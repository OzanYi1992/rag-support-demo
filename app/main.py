"""HTTP-Schicht ueber dem bestehenden RAG-Pfad.

Diese Schicht entscheidet nichts. Sie loest ein url_token auf einen Mandanten
auf, reicht dessen slug als tenant_id an answer() weiter und gibt zurueck, was
herauskommt. Retrieval, Schwellwerte und Eskalation liegen unveraendert in
app/rag.py und app/escalation.py.

Start:

    .venv/bin/uvicorn app.main:create_app --factory --reload

Bewusst eine Fabrik statt eines Modul-Level-`app`: Ein `app = create_app()` auf
Modulebene wuerde beim Import die Settings laden und damit die Konfiguration
lesen. Jeder Testimport haenge dann an einer vollstaendigen Umgebung.

Zur Mandantentrennung (ADR-001, ADR-007):

* Es gibt genau eine Stelle, die aus einem url_token einen Mandanten macht:
  `_mandant()`. Danach wandert `tenant.slug` als `tenant_id` weiter.
* Kein Endpunkt nimmt eine tenant_id entgegen, keiner gibt einen slug aus,
  keiner listet auf. Auch `/health` nicht - der Endpunkt ist oeffentlich, und
  die Token sind die Zugangskontrolle.
* Unbekanntes und ungueltiges Token ergeben nach aussen dieselbe Antwort. Aus
  der Antwort ist nicht ableitbar, ob ein Token existiert. Im Log sind die
  Faelle unterscheidbar.
"""

from __future__ import annotations

import html
import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import Settings
from app.embeddings import E5Embeddings
from app.llm import LlmClient
from app.rag import Answer, answer
from app.ratelimit import RateLimiter
from app.tenants import (
    MIN_TOKEN_LENGTH,
    AmbiguousUrlToken,
    InvalidTenantSlug,
    TenantConfig,
    TenantNotFound,
    resolve_token,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Ein einziger Wortlaut fuer jeden Fall, in dem kein Mandant aufgeloest werden
# konnte. Verschiedene Texte waeren ein Orakel: wer den Unterschied zwischen
# "ungueltig" und "unbekannt" sieht, kann Token erraten.
NICHT_GEFUNDEN = "Diese Adresse gibt es nicht."

_log = logging.getLogger("rag.api")


def _ereignis(name: str, tenant_id: str | None, **felder: object) -> None:
    """Strukturiertes Ereignis mit tenant_id als Dimension (ADR-002).

    Vorlaeufig, bis Phase 6 OpenTelemetry einzieht. Was hier schon gilt und
    dort gelten wird: tenant_id ist Pflichtfeld, und Frageinhalte, Antworttexte
    und Dokumentinhalte gehoeren nicht hinein.

    Das url_token wird NICHT geloggt. Es ist die Zugangskontrolle; ein Log, das
    Token mitschreibt, ist eine Schluesselliste.
    """
    _log.info(json.dumps({"ereignis": name, "tenant_id": tenant_id, **felder}))


class ChatAnfrage(BaseModel):
    """Rumpf von POST /t/{url_token}/chat."""

    question: str = Field(min_length=1, max_length=2000)
    response_language: str | None = Field(default=None, max_length=16)


def create_app(
    settings: Settings | None = None,
    *,
    llm: LlmClient | None = None,
    embeddings: E5Embeddings | None = None,
    rate_limit: int = 30,
) -> FastAPI:
    """Baut die Anwendung.

    `llm` und `embeddings` sind Einspeisepunkte fuer Tests. Bleiben sie None,
    loest answer() sie selbst aus den Settings auf - die Produktionsvariante.
    """
    aktive_settings = settings or Settings()
    app = FastAPI(
        title="rag-support-demo",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Der Zaehler haengt an DIESER Anwendung, nicht am Modul. Damit hat jeder
    # Test seine eigene Anwendung und sein eigenes Kontingent.
    app.state.limiter = RateLimiter(max_requests=rate_limit, window_seconds=60.0)
    app.state.settings = aktive_settings
    app.state.llm = llm
    app.state.embeddings = embeddings

    # docs_url/redoc_url/openapi_url sind abgeschaltet: Ein Schema-Endpunkt
    # listet Pfade und Modelle und ist auf einer oeffentlich verlinkten Demo
    # eine Einladung.

    # CORS: leere Liste heisst, dass kein fremder Ursprung erlaubt ist.
    # Gleichursprungs-Anfragen - die Oberflaeche wird vom selben Host
    # ausgeliefert - laufen ohne CORS. Wer das oeffnet, oeffnet den Chat fuer
    # jede fremde Seite und bezahlt deren Aufrufe.
    erlaubte_urspruenge = [
        eintrag.strip()
        for eintrag in aktive_settings.cors_allowed_origins.split(",")
        if eintrag.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=erlaubte_urspruenge,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    def _mandant(url_token: str) -> TenantConfig:
        """Die einzige Stelle, die aus einem Token einen Mandanten macht."""
        if len(url_token) < MIN_TOKEN_LENGTH:
            # Kann kein gueltiges Token sein. Ohne diese Abkuerzung wuerde jede
            # Muellanfrage alle Mandantenverzeichnisse durchsuchen.
            _ereignis("token_abgelehnt", None, grund="zu_kurz")
            raise HTTPException(status_code=404, detail=NICHT_GEFUNDEN)
        try:
            return resolve_token(url_token, aktive_settings.tenants_dir)
        except TenantNotFound:
            _ereignis("token_abgelehnt", None, grund="unbekannt")
            raise HTTPException(status_code=404, detail=NICHT_GEFUNDEN) from None
        except InvalidTenantSlug:
            _ereignis("token_abgelehnt", None, grund="ungueltiger_slug")
            raise HTTPException(status_code=404, detail=NICHT_GEFUNDEN) from None
        except AmbiguousUrlToken:
            # Kein Nutzerfehler, sondern ein Konfigurationsfehler: zwei
            # Mandanten teilen ein Token. Das darf NICHT als "gibt es nicht"
            # untergehen, sonst sucht niemand danach.
            _ereignis("token_kollision", None, grund="mehrdeutig")
            raise HTTPException(status_code=500, detail="Konfigurationsfehler.") from None

    def _limit_pruefen(tenant: TenantConfig, url_token: str) -> None:
        erlaubt, frei_in = app.state.limiter.pruefe(url_token)
        if not erlaubt:
            _ereignis("rate_limit", tenant.slug, frei_in_sekunden=frei_in)
            raise HTTPException(
                status_code=429,
                detail="Zu viele Anfragen. Bitte kurz warten.",
                headers={"Retry-After": str(frei_in)},
            )

    @app.get("/health")
    def health() -> dict[str, str]:
        """Oeffentlich und absichtlich nichtssagend.

        Keine Mandantenliste, keine Zaehlung, keine Version. Wer diesen
        Endpunkt erreicht, erfaehrt genau, dass der Prozess laeuft.
        """
        return {"status": "ok"}

    @app.get("/t/{url_token}/", response_class=HTMLResponse)
    def oberflaeche(url_token: str) -> HTMLResponse:
        tenant = _mandant(url_token)
        vorlage = STATIC_DIR / "index.html"
        if not vorlage.is_file():
            raise HTTPException(status_code=500, detail="Oberflaeche fehlt.")

        # Der Mandantenname kommt aus der TenantConfig und wird serverseitig
        # gesetzt. Es gibt keinen Endpunkt, ueber den die Oberflaeche ihn
        # nachladen koennte - das waere ein Endpunkt, der Mandantendaten
        # ausgibt.
        seite = (
            vorlage.read_text(encoding="utf-8")
            .replace("{{display_name}}", html.escape(tenant.display_name))
            .replace("{{url_token}}", html.escape(url_token))
        )
        _ereignis("oberflaeche", tenant.slug)
        return HTMLResponse(seite)

    @app.post("/t/{url_token}/chat")
    def chat(url_token: str, anfrage: ChatAnfrage) -> Answer:
        tenant = _mandant(url_token)
        _limit_pruefen(tenant, url_token)

        begonnen = time.monotonic()
        ergebnis = answer(
            tenant.slug,
            anfrage.question,
            response_language=anfrage.response_language,
            settings=aktive_settings,
            llm=app.state.llm,
            embeddings=app.state.embeddings,
        )
        gesamt_ms = int((time.monotonic() - begonnen) * 1000)

        # Kein Fragetext, kein Antworttext, keine Quellnamen im Log.
        _ereignis(
            "chat",
            tenant.slug,
            escalated=ergebnis.escalated,
            escalation_reason=ergebnis.escalation_reason,
            treffer=len(ergebnis.sources),
            lang=ergebnis.lang,
            prompt_tokens=ergebnis.prompt_tokens,
            completion_tokens=ergebnis.completion_tokens,
            model=ergebnis.model,
            gesamt_ms=gesamt_ms,
        )
        return ergebnis

    @app.exception_handler(404)
    def nicht_gefunden(_: Request, __: Exception) -> JSONResponse:
        """Auch unbekannte Pfade antworten mit demselben Wortlaut."""
        return JSONResponse(status_code=404, content={"detail": NICHT_GEFUNDEN})

    return app
