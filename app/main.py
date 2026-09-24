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

import hashlib
import html
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from os import stat_result
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import Response
from starlette.types import Scope

from app.config import Settings
from app.embeddings import E5Embeddings, get_embeddings
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
from app.texts import Texte, texte_fuer

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Die Dateien, die die Seite einbindet und die deshalb einen Fassungsschluessel
# in der Adresse tragen. index.html gehoert NICHT dazu - sie ist die Vorlage und
# wird nie aus dem Cache bedient (siehe _StatischeDateien).
VERSIONIERTE_DATEIEN: tuple[str, ...] = ("app.js", "style.css")

# Wie lange der Browser eine versionierte Datei behalten darf. Ein Jahr und
# "immutable", weil sich unter einer gegebenen Adresse nichts mehr aendern KANN:
# Aendert sich der Inhalt, aendert sich der Schluessel und damit die Adresse.
_CACHE_VERSIONIERT = "public, max-age=31536000, immutable"


def statische_fassung(verzeichnis: Path = STATIC_DIR) -> str:
    """Kurzer Schluessel ueber den INHALT der eingebundenen Dateien.

    Weg A aus OP-054: Die Seite bindet `/static/app.js?v=<schluessel>` ein, also
    bekommt der Browser nach einer Aenderung eine andere Adresse und kann die
    alte Datei nicht mehr liefern.

    Der Schluessel ist ein Inhaltshash und ausdruecklich NICHT der Commit. Beide
    sind Weg A, aber der Inhaltshash ist der praezisere Schluessel: Er wechselt
    genau dann, wenn sich etwas an der ausgelieferten Datei aendert, und nicht
    bei jedem Commit an einer beliebigen Stelle des Projekts. Er braucht
    ausserdem kein Bauargument - im Container gilt dasselbe Verfahren wie lokal,
    ohne dass jemand `--build-arg` vergessen kann.

    Berechnet wird er EINMAL beim Bau der Anwendung, nicht je Anfrage. Wer
    waehrend des Betriebs eine Datei unter `static/` aendert, muss den Prozess
    neu starten - im Betrieb ist das ein neues Image, lokal reicht `--reload`.
    """
    schluessel = hashlib.sha256()
    for name in sorted(VERSIONIERTE_DATEIEN):
        datei = verzeichnis / name
        if datei.is_file():
            schluessel.update(datei.read_bytes())
    return schluessel.hexdigest()[:12]


class _StatischeDateien(StaticFiles):
    """StaticFiles mit ausdruecklichen Cache-Kopfzeilen.

    Bis zum 2026-09-24 lieferte `/static/` **keinen** `Cache-Control`-Kopf. Ohne
    ihn entscheidet der Browser nach eigener Heuristik, wie lange er eine Datei
    behaelt - und am 2026-09-23 hat genau das dazu gefuehrt, dass ein `app.js`
    von vor EN-1 mit deutschen Zeichenketten ausgeliefert wurde, obwohl die Datei
    im Image englisch war. Gemeldet als Sprachfehler, war es ein Cachefehler
    (OP-054).

    Zwei Faelle, und sie sind entgegengesetzt:

    * **Versionierte Dateien** duerfen sehr lange behalten werden. Ihre Adresse
      traegt den Inhaltsschluessel; eine Aenderung erzeugt eine neue Adresse.
    * **index.html** darf NIE aus dem Cache kommen. Sie ist die Vorlage mit den
      Platzhaltern und traegt die Adressen der versionierten Dateien. Wird sie
      zwischengespeichert, verweist sie weiter auf die alte Fassung, und das
      Verfahren oben laeuft leer.
    """

    def file_response(
        self,
        full_path: str | Path,
        stat_res: stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        antwort = super().file_response(full_path, stat_res, scope, status_code)
        name = Path(str(full_path)).name
        antwort.headers["Cache-Control"] = (
            "no-store" if name == "index.html" else _CACHE_VERSIONIERT
        )
        return antwort


# Ein einziger Wortlaut fuer jeden Fall, in dem kein Mandant aufgeloest werden
# konnte. Verschiedene Texte waeren ein Orakel: wer den Unterschied zwischen
# "ungueltig" und "unbekannt" sieht, kann Token erraten.
#
# Der Wortlaut steht BEWUSST nicht im Textkatalog und richtet sich nicht nach
# einer Mandantensprache. Er faellt, bevor ein Mandant aufgeloest ist - es gibt
# an dieser Stelle keine Sprache, aus der er sich ableiten liesse. Selbst wenn
# es sie gaebe, waere ein sprachabhaengiger Wortlaut genau das Orakel, das diese
# Konstante verhindern soll.
#
# Zweisprachig und fest, weil ein zerbrochener Link auch bei einem
# englischsprachigen Empfaenger ankommt, der mit einem rein deutschen Satz
# nichts anfangen kann. Beide Haelften stehen immer zusammen; damit ist der
# Wortlaut fuer jeden Aufruf identisch.
NICHT_GEFUNDEN = "Diese Adresse gibt es nicht. / This address does not exist."

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


def _pfadmuster(pfad: str) -> str:
    """Ersetzt ein url_token im Pfad durch einen Platzhalter.

    Das Token ist die Zugangskontrolle (ADR-007). Ein Log, das Token
    mitschreibt, ist eine Schluesselliste - und Logs wandern in Phase 6 nach
    Application Insights, also aus meinem Zugriff heraus.

    Fuer die Diagnose reicht das Muster vollkommen: Wer `/t/{token}/` im Log
    sieht, weiss, dass die Route stimmte und die Aufloesung scheiterte. Wer `/`
    sieht, weiss, dass jemand die Wurzel aufgerufen hat, wo es nichts gibt.

    Alles ausserhalb von `/t/` wird unveraendert uebernommen - Pfade wie `/`,
    `/favicon.ico` oder `/admin` tragen kein Geheimnis und sind als Wortlaut
    nuetzlicher als ein Platzhalter.
    """
    teile = pfad.split("/")
    # ["", "t", "<token>", ...] - erst ab drei Teilen gibt es ein Token.
    if len(teile) >= 3 and teile[1] == "t" and teile[2]:
        teile[2] = "{token}"
        return "/".join(teile)
    return pfad


def _texte_als_json(texte: Texte) -> str:
    """Serialisiert die Browsertexte fuer den <script>-Block der Seite.

    `<`, `>` und `&` werden als \\u-Folgen geschrieben. Ohne das koennte ein
    Text, der zufaellig `</script>` enthaelt, den Block vorzeitig schliessen -
    und alles danach waere Markup statt Daten. Die Texte stammen zwar aus dem
    Katalog dieses Repos und nicht von aussen, aber die Maskierung kostet
    nichts und haelt die Stelle auch dann sicher, wenn spaeter jemand einen
    Text ergaenzt, ohne an diesen Block zu denken.

    Bewusst NICHT html.escape: Der Inhalt eines <script>-Elements ist kein
    HTML-Text. Ein &quot; darin waere kaputtes JSON.
    """
    roh = json.dumps(texte.fuer_javascript(), ensure_ascii=False)
    return roh.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


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

    @asynccontextmanager
    async def lebenszyklus(laufende_app: FastAPI) -> AsyncIterator[None]:
        """Waermt den Embedder vor, bevor der Port Verkehr annimmt (ADR-023).

        Zwei Gruende, und der zweite wiegt schwerer als der erste:

        1. `/health` wird ehrlich. Ohne Vorwaermen meldet der Endpunkt rund
           sechs Sekunden lang Gesundheit, bevor eine Frage beantwortet werden
           kann - der Embedder entstuende erst beim ersten Bedarf. Ein
           Orchestrator schickt Verkehr, sobald die Bereitschaftspruefung
           Erfolg meldet, und traefe einen Prozess, der noch laedt.

        2. Ein kaputter oder fehlender Modellcache scheitert jetzt beim START
           und nicht bei der ersten Anfrage eines Interessenten. Dieselbe
           Regel, die fuer Secrets gilt: laut und sofort statt spaet und beim
           Kunden. Der Modellcache ist fuer diese Anwendung genauso eine
           Startbedingung wie ein Schluessel.

        Es entsteht KEIN zweiter Embedder: `get_embeddings` liefert den
        prozessweit geteilten (ADR-020). Dieses Vorwaermen aendert das WANN,
        nicht das WIE OFT. Ein eingespeister wird nicht ueberschrieben -
        sonst zoegen Tests, die einen Ersatz einspeisen, beim Start das echte
        Modell, und Unit-Tests duerfen nicht ins Netz.
        """
        if laufende_app.state.embeddings is None:
            laufende_app.state.embeddings = get_embeddings(laufende_app.state.settings)
        yield

    app = FastAPI(
        title="rag-support-demo",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lebenszyklus,
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

    # Der Fassungsschluessel wird EINMAL berechnet, nicht je Anfrage: Er haengt
    # am Dateiinhalt, und der aendert sich im Betrieb nicht.
    app.state.statische_fassung = statische_fassung()

    if STATIC_DIR.is_dir():
        app.mount("/static", _StatischeDateien(directory=STATIC_DIR), name="static")

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
                # Der Mandant ist hier bereits aufgeloest, also gibt es eine
                # Sprache. Anders als beim 404 verraet dieser Text nichts:
                # Wer ihn sieht, hat schon ein gueltiges Token.
                detail=texte_fuer(tenant.language).ratenlimit_detail,
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
        texte = texte_fuer(tenant.language)

        vorlage = STATIC_DIR / "index.html"
        if not vorlage.is_file():
            raise HTTPException(status_code=500, detail=texte.oberflaeche_fehlt)

        # Der Mandantenname kommt aus der TenantConfig und wird serverseitig
        # gesetzt. Es gibt keinen Endpunkt, ueber den die Oberflaeche ihn
        # nachladen koennte - das waere ein Endpunkt, der Mandantendaten
        # ausgibt.
        #
        # Dasselbe gilt fuer die Sprache: Sie wird hier eingesetzt und nicht
        # im Browser aus einer Kennung abgeleitet. Die Seite eines Mandanten
        # ist in genau einer Sprache, und welche das ist, entscheidet die
        # tenant.yaml - nicht die Browsereinstellung des Besuchers.
        ersetzungen = {
            "{{lang}}": tenant.language,
            "{{titel}}": html.escape(texte.titel),
            "{{display_name}}": html.escape(tenant.display_name),
            "{{begruessung}}": html.escape(
                texte.begruessung.format(
                    display_name=tenant.display_name, topics=tenant.topics.strip()
                )
            ),
            "{{frage_label}}": html.escape(texte.frage_label),
            "{{frage_platzhalter}}": html.escape(texte.frage_platzhalter),
            "{{senden}}": html.escape(texte.senden),
            "{{fusszeile}}": html.escape(texte.fusszeile),
            "{{url_token}}": html.escape(url_token),
            # Steht im <script>-Element und wird deshalb anders maskiert.
            "{{texte_json}}": _texte_als_json(texte),
            # Inhaltsschluessel der eingebundenen Dateien. Nicht maskiert, weil
            # es ein Hexadezimalwert aus dem eigenen Prozess ist und kein
            # Fremdtext - aber ein Wert in einem Attribut, also ist die
            # Herkunft der Grund, nicht die Bequemlichkeit.
            "{{fassung}}": app.state.statische_fassung,
        }

        seite = vorlage.read_text(encoding="utf-8")
        for platzhalter, wert in ersetzungen.items():
            seite = seite.replace(platzhalter, wert)

        _ereignis("oberflaeche", tenant.slug, sprache=tenant.language)
        # no-store und nicht no-cache: Die Seite traegt die Adressen der
        # versionierten Dateien. Eine zwischengespeicherte Seite verweist weiter
        # auf die alte Fassung, und das Verfahren aus OP-054 laeuft leer. Sie
        # traegt ausserdem das url_token - eine Kopie davon im Browsercache ist
        # nichts, was ohne Not entstehen soll.
        return HTMLResponse(seite, headers={"Cache-Control": "no-store"})

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
    def nicht_gefunden(request: Request, __: Exception) -> JSONResponse:
        """Auch unbekannte Pfade antworten mit demselben Wortlaut.

        Der Wortlaut ist ueberall gleich, damit aus der Antwort nicht ableitbar
        ist, ob ein Token existiert. Der Preis dafuer ist, dass die Diagnose
        ueber das Log laufen muss - und genau dort fehlte bis 2026-08-17 der
        Eintrag fuer den haeufigsten Fall: einen Aufruf, der gar keine Route
        trifft. Ein 404 auf `/` hinterliess keine Spur.

        Das ist die dritte Auspraegung von P-018: nicht falsch meldend und
        nicht unsichtbar, sondern stumm an der Stelle, an der man hinsieht.
        """
        _ereignis("route_unbekannt", None, pfad=_pfadmuster(request.url.path))
        return JSONResponse(status_code=404, content={"detail": NICHT_GEFUNDEN})

    return app
