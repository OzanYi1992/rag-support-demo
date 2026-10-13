"""Tests des Providerzugangs - und zwar am PAYLOAD, nicht am Objekt.

Der Grund fuer diese Datei ist ein Fehlschlag vom 2026-09-24. Die erste Fassung
der Temperaturdurchreichung wurde gegen ein Settings-Objekt mit dem
Modellnamen "x" geprueft. Der Wert stand danach brav am Modellobjekt, der Test
war gruen - und im Betrieb fehlte der Parameter im Payload, weil
langchain-openai ihn fuer die gpt-5-Familie still verwirft. Ein gruener Test,
dessen Gegenstand der falsche war (P-023).

Deshalb prueft hier alles den Payload, den `_get_request_payload` liefert, und
mit den ECHTEN Modellnamen. Netz wird dabei nicht angefasst: Der Payload
entsteht vor dem Absenden.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import httpx
import openai
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai.chat_models.base import BaseChatOpenAI

from app.config import Settings
from app.llm import (
    LlmAnfrageAbgelehnt,
    LlmNichtErreichbar,
    StructuredChatClient,
    build_llm,
)
from app.prompts import antwortmodell_fuer

# Das Modell, das im Betrieb konfiguriert ist. Steht hier als Literal, damit
# der Test nicht mitwandert, wenn jemand die Konfiguration aendert - dann soll
# er auffallen.
MODELL_IM_BETRIEB = "gpt-5.4-mini-2026-03-17"

# Ein Modell aus einer Familie, die temperature annimmt. Die Gegenprobe zur
# Aussage "der Parameter wird verworfen" - ohne sie waere nicht zu
# unterscheiden, ob unser Code ihn nie setzt.
MODELL_MIT_TEMPERATUR = "gpt-4o-mini"

NACHRICHTEN = [SystemMessage(content="system"), HumanMessage(content="user")]


def _payload(modellname: str, temperatur: float) -> dict:
    """Der Payload, den der fertig gebaute Client absenden wuerde."""
    settings = Settings(
        llm_provider="openai",
        openai_api_key="platzhalter",
        openai_model=modellname,
        llm_temperature=temperatur,
    )
    klient = build_llm(settings, None, answer_model=antwortmodell_fuer("de"))
    gebunden = _bindung(klient._runnable)
    assert gebunden is not None, "Chatmodell in der Kette nicht gefunden"
    return gebunden.bound._get_request_payload(NACHRICHTEN, stop=None, **gebunden.kwargs)


def _bindung(objekt: object, tiefe: int = 0, typ: type = BaseChatOpenAI) -> object | None:
    """Sucht das gebundene Chatmodell in der Kette.

    Rekursiv, weil `include_raw=True` die Kette verzweigt: Das Modell liegt
    unter steps[0].steps__["raw"] und nicht auf der obersten Ebene. `typ` ist
    die Klasse des Chatmodells; ohne Angabe die von OpenAI und Azure.
    """
    if tiefe > 6:
        return None
    if isinstance(getattr(objekt, "bound", None), typ):
        return objekt
    for name in ("steps", "middle"):
        if isinstance(getattr(objekt, name, None), list):
            for teil in getattr(objekt, name):
                gefunden = _bindung(teil, tiefe + 1, typ)
                if gefunden is not None:
                    return gefunden
    abbildung = getattr(objekt, "steps__", None)
    if isinstance(abbildung, dict):
        for teil in abbildung.values():
            gefunden = _bindung(teil, tiefe + 1, typ)
            if gefunden is not None:
                return gefunden
    for name in ("first", "last", "bound", "runnable", "default"):
        teil = getattr(objekt, name, None)
        if teil is not None and teil is not objekt:
            gefunden = _bindung(teil, tiefe + 1, typ)
            if gefunden is not None:
                return gefunden
    return None


def test_temperatur_erreicht_den_payload():
    """Der konfigurierte Wert wird gesendet, nicht nur gesetzt.

    Das ist der Zweck der Einstellung: Ohne sie entschied der Anbieter, und eine
    Aenderung seines Standards haette jede Messung verschoben, ohne dass eine
    Zeile Code anders wird.
    """
    assert _payload(MODELL_MIT_TEMPERATUR, 0.0)["temperature"] == 0.0
    assert _payload(MODELL_MIT_TEMPERATUR, 0.7)["temperature"] == 0.7


def test_nur_temperatur_eins_erreicht_das_konfigurierte_modell():
    """Die gpt-5-Familie nimmt ausschliesslich temperature=1.

    langchain-openai verwirft jeden anderen Wert in einem Validator, STILL -
    kein Fehler, keine Warnung, der Parameter fehlt danach im Payload.

    Dieser Test haelt die Einschraenkung fest, damit niemand den Standard auf
    0.0 dreht und glaubt, er wirke. Wird er eines Tages rot, weil auch 0.0
    ankommt, dann hat der Anbieter oder die Bibliothek die Einschraenkung
    aufgehoben - und dann ist die Wahl der Temperatur neu zu messen, nicht der
    Test zu loeschen.
    """
    assert _payload(MODELL_IM_BETRIEB, 1.0)["temperature"] == 1.0
    assert "temperature" not in _payload(MODELL_IM_BETRIEB, 0.0)


def test_das_strukturschema_folgt_der_uebergebenen_sprache():
    """Was beim Modell ankommt, ist das Schema der Mandantensprache.

    Geprueft am Payload und nicht am Klassenobjekt: Zwischen beiden liegt
    `with_structured_output`, und genau dort ist der Fehler vom 2026-09-24
    unentdeckt durchgelaufen.
    """
    from openai.lib._parsing._completions import type_to_response_format_param

    for sprache, erwartet in (("de", "Die Antwort"), ("en", "The answer")):
        settings = Settings(
            llm_provider="openai",
            openai_api_key="platzhalter",
            openai_model=MODELL_IM_BETRIEB,
            llm_temperature=1.0,
        )
        klient = build_llm(settings, None, answer_model=antwortmodell_fuer(sprache))
        gebunden = _bindung(klient._runnable)
        payload = gebunden.bound._get_request_payload(NACHRICHTEN, stop=None, **gebunden.kwargs)
        schema = type_to_response_format_param(payload["response_format"])
        beschreibung = schema["json_schema"]["schema"]["properties"]["answer"]["description"]
        assert beschreibung.startswith(erwartet), f"{sprache}: {beschreibung!r}"


# =============================================================================
# ZEITLIMIT, WIEDERHOLUNGEN UND DIE UEBERSETZUNG DER ANBIETERFEHLER
#
# Ohne Zeitlimit wartete jeder der drei Clients unbegrenzt auf einen Anbieter,
# der nicht antwortet, und wiederholte zweimal. Geprueft wird zweierlei: dass
# die Werte am SDK-Client ankommen, fuer alle drei Provider, und dass sie
# wirken, gegen einen Server in diesem Prozess. Das Attribut allein waere der
# Fehler vom Temperaturtest oben: am Objekt richtig, im Betrieb wirkungslos.
#
# Die Wirkung wird nur am Azure-Client geprueft. Nur dessen Adresse kommt
# ueber die Settings; OpenAI und Anthropic liessen sich nur ueber eine
# Umgebungsvariable umleiten, und Tests setzen keine.
# =============================================================================

AZURE_API_VERSION = "2024-10-21"

# Anthropic ist im Betrieb nicht konfiguriert. Ein echter Modellname trotzdem,
# damit kein Validator, der am Namen haengt, den Test anders laufen laesst als
# einen echten Aufruf.
ANTHROPIC_MODELL = "claude-sonnet-5-5"


def _settings(provider: str, **zusatz: object) -> Settings:
    """Settings fuer einen Provider, mit Platzhalter-Schluessel."""
    felder: dict[str, object] = {"llm_provider": provider}
    if provider == "openai":
        felder |= {"openai_api_key": "platzhalter", "openai_model": MODELL_IM_BETRIEB}
    elif provider == "azure":
        felder |= {
            "azure_openai_api_key": "platzhalter",
            "azure_openai_endpoint": "https://platzhalter.openai.azure.com",
            "azure_openai_api_version": AZURE_API_VERSION,
            "azure_openai_deployment": MODELL_IM_BETRIEB,
        }
    else:
        felder |= {"anthropic_api_key": "platzhalter", "anthropic_model": ANTHROPIC_MODELL}
    return Settings(**(felder | zusatz))


def _sdk_client(provider: str, settings: Settings) -> object:
    """Der SDK-Client unter dem Chatmodell, so wie build_llm ihn baut."""
    klient = build_llm(settings, None, answer_model=antwortmodell_fuer("de"))
    typ = ChatAnthropic if provider == "anthropic" else BaseChatOpenAI
    gebunden = _bindung(klient._runnable, typ=typ)
    assert gebunden is not None, f"{provider}: Chatmodell in der Kette nicht gefunden"
    modell = gebunden.bound
    return modell._client if provider == "anthropic" else modell.root_client


@pytest.mark.parametrize("provider", ["openai", "azure", "anthropic"])
def test_zeitlimit_und_wiederholungen_kommen_an_allen_drei_clients_an(provider: str) -> None:
    """Zeitlimit und Wiederholungen stehen am SDK-Client, nicht nur am Chatmodell.

    Gegenprobe im selben Test: Ein anderer Wert kommt ebenfalls an. Ohne sie
    waere nicht zu unterscheiden, ob der Wert durchgereicht wird oder zufaellig
    der Voreinstellung des SDK gleicht.
    """
    sdk = _sdk_client(provider, _settings(provider, llm_timeout_s=20.0, llm_max_retries=0))
    assert sdk.timeout == 20.0
    assert sdk.max_retries == 0

    anders = _sdk_client(provider, _settings(provider, llm_timeout_s=7.5, llm_max_retries=2))
    assert anders.timeout == 7.5
    assert anders.max_retries == 2


def _stummer_server() -> socket.socket:
    """Nimmt Verbindungen an und antwortet nie.

    Der Kernel schliesst den Verbindungsaufbau ueber die Warteschlange ab, auch
    ohne accept(). Der Client schickt seine Anfrage und wartet auf eine
    Antwort, die nicht kommt.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    return server


def _im_faden(klient: object, waechter_s: float) -> tuple[threading.Thread, dict[str, object]]:
    """Ruft generate in einem Faden und wartet hoechstens `waechter_s`.

    Ohne Faden liesse ein fehlendes Zeitlimit den Test haengen statt rot werden.
    """
    ergebnis: dict[str, object] = {}

    def lauf() -> None:
        start = time.monotonic()
        try:
            klient.generate("system", "user")
        except Exception as fehler:  # noqa: BLE001 - der Test prueft die Klasse selbst
            ergebnis["fehler"] = fehler
        ergebnis["dauer"] = time.monotonic() - start

    faden = threading.Thread(target=lauf, daemon=True)
    faden.start()
    faden.join(timeout=waechter_s)
    return faden, ergebnis


def test_zeitlimit_wirkt_gegen_einen_stummen_server() -> None:
    """Die Wirkung, nicht das Attribut: Nach dem Zeitlimit ist Schluss.

    Erwartet LlmNichtErreichbar mit dem Grund zeitlimit in unter drei Sekunden
    bei einem Zeitlimit von 0,5 s. Der Waechter wartet fuenf Sekunden; laeuft
    der Faden dann noch, gibt es kein wirksames Zeitlimit.
    """
    server = _stummer_server()
    try:
        port = server.getsockname()[1]
        settings = _settings(
            "azure",
            azure_openai_endpoint=f"http://127.0.0.1:{port}",
            llm_timeout_s=0.5,
            llm_max_retries=0,
        )
        klient = build_llm(settings, None, answer_model=antwortmodell_fuer("de"))
        faden, ergebnis = _im_faden(klient, waechter_s=5.0)

        assert not faden.is_alive(), "nach 5 s noch kein Ende: das Zeitlimit wirkt nicht"
        fehler = ergebnis["fehler"]
        assert isinstance(fehler, LlmNichtErreichbar), repr(fehler)
        assert fehler.grund == "zeitlimit"
        assert ergebnis["dauer"] < 3.0
    finally:
        server.close()


@contextmanager
def _antwortserver(status: int, kopf: dict[str, str], rumpf: dict) -> Iterator[dict[str, object]]:
    """Ein HTTP-Server in diesem Prozess, der jede Anfrage gleich beantwortet.

    Zaehlt die Anfragen. Der Rumpf ist JSON, so wie ihn die Anbieter schicken.
    """
    zustand: dict[str, object] = {"anfragen": 0}
    roh = json.dumps(rumpf).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - Name von http.server vorgegeben
            laenge = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(laenge)
            zustand["anfragen"] = int(zustand["anfragen"]) + 1
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(roh)))
            for name, wert in kopf.items():
                self.send_header(name, wert)
            self.end_headers()
            self.wfile.write(roh)

        def log_message(self, *_: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    faden = threading.Thread(target=server.serve_forever, daemon=True)
    faden.start()
    zustand["port"] = server.server_address[1]
    try:
        yield zustand
    finally:
        server.shutdown()
        server.server_close()


RUMPF_429 = {"error": {"code": "429", "message": "Rate limit reached."}}


def test_keine_wiederholung_bei_429() -> None:
    """Mit llm_max_retries=0 geht genau eine Anfrage hinaus.

    Das SDK wiederholt sonst und wartet dazwischen so lange, wie retry-after
    verlangt, bei OpenAI und Azure bis 120 s. Gegenprobe im selben Test: Mit
    einer Wiederholung sind es zwei Anfragen. Ohne sie waere nicht zu
    unterscheiden, ob der Server die zweite gar nicht zaehlt.
    """
    with _antwortserver(429, {"retry-after": "1"}, RUMPF_429) as server:
        settings = _settings(
            "azure",
            azure_openai_endpoint=f"http://127.0.0.1:{server['port']}",
            llm_timeout_s=5.0,
            llm_max_retries=0,
        )
        klient = build_llm(settings, None, answer_model=antwortmodell_fuer("de"))
        faden, ergebnis = _im_faden(klient, waechter_s=10.0)
        assert not faden.is_alive()
        anfragen = server["anfragen"]
        assert anfragen == 1
        fehler = ergebnis["fehler"]
        assert isinstance(fehler, LlmNichtErreichbar), repr(fehler)
        assert fehler.grund == "status_429"

    with _antwortserver(429, {"retry-after": "1"}, RUMPF_429) as server:
        settings = _settings(
            "azure",
            azure_openai_endpoint=f"http://127.0.0.1:{server['port']}",
            llm_timeout_s=5.0,
            llm_max_retries=1,
        )
        klient = build_llm(settings, None, answer_model=antwortmodell_fuer("de"))
        faden, ergebnis = _im_faden(klient, waechter_s=10.0)
        assert not faden.is_alive()
        assert server["anfragen"] == 2


@pytest.mark.parametrize(
    ("status", "rumpf", "grund"),
    [
        # So antwortet Azure, wenn sein Inhaltsfilter die Eingabe blockt.
        (
            400,
            {
                "error": {
                    "code": "content_filter",
                    "message": "The response was filtered.",
                    "param": "prompt",
                    "status": 400,
                }
            },
            "inhaltsfilter",
        ),
        (400, {"error": {"code": "invalid_request", "message": "Bad request."}}, "status_400"),
        (401, {"error": {"code": "401", "message": "Access denied."}}, "status_401"),
    ],
)
def test_abgewiesene_anfrage_wird_llm_anfrage_abgelehnt(
    status: int, rumpf: dict, grund: str
) -> None:
    """Ein Fehler, den ein neuer Versuch nicht behebt, ist kein Ausfall.

    Ueber das echte SDK, damit der Fehlercode aus dem Rumpf so ankommt, wie das
    SDK ihn liest, und nicht so, wie der Test ihn annimmt.
    """
    with _antwortserver(status, {}, rumpf) as server:
        settings = _settings(
            "azure",
            azure_openai_endpoint=f"http://127.0.0.1:{server['port']}",
            llm_timeout_s=5.0,
            llm_max_retries=0,
        )
        klient = build_llm(settings, None, answer_model=antwortmodell_fuer("de"))
        faden, ergebnis = _im_faden(klient, waechter_s=10.0)
        assert not faden.is_alive()
        fehler = ergebnis["fehler"]
        assert isinstance(fehler, LlmAnfrageAbgelehnt), repr(fehler)
        assert not isinstance(fehler, LlmNichtErreichbar)
        assert fehler.grund == grund
        assert server["anfragen"] == 1


# --- die Uebersetzung, fuer beide SDKs -------------------------------------

_ANFRAGE = httpx.Request("POST", "https://anbieter.invalid/v1/chat")


def _status(sdk: object, code: int, rumpf: object = None) -> Exception:
    return sdk.APIStatusError("x", response=httpx.Response(code, request=_ANFRAGE), body=rumpf)


class _WerfendesModell:
    """Ein Chatmodell, dessen Aufruf mit dem gegebenen Fehler endet."""

    def __init__(self, fehler: Exception) -> None:
        self._fehler = fehler

    def with_structured_output(self, schema: object, include_raw: bool) -> _WerfendesModell:
        return self

    def invoke(self, nachrichten: object) -> None:
        raise self._fehler


def _generate_mit(fehler: Exception) -> Exception:
    klient = StructuredChatClient(_WerfendesModell(fehler), "x", antwortmodell_fuer("de"))
    with pytest.raises(Exception) as gefangen:
        klient.generate("system", "user")
    return gefangen.value


NICHT_ERREICHBAR = [
    (openai.APITimeoutError(request=_ANFRAGE), "zeitlimit"),
    (anthropic.APITimeoutError(request=_ANFRAGE), "zeitlimit"),
    (openai.APIConnectionError(request=_ANFRAGE), "verbindung"),
    (anthropic.APIConnectionError(request=_ANFRAGE), "verbindung"),
    (_status(openai, 408), "status_408"),
    (_status(openai, 409), "status_409"),
    (_status(openai, 429), "status_429"),
    (_status(anthropic, 429), "status_429"),
    (_status(openai, 500), "status_500"),
    (_status(openai, 503), "status_503"),
    (_status(anthropic, 529), "status_529"),
]

ABGELEHNT = [
    (_status(openai, 400, {"code": "content_filter", "message": "x"}), "inhaltsfilter"),
    (_status(openai, 400, {"code": "invalid_request", "message": "x"}), "status_400"),
    (_status(openai, 400), "status_400"),
    # Das anthropic-SDK fuehrt keinen Fehlercode als Attribut.
    (_status(anthropic, 400, {"code": "content_filter"}), "status_400"),
    (_status(openai, 401), "status_401"),
    (_status(anthropic, 401), "status_401"),
    (_status(openai, 403), "status_403"),
    (_status(openai, 404), "status_404"),
    (_status(openai, 422), "status_422"),
]


@pytest.mark.parametrize(("fehler", "grund"), NICHT_ERREICHBAR)
def test_anbieterfehler_werden_zu_llm_nicht_erreichbar(fehler: Exception, grund: str) -> None:
    """Zeitlimit, Verbindung, 408, 409, 429 und ab 500, aus beiden SDKs."""
    ergebnis = _generate_mit(fehler)
    assert isinstance(ergebnis, LlmNichtErreichbar), repr(ergebnis)
    assert ergebnis.grund == grund
    assert ergebnis.__cause__ is fehler


@pytest.mark.parametrize(("fehler", "grund"), ABGELEHNT)
def test_anbieterfehler_werden_zu_llm_anfrage_abgelehnt(fehler: Exception, grund: str) -> None:
    """Jeder andere Status. Ein Inhaltsfilter ist kein Ausfall."""
    ergebnis = _generate_mit(fehler)
    assert isinstance(ergebnis, LlmAnfrageAbgelehnt), repr(ergebnis)
    assert not isinstance(ergebnis, LlmNichtErreichbar)
    assert ergebnis.grund == grund


@pytest.mark.parametrize(
    "fehler",
    [
        ValueError("kein Anbieterfehler"),
        openai.APIResponseValidationError(httpx.Response(200, request=_ANFRAGE), None),
        anthropic.APIResponseValidationError(httpx.Response(200, request=_ANFRAGE), None),
    ],
)
def test_andere_fehler_gehen_unveraendert_durch(fehler: Exception) -> None:
    """Was nicht eindeutig zugeordnet ist, bleibt dieselbe Ausnahme.

    Auch eine andere Unterklasse von APIError: Ein Validierungsfehler der
    Antwort ist kein Ausfall und keine Ablehnung, und eine Uebersetzung, die
    ihn schluckt, verbaerge einen Fehler in unserem Aufbau.
    """
    assert _generate_mit(fehler) is fehler
