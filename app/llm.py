"""LLM-Zugang, drei Provider hinter einer Schnittstelle.

ADR-006: OpenAI lokal, Azure OpenAI im Deploy, Anthropic als dritter, damit
Austauschbarkeit vorfuehrbar statt behauptet ist. **Im aufrufenden Code gibt es
keine Verzweigung nach Provider** - ein `if provider == "azure"` ausserhalb
dieser Datei ist ein Fehler, kein Stil.

Die Tokenzahlen gehen von Anfang an mit. Phase 6 braucht sie, und sie
nachtraeglich einzuziehen hiesse, jede Aufrufstelle anzufassen.

Alle drei Provider bekommen `temperature` aus den Settings. Seit dem 2026-09-24
und mit Absicht: Ohne den Parameter entschied der Anbieter, und eine Aenderung
seines Standards haette jede Messung dieses Projekts verschoben, ohne dass hier
eine Zeile anders wird. Siehe `Settings.llm_temperature`.

Ebenso Zeitlimit und Wiederholungen, siehe `Settings.llm_timeout_s`. Und nur
diese Datei kennt die Fehlerklassen der Anbieter: Sie uebersetzt sie in
`LlmNichtErreichbar` und `LlmAnfrageAbgelehnt`, und die Route entscheidet
danach, ohne openai oder anthropic zu importieren.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.prompts import EigenstaendigeFrage, GroundedAnswer


class LlmConfigError(ValueError):
    """Die Konfiguration des gewaehlten Providers ist unvollstaendig.

    Wird beim Erzeugen geworfen, nicht beim Aufruf. Ein fehlender Endpunkt darf
    nicht erst auffallen, wenn ein Interessent bereits eine Frage gestellt hat.
    """


class LlmAufrufFehler(Exception):
    """Gemeinsame Basis der beiden uebersetzten Anbieterfehler.

    `grund` ist ein fester Code und nie Text des Anbieters: Er geht ins
    Ereignis, und eine Fehlermeldung des Anbieters kann Teile der Anfrage
    zitieren.
    """

    def __init__(self, grund: str) -> None:
        super().__init__(grund)
        self.grund = grund
        # Welcher Aufruf gescheitert ist: "antwort" oder, bei einer Folgefrage,
        # "umformung". Setzt der Aufrufer, der die Umformung faehrt; das
        # Ereignis traegt es als festen Code.
        self.aufruf = "antwort"


class LlmNichtErreichbar(LlmAufrufFehler):
    """Der Anbieter ist gerade nicht erreichbar oder ueberlastet.

    Gruende: `zeitlimit`, `verbindung`, `status_408`, `status_409`,
    `status_429` und `status_<code>` ab 500. Ein erneuter Versuch spaeter kann
    gelingen, deshalb sagt die Oberflaeche genau das.
    """


class LlmAnfrageAbgelehnt(LlmAufrufFehler):
    """Der Anbieter hat diese Anfrage abgewiesen, und ein erneuter Versuch
    aendert daran nichts.

    Gruende: `inhaltsfilter`, wenn Azure die Eingabe mit seinem Inhaltsfilter
    blockt (HTTP 400, Fehlercode content_filter), sonst `status_<code>`, etwa
    `status_401` bei einem falschen Schluessel. "Gerade nicht erreichbar" waere
    hier eine falsche Auskunft.
    """


# Status, bei denen ein spaeterer Versuch gelingen kann. 408 Zeitueberschreitung
# beim Anbieter, 409 Konflikt, 429 Ratenlimit, ab 500 Fehler auf seiner Seite.
_VORUEBERGEHEND = frozenset({408, 409, 429})


def anbieterfehler_uebersetzen(fehler: Exception) -> LlmAufrufFehler | None:
    """Ordnet einen Fehler des Anbieters einer der beiden Ausnahmen zu.

    Gibt None zurueck fuer alles, was hier nicht zugeordnet wird, auch fuer
    andere Unterklassen von APIError wie einen Validierungsfehler der Antwort.
    Der Aufrufer wirft diese dann unveraendert weiter: Was nicht eindeutig
    zugeordnet ist, soll als Fehler sichtbar bleiben und nicht als "gerade
    nicht erreichbar" verschwinden.

    Die Reihenfolge zaehlt: Das Zeitlimit ist in beiden SDKs eine Unterklasse
    des Verbindungsfehlers.
    """
    if isinstance(fehler, (openai.APITimeoutError, anthropic.APITimeoutError)):
        return LlmNichtErreichbar("zeitlimit")
    if isinstance(fehler, (openai.APIConnectionError, anthropic.APIConnectionError)):
        return LlmNichtErreichbar("verbindung")
    if isinstance(fehler, (openai.APIStatusError, anthropic.APIStatusError)):
        status = fehler.status_code
        if status in _VORUEBERGEHEND or status >= 500:
            return LlmNichtErreichbar(f"status_{status}")
        # Nur das openai-SDK fuehrt den Fehlercode aus dem Rumpf als Attribut.
        # Azure antwortet mit 400 und content_filter, wenn sein Inhaltsfilter
        # eine Eingabe blockt.
        if (
            status == 400
            and isinstance(fehler, openai.APIStatusError)
            and fehler.code == "content_filter"
        ):
            return LlmAnfrageAbgelehnt("inhaltsfilter")
        return LlmAnfrageAbgelehnt(f"status_{status}")
    return None


@dataclass(frozen=True)
class LlmResult:
    """Ergebnis eines Modellaufrufs."""

    # None, wenn das Modell keine gueltige Struktur geliefert hat. Dann traegt
    # parsing_error die Begruendung. Das ist ein eigener Eskalationsweg - ohne
    # ihn entstuende aus einem Parsefehler eine leere Antwort, still.
    parsed: GroundedAnswer | None
    parsing_error: str | None

    prompt_tokens: int | None
    completion_tokens: int | None

    # Das Modell, das TATSAECHLICH geantwortet hat, aus der API-Antwort. Nicht
    # der konfigurierte Name: Ein Alias loest serverseitig oft auf einen
    # datierten Schnappschuss auf, und fuer Telemetrie (ADR-002) und Phase 5
    # zaehlt der aufgeloeste Wert.
    model: str


class LlmClient(Protocol):
    """Was der aufrufende Code von einem Provider braucht."""

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult: ...


class StructuredChatClient:
    """Ruft ein Chat-Modell auf und erzwingt die GroundedAnswer-Struktur.

    Setzt Tool- beziehungsweise Function-Calling voraus, weil
    `with_structured_output` darauf beruht. Ein Modell ohne das liefert dauerhaft
    einen Parsefehler - dann greift der dritte Eskalationsweg bei jeder Anfrage,
    und das faellt sofort auf.
    """

    def __init__(
        self,
        chat_model: BaseChatModel,
        configured_model: str,
        answer_model: type[GroundedAnswer],
    ) -> None:
        # include_raw=True liefert ein dict mit "raw", "parsed" und
        # "parsing_error". Nur so sind Struktur UND Tokenzahlen aus einem
        # einzigen Aufruf zu haben.
        #
        # `answer_model` traegt die Feldbeschreibungen in der Sprache des
        # Mandanten. Es ist ein Pflichtargument und kein Standardwert: Die
        # Basisklasse GroundedAnswer hat gar keine Beschreibungen, und ein
        # Standardwert waere der Weg, auf dem wieder ein Schema ohne oder in der
        # falschen Sprache beim Modell landet.
        self._runnable = chat_model.with_structured_output(answer_model, include_raw=True)
        self._configured_model = configured_model

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
        ergebnis = _strukturiert_aufrufen(self._runnable, system_prompt, user_prompt)

        roh = ergebnis.get("raw")
        parsed = ergebnis.get("parsed")
        fehler = ergebnis.get("parsing_error")

        verbrauch = getattr(roh, "usage_metadata", None) or {}
        metadaten = getattr(roh, "response_metadata", None) or {}

        return LlmResult(
            parsed=parsed if isinstance(parsed, GroundedAnswer) else None,
            parsing_error=str(fehler) if fehler is not None else None,
            prompt_tokens=verbrauch.get("input_tokens"),
            completion_tokens=verbrauch.get("output_tokens"),
            model=metadaten.get("model_name") or self._configured_model,
        )


def _require(settings: Settings, felder: dict[str, str | None], provider: str) -> None:
    """Prueft, dass alle noetigen Felder gesetzt sind, und nennt die fehlenden."""
    fehlend = sorted(name for name, wert in felder.items() if not wert)
    if fehlend:
        raise LlmConfigError(
            f"LLM_PROVIDER ist '{provider}', aber diese Werte fehlen in .env: "
            f"{', '.join(fehlend)}. Sie werden ueber pydantic-settings gelesen; "
            f"Schluessel gehoeren nie in die Shell (pitfalls.md, P-002)."
        )


def _strukturiert_aufrufen(
    runnable: Any,  # noqa: ANN401
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Ruft ein strukturiertes Chatmodell auf und uebersetzt Anbieterfehler.

    Gemeinsam fuer Antwort und Umformung, damit beide dieselben zwei Ausnahmen
    kennen und nur diese Datei die Fehlerklassen der SDKs.
    """
    try:
        ergebnis: dict[str, Any] = runnable.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except (openai.APIError, anthropic.APIError) as fehler:
        uebersetzt = anbieterfehler_uebersetzen(fehler)
        if uebersetzt is None:
            raise
        raise uebersetzt from fehler
    return ergebnis


@dataclass(frozen=True)
class UmformungsErgebnis:
    """Ergebnis der Umformung einer Folgefrage."""

    # None, wenn das Modell keine gueltige Struktur geliefert hat.
    frage: str | None
    parsing_error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    model: str


class Umformer(Protocol):
    """Was der aufrufende Code fuer die Umformung braucht."""

    def umformen(self, system_prompt: str, user_prompt: str) -> UmformungsErgebnis: ...


class StructuredUmformer:
    """Formt eine Folgefrage um und erzwingt eine Struktur mit genau einem Feld."""

    def __init__(
        self,
        chat_model: BaseChatModel,
        configured_model: str,
        schema: type[EigenstaendigeFrage],
    ) -> None:
        self._runnable = chat_model.with_structured_output(schema, include_raw=True)
        self._configured_model = configured_model

    def umformen(self, system_prompt: str, user_prompt: str) -> UmformungsErgebnis:
        ergebnis = _strukturiert_aufrufen(self._runnable, system_prompt, user_prompt)
        roh = ergebnis.get("raw")
        parsed = ergebnis.get("parsed")
        fehler = ergebnis.get("parsing_error")
        verbrauch = getattr(roh, "usage_metadata", None) or {}
        metadaten = getattr(roh, "response_metadata", None) or {}
        return UmformungsErgebnis(
            frage=parsed.question if isinstance(parsed, EigenstaendigeFrage) else None,
            parsing_error=str(fehler) if fehler is not None else None,
            prompt_tokens=verbrauch.get("input_tokens"),
            completion_tokens=verbrauch.get("output_tokens"),
            model=metadaten.get("model_name") or self._configured_model,
        )


def _chatmodell(settings: Settings, model_override: str | None) -> tuple[BaseChatModel, str]:
    """Baut das Chatmodell des konfigurierten Providers, samt Modellname.

    Gemeinsam fuer `build_llm` und `build_umformer`: Beide Aufrufe einer
    Folgefrage laufen mit demselben Modell, demselben Zeitlimit und denselben
    Wiederholungen. Nach Provider verzweigt nur diese Funktion.

    Geprueft wird die VOLLSTAENDIGE Konfiguration des Providers, nicht nur der
    Schluessel. Bei Azure fehlten sonst Endpunkt und API-Version, und es
    scheiterte erst beim Aufruf statt beim Start.

    `timeout` und `max_retries` gehen an alle drei Clients, unter diesen Namen.
    Fehlt einer, wartet dieser Provider unbegrenzt beziehungsweise wiederholt
    zweimal, und das faellt erst bei einem Ausfall auf.
    """
    provider = settings.llm_provider
    modell = model_override or settings.model_name

    if provider == "openai":
        _require(settings, {"OPENAI_API_KEY": settings.openai_api_key}, provider)
        from langchain_openai import ChatOpenAI

        return (
            ChatOpenAI(
                model=modell,
                api_key=settings.openai_api_key,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            ),
            modell,
        )

    if provider == "azure":
        _require(
            settings,
            {
                "AZURE_OPENAI_API_KEY": settings.azure_openai_api_key,
                "AZURE_OPENAI_ENDPOINT": settings.azure_openai_endpoint,
                "AZURE_OPENAI_API_VERSION": settings.azure_openai_api_version,
            },
            provider,
        )
        from langchain_openai import AzureChatOpenAI

        return (
            AzureChatOpenAI(
                azure_deployment=modell,
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            ),
            modell,
        )

    if provider == "anthropic":
        _require(settings, {"ANTHROPIC_API_KEY": settings.anthropic_api_key}, provider)
        from langchain_anthropic import ChatAnthropic

        return (
            ChatAnthropic(
                model=modell,
                api_key=settings.anthropic_api_key,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout_s,
                max_retries=settings.llm_max_retries,
            ),
            modell,
        )

    raise LlmConfigError(f"Unbekannter LLM_PROVIDER: {provider!r}.")


def build_llm(
    settings: Settings,
    model_override: str | None = None,
    *,
    answer_model: type[GroundedAnswer],
) -> LlmClient:
    """Baut den Client fuer den konfigurierten Provider.

    Bewusst eine Funktion und kein Modul-Level-Objekt. `model_override`
    stammt aus der TenantConfig und sticht die globale Einstellung.

    `answer_model` kommt aus `antwortmodell_fuer(tenant.language)` und traegt die
    Feldbeschreibungen des Strukturschemas. Nur mit Schluesselwort und ohne
    Standardwert, damit an keiner Aufrufstelle vergessen werden kann, in welcher
    Sprache das Modell angesprochen wird.

    Provider, Pruefung der Konfiguration, Zeitlimit und Wiederholungen stehen in
    `_chatmodell`.
    """
    chat_model, modell = _chatmodell(settings, model_override)
    return StructuredChatClient(chat_model, modell, answer_model)


def build_umformer(
    settings: Settings,
    model_override: str | None = None,
    *,
    schema: type[EigenstaendigeFrage],
) -> Umformer:
    """Baut den Client fuer die Umformung einer Folgefrage.

    Dasselbe Modell wie fuer die Antwort, aus `_chatmodell`. `schema` kommt aus
    `umformungsmodell_fuer(tenant.language)`, nur mit Schluesselwort und ohne
    Standardwert, aus demselben Grund wie `answer_model` bei `build_llm`.
    """
    chat_model, modell = _chatmodell(settings, model_override)
    return StructuredUmformer(chat_model, modell, schema)
