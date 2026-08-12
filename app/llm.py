"""LLM-Zugang, drei Provider hinter einer Schnittstelle.

ADR-006: OpenAI lokal, Azure OpenAI im Deploy, Anthropic als dritter, damit
Austauschbarkeit vorfuehrbar statt behauptet ist. **Im aufrufenden Code gibt es
keine Verzweigung nach Provider** - ein `if provider == "azure"` ausserhalb
dieser Datei ist ein Fehler, kein Stil.

Die Tokenzahlen gehen von Anfang an mit. Phase 6 braucht sie, und sie
nachtraeglich einzuziehen hiesse, jede Aufrufstelle anzufassen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import Settings
from app.prompts import GroundedAnswer


class LlmConfigError(ValueError):
    """Die Konfiguration des gewaehlten Providers ist unvollstaendig.

    Wird beim Erzeugen geworfen, nicht beim Aufruf. Ein fehlender Endpunkt darf
    nicht erst auffallen, wenn ein Interessent bereits eine Frage gestellt hat.
    """


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

    def __init__(self, chat_model: BaseChatModel, configured_model: str) -> None:
        # include_raw=True liefert ein dict mit "raw", "parsed" und
        # "parsing_error". Nur so sind Struktur UND Tokenzahlen aus einem
        # einzigen Aufruf zu haben.
        self._runnable = chat_model.with_structured_output(GroundedAnswer, include_raw=True)
        self._configured_model = configured_model

    def generate(self, system_prompt: str, user_prompt: str) -> LlmResult:
        ergebnis: dict[str, Any] = self._runnable.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )

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


def build_llm(settings: Settings, model_override: str | None = None) -> LlmClient:
    """Baut den Client fuer den konfigurierten Provider.

    Bewusst eine Funktion und kein Modul-Level-Objekt (ADR-001). `model_override`
    stammt aus der TenantConfig und sticht die globale Einstellung.

    Geprueft wird die VOLLSTAENDIGE Konfiguration des Providers, nicht nur der
    Schluessel. Bei Azure fehlten sonst Endpunkt und API-Version, und es
    scheiterte erst beim Aufruf statt beim Start.
    """
    provider = settings.llm_provider
    modell = model_override or settings.model_name

    if provider == "openai":
        _require(settings, {"OPENAI_API_KEY": settings.openai_api_key}, provider)
        from langchain_openai import ChatOpenAI

        return StructuredChatClient(
            ChatOpenAI(model=modell, api_key=settings.openai_api_key), modell
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

        return StructuredChatClient(
            AzureChatOpenAI(
                azure_deployment=modell,
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            ),
            modell,
        )

    if provider == "anthropic":
        _require(settings, {"ANTHROPIC_API_KEY": settings.anthropic_api_key}, provider)
        from langchain_anthropic import ChatAnthropic

        return StructuredChatClient(
            ChatAnthropic(model=modell, api_key=settings.anthropic_api_key), modell
        )

    raise LlmConfigError(f"Unbekannter LLM_PROVIDER: {provider!r}.")
