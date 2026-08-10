"""Konfiguration der Anwendung.

Einzige Quelle fuer Konfiguration. Kein `os.environ`, kein `os.getenv` ausserhalb
dieser Datei (CLAUDE.md).

Die Feldnamen entsprechen exakt den Variablennamen in `.env.example`. Bei einem
Konflikt zwischen einem Planungsdokument und einer committeten Datei gewinnt die
committete Datei (knowledge/conventions.md).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LlmProvider = Literal["openai", "azure", "anthropic"]

# Welcher Schluessel fuer welchen Provider Pflicht ist. Siehe ADR-006.
_REQUIRED_KEY_PER_PROVIDER: dict[str, str] = {
    "openai": "openai_api_key",
    "azure": "azure_openai_api_key",
    "anthropic": "anthropic_api_key",
}

# Welches Feld den Modellnamen je Provider traegt. Siehe ADR-006.
_MODEL_FIELD_PER_PROVIDER: dict[str, str] = {
    "openai": "openai_model",
    "azure": "azure_openai_deployment",
    "anthropic": "anthropic_model",
}


class Settings(BaseSettings):
    """Laufzeitkonfiguration der Anwendung.

    Nicht enthalten sind die Deployment-Variablen aus `.env.example`
    (`AZURE_SUBSCRIPTION_ID`, `GHCR_IMAGE` und weitere). Die gehoeren zu
    Infrastruktur und CI, nicht zur laufenden Anwendung. `extra="ignore"` sorgt
    dafuer, dass sie in `.env` stehen duerfen, ohne hier aufzutauchen.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Anwendung ---------------------------------------------------------
    app_env: str = "local"
    log_level: str = "INFO"

    # --- LLM-Provider ------------------------------------------------------
    # Default openai, weil ADR-006 OpenAI als Provider fuer die lokale
    # Entwicklung festlegt.
    llm_provider: LlmProvider = "openai"

    openai_api_key: str | None = None
    openai_model: str | None = None

    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_region: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_deployment: str | None = None

    anthropic_api_key: str | None = None
    anthropic_model: str | None = None

    # --- Embeddings --------------------------------------------------------
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32

    # --- Retrieval ---------------------------------------------------------
    # retrieval_score_threshold hat BEWUSST keinen Default. Ein geschaetzter
    # Schwellwert ist wertlos (pitfalls.md, P-005), und der Startwert ist noch
    # nicht kalibriert (open-points.md, OP-004).
    #
    # Wichtig fuer den Retrieval-Pfad: Ist der Wert None, muss laut abgebrochen
    # werden. Eine defensive Formulierung wie
    #     if threshold is not None and score < threshold
    # laesst die Eskalation nie feuern und niemand merkt es - dieselbe stille
    # Umkehr wie bei rohen Distanzen (ADR-008). Siehe open-points.md, OP-013.
    retrieval_score_threshold: float | None = None
    retrieval_top_k: int = 4

    chunk_size: int = 1000
    chunk_overlap: int = 150

    # --- Pfade -------------------------------------------------------------
    tenants_dir: Path = Path("tenants")
    index_dir: Path = Path("data/index")

    # --- Telemetrie --------------------------------------------------------
    # Instrumentierung ist OpenTelemetry, Ziel ist Application Insights
    # (ADR-012). Die beiden OTLP-Felder stehen NEBEN dem Azure-Pfad, nicht an
    # seiner Stelle.
    telemetry_enabled: bool = False
    applicationinsights_connection_string: str | None = None
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "rag-support-demo"

    # `Any` ist hier nicht zu vermeiden und deshalb nach conventions.md begruendet:
    # Ein Validator mit mode="before" bekommt die Rohdaten, so wie sie
    # hereinkommen. Das ist ueblicherweise ein dict aus Umgebung und .env, kann
    # aber auch eine Instanz oder ein beliebiges Mapping sein. Eine engere
    # Annotation waere schlicht falsch.
    @model_validator(mode="before")
    @classmethod
    def _leere_werte_als_nicht_gesetzt(cls, data: Any) -> Any:  # noqa: ANN401
        """Leere Zeichenketten wie fehlende Werte behandeln.

        `.env.example` fuehrt alle Variablen mit leerem Wert. Wer die Datei nach
        `.env` kopiert - wozu ihr eigener Kopf auffordert - und nur den
        API-Schluessel eintraegt, haette sonst `RETRIEVAL_TOP_K=""` und damit
        einen Typfehler beim Start, obwohl ein brauchbarer Default existiert.

        Eine leere Zeichenkette ist keine Angabe. Sie wird entfernt, damit der
        Default greift.
        """
        if not isinstance(data, dict):
            return data
        return {k: v for k, v in data.items() if not (isinstance(v, str) and v.strip() == "")}

    @model_validator(mode="after")
    def _schluessel_des_gewaehlten_providers_pflicht(self) -> Settings:
        """Erzwingt den API-Schluessel des ausgewaehlten Providers.

        CLAUDE.md verlangt, dass ein fehlendes Secret den Start laut und sofort
        scheitern laesst. Geprueft wird aber nur der Schluessel des Providers aus
        `LLM_PROVIDER` - Schluessel fuer die beiden anderen zu verlangen, die man
        gar nicht benutzt, waere Schikane ohne Sicherheitsgewinn.
        """
        feld = _REQUIRED_KEY_PER_PROVIDER[self.llm_provider]
        if not getattr(self, feld):
            env_name = feld.upper()
            raise ValueError(
                f"LLM_PROVIDER ist '{self.llm_provider}', aber {env_name} ist nicht gesetzt. "
                f"Der Schluessel gehoert in .env und wird ueber pydantic-settings gelesen."
            )
        return self

    @property
    def model_name(self) -> str:
        """Modellname des aktiven Providers.

        Ersetzt ein einzelnes `MODEL_NAME`: Nach ADR-006 traegt jeder Provider
        sein eigenes Feld, und `.env.example` fuehrt sie getrennt. Der aufrufende
        Code verzweigt nicht nach Provider - das ist genau die Verzweigung, die
        ADR-006 ausserhalb der Factory verbietet.
        """
        feld = _MODEL_FIELD_PER_PROVIDER[self.llm_provider]
        wert = getattr(self, feld)
        if not wert:
            env_name = feld.upper()
            raise ValueError(
                f"LLM_PROVIDER ist '{self.llm_provider}', aber {env_name} ist nicht gesetzt."
            )
        return str(wert)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Liefert die Konfiguration, einmal gebaut und danach zwischengespeichert.

    Bewusst eine Funktion und kein Modul-Level-Objekt: Ein `settings = Settings()`
    beim Import laesst jeden Import scheitern, sobald eine Variable fehlt - auch in
    Tests, die mit Konfiguration nichts zu tun haben.

    Zum ADR-001-Hook: Er warnt hier wegen `@lru_cache` ohne `tenant_id` im
    Parameter. Das ist ein bekannter Fehlalarm - `Settings` ist
    mandantenunabhaengig, der Cache teilt keinen Mandantenzustand. Der Fehlalarm
    ist in knowledge/pitfalls.md, P-014 verzeichnet; ein ZWEITER Fehlalarm
    desselben Hooks waere kein Dokumentationsfall mehr, sondern ein Auftrag, den
    Hook zu praezisieren.
    """
    return Settings()
