"""Fixtures fuer die Mandantentests.

Keine Fixture setzt Umgebungsvariablen, und kein Test baut `Settings()`. Die
Mandantenfunktionen bekommen ihr Wurzelverzeichnis injiziert - dadurch laufen
diese Tests ohne `.env` und ohne LLM-Schluessel, mit denen sie nichts zu tun
haben.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

FAKE_DIMENSION = 16


class FakeBackend:
    """Deterministischer Embedder ohne Modell und ohne Netz.

    Bildet Woerter auf feste Positionen ab und normalisiert das Ergebnis auf
    Laenge 1 - dieselbe Voraussetzung, die ADR-008 an das echte Modell stellt.
    Aehnliche Texte bekommen aehnliche Vektoren, das reicht, um den Mechanismus
    zu pruefen: Ablage, Mandantentrennung, Score-Bereich.

    Was er NICHT kann, ist Semantik ueber Sprachgrenzen. Der Nachweis, dass eine
    deutsche Frage ein englisches Dokument findet, braucht das echte Modell und
    steht deshalb im Integrationstest.
    """

    def __init__(self, dimension: int = FAKE_DIMENSION, normalize: bool = True) -> None:
        self.dimension = dimension
        self.normalize = normalize
        self.seen: list[str] = []

    def _vector(self, text: str) -> list[float]:
        self.seen.append(text)
        werte = [0.0] * self.dimension
        for wort in text.lower().split():
            h = hashlib.sha256(wort.encode("utf-8")).digest()
            werte[h[0] % self.dimension] += 1.0
        if not any(werte):
            werte[0] = 1.0
        if self.normalize:
            laenge = math.sqrt(sum(v * v for v in werte))
            werte = [v / laenge for v in werte]
        return werte

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


class FakeLlm:
    """LLM-Ersatz ohne Netz.

    Zeichnet jeden Aufruf auf. Das ist der wichtigere Teil: Der Test, dass bei
    retrievalseitiger Eskalation KEIN Modell aufgerufen wird, laesst sich nur
    ueber `calls` fuehren. Ein `escalated=True` bei gleichzeitigem Aufruf waere
    ADR-003 formal erfuellt und inhaltlich verletzt.
    """

    def __init__(
        self,
        parsed: object | None = None,
        parsing_error: str | None = None,
        model: str = "fake-model-2026",
    ) -> None:
        self._parsed = parsed
        self._parsing_error = parsing_error
        self._model = model
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> object:
        from app.llm import LlmResult

        self.calls.append((system_prompt, user_prompt))
        return LlmResult(
            parsed=self._parsed,
            parsing_error=self._parsing_error,
            prompt_tokens=123,
            completion_tokens=45,
            model=self._model,
        )


def lege_mandant_an(
    root: Path,
    slug: str,
    text: str,
    token: str,
    escalation_message: str = "Dazu finde ich in den Unterlagen nichts.",
) -> None:
    """Legt einen Mandanten mit einem Dokument an. Fuer Tests gegen tmp_path."""
    tenant_dir = root / slug
    (tenant_dir / "docs").mkdir(parents=True)
    (tenant_dir / "tenant.yaml").write_text(
        yaml.safe_dump(
            {
                "display_name": slug,
                "languages": ["de"],
                "escalation_message": escalation_message,
                "url_token": token,
                "public_image_allowed": True,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (tenant_dir / "docs" / "doku.md").write_text(text, encoding="utf-8")


@pytest.fixture
def demo_tenants_dir() -> Path:
    """Das echte tenants/-Verzeichnis des Repos mit den beiden Demo-Mandanten."""
    return REPO_ROOT / "tenants"


@pytest.fixture
def tmp_tenants_dir(tmp_path: Path) -> Path:
    """Ein Wurzelverzeichnis mit einem Mandanten OHNE public_image_allowed.

    Bewusst nicht einer der Demo-Mandanten: Der Default-Test soll das Modell
    pruefen, nicht die YAML-Datei. Wuerde er gegen demo-acme laufen, pruefte er
    nur, dass dort `true` steht.
    """
    slug = "ohne-flag"
    tenant_dir = tmp_path / slug
    tenant_dir.mkdir()
    (tenant_dir / "tenant.yaml").write_text(
        yaml.safe_dump(
            {
                "display_name": "Ohne Flag",
                "languages": ["de"],
                "escalation_message": "Dazu finde ich nichts.",
                "url_token": "ohne-flag-token-1234567890",
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def tmp_tenants_dir_mit_doppeltem_token(tmp_path: Path) -> Path:
    """Zwei Mandanten mit demselben url_token - der kopierte-Datei-Fehler."""
    token = "geteiltes-token-1234567890"
    for slug in ("erster-mandant", "zweiter-mandant"):
        tenant_dir = tmp_path / slug
        tenant_dir.mkdir()
        (tenant_dir / "tenant.yaml").write_text(
            yaml.safe_dump(
                {
                    "display_name": slug,
                    "languages": ["de"],
                    "escalation_message": "Dazu finde ich nichts.",
                    "url_token": token,
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    return tmp_path
