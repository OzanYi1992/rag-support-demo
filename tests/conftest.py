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


class FakeLlmFolge:
    """LLM-Ersatz ohne Netz, der je Aufruf das naechste Ergebnis der Folge liefert.

    Fuer den zweiten Aufruf zur Antwortsprache: Erster und zweiter Aufruf sollen
    verschieden antworten koennen. Ein Eintrag ist eine Antwortstruktur, None
    fuer eine nicht lesbare Struktur, oder eine Ausnahme, die geworfen wird.

    Zeichnet jeden Aufruf auf. Ein Aufruf mehr als vorgesehen ist ein Fehler im
    Test und kein stiller Rueckfall auf den letzten Eintrag: Genau so faellt ein
    dritter Aufruf auf. Die Tokens unterscheiden sich je Aufruf, 100 und 10 plus
    die Nummer ab 0, damit ein Test sieht, welche Zahl zu welchem Aufruf gehoert.
    """

    def __init__(self, *folge: object, model: str = "fake-model-2026") -> None:
        self._folge = list(folge)
        self._model = model
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_prompt: str, user_prompt: str) -> object:
        from app.llm import LlmResult

        nummer = len(self.calls)
        self.calls.append((system_prompt, user_prompt))
        if nummer >= len(self._folge):
            raise AssertionError(f"Aufruf {nummer + 1} ist nicht vorgesehen.")
        eintrag = self._folge[nummer]
        if isinstance(eintrag, BaseException):
            raise eintrag
        return LlmResult(
            parsed=eintrag,  # type: ignore[arg-type]
            parsing_error=None if eintrag is not None else "kein JSON",
            prompt_tokens=100 + nummer,
            completion_tokens=10 + nummer,
            model=self._model,
        )


class FakeUmformer:
    """Umformer-Ersatz ohne Netz, fuer Folgefragen.

    Liefert eine feste eigenstaendige Frage, oder None wie bei einem
    Parsefehler, oder wirft `fehler`. Zeichnet jeden Aufruf auf: Ob die
    Umformung ueberhaupt lief, laesst sich nur ueber `calls` pruefen.
    """

    def __init__(
        self,
        frage: str | None = None,
        fehler: Exception | None = None,
        model: str = "fake-model-2026",
    ) -> None:
        self._frage = frage
        self._fehler = fehler
        self._model = model
        self.calls: list[tuple[str, str]] = []

    def umformen(self, system_prompt: str, user_prompt: str) -> object:
        from app.llm import UmformungsErgebnis

        self.calls.append((system_prompt, user_prompt))
        if self._fehler is not None:
            raise self._fehler
        return UmformungsErgebnis(
            frage=self._frage,
            parsing_error=None if self._frage is not None else "kein JSON",
            prompt_tokens=211,
            completion_tokens=17,
            model=self._model,
        )


# Eskalationsnachrichten je Sprache. Der Default haengt an `language`, weil ein
# Mandant mit `language: en` und einer deutschen Nachricht seit EN-2 beim Laden
# abgewiesen wird - zu Recht. Ein fester deutscher Default haette jeden
# englischen Testmandanten unbaubar gemacht.
STANDARD_ESKALATION: dict[str, str] = {
    "de": "Dazu finde ich in den Unterlagen nichts.",
    "en": "I cannot find anything about that in the documents.",
}

# Dasselbe fuer `topics`, aus demselben Grund: Das Feld ist Pflicht, und es wird
# beim Laden gegen die Sprache des Mandanten geprueft. Ein fester deutscher Text
# haette jeden englischen Testmandanten unbaubar gemacht.
STANDARD_THEMEN: dict[str, str] = {
    "de": "Versand, Retouren und Zahlung",
    "en": "shipping, returns and payment",
}


def lege_mandant_an(
    root: Path,
    slug: str,
    text: str,
    token: str,
    escalation_message: str | None = None,
    language: str = "de",
    topics: str | None = None,
) -> None:
    """Legt einen Mandanten mit einem Dokument an. Fuer Tests gegen tmp_path.

    `language` ist der Schalter fuer Oberflaechensprache und Prompt-Regelwerk.
    Default "de", damit bestehende Aufrufe unveraendert bleiben.

    `escalation_message` folgt der Sprache, wenn nichts uebergeben wird. Wer
    einen Text uebergibt, ist selbst dafuer verantwortlich, dass er zur Sprache
    passt - auch das pruefen Tests, dort absichtlich mit falschem Text.
    """
    if escalation_message is None:
        escalation_message = STANDARD_ESKALATION[language]
    if topics is None:
        topics = STANDARD_THEMEN[language]
    tenant_dir = root / slug
    (tenant_dir / "docs").mkdir(parents=True)
    (tenant_dir / "tenant.yaml").write_text(
        yaml.safe_dump(
            {
                "display_name": slug,
                "language": language,
                "topics": topics,
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
                "language": "de",
                "topics": "Versand und Retouren",
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
                    "language": "de",
                    "topics": "Versand und Retouren",
                    "escalation_message": "Dazu finde ich nichts.",
                    "url_token": token,
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    return tmp_path
