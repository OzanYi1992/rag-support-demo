"""Durchsetzung von ADR-020: geteilter, ZUSTANDSLOSER Embedder.

ADR-020 erlaubt das Teilen auf Grundlage von zwei Eigenschaften:
mandantenunabhaengig und zustandslos. Die erste ist strukturell - dieselbe
Zeichenkette ergibt fuer jeden Mandanten denselben Vektor. Die zweite ist eine
Zusicherung, und eine Zusicherung ohne Test ist eine Behauptung (P-009, P-010).

Diese Datei ist die Durchsetzung. Zwei der Tests werden ROT, wenn jemand einen
Cache ueber encodierte Texte einbaut - genau der Fall, in dem der Embedder
Mandanteninhalte hielte und ADR-001 mit voller Haerte griffe.

WER DIESE TESTS REPARIEREN WILL, WEIL SIE STOEREN: Sie stoeren nicht, sie
melden. Ein Cache ueber Texte ist nach ADR-020 nicht zulaessig. Wenn er
gebraucht wird, gehoert die Entscheidung in einen neuen ADR, der ADR-020
abloest - nicht in eine Anpassung dieser Datei.

Alle Tests laufen mit FakeBackend. Kein Modell, kein Netz, kein Download.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.embeddings import (
    E5Embeddings,
    _embedder_cache_leeren,
    build_embeddings,
    get_embeddings,
)
from tests.conftest import FAKE_DIMENSION, FakeBackend

PFLICHT = {
    "llm_provider": "openai",
    "openai_api_key": "platzhalter",
    "openai_model": "platzhalter",
}


@pytest.fixture(autouse=True)
def cache_leeren() -> None:
    """Jeder Test startet mit leerem Cache, sonst haengen sie voneinander ab."""
    _embedder_cache_leeren()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        **PFLICHT,
        tenants_dir=tmp_path / "tenants",
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
    )


def _fake_bauen(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Ersetzt die Fabrik durch eine, die FakeBackend liefert und mitzaehlt.

    Der Zaehler ist der eigentliche Nachweis: Er zeigt, wie oft wirklich gebaut
    wurde, statt nur Objektidentitaet zu vergleichen.
    """
    zaehler = [0]

    def fake_build(s: Settings) -> E5Embeddings:
        zaehler[0] += 1
        return E5Embeddings(FakeBackend(), expected_dimension=s.embedding_dimension)

    monkeypatch.setattr("app.embeddings.build_embeddings", fake_build)
    return zaehler


# --- 1. Identitaet ----------------------------------------------------------


def test_gleiche_konfiguration_liefert_dasselbe_objekt(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    zaehler = _fake_bauen(monkeypatch)

    erster = get_embeddings(settings)
    zweiter = get_embeddings(settings)

    assert erster is zweiter
    assert zaehler[0] == 1, "Der Embedder wurde mehr als einmal gebaut."


def test_andere_konfiguration_liefert_ein_anderes_objekt(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Der Cache-Schluessel muss die embedding-relevanten Werte enthalten.

    Ohne diesen Test wuerde ein Modellwechsel stillschweigend den alten Embedder
    weiterbenutzen - und der Index waere gegen ein anderes Modell gebaut als die
    Anfrage (ADR-016).
    """
    zaehler = _fake_bauen(monkeypatch)

    erster = get_embeddings(settings)
    anderes_modell = settings.model_copy(update={"embedding_model": "ein/anderes"})
    zweiter = get_embeddings(anderes_modell)

    assert erster is not zweiter
    assert zaehler[0] == 2


def test_llm_modell_aendert_den_embedder_nicht(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zwei Konfigurationen, die sich nur im LLM unterscheiden, teilen sich den
    Embedder. Sonst laedt ein Providerwechsel 0,47 GB ohne Grund neu."""
    zaehler = _fake_bauen(monkeypatch)

    get_embeddings(settings)
    get_embeddings(settings.model_copy(update={"openai_model": "ein-anderes-llm"}))

    assert zaehler[0] == 1


# --- 2. Kein Zustandswachstum -----------------------------------------------


def _zustandsabdruck(objekt: object) -> dict[str, Any]:
    """Attributnamen und die Groessen aller Container-Attribute.

    Waechst hier irgendetwas durch blosses Encodieren, haelt das Objekt etwas
    fest - und das darf es nach ADR-020 nicht.
    """
    abdruck: dict[str, Any] = {}
    for name, wert in vars(objekt).items():
        abdruck[f"{name}:typ"] = type(wert).__name__
        if isinstance(wert, (dict, list, set, tuple, frozenset)):
            abdruck[f"{name}:groesse"] = len(wert)
    return abdruck


def test_encodieren_laesst_den_zustand_unveraendert() -> None:
    """Der Test, der rot wird, wenn jemand einen Cache einbaut."""
    embedder = E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION)

    # Einmal encodieren, damit einmalige Initialisierung (z.B. _checked) bereits
    # passiert ist und nicht als Wachstum zaehlt.
    embedder.embed_query("aufwaermen")
    vorher = _zustandsabdruck(embedder)

    for i in range(25):
        embedder.embed_query(f"eine voellig andere Frage Nummer {i}")
        embedder.embed_documents([f"ein voellig anderer Abschnitt Nummer {i}"])

    nachher = _zustandsabdruck(embedder)

    assert nachher == vorher, (
        "Der Zustand des Embedders hat sich durch Encodieren veraendert. "
        "Nach ADR-020 haelt er Modellgewichte und Konfiguration, sonst nichts. "
        "Ein Cache ueber encodierte Texte hielte Mandanteninhalte und faellt "
        "unter ADR-001."
    )


# --- 3. Kein encodierter Text bleibt haengen --------------------------------


def _erreichbarer_text(objekt: object, tiefe: int = 3) -> str:
    """Sammelt den erreichbaren Zustand als Zeichenkette.

    Begrenzt in der Tiefe, weil unter dem Backend ein vollstaendiges Modell
    haengen kann. Fuer den Zweck reicht das: Ein Cache wuerde im Wrapper oder
    unmittelbar darunter liegen, nicht in den Modellgewichten.
    """
    if tiefe < 0:
        return ""
    teile: list[str] = []
    if isinstance(objekt, (str, bytes)):
        return str(objekt)
    if isinstance(objekt, dict):
        for k, v in objekt.items():
            teile.append(_erreichbarer_text(k, tiefe - 1))
            teile.append(_erreichbarer_text(v, tiefe - 1))
        return " ".join(teile)
    if isinstance(objekt, (list, tuple, set, frozenset)):
        for v in objekt:
            teile.append(_erreichbarer_text(v, tiefe - 1))
        return " ".join(teile)
    if hasattr(objekt, "__dict__"):
        return _erreichbarer_text(vars(objekt), tiefe - 1)
    return ""


MARKER = "ZWEIUNDVIERZIG-EINZIGARTIGER-MARKER-4711"


def test_encodierter_text_bleibt_nicht_im_embedder() -> None:
    """Der zweite Test, der rot wird, wenn jemand einen Cache einbaut."""
    embedder = E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION)

    embedder.embed_query(MARKER)
    embedder.embed_documents([MARKER + "-ALS-ABSCHNITT"])

    assert MARKER not in _erreichbarer_text(embedder), (
        "Ein encodierter Text ist im Embedder haengen geblieben. "
        "Damit haelt ein prozessweit geteiltes Objekt Mandanteninhalte - "
        "ADR-001 greift, und ADR-020 traegt nicht mehr."
    )


# --- Gegenprobe: prüfen die beiden Tests wirklich? --------------------------


class _EmbedderMitCache(E5Embeddings):
    """Bewusst zustandsbehaftet. Existiert NUR fuer die Gegenprobe.

    Genau das, was ADR-020 verbietet: ein Cache ueber encodierte Texte. Wuerde
    jemand so etwas im Anwendungscode einbauen, muessten die beiden Tests oben
    anschlagen. Dass sie es bei diesem Objekt tun, ist der Nachweis, dass sie
    pruefen und nicht nur zusehen (conventions.md, Regel 1).
    """

    def __init__(self, backend: object, expected_dimension: int) -> None:
        super().__init__(backend, expected_dimension=expected_dimension)
        self._text_cache: dict[str, list[float]] = {}

    def embed_query(self, text: str) -> list[float]:
        if text not in self._text_cache:
            self._text_cache[text] = super().embed_query(text)
        return self._text_cache[text]


def test_gegenprobe_ein_cache_macht_beide_tests_rot() -> None:
    boeser = _EmbedderMitCache(FakeBackend(), expected_dimension=FAKE_DIMENSION)

    boeser.embed_query("aufwaermen")
    vorher = _zustandsabdruck(boeser)
    for i in range(25):
        boeser.embed_query(f"eine voellig andere Frage Nummer {i}")
    nachher = _zustandsabdruck(boeser)

    assert nachher != vorher, (
        "Die Wachstumspruefung haette hier anschlagen muessen. Sie ist blind und schuetzt nichts."
    )

    boeser.embed_query(MARKER)
    assert MARKER in _erreichbarer_text(boeser), (
        "Die Textpruefung haette hier anschlagen muessen. Sie ist blind und schuetzt nichts."
    )


# --- Die Fabrik bleibt daneben bestehen -------------------------------------


def test_build_embeddings_liefert_weiterhin_frische_objekte(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`build_embeddings` ist die reine Fabrik und wird NICHT geteilt.

    Ohne diese Trennung waere das Teilen nicht testbar, ohne 0,47 GB zu laden.
    """
    monkeypatch.setattr(
        "app.embeddings.HuggingFaceEmbeddings",
        lambda **_: FakeBackend(),
    )
    monkeypatch.setattr("app.embeddings._read_max_seq_length", lambda _: 512)

    assert build_embeddings(settings) is not build_embeddings(settings)
