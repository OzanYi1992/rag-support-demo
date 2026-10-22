"""Die Spracherkennung fuer Fragen und Antworten, und der geteilte Erkenner.

Zwei Dinge haelt diese Datei fest:

1. Was `sprache_bestimmen()` erkennt und was nicht. Vor allem die nordischen
   Fragen: Ein Erkenner, der nur zwischen den Katalogsprachen waehlen darf,
   ordnet sie sicher einer davon zu. Jeder dieser Faelle traegt deshalb die
   Gegenprobe mit einem solchen Erkenner, sonst waere nicht zu unterscheiden, ob
   die erste Stufe wirkt oder ob die Fragen ohnehin unter der Schwelle laegen.
2. Dass der Erkenner geteilt werden darf: einmal je Prozess, mandantenunabhaengig
   und zustandslos. Die Zustandslosigkeit ist eine Zusicherung, und der Test
   dazu traegt seine Gegenprobe.

Alles ohne Netz: Das Modell liegt im Paket.
"""

from __future__ import annotations

import hashlib
import threading
from array import array
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from py3langid.langid import MODEL_FILE, LanguageIdentifier

from app.config import Settings
from app.embeddings import E5Embeddings
from app.main import create_app
from app.sprache import (
    SCHWELLE,
    _erkenner_leeren,
    erkenner,
    katalogverteilung,
    sprache_bestimmen,
)
from app.texts import VERFUEGBARE_SPRACHEN
from tests.conftest import FAKE_DIMENSION, FakeBackend

ENGLISCH = ["Do you ship to Norway?", "How long is the return window?"]
DEUTSCH = ["Wie lange dauert Lieferung Norwegen?", "Wann bekomme ich mein Geld zurueck?"]
NICHT_ERKENNBAR = ["Sure?", "ok", "is usa in zone a"]

# Je zwei Fragen schwedisch, norwegisch, daenisch und finnisch, die finnischen
# ohne Umlaute geschrieben.
NORDISCH = [
    "Vad kostar frakten?",
    "Kan jag returnera mina skor?",
    "Leverer dere til Norge?",
    "Hvor lang tid tar leveringen?",
    "Hvad koster fragten?",
    "Hvor lang er leveringstiden?",
    "Onko teilla kenkia koossa 42?",
    "Voinko palauttaa tuotteen?",
]

# Eine gemischte Antwort aus einer Messung, woertlich: halb deutsch, halb
# englisch. Sie soll nicht als eine der beiden Sprachen durchgehen.
GEMISCHT = (
    "Ja. Lieferung nach Norwegen ist Zone B, und für Zone B beträgt die "
    "Standardlieferzeit 5 working days. Norway is listed in Zone B in „Where We "
    "Ship“, and the Zone B standard delivery time is 5 working days in „Where We "
    "Ship“."
)

ALLE = ENGLISCH + DEUTSCH + NICHT_ERKENNBAR + NORDISCH + [GEMISCHT]


def _fingerabdruck(objekt: LanguageIdentifier) -> str:
    """sha256 ueber alle Attribute des Erkenners, Tabellen eingeschlossen."""
    h = hashlib.sha256()

    def hinein(wert: object) -> None:
        if isinstance(wert, np.ndarray):
            h.update(repr((wert.shape, wert.dtype.str)).encode())
            h.update(np.ascontiguousarray(wert).tobytes())
        elif isinstance(wert, array):
            h.update(wert.typecode.encode())
            h.update(wert.tobytes())
        elif isinstance(wert, (list, tuple)):
            h.update(f"<{type(wert).__name__} {len(wert)}>".encode())
            for teil in wert:
                hinein(teil)
        else:
            h.update(repr(wert).encode())

    for name in LanguageIdentifier.__slots__:
        h.update(name.encode())
        hinein(getattr(objekt, name))
    return h.hexdigest()


@pytest.fixture(scope="module")
def eingeschraenkt() -> tuple[LanguageIdentifier, str, str]:
    """Ein EIGENES Objekt, auf die Katalogsprachen eingeschraenkt, nie das der
    Anwendung. Dazu sein Fingerabdruck vor und nach `set_languages()`."""
    objekt = LanguageIdentifier.from_model_file(MODEL_FILE, norm_probs=True)
    vorher = _fingerabdruck(objekt)
    objekt.set_languages(list(VERFUEGBARE_SPRACHEN))
    return objekt, vorher, _fingerabdruck(objekt)


# --- Was erkannt wird ---------------------------------------------------------


@pytest.mark.parametrize("frage", ENGLISCH)
def test_englische_fragen(frage: str) -> None:
    assert sprache_bestimmen(frage) == "en"


@pytest.mark.parametrize("frage", DEUTSCH)
def test_deutsche_fragen(frage: str) -> None:
    assert sprache_bestimmen(frage) == "de"


@pytest.mark.parametrize("frage", NICHT_ERKENNBAR)
def test_kurze_eingaben_ohne_merkmal_sind_nicht_erkennbar(frage: str) -> None:
    assert sprache_bestimmen(frage) is None


@pytest.mark.parametrize("frage", NORDISCH)
def test_nordische_fragen_sind_nicht_erkennbar(
    frage: str, eingeschraenkt: tuple[LanguageIdentifier, str, str]
) -> None:
    """Vorne steht eine nordische Sprache, also nicht erkennbar.

    Gegenprobe: Nur auf die Katalogsprachen eingeschraenkt ordnet der Erkenner
    dieselbe Frage mit mindestens der Schwelle einer davon zu. Ohne die erste
    Stufe galte sie also als Deutsch oder Englisch.
    """
    objekt, _, _ = eingeschraenkt
    vorne, _ = katalogverteilung(frage)

    assert sprache_bestimmen(frage) is None
    assert vorne not in VERFUEGBARE_SPRACHEN
    sprache, konfidenz = objekt.classify(frage)
    assert sprache in VERFUEGBARE_SPRACHEN
    assert konfidenz >= SCHWELLE, (frage, sprache, konfidenz)


def test_eine_gemischte_antwort_ist_nicht_erkennbar() -> None:
    vorne, anteile = katalogverteilung(GEMISCHT)
    assert vorne in VERFUEGBARE_SPRACHEN
    assert anteile[vorne] < SCHWELLE
    assert sprache_bestimmen(GEMISCHT) is None


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_ein_leerer_text_ist_nicht_erkennbar(text: str) -> None:
    assert sprache_bestimmen(text) is None


@pytest.mark.parametrize("text", ALLE)
def test_der_anteil_gleicht_dem_eingeschraenkten_erkenner(
    text: str, eingeschraenkt: tuple[LanguageIdentifier, str, str]
) -> None:
    """Der Anteil an den Katalogsprachen aus der Rangfolge ueber ALLE Sprachen
    ist die Konfidenz eines eingeschraenkten Erkenners. Ein Objekt reicht
    deshalb fuer beide Stufen.

    Gleich bis auf die Rundung: Die Tabellen sind float32, und die
    eingeschraenkte Matrixmultiplikation summiert ueber andere Spalten in anderer
    Reihenfolge. Bei der gemischten Antwort, 234 Byte, weichen die Rohwerte um
    rund 2e-4 bei Betraegen um 935 ab, einige Einheiten der letzten Stelle, und
    der Anteil um rund 3e-6. Kurze Fragen weichen gar nicht ab.
    """
    objekt, _, _ = eingeschraenkt
    _, anteile = katalogverteilung(text)
    sprache, konfidenz = objekt.classify(text)

    assert anteile[sprache] == pytest.approx(konfidenz, abs=1e-5)
    assert anteile[sprache] == pytest.approx(max(anteile.values()), abs=1e-5)
    assert sum(anteile.values()) == pytest.approx(1.0, abs=1e-9)


# --- Der geteilte Erkenner ----------------------------------------------------


def _zaehlend(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Zaehlt, wie oft das Modell wirklich geladen wird."""
    zaehler = [0]
    original = LanguageIdentifier.from_model_file

    def laden(*args: object, **kwargs: object) -> LanguageIdentifier:
        zaehler[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(LanguageIdentifier, "from_model_file", staticmethod(laden))
    _erkenner_leeren()
    return zaehler


def test_der_erkenner_entsteht_einmal(monkeypatch: pytest.MonkeyPatch) -> None:
    zaehler = _zaehlend(monkeypatch)

    erster = erkenner()
    sprache_bestimmen("Do you ship to Norway?")
    zweiter = erkenner()

    assert erster is zweiter
    assert zaehler[0] == 1


def test_gleichzeitige_erste_aufrufe_laden_einmal(monkeypatch: pytest.MonkeyPatch) -> None:
    zaehler = _zaehlend(monkeypatch)
    gefunden: list[LanguageIdentifier] = []

    def holen() -> None:
        gefunden.append(erkenner())

    faeden = [threading.Thread(target=holen) for _ in range(8)]
    for faden in faeden:
        faden.start()
    for faden in faeden:
        faden.join()

    assert zaehler[0] == 1
    assert all(objekt is gefunden[0] for objekt in gefunden)


def test_der_start_waermt_den_erkenner_vor_und_laedt_keinen_zweiten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zaehler = _zaehlend(monkeypatch)
    settings = Settings(
        llm_provider="openai",
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=tmp_path / "tenants",
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
    )
    embeddings = E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION)
    app = create_app(settings, embeddings=embeddings)

    assert zaehler[0] == 0, "erst der Start laedt, nicht create_app"
    with TestClient(app):
        assert zaehler[0] == 1
        erkenner()
    assert zaehler[0] == 1


def test_der_erkenner_bleibt_beim_erkennen_unveraendert(
    eingeschraenkt: tuple[LanguageIdentifier, str, str],
) -> None:
    """Zustandslos: Erkennen aendert am geteilten Objekt nichts.

    Gegenprobe: `set_languages()` auf einem eigenen Objekt aendert den
    Fingerabdruck. Ohne sie waere nicht zu unterscheiden, ob der Abdruck die
    Tabellen ueberhaupt erfasst.
    """
    geteilt = erkenner()
    vorher = _fingerabdruck(geteilt)
    for _ in range(3):
        for text in ALLE:
            sprache_bestimmen(text)
            geteilt.classify(text)

    assert _fingerabdruck(geteilt) == vorher

    _, eigen_vorher, eigen_nachher = eingeschraenkt
    assert eigen_vorher == vorher, "ein frisches Objekt gleicht dem geteilten"
    assert eigen_nachher != eigen_vorher
