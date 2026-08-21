"""Tests des Messwerkzeugs.

Zwei Zusicherungen werden hier durchgesetzt, nicht behauptet (P-009, P-010):

1. Keine Ergebnisdatei ohne Metrikangabe. Eine Datei ohne sie ist spaeter nicht
   einzuordnen, und daraus entsteht der Vergleich von Zahlen, die nicht
   vergleichbar sind - Dateitreffer gegen Chunktreffer.
2. Die Abweichungsklasse wird AUS DEN DATEN abgeleitet. Wo die Grundlage fehlt,
   wird das ausgewiesen statt geraten.
"""

from __future__ import annotations

import pytest

from eval.run import (
    ABW_IM_KONTEXT,
    ABW_KEINE,
    ABW_NICHT_BEWERTBAR,
    ABW_NICHT_IM_KONTEXT,
    ABW_UNBESTIMMT,
    METRIK_DATEI_UND_CHUNK,
    Frageergebnis,
    _abweichungsklasse,
    pruefe_kopf,
)


def _kopf(**abweichend: object) -> dict[str, object]:
    basis = {
        "metrik": METRIK_DATEI_UND_CHUNK,
        "metrik_bedeutung": "rang = Datei, rang_chunk = Chunk",
    }
    basis.update(abweichend)
    return {"kopf": basis}


# --- Schreibsperre ----------------------------------------------------------


def test_kopf_mit_gueltiger_metrik_geht_durch() -> None:
    pruefe_kopf(_kopf())


def test_kopf_ohne_metrik_wird_abgelehnt() -> None:
    bericht = _kopf()
    del bericht["kopf"]["metrik"]
    with pytest.raises(ValueError, match="Metrikangabe"):
        pruefe_kopf(bericht)


def test_kopf_mit_unbekannter_metrik_wird_abgelehnt() -> None:
    with pytest.raises(ValueError, match="Metrikangabe"):
        pruefe_kopf(_kopf(metrik="irgendwas"))


def test_kopf_ohne_erklaerung_wird_abgelehnt() -> None:
    """Ein Kuerzel ohne Erklaerung ist in vier Wochen so wenig wert wie nichts."""
    with pytest.raises(ValueError, match="metrik_bedeutung"):
        pruefe_kopf(_kopf(metrik_bedeutung=""))


# --- Abweichungsklasse ------------------------------------------------------


def _frage(**abweichend: object) -> Frageergebnis:
    grund = {
        "id": "test-01",
        "kategorie": "direkt",
        "frage": "egal",
        "erwartete_quelle": "a.md",
        "erwartete_textstelle": "eine Stelle",
        "erwartet_eskalation": False,
    }
    grund.update(abweichend)
    return Frageergebnis(**grund)  # type: ignore[arg-type]


def test_ohne_llm_ist_die_klasse_nicht_bewertbar() -> None:
    assert _abweichungsklasse(_frage()) == ABW_NICHT_BEWERTBAR


def test_erwartungsgemaess_ergibt_keine_abweichung() -> None:
    f = _frage(eskaliert=False, eskalation_wie_erwartet=True, antwort_im_kontext=True)
    assert _abweichungsklasse(f) == ABW_KEINE


def test_abweichung_ohne_kontext_wird_als_retrievalfall_gefuehrt() -> None:
    """Eskaliert, und die Antwort lag nicht im Kontext: kein Torproblem."""
    f = _frage(eskaliert=True, eskalation_wie_erwartet=False, antwort_im_kontext=False)
    assert _abweichungsklasse(f) == ABW_NICHT_IM_KONTEXT


def test_abweichung_mit_kontext_wird_als_torfall_gefuehrt() -> None:
    """Eskaliert, obwohl die Antwort im Kontext lag: das Tor hat entschieden."""
    f = _frage(eskaliert=True, eskalation_wie_erwartet=False, antwort_im_kontext=True)
    assert _abweichungsklasse(f) == ABW_IM_KONTEXT


def test_ohne_textstelle_ist_die_klasse_unbestimmt() -> None:
    """Fehlt die Grundlage, wird das ausgewiesen - nicht geraten.

    Das ist der Fall, der die beiden Klassen ununterscheidbar macht. Ihn als
    'nicht im Kontext' zu fuehren waere eine Behauptung ueber Daten, die nicht
    erhoben wurden.
    """
    f = _frage(
        erwartete_textstelle=None,
        eskaliert=True,
        eskalation_wie_erwartet=False,
        antwort_im_kontext=None,
    )
    assert _abweichungsklasse(f) == ABW_UNBESTIMMT
