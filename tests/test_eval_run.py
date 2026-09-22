"""Tests des Messwerkzeugs.

Zwei Zusicherungen werden hier durchgesetzt, nicht behauptet (P-009, P-010):

1. Keine Ergebnisdatei ohne Metrikangabe. Eine Datei ohne sie ist spaeter nicht
   einzuordnen, und daraus entsteht der Vergleich von Zahlen, die nicht
   vergleichbar sind - Dateitreffer gegen Chunktreffer.
2. Die Abweichungsklasse wird AUS DEN DATEN abgeleitet. Wo die Grundlage fehlt,
   wird das ausgewiesen statt geraten.
"""

from __future__ import annotations

import json

import httpx
import pytest

from eval.run import (
    ABW_IM_KONTEXT,
    ABW_KEINE,
    ABW_NICHT_BEWERTBAR,
    ABW_NICHT_IM_KONTEXT,
    ABW_UNBESTIMMT,
    METRIK_DATEI_UND_CHUNK,
    QUELLE_NICHT_ANWENDBAR,
    SCORE_TOLERANZ,
    CloudZiel,
    Frageergebnis,
    _abweichungsklasse,
    cloud_antwort,
    main,
    pruefe_goldsatz,
    pruefe_kopf,
    vergleiche_scores,
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


# --- erwartete_quelle darf nicht blank null sein ----------------------------


def test_goldsatz_mit_platzhalter_geht_durch() -> None:
    gold = {"fragen": [{"id": "a-1", "erwartete_quelle": QUELLE_NICHT_ANWENDBAR}]}
    pruefe_goldsatz(gold, "test.yaml")


def test_goldsatz_mit_dateiname_geht_durch() -> None:
    pruefe_goldsatz({"fragen": [{"id": "a-1", "erwartete_quelle": "doku.md"}]}, "t.yaml")


def test_goldsatz_mit_null_wird_abgelehnt() -> None:
    """Der Kern von P-019: Dasselbe Zeichen trug zwei Bedeutungen.

    null hiess mal "es gibt hier keine richtige Quelle" und mal "eine gaebe es,
    sie wurde nur nie bestimmt". Nichts am Zeichen unterschied die beiden, und
    ein Leser sah keinen Anlass zur Rueckfrage.
    """
    with pytest.raises(ValueError, match="nicht_anwendbar"):
        pruefe_goldsatz({"fragen": [{"id": "a-1", "erwartete_quelle": None}]}, "t.yaml")


def test_goldsatz_ohne_das_feld_wird_abgelehnt() -> None:
    with pytest.raises(ValueError, match="a-1"):
        pruefe_goldsatz({"fragen": [{"id": "a-1"}]}, "t.yaml")


def test_die_echten_goldsaetze_sind_gueltig() -> None:
    """Positivtest: Ohne ihn waere die Pruefung eine Behauptung (P-010)."""
    import pathlib

    import yaml

    from eval.run import EVAL_DIR

    geprueft = 0
    for pfad in sorted(pathlib.Path(EVAL_DIR).glob("*/gold.yaml")):
        pruefe_goldsatz(yaml.safe_load(pfad.read_text(encoding="utf-8")), str(pfad))
        geprueft += 1
    assert geprueft == 2, "Es sollten zwei Goldsaetze geprueft worden sein."


# --- Cloud-Lauf ---------------------------------------------------------------
#
# Drei Zusicherungen: Der Score-Vergleich ist beziffert und nicht nur behauptet,
# der Generierungsschritt trifft genau den Chat-Pfad der Instanz, und das
# url_token taucht in keiner Ausgabe auf - auch nicht in einer Fehlermeldung.

TEST_TOKEN = "nur-ein-testwert-fuer-den-pfad-0001"


def _cloud(handler: object) -> CloudZiel:
    client = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return CloudZiel("https://instanz.invalid", "sha256:abc", client)


def test_scores_innerhalb_der_toleranz_gelten_als_gleich() -> None:
    gleich, abweichung = vergleiche_scores([0.91, 0.87], [0.9101, 0.8699])
    assert gleich is True
    assert abweichung is not None and abweichung <= SCORE_TOLERANZ


def test_scores_ausserhalb_der_toleranz_gelten_als_verschieden() -> None:
    """Die Gegenprobe: Ein anderer Index darf nicht als gleich durchgehen."""
    gleich, abweichung = vergleiche_scores([0.91, 0.87], [0.91, 0.80])
    assert gleich is False
    assert abweichung == pytest.approx(0.07)


def test_verschiedene_laenge_gilt_als_verschieden() -> None:
    assert vergleiche_scores([0.9, 0.8], [0.9]) == (False, None)


def test_cloud_antwort_trifft_den_chatpfad() -> None:
    gesehen: dict[str, object] = {}

    def handler(anfrage: httpx.Request) -> httpx.Response:
        gesehen["pfad"] = anfrage.url.path
        gesehen["rumpf"] = json.loads(anfrage.content)
        return httpx.Response(200, json={"escalated": False, "retrieval_scores": [0.9]})

    daten = cloud_antwort(_cloud(handler), TEST_TOKEN, "Wie lange gilt die RMA?")

    assert gesehen["pfad"] == f"/t/{TEST_TOKEN}/chat"
    assert gesehen["rumpf"] == {"question": "Wie lange gilt die RMA?"}
    assert TEST_TOKEN not in json.dumps(daten)


def test_cloud_fehler_nennt_das_token_nicht() -> None:
    """raise_for_status() wuerde die URL samt Token in die Meldung schreiben."""

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"detail": "Zu viele Anfragen."})

    with pytest.raises(RuntimeError) as fehler:
        cloud_antwort(_cloud(handler), TEST_TOKEN, "egal")

    assert "429" in str(fehler.value)
    assert TEST_TOKEN not in str(fehler.value)


def test_cloud_lauf_ohne_image_digest_wird_abgelehnt() -> None:
    """Ein Cloud-Ergebnis ohne Image-Bezug waere spaeter nicht einzuordnen."""
    with pytest.raises(SystemExit):
        main(["demo-acme", "--base-url", "https://instanz.invalid"])


def test_cloud_lauf_mit_nur_retrieval_wird_abgelehnt() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "demo-acme",
                "--base-url",
                "https://instanz.invalid",
                "--image-digest",
                "sha256:abc",
                "--retrieval-only",
            ]
        )
