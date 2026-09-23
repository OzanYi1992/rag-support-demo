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
    KONTROLLE_RETRIEVAL,
    KONTROLLE_SPRACHABSTAND,
    METRIK_DATEI_UND_CHUNK,
    QUELLE_NICHT_ANWENDBAR,
    SCORE_TOLERANZ,
    CloudZiel,
    Frageergebnis,
    _abweichungsklasse,
    _hat_ziel_verfehlt,
    _kontrolle_fahren,
    aggregiere,
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
    assert geprueft == 3, "Es sollten drei Goldsaetze geprueft worden sein."


def test_jede_erwartete_textstelle_steht_woertlich_im_korpus() -> None:
    """Eine Textstelle mit Tippfehler misst dauerhaft null, ohne Fehlermeldung.

    `rang_chunk` bleibt dann None, `antwort_im_kontext` False - und das sieht
    im Ergebnis aus wie ein Retrievalproblem. Der Goldsatz waere kaputt, die
    Zahl waere falsch, und nichts wuerde darauf hinweisen.

    Beim Anlegen des Fellgate-Goldsatzes hat genau das zugeschlagen: zwei
    Textstellen liefen im Dokument ueber einen Zeilenumbruch, standen im
    Goldsatz aber einzeilig. Dieser Test faengt das.
    """
    import pathlib

    import yaml

    from eval.run import EVAL_DIR, QUELLE_PLATZHALTER

    wurzel = pathlib.Path(EVAL_DIR).parent / "tenants"
    geprueft = 0
    fehlend: list[str] = []

    for pfad in sorted(pathlib.Path(EVAL_DIR).glob("*/gold.yaml")):
        gold = yaml.safe_load(pfad.read_text(encoding="utf-8"))
        docs = wurzel / gold["mandant"] / "docs"
        if not docs.is_dir():
            pytest.skip(f"Kein Korpus unter {docs} - der Test waere blind.")
        for f in gold["fragen"]:
            stelle = f.get("erwartete_textstelle")
            quelle = f.get("erwartete_quelle")
            if not stelle or quelle in QUELLE_PLATZHALTER:
                continue
            datei = docs / quelle
            if not datei.is_file():
                fehlend.append(f"{f['id']}: Datei {quelle} fehlt")
                continue
            geprueft += 1
            if stelle not in datei.read_text(encoding="utf-8"):
                fehlend.append(f"{f['id']}: {stelle!r} steht nicht in {quelle}")

    # Gegenprobe: Haette die Schleife nichts geprueft, waere ein leeres
    # `fehlend` kein Ergebnis, sondern ein Werkzeugfehler.
    assert geprueft > 10, f"Nur {geprueft} Textstellen geprueft - der Test waere fast blind."
    assert not fehlend, "Erwartete Textstellen fehlen im Korpus:\n  " + "\n  ".join(fehlend)


def test_jede_kontrollfrage_zielt_auf_dieselbe_stelle() -> None:
    """Eine Kontrollfrage auf ein anderes Ziel beantwortet eine andere Frage.

    Der Befund waere dann nicht deutbar: Ein Treffer der Kontrolle hiesse nicht
    mehr "die Stelle ist erreichbar", sondern nur "irgendeine Stelle ist
    erreichbar". Geprueft wird deshalb, dass ueberhaupt eine Stelle vorliegt,
    an der sich beide messen lassen.
    """
    import pathlib

    import yaml

    from eval.run import EVAL_DIR

    mit_kontrolle = 0
    for pfad in sorted(pathlib.Path(EVAL_DIR).glob("*/gold.yaml")):
        gold = yaml.safe_load(pfad.read_text(encoding="utf-8"))
        for f in gold["fragen"]:
            if not f.get("kontrollfrage"):
                continue
            mit_kontrolle += 1
            assert f.get("erwartete_textstelle"), (
                f"{f['id']} hat eine Kontrollfrage, aber keine erwartete Textstelle - "
                f"dann gibt es keinen gemeinsamen Massstab."
            )
            assert f["kontrollfrage"] != f["frage"], (
                f"{f['id']}: Kontrollfrage und Frage sind identisch."
            )
    assert mit_kontrolle >= 3, "Zu wenige Kontrollfragen - der Test waere fast blind."


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


# --- Kontrollfrage ----------------------------------------------------------
#
# Wozu sie da ist: Eine kundennah formulierte Frage kann aus zwei Gruenden
# scheitern - das Retrieval gibt die Stelle nicht her, oder die Frage ist
# schlecht formuliert. Ohne Kontrollfall sind die beiden nicht zu trennen, und
# ein roter Eintrag im Goldsatz ist dann nicht deutbar.


class _FakeSuche:
    """Liefert je Frage eine feste Trefferliste. Kein Modell, kein Index."""

    def __init__(self, nach_frage: dict[str, list[tuple[str, str]]]) -> None:
        self._nach_frage = nach_frage
        self.gefragt: list[str] = []

    def __call__(
        self, slug: str, frage: str, k: int, settings: object, embeddings: object
    ) -> list[object]:
        self.gefragt.append(frage)

        class _Treffer:
            def __init__(self, quelle: str, text: str) -> None:
                self.source_file = quelle
                self.text = text
                self.score = 0.9
                self.tenant_slug = slug

        return [_Treffer(q, t) for q, t in self._nach_frage.get(frage, [])]


def _fahre_kontrolle(
    monkeypatch: pytest.MonkeyPatch,
    erg: Frageergebnis,
    nach_frage: dict[str, list[tuple[str, str]]],
    top_k: int = 2,
) -> _FakeSuche:
    suche = _FakeSuche(nach_frage)
    monkeypatch.setattr("eval.run.search_tenant", suche)
    _kontrolle_fahren("demo-test", erg, settings=None, top_k=top_k, gesamtzahl=10, embedder=None)  # type: ignore[arg-type]
    return suche


def test_verfehlt_wenn_datei_nicht_in_top_k() -> None:
    assert _hat_ziel_verfehlt(_frage(in_top_k=False, antwort_im_kontext=True))


def test_verfehlt_wenn_antwort_nicht_im_kontext() -> None:
    """Datei auf Rang 3 nuetzt nichts, wenn der Chunk mit der Zahl fehlt."""
    assert _hat_ziel_verfehlt(_frage(in_top_k=True, antwort_im_kontext=False))


def test_nicht_verfehlt_wenn_beides_stimmt() -> None:
    """Die Gegenprobe. Ohne sie faende der Test oben alles verfehlt."""
    assert not _hat_ziel_verfehlt(_frage(in_top_k=True, antwort_im_kontext=True))


def test_kontrolle_laeuft_gar_nicht_wenn_die_frage_getroffen_hat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sonst verdoppelt sie die Retrievalzeit jedes Laufs, ohne etwas zu sagen."""
    erg = _frage(in_top_k=True, antwort_im_kontext=True, kontrollfrage="Dokumentfassung")
    suche = _fahre_kontrolle(monkeypatch, erg, {})

    assert suche.gefragt == []
    assert erg.kontroll_befund is None


def test_kontrolle_laeuft_nicht_ohne_kontrollfrage(monkeypatch: pytest.MonkeyPatch) -> None:
    erg = _frage(in_top_k=False, antwort_im_kontext=False, kontrollfrage=None)
    suche = _fahre_kontrolle(monkeypatch, erg, {})

    assert suche.gefragt == []
    assert erg.kontroll_befund is None


def test_befund_retrieval_wenn_auch_die_dokumentfassung_nichts_findet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Beide scheitern -> das Retrieval gibt die Stelle nicht her."""
    erg = _frage(in_top_k=False, antwort_im_kontext=False, kontrollfrage="Dokumentfassung")
    suche = _fahre_kontrolle(
        monkeypatch, erg, {"Dokumentfassung": [("b.md", "etwas ganz anderes")]}
    )

    assert suche.gefragt == ["Dokumentfassung"]
    assert erg.kontroll_befund == KONTROLLE_RETRIEVAL
    assert erg.kontroll_in_top_k is False


def test_befund_sprachabstand_wenn_nur_die_dokumentfassung_findet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nur die Dokumentfassung trifft -> Abstand Kundensprache/Dokumentsprache.

    Das ist ein Befund ueber das System, kein Fehler im Goldsatz. Genau diese
    Unterscheidung ist der Zweck der Kontrollfrage.
    """
    erg = _frage(in_top_k=False, antwort_im_kontext=False, kontrollfrage="Dokumentfassung")
    _fahre_kontrolle(monkeypatch, erg, {"Dokumentfassung": [("a.md", "eine Stelle steht hier")]})

    assert erg.kontroll_befund == KONTROLLE_SPRACHABSTAND
    assert erg.kontroll_rang == 1
    assert erg.kontroll_antwort_im_kontext is True


def test_kontrolle_misst_mit_demselben_massstab(monkeypatch: pytest.MonkeyPatch) -> None:
    """Richtige Datei, aber die Textstelle fehlt -> weiterhin Retrievalbefund.

    Waere der Massstab hier lockerer als bei der eigentlichen Frage, verglichen
    Befund und Fehlschlag zwei verschiedene Dinge.
    """
    erg = _frage(in_top_k=False, antwort_im_kontext=False, kontrollfrage="Dokumentfassung")
    _fahre_kontrolle(monkeypatch, erg, {"Dokumentfassung": [("a.md", "steht nicht drin")]})

    assert erg.kontroll_rang == 1
    assert erg.kontroll_antwort_im_kontext is False
    assert erg.kontroll_befund == KONTROLLE_RETRIEVAL


def test_aggregation_weist_die_beiden_befunde_getrennt_aus() -> None:
    """Zusammengezaehlt verdeckten sie genau den Unterschied, der zaehlt."""
    ergebnisse = [
        _frage(id="a", kontrollfrage="x", kontroll_befund=KONTROLLE_RETRIEVAL),
        _frage(id="b", kontrollfrage="x", kontroll_befund=KONTROLLE_SPRACHABSTAND),
        _frage(id="c", kontrollfrage="x", kontroll_befund=KONTROLLE_SPRACHABSTAND),
        _frage(id="d", kontrollfrage="x", in_top_k=True, antwort_im_kontext=True),
    ]
    a = aggregiere(ergebnisse, top_k=4, preise={}, modell=None)["direkt"]

    assert a["kontrolle_retrieval"] == 1
    assert a["kontrolle_sprachabstand"] == 2
    assert a["kontrolle_nicht_gefahren"] == 1


def test_aggregation_meldet_eine_fehlende_kontrollfrage() -> None:
    """Ein verfehlter Eintrag ohne Kontrollfall ist nicht deutbar - das muss auffallen."""
    ergebnisse = [_frage(id="ohne-kontrolle", in_top_k=False, antwort_im_kontext=False)]
    a = aggregiere(ergebnisse, top_k=4, preise={}, modell=None)["direkt"]

    assert a["kontrollfrage_fehlt"] == ["ohne-kontrolle"]


def test_aggregation_meldet_nichts_wenn_die_frage_getroffen_hat() -> None:
    """Gegenprobe zum Test darueber."""
    ergebnisse = [_frage(id="getroffen", in_top_k=True, antwort_im_kontext=True)]
    a = aggregiere(ergebnisse, top_k=4, preise={}, modell=None)["direkt"]

    assert a["kontrollfrage_fehlt"] == []


def test_cloud_lauf_ohne_revision_wird_abgelehnt() -> None:
    """Ein Cloud-Ergebnis ohne die antwortende Revision ist nicht einzuordnen.

    Der Digest sagt, was deployt wurde. In der Luecke eines Rollouts antwortet
    der Vorgaenger, und das Ergebnis sieht genauso aus - am 2026-09-23 sind so
    drei Messwerte am falschen Gegenstand entstanden.
    """
    with pytest.raises(SystemExit):
        main(["demo-acme", "--base-url", "https://x.invalid", "--image-digest", "sha256:abc"])


def test_cloud_ziel_traegt_die_revision() -> None:
    """Gegenprobe zum Test darueber: mit Revision laesst sich das Ziel bauen."""
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    ziel = CloudZiel("https://x.invalid", "sha256:abc", client, "ca-ragdemo--0000002")
    assert ziel.revision == "ca-ragdemo--0000002"
    client.close()


def test_cloud_ziel_ohne_revision_ist_moeglich_aber_leer() -> None:
    """Das Feld hat einen Default, damit bestehende Aufrufe nicht brechen.

    Erzwungen wird die Angabe auf der CLI, nicht im Datentyp - sonst waere jeder
    Test, der ein CloudZiel baut, an einen Wert gebunden, der ihn nicht betrifft.
    """
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    assert CloudZiel("https://x.invalid", "sha256:abc", client).revision is None
    client.close()
