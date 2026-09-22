"""Tests der Textkataloge.

Der Fehler, gegen den diese Datei gebaut ist, ist keiner, der abstuerzt: Eine
vergessene Uebersetzung ist ein deutscher Satz in einer englischen Oberflaeche.
Er faellt nicht im Test auf, nicht im Log und nicht beim Start - sondern vor
einem Interessenten.
"""

from __future__ import annotations

import pytest

from app.prompts import _REGELWERK
from app.texts import (
    JAVASCRIPT_SCHLUESSEL,
    TEXTE,
    VERFUEGBARE_SPRACHEN,
    Texte,
    passt_zur_sprache,
    texte_fuer,
)


def test_beide_sprachen_vorhanden() -> None:
    assert set(TEXTE) == {"de", "en"}
    assert VERFUEGBARE_SPRACHEN == ("de", "en")


def test_kein_feld_ist_leer() -> None:
    """Ein leeres Feld ist eine unsichtbare Luecke in der Oberflaeche."""
    for sprache, texte in TEXTE.items():
        for name in Texte.model_fields:
            wert = getattr(texte, name)
            assert wert.strip(), f"{sprache}.{name} ist leer"


def test_kein_feld_ist_in_beiden_sprachen_gleich() -> None:
    """Ein in beiden Katalogen identischer Wert ist eine vergessene Uebersetzung.

    Das ist der Test, der die Luecke faengt, die sonst niemand sieht. Sollte
    einmal ein Feld absichtlich in beiden Sprachen gleich lauten - ein
    Eigenname etwa -, gehoert es hier namentlich ausgenommen und nicht der Test
    abgeschwaecht. Die Ausnahme ist dann dokumentiert, die Regel bleibt scharf.
    """
    for name in Texte.model_fields:
        de = getattr(TEXTE["de"], name)
        en = getattr(TEXTE["en"], name)
        assert de != en, f"{name} lautet in de und en gleich - vermutlich nicht uebersetzt"


def test_javascript_schluessel_existieren_alle() -> None:
    """Ein Tippfehler in der Liste waere sonst erst im Browser sichtbar."""
    for name in JAVASCRIPT_SCHLUESSEL:
        assert name in Texte.model_fields, f"{name} ist kein Feld von Texte"


def test_fuer_javascript_liefert_nur_die_browsertexte() -> None:
    """Was der Server allein braucht, hat in der Seite nichts zu suchen."""
    for texte in TEXTE.values():
        js = texte.fuer_javascript()
        assert set(js) == set(JAVASCRIPT_SCHLUESSEL)
        assert "ratenlimit_detail" not in js
        assert "oberflaeche_fehlt" not in js


def test_platzhalter_bleiben_in_beiden_sprachen_erhalten() -> None:
    """Ein in der Uebersetzung verlorener Platzhalter bricht den Satz still."""
    for sprache, texte in TEXTE.items():
        assert "{display_name}" in texte.begruessung, sprache
        assert "{sekunden}" in texte.ratenlimit_mit_zeit, sprache
        # Der Satz ohne Zeitangabe darf KEINEN Platzhalter tragen - er wird
        # genau dann benutzt, wenn es keine Zahl einzusetzen gibt.
        assert "{" not in texte.ratenlimit_ohne_zeit, sprache


def test_regelwerk_und_textkatalog_decken_dieselben_sprachen_ab() -> None:
    """Sonst faellt eine Sprache erst beim ersten Modellaufruf auf.

    Die Oberflaeche wuerde laden, und der System-Prompt liefe in einen
    KeyError - nicht beim Start, sondern bei der ersten Frage eines
    Interessenten.
    """
    assert set(_REGELWERK) == set(TEXTE)


def test_texte_fuer_unbekannte_sprache_scheitert_laut() -> None:
    """Kein stiller Rueckfall auf Deutsch.

    Ein Rueckfall liesse einen Tippfehler in der tenant.yaml als deutsche
    Oberflaeche bei einem englischen Mandanten enden - ohne Fehlermeldung.
    """
    with pytest.raises(KeyError) as fehler:
        texte_fuer("fr")
    assert "fr" in str(fehler.value)
    # Die Meldung nennt, was es stattdessen gibt.
    assert "de" in str(fehler.value) and "en" in str(fehler.value)


def test_englischer_katalog_traegt_keine_umlaute() -> None:
    """Grobe, aber wirksame Gegenprobe gegen kopierte deutsche Saetze."""
    englisch = TEXTE["en"]
    for name in Texte.model_fields:
        wert = getattr(englisch, name)
        for zeichen in "äöüÄÖÜß":
            assert zeichen not in wert, f"en.{name} enthaelt {zeichen!r}"


# --- Sprachpruefung fuer Texte, die kein Katalog traegt ---------------------

DE_ECHT = (
    "Dazu finde ich in den Unterlagen von ACME nichts Belastbares. "
    "Bitte wenden Sie sich an den Support."
)
EN_ECHT = (
    "I cannot find anything reliable about that in the documents. Please write to our support desk."
)


def test_richtige_sprache_wird_durchgelassen() -> None:
    """Der Positivtest. Ohne ihn koennte die Pruefung alles abweisen."""
    assert passt_zur_sprache(DE_ECHT, "de") is None
    assert passt_zur_sprache(EN_ECHT, "en") is None


def test_falsche_sprache_wird_abgewiesen() -> None:
    """Der Negativtest, und der eigentliche Zweck der Pruefung."""
    grund = passt_zur_sprache(DE_ECHT, "en")
    assert grund is not None and "'de'" in grund

    grund = passt_zur_sprache(EN_ECHT, "de")
    assert grund is not None and "'en'" in grund


def test_deutsche_sonderzeichen_in_einem_englischen_text() -> None:
    """Ein hartes Signal, das auch ohne Marker greift.

    Der Satz traegt absichtlich kaum deutsche Funktionswoerter - er soll allein
    an den Umlauten scheitern, damit die zweite Regel nachweislich wirkt und
    nicht nur die erste.
    """
    grund = passt_zur_sprache(
        "Gruesse aus Muenchen: Groesse XL, Qualitaet geprueft.".replace("ue", "ü")
        .replace("oe", "ö")
        .replace("ae", "ä"),
        "en",
    )
    assert grund is not None
    assert "Englischen nicht gibt" in grund


def test_die_kataloge_selbst_bestehen_ihre_eigene_pruefung() -> None:
    """Gegenprobe gegen die Heuristik: Die gepflegten Texte muessen durchlaufen.

    Weist sie hier etwas ab, ist die Heuristik zu scharf - und nicht der Text
    falsch.
    """
    for sprache, texte in TEXTE.items():
        for name in Texte.model_fields:
            wert = getattr(texte, name)
            if len(wert.split()) < 4:
                # Ein bis drei Woerter tragen keine Funktionswoerter. Dort kann
                # die Heuristik nichts sehen, und das soll sie auch nicht.
                continue
            assert passt_zur_sprache(wert, sprache) is None, f"{sprache}.{name}"


def test_kurzer_text_ohne_marker_laeuft_durch() -> None:
    """Die Heuristik ist konservativ: Was sie nicht sieht, weist sie nicht ab.

    Sonst waere sie an Randfaellen laut und am eigentlichen Fehler still.
    """
    assert passt_zur_sprache("Support: 12345", "de") is None
    assert passt_zur_sprache("Support: 12345", "en") is None
