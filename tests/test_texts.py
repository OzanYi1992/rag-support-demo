"""Tests der Textkataloge.

Der Fehler, gegen den diese Datei gebaut ist, ist keiner, der abstuerzt: Eine
vergessene Uebersetzung ist ein deutscher Satz in einer englischen Oberflaeche.
Er faellt nicht im Test auf, nicht im Log und nicht beim Start - sondern vor
einem Interessenten.
"""

from __future__ import annotations

import re

import pytest

from app.prompts import _REGELWERK, GroundedAnswer, antwortmodell_fuer
from app.texts import (
    _DEUTSCHE_ZEICHEN,
    _MARKER,
    JAVASCRIPT_SCHLUESSEL,
    SCHEMABESCHREIBUNGEN,
    TEXTE,
    VERFUEGBARE_SPRACHEN,
    Schemabeschreibungen,
    Texte,
    passt_zur_sprache,
    schemabeschreibungen_fuer,
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


# =============================================================================
# DIE FELDBESCHREIBUNGEN DER ANTWORTSTRUKTUR
#
# Diese Texte erreichen das Modell ueber `response_format`. Kein Mensch sieht
# sie, keine Oberflaeche zeigt sie, und bis zum 2026-09-24 hat sie kein Test
# angesehen. Sie waren fest deutsch, und ein englischer Mandant antwortete
# messbar in 23 von 30 Faellen auf Deutsch.
# =============================================================================


def test_schemabeschreibungen_und_textkatalog_decken_dieselben_sprachen_ab() -> None:
    """Sonst laeuft eine Sprache erst beim ersten Modellaufruf in einen KeyError.

    Dieselbe Begruendung wie beim Regelwerk, und derselbe Fehlerzeitpunkt: nicht
    beim Start, sondern bei der ersten Frage eines Interessenten.
    """
    assert set(SCHEMABESCHREIBUNGEN) == set(TEXTE)


def test_kein_schemafeld_ist_leer() -> None:
    """Eine leere Beschreibung ist ein Feld, zu dem das Modell keine Anweisung hat."""
    for sprache, beschreibung in SCHEMABESCHREIBUNGEN.items():
        for name in Schemabeschreibungen.model_fields:
            wert = getattr(beschreibung, name)
            assert wert.strip(), f"{sprache}.{name} ist leer"


def test_kein_schemafeld_ist_in_beiden_sprachen_gleich() -> None:
    """Ein in beiden Katalogen identischer Wert ist eine vergessene Uebersetzung.

    Derselbe Test wie fuer `Texte`, und er gehoert hierher, weil genau diese
    Luecke den Fehler vom 2026-09-24 ausgemacht hat: Der englische Mandant bekam
    den deutschen Wortlaut, weil es keinen zweiten gab.
    """
    for name in Schemabeschreibungen.model_fields:
        de = getattr(SCHEMABESCHREIBUNGEN["de"], name)
        en = getattr(SCHEMABESCHREIBUNGEN["en"], name)
        assert de != en, f"{name} lautet in de und en gleich - vermutlich nicht uebersetzt"


def test_schemafelder_heissen_wie_die_felder_der_antwortstruktur() -> None:
    """Ein umbenanntes Feld verliert sonst still seine Beschreibung.

    `create_model` wuerde ein Feld, das es in der Struktur nicht gibt, einfach
    zusaetzlich anlegen - und das eigentliche Feld bliebe unbeschrieben. Das
    faellt ohne diesen Test niemandem auf, weil beides ein gueltiges Schema
    ergibt.
    """
    erwartet = set(GroundedAnswer.model_fields) | {"struktur"}
    assert set(Schemabeschreibungen.model_fields) == erwartet


def test_deutsches_schema_ist_wortgleich_mit_dem_alten_stand() -> None:
    """Fuer die deutschen Mandanten darf sich NICHTS geaendert haben.

    Die Sollwerte stehen hier als Literale und kommen ausdruecklich NICHT aus
    dem Katalog. Ein Test, der den Katalog gegen sich selbst prueft, bestaetigt
    jede Aenderung, auch die versehentliche. Diese Zeichenketten sind der
    Wortlaut, der bis zum 2026-09-24 in app/prompts.py stand.
    """
    de = SCHEMABESCHREIBUNGEN["de"]
    assert de.struktur == (
        "Was das Modell zurueckgeben muss.\n"
        "\n"
        "`answerable` ist das zweite Eskalationstor. Es ist bewusst ein eigenes Feld\n"
        'und keine Formulierung im Antworttext: Ein "das steht leider nicht in den\n'
        'Unterlagen" mitten in einem ansonsten erfundenen Absatz waere nicht\n'
        "auswertbar."
    )
    assert de.answerable == (
        "true, wenn die Frage aus dem gelieferten Kontext vollstaendig beantwortet "
        "werden kann. false, wenn der Kontext die Frage nicht oder nur teilweise abdeckt."
    )
    assert de.answer == (
        "Die Antwort, ausschliesslich aus dem Kontext. Leer lassen, wenn answerable false ist."
    )
    assert de.sources == (
        "Dateinamen aus dem Kontext, auf denen die Antwort beruht. Nur Dateinamen, "
        "die im Kontext vorkommen."
    )
    assert de.language == "Sprache der Antwort als ISO-639-1-Kuerzel, etwa 'de' oder 'en'."


def test_schemabeschreibungen_fuer_unbekannte_sprache_scheitert_laut() -> None:
    """Kein stiller Rueckfall auf Deutsch - hier waere er genau der Fehler."""
    with pytest.raises(KeyError) as fehler:
        schemabeschreibungen_fuer("fr")
    assert "fr" in str(fehler.value)
    assert "de" in str(fehler.value) and "en" in str(fehler.value)


# =============================================================================
# DAS SCHEMA, WIE DAS MODELL ES SIEHT
#
# Diese Pruefungen gehen ueber `model_json_schema()` und brauchen deshalb keinen
# Provider, keinen Schluessel und kein Netz. Genau das ist der Punkt: Die Stelle
# war bis zum 2026-09-24 nicht schwach geprueft, sondern fuer die Suite
# unsichtbar - `FakeLlm` speist ein FERTIGES GroundedAnswer-Objekt ein, also
# entsteht in keinem Test je ein Schema. Die Absicherung kostet nichts, sie
# fehlte nur.
# =============================================================================


def _schematexte(sprache: str) -> dict[str, str]:
    """Alle Zeichenketten, die aus dem Schema an das Modell gehen."""
    schema = antwortmodell_fuer(sprache).model_json_schema()
    texte = {"struktur": schema["description"]}
    texte.update({name: feld["description"] for name, feld in schema["properties"].items()})
    return texte


def _marker_treffer(text: str, sprache: str) -> list[str]:
    """Funktionswoerter einer Sprache im Text, mit Wortgrenze.

    Die Wortgrenze ist nicht Kosmetik: Ohne sie steckt "der" in "order" und
    "und" in "refund", und jeder englische Satz gilt als deutsch. Die erste
    Fassung dieser Pruefung am 2026-09-24 hat genau so zwei Fehlalarme erzeugt.
    """
    return sorted(
        wort
        for wort in _MARKER[sprache]
        if re.search(r"\b" + re.escape(wort) + r"\b", text, re.IGNORECASE)
    )


def test_jedes_schema_traegt_nur_marker_seiner_eigenen_sprache() -> None:
    """Der Test, den es nicht gab.

    Er ist bewusst ueber alle Sprachen geschrieben und nicht nur gegen Englisch:
    Eine dritte Sprache bekommt die Pruefung damit geschenkt, statt dass jemand
    daran denken muss.

    Der Fehler, den er faengt: Ein englischer Mandant bekam deutsch beschriebene
    Felder, und das Modell antwortete daraufhin gemessen in 23 von 30 Faellen auf
    Deutsch - bei englischer Frage und englischem Korpus.
    """
    for sprache in VERFUEGBARE_SPRACHEN:
        for name, text in _schematexte(sprache).items():
            for fremde in VERFUEGBARE_SPRACHEN:
                if fremde == sprache:
                    continue
                treffer = _marker_treffer(text, fremde)
                assert not treffer, (
                    f"Schema {sprache!r}, Feld {name!r} traegt Woerter der Sprache "
                    f"{fremde!r}: {treffer}"
                )


def test_kein_schema_ausser_dem_deutschen_traegt_deutsche_zeichen() -> None:
    """Umlaute und Eszett sind das harte Signal, unabhaengig von Wortlisten."""
    for sprache in VERFUEGBARE_SPRACHEN:
        if sprache == "de":
            continue
        for name, text in _schematexte(sprache).items():
            getroffen = sorted(set(text) & _DEUTSCHE_ZEICHEN)
            assert not getroffen, f"Schema {sprache!r}, Feld {name!r}: {getroffen}"


def test_die_markerpruefung_am_schema_ist_nicht_blind() -> None:
    """Gegenprobe: Findet dieselbe Pruefung das Deutsche, wenn es da ist?

    Ohne sie waere eine Pruefung, die nie etwas findet, von einer, die nichts
    zu finden hat, nicht zu unterscheiden. Das deutsche Schema MUSS deutsche
    Marker tragen - es ist deutsch.
    """
    treffer = {name: _marker_treffer(text, "de") for name, text in _schematexte("de").items()}
    assert all(treffer.values()), f"kein deutscher Marker gefunden: {treffer}"
    assert sum(len(t) for t in treffer.values()) >= 10


def test_die_basisstruktur_traegt_ueberhaupt_keinen_text() -> None:
    """GroundedAnswer selbst darf nichts an das Modell beitragen.

    Sie hat bewusst keinen Docstring - Pydantic wuerde ihn als `description` in
    das Schema uebernehmen. Sollte jemand die Basisklasse versehentlich direkt
    an `with_structured_output` geben, bekommt das Modell dann ein Schema ohne
    Beschreibungen. Das ist schlechter als richtig, aber es ist nicht FALSCH -
    und vor allem nicht in der falschen Sprache.
    """
    schema = GroundedAnswer.model_json_schema()
    assert "description" not in schema
    for name, feld in schema["properties"].items():
        assert "description" not in feld, f"{name} traegt eine Beschreibung"
