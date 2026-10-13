"""Tests der Textkataloge.

Der Fehler, gegen den diese Datei gebaut ist, ist keiner, der abstuerzt: Eine
vergessene Uebersetzung ist ein deutscher Satz in einer englischen Oberflaeche.
Er faellt nicht im Test auf, nicht im Log und nicht beim Start - sondern vor
einem Interessenten.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.prompts import (
    _REGELWERK,
    _UMFORMUNG,
    GroundedAnswer,
    antwortmodell_fuer,
    umformungsmodell_fuer,
)
from app.texts import (
    _DEUTSCHE_ZEICHEN,
    _MARKER,
    JAVASCRIPT_SCHLUESSEL,
    KURZFORMELN,
    SCHEMABESCHREIBUNGEN,
    TEXTE,
    UMFORMUNGSBESCHREIBUNGEN,
    VERFUEGBARE_SPRACHEN,
    Schemabeschreibungen,
    Texte,
    kurzformel_erkennen,
    levenshtein_abstand,
    normalisieren,
    passt_zur_sprache,
    schemabeschreibungen_fuer,
    texte_fuer,
    tippfehler_toleranz,
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
        # Seit dem 2026-09-24: Die Begruessung nennt, worueber Auskunft kommt.
        # Ohne diesen Platzhalter wuesste ein Interessent nicht, was die
        # Wissensbasis abdeckt - er fragt dann etwas Ungedecktes, bekommt eine
        # korrekte Eskalation und haelt das System fuer schwach.
        assert "{topics}" in texte.begruessung, sprache
        # Die Antwort auf einen Gruss nennt dieselben Themen.
        assert "{topics}" in texte.begruessung_antwort, sprache
        assert "{sekunden}" in texte.ratenlimit_mit_zeit, sprache
        # Der Rahmentitel nennt die Datei. Ohne den Platzhalter hiessen alle
        # aufgeklappten Dokumente gleich, und ein Screenreader saehe nicht,
        # welches offen ist.
        assert "{datei}" in texte.dokument_rahmen_titel, sprache
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


def test_anbieter_nicht_erreichbar_steht_im_browserkatalog() -> None:
    """Der Text fuer einen Ausfall des Modellanbieters, in beiden Sprachen.

    Er steht in der Seite, weil app.js ihn bei HTTP 503 zeigt. Und er ist ein
    anderer als anfrage_fehlgeschlagen und als jeder Eskalationstext: Ein
    Ausfall ist weder ein Fehler der Anfrage noch eine Luecke der Unterlagen.
    """
    assert "anbieter_nicht_erreichbar" in JAVASCRIPT_SCHLUESSEL
    assert TEXTE["de"].anbieter_nicht_erreichbar == (
        "Der Assistent ist gerade nicht erreichbar. "
        "Bitte versuchen Sie es in einer Minute noch einmal."
    )
    assert TEXTE["en"].anbieter_nicht_erreichbar == (
        "The assistant cannot be reached right now. Please try again in a minute."
    )
    for sprache, texte in TEXTE.items():
        js = texte.fuer_javascript()
        assert js["anbieter_nicht_erreichbar"] == texte.anbieter_nicht_erreichbar, sprache
        assert texte.anbieter_nicht_erreichbar != texte.anfrage_fehlgeschlagen, sprache


# --- Begruessung, Dank und Abschied ohne Modell -------------------------------


@pytest.mark.parametrize(
    "eingabe",
    [
        "Hi!",
        "  hello ",
        "Guten Tag.",
        "HALLO",
        "Grüß Gott",
        "grüss gott!",
        "Good   Morning",
        "hi\tthere",
    ],
)
def test_begruessung_normalisierung_trifft(eingabe: str) -> None:
    """Satzzeichen, Gross- und Kleinschreibung, Leerraum und ß.

    "Grüß Gott" trifft nur, weil dieselbe Normalisierung ueber die Liste
    laeuft: casefold macht aus ß ein ss, in der Eingabe wie in der Formel.
    """
    assert kurzformel_erkennen(eingabe) == "greeting"


@pytest.mark.parametrize(
    "eingabe",
    [
        "Hi, do you ship to Norway?",
        "Hallo, wo ist meine Bestellung?",
        "hi hi",
        "hello world",
        "",
        "   ",
        "?!",
        "thanks, and do you ship to Norway?",
    ],
)
def test_ein_satz_mit_gruss_ist_keine_kurzformel(eingabe: str) -> None:
    """Nur die ganze Eingabe zaehlt. Ein Gruss am Anfang einer Frage trifft nicht."""
    assert kurzformel_erkennen(eingabe) is None


def test_normalisieren() -> None:
    assert normalisieren("  Grüß   Gott!! ") == "grüss gott"
    assert normalisieren("Hallå.") == "hallå"
    assert normalisieren("thank-you") == "thank you"


@pytest.mark.parametrize("eingabe", ["hej", "Hei!", "moi", "Hallå", "HALLÅ!"])
def test_nordische_gruesse(eingabe: str) -> None:
    """hej, hei, moi und das schwedische hallå stehen in der englischen Liste."""
    assert kurzformel_erkennen(eingabe) == "greeting"
    assert "hallå" in KURZFORMELN["en"].begruessung
    # Gegenprobe: ohne den Ring ist es ein anderes Wort und trifft nicht.
    assert kurzformel_erkennen("halla") is None


def test_der_ring_steht_als_echtes_zeichen_in_der_liste() -> None:
    """U+00E5 als ein Zeichen, nicht als a mit kombinierendem Ring."""
    assert "hallå" in KURZFORMELN["en"].begruessung
    assert "hallå" not in KURZFORMELN["en"].begruessung


def test_kurzformellisten_decken_die_katalogsprachen_ab() -> None:
    """Dieselben Sprachschluessel wie der Textkatalog, keine Liste leer."""
    assert set(KURZFORMELN) == set(TEXTE)
    for sprache, formeln in KURZFORMELN.items():
        for art in ("begruessung", "dank", "abschied", "faehigkeiten", "identitaet"):
            assert getattr(formeln, art), f"{sprache}.{art} ist leer"


def test_jede_formel_gehoert_zu_genau_einer_art() -> None:
    """Ueber alle Sprachen: Keine Formel gehoert zu zwei Arten, etwa Gruss und Dank.

    Sonst entschiede die Reihenfolge der Pruefung, was geantwortet wird.
    """
    arten: dict[str, set[str]] = {}
    for formeln in KURZFORMELN.values():
        for art in ("begruessung", "dank", "abschied", "faehigkeiten", "identitaet"):
            for formel in getattr(formeln, art):
                arten.setdefault(normalisieren(formel), set()).add(art)
                # Jede Formel ist schon in Normalform geschrieben, bis auf ß.
                assert normalisieren(formel) == formel.replace("ß", "ss"), formel
    mehrdeutig = {formel: a for formel, a in arten.items() if len(a) > 1}
    assert mehrdeutig == {}


@pytest.mark.parametrize(
    ("eingabe", "art"),
    [("Thanks!", "thanks"), ("Danke schön", "thanks"), ("Bye!", "goodbye"), ("Tschüss", "goodbye")],
)
def test_dank_und_abschied_werden_erkannt(eingabe: str, art: str) -> None:
    assert kurzformel_erkennen(eingabe) == art


# Der Satz, der sagt, dass ein KI-System antwortet. Er steht als zweiter Satz
# in der Begruessung der Seite und ist damit das Erste, was vor der ersten
# Eingabe zu lesen ist.
KI_SATZ = {
    "de": "Hier antwortet ein KI-System, kein Mensch.",
    "en": "You are chatting with an AI system, not a person.",
}

BEGRUESSUNG = {
    "de": (
        "Guten Tag. Hier antwortet ein KI-System, kein Mensch. Ich beantworte Fragen zu "
        "{topics} — auf Grundlage der hinterlegten Unterlagen von {display_name}. Wenn die "
        "Unterlagen eine Frage nicht abdecken, sage ich das, statt zu raten."
    ),
    "en": (
        "Hello. You are chatting with an AI system, not a person. I answer questions about "
        "{topics} — based on the documents stored for {display_name}. If those documents do "
        "not cover a question, I say so instead of guessing."
    ),
}


def test_ki_satz_in_beiden_katalogen() -> None:
    """Der Satz steht woertlich als zweiter Satz der Begruessung, in beiden Sprachen.

    Der uebrige Text ist der von vorher, Zeichen fuer Zeichen, auch mit dem
    Gedankenstrich.
    """
    for sprache, satz in KI_SATZ.items():
        begruessung = TEXTE[sprache].begruessung
        assert begruessung == BEGRUESSUNG[sprache], sprache
        anrede, rest = begruessung.split(". ", 1)
        assert rest.startswith(satz), sprache
        assert "—" in begruessung, sprache


def test_antworten_ohne_modell_im_katalog() -> None:
    """Die drei Texte, woertlich, mit echten Umlauten im deutschen Katalog."""
    assert TEXTE["de"].begruessung_antwort == (
        "Guten Tag. Ich beantworte Fragen zu {topics}. Was möchten Sie wissen?"
    )
    assert TEXTE["en"].begruessung_antwort == (
        "Hello. I answer questions about {topics}. What would you like to know?"
    )
    assert TEXTE["de"].dank_antwort == "Gern geschehen. Haben Sie noch eine Frage?"
    assert TEXTE["en"].dank_antwort == (
        "You're welcome. Is there anything else you would like to know?"
    )
    assert TEXTE["de"].abschied_antwort == "Auf Wiedersehen und vielen Dank für Ihre Fragen."
    assert TEXTE["en"].abschied_antwort == "Goodbye, and thank you for your questions."
    # Serverseitig, nicht in der Seite.
    for name in ("begruessung_antwort", "dank_antwort", "abschied_antwort"):
        assert name not in JAVASCRIPT_SCHLUESSEL


# --- Fragen nach dem Assistenten: was er kann und wer antwortet --------------

# Woerter, die ein Mensch auf Deutsch gross schreibt, und die englischen
# Ausnahmen. Nur fuer `_wie_getippt`: Die Listen stehen in Normalform.
_GROSS = {
    "de": {
        "sie": "Sie",
        "ihr": "Ihr",
        "bot": "Bot",
        "ki": "KI",
        "mensch": "Mensch",
        "menschen": "Menschen",
        "roboter": "Roboter",
        "zweck": "Zweck",
    },
    "en": {"i": "I", "ai": "AI"},
}


def _wie_getippt(formel: str, sprache: str) -> str:
    """Die Formel so, wie ein Mensch sie tippt: Grossschreibung, Apostroph,
    Satzzeichen am Ende."""
    woerter = [_GROSS[sprache].get(wort, wort) for wort in formel.split()]
    text = " ".join(woerter).replace("what s ", "what's ")
    text = text[0].upper() + text[1:]
    return text + ("!" if formel in ("help", "hilfe") else "?")


_ASSISTENTENFRAGEN = [
    (sprache, art, formel)
    for sprache, formeln in KURZFORMELN.items()
    for art, liste in (("capabilities", formeln.faehigkeiten), ("identity", formeln.identitaet))
    for formel in liste
]


def test_die_listen_fuer_fragen_nach_dem_assistenten_sind_vollstaendig() -> None:
    """36 Formeln fuer capabilities und 34 fuer identity, ueber beide Sprachen."""
    assert sum(1 for _, art, _ in _ASSISTENTENFRAGEN if art == "capabilities") == 36
    assert sum(1 for _, art, _ in _ASSISTENTENFRAGEN if art == "identity") == 34
    # Echte Umlaute in den deutschen Listen, nicht die Umschreibung.
    deutsch = " ".join(f for s, _, f in _ASSISTENTENFRAGEN if s == "de")
    assert "können" in deutsch and "wofür" in deutsch
    assert "koennen" not in deutsch and "wofuer" not in deutsch


@pytest.mark.parametrize(("sprache", "art", "formel"), _ASSISTENTENFRAGEN)
def test_jede_frage_nach_dem_assistenten_wird_genau_erkannt(
    sprache: str, art: str, formel: str
) -> None:
    eingabe = _wie_getippt(formel, sprache)
    assert eingabe != formel
    assert kurzformel_erkennen(eingabe) == art, eingabe


def test_wie_getippt_sieht_aus_wie_ein_mensch_tippt() -> None:
    """Gegenprobe zur Hilfsfunktion des Tests oben."""
    assert _wie_getippt("sind sie ein bot", "de") == "Sind Sie ein Bot?"
    assert _wie_getippt("ist das eine ki", "de") == "Ist das eine KI?"
    assert _wie_getippt("what s your purpose", "en") == "What's your purpose?"
    assert _wie_getippt("am i talking to a bot", "en") == "Am I talking to a bot?"
    assert _wie_getippt("help", "en") == "Help!"


@pytest.mark.parametrize(
    ("a", "b", "abstand"),
    [
        ("", "", 0),
        ("", "abc", 3),
        ("bot", "bot", 0),
        ("bot", "bott", 1),
        ("how do you help", "how do ou help", 1),
        ("main", "mian", 2),
        ("kitten", "sitting", 3),
        ("wofür", "wofuer", 2),
    ],
)
def test_levenshtein_abstand(a: str, b: str, abstand: int) -> None:
    assert levenshtein_abstand(a, b) == abstand
    assert levenshtein_abstand(b, a) == abstand


@pytest.mark.parametrize(
    ("laenge", "toleranz"),
    [(0, 0), (7, 0), (8, 1), (15, 1), (16, 2), (40, 2)],
)
def test_tippfehler_toleranz(laenge: int, toleranz: int) -> None:
    """Unter 8 Zeichen nur genau, ab 8 hoechstens 1, ab 16 hoechstens 2."""
    assert tippfehler_toleranz(laenge) == toleranz


@pytest.mark.parametrize(
    ("eingabe", "art"),
    [
        ("How do ou help?", "capabilities"),
        ("Are you a bott?", "identity"),
        ("What is your mian purpose?", "capabilities"),
        ("Wobei konnen Sie mir helfen?", "capabilities"),
        ("Bist du ein Robotr?", "identity"),
    ],
)
def test_tippfehler_bei_fragen_nach_dem_assistenten(eingabe: str, art: str) -> None:
    """Der Pflichtfall "How do ou help?" und weitere Tippfehler, je ein Zeichen
    oder eine vertauschte Nachbarschaft."""
    assert kurzformel_erkennen(eingabe) == art


@pytest.mark.parametrize(
    "eingabe",
    [
        # Unter 8 Zeichen nur genau.
        "hepl",
        "Hilfee",
        # Gruss, Dank und Abschied haben keine Toleranz.
        "Good mornin",
        "thank yu",
        "goodbey",
        # Die kuerzere Laenge zaehlt: "help" hat 4 Zeichen.
        "help me",
        # Umschrift statt Umlaut bei einer kurzen Formel: Abstand 2 bei 14 Zeichen.
        "was koennen sie",
    ],
)
def test_ausserhalb_der_toleranz_trifft_nichts(eingabe: str) -> None:
    assert kurzformel_erkennen(eingabe) is None


def _gold_und_kontrollfragen() -> tuple[list[str], list[str]]:
    import yaml

    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    gold: list[str] = []
    kontrolle: list[str] = []
    for datei in sorted(eval_dir.glob("*/gold.yaml")):
        for eintrag in yaml.safe_load(datei.read_text(encoding="utf-8"))["fragen"]:
            gold.append(eintrag["frage"])
            if eintrag.get("kontrollfrage"):
                kontrolle.append(eintrag["kontrollfrage"])
    return gold, kontrolle


_GOLDFRAGEN, _KONTROLLFRAGEN = _gold_und_kontrollfragen()


def _verlaufsfragen() -> list[str]:
    """Die Folgefragen der Falldateien, ausser denen nach dem Assistenten selbst,
    und die erste Runde der Gegenprobe."""
    import yaml

    eval_dir = Path(__file__).resolve().parents[1] / "eval"
    fragen: list[str] = []
    for datei in sorted(eval_dir.glob("*/verlauf.yaml")):
        daten = yaml.safe_load(datei.read_text(encoding="utf-8"))
        fragen += [f["folgefrage"] for f in daten["faelle"] if not f.get("erwartete_art")]
        if daten.get("gegenprobe_vorrunde"):
            fragen.append(daten["gegenprobe_vorrunde"]["question"])
    return fragen


_VERLAUFSFRAGEN = _verlaufsfragen()

_ECHTE_FRAGEN_MIT_AEHNLICHEM_WORTLAUT = [
    "how do you ship",
    "how can i help",
    "what do you sell",
    "are you open",
    "what can i return",
    "how do i pay",
    "is this a bot question",
]


def test_die_gold_und_kontrollfragen_sind_geladen() -> None:
    """Gegenprobe gegen einen blinden Test unten: 45 Goldfragen, 11 Kontrollfragen."""
    assert len(_GOLDFRAGEN) == 45
    assert len(_KONTROLLFRAGEN) == 11
    # Sechs Folgefragen ohne erwartete Art und die erste Runde der Gegenprobe.
    assert len(_VERLAUFSFRAGEN) == 7


@pytest.mark.parametrize(
    "eingabe",
    _GOLDFRAGEN + _KONTROLLFRAGEN + _VERLAUFSFRAGEN + _ECHTE_FRAGEN_MIT_AEHNLICHEM_WORTLAUT,
)
def test_keine_echte_frage_wird_als_kurzformel_erkannt(eingabe: str) -> None:
    """Der Pflichttest gegen Fehltreffer. Eine Fachfrage, die hier traefe,
    bekaeme eine feste Antwort statt einer aus den Unterlagen."""
    assert kurzformel_erkennen(eingabe) is None


def test_ein_gleichstand_zwischen_beiden_arten_gilt_als_keine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treffen beide Arten mit demselben Abstand, gilt keine.

    In den echten Listen gibt es keinen Gleichstand: Die naechsten Formeln der
    beiden Arten, "what are you for" und "what are you", liegen 4 auseinander.
    Deshalb mit eingespeisten Listen.
    """
    from app import texts
    from app.texts import Kurzformeln

    leer: tuple[str, ...] = ()
    monkeypatch.setattr(
        texts,
        "KURZFORMELN",
        {
            "en": Kurzformeln(
                begruessung=leer,
                dank=leer,
                abschied=leer,
                faehigkeiten=("abcdefgh x",),
                identitaet=("abcdefgh y",),
            )
        },
    )
    assert kurzformel_erkennen("abcdefgh z") is None
    # Gegenprobe: Mit verschiedenem Abstand gewinnt der kleinere.
    assert kurzformel_erkennen("abcdefgh xx") == "capabilities"
    assert kurzformel_erkennen("abcdefgh y") == "identity"


def test_die_antworten_auf_fragen_nach_dem_assistenten_im_katalog() -> None:
    """Woertlich, mit echten Umlauten, ohne Gedankenstrich, nur serverseitig."""
    assert TEXTE["de"].faehigkeiten_antwort == (
        "Ich beantworte Fragen zu {topics}, auf Grundlage der hinterlegten Unterlagen von "
        "{display_name}. Was möchten Sie wissen?"
    )
    assert TEXTE["en"].faehigkeiten_antwort == (
        "I answer questions about {topics}, based on the documents stored for "
        "{display_name}. What would you like to know?"
    )
    assert TEXTE["de"].identitaet_antwort == (
        "Hier antwortet ein KI-System, kein Mensch. Ich beantworte Fragen zu {topics}, auf "
        "Grundlage der hinterlegten Unterlagen von {display_name}."
    )
    assert TEXTE["en"].identitaet_antwort == (
        "You are chatting with an AI system, not a person. I answer questions about "
        "{topics}, based on the documents stored for {display_name}."
    )
    for sprache, texte in TEXTE.items():
        for name in ("faehigkeiten_antwort", "identitaet_antwort"):
            assert "—" not in getattr(texte, name), (sprache, name)
            assert name not in JAVASCRIPT_SCHLUESSEL
        # Die Identitaetsantwort beginnt mit dem KI-Satz der Begruessung.
        assert texte.identitaet_antwort.startswith(KI_SATZ[sprache]), sprache


# --- Gespraech: Umformungsprompt und Anzeige ----------------------------------


def _umformungstexte(sprache: str) -> dict[str, str]:
    """Alle Zeichenketten, die bei der Umformung an das Modell gehen."""
    umformung = _UMFORMUNG[sprache]
    schema = umformungsmodell_fuer(sprache).model_json_schema()
    return {
        "system": umformung.system,
        "rahmen": " ".join(umformung.rahmen),
        "struktur": schema["description"],
        "question": schema["properties"]["question"]["description"],
    }


def test_umformung_deckt_dieselben_sprachen_ab_wie_das_regelwerk() -> None:
    """Sonst liefe eine Sprache erst bei der ersten Folgefrage in einen KeyError."""
    assert set(_UMFORMUNG) == set(_REGELWERK) == set(TEXTE)
    assert set(UMFORMUNGSBESCHREIBUNGEN) == set(TEXTE)


def test_umformungsprompt_traegt_nur_marker_seiner_sprache() -> None:
    """System, Rahmen und Feldbeschreibungen der Umformung tragen keine
    Funktionswoerter einer anderen Sprache, und ausser im Deutschen keine
    deutschen Zeichen. Derselbe Fehler wie beim Antwortschema: Ein Prompt in der
    falschen Sprache kippt die Sprache der Antwort, gemessen."""
    for sprache in VERFUEGBARE_SPRACHEN:
        texte = _umformungstexte(sprache)
        for name, text in texte.items():
            assert text.strip(), (sprache, name)
            for fremde in VERFUEGBARE_SPRACHEN:
                if fremde == sprache:
                    continue
                treffer = _marker_treffer(text, fremde)
                assert not treffer, f"Umformung {sprache!r}, {name!r}: {treffer}"
            if sprache != "de":
                assert not set(text) & _DEUTSCHE_ZEICHEN, (sprache, name)


def test_umformungsprompt_marker_gegenprobe() -> None:
    """Gegenprobe: Die Pruefung oben schlaegt an einem vertauschten Text an."""
    assert _marker_treffer(_UMFORMUNG["de"].system, "de")
    assert _marker_treffer(_UMFORMUNG["en"].system, "en")
    assert _marker_treffer(_UMFORMUNG["de"].system + " the", "en") == ["the"]


def test_gespraechstexte_im_katalog() -> None:
    """Woertlich, mit echtem Umlaut. "Verstanden als" traegt den Platzhalter fuer
    die eigenstaendige Frage und geht in den Browser, der Knopf nicht."""
    assert TEXTE["de"].verstanden_als == "Verstanden als: {frage}"
    assert TEXTE["en"].verstanden_als == "Understood as: {frage}"
    assert TEXTE["de"].neues_gespraech == "Neues Gespräch"
    assert TEXTE["en"].neues_gespraech == "New conversation"
    assert "verstanden_als" in JAVASCRIPT_SCHLUESSEL
    assert "neues_gespraech" not in JAVASCRIPT_SCHLUESSEL
