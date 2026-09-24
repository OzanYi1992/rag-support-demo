"""Oberflaechen-, Hinweis- und Fehlertexte je Sprache.

Zwei Kataloge, `de` und `en`, an genau einer Stelle. Kein gettext, keine
Uebersetzungsbibliothek: Fuer zwei Sprachen und eine Handvoll Saetze waere eine
Uebersetzungsschicht Aufwand ohne Gegenwert.

Der Katalog ist gleichzeitig die Liste der zulaessigen Mandantensprachen. Die
Mandantenkonfiguration prueft gegen `VERFUEGBARE_SPRACHEN`, nicht gegen eine
zweite, eigene Aufzaehlung - sonst koennten Schema und Katalog auseinanderlaufen
und ein Mandant traege eine Sprache, fuer die es keine Texte gibt.

Was hier NICHT hineingehoert:

* Die Eskalationsnachricht. Sie steht je Mandant in dessen `tenant.yaml`, weil
  sie die Supportadresse dieses Mandanten nennt. Ein globaler Katalog kann sie
  nicht tragen.
* Der Wortlaut fuer ein unbekanntes url_token. Er faellt, BEVOR ein Mandant
  aufgeloest ist - an der Stelle gibt es keine Sprache. Zwei Wortlaute waeren
  ausserdem ein Orakel: Wer einen Unterschied sieht, kann Token einkreisen.
  Siehe `app/main.py`.
* Das Regelwerk des System-Prompts. Es steht in `app/prompts.py`.

Was hier seit dem 2026-09-24 SEHR WOHL hineingehoert, obwohl es sich an das
Modell richtet und nicht an einen Menschen: die Feldbeschreibungen der
Antwortstruktur, siehe `Schemabeschreibungen`. Damit tragen zwei Dateien je
Sprache formulierten Text fuer das Modell - das Regelwerk dort, die
Beschreibungen hier. Diese Grenze ist historisch und nicht sachlich; ob sie
zusammengelegt wird, ist eine offene Frage (open-points.md, OP-055).
"""

from __future__ import annotations

from pydantic import BaseModel

# Schluessel, die die Oberflaeche im Browser braucht. Sie werden als JSON in die
# Seite geschrieben; alles andere bleibt serverseitig.
JAVASCRIPT_SCHLUESSEL: tuple[str, ...] = (
    "sucht",
    "eskalation_titel",
    "eskalation_hinweis",
    "quelle",
    "quellen",
    "ratenlimit_mit_zeit",
    "ratenlimit_ohne_zeit",
    "anfrage_fehlgeschlagen",
    "mehr_anzeigen",
    "weniger_anzeigen",
    "eskalation_themen",
)


class Texte(BaseModel, frozen=True):
    """Alle Texte einer Sprache.

    Ein Pydantic-Modell und kein `dict`: Eine im zweiten Katalog vergessene Zeile
    ist damit ein Fehler beim Import und nicht eine deutsche Zeile in einer
    englischen Oberflaeche, die erst vor einem Interessenten auffaellt.
    """

    # --- Seite ------------------------------------------------------------
    titel: str
    begruessung: str
    frage_label: str
    frage_platzhalter: str
    senden: str
    fusszeile: str

    # --- Oberflaeche im Browser ------------------------------------------
    sucht: str
    eskalation_titel: str
    eskalation_hinweis: str
    quelle: str
    quellen: str
    ratenlimit_mit_zeit: str
    ratenlimit_ohne_zeit: str
    anfrage_fehlgeschlagen: str

    # Aufklappen langer Antworten. Die LAENGE wird in der Anzeige begrenzt und
    # nicht im Prompt: Eine Knappheitsforderung im Prompt hat am 2026-09-24 das
    # Groundedness-Tor gelockert (P-030). Eine Kuerzung im Browser kann das
    # nicht, weil sie den Prompt nicht anfasst.
    mehr_anzeigen: str
    weniger_anzeigen: str

    # Der Satz im Eskalationskasten, der nennt, worueber Auskunft moeglich ist.
    # Traegt den Platzhalter {topics} und bekommt denselben Text wie die
    # Begruessung - nicht eine zweite, eigene Aufzaehlung, die auseinanderlaufen
    # koennte.
    #
    # Warum es ihn gibt: Heute sieht jede Eskalation gleich aus, egal ob die
    # Frage unbeantwortbar oder nur zu knapp war. Viermal derselbe Satz liest
    # sich beim Interessenten wie eine statische Seite ohne KI.
    eskalation_themen: str

    # --- Serverseitige Meldungen -----------------------------------------
    ratenlimit_detail: str
    oberflaeche_fehlt: str

    def fuer_javascript(self) -> dict[str, str]:
        """Die Teilmenge, die im Browser gebraucht wird.

        Ausdruecklich eine Auswahl und nicht das ganze Modell: Was der Server
        allein braucht, hat in einer ausgelieferten Seite nichts zu suchen.
        """
        return {name: getattr(self, name) for name in JAVASCRIPT_SCHLUESSEL}


_DE = Texte(
    titel="Support-Assistent",
    begruessung=(
        "Guten Tag. Ich beantworte Fragen zu {topics} — auf Grundlage der "
        "hinterlegten Unterlagen von {display_name}. Wenn die Unterlagen eine "
        "Frage nicht abdecken, sage ich das, statt zu raten."
    ),
    frage_label="Ihre Frage",
    frage_platzhalter="Ihre Frage — Enter zum Senden, Umschalt+Enter für eine neue Zeile",
    senden="Senden",
    fusszeile="Demo. Antworten stammen ausschließlich aus den hinterlegten Unterlagen.",
    sucht="Sucht in den Unterlagen …",
    eskalation_titel="Nicht in den Unterlagen. ",
    eskalation_hinweis=(
        "Diese Frage wird von den hinterlegten Inhalten nicht abgedeckt. "
        "Der Assistent rät in diesem Fall nicht, sondern verweist weiter."
    ),
    quelle="Quelle",
    quellen="Quellen",
    ratenlimit_mit_zeit="Zu viele Anfragen. Bitte {sekunden} Sekunden warten.",
    ratenlimit_ohne_zeit="Zu viele Anfragen. Bitte einen Moment warten.",
    anfrage_fehlgeschlagen="Die Anfrage ist fehlgeschlagen.",
    mehr_anzeigen="Ganze Antwort anzeigen",
    weniger_anzeigen="Antwort einklappen",
    eskalation_themen="Auskunft ist möglich zu {topics}.",
    ratenlimit_detail="Zu viele Anfragen. Bitte kurz warten.",
    oberflaeche_fehlt="Oberflaeche fehlt.",
)

_EN = Texte(
    titel="Support Assistant",
    begruessung=(
        "Hello. I answer questions about {topics} — based on the documents "
        "stored for {display_name}. If those documents do not cover a question, "
        "I say so instead of guessing."
    ),
    frage_label="Your question",
    frage_platzhalter="Your question — Enter to send, Shift+Enter for a new line",
    senden="Send",
    fusszeile="Demo. Answers come exclusively from the stored documents.",
    sucht="Searching the documents …",
    eskalation_titel="Not in the documents. ",
    eskalation_hinweis=(
        "The stored content does not cover this question. In that case the "
        "assistant does not guess; it refers you on."
    ),
    quelle="Source",
    quellen="Sources",
    ratenlimit_mit_zeit="Too many requests. Please wait {sekunden} seconds.",
    ratenlimit_ohne_zeit="Too many requests. Please wait a moment.",
    anfrage_fehlgeschlagen="The request failed.",
    mehr_anzeigen="Show the full answer",
    weniger_anzeigen="Collapse the answer",
    eskalation_themen="I can answer about {topics}.",
    ratenlimit_detail="Too many requests. Please wait a moment.",
    oberflaeche_fehlt="User interface missing.",
)

TEXTE: dict[str, Texte] = {"de": _DE, "en": _EN}

# Die zulaessigen Werte fuer das Feld `language` eines Mandanten. Abgeleitet,
# nicht zweitgefuehrt.
VERFUEGBARE_SPRACHEN: tuple[str, ...] = tuple(sorted(TEXTE))


# Kurze Markerlisten fuer die Sprachpruefung unten. Bewusst Funktionswoerter:
# Sie kommen in fast jedem Satz vor, stehen in beiden Sprachen praktisch nie
# gleichzeitig, und sie haengen nicht am Thema des Textes.
_MARKER: dict[str, frozenset[str]] = {
    "de": frozenset(
        {
            "der",
            "die",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "einen",
            "und",
            "oder",
            "nicht",
            "kein",
            "keine",
            "ist",
            "sind",
            "wir",
            "sie",
            "ihnen",
            "ihre",
            "sich",
            "dazu",
            "bitte",
            "finde",
            "steht",
            "unterlagen",
            "mit",
            "von",
            "auf",
            "wenden",
            "weiter",
            "direkt",
        }
    ),
    "en": frozenset(
        {
            "the",
            "and",
            "not",
            "cannot",
            "can",
            "that",
            "this",
            "with",
            "from",
            "for",
            "about",
            "please",
            "you",
            "your",
            "our",
            "will",
            "find",
            "anything",
            "documents",
            "someone",
            "help",
            "write",
            "there",
            "them",
            "have",
        }
    ),
}

# Zeichen, die es nur im Deutschen gibt. Ein hartes Signal: Steht eines davon in
# einem englischen Text, ist es kein englischer Text.
_DEUTSCHE_ZEICHEN = frozenset("äöüÄÖÜß")


def passt_zur_sprache(text: str, language: str) -> str | None:
    """Prueft grob, ob ein Text in der erwarteten Sprache verfasst ist.

    Gibt None zurueck, wenn nichts dagegen spricht, sonst eine Begruendung.

    Wofuer das da ist: Die `escalation_message` eines Mandanten ist der einzige
    Text, den kein Katalog traegt - sie nennt die Supportadresse dieses
    Mandanten. Damit ist sie auch der einzige, den kein Katalogtest schuetzt.
    Ein Mandant mit `language: en` und einer deutschen Eskalationsnachricht
    besteht sonst jede Pruefung und faellt erst vor einem Interessenten auf,
    und zwar genau bei der Frage, die das System richtig beantwortet: der nicht
    gedeckten.

    Das ist eine Heuristik und will keine Spracherkennung sein. Sie ist
    bewusst KONSERVATIV: Abgewiesen wird nur, wenn die andere Sprache deutlich
    gewinnt. Ein kurzer oder ungewoehnlicher Text, bei dem kein Marker greift,
    laeuft durch. Der Fehler, den sie fangen soll, ist ein ganzer Satz in der
    falschen Sprache - und der bringt reichlich Marker mit.

    Kein Sprachmodell: Das waere ein Netzaufruf beim Laden eines Mandanten und
    eine Abhaengigkeit an einer Stelle, die keine braucht.
    """
    if language not in _MARKER:
        raise KeyError(f"Keine Marker fuer die Sprache {language!r}.")

    if language == "en" and set(text) & _DEUTSCHE_ZEICHEN:
        getroffen = "".join(sorted(set(text) & _DEUTSCHE_ZEICHEN))
        return f"enthaelt die Zeichen {getroffen!r}, die es im Englischen nicht gibt"

    woerter = {wort.strip(".,;:!?()\"'").lower() for wort in text.split()}
    eigene = len(woerter & _MARKER[language])
    fremde = {
        andere: len(woerter & marker) for andere, marker in _MARKER.items() if andere != language
    }
    staerkste, punkte = max(fremde.items(), key=lambda paar: paar[1])

    if punkte > eigene:
        return (
            f"enthaelt {punkte} typische Woerter der Sprache {staerkste!r}, "
            f"aber nur {eigene} der erwarteten Sprache {language!r}"
        )
    return None


def texte_fuer(language: str) -> Texte:
    """Liefert den Katalog einer Sprache.

    Kein Rueckfall auf Deutsch bei unbekanntem Wert: Ein stiller Rueckfall
    liesse einen Tippfehler in der `tenant.yaml` als deutsche Oberflaeche bei
    einem englischen Mandanten enden, ohne Fehlermeldung. Geprueft wird der Wert
    ohnehin schon beim Laden des Mandanten; hier steht die zweite Ebene.
    """
    if language not in TEXTE:
        raise KeyError(
            f"Kein Textkatalog fuer die Sprache {language!r}. "
            f"Vorhanden: {', '.join(VERFUEGBARE_SPRACHEN)}."
        )
    return TEXTE[language]


# =============================================================================
# DIE FELDBESCHREIBUNGEN DER ANTWORTSTRUKTUR
#
# Warum diese Texte eine Sprache haben muessen, und warum das teuer war:
#
# Bis zum 2026-09-24 waren sie fest deutsch, fuer JEDEN Mandanten. Sie gehen
# nicht ueber den System- oder Userprompt an das Modell, sondern ueber
# `response_format` - also an einer Stelle, die kein Mensch je zu Gesicht
# bekommt und die keine Testhilfe beruehrt.
#
# Gemessen am 2026-09-24 gegen `demo-fellgate` (language: en), Frage
# "Can I Return my boots", System- und Userprompt vollstaendig englisch:
# 23 von 30 Antworten kamen auf DEUTSCH. Mit inhaltlich gleichen, aber
# englischen Beschreibungen: 0 von 10. Eine Variable, eindeutige Wirkung.
#
# Das ist P-024 eine Ebene tiefer. Beim ersten Mal war es der Rahmen des
# Userprompts, hier die Beschreibung des Feldes, in das die Antwort geschrieben
# wird. Beide Male eine Anweisung in der falschen Sprache an einer Stelle, die
# nur das Modell liest.
# =============================================================================


class Schemabeschreibungen(BaseModel, frozen=True):
    """Die Feldbeschreibungen der Antwortstruktur in einer Sprache.

    Die Feldnamen sind genau die des Schemas, damit die Zuordnung ohne
    Nachdenken stimmt. `struktur` ist die Beschreibung der Struktur selbst; sie
    landet im Schema als `description` auf der obersten Ebene.

    Ein frozenes Modell und kein `dict`, aus demselben Grund wie bei `Texte`:
    Eine im zweiten Katalog vergessene Zeile ist ein Fehler beim Import und
    nicht eine deutsche Zeile im Prompt eines englischen Mandanten.
    """

    struktur: str
    answerable: str
    answer: str
    sources: str
    language: str


# Der deutsche Wortlaut ist UNVERAENDERT der, der bis zum 2026-09-24 in
# app/prompts.py stand - Zeichen fuer Zeichen, inklusive der Zeilenumbrueche im
# Strukturtext. Fuer die deutschen Mandanten aendert sich damit nichts, und
# `test_deutsches_schema_ist_wortgleich_mit_dem_alten_stand` haelt das fest.
_SCHEMA_DE = Schemabeschreibungen(
    struktur=(
        "Was das Modell zurueckgeben muss.\n"
        "\n"
        "`answerable` ist das zweite Eskalationstor. Es ist bewusst ein eigenes Feld\n"
        'und keine Formulierung im Antworttext: Ein "das steht leider nicht in den\n'
        'Unterlagen" mitten in einem ansonsten erfundenen Absatz waere nicht\n'
        "auswertbar."
    ),
    answerable=(
        "true, wenn die Frage aus dem gelieferten Kontext vollstaendig "
        "beantwortet werden kann. false, wenn der Kontext die Frage nicht "
        "oder nur teilweise abdeckt."
    ),
    answer=(
        "Die Antwort, ausschliesslich aus dem Kontext. Leer lassen, wenn answerable false ist."
    ),
    sources=(
        "Dateinamen aus dem Kontext, auf denen die Antwort beruht. Nur "
        "Dateinamen, die im Kontext vorkommen."
    ),
    language="Sprache der Antwort als ISO-639-1-Kuerzel, etwa 'de' oder 'en'.",
)

# Inhaltlich dieselben Aussagen, nicht mehr und nicht weniger. Insbesondere
# bleibt Punkt 4 des Regelwerks erhalten - "oder nur teilweise abdeckt" ist die
# Haelfte des Groundedness-Tors und darf in der Uebersetzung nicht weicher
# werden.
_SCHEMA_EN = Schemabeschreibungen(
    struktur=(
        "What the model must return.\n"
        "\n"
        "`answerable` is the second escalation gate. It is deliberately a field of\n"
        'its own and not a phrase in the answer text: a "that is unfortunately not\n'
        'in the documents" in the middle of an otherwise invented paragraph would\n'
        "not be evaluable."
    ),
    answerable=(
        "true if the question can be answered completely from the context "
        "provided. false if the context does not cover the question, or "
        "covers it only partially."
    ),
    answer=("The answer, exclusively from the context. Leave empty when answerable is false."),
    sources=(
        "File names from the context that the answer rests on. Only file "
        "names that appear in the context."
    ),
    language="Language of the answer as an ISO 639-1 code, for example 'de' or 'en'.",
)

SCHEMABESCHREIBUNGEN: dict[str, Schemabeschreibungen] = {"de": _SCHEMA_DE, "en": _SCHEMA_EN}


def schemabeschreibungen_fuer(language: str) -> Schemabeschreibungen:
    """Liefert die Feldbeschreibungen einer Sprache.

    Kein Rueckfall auf Deutsch, aus demselben Grund wie bei `texte_fuer`, und
    hier waere der Rueckfall genau der Fehler, den diese Datei abstellt.
    """
    if language not in SCHEMABESCHREIBUNGEN:
        raise KeyError(
            f"Keine Schemabeschreibungen fuer die Sprache {language!r}. "
            f"Vorhanden: {', '.join(sorted(SCHEMABESCHREIBUNGEN))}."
        )
    return SCHEMABESCHREIBUNGEN[language]
