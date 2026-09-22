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
* Das Regelwerk des System-Prompts. Es richtet sich an das Modell, nicht an
  einen Menschen, und steht deshalb in `app/prompts.py`.
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
        "Guten Tag. Ich beantworte Fragen auf Grundlage der hinterlegten "
        "Unterlagen von {display_name}. Wenn die Unterlagen eine Frage nicht "
        "abdecken, sage ich das, statt zu raten."
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
    ratenlimit_detail="Zu viele Anfragen. Bitte kurz warten.",
    oberflaeche_fehlt="Oberflaeche fehlt.",
)

_EN = Texte(
    titel="Support Assistant",
    begruessung=(
        "Hello. I answer questions based on the documents stored for "
        "{display_name}. If those documents do not cover a question, I say so "
        "instead of guessing."
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
    ratenlimit_detail="Too many requests. Please wait a moment.",
    oberflaeche_fehlt="User interface missing.",
)

TEXTE: dict[str, Texte] = {"de": _DE, "en": _EN}

# Die zulaessigen Werte fuer das Feld `language` eines Mandanten. Abgeleitet,
# nicht zweitgefuehrt.
VERFUEGBARE_SPRACHEN: tuple[str, ...] = tuple(sorted(TEXTE))


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
