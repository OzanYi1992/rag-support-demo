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
    "dokument_rahmen_titel",
    "anbieter_nicht_erreichbar",
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

    # Der Titel des Rahmens, in dem ein Quelldokument aufklappt. Ein
    # Screenreader liest ihn vor, wenn er den Rahmen erreicht. Traegt den
    # Platzhalter {datei}, den die Oberflaeche mit dem Dateinamen fuellt.
    dokument_rahmen_titel: str

    # Die Meldung, wenn der Modellanbieter nicht antwortet oder ueberlastet ist
    # (HTTP 503). Bewusst NICHT die Eskalationsmeldung des Mandanten: Die sagt,
    # die Unterlagen deckten die Frage nicht, und das waere bei einem Ausfall
    # eine falsche Auskunft ueber die Wissensbasis.
    anbieter_nicht_erreichbar: str

    # --- Serverseitige Meldungen -----------------------------------------
    ratenlimit_detail: str
    oberflaeche_fehlt: str

    # --- Antworten ohne Modell ---------------------------------------------
    # Auf eine Begruessung, einen Dank oder einen Abschied antwortet der
    # Assistent aus diesem Katalog, ohne Retrieval und ohne Modellaufruf. Die
    # Eingabe muss dafuer genau einer Formel aus KURZFORMELN entsprechen.
    # Vorher eskalierte ein "Hi" mit dem Satz, die Unterlagen deckten das
    # nicht, und kostete einen Modellaufruf.
    #
    # Die Begruessungsantwort traegt {topics} und bekommt denselben Text wie
    # die Begruessung der Seite.
    begruessung_antwort: str
    dank_antwort: str
    abschied_antwort: str

    # Auf eine Frage nach dem Assistenten selbst, was er kann und wozu er da ist
    # oder ob ein Mensch antwortet, ebenfalls aus diesem Katalog. Beide tragen
    # {topics} und {display_name}, wie die Begruessung der Seite.
    faehigkeiten_antwort: str
    identitaet_antwort: str

    # Der Titel des Dokuments ueber den Assistenten. Er ersetzt dessen
    # Dateinamen, wo das Modell ihn im Fliesstext nennt ("according to
    # about-this-assistant.md"). Steht in der Sprache der Antwort, mit den
    # Anfuehrungszeichen dieser Sprache.
    assistentendokument_titel: str

    def fuer_javascript(self) -> dict[str, str]:
        """Die Teilmenge, die im Browser gebraucht wird.

        Ausdruecklich eine Auswahl und nicht das ganze Modell: Was der Server
        allein braucht, hat in einer ausgelieferten Seite nichts zu suchen.
        """
        return {name: getattr(self, name) for name in JAVASCRIPT_SCHLUESSEL}


_DE = Texte(
    titel="Support-Assistent",
    begruessung=(
        "Guten Tag. Hier antwortet ein KI-System, kein Mensch. Ich beantworte "
        "Fragen zu {topics} — auf Grundlage der hinterlegten Unterlagen von "
        "{display_name}. Wenn die Unterlagen eine Frage nicht abdecken, sage ich "
        "das, statt zu raten."
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
    dokument_rahmen_titel="Quelldokument {datei}",
    anbieter_nicht_erreichbar=(
        "Der Assistent ist gerade nicht erreichbar. "
        "Bitte versuchen Sie es in einer Minute noch einmal."
    ),
    ratenlimit_detail="Zu viele Anfragen. Bitte kurz warten.",
    oberflaeche_fehlt="Oberflaeche fehlt.",
    begruessung_antwort="Guten Tag. Ich beantworte Fragen zu {topics}. Was möchten Sie wissen?",
    dank_antwort="Gern geschehen. Haben Sie noch eine Frage?",
    abschied_antwort="Auf Wiedersehen und vielen Dank für Ihre Fragen.",
    faehigkeiten_antwort=(
        "Ich beantworte Fragen zu {topics}, auf Grundlage der hinterlegten Unterlagen von "
        "{display_name}. Was möchten Sie wissen?"
    ),
    identitaet_antwort=(
        "Hier antwortet ein KI-System, kein Mensch. Ich beantworte Fragen zu {topics}, auf "
        "Grundlage der hinterlegten Unterlagen von {display_name}."
    ),
    assistentendokument_titel="„Über den Assistenten“",
)

_EN = Texte(
    titel="Support Assistant",
    begruessung=(
        "Hello. You are chatting with an AI system, not a person. I answer "
        "questions about {topics} — based on the documents stored for "
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
    mehr_anzeigen="Show the full answer",
    weniger_anzeigen="Collapse the answer",
    eskalation_themen="I can answer about {topics}.",
    dokument_rahmen_titel="Source document {datei}",
    anbieter_nicht_erreichbar=(
        "The assistant cannot be reached right now. Please try again in a minute."
    ),
    ratenlimit_detail="Too many requests. Please wait a moment.",
    oberflaeche_fehlt="User interface missing.",
    begruessung_antwort="Hello. I answer questions about {topics}. What would you like to know?",
    dank_antwort="You're welcome. Is there anything else you would like to know?",
    abschied_antwort="Goodbye, and thank you for your questions.",
    faehigkeiten_antwort=(
        "I answer questions about {topics}, based on the documents stored for "
        "{display_name}. What would you like to know?"
    ),
    identitaet_antwort=(
        "You are chatting with an AI system, not a person. I answer questions about "
        "{topics}, based on the documents stored for {display_name}."
    ),
    assistentendokument_titel="“About This Assistant”",
)

TEXTE: dict[str, Texte] = {"de": _DE, "en": _EN}

# Die zulaessigen Werte fuer das Feld `language` eines Mandanten. Abgeleitet,
# nicht zweitgefuehrt.
VERFUEGBARE_SPRACHEN: tuple[str, ...] = tuple(sorted(TEXTE))


# =============================================================================
# BEGRUESSUNG, DANK UND ABSCHIED OHNE MODELL
#
# Erkannt wird nur die GANZE Eingabe, nach der Normalisierung genau gleich
# einer Formel. "Hi, do you ship to Norway?" ist keine Begruessung, sondern
# eine Frage mit Gruss davor, und laeuft den normalen Weg. Eine Erkennung, die
# weiter greift, beantwortete echte Fragen mit einem Gruss.
#
# Geprueft wird gegen die Formeln ALLER Sprachen, geantwortet in der Sprache des
# Mandanten. Ein "Hallo" beim englischen Mandanten bekommt die englische
# Antwort: Sie nennt die Themen des Mandanten, und die stehen in der
# tenant.yaml nur in seiner Sprache.
#
# Die englische Liste traegt dazu nordische Gruesse, weil die Empfaenger der
# Demo nordisch sind und die Antwort ohnehin englisch ist: hej, hei, moi und
# das schwedische hallå. Gruesse anderer Sprachen erkennt die Liste nicht; sie
# laufen den normalen Weg.
#
# FRAGEN NACH DEM ASSISTENTEN SELBST, auf demselben Weg und ebenfalls ohne
# Modell: was er kann und wozu er da ist ("capabilities"), und ob ein Mensch,
# ein Bot oder eine KI antwortet ("identity"). Das Dokument ueber den
# Assistenten im Korpus beantwortet solche Fragen nicht verlaesslich: Gemessen
# eskalierte "Are you a bot?" bei Fellgate in 18 von 20 Laeufen, obwohl das
# Dokument auf Rang 1 im Kontext lag.
#
# Nur fuer diese beiden Arten gilt eine Toleranz fuer Tippfehler, weil genau
# solche Fragen schnell und ungenau getippt werden ("How do ou help?"). Fuer
# Gruss, Dank und Abschied gilt sie nicht: Die Formeln sind kurz, und ein
# Abstand von 1 machte aus "hi" jedes Wort mit zwei Buchstaben.
# =============================================================================


class Kurzformeln(BaseModel, frozen=True):
    """Die Formeln einer Sprache, je Art. Ungeordnet, verglichen nach
    `normalisieren`."""

    begruessung: tuple[str, ...]
    dank: tuple[str, ...]
    abschied: tuple[str, ...]
    faehigkeiten: tuple[str, ...]
    identitaet: tuple[str, ...]


KURZFORMELN: dict[str, Kurzformeln] = {
    "de": Kurzformeln(
        begruessung=(
            "hallo",
            "hi",
            "hey",
            "moin",
            "servus",
            "guten tag",
            "guten morgen",
            "guten abend",
            "grüß gott",
        ),
        dank=("danke", "danke schön", "danke sehr", "vielen dank"),
        abschied=("tschüss", "auf wiedersehen", "bis bald", "ciao"),
        faehigkeiten=(
            "was kannst du",
            "was können sie",
            "wobei können sie mir helfen",
            "wobei kannst du mir helfen",
            "wobei können sie helfen",
            "wie können sie mir helfen",
            "wie kannst du mir helfen",
            "wie helfen sie",
            "wofür bist du da",
            "wofür sind sie da",
            "was ist ihr zweck",
            "was ist dein zweck",
            "was machst du",
            "was machen sie",
            "was kann ich fragen",
            "was kann ich sie fragen",
            "hilfe",
        ),
        identitaet=(
            "bist du ein bot",
            "sind sie ein bot",
            "bist du ein roboter",
            "sind sie ein roboter",
            "bist du ein mensch",
            "sind sie ein mensch",
            "bist du eine ki",
            "sind sie eine ki",
            "ist das ein bot",
            "ist das eine ki",
            "spreche ich mit einem menschen",
            "spreche ich mit einem bot",
            "rede ich mit einem menschen",
            "wer bist du",
            "wer sind sie",
        ),
    ),
    "en": Kurzformeln(
        begruessung=(
            "hi",
            "hello",
            "hey",
            "hi there",
            "hello there",
            "hey there",
            "good morning",
            "good afternoon",
            "good evening",
            "hej",
            "hei",
            "moi",
            "hallå",
        ),
        dank=("thanks", "thank you", "thanks a lot", "thank you very much", "cheers"),
        abschied=("bye", "goodbye", "bye bye", "see you"),
        faehigkeiten=(
            "what can you do",
            "what can you help with",
            "what can you help me with",
            "how can you help",
            "how can you help me",
            "how do you help",
            "how do you help me",
            "what do you do",
            "what are you for",
            "what is your purpose",
            "what is your main purpose",
            "what s your purpose",
            "whats your purpose",
            "what can i ask",
            "what can i ask you",
            "what questions can you answer",
            "what do you know",
            "what is this",
            "help",
        ),
        identitaet=(
            "are you a bot",
            "are you a robot",
            "are you a chatbot",
            "are you human",
            "are you a human",
            "are you a person",
            "are you a real person",
            "are you real",
            "are you ai",
            "are you an ai",
            "is this a bot",
            "is this ai",
            "is this a real person",
            "am i talking to a bot",
            "am i talking to a human",
            "am i talking to a person",
            "am i talking to a real person",
            "who are you",
            "what are you",
        ),
    ),
}

# Die Arten, wie sie in `Answer.kind` und im Ereignis stehen. Englisch wie die
# uebrigen Felder der Antwort.
ART_BEGRUESSUNG = "greeting"
ART_DANK = "thanks"
ART_ABSCHIED = "goodbye"
ART_FAEHIGKEITEN = "capabilities"
ART_IDENTITAET = "identity"


def normalisieren(text: str) -> str:
    """Kleinschreibung, jedes Zeichen ausser Buchstabe und Ziffer ein Leerzeichen,
    Leerraum zusammengefasst und am Rand entfernt.

    `casefold` statt `lower`, damit "GRÜSS" und "grüß" gleich werden. Dieselbe
    Funktion laeuft ueber die Formeln, sonst traefe "Grüß Gott" nie: casefold
    macht aus ß ein ss, in der Eingabe wie in der Liste.
    """
    zeichen = (z if z.isalnum() else " " for z in text.casefold())
    return " ".join("".join(zeichen).split())


def levenshtein_abstand(a: str, b: str) -> int:
    """Wie viele Zeichen eingefuegt, geloescht oder ersetzt werden muessen, damit
    aus `a` `b` wird. Eine vertauschte Nachbarschaft ("mian" statt "main") zaehlt
    zwei.

    Eigene Funktion statt einer Abhaengigkeit: zwei Zeilen der Matrix, fuer
    Eingaben von der Laenge einer Frage.
    """
    if len(a) < len(b):
        a, b = b, a
    vorher = list(range(len(b) + 1))
    for i, zeichen_a in enumerate(a, 1):
        jetzt = [i]
        for j, zeichen_b in enumerate(b, 1):
            jetzt.append(
                min(vorher[j] + 1, jetzt[j - 1] + 1, vorher[j - 1] + (zeichen_a != zeichen_b))
            )
        vorher = jetzt
    return vorher[-1]


def tippfehler_toleranz(laenge: int) -> int:
    """Der hoechste zulaessige Abstand bei dieser Laenge.

    Unter 8 Zeichen nur genau, ab 8 Zeichen hoechstens 1, ab 16 hoechstens 2.
    `laenge` ist die kuerzere von Eingabe und Formel: Die Toleranz waechst erst,
    wenn beide lang genug sind.
    """
    if laenge >= 16:
        return 2
    if laenge >= 8:
        return 1
    return 0


def _arten(formeln: Kurzformeln) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        (ART_BEGRUESSUNG, formeln.begruessung),
        (ART_DANK, formeln.dank),
        (ART_ABSCHIED, formeln.abschied),
        (ART_FAEHIGKEITEN, formeln.faehigkeiten),
        (ART_IDENTITAET, formeln.identitaet),
    )


def _nach_tippfehler(normal: str) -> str | None:
    """Die Frage nach dem Assistenten, deren Formel der Eingabe am naechsten
    liegt, innerhalb der Toleranz.

    Treffen beide Arten mit demselben kleinsten Abstand, gilt keine: Dann ist
    nicht zu entscheiden, was gemeint ist, und die Eingabe laeuft den normalen
    Weg.
    """
    bester: dict[str, int] = {}
    for formeln in KURZFORMELN.values():
        for art, liste in (
            (ART_FAEHIGKEITEN, formeln.faehigkeiten),
            (ART_IDENTITAET, formeln.identitaet),
        ):
            for formel in liste:
                eintrag = normalisieren(formel)
                grenze = tippfehler_toleranz(min(len(normal), len(eintrag)))
                if abs(len(normal) - len(eintrag)) > grenze:
                    continue
                abstand = levenshtein_abstand(normal, eintrag)
                if abstand <= grenze and abstand < bester.get(art, grenze + 1):
                    bester[art] = abstand
    if not bester:
        return None
    kleinster = min(bester.values())
    arten = [art for art, abstand in bester.items() if abstand == kleinster]
    return arten[0] if len(arten) == 1 else None


def kurzformel_erkennen(eingabe: str) -> str | None:
    """Die Art der Eingabe, wenn sie einer Formel entspricht, sonst None.

    Zuerst genau, ueber alle Arten und alle Sprachen; ein genauer Treffer geht
    vor. Danach, nur fuer die Fragen nach dem Assistenten, mit der Toleranz fuer
    Tippfehler aus `_nach_tippfehler`.

    Beruehrt nur die Eingabe und den Katalog, keinen Mandanteninhalt, und nimmt
    deshalb keine tenant_id.
    """
    normal = normalisieren(eingabe)
    if not normal:
        return None
    for formeln in KURZFORMELN.values():
        for art, liste in _arten(formeln):
            if normal in {normalisieren(formel) for formel in liste}:
                return art
    return _nach_tippfehler(normal)


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
