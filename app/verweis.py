"""Verweise auf ein Dokument ueber den Assistenten aus dem Antworttext nehmen.

Jeder Demomandant hat ein Dokument, das den Assistenten selbst beschreibt: dass
ein KI-System antwortet, nur aus den hinterlegten Unterlagen, und zu welchen
Themen. Es steht im Index wie jedes andere, damit eine Frage wie "Wobei koennen
Sie mir helfen?" einen Beleg hat. Als Quelle ausgewiesen wird es nie.

Die Bedingung dafuer: Bei einer Frage, fuer die kein Fachdokument noetig ist,
steht weder unter noch in der Antwort ein Dokument. Unter der Antwort sorgt
`_quellen_verdichten` in app/rag.py dafuer. IN der Antwort sorgt diese Datei
dafuer, denn das Regelwerk laesst das Modell die Quelldatei auch in den Text
schreiben, in wechselnder Form:

    [Source: x.md]   [x.md]   Source: x.md.   Sources: a.md; b.md.
    according to x.md        Source: x.md says that ...

Die Regeln, in dieser Reihenfolge:

1. Nur bei nicht eskalierten Antworten, und nur fuer die Namen aus
   `assistant_documents` dieses Mandanten, mit und ohne `.md`, ohne Ruecksicht
   auf Gross- und Kleinschreibung.
2. Eine Angabe, die nur Assistentendokumente nennt, faellt ganz weg, samt
   Klammern, Bezeichnung und dem Leerraum davor: in eckigen oder runden
   Klammern, oder als "Source: x.md." am Satz- oder Textende, mit Source,
   Sources, Quelle oder Quellen.
3. Nennt eine Angabe auch andere Dateien, faellt nur der Name samt seinem
   Trennzeichen weg. Die anderen bleiben Zeichen fuer Zeichen.
4. Steht der Name danach noch im Fliesstext, ersetzt ihn der Titel des
   Dokuments aus dem Katalog, in der Antwortsprache.
5. Danach steht der Name in keiner Schreibweise mehr im Text, und es bleibt
   keine leere Klammer, kein doppelter Leerraum und kein Satzzeichen ohne Wort
   davor. Waere der Text danach leer, bleibt er unveraendert.
6. Jeder andere Text bleibt Zeichen fuer Zeichen gleich. Zweimal angewendet
   ergibt dasselbe wie einmal.

Das ist ein Eingriff in Modelltext nach den Toren. Er nimmt nur Verweise
heraus und setzt keinen Inhalt ein; das Tor hat ueber die Antwort schon
entschieden, bevor diese Datei sie sieht.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.tenants import TenantConfig
from app.texts import texte_fuer

# Die Bezeichnungen vor einer Angabe, in beiden Sprachen.
_BEZEICHNUNG = r"(?:sources?|quellen?)\s*:\s*"

# Ein Dateiname, wie das Modell ihn in einer Angabe nennt: ein Wort mit Endung.
# "[Note]" oder "Source: the FAQ" sind damit keine Angabe und bleiben stehen.
_DATEI = r"[\w][\w.\-]*\.\w+"

# Trennzeichen zwischen zwei Namen einer Angabe.
_TRENNER = r"\s*[;,]\s*"

# Satzzeichen, vor denen kein Leerraum stehen bleibt, und nach denen kein
# zweites stehen darf.
_SATZZEICHEN = ".,;:!?"


def _ohne_endung(name: str) -> str:
    return name[:-3] if name.lower().endswith(".md") else name


def _namensmuster(namen: Iterable[str]) -> str:
    """Ein Muster, das jeden der Namen trifft, mit und ohne .md.

    Die Raender verhindern Treffer in anderen Namen: "about-this-assistant.md"
    trifft weder in "my-about-this-assistant.md" noch in
    "about-this-assistant-v2.md" noch in "about-this-assistant.mdx".
    """
    stamm = sorted({_ohne_endung(n) for n in namen if _ohne_endung(n)}, key=len, reverse=True)
    if not stamm:
        return r"(?!x)x"
    auswahl = "|".join(re.escape(s) for s in stamm)
    return rf"(?<![\w./\-])(?:{auswahl})(?:\.md)?(?![\w\-])(?!\.\w)"


def ist_assistentendokument(name: str, namen: Iterable[str]) -> bool:
    """Ob ein Dateiname einer der Namen ist, mit und ohne .md, ohne Gross- und
    Kleinschreibung."""
    return re.fullmatch(_namensmuster(namen), name.strip(), flags=re.IGNORECASE) is not None


def _verbinden(links: str, rechts: str) -> str:
    """Fuegt die beiden Seiten um eine entfernte Stelle zusammen.

    Behaelt hoechstens einen der beiden Leerraeume, einen mit Zeilenumbruch
    bevorzugt, und keinen vor einem Satzzeichen oder nach einer oeffnenden
    Klammer. Folgt auf ein Satzzeichen ein zweites, faellt das zweite weg: Es
    gehoerte zur entfernten Angabe.
    """
    links_rumpf = links.rstrip()
    rechts_rumpf = rechts.lstrip()
    leer_links = links[len(links_rumpf) :]
    leer_rechts = rechts[: len(rechts) - len(rechts_rumpf)]

    if rechts_rumpf[:1] and rechts_rumpf[0] in _SATZZEICHEN:
        if not links_rumpf or links_rumpf[-1] in _SATZZEICHEN:
            rechts_rumpf = rechts_rumpf[1:].lstrip()
            leer_rechts = " " if rechts_rumpf else ""
        else:
            return links_rumpf + rechts_rumpf
    if not links_rumpf:
        return rechts_rumpf
    if not rechts_rumpf:
        return links_rumpf
    if links_rumpf[-1] in "([":
        return links_rumpf + rechts_rumpf
    if "\n" in leer_links:
        zwischen = leer_links
    elif "\n" in leer_rechts:
        zwischen = leer_rechts
    else:
        zwischen = leer_links or leer_rechts
    return links_rumpf + zwischen + rechts_rumpf


def _liste_kuerzen(liste: str, name_muster: re.Pattern[str]) -> tuple[str | None, int]:
    """Nimmt die Assistentendokumente aus einer Liste von Namen.

    Gibt (neue Liste, Zahl der entfernten Namen) zurueck. None heisst: Die
    Liste nannte nur Assistentendokumente, die ganze Angabe faellt weg. Die
    uebrigen Namen und ihre Trennzeichen bleiben Zeichen fuer Zeichen.
    """
    # Abwechselnd Name, Trenner, Name, ... - die Trenner bleiben erhalten.
    teile = re.split(f"({_TRENNER})", liste)
    namen = teile[0::2]
    trenner = teile[1::2]
    weg = [name_muster.fullmatch(n) is not None for n in namen]
    if not any(weg):
        return liste, 0
    if all(weg):
        return None, len(namen)

    behalten: list[str] = []
    for i, name in enumerate(namen):
        if weg[i]:
            continue
        if behalten:
            # Der Trenner vor diesem Namen. Steht ein entfernter Name davor,
            # nimmt dessen Trenner den Platz ein; der Wortlaut ist derselbe Typ.
            behalten.append(trenner[i - 1])
        behalten.append(name)
    return "".join(behalten), sum(weg)


def assistentenverweise_entfernen(
    tenant_id: str, tenant: TenantConfig, text: str, antwortsprache: str
) -> tuple[str, int]:
    """Nimmt Verweise auf die Assistentendokumente dieses Mandanten aus dem Text.

    Gibt (Text, Zahl der entfernten oder ersetzten Verweise) zurueck. Nur fuer
    nicht eskalierte Antworten aufzurufen; eine Eskalation traegt den Text des
    Mandanten und keinen Modelltext.

    `tenant_id` an erster Stelle: Der Antworttext ist Mandanteninhalt, aus
    seinen Dokumenten erzeugt. `tenant` muss derselbe Mandant sein, sonst ist
    das ein Fehler im Aufrufer.

    `antwortsprache` ist das Feld language der Antwort. Ist es weder de noch
    en, gilt die Sprache des Mandanten fuer den Titel.
    """
    if tenant.slug != tenant_id:
        raise ValueError("tenant passt nicht zur tenant_id.")
    if not tenant.assistant_documents or not text:
        return text, 0

    muster = _namensmuster(tenant.assistant_documents)
    name_muster = re.compile(muster, re.IGNORECASE)
    eintrag = rf"(?:{muster}|{_DATEI})"
    liste = rf"{eintrag}(?:{_TRENNER}{eintrag})*"

    neu = text
    anzahl = 0

    # Regel 2 und 3, in Klammern: [Source: x.md], [x.md], (Quelle: x.md).
    klammer = re.compile(
        rf"(?P<auf>[\[(])(?P<bez>{_BEZEICHNUNG})?(?P<liste>{liste})\s*(?P<zu>[\])])",
        re.IGNORECASE,
    )
    neu, n = _angaben_bearbeiten(neu, klammer, name_muster, klammer=True)
    anzahl += n

    # Regel 2 und 3, ohne Klammern, am Satzende oder am Textende:
    # "Source: x.md." und "Sources: a.md; b.md."
    satz = re.compile(
        rf"(?<![\w\[(])(?P<bez>{_BEZEICHNUNG})(?P<liste>{liste})(?P<punkt>\.(?=\s|$)|(?=\s*$))",
        re.IGNORECASE,
    )
    neu, n = _angaben_bearbeiten(neu, satz, name_muster, klammer=False)
    anzahl += n

    # Regel 4: was noch steht, ist Fliesstext.
    sprache = antwortsprache.strip().lower()
    if sprache not in ("de", "en"):
        sprache = tenant.language
    titel = texte_fuer(sprache).assistentendokument_titel
    neu, n = name_muster.subn(lambda _: titel, neu)
    anzahl += n

    # Regel 5, letzter Satz.
    if not neu.strip():
        return text, 0
    return neu, anzahl


def _angaben_bearbeiten(
    text: str, angabe: re.Pattern[str], name_muster: re.Pattern[str], *, klammer: bool
) -> tuple[str, int]:
    """Wendet Regel 2 und 3 auf jede Angabe an, von hinten nach vorn.

    Von hinten, damit die Positionen der vorderen Treffer gueltig bleiben.
    """
    treffer = list(angabe.finditer(text))
    anzahl = 0
    for t in reversed(treffer):
        if klammer and (t["auf"] == "[") != (t["zu"] == "]"):
            continue
        gekuerzt, n = _liste_kuerzen(t["liste"], name_muster)
        if n == 0:
            continue
        anzahl += n
        if gekuerzt is None:
            text = _verbinden(text[: t.start()], text[t.end() :])
        else:
            start, ende = t.span("liste")
            text = text[:start] + gekuerzt + text[ende:]
    return text, anzahl
