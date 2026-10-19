"""Verweise auf die Dokumente des Mandanten aus dem Antworttext nehmen.

Das Regelwerk laesst das Modell zu jeder Aussage die Quelldatei nennen, und es
schreibt sie in wechselnder Form in den Text:

    [x.md]   [Source: x.md]   [a.md; b.md]   (Quelle: x.md)
    Source: x.md.   Sources: a.md, b.md and c.md   Source: `x.md`.
    according to x.md        Source: x.md says that ...

Unter der Antwort stehen die Quellen ohnehin, zum Aufklappen. Im Text stehen
sie doppelt, und ein Dateiname liest sich dort wie ein Rest der Technik. Fuer
das Dokument ueber den Assistenten gilt das erst recht: Es erscheint nie als
Quelle, weder unter noch in der Antwort; unter der Antwort sorgt
`_quellen_verdichten` in app/rag.py dafuer.

Die Regeln, in dieser Reihenfolge:

1. Nur bei nicht eskalierten Antworten, und nur fuer die Dokumente DIESES
   Mandanten: die Dateien in seinem Dokumentordner und die Namen aus
   `assistant_documents`. Ohne Ruecksicht auf Gross- und Kleinschreibung, mit
   und ohne Backticks. Ohne .md nur bei den Assistentendokumenten: Der Stamm
   eines Fachdokuments ist oft ein gewoehnliches Wort ("widerruf", "faq").
2. Eine Angabe, die nur Dokumente des Mandanten nennt, faellt ganz weg, samt
   Klammern, Bezeichnung und dem Leerraum davor: in eckigen oder runden
   Klammern, oder als "Source: x.md." am Satz- oder Textende, mit Source,
   Sources, Quelle oder Quellen. Mehrere Namen trennt ein Semikolon, ein Komma,
   "and" oder "und".
3. Nennt eine Angabe auch andere Dateien, faellt nur der Name samt seinem
   Trennzeichen weg. Die anderen bleiben Zeichen fuer Zeichen: Eine erfundene
   Quelle ist ein Befund und bleibt sichtbar.
4. Steht der Name danach noch im Fliesstext, ersetzt ihn der Titel des
   Dokuments, in den Anfuehrungszeichen der Antwortsprache. Titel ist die erste
   Zeile der Datei, die mit "# " beginnt, ohne diese Zeichen, sonst der Name
   ohne .md. Ein Assistentendokument bekommt seinen Titel aus dem Katalog.
5. Danach steht kein Name eines Dokuments des Mandanten mehr im Text, und es
   bleibt keine leere Klammer, kein doppelter Leerraum und kein Satzzeichen
   ohne Wort davor. Waere der Text danach leer, bleibt er unveraendert.
6. Jeder andere Text bleibt Zeichen fuer Zeichen gleich. Zweimal angewendet
   ergibt dasselbe wie einmal.

Das ist ein Eingriff in Modelltext nach den Toren. Er nimmt nur Verweise
heraus und setzt keinen Inhalt ein; das Tor hat ueber die Antwort schon
entschieden, bevor diese Datei sie sieht.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.tenants import TenantConfig, dokumentordner
from app.texts import texte_fuer

# Welche Dateien ein Dokument sind: dieselben, die der Ingest liest.
DOKUMENTMUSTER = "*.md"

# Die Bezeichnungen vor einer Angabe, in beiden Sprachen.
_BEZEICHNUNG = r"(?:sources?|quellen?)\s*:\s*"

# Ein Dateiname, wie das Modell ihn in einer Angabe nennt: ein Wort mit Endung,
# auch in Backticks. "[Note]" oder "Source: the FAQ" sind damit keine Angabe
# und bleiben stehen.
_DATEI = r"(?:`[\w][\w.\-]*\.\w+`|[\w][\w.\-]*\.\w+)"

# Trennzeichen zwischen zwei Namen einer Angabe: Semikolon, Komma, "and" oder
# "und", auch ", and".
_TRENNER = r"(?:\s*[;,]\s*(?:(?:and|und)\s+)?|\s+(?:and|und)\s+)"

# Satzzeichen, vor denen kein Leerraum stehen bleibt, und nach denen kein
# zweites stehen darf.
_SATZZEICHEN = ".,;:!?"


@dataclass(frozen=True)
class Mandantendokumente:
    """Die Dokumente eines Mandanten mit ihren Titeln.

    Traegt die `tenant_id`, aus deren Ordner sie gelesen wurden. Die Entfernung
    prueft sie, damit die Liste eines anderen Mandanten nicht still in einen
    Antworttext geraet.
    """

    tenant_id: str
    # Dateiname -> Titel, ohne Anfuehrungszeichen.
    titel: dict[str, str]


def _ohne_endung(name: str) -> str:
    return name[:-3] if name.lower().endswith(".md") else name


def _titel_aus(datei: Path) -> str:
    """Die erste Zeile, die mit "# " beginnt, ohne diese Zeichen; sonst der
    Name ohne .md."""
    try:
        zeilen = datei.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        zeilen = []
    for zeile in zeilen:
        if zeile.startswith("# "):
            titel = zeile[2:].strip()
            if titel:
                return titel
            break
    return _ohne_endung(datei.name)


def mandantendokumente(tenant_id: str, tenants_dir: Path) -> Mandantendokumente:
    """Liest Namen und Titel der Dokumente genau dieses Mandanten.

    `tenant_id` an erster Stelle: Der Ordner wird aus ihr aufgeloest, ueber
    `dokumentordner`, mit Pruefung des Slugs. Kein Cache: Es sind wenige kleine
    Dateien, gelesen je beantworteter Frage.
    """
    ordner = dokumentordner(tenant_id, tenants_dir)
    titel: dict[str, str] = {}
    if ordner.is_dir():
        for datei in sorted(ordner.glob(DOKUMENTMUSTER)):
            if datei.is_file():
                titel[datei.name] = _titel_aus(datei)
    return Mandantendokumente(tenant_id=tenant_id, titel=titel)


def _namensmuster(namen: Iterable[str], nur_mit_endung: Iterable[str] = ()) -> str:
    """Ein Muster, das jeden der Namen trifft, auch in Backticks.

    `namen` treffen mit und ohne .md, `nur_mit_endung` nur mit .md. Die Raender
    verhindern Treffer in anderen Namen: "about-this-assistant.md" trifft weder
    in "my-about-this-assistant.md" noch in "about-this-assistant-v2.md" noch in
    "about-this-assistant.mdx".
    """
    stamm = {_ohne_endung(n) for n in namen if _ohne_endung(n)}
    voll = {_ohne_endung(n) for n in nur_mit_endung if _ohne_endung(n)} - stamm
    teile = [(s, rf"{re.escape(s)}(?:\.md)?") for s in stamm]
    teile += [(s, rf"{re.escape(s)}\.md") for s in voll]
    if not teile:
        return r"(?!x)x"
    kern = "|".join(m for _, m in sorted(teile, key=lambda t: (-len(t[0]), t[0])))
    return rf"(?:`(?:{kern})`|(?<![\w./\-])(?:{kern})(?![\w\-])(?!\.\w))"


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
    """Nimmt die Dokumente des Mandanten aus einer Liste von Namen.

    Gibt (neue Liste, Zahl der entfernten Namen) zurueck. None heisst: Die
    Liste nannte nur Dokumente des Mandanten, die ganze Angabe faellt weg. Die
    uebrigen Namen und ihre Trennzeichen bleiben Zeichen fuer Zeichen.
    """
    # Abwechselnd Name, Trenner, Name, ... - die Trenner bleiben erhalten.
    teile = re.split(f"({_TRENNER})", liste, flags=re.IGNORECASE)
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


def quellverweise_entfernen(
    tenant_id: str,
    tenant: TenantConfig,
    text: str,
    antwortsprache: str,
    dokumente: Mandantendokumente,
) -> tuple[str, int]:
    """Nimmt Verweise auf die Dokumente dieses Mandanten aus dem Text.

    Gibt (Text, Zahl der entfernten oder ersetzten Namen) zurueck. Nur fuer
    nicht eskalierte Antworten aufzurufen; eine Eskalation traegt den Text des
    Mandanten und keinen Modelltext.

    `tenant_id` an erster Stelle: Der Antworttext ist Mandanteninhalt, aus
    seinen Dokumenten erzeugt. `tenant` und `dokumente` muessen derselbe
    Mandant sein, sonst ist das ein Fehler im Aufrufer.

    `antwortsprache` ist das Feld language der Antwort. Ist es weder de noch
    en, gilt die Sprache des Mandanten fuer Titel und Anfuehrungszeichen.
    """
    if tenant.slug != tenant_id or dokumente.tenant_id != tenant_id:
        raise ValueError("tenant oder Dokumentliste passt nicht zur tenant_id.")
    assistent = list(tenant.assistant_documents)
    fach = [n for n in dokumente.titel if not ist_assistentendokument(n, assistent)]
    if not (assistent or fach) or not text:
        return text, 0

    muster = _namensmuster(assistent, fach)
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
    # "Source: x.md." und "Sources: a.md, b.md and c.md."
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
    texte = texte_fuer(sprache)
    titel_je_stamm = {_ohne_endung(d).lower(): t for d, t in dokumente.titel.items()}

    def titel_fuer(treffer: re.Match[str]) -> str:
        name = treffer.group(0).strip("`")
        if ist_assistentendokument(name, assistent):
            return texte.assistentendokument_titel
        stamm = _ohne_endung(name)
        return texte.dokumenttitel_im_text.format(titel=titel_je_stamm.get(stamm.lower(), stamm))

    neu, n = name_muster.subn(titel_fuer, neu)
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
