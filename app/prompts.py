"""System-Prompt und die Struktur, die das Modell zurueckgeben muss.

Die Struktur ist der Kern dieser Datei. Ein Modell, das nur Text zurueckgibt,
laesst sich nicht danach fragen, ob es die Antwort wirklich im Kontext gefunden
hat - es wuerde die Frage im selben Fliesstext beantworten, den es gerade
erfunden hat. Ein eigenes Feld dafuer trennt die Aussage von der Antwort.

Drei Dinge gehen an das Modell, und alle drei sind sprachabhaengig:

  Regelwerk           System-Prompt, `_REGELWERK` hier
  Rahmen              Userprompt, `_RAHMEN_DE` / `_RAHMEN_EN` hier
  Feldbeschreibungen  Strukturschema, Katalog in `app/texts.py`

Die dritte Zeile ist am 2026-09-24 dazugekommen. Sie fehlte, und weil sie das
Modell ueber `response_format` erreicht und kein Mensch sie liest, ist sie
niemandem aufgefallen - bis ein englischer Mandant 23 von 30 Fragen auf Deutsch
beantwortete. Wer hier eine vierte modellgerichtete Zeichenkette ergaenzt,
prueft sie gegen alle Sprachen, nicht nur gegen die eigene.
"""

from __future__ import annotations

from functools import cache
from typing import NamedTuple

from pydantic import BaseModel, Field, create_model

from app.search import SearchHit
from app.tenants import TenantConfig
from app.texts import schemabeschreibungen_fuer


# Bewusst OHNE Docstring, und das ist kein Versehen: Pydantic uebernimmt den
# Docstring einer Klasse als `description` der obersten Ebene in das JSON-Schema.
# Diese Klasse ist nur die Struktur - Namen, Typen, Standardwerte - und soll dem
# Modell keinen Text beitragen. Die Beschreibungen kommen sprachabhaengig aus
# `antwortmodell_fuer()`.
#
# Was die Struktur leistet, gehoert trotzdem aufgeschrieben, deshalb hier:
# `answerable` ist das zweite Eskalationstor (ADR-019). Es ist ein eigenes Feld
# und keine Formulierung im Antworttext, weil ein "das steht leider nicht in den
# Unterlagen" mitten in einem ansonsten erfundenen Absatz nicht auswertbar waere.
#
# Diese Klasse DIREKT an with_structured_output zu geben ist ein Fehler - das
# Modell bekaeme ein Schema ohne jede Feldbeschreibung. Deshalb verlangt
# build_llm() das Antwortmodell als Pflichtargument; es gibt keinen Standardwert,
# auf den man versehentlich zurueckfallen kann.
class GroundedAnswer(BaseModel):
    answerable: bool
    answer: str
    sources: list[str] = Field(default_factory=list)
    language: str = ""


@cache
def antwortmodell_fuer(language: str) -> type[GroundedAnswer]:
    """Die Antwortstruktur mit den Feldbeschreibungen dieser Sprache.

    Warum das eine Funktion ist und keine zwei Klassen: Die Beschreibungen
    stehen im Katalog in `app/texts.py`, zusammen mit der Liste der Sprachen.
    Zwei handgeschriebene Klassen waeren eine zweite Aufzaehlung der Sprachen -
    genau die Doppelfuehrung, die bei `languages` vor EN-1 schon einmal
    auseinandergelaufen ist.

    Der Cache haengt an der SPRACHE, nicht an einem Mandanten. Er beruehrt
    ADR-001 nicht: Das Ergebnis ist eine Klasse ohne jeden Mandantenbezug, kein
    Index und kein Retriever. Zwei Mandanten derselben Sprache teilen sie
    gefahrlos, weil sie nichts als Beschreibungstexte traegt.

    Der Klassenname bleibt "GroundedAnswer", damit der Name im Schema und in den
    Fehlermeldungen des Providers derselbe bleibt wie bisher.
    """
    beschreibung = schemabeschreibungen_fuer(language)
    return create_model(
        "GroundedAnswer",
        __base__=GroundedAnswer,
        __doc__=beschreibung.struktur,
        answerable=(bool, Field(description=beschreibung.answerable)),
        answer=(str, Field(description=beschreibung.answer)),
        sources=(list[str], Field(default_factory=list, description=beschreibung.sources)),
        language=(str, Field(default="", description=beschreibung.language)),
    )


_BASIS_REGELN_DE = """\
Du bist ein Support-Assistent fuer {display_name}.

Regeln, die ausnahmslos gelten:

1. Antworte AUSSCHLIESSLICH aus dem gelieferten Kontext. Dein eigenes Wissen ist
   hier unzulaessig, auch wenn du die Antwort sicher kennst. Steht eine Zahl,
   eine Frist oder eine Bedingung nicht im Kontext, dann existiert sie fuer
   diese Antwort nicht.
2. Nenne zu jeder Aussage die Quelldatei, aus der sie stammt.
3. Kannst du die Frage aus dem Kontext nicht vollstaendig beantworten, setze
   answerable auf false und lass answer leer. Rate nicht. Fuelle nicht auf.
   Formuliere keine Teilantwort, die vollstaendig aussieht.
4. Eine Frage, die der Kontext nur streift, ist NICHT beantwortbar. Beispiel:
   Wird eine Gebuehr erwaehnt, aber ihre Hoehe nicht genannt, ist die Frage nach
   der Hoehe nicht beantwortbar.
"""

_BASIS_REGELN_EN = """\
You are a support assistant for {display_name}.

Rules that apply without exception:

1. Answer EXCLUSIVELY from the context provided. Your own knowledge is not
   admissible here, even if you are certain of the answer. If a number, a
   deadline or a condition is not in the context, then for this answer it does
   not exist.
2. For every statement, name the source file it comes from.
3. If you cannot answer the question completely from the context, set
   answerable to false and leave answer empty. Do not guess. Do not fill gaps.
   Do not formulate a partial answer that looks complete.
4. A question the context merely touches on is NOT answerable. Example: if a
   fee is mentioned but its amount is not stated, the question about the amount
   is not answerable.
"""

_SPRACHE_FOLGT_FRAGE_DE = """\
5. Antworte in der Sprache der FRAGE, unabhaengig davon, in welcher Sprache die
   Quelldokumente verfasst sind. Eine deutsche Frage bekommt eine deutsche
   Antwort, auch wenn der Beleg englisch ist. Trage die verwendete Sprache in
   language ein.
"""

_SPRACHE_FOLGT_FRAGE_EN = """\
5. Answer in the language of the QUESTION, regardless of the language the
   source documents are written in. A German question gets a German answer,
   even if the evidence is in English. Record the language you used in
   language.
"""

_SPRACHE_VORGEGEBEN_DE = """\
5. Antworte in der Sprache mit dem Kuerzel "{response_language}", unabhaengig von
   der Sprache der Frage und der Quelldokumente. Trage "{response_language}" in
   language ein.
"""

_SPRACHE_VORGEGEBEN_EN = """\
5. Answer in the language with the code "{response_language}", regardless of the
   language of the question and of the source documents. Record
   "{response_language}" in language.
"""

_ZUSATZ_DE = "\nZusaetzlich fuer diesen Mandanten:\n"
_ZUSATZ_EN = "\nAdditionally for this tenant:\n"

# Der Rahmen des Userprompts - die Woerter, die den Kontextblock und die Frage
# beschriften.
#
# Bis zum 2026-09-23 waren diese vier Zeichenketten fest deutsch, fuer JEDEN
# Mandanten. Ein englischer Mandant bekam damit bei jeder einzelnen Anfrage
# einen deutsch gerahmten Prompt: "Kontext:", "[Quelle: ...]", "Frage:". Der
# System-Prompt war englisch, der Userprompt deutsch - und gemessen hat das
# Modell daraufhin drei von elf englischen Fragen auf Deutsch beantwortet.
#
# Das ist kein Schoenheitsfehler. Vor einem nordischen Empfaenger ist eine
# deutsche Antwort auf eine englische Frage genau der Fehler, gegen den die
# ganze EN-Reihe gebaut ist.
_RAHMEN_DE = ("Kontext", "Quelle", "Frage", "(kein Kontext gefunden)")
_RAHMEN_EN = ("Context", "Source", "Question", "(no context found)")


class _Regelwerk(NamedTuple):
    """Die vier Bausteine des System-Prompts in einer Sprache."""

    basis: str
    sprache_folgt_frage: str
    sprache_vorgegeben: str
    zusatz: str
    # (Kontext, Quelle, Frage, kein-Kontext) - der Rahmen des Userprompts.
    rahmen: tuple[str, str, str, str]


# Das Regelwerk steht hier und nicht im Textkatalog: Es richtet sich an das
# Modell, nicht an einen Menschen. Die Schluessel sind dieselben wie dort, und
# ein Test haelt beide Mengen deckungsgleich - eine Sprache mit Oberflaeche aber
# ohne Regelwerk wuerde sonst erst beim ersten Aufruf auffallen.
_REGELWERK: dict[str, _Regelwerk] = {
    "de": _Regelwerk(
        basis=_BASIS_REGELN_DE,
        sprache_folgt_frage=_SPRACHE_FOLGT_FRAGE_DE,
        sprache_vorgegeben=_SPRACHE_VORGEGEBEN_DE,
        zusatz=_ZUSATZ_DE,
        rahmen=_RAHMEN_DE,
    ),
    "en": _Regelwerk(
        basis=_BASIS_REGELN_EN,
        sprache_folgt_frage=_SPRACHE_FOLGT_FRAGE_EN,
        sprache_vorgegeben=_SPRACHE_VORGEGEBEN_EN,
        zusatz=_ZUSATZ_EN,
        rahmen=_RAHMEN_EN,
    ),
}


def build_system_prompt(tenant: TenantConfig, response_language: str | None = None) -> str:
    """Baut den System-Prompt fuer einen Mandanten.

    Zwei Sprachen, die nicht dasselbe meinen:

    `tenant.language` bestimmt, in welcher Sprache das REGELWERK formuliert ist.
    Ein englischer Mandant bekommt englische Regeln - sein `system_prompt_extra`
    ist ebenfalls englisch, und ein Prompt aus zwei Sprachen ist schlechter
    lesbar, fuer das Modell wie fuer den, der ihn spaeter prueft.

    `response_language` uebersteuert die ANTWORTSPRACHE. Ist es None, gilt Regel
    5 in der Fassung "Sprache der Frage" - unveraendert und unabhaengig davon,
    welche Sprache der Mandant traegt. Ein englischer Mandant antwortet auf eine
    deutsche Frage also deutsch, solange nichts anderes verlangt wird. Ist es
    gesetzt, wird in dieser Sprache geantwortet; der Goldsatz braucht das als
    deterministischen Test.
    """
    regeln = _REGELWERK[tenant.language]
    teile = [regeln.basis.format(display_name=tenant.display_name)]

    if response_language:
        teile.append(regeln.sprache_vorgegeben.format(response_language=response_language))
    else:
        teile.append(regeln.sprache_folgt_frage)

    if tenant.system_prompt_extra.strip():
        teile.append(regeln.zusatz)
        teile.append(tenant.system_prompt_extra.strip())

    return "\n".join(teile)


def build_user_prompt(question: str, hits: list[SearchHit], language: str = "de") -> str:
    """Baut den Kontextblock und die Frage, im Rahmen der Mandantensprache.

    Der Score steht bewusst NICHT im Kontext. Er ist eine interne Kennzahl; dem
    Modell hilft er nicht bei der Antwort und koennte es dazu verleiten, einen
    hohen Wert als Beleg fuer Abdeckung zu lesen. Genau diese Verwechslung -
    Aehnlichkeit statt Abdeckung - ist der Grund fuer das zweite Tor.

    `language` ist die Sprache des MANDANTEN, nicht die der Frage. Der Rahmen
    gehoert zum Prompt und nicht zur Antwort: Er beschriftet dem Modell, was
    Kontext ist und was Frage. Steht er in einer anderen Sprache als das
    Regelwerk darueber, bekommt das Modell einen zweisprachigen Prompt - und
    richtet sich bei der Antwortsprache messbar danach.

    Default "de", damit ein Aufruf ohne Angabe sich verhaelt wie bisher.
    """
    kontext_wort, quelle_wort, frage_wort, leer = _REGELWERK[language].rahmen

    abschnitte = [f"[{quelle_wort}: {hit.source_file}]\n{hit.text}" for hit in hits]
    kontext = "\n\n---\n\n".join(abschnitte) if abschnitte else leer

    return f"{kontext_wort}:\n\n{kontext}\n\n---\n\n{frage_wort}: {question}"
