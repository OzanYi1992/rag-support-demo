"""System-Prompt und die Struktur, die das Modell zurueckgeben muss.

Die Struktur ist der Kern dieser Datei. Ein Modell, das nur Text zurueckgibt,
laesst sich nicht danach fragen, ob es die Antwort wirklich im Kontext gefunden
hat - es wuerde die Frage im selben Fliesstext beantworten, den es gerade
erfunden hat. Ein eigenes Feld dafuer trennt die Aussage von der Antwort.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.search import SearchHit
from app.tenants import TenantConfig


class GroundedAnswer(BaseModel):
    """Was das Modell zurueckgeben muss.

    `answerable` ist das zweite Eskalationstor. Es ist bewusst ein eigenes Feld
    und keine Formulierung im Antworttext: Ein "das steht leider nicht in den
    Unterlagen" mitten in einem ansonsten erfundenen Absatz waere nicht
    auswertbar.
    """

    answerable: bool = Field(
        description=(
            "true, wenn die Frage aus dem gelieferten Kontext vollstaendig "
            "beantwortet werden kann. false, wenn der Kontext die Frage nicht "
            "oder nur teilweise abdeckt."
        )
    )
    answer: str = Field(
        description=(
            "Die Antwort, ausschliesslich aus dem Kontext. Leer lassen, wenn answerable false ist."
        )
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "Dateinamen aus dem Kontext, auf denen die Antwort beruht. Nur "
            "Dateinamen, die im Kontext vorkommen."
        ),
    )
    language: str = Field(
        default="",
        description=("Sprache der Antwort als ISO-639-1-Kuerzel, etwa 'de' oder 'en'."),
    )


_BASIS_REGELN = """\
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

_SPRACHE_FOLGT_FRAGE = """\
5. Antworte in der Sprache der FRAGE, unabhaengig davon, in welcher Sprache die
   Quelldokumente verfasst sind. Eine deutsche Frage bekommt eine deutsche
   Antwort, auch wenn der Beleg englisch ist. Trage die verwendete Sprache in
   language ein.
"""

_SPRACHE_VORGEGEBEN = """\
5. Antworte in der Sprache mit dem Kuerzel "{response_language}", unabhaengig von
   der Sprache der Frage und der Quelldokumente. Trage "{response_language}" in
   language ein.
"""


def build_system_prompt(tenant: TenantConfig, response_language: str | None = None) -> str:
    """Baut den System-Prompt fuer einen Mandanten.

    `response_language` uebersteuert die Sprachregel. Ist es None, gilt Regel 5
    in der Fassung "Sprache der Frage". Ist es gesetzt, wird in dieser Sprache
    geantwortet - der Fall, den der Goldsatz in Phase 5 als deterministischen
    Test braucht.
    """
    teile = [_BASIS_REGELN.format(display_name=tenant.display_name)]

    if response_language:
        teile.append(_SPRACHE_VORGEGEBEN.format(response_language=response_language))
    else:
        teile.append(_SPRACHE_FOLGT_FRAGE)

    if tenant.system_prompt_extra.strip():
        teile.append("\nZusaetzlich fuer diesen Mandanten:\n")
        teile.append(tenant.system_prompt_extra.strip())

    return "\n".join(teile)


def build_user_prompt(question: str, hits: list[SearchHit]) -> str:
    """Baut den Kontextblock und die Frage.

    Der Score steht bewusst NICHT im Kontext. Er ist eine interne Kennzahl; dem
    Modell hilft er nicht bei der Antwort und koennte es dazu verleiten, einen
    hohen Wert als Beleg fuer Abdeckung zu lesen. Genau diese Verwechslung -
    Aehnlichkeit statt Abdeckung - ist der Grund fuer das zweite Tor.
    """
    abschnitte = [f"[Quelle: {hit.source_file}]\n{hit.text}" for hit in hits]
    kontext = "\n\n---\n\n".join(abschnitte) if abschnitte else "(kein Kontext gefunden)"

    return f"Kontext:\n\n{kontext}\n\n---\n\nFrage: {question}"
