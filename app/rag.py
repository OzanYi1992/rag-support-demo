"""Antwortgenerierung mit zwei Eskalationstoren.

Der Ablauf ist bewusst so geordnet, dass das erste Tor VOR dem LLM-Aufruf faellt.
Eine Eskalation, die erst nach dem Aufruf entschieden wird, kostet Geld und
Latenz - und vor allem waere ADR-003 dann formal erfuellt und inhaltlich
verletzt, weil das Modell den Kontext bereits gesehen hat.

Zwei Tore, drei Gruende (ADR-019):

  Retrieval    vor dem Aufruf, Strategie aus app/escalation.py
  Groundedness nach dem Aufruf, answerable-Feld des Modells
  Parsefehler  wenn die Struktur nicht lesbar ist

Alle drei sind in der Answer unterscheidbar, damit Phase 5 messen kann, welches
Tor greift - nicht nur, DASS eskaliert wurde.
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.embeddings import E5Embeddings
from app.escalation import EscalationStrategy, build_escalation_strategy
from app.llm import LlmClient, build_llm
from app.prompts import antwortmodell_fuer, build_system_prompt, build_user_prompt
from app.search import SearchHit, search_tenant
from app.tenants import TenantConfig, load_tenant

REASON_NOT_GROUNDED = "not_grounded"
REASON_UNPARSEABLE = "generation_unparseable"


class Answer(BaseModel):
    """Antwort auf eine Frage, mit allem, was Telemetrie und Eval brauchen."""

    text: str

    # Eine Datei, ein Eintrag. Verdichtet in `_quellen_verdichten()`, nicht in
    # der Oberflaeche: Jeder Verbraucher soll dasselbe sehen.
    sources: list[str]

    # Der beste Score der Chunks JE QUELLDATEI, gleiche Reihenfolge wie
    # `sources`. None, wenn das Modell eine Datei nennt, die im Kontext nicht
    # vorkam - dann gibt es keinen Score, und es waere falsch, einen zu zeigen.
    #
    # Warum das Feld ueberhaupt existiert: Die Oberflaeche hat bis zum
    # 2026-09-24 `sources[i]` mit `retrieval_scores[i]` gepaart. Die beiden
    # Listen bedeuten Verschiedenes - die eine zaehlt Dateien, die das Modell
    # zitiert, die andere Chunks in der Trefferreihenfolge. Die Zahl neben einem
    # Dateinamen war damit der Score des i-ten CHUNKS, nicht der dieser Datei.
    source_scores: list[float | None]

    escalated: bool
    escalation_reason: str | None
    escalation_metric: float | None

    # Alle Scores der Trefferliste, nicht nur der beste. Phase 5 braucht die
    # Verteilung, um ueber das Mass zu entscheiden (OP-018).
    retrieval_scores: list[float]

    latency_ms_retrieval: int
    # None, wenn retrievalseitig eskaliert wurde. Das ist der Beleg dafuer, dass
    # kein Modell aufgerufen wurde - nicht nur eine Behauptung im Flag.
    latency_ms_generation: int | None

    prompt_tokens: int | None
    completion_tokens: int | None
    model: str | None

    # Tatsaechliche Antwortsprache, nicht der Eingabewert von response_language.
    # Phase 6 will das als Dimension.
    lang: str


def _eskaliert(
    tenant: TenantConfig,
    reason: str,
    metric: float | None,
    scores: list[float],
    latency_retrieval: int,
    latency_generation: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    model: str | None = None,
    lang: str = "",
) -> Answer:
    """Baut die Eskalationsantwort aus der Nachricht des Mandanten.

    Der Modelltext wird hier NIE ausgeliefert. Auch dann nicht, wenn das Modell
    etwas geschrieben hat - bei answerable=false ist genau dieser Text der, den
    ADR-003 verhindern soll.
    """
    return Answer(
        text=tenant.escalation_message.strip(),
        sources=[],
        source_scores=[],
        escalated=True,
        escalation_reason=reason,
        escalation_metric=metric,
        retrieval_scores=scores,
        latency_ms_retrieval=latency_retrieval,
        latency_ms_generation=latency_generation,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
        lang=lang,
    )


def _quellen_verdichten(
    genannt: list[str], treffer: list[SearchHit]
) -> tuple[list[str], list[float | None]]:
    """Eine Datei, ein Eintrag - mit dem besten Score der Chunks dieser Datei.

    Zwei Dinge auf einmal, weil sie zusammengehoeren:

    1. **Entdoppeln.** Das Modell nennt eine Datei so oft, wie es Chunks daraus
       benutzt hat. Fuenf Chunks aus derselben Datei ergaben fuenf Eintraege in
       der Quellenliste, und die Oberflaeche zeigte sie fuenfmal.
    2. **Den richtigen Score zuordnen.** Der Score gehoert zu einem Chunk, die
       Anzeige zu einer Datei. Der beste Chunk einer Datei ist der, der sie in
       die Trefferliste gebracht hat - also der richtige Wert fuer die Anzeige.

    Die Reihenfolge der ersten Nennung bleibt erhalten. Sie ist die des Modells
    und damit die seiner Begruendung; nach Score zu sortieren wuerde eine
    Rangfolge behaupten, die das Modell nicht gemeint hat.

    Nennt das Modell eine Datei, die im Kontext nicht vorkam, bleibt der Score
    None. Der Eintrag wird NICHT entfernt: Eine erfundene Quelle ist ein Befund
    und gehoert sichtbar, nicht stillschweigend weggeraeumt.
    """
    beste: dict[str, float] = {}
    for hit in treffer:
        if hit.source_file not in beste or hit.score > beste[hit.source_file]:
            beste[hit.source_file] = hit.score

    quellen: list[str] = []
    scores: list[float | None] = []
    for name in genannt:
        saubere = name.strip()
        if not saubere or saubere in quellen:
            continue
        quellen.append(saubere)
        scores.append(beste.get(saubere))
    return quellen, scores


def answer(
    tenant_id: str,
    question: str,
    response_language: str | None = None,
    settings: Settings | None = None,
    strategy: EscalationStrategy | None = None,
    llm: LlmClient | None = None,
    embeddings: E5Embeddings | None = None,
) -> Answer:
    """Beantwortet eine Frage fuer genau einen Mandanten.

    `tenant_id` ist Pflichtparameter an erster Position, ohne Default und ohne
    Optional (ADR-001). Es gibt kein Index-, Retriever- oder Modellobjekt auf
    Modulebene; alles wird pro Aufruf aufgeloest.

    `response_language`:
      None      Prompt-Regel gilt, geantwortet wird in der Sprache der Frage.
      gesetzt   Es wird in DIESER Sprache geantwortet, unabhaengig von der
                Sprache der Frage. Der Goldsatz in Phase 5 braucht das als
                deterministischen Test - haengt die Antwortsprache an der
                Spracherkennung des Modells, ist der Test wackelig.

    `strategy`, `llm` und `embeddings` sind injizierbar, damit Tests ohne Netz
    und ohne Schluessel auskommen. Keines davon hat einen Mandantenbezug.
    """
    aktive_settings = settings if settings is not None else get_settings()
    tenant = load_tenant(tenant_id, aktive_settings.tenants_dir)

    # --- Retrieval ---------------------------------------------------------
    start = time.perf_counter()
    treffer: list[SearchHit] = search_tenant(
        tenant_id,
        question,
        k=tenant.retrieval_top_k or aktive_settings.retrieval_top_k,
        settings=aktive_settings,
        embeddings=embeddings,
    )
    latenz_retrieval = int((time.perf_counter() - start) * 1000)
    scores = [t.score for t in treffer]

    # --- Erstes Tor: Retrieval --------------------------------------------
    aktive_strategie = (
        strategy if strategy is not None else build_escalation_strategy(aktive_settings)
    )
    entscheidung = aktive_strategie.should_escalate(treffer)
    if entscheidung.escalate:
        # Sofort zurueck. KEIN LLM-Aufruf - latency_ms_generation und die
        # Tokenzahlen bleiben None und belegen das.
        return _eskaliert(
            tenant,
            reason=entscheidung.reason or "retrieval",
            metric=entscheidung.metric_value,
            scores=scores,
            latency_retrieval=latenz_retrieval,
        )

    # --- Generierung -------------------------------------------------------
    aktives_llm = (
        llm
        if llm is not None
        else build_llm(
            aktive_settings,
            tenant.model_override,
            # Die Sprache des MANDANTEN, nicht die der Frage: Das Schema gehoert
            # zum Prompt und beschreibt dem Modell die Struktur. Ein englischer
            # Mandant mit deutsch beschriebenen Feldern antwortet messbar
            # deutsch, auch auf eine englische Frage (P-024, zweite Ebene).
            answer_model=antwortmodell_fuer(tenant.language),
        )
    )
    system_prompt = build_system_prompt(tenant, response_language)
    # response_language geht auch hier hinein: Die Knappheitsforderung am Ende
    # nennt die Sprache, und sie muss dasselbe sagen wie Regel 5 im
    # System-Prompt. Zwei widerspruechliche Angaben in einem Prompt waeren
    # schlimmer als keine.
    user_prompt = build_user_prompt(question, treffer, tenant.language, response_language)

    start = time.perf_counter()
    ergebnis = aktives_llm.generate(system_prompt, user_prompt)
    latenz_generierung = int((time.perf_counter() - start) * 1000)

    # --- Dritter Weg: Struktur nicht lesbar --------------------------------
    if ergebnis.parsed is None:
        return _eskaliert(
            tenant,
            reason=REASON_UNPARSEABLE,
            metric=entscheidung.metric_value,
            scores=scores,
            latency_retrieval=latenz_retrieval,
            latency_generation=latenz_generierung,
            prompt_tokens=ergebnis.prompt_tokens,
            completion_tokens=ergebnis.completion_tokens,
            model=ergebnis.model,
        )

    # --- Zweites Tor: Groundedness ----------------------------------------
    if not ergebnis.parsed.answerable:
        return _eskaliert(
            tenant,
            reason=REASON_NOT_GROUNDED,
            metric=entscheidung.metric_value,
            scores=scores,
            latency_retrieval=latenz_retrieval,
            latency_generation=latenz_generierung,
            prompt_tokens=ergebnis.prompt_tokens,
            completion_tokens=ergebnis.completion_tokens,
            model=ergebnis.model,
            lang=ergebnis.parsed.language,
        )

    quellen, quellscores = _quellen_verdichten(ergebnis.parsed.sources, treffer)
    return Answer(
        text=ergebnis.parsed.answer,
        sources=quellen,
        source_scores=quellscores,
        escalated=False,
        escalation_reason=None,
        escalation_metric=entscheidung.metric_value,
        retrieval_scores=scores,
        latency_ms_retrieval=latenz_retrieval,
        latency_ms_generation=latenz_generierung,
        prompt_tokens=ergebnis.prompt_tokens,
        completion_tokens=ergebnis.completion_tokens,
        model=ergebnis.model,
        lang=ergebnis.parsed.language,
    )
