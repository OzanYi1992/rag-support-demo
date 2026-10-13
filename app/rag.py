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
from dataclasses import dataclass

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.embeddings import E5Embeddings
from app.escalation import EscalationStrategy, build_escalation_strategy
from app.llm import (
    LlmAnfrageAbgelehnt,
    LlmClient,
    LlmNichtErreichbar,
    Umformer,
    build_llm,
    build_umformer,
)
from app.prompts import (
    Runde,
    antwortmodell_fuer,
    build_system_prompt,
    build_user_prompt,
    umformungsmodell_fuer,
    umformungsprompts_bauen,
)
from app.search import SearchHit, search_tenant
from app.tenants import TenantConfig, load_tenant
from app.texts import (
    ART_ABSCHIED,
    ART_BEGRUESSUNG,
    ART_DANK,
    ART_FAEHIGKEITEN,
    ART_IDENTITAET,
    kurzformel_erkennen,
    kurzformel_genau,
    texte_fuer,
)
from app.verweis import assistentenverweise_entfernen, ist_assistentendokument

REASON_NOT_GROUNDED = "not_grounded"
REASON_UNPARSEABLE = "generation_unparseable"

# Die Art der Antwort. "rag" ist der Weg ueber Retrieval und Modell, auch wenn
# er eskaliert. Die anderen kommen ohne beides aus dem Katalog.
ART_RAG = "rag"

# Was mit einer Folgefrage geschah, als fester Code fuer Antwort und Ereignis.
UMFORMUNG_KEINE = "keine"  # kein Verlauf, oder eine genaue Formel im Gespraech
UMFORMUNG_UMGEFORMT = "umgeformt"
UMFORMUNG_VERWORFEN = "verworfen"  # zurueck auf die getippte Frage

# Warum eine Umformung verworfen wurde.
VERWORFEN_PARSEFEHLER = "parsefehler"
VERWORFEN_LEER = "leer"
VERWORFEN_ZU_LANG = "zu_lang"
VERWORFEN_ABGELEHNT = "abgelehnt"

# Laenger darf eine eigenstaendige Frage nicht sein; dieselbe Grenze wie fuer
# die getippte Frage.
UMFORMUNG_MAX_ZEICHEN = 2000


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

    # "rag", oder "greeting", "thanks", "goodbye", "capabilities" und
    # "identity" fuer eine Antwort ohne Retrieval und ohne Modell. Die
    # Oberflaeche braucht das Feld fuer die Antwort selbst nicht: Ohne Quellen
    # haengt sie nichts an, ohne Eskalation zeigt sie keinen Kasten. Sie
    # nimmt aber nur Runden der Art "rag" in den Verlauf.
    kind: str = ART_RAG

    # Wie viele Verweise auf ein Dokument ueber den Assistenten aus dem
    # Antworttext genommen oder durch seinen Titel ersetzt wurden. Eine Zahl,
    # kein Text: Sie geht ins Ereignis.
    removed_references: int = 0

    # --- Gespraech ---------------------------------------------------------
    # Die Frage, mit der gesucht und geantwortet wurde, wenn eine Folgefrage
    # umgeformt wurde. None ohne Verlauf und bei einer verworfenen Umformung.
    # Die Oberflaeche zeigt sie als "Verstanden als", ins Ereignis geht sie
    # nicht: Sie ist Text.
    standalone_question: str | None = None
    # Wie viele fruehere Runden die Anfrage mitbrachte.
    history_turns: int = 0
    # "keine", "umgeformt" oder "verworfen", und bei "verworfen" der Grund:
    # "parsefehler", "leer", "zu_lang" oder "abgelehnt".
    rewrite_status: str = UMFORMUNG_KEINE
    rewrite_reason: str | None = None
    # Der zweite Aufruf, getrennt von dem der Antwort. Alle None, wenn nicht
    # umgeformt wurde.
    latency_ms_rewrite: int | None = None
    rewrite_prompt_tokens: int | None = None
    rewrite_completion_tokens: int | None = None
    rewrite_model: str | None = None


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
    genannt: list[str],
    treffer: list[SearchHit],
    ausgenommen: list[str] | tuple[str, ...] = (),
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

    `ausgenommen` sind die Dokumente ueber den Assistenten aus der tenant.yaml
    des Mandanten. Sie werden nie als Quelle ausgewiesen, mit und ohne .md und
    ohne Ruecksicht auf Gross- und Kleinschreibung. Gefiltert werden nur diese
    Namen; eine erfundene Quelle bleibt sichtbar wie oben.
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
        if ausgenommen and ist_assistentendokument(saubere, ausgenommen):
            continue
        quellen.append(saubere)
        scores.append(beste.get(saubere))
    return quellen, scores


def kurzformel_beantworten(tenant_id: str, tenant: TenantConfig, art: str) -> Answer:
    """Antwort auf eine Begruessung, einen Dank, einen Abschied oder eine Frage
    nach dem Assistenten, aus dem Katalog in der Sprache des Mandanten.

    Kein Retrieval, kein Modell: keine Scores, keine Tokens,
    `latency_ms_generation` und `model` None. `sources` ist leer.

    `tenant_id` an erster Stelle, obwohl `tenant` schon aufgeloest ist: Die
    Begruessung und die Antworten auf Fragen nach dem Assistenten setzen die
    Themen und den Namen dieses Mandanten ein, also Mandanteninhalt.
    Passen beide nicht zusammen, ist das ein Fehler im Aufrufer und kein Fall
    fuer eine Antwort.
    """
    if tenant.slug != tenant_id:
        raise ValueError("tenant passt nicht zur tenant_id.")
    texte = texte_fuer(tenant.language)
    if art == ART_BEGRUESSUNG:
        text = texte.begruessung_antwort.format(topics=tenant.topics.strip())
    elif art == ART_DANK:
        text = texte.dank_antwort
    elif art == ART_ABSCHIED:
        text = texte.abschied_antwort
    elif art == ART_FAEHIGKEITEN:
        text = texte.faehigkeiten_antwort.format(
            topics=tenant.topics.strip(), display_name=tenant.display_name
        )
    elif art == ART_IDENTITAET:
        text = texte.identitaet_antwort.format(
            topics=tenant.topics.strip(), display_name=tenant.display_name
        )
    else:
        raise ValueError(f"Unbekannte Art: {art!r}")
    return Answer(
        text=text,
        sources=[],
        source_scores=[],
        escalated=False,
        escalation_reason=None,
        escalation_metric=None,
        retrieval_scores=[],
        latency_ms_retrieval=0,
        latency_ms_generation=None,
        prompt_tokens=None,
        completion_tokens=None,
        model=None,
        lang=tenant.language,
        kind=art,
    )


@dataclass(frozen=True)
class Umformung:
    """Was die Umformung einer Folgefrage ergeben hat."""

    # Die Frage, die in Suche und Antwortprompt geht: die eigenstaendige Frage,
    # oder die getippte, wenn die Umformung verworfen wurde.
    suchfrage: str
    eigenstaendige_frage: str | None
    status: str
    grund: str | None
    latenz_ms: int
    prompt_tokens: int | None
    completion_tokens: int | None
    modell: str | None

    def als_felder(self, runden: int) -> dict[str, object]:
        """Die Felder einer `Answer`, die diese Umformung beschreiben."""
        return {
            "standalone_question": self.eigenstaendige_frage,
            "history_turns": runden,
            "rewrite_status": self.status,
            "rewrite_reason": self.grund,
            "latency_ms_rewrite": self.latenz_ms,
            "rewrite_prompt_tokens": self.prompt_tokens,
            "rewrite_completion_tokens": self.completion_tokens,
            "rewrite_model": self.modell,
        }


def eigenstaendige_frage(
    tenant_id: str,
    tenant: TenantConfig,
    verlauf: list[Runde],
    frage: str,
    umformer: Umformer,
) -> Umformung:
    """Formt eine Folgefrage zu einer Frage um, die ohne den Verlauf verstaendlich ist.

    `tenant_id` an erster Stelle, weil der Verlauf Mandanteninhalt ist. Passt
    `tenant.slug` nicht, ist das ein Fehler im Aufrufer.

    Ist die Umformung unbrauchbar, gilt die getippte Frage, und die Tore
    entscheiden wie ohne Verlauf. Unbrauchbar heisst: keine lesbare Struktur,
    eine leere Frage, mehr als 2 000 Zeichen, oder der Anbieter lehnt den
    Aufruf ab, etwa sein Inhaltsfilter. Der Verlauf kann den Text enthalten,
    den der Filter ablehnt, die getippte Frage nicht.

    Ist der Anbieter nicht erreichbar, geht `LlmNichtErreichbar` weiter, mit
    `aufruf="umformung"`: Er ist dann fuer die Antwort ebenso wenig erreichbar.
    """
    if tenant.slug != tenant_id:
        raise ValueError("tenant passt nicht zur tenant_id.")
    system_prompt, user_prompt = umformungsprompts_bauen(tenant_id, tenant, verlauf, frage)

    start = time.perf_counter()
    try:
        ergebnis = umformer.umformen(system_prompt, user_prompt)
    except LlmNichtErreichbar as fehler:
        fehler.aufruf = "umformung"
        raise
    except LlmAnfrageAbgelehnt:
        return Umformung(
            suchfrage=frage,
            eigenstaendige_frage=None,
            status=UMFORMUNG_VERWORFEN,
            grund=VERWORFEN_ABGELEHNT,
            latenz_ms=int((time.perf_counter() - start) * 1000),
            prompt_tokens=None,
            completion_tokens=None,
            modell=None,
        )
    latenz = int((time.perf_counter() - start) * 1000)

    neu = ergebnis.frage.strip() if ergebnis.frage is not None else None
    if neu is None:
        grund: str | None = VERWORFEN_PARSEFEHLER
    elif not neu:
        grund = VERWORFEN_LEER
    elif len(neu) > UMFORMUNG_MAX_ZEICHEN:
        grund = VERWORFEN_ZU_LANG
    else:
        grund = None

    return Umformung(
        suchfrage=frage if grund is not None or neu is None else neu,
        eigenstaendige_frage=None if grund is not None else neu,
        status=UMFORMUNG_VERWORFEN if grund is not None else UMFORMUNG_UMGEFORMT,
        grund=grund,
        latenz_ms=latenz,
        prompt_tokens=ergebnis.prompt_tokens,
        completion_tokens=ergebnis.completion_tokens,
        modell=ergebnis.model,
    )


def answer(
    tenant_id: str,
    question: str,
    response_language: str | None = None,
    settings: Settings | None = None,
    strategy: EscalationStrategy | None = None,
    llm: LlmClient | None = None,
    embeddings: E5Embeddings | None = None,
    history: list[Runde] | None = None,
    umformer: Umformer | None = None,
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

    `history` sind die frueheren Runden des Gespraechs aus dem Browser. Der
    Server haelt sie nicht ueber die Anfrage hinaus. Ohne Verlauf laeuft alles
    Zeichen fuer Zeichen wie vorher. Mit Verlauf:

      1. Entspricht die getippte Frage GENAU einer Formel, kommt die feste
         Antwort, ohne Umformung und ohne Modell.
      2. Sonst wird die Frage umgeformt, siehe `eigenstaendige_frage`.
      3. Dann laeuft die Erkennung mit Toleranz auf die Frage, die sonst in die
         Suche ginge. Trifft sie, kommt die feste Antwort ohne Suche und ohne
         Antwortaufruf.
      4. Sonst der Weg ueber Suche und Modell mit dieser Frage. Den Verlauf
         sieht der Antwortprompt nie; er ist kein Beleg.

    Die Reihenfolge hat einen Grund: Eine kurze Folgefrage wie "How can you
    tell?" liegt innerhalb der Toleranz einer Formel, meint aber die vorige
    Antwort. Nach der Umformung traegt sie ihren Bezug und trifft nicht mehr.

    `strategy`, `llm`, `umformer` und `embeddings` sind injizierbar, damit Tests
    ohne Netz und ohne Schluessel auskommen. Keines davon hat einen
    Mandantenbezug.
    """
    aktive_settings = settings if settings is not None else get_settings()
    tenant = load_tenant(tenant_id, aktive_settings.tenants_dir)
    verlauf = list(history or [])

    if not verlauf:
        # --- Begruessung, Dank, Abschied, Fragen nach dem Assistenten ------
        # Ohne Retrieval und ohne Modell. Nur, wenn die GANZE Eingabe einer
        # Formel entspricht, bei den Fragen nach dem Assistenten mit Toleranz
        # fuer Tippfehler. Jede andere Eingabe laeuft unveraendert den Weg
        # darunter.
        art = kurzformel_erkennen(question)
        if art is not None:
            return kurzformel_beantworten(tenant_id, tenant, art)
        return _beantworten(
            tenant_id,
            tenant,
            question,
            response_language,
            aktive_settings,
            strategy,
            llm,
            embeddings,
        )

    # --- Mit Verlauf ---------------------------------------------------------
    art = kurzformel_genau(question)
    if art is not None:
        return kurzformel_beantworten(tenant_id, tenant, art).model_copy(
            update={"history_turns": len(verlauf)}
        )

    aktiver_umformer = (
        umformer
        if umformer is not None
        else build_umformer(
            aktive_settings,
            tenant.model_override,
            schema=umformungsmodell_fuer(tenant.language),
        )
    )
    umformung = eigenstaendige_frage(tenant_id, tenant, verlauf, question, aktiver_umformer)
    felder = umformung.als_felder(len(verlauf))

    art = kurzformel_erkennen(umformung.suchfrage)
    if art is not None:
        return kurzformel_beantworten(tenant_id, tenant, art).model_copy(update=felder)
    return _beantworten(
        tenant_id,
        tenant,
        umformung.suchfrage,
        response_language,
        aktive_settings,
        strategy,
        llm,
        embeddings,
    ).model_copy(update=felder)


def _beantworten(
    tenant_id: str,
    tenant: TenantConfig,
    frage: str,
    response_language: str | None,
    aktive_settings: Settings,
    strategy: EscalationStrategy | None,
    llm: LlmClient | None,
    embeddings: E5Embeddings | None,
) -> Answer:
    """Der Weg ueber Retrieval und Modell, mit beiden Toren.

    `frage` ist die getippte Frage oder, bei einer Folgefrage, die
    eigenstaendige. Suche und Antwortprompt bekommen dieselbe Frage, damit
    beide Tore dieselbe Frage beurteilen.

    `tenant_id` und `tenant` muessen zusammenpassen: Die Suche laeuft im Index
    von `tenant_id`, die Prompts tragen Name, Zusatz und Sprache von `tenant`.
    Passen beide nicht, ist das ein Fehler im Aufrufer, und er faellt hier auf,
    vor der Suche, nicht erst nach der Antwort.
    """
    if tenant.slug != tenant_id:
        raise ValueError("tenant passt nicht zur tenant_id.")

    # --- Retrieval ---------------------------------------------------------
    start = time.perf_counter()
    treffer: list[SearchHit] = search_tenant(
        tenant_id,
        frage,
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
    user_prompt = build_user_prompt(frage, treffer, tenant.language)

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

    quellen, quellscores = _quellen_verdichten(
        ergebnis.parsed.sources, treffer, ausgenommen=tenant.assistant_documents
    )
    # Nach beiden Toren und nur hier: Eine Eskalation traegt den Text des
    # Mandanten, keinen Modelltext.
    text, entfernt = assistentenverweise_entfernen(
        tenant_id, tenant, ergebnis.parsed.answer, ergebnis.parsed.language
    )
    return Answer(
        text=text,
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
        removed_references=entfernt,
    )
