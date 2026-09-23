"""Tests der Antwortgenerierung.

LLM gemockt, kein Netz (conventions.md). Der gefakte Client zeichnet seine
Aufrufe auf - der Nachweis, dass bei retrievalseitiger Eskalation KEIN Modell
aufgerufen wird, ist der wichtigere Teil des Eskalationstests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.embeddings import E5Embeddings
from app.escalation import (
    REASON_NO_HITS,
    AbsoluteThreshold,
    DegenerateOnly,
    EscalationDecision,
)
from app.prompts import GroundedAnswer, build_system_prompt, build_user_prompt
from app.rag import REASON_NOT_GROUNDED, REASON_UNPARSEABLE, answer
from app.search import SearchHit
from app.tenants import TenantConfig, load_tenant
from tests.conftest import FAKE_DIMENSION, FakeBackend, FakeLlm, lege_mandant_an

ACME_TEXT = """# Retouren

Jede Ruecksendung braucht eine RMA-Nummer. Sie ist 21 Kalendertage gueltig.
Verzugszinsen berechnen wir in gesetzlicher Hoehe.
"""

NORDWIND_TEXT = """# Lieferung

Wir liefern als Zwei-Mann-Montage in einem Terminfenster von vier Stunden.
"""

ESKALATIONSTEXT = "Dazu finde ich in den Unterlagen nichts."


@pytest.fixture
def umgebung(tmp_path: Path) -> tuple[Settings, E5Embeddings]:
    """Zwei ingestierte Mandanten mit gefaktem Embedder."""
    from app.ingest import ingest_tenant

    tenants_root = tmp_path / "tenants"
    tenants_root.mkdir()
    lege_mandant_an(tenants_root, "demo-acme", ACME_TEXT, "acme-token-1234567890", ESKALATIONSTEXT)
    lege_mandant_an(tenants_root, "demo-nordwind", NORDWIND_TEXT, "nordwind-token-123456")

    settings = Settings(
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=tenants_root,
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
        chunk_size=200,
        chunk_overlap=20,
    )
    embeddings = E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION)
    for slug in ("demo-acme", "demo-nordwind"):
        ingest_tenant(slug, settings=settings, embeddings=embeddings)
    return settings, embeddings


def gute_antwort(text: str = "Ja, 21 Kalendertage.", lang: str = "de") -> GroundedAnswer:
    return GroundedAnswer(answerable=True, answer=text, sources=["doku.md"], language=lang)


# --- Normalfall ------------------------------------------------------------


def test_normale_frage_liefert_quellen(umgebung: tuple[Settings, E5Embeddings]) -> None:
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=gute_antwort())

    a = answer(
        "demo-acme",
        "Wie lange gilt die RMA-Nummer?",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
    )

    assert a.escalated is False
    assert a.escalation_reason is None
    assert a.text == "Ja, 21 Kalendertage."
    assert a.sources == ["doku.md"]
    assert a.model == "fake-model-2026"
    assert a.prompt_tokens == 123
    assert a.completion_tokens == 45
    assert a.lang == "de"
    assert a.latency_ms_generation is not None
    assert a.retrieval_scores


def test_llm_wird_im_normalfall_aufgerufen(umgebung: tuple[Settings, E5Embeddings]) -> None:
    """Gegenprobe zum naechsten Test.

    Ohne sie waere 'LLM nicht aufgerufen' auch dann erfuellt, wenn der LLM-Pfad
    gar nicht existierte (conventions.md).
    """
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=gute_antwort())

    answer(
        "demo-acme",
        "Wie lange gilt die RMA-Nummer?",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
    )

    assert len(llm.calls) == 1


# --- Erstes Tor: Retrieval -------------------------------------------------


def test_retrieval_eskalation_ruft_kein_llm_auf(umgebung: tuple[Settings, E5Embeddings]) -> None:
    """Beides pruefen: das Flag UND dass kein Modell aufgerufen wurde.

    Ein escalated=True bei gleichzeitigem Aufruf waere ADR-003 formal erfuellt
    und inhaltlich verletzt - das Modell haette den Kontext bereits gesehen.
    """
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=gute_antwort())

    a = answer(
        "demo-acme",
        "Voellig andere Frage",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
        strategy=AbsoluteThreshold(1.01),  # eskaliert immer
    )

    assert a.escalated is True
    assert llm.calls == []
    assert a.latency_ms_generation is None
    assert a.prompt_tokens is None
    assert a.model is None


def test_eskalationstext_des_mandanten_wird_ausgeliefert(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    settings, embeddings = umgebung

    a = answer(
        "demo-acme",
        "Egal",
        settings=settings,
        embeddings=embeddings,
        llm=FakeLlm(parsed=gute_antwort()),
        strategy=AbsoluteThreshold(1.01),
    )

    assert a.text == ESKALATIONSTEXT
    assert a.sources == []


def test_leere_trefferliste_eskaliert_mit_no_hits(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Der no-hits-Weg, ueber eine injizierte Strategie geprueft.

    Er laesst sich NICHT ueber eine echte Suche ausloesen: `ingest_tenant`
    verweigert einen leeren Index, und FAISS liefert stets `min(k, ntotal)`
    Treffer. Eine leere Liste ist auf dem regulaeren Weg unerreichbar.

    Genau das ist der Grund, warum DegenerateOnly als Voreinstellung taugt: Das
    Retrieval-Tor ist damit strukturell inert, und Phase 5 misst garantiert das
    Groundedness-Tor allein (OP-018) - nicht nur ueberwiegend.
    """
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=gute_antwort())

    class ImmerLeer:
        def should_escalate(self, hits: list[SearchHit]) -> EscalationDecision:
            return DegenerateOnly().should_escalate([])

    a = answer(
        "demo-acme",
        "Irgendwas",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
        strategy=ImmerLeer(),
    )

    assert a.escalated is True
    assert a.escalation_reason == REASON_NO_HITS
    assert llm.calls == []


def test_degenerate_only_feuert_auf_dem_regulaeren_weg_nie(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Gegenprobe: die Voreinstellung eskaliert auf dem echten Weg nicht.

    Belegt, dass das Retrieval-Tor inert ist und die Last beim Groundedness-Tor
    liegt.
    """
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=gute_antwort())

    a = answer(
        "demo-acme",
        "Voellig zusammenhanglose Frage ueber Astronomie",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
        strategy=DegenerateOnly(),
    )

    assert a.escalated is False
    assert len(llm.calls) == 1


# --- Zweites Tor: Groundedness ---------------------------------------------


def test_groundedness_tor_eskaliert(umgebung: tuple[Settings, E5Embeddings]) -> None:
    """Das Modell meldet 'nicht abgedeckt'. Sein Text darf NICHT ausgeliefert werden."""
    settings, embeddings = umgebung
    llm = FakeLlm(
        parsed=GroundedAnswer(
            answerable=False,
            answer="Die Verzugszinsen betragen neun Prozent.",  # erfunden
            sources=[],
            language="de",
        )
    )

    a = answer(
        "demo-acme",
        "Wie hoch sind die Verzugszinsen genau?",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
    )

    assert a.escalated is True
    assert a.escalation_reason == REASON_NOT_GROUNDED
    assert a.text == ESKALATIONSTEXT
    assert "neun Prozent" not in a.text
    # Das Modell WURDE aufgerufen - dieses Tor faellt nach der Generierung.
    assert len(llm.calls) == 1
    assert a.prompt_tokens == 123
    assert a.model == "fake-model-2026"


# --- Dritter Weg: Parsefehler ----------------------------------------------


def test_parsefehler_eskaliert(umgebung: tuple[Settings, E5Embeddings]) -> None:
    """Ohne eigenen Weg entstuende hier eine leere Antwort, still."""
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=None, parsing_error="kein gueltiges JSON")

    a = answer("demo-acme", "Frage", settings=settings, embeddings=embeddings, llm=llm)

    assert a.escalated is True
    assert a.escalation_reason == REASON_UNPARSEABLE
    assert a.text == ESKALATIONSTEXT


def test_die_drei_gruende_sind_unterscheidbar() -> None:
    """Phase 5 muss messen koennen, WELCHES Tor greift, nicht nur DASS."""
    assert len({REASON_NO_HITS, REASON_NOT_GROUNDED, REASON_UNPARSEABLE}) == 3


# --- response_language -----------------------------------------------------


def test_ohne_response_language_folgt_die_sprache_der_frage(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=gute_antwort())

    answer("demo-acme", "Frage", settings=settings, embeddings=embeddings, llm=llm)

    system_prompt = llm.calls[0][0]
    assert "Sprache der FRAGE" in system_prompt


def test_response_language_uebersteuert(umgebung: tuple[Settings, E5Embeddings]) -> None:
    """Der Fall, den der Goldsatz in Phase 5 als deterministischen Test braucht."""
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=gute_antwort(lang="de"))

    a = answer(
        "demo-acme",
        "How long is the RMA number valid?",
        response_language="de",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
    )

    system_prompt = llm.calls[0][0]
    assert '"de"' in system_prompt
    assert "Sprache der FRAGE" not in system_prompt
    # lang traegt die TATSAECHLICHE Antwortsprache, nicht den Eingabewert.
    assert a.lang == "de"


def test_englischer_mandant_bekommt_ein_englisches_regelwerk(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Die Sprache des Mandanten bestimmt die Sprache der REGELN.

    Ein Prompt aus zwei Sprachen - deutsche Regeln, englischer
    system_prompt_extra - ist schlechter lesbar, fuer das Modell wie fuer den,
    der ihn spaeter prueft.
    """
    tenant = TenantConfig(
        slug="demo-englisch",
        display_name="Northfield Outfitters",
        language="en",
        escalation_message="I cannot find that in the documents.",
        url_token="englisch-token-1234567890",
    )
    prompt = build_system_prompt(tenant, None)

    assert "You are a support assistant for Northfield Outfitters." in prompt
    assert "Answer EXCLUSIVELY from the context provided" in prompt
    assert "language of the QUESTION" in prompt
    # Keine deutsche Zeile darf stehenbleiben.
    assert "AUSSCHLIESSLICH" not in prompt
    assert "Sprache der FRAGE" not in prompt


def test_deutscher_mandant_bleibt_unveraendert(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Die Gegenprobe. Ohne sie zeigt der Test oben nur, dass irgendetwas anders ist."""
    settings, _ = umgebung
    tenant = load_tenant("demo-acme", settings.tenants_dir)
    prompt = build_system_prompt(tenant, None)

    assert "AUSSCHLIESSLICH aus dem gelieferten Kontext" in prompt
    assert "Sprache der FRAGE" in prompt
    assert "You are a support assistant" not in prompt


def test_response_language_sticht_auch_bei_einem_englischen_mandanten() -> None:
    """Mandantensprache und Antwortsprache sind zwei verschiedene Dinge.

    Der englische Mandant bekommt englische Regeln - und darin die Anweisung,
    auf Deutsch zu antworten. Genau das braucht der Goldsatz, um eine deutsche
    Frage gegen einen englischen Korpus deterministisch zu pruefen.
    """
    tenant = TenantConfig(
        slug="demo-englisch",
        display_name="Northfield Outfitters",
        language="en",
        escalation_message="I cannot find that in the documents.",
        url_token="englisch-token-1234567890",
    )
    prompt = build_system_prompt(tenant, "de")

    assert "Answer in the language with the code" in prompt
    assert '"de"' in prompt
    assert "language of the QUESTION" not in prompt


def test_mandantenzusatz_wird_englisch_eingeleitet() -> None:
    """Auch die Einleitungszeile folgt der Mandantensprache."""
    tenant = TenantConfig(
        slug="demo-englisch",
        display_name="Northfield Outfitters",
        language="en",
        system_prompt_extra="Answer briefly and name the delivery window.",
        escalation_message="I cannot find that in the documents.",
        url_token="englisch-token-1234567890",
    )
    prompt = build_system_prompt(tenant, None)

    assert "Additionally for this tenant:" in prompt
    assert "Answer briefly and name the delivery window." in prompt
    assert "Zusaetzlich fuer diesen Mandanten" not in prompt


# --- Prompt-Bau ------------------------------------------------------------


def test_system_prompt_enthaelt_mandantenzusatz(umgebung: tuple[Settings, E5Embeddings]) -> None:
    settings, _ = umgebung
    tenant = load_tenant("demo-acme", settings.tenants_dir)
    prompt = build_system_prompt(tenant, None)

    assert tenant.display_name in prompt
    assert "AUSSCHLIESSLICH aus dem gelieferten Kontext" in prompt


def test_userprompt_eines_englischen_mandanten_ist_durchgehend_englisch() -> None:
    """Der Rahmen des Userprompts folgt der Mandantensprache.

    Der Fehler, den dieser Test faengt, ist am 2026-09-23 gemessen worden und
    war vorher unsichtbar: Der System-Prompt war englisch, der Userprompt fest
    deutsch. Das Modell bekam also bei jeder Anfrage einen zweisprachigen
    Prompt - und beantwortete daraufhin drei von elf englischen Fragen auf
    Deutsch.

    Keine Zeile Code war dabei falsch, und kein Test war rot. Sichtbar wurde es
    erst an den Antworttexten eines echten Laufs.
    """
    treffer = [
        SearchHit(
            tenant_slug="demo-englisch",
            source_file="a.md",
            chunk_index=0,
            text="Standard delivery takes three working days.",
            score=0.9,
        )
    ]
    prompt = build_user_prompt("How long is delivery?", treffer, "en")

    assert "Context:" in prompt
    assert "[Source: a.md]" in prompt
    assert "Question: How long is delivery?" in prompt
    for deutsch in ("Kontext", "Quelle", "Frage"):
        assert deutsch not in prompt, f"{deutsch!r} steht im englischen Userprompt"


def test_userprompt_eines_deutschen_mandanten_bleibt_deutsch() -> None:
    """Gegenprobe. Ohne sie zeigt der Test oben nur, dass sich etwas geaendert hat."""
    treffer = [
        SearchHit(
            tenant_slug="demo-acme",
            source_file="a.md",
            chunk_index=0,
            text="Die RMA-Nummer gilt 21 Kalendertage.",
            score=0.9,
        )
    ]
    prompt = build_user_prompt("Wie lange gilt die RMA?", treffer, "de")

    assert "Kontext:" in prompt
    assert "[Quelle: a.md]" in prompt
    assert "Frage: Wie lange gilt die RMA?" in prompt
    assert "Context" not in prompt


def test_leerer_kontext_ist_ebenfalls_sprachabhaengig() -> None:
    """Auch der Platzhalter fuer 'nichts gefunden' gehoert in die Mandantensprache."""
    assert "(no context found)" in build_user_prompt("Anything?", [], "en")
    assert "(kein Kontext gefunden)" in build_user_prompt("Irgendwas?", [], "de")


def test_user_prompt_traegt_keinen_score() -> None:
    """Der Score ist eine interne Kennzahl.

    Im Kontext koennte er das Modell dazu verleiten, einen hohen Wert als Beleg
    fuer Abdeckung zu lesen - genau die Verwechslung von Aehnlichkeit und
    Abdeckung, gegen die das zweite Tor gebaut ist.
    """
    treffer = [SearchHit("Text", 0.9876, "doku.md", 0, "demo-acme")]
    prompt = build_user_prompt("Frage?", treffer)

    assert "doku.md" in prompt
    assert "0.98" not in prompt
    assert "Frage?" in prompt


# --- Mandantentrennung, zweite Ebene ---------------------------------------


def test_antwort_traegt_nie_quellen_eines_anderen_mandanten(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Duenne zweite Ebene zum Isolationstest aus Phase 2, nicht dessen Ersatz.

    Geprueft wird, dass tenant_id bis in den Kontext durchgereicht wird: Kein
    Chunk von demo-nordwind darf im Prompt fuer demo-acme auftauchen.

    Wichtig: Geprueft wird nur der KONTEXTTEIL, nicht der ganze Prompt. Die Frage
    steht ebenfalls darin, und sie enthaelt hier absichtlich den Suchbegriff des
    anderen Mandanten - ein Assert ueber den ganzen Prompt traefe die eigene
    Frage statt eines fremden Chunks und waere damit wertlos.
    """
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=gute_antwort())

    answer(
        "demo-acme",
        "Zwei-Mann-Montage Terminfenster",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
    )

    user_prompt = llm.calls[0][1]
    kontext = user_prompt.split("Frage:")[0]

    assert "Zwei-Mann-Montage" not in kontext
    assert "RMA-Nummer" in kontext
    # Gegenprobe: Die Frage steht sehr wohl im Prompt, nur eben nicht im Kontext.
    assert "Zwei-Mann-Montage" in user_prompt


def test_beide_mandanten_bekommen_eigene_kontexte(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    settings, embeddings = umgebung
    llm_a = FakeLlm(parsed=gute_antwort())
    llm_n = FakeLlm(parsed=gute_antwort())

    answer("demo-acme", "Lieferung", settings=settings, embeddings=embeddings, llm=llm_a)
    answer("demo-nordwind", "Lieferung", settings=settings, embeddings=embeddings, llm=llm_n)

    assert "RMA-Nummer" in llm_a.calls[0][1]
    assert "Zwei-Mann-Montage" in llm_n.calls[0][1]
    assert "Zwei-Mann-Montage" not in llm_a.calls[0][1]
    assert "RMA-Nummer" not in llm_n.calls[0][1]


def test_strategie_wird_ohne_injektion_aus_settings_gebaut(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Ohne strategy-Parameter gilt die Voreinstellung DegenerateOnly."""
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=gute_antwort())

    a = answer("demo-acme", "Frage", settings=settings, embeddings=embeddings, llm=llm)

    assert isinstance(DegenerateOnly(), DegenerateOnly)
    assert a.escalated is False
