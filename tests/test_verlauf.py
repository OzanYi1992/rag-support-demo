"""Tests des Gespraechs: Verlauf, eigenstaendige Frage und feste Antworten.

Was hier gehalten wird:

1. Ohne Verlauf laeuft alles wie vorher, und die Umformung wird nie gerufen.
2. Mit Verlauf suchen Retrieval und Antwortprompt mit der eigenstaendigen
   Frage. Der Verlauf erreicht den Antwortprompt nie.
3. Eine unbrauchbare oder abgelehnte Umformung faellt auf die getippte Frage
   zurueck.
4. Feste Antworten im Gespraech: Eine genaue Formel antwortet ohne Umformung.
   Die Toleranz fuer Tippfehler gilt erst fuer die Frage nach der Umformung,
   damit eine Folgefrage mit Bezug keine feste Antwort bekommt.

Ersatzobjekte, die bei jedem Zugriff werfen, belegen, dass ein Weg OHNE Suche
oder ohne Modell auskommt, und nicht nur, dass ein Flag das behauptet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.embeddings import E5Embeddings
from app.llm import LlmAnfrageAbgelehnt, LlmNichtErreichbar
from app.prompts import GroundedAnswer, Runde, umformungsprompts_bauen
from app.rag import _beantworten, answer, eigenstaendige_frage
from app.tenants import load_tenant
from tests.conftest import FAKE_DIMENSION, FakeBackend, FakeLlm, FakeUmformer, lege_mandant_an

ENGLISCH_TEXT = """# Shipping

Standard delivery takes three working days. Express delivery arrives the next
working day if the order is placed before noon.
"""

ACME_TEXT = """# Retouren

Jede Ruecksendung braucht eine RMA-Nummer. Sie ist 21 Kalendertage gueltig.
"""

# Ein Marker in einer frueheren Antwort. Er darf nur im Prompt der Umformung
# stehen, nie im Prompt der Antwort.
MARKER_VERLAUF = "MARKER-VERLAUF-7Q4Z"

VERLAUF = [
    Runde(
        question="Do you ship to Norway?",
        answer=f"Yes, Norway is in Zone B. {MARKER_VERLAUF}",
    )
]


class _Verboten:
    """Ersatz, der jeden Zugriff als Fehler meldet, auch das Lesen eines Attributs."""

    def __init__(self, rolle: str) -> None:
        self._rolle = rolle

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"{self._rolle} wurde benutzt: {name}")


@pytest.fixture
def umgebung(tmp_path: Path) -> tuple[Settings, E5Embeddings]:
    """Ein englischer und ein deutscher Mandant, je mit einem Dokument."""
    from app.ingest import ingest_tenant

    root = tmp_path / "tenants"
    root.mkdir()
    lege_mandant_an(
        root, "demo-englisch", ENGLISCH_TEXT, "englisch-token-1234567890", language="en"
    )
    lege_mandant_an(root, "demo-acme", ACME_TEXT, "acme-token-1234567890")
    settings = Settings(
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=root,
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
        chunk_size=200,
        chunk_overlap=20,
    )
    embeddings = E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION)
    for slug in ("demo-englisch", "demo-acme"):
        ingest_tenant(slug, settings=settings, embeddings=embeddings)
    return settings, embeddings


@pytest.fixture
def gesucht(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Zeichnet jede Frage auf, mit der `answer()` sucht, und sucht dann echt."""
    import app.rag

    fragen: list[str] = []
    echt = app.rag.search_tenant

    def aufzeichnen(tenant_id: str, frage: str, **rest: object) -> object:
        fragen.append(frage)
        return echt(tenant_id, frage, **rest)  # type: ignore[arg-type]

    monkeypatch.setattr("app.rag.search_tenant", aufzeichnen)
    return fragen


def _antwort_en() -> GroundedAnswer:
    return GroundedAnswer(
        answerable=True, answer="Three working days.", sources=["doku.md"], language="en"
    )


# --- Suche und Antwortprompt mit der eigenstaendigen Frage ---------------------


def test_mit_verlauf_sucht_die_eigenstaendige_frage(
    umgebung: tuple[Settings, E5Embeddings], gesucht: list[str]
) -> None:
    """Pflichttest. Suche und Userprompt bekommen die Frage des Umformers,
    nicht die getippte. Gegenprobe im selben Test: ohne Verlauf die getippte
    Frage, und der Umformer wird nicht gerufen."""
    settings, embeddings = umgebung
    umformer = FakeUmformer("How long does delivery to Norway take?")
    llm = FakeLlm(parsed=_antwort_en())

    a = answer(
        "demo-englisch",
        "how long does that take?",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
        history=VERLAUF,
        umformer=umformer,
    )

    assert gesucht == ["How long does delivery to Norway take?"]
    assert len(umformer.calls) == 1
    assert len(llm.calls) == 1
    _, user_prompt = llm.calls[0]
    assert user_prompt.endswith("Question: How long does delivery to Norway take?")
    assert "how long does that take?" not in user_prompt
    assert a.kind == "rag"
    assert a.standalone_question == "How long does delivery to Norway take?"
    assert a.history_turns == 1
    assert a.rewrite_status == "umgeformt"
    assert a.rewrite_reason is None
    assert a.rewrite_prompt_tokens == 211 and a.rewrite_completion_tokens == 17
    assert a.latency_ms_rewrite is not None
    assert a.rewrite_model == "fake-model-2026"

    # Gegenprobe: ohne Verlauf die getippte Frage, kein Umformer.
    gesucht.clear()
    nie = FakeUmformer("darf nie kommen")
    b = answer(
        "demo-englisch",
        "how long does that take?",
        settings=settings,
        embeddings=embeddings,
        llm=FakeLlm(parsed=_antwort_en()),
        umformer=nie,
    )
    assert gesucht == ["how long does that take?"]
    assert nie.calls == []
    assert b.standalone_question is None
    assert b.history_turns == 0
    assert b.rewrite_status == "keine"
    assert b.latency_ms_rewrite is None


def test_der_verlauf_erreicht_den_antwortprompt_nie(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Ein Marker in einer frueheren Antwort steht im Prompt der Umformung und in
    keinem der beiden Prompts der Antwort. Faengt einen Verlauf, der zum Beleg
    wird."""
    settings, embeddings = umgebung
    umformer = FakeUmformer("How long does delivery to Norway take?")
    llm = FakeLlm(parsed=_antwort_en())

    answer(
        "demo-englisch",
        "how long does that take?",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
        history=VERLAUF,
        umformer=umformer,
    )

    _, umformung_user = umformer.calls[0]
    assert MARKER_VERLAUF in umformung_user
    system_prompt, user_prompt = llm.calls[0]
    assert MARKER_VERLAUF not in system_prompt
    assert MARKER_VERLAUF not in user_prompt
    assert "Do you ship to Norway?" not in user_prompt


# --- unbrauchbare Umformung ---------------------------------------------------


@pytest.mark.parametrize(
    ("frage", "grund"),
    [(None, "parsefehler"), ("   ", "leer"), ("x" * 2001, "zu_lang")],
)
def test_umformung_unbrauchbar_faellt_auf_die_originalfrage_zurueck(
    umgebung: tuple[Settings, E5Embeddings],
    gesucht: list[str],
    frage: str | None,
    grund: str,
) -> None:
    """Parsefehler, leere Frage und 2 001 Zeichen: Suche und Antwort mit der
    getippten Frage, keine eigenstaendige Frage, und der Grund als fester Code."""
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=_antwort_en())

    a = answer(
        "demo-englisch",
        "how long does that take?",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
        history=VERLAUF,
        umformer=FakeUmformer(frage),
    )

    assert gesucht == ["how long does that take?"]
    assert llm.calls[0][1].endswith("Question: how long does that take?")
    assert a.standalone_question is None
    assert a.rewrite_status == "verworfen"
    assert a.rewrite_reason == grund
    assert a.latency_ms_rewrite is not None


def test_eine_umformung_mit_genau_2000_zeichen_gilt(
    umgebung: tuple[Settings, E5Embeddings], gesucht: list[str]
) -> None:
    """Gegenprobe zur Grenze: 2 000 Zeichen sind noch eine Frage."""
    settings, embeddings = umgebung
    lang = "y" * 2000
    a = answer(
        "demo-englisch",
        "how long does that take?",
        settings=settings,
        embeddings=embeddings,
        llm=FakeLlm(parsed=_antwort_en()),
        history=VERLAUF,
        umformer=FakeUmformer(lang),
    )
    assert gesucht == [lang]
    assert a.rewrite_status == "umgeformt"


def test_abgelehnte_umformung_faellt_auf_die_originalfrage_zurueck(
    umgebung: tuple[Settings, E5Embeddings], gesucht: list[str]
) -> None:
    """Lehnt der Anbieter die Umformung ab, etwa sein Inhaltsfilter, gilt die
    getippte Frage: Der Verlauf kann den abgelehnten Text tragen, die Frage
    nicht."""
    settings, embeddings = umgebung
    llm = FakeLlm(parsed=_antwort_en())

    a = answer(
        "demo-englisch",
        "how long does that take?",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
        history=VERLAUF,
        umformer=FakeUmformer(fehler=LlmAnfrageAbgelehnt("inhaltsfilter")),
    )

    assert gesucht == ["how long does that take?"]
    assert len(llm.calls) == 1
    assert a.rewrite_status == "verworfen"
    assert a.rewrite_reason == "abgelehnt"
    assert a.standalone_question is None


def test_umformer_nicht_erreichbar_geht_als_umformung_weiter(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Ist der Anbieter fuer die Umformung nicht erreichbar, gibt es keine
    Antwort: Die Ausnahme geht weiter, mit `aufruf="umformung"`, und weder
    Suche noch Antwortmodell werden benutzt."""
    settings, _ = umgebung
    with pytest.raises(LlmNichtErreichbar) as fehler:
        answer(
            "demo-englisch",
            "how long does that take?",
            settings=settings,
            embeddings=_Verboten("Embedder"),  # type: ignore[arg-type]
            llm=_Verboten("Modell"),  # type: ignore[arg-type]
            history=VERLAUF,
            umformer=FakeUmformer(fehler=LlmNichtErreichbar("zeitlimit")),
        )
    assert fehler.value.aufruf == "umformung"
    assert fehler.value.grund == "zeitlimit"


# --- feste Antworten im Gespraech ---------------------------------------------


@pytest.mark.parametrize(
    ("eingabe", "art"),
    [("Thanks", "thanks"), ("What can you do?", "capabilities"), ("Sind Sie ein Bot?", "identity")],
)
def test_mit_verlauf_und_genauer_formel_ohne_umformung_und_ohne_modell(
    umgebung: tuple[Settings, E5Embeddings], eingabe: str, art: str
) -> None:
    """Eine genaue Formel bekommt auch im Gespraech die feste Antwort, ohne
    Umformung, ohne Suche und ohne Modell. Alle drei Ersatzobjekte werfen."""
    settings, _ = umgebung
    a = answer(
        "demo-englisch",
        eingabe,
        settings=settings,
        embeddings=_Verboten("Embedder"),  # type: ignore[arg-type]
        llm=_Verboten("Modell"),  # type: ignore[arg-type]
        history=VERLAUF,
        umformer=_Verboten("Umformer"),  # type: ignore[arg-type]
    )
    assert a.kind == art
    assert a.sources == []
    assert a.escalated is False
    assert a.history_turns == 1
    assert a.rewrite_status == "keine"
    assert a.standalone_question is None


def test_ok_mit_verlauf_ergibt_die_feste_antwort(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """ "ok" nach einer Antwort: feste Antwort ohne Umformung, ohne Suche und ohne
    Modell. Vorher ging die Bestaetigung in die Umformung und dann in die Suche.
    Umformer und Antwortclient werfen bei jedem Zugriff."""
    settings, _ = umgebung
    a = answer(
        "demo-englisch",
        "ok",
        settings=settings,
        embeddings=_Verboten("Embedder"),  # type: ignore[arg-type]
        llm=_Verboten("Modell"),  # type: ignore[arg-type]
        history=VERLAUF,
        umformer=_Verboten("Umformer"),  # type: ignore[arg-type]
    )
    assert a.kind == "acknowledgment"
    assert a.text == "Sure. What would you like to know?"
    assert a.sources == [] and a.escalated is False
    assert a.history_turns == 1
    assert a.rewrite_status == "keine"
    assert a.standalone_question is None


def test_heinis_your_programmer_wird_ueber_die_umformung_zu_identity(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Aus dem Telefontest: "Heinis your programmer" trifft selbst keine Formel.
    Die Umformung macht daraus "Who is your programmer?", und das ergibt die
    feste Antwort auf die Frage, wer antwortet. Weder Suche noch Antwortaufruf."""
    settings, _ = umgebung
    a = answer(
        "demo-englisch",
        "Heinis your programmer",
        settings=settings,
        embeddings=_Verboten("Embedder"),  # type: ignore[arg-type]
        llm=_Verboten("Modell"),  # type: ignore[arg-type]
        history=VERLAUF,
        umformer=FakeUmformer(frage="Who is your programmer?"),  # type: ignore[arg-type]
    )
    assert a.kind == "identity"
    assert a.standalone_question == "Who is your programmer?"
    assert a.rewrite_status == "umgeformt"
    assert a.sources == [] and a.escalated is False


def test_mit_verlauf_und_tippfehler_traegt_die_umformung_die_erkennung(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """ "How do ou help?" ist keine genaue Formel und geht in die Umformung. Gibt
    der Umformer sie unveraendert zurueck, greift die Toleranz: capabilities,
    ohne Suche und ohne Antwortaufruf, und die Felder der Umformung sind
    gesetzt."""
    settings, _ = umgebung
    umformer = FakeUmformer("How do ou help?")
    a = answer(
        "demo-englisch",
        "How do ou help?",
        settings=settings,
        embeddings=_Verboten("Embedder"),  # type: ignore[arg-type]
        llm=_Verboten("Modell"),  # type: ignore[arg-type]
        history=VERLAUF,
        umformer=umformer,
    )
    assert len(umformer.calls) == 1
    assert a.kind == "capabilities"
    assert a.sources == []
    assert a.escalated is False
    assert a.standalone_question == "How do ou help?"
    assert a.rewrite_status == "umgeformt"
    assert a.history_turns == 1
    assert a.rewrite_prompt_tokens == 211
    assert a.latency_ms_rewrite is not None


def test_eine_folgefrage_mit_bezug_trifft_nach_der_umformung_keine_formel(
    umgebung: tuple[Settings, E5Embeddings], gesucht: list[str]
) -> None:
    """ "How can you tell?" liegt innerhalb der Toleranz von "how can you help",
    meint im Gespraech aber die vorige Antwort. Nach der Umformung traegt die
    Frage ihren Bezug und laeuft den normalen Weg. Gegenprobe im selben Test:
    ohne Verlauf trifft dieselbe Eingabe capabilities."""
    settings, embeddings = umgebung
    umformer = FakeUmformer("How can you tell that the jacket is waterproof?")
    llm = FakeLlm(parsed=_antwort_en())

    a = answer(
        "demo-englisch",
        "How can you tell?",
        settings=settings,
        embeddings=embeddings,
        llm=llm,
        history=VERLAUF,
        umformer=umformer,
    )
    assert a.kind == "rag"
    assert gesucht == ["How can you tell that the jacket is waterproof?"]
    assert len(llm.calls) == 1
    assert llm.calls[0][1].endswith("Question: How can you tell that the jacket is waterproof?")

    # Gegenprobe: ohne Verlauf die feste Antwort.
    b = answer(
        "demo-englisch",
        "How can you tell?",
        settings=settings,
        embeddings=_Verboten("Embedder"),  # type: ignore[arg-type]
        llm=_Verboten("Modell"),  # type: ignore[arg-type]
    )
    assert b.kind == "capabilities"


def test_verworfene_umformung_und_tippfehler_ergibt_identity(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Bei verworfener Umformung laeuft die Erkennung mit Toleranz auf die
    getippte Frage: "Are you a bott?" ergibt identity, ohne Suche und Modell."""
    settings, _ = umgebung
    a = answer(
        "demo-englisch",
        "Are you a bott?",
        settings=settings,
        embeddings=_Verboten("Embedder"),  # type: ignore[arg-type]
        llm=_Verboten("Modell"),  # type: ignore[arg-type]
        history=VERLAUF,
        umformer=FakeUmformer(None),
    )
    assert a.kind == "identity"
    assert a.rewrite_status == "verworfen"
    assert a.rewrite_reason == "parsefehler"
    assert a.standalone_question is None


# --- Umformungsprompt und Mandant ---------------------------------------------


def test_umformungsprompt_im_rahmen_der_mandantensprache(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Der Rahmen folgt der Sprache des Mandanten. Eine eskalierte Runde steht
    mit leerer Antwort und ohne Leerzeichen am Zeilenende da, und der Prompt
    endet mit der letzten Frage."""
    settings, _ = umgebung
    verlauf = [
        Runde(question="Do you ship to Norway?", answer="Yes."),
        Runde(question="Student discount?", answer=""),
    ]
    englisch = load_tenant("demo-englisch", settings.tenants_dir)
    system, user = umformungsprompts_bauen("demo-englisch", englisch, verlauf, "how long?")
    assert system.startswith("You rewrite the last question")
    assert user == (
        "Conversation:\n\n"
        "Question: Do you ship to Norway?\nAnswer: Yes.\n\n"
        "Question: Student discount?\nAnswer:\n\n"
        "---\n\n"
        "Last question: how long?"
    )

    acme = load_tenant("demo-acme", settings.tenants_dir)
    system, user = umformungsprompts_bauen("demo-acme", acme, verlauf, "and after that?")
    assert system.startswith("Du formulierst die letzte Frage")
    assert user.startswith("Gespraech:\n\nFrage: Do you ship to Norway?\nAntwort: Yes.")
    assert user.endswith("---\n\nLetzte Frage: and after that?")


def test_umformung_prueft_den_mandanten(umgebung: tuple[Settings, E5Embeddings]) -> None:
    """Der Verlauf eines Mandanten geht nie unter der tenant_id eines anderen hinaus."""
    settings, _ = umgebung
    acme = load_tenant("demo-acme", settings.tenants_dir)
    with pytest.raises(ValueError):
        umformungsprompts_bauen("demo-englisch", acme, VERLAUF, "x?")
    nie = FakeUmformer("x")
    with pytest.raises(ValueError):
        eigenstaendige_frage("demo-englisch", acme, VERLAUF, "x?", nie)
    assert nie.calls == []


def test_der_weg_ueber_suche_und_modell_prueft_den_mandanten(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Passen tenant_id und tenant nicht zusammen, faellt das vor der Suche auf:
    Embedder und Modell werfen bei jedem Zugriff und werden nie erreicht.
    Gegenprobe: mit dem passenden Mandanten laeuft derselbe Weg durch."""
    settings, embeddings = umgebung
    acme = load_tenant("demo-acme", settings.tenants_dir)
    with pytest.raises(ValueError, match="tenant passt nicht"):
        _beantworten(
            "demo-englisch",
            acme,
            "How long does delivery take?",
            None,
            settings,
            None,
            _Verboten("Modell"),  # type: ignore[arg-type]
            _Verboten("Embedder"),  # type: ignore[arg-type]
        )
    llm = FakeLlm(parsed=_antwort_en())
    a = _beantworten(
        "demo-acme", acme, "Wie lange gilt die RMA?", None, settings, None, llm, embeddings
    )
    assert a.kind == "rag"
    assert len(llm.calls) == 1
