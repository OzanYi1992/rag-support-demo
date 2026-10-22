"""Der Server legt die Antwortsprache fest, prueft die Antwort und fragt hoechstens
einmal nach.

Was diese Datei festhaelt:

* woher die Sprache kommt: vorgegeben, aus der Frage, aus der Umformung, oder
  vom Mandanten;
* dass der erste Aufruf laeuft wie ohne Sprachpruefung, und ein zweiter nur
  folgt, wenn die belegte erste Antwort die Sprache nicht haelt;
* dass ein zweiter Aufruf ohne belegte Antwort die erste stehen laesst, auch
  wenn der Anbieter ihn ablehnt oder nicht erreichbar ist;
* dass es nie einen dritten gibt.

Alles mit Ersatzmodell und Ersatz-Embedder, ohne Netz. Der Erkenner ist der
echte: Er laedt ohne Netz, und ein Ersatz waere hier genau das, was gemessen
werden soll.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.embeddings import E5Embeddings
from app.llm import LlmAnfrageAbgelehnt, LlmNichtErreichbar
from app.prompts import GroundedAnswer, Runde, build_system_prompt, build_user_prompt
from app.rag import (
    _belegt_aufbereiten,
    _sprache_nachfordern,
    answer,
    antwortsprache_festlegen,
)
from app.search import search_tenant
from app.tenants import load_tenant
from tests.conftest import (
    FAKE_DIMENSION,
    FakeBackend,
    FakeLlmFolge,
    FakeUmformer,
    lege_mandant_an,
)

DE_TEXT = """# Lieferung

Wir liefern nach Norwegen in fuenf Werktagen. Die RMA-Nummer ist 21
Kalendertage gueltig.
"""

EN_TEXT = """# Shipping

We ship to Norway. Delivery to Norway takes five working days.
"""

EN = "Yes, we ship to Norway within five working days."
DE = "Ja, wir liefern innerhalb von fünf Werktagen nach Norwegen."
DE_ZWEI = "Die Lieferung nach Norwegen dauert fünf Werktage."
NO = "Ja, vi leverer til Norge innen fem virkedager."

FRAGE_EN = "Do you ship to Norway?"
FRAGE_DE = "Wie lange dauert die Lieferung nach Norwegen?"


def _belegt(text: str, language: str) -> GroundedAnswer:
    return GroundedAnswer(answerable=True, answer=text, sources=["doku.md"], language=language)


def _unbelegt(language: str = "en") -> GroundedAnswer:
    return GroundedAnswer(answerable=False, answer="", sources=[], language=language)


@pytest.fixture
def umgebung(tmp_path: Path) -> tuple[Settings, E5Embeddings]:
    """Ein deutscher und ein englischer Mandant, je mit einem Dokument."""
    from app.ingest import ingest_tenant

    wurzel = tmp_path / "tenants"
    wurzel.mkdir()
    lege_mandant_an(wurzel, "demo-de", DE_TEXT, "de-token-1234567890123")
    lege_mandant_an(wurzel, "demo-en", EN_TEXT, "en-token-1234567890123", language="en")
    settings = Settings(
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=wurzel,
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
        chunk_size=200,
        chunk_overlap=20,
    )
    embeddings = E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION)
    for slug in ("demo-de", "demo-en"):
        ingest_tenant(slug, settings=settings, embeddings=embeddings)
    return settings, embeddings


def _fragen(
    umgebung: tuple[Settings, E5Embeddings],
    slug: str,
    frage: str,
    llm: FakeLlmFolge,
    **weiter: object,
) -> object:
    settings, embeddings = umgebung
    return answer(slug, frage, settings=settings, embeddings=embeddings, llm=llm, **weiter)


def _ohne_zweiten_aufruf(a: object) -> None:
    assert a.language_retry is False  # type: ignore[attr-defined]
    assert a.language_retry_result is None  # type: ignore[attr-defined]
    assert a.latency_ms_language_retry is None  # type: ignore[attr-defined]
    assert a.language_retry_prompt_tokens is None  # type: ignore[attr-defined]
    assert a.language_retry_completion_tokens is None  # type: ignore[attr-defined]


# --- Die erste Antwort haelt die Sprache --------------------------------------


def test_englische_frage_englische_antwort_ein_aufruf(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    llm = FakeLlmFolge(_belegt(EN, "en"))

    a = _fragen(umgebung, "demo-en", FRAGE_EN, llm)

    assert len(llm.calls) == 1
    assert a.text == EN
    assert a.answer_language == "en"
    assert a.answer_language_source == "frage"
    assert a.language_kept is True
    _ohne_zweiten_aufruf(a)
    # Der erste Aufruf laeuft wie ohne Sprachpruefung: ohne Vorgabe.
    settings, _ = umgebung
    assert llm.calls[0][0] == build_system_prompt(load_tenant("demo-en", settings.tenants_dir))


# --- Die erste Antwort haelt die Sprache nicht --------------------------------


def test_deutsche_erste_antwort_bekommt_genau_einen_zweiten_aufruf(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Der zweite Systemprompt ist die bestehende Fassung fuer eine vorgegebene
    Sprache, der Userprompt ist derselbe, also dieselben Treffer. Die zweite
    Antwort wird ausgeliefert, mit ihrer Selbstauskunft; Tokens und Latenz der
    beiden Aufrufe stehen getrennt."""
    settings, embeddings = umgebung
    tenant = load_tenant("demo-de", settings.tenants_dir)
    llm = FakeLlmFolge(_belegt(DE, "en"), _belegt(EN, "en"))

    a = _fragen(umgebung, "demo-de", FRAGE_EN, llm)

    assert len(llm.calls) == 2
    (system_1, user_1), (system_2, user_2) = llm.calls
    assert system_1 == build_system_prompt(tenant)
    assert system_2 == build_system_prompt(tenant, "en")
    assert user_2 == user_1
    treffer = search_tenant(
        "demo-de", FRAGE_EN, k=settings.retrieval_top_k, settings=settings, embeddings=embeddings
    )
    assert user_1 == build_user_prompt(FRAGE_EN, treffer, tenant.language)

    assert a.text == EN
    assert a.lang == "en"
    assert a.answer_language == "en"
    assert a.answer_language_source == "frage"
    assert a.language_kept is True
    assert a.language_retry is True
    assert a.language_retry_result == "ersetzt"
    assert isinstance(a.latency_ms_language_retry, int)
    assert (a.prompt_tokens, a.completion_tokens) == (100, 10)
    assert (a.language_retry_prompt_tokens, a.language_retry_completion_tokens) == (101, 11)
    assert a.escalated is False


def test_auch_die_zweite_deutsch_wird_ausgeliefert_ohne_dritten_aufruf(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    llm = FakeLlmFolge(_belegt(DE, "de"), _belegt(DE_ZWEI, "de"))

    a = _fragen(umgebung, "demo-de", FRAGE_EN, llm)

    assert len(llm.calls) == 2
    assert a.text == DE_ZWEI
    assert a.language_retry_result == "ersetzt"
    assert a.language_kept is False


@pytest.mark.parametrize(
    ("zweiter", "code", "tokens"),
    [
        (_unbelegt(), "eskaliert", (101, 11)),
        (None, "parsefehler", (101, 11)),
        (LlmNichtErreichbar("zeitlimit"), "nicht_erreichbar", (None, None)),
        (LlmAnfrageAbgelehnt("inhaltsfilter"), "abgelehnt", (None, None)),
    ],
)
def test_ohne_belegte_zweite_bleibt_die_erste(
    umgebung: tuple[Settings, E5Embeddings],
    zweiter: object,
    code: str,
    tokens: tuple[int | None, int | None],
) -> None:
    """Eskaliert der zweite Aufruf, ist er unlesbar, oder scheitert er am
    Anbieter, bleibt die erste belegte Antwort, ohne Eskalation und ohne Fehler."""
    llm = FakeLlmFolge(_belegt(DE, "de"), zweiter)

    a = _fragen(umgebung, "demo-de", FRAGE_EN, llm)

    assert len(llm.calls) == 2
    assert a.text == DE
    assert a.lang == "de"
    assert a.escalated is False
    assert a.sources == ["doku.md"]
    assert a.language_retry is True
    assert a.language_retry_result == code
    assert a.language_kept is False
    assert isinstance(a.latency_ms_language_retry, int)
    assert (a.language_retry_prompt_tokens, a.language_retry_completion_tokens) == tokens
    assert (a.prompt_tokens, a.completion_tokens) == (100, 10)


@pytest.mark.parametrize("erste", [_unbelegt("de"), None])
def test_eine_eskalierte_erste_bekommt_keinen_zweiten_aufruf(
    umgebung: tuple[Settings, E5Embeddings], erste: object
) -> None:
    llm = FakeLlmFolge(erste)

    a = _fragen(umgebung, "demo-de", FRAGE_EN, llm)

    assert len(llm.calls) == 1
    assert a.escalated is True
    assert a.answer_language == "en"
    assert a.answer_language_source == "frage"
    assert a.language_kept is None
    _ohne_zweiten_aufruf(a)


# --- Vorgegeben, Mandant, feste Antwort ---------------------------------------


def test_eine_vorgegebene_sprache_wird_nicht_geprueft(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Ein Aufruf mit dieser Sprache. Auch eine Antwort in einer anderen
    Sprache loest keinen zweiten aus: Er haette dieselbe Vorgabe."""
    settings, _ = umgebung
    tenant = load_tenant("demo-en", settings.tenants_dir)
    llm = FakeLlmFolge(_belegt(EN, "de"))

    a = _fragen(umgebung, "demo-en", FRAGE_EN, llm, response_language="de")

    assert len(llm.calls) == 1
    assert llm.calls[0][0] == build_system_prompt(tenant, "de")
    assert a.answer_language == "de"
    assert a.answer_language_source == "vorgegeben"
    assert a.language_kept is None
    _ohne_zweiten_aufruf(a)


def test_norwegisch_gilt_als_nicht_erkennbar_und_bekommt_die_sprache_des_mandanten(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Eine Frage in einer Sprache ausserhalb des Katalogs bekommt die Sprache
    des Mandanten. Eine norwegische erste Antwort haelt sie nicht und loest den
    zweiten Aufruf aus."""
    settings, _ = umgebung
    tenant = load_tenant("demo-en", settings.tenants_dir)
    llm = FakeLlmFolge(_belegt(NO, "no"), _belegt(EN, "en"))

    a = _fragen(umgebung, "demo-en", "Leverer dere til Norge?", llm)

    assert a.answer_language == "en"
    assert a.answer_language_source == "mandant"
    assert len(llm.calls) == 2
    assert llm.calls[1][0] == build_system_prompt(tenant, "en")
    assert a.text == EN
    assert a.language_retry_result == "ersetzt"


@pytest.mark.parametrize("vorgabe", [None, "en"])
def test_eine_feste_antwort_traegt_die_sprache_des_mandanten(
    umgebung: tuple[Settings, E5Embeddings], vorgabe: str | None
) -> None:
    llm = FakeLlmFolge()

    a = _fragen(umgebung, "demo-de", "Hallo", llm, response_language=vorgabe)

    assert a.kind == "greeting"
    assert llm.calls == []
    assert a.answer_language == "de"
    assert a.answer_language_source == "mandant"
    assert a.language_kept is None
    _ohne_zweiten_aufruf(a)


# --- Mit Verlauf --------------------------------------------------------------

VERLAUF = [Runde(question="Wohin liefern Sie?", answer="Wir liefern nach Norwegen.")]


def test_mit_verlauf_gilt_die_sprache_der_umformung_wenn_die_getippte_nicht_erkennbar_ist(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """ "Sure?" ist nicht erkennbar, die eigenstaendige Frage englisch: Englisch
    aus der Umformung, beim deutschen Mandanten.

    Gegenproben: Ist die getippte Frage erkennbar, geht sie vor. Wird die
    Umformung verworfen, gilt die Sprache des Mandanten.
    """
    llm = FakeLlmFolge(_belegt(EN, "en"))
    a = _fragen(
        umgebung,
        "demo-de",
        "Sure?",
        llm,
        history=VERLAUF,
        umformer=FakeUmformer(FRAGE_EN),
    )
    assert a.rewrite_status == "umgeformt"
    assert a.answer_language == "en"
    assert a.answer_language_source == "umformung"
    assert a.language_kept is True
    assert len(llm.calls) == 1

    vorrang = _fragen(
        umgebung,
        "demo-de",
        FRAGE_DE,
        FakeLlmFolge(_belegt(DE, "de")),
        history=VERLAUF,
        umformer=FakeUmformer(FRAGE_EN),
    )
    assert vorrang.answer_language == "de"
    assert vorrang.answer_language_source == "frage"

    verworfen = _fragen(
        umgebung,
        "demo-de",
        "Sure?",
        FakeLlmFolge(_belegt(DE, "de")),
        history=VERLAUF,
        umformer=FakeUmformer(None),
    )
    assert verworfen.rewrite_status == "verworfen"
    assert verworfen.answer_language == "de"
    assert verworfen.answer_language_source == "mandant"


# --- Nur fuer die Messung: ohne zweiten Aufruf --------------------------------


def test_ohne_sprachpruefung_kein_zweiter_aufruf_aber_ein_urteil(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    llm = FakeLlmFolge(_belegt(DE, "en"))

    a = _fragen(umgebung, "demo-de", FRAGE_EN, llm, sprachpruefung=False)

    assert len(llm.calls) == 1
    assert a.text == DE
    assert a.answer_language == "en"
    assert a.language_kept is False
    _ohne_zweiten_aufruf(a)


# --- tenant_id ----------------------------------------------------------------


def test_die_neuen_funktionen_pruefen_den_mandanten(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Jede der drei wirft, wenn tenant_id und tenant nicht zusammenpassen, und
    das Modell wird nie erreicht. Gegenprobe: mit dem passenden Mandanten
    laufen sie durch."""
    settings, _ = umgebung
    fremd = load_tenant("demo-de", settings.tenants_dir)
    llm = FakeLlmFolge()

    with pytest.raises(ValueError, match="tenant passt nicht"):
        antwortsprache_festlegen("demo-en", fremd, FRAGE_EN, None)
    with pytest.raises(ValueError, match="tenant passt nicht"):
        _sprache_nachfordern("demo-en", fremd, llm, "Frage", "en")
    with pytest.raises(ValueError, match="tenant passt nicht"):
        _belegt_aufbereiten("demo-en", fremd, _belegt(EN, "en"), [], settings)
    assert llm.calls == []

    assert antwortsprache_festlegen("demo-de", fremd, FRAGE_EN, None) == ("en", "frage")
    passend = FakeLlmFolge(_belegt(EN, "en"))
    assert _sprache_nachfordern("demo-de", fremd, passend, "Frage", "en").ergebnis == "ersetzt"
    assert _belegt_aufbereiten("demo-de", fremd, _belegt(EN, "en"), [], settings).text == EN
