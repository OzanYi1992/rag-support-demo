"""Tests von Ingest und Suche auf Mechanismusebene.

Alle Tests hier laufen mit dem gefakten Embedder aus conftest.py - kein Modell,
kein Netz (conventions.md). Sie pruefen die Mechanik: Praefixe, Metadaten,
Ablage, Score-Bereich und die Mandantentrennung.

Was sie NICHT pruefen koennen, ist Semantik. Der Nachweis, dass eine deutsche
Frage ein englisches Dokument findet und dass Fremdmandanten-Merkmale niedrige
Scores bekommen, braucht das echte Modell und steht in
tests/test_integration_retrieval.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.config import Settings
from app.embeddings import PASSAGE_PREFIX, QUERY_PREFIX, E5Embeddings
from app.index_store import load_index
from app.ingest import ingest_tenant
from app.search import search_tenant
from tests.conftest import FAKE_DIMENSION, FakeBackend

ACME_TEXT = """# Retouren

Jede Ruecksendung braucht eine RMA-Nummer. Sie ist 21 Kalendertage gueltig.
Ohne RMA-Nummer nehmen wir keine Ruecksendung an.
"""

NORDWIND_TEXT = """# Lieferung

Wir liefern als Zwei-Mann-Montage in einem Terminfenster von vier Stunden.
Die Monteure bauen die Moebel im Aufstellraum auf.
"""


def _lege_mandant_an(root: Path, slug: str, text: str, token: str) -> None:
    tenant_dir = root / slug
    (tenant_dir / "docs").mkdir(parents=True)
    (tenant_dir / "tenant.yaml").write_text(
        yaml.safe_dump(
            {
                "display_name": slug,
                "languages": ["de"],
                "escalation_message": "Dazu finde ich nichts.",
                "url_token": token,
                "public_image_allowed": True,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (tenant_dir / "docs" / "doku.md").write_text(text, encoding="utf-8")


@pytest.fixture
def zwei_mandanten(tmp_path: Path) -> tuple[Settings, E5Embeddings, FakeBackend]:
    """Zwei Mandanten mit unterscheidbaren Inhalten, ingestiert."""
    tenants_root = tmp_path / "tenants"
    tenants_root.mkdir()
    _lege_mandant_an(tenants_root, "demo-acme", ACME_TEXT, "acme-token-1234567890")
    _lege_mandant_an(tenants_root, "demo-nordwind", NORDWIND_TEXT, "nordwind-token-123456")

    settings = Settings(
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=tenants_root,
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
        chunk_size=200,
        chunk_overlap=20,
    )
    backend = FakeBackend()
    embeddings = E5Embeddings(backend, expected_dimension=FAKE_DIMENSION, max_seq_length=512)

    for slug in ("demo-acme", "demo-nordwind"):
        ingest_tenant(slug, settings=settings, embeddings=embeddings)

    return settings, embeddings, backend


# --- Ingest ----------------------------------------------------------------


def test_ingest_erzeugt_chunks(zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend]) -> None:
    settings, embeddings, _ = zwei_mandanten
    report = ingest_tenant("demo-acme", settings=settings, embeddings=embeddings)

    assert report.files == 1
    assert report.chunks >= 1
    assert report.index_bytes > 0
    assert report.duration_seconds >= 0


def test_page_content_traegt_kein_praefix(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    """Das Praefix lebt im Wrapper, nicht im gespeicherten Text (ADR-009).

    Stuende es im Chunk-Text, landete es in jedem LLM-Kontext und in jeder
    Quellenanzeige - der Interessent saehe 'passage: ' in den Belegstellen
    seiner eigenen Inhalte.
    """
    settings, _, _ = zwei_mandanten
    geladen = load_index("demo-acme", settings.index_dir)

    assert geladen.chunks
    for chunk in geladen.chunks:
        assert not chunk.text.startswith(PASSAGE_PREFIX)
        assert PASSAGE_PREFIX not in chunk.text


def test_wrapper_setzt_praefixe_tatsaechlich(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    """Gegenprobe zum vorigen Test.

    Ohne sie waere 'kein Praefix im Text' auch dann erfuellt, wenn das Praefix
    gar nicht gesetzt wuerde - conventions.md verlangt zu jeder Negativaussage
    eine Gegenprobe.
    """
    _, _, backend = zwei_mandanten

    assert any(t.startswith(PASSAGE_PREFIX) for t in backend.seen)


def test_tenant_slug_in_allen_metadaten(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    settings, _, _ = zwei_mandanten
    for slug in ("demo-acme", "demo-nordwind"):
        geladen = load_index(slug, settings.index_dir)
        assert geladen.chunks
        assert all(c.tenant_slug == slug for c in geladen.chunks)


def test_source_file_und_chunk_index_gesetzt(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    settings, _, _ = zwei_mandanten
    geladen = load_index("demo-acme", settings.index_dir)

    assert all(c.source_file == "doku.md" for c in geladen.chunks)
    assert [c.chunk_index for c in geladen.chunks] == list(range(len(geladen.chunks)))


def test_mandant_ohne_dokumente_bricht_ab(tmp_path: Path) -> None:
    """Ein leerer Index wuerde jede Frage eskalieren lassen - lieber laut."""
    tenants_root = tmp_path / "tenants"
    tenants_root.mkdir()
    _lege_mandant_an(tenants_root, "demo-leer", "", "leer-token-1234567890")
    (tenants_root / "demo-leer" / "docs" / "doku.md").unlink()

    settings = Settings(
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=tenants_root,
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
    )
    embeddings = E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION)

    with pytest.raises(ValueError, match="keine Dokumente"):
        ingest_tenant("demo-leer", settings=settings, embeddings=embeddings)


# --- Tokengrenze (ADR-017) -------------------------------------------------


class TokenBackend(FakeBackend):
    """Fake mit Tokenizer, damit der Zaehler ueberhaupt zaehlen kann."""

    class _Tokenizer:
        @staticmethod
        def encode(text: str) -> list[int]:
            return [0] * len(text.split())

    def __init__(self) -> None:
        super().__init__()
        self.client = type("Client", (), {"tokenizer": TokenBackend._Tokenizer()})()


def test_tokenzaehler_schlaegt_bei_zu_langem_chunk_an(tmp_path: Path) -> None:
    """Positivtest fuer den Zaehler aus ADR-017.

    Bei 800 Zeichen wird er im Betrieb praktisch nie anschlagen. Ohne diesen
    Test waere er ein totes Muster wie das PEM-Muster aus P-010 - vorhanden,
    nie ausgeloest, niemand merkt es.
    """
    tenants_root = tmp_path / "tenants"
    tenants_root.mkdir()
    langer_text = " ".join(f"wort{i}" for i in range(400))
    _lege_mandant_an(tenants_root, "demo-lang", langer_text, "lang-token-1234567890")

    settings = Settings(
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=tenants_root,
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
        chunk_size=100_000,
        chunk_overlap=0,
    )
    embeddings = E5Embeddings(TokenBackend(), expected_dimension=FAKE_DIMENSION, max_seq_length=50)

    report = ingest_tenant("demo-lang", settings=settings, embeddings=embeddings)

    assert report.max_seq_length == 50
    assert report.chunks_at_token_limit >= 1
    assert "gekuerzt" in report.summary()


def test_ohne_tokenizer_wird_nicht_gezaehlt(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    """None heisst 'nicht geprueft', nicht 'null Treffer'."""
    settings, embeddings, _ = zwei_mandanten
    report = ingest_tenant("demo-acme", settings=settings, embeddings=embeddings)

    assert report.max_seq_length is None
    assert "nicht geprueft" in report.summary()


# --- Suche -----------------------------------------------------------------


def test_suche_liefert_treffer(zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend]) -> None:
    settings, embeddings, _ = zwei_mandanten
    treffer = search_tenant(
        "demo-acme", "RMA-Nummer", k=3, settings=settings, embeddings=embeddings
    )

    assert treffer
    assert all(t.text for t in treffer)


def test_query_praefix_wird_bei_der_suche_gesetzt(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    settings, embeddings, backend = zwei_mandanten
    backend.seen.clear()
    search_tenant("demo-acme", "RMA-Nummer", k=1, settings=settings, embeddings=embeddings)

    assert backend.seen == [QUERY_PREFIX + "RMA-Nummer"]


def test_scores_liegen_im_intervall(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    """ADR-008: relevance_score in 0..1, groesser ist besser."""
    settings, embeddings, _ = zwei_mandanten
    treffer = search_tenant("demo-acme", "RMA", k=5, settings=settings, embeddings=embeddings)

    assert all(0.0 <= t.score <= 1.0 for t in treffer)


def test_treffer_sind_absteigend_sortiert(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    """Faengt die Inversion: groesser muss besser heissen, nicht umgekehrt."""
    settings, embeddings, _ = zwei_mandanten
    treffer = search_tenant("demo-acme", "RMA", k=5, settings=settings, embeddings=embeddings)

    scores = [t.score for t in treffer]
    assert scores == sorted(scores, reverse=True)


def test_k_begrenzt_die_trefferzahl(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    settings, embeddings, _ = zwei_mandanten
    treffer = search_tenant("demo-acme", "RMA", k=1, settings=settings, embeddings=embeddings)

    assert len(treffer) == 1


# --- Mandantentrennung (ADR-001) -------------------------------------------


def test_suche_liefert_nur_eigene_chunks(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    """Der Kern von ADR-001, auf der Schicht, auf der ein Leck entstuende.

    Der gepruefte tenant_slug stammt aus dem Sidecar, nicht aus dem Parameter -
    er ist damit ein Beleg und keine Wiederholung der Eingabe.
    """
    settings, embeddings, _ = zwei_mandanten

    for slug, fremd in (("demo-acme", "demo-nordwind"), ("demo-nordwind", "demo-acme")):
        treffer = search_tenant(
            slug, "Lieferung Montage RMA", k=10, settings=settings, embeddings=embeddings
        )
        assert treffer
        assert all(t.tenant_slug == slug for t in treffer)
        assert not any(t.tenant_slug == fremd for t in treffer)


def test_fremdes_merkmal_liefert_keinen_fremden_treffer(
    zwei_mandanten: tuple[Settings, E5Embeddings, FakeBackend],
) -> None:
    """Frage nach dem Merkmal des anderen Mandanten, in beide Richtungen."""
    settings, embeddings, _ = zwei_mandanten

    treffer = search_tenant(
        "demo-acme",
        "Zwei-Mann-Montage Terminfenster",
        k=10,
        settings=settings,
        embeddings=embeddings,
    )
    assert all(t.tenant_slug == "demo-acme" for t in treffer)

    treffer = search_tenant(
        "demo-nordwind", "RMA-Nummer Ruecksendung", k=10, settings=settings, embeddings=embeddings
    )
    assert all(t.tenant_slug == "demo-nordwind" for t in treffer)
