"""Tests der Embedding-Kapselung.

Schwerpunkt sind die drei Pruefungen, die still scheitern wuerden: fehlende
Praefixe, falsche Dimension, fehlende Normalisierung. Alle drei liefern Zahlen,
die plausibel aussehen.
"""

from __future__ import annotations

import pytest

from app.embeddings import (
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    E5Embeddings,
    EmbeddingConfigError,
)
from tests.conftest import FAKE_DIMENSION, FakeBackend

# --- Praefixe (ADR-009) ----------------------------------------------------


def test_passage_praefix_wird_gesetzt(fake_backend: FakeBackend) -> None:
    embeddings = E5Embeddings(fake_backend, expected_dimension=FAKE_DIMENSION)
    embeddings.embed_documents(["Der Versand dauert zwei Werktage."])

    assert fake_backend.seen == [PASSAGE_PREFIX + "Der Versand dauert zwei Werktage."]


def test_query_praefix_wird_gesetzt(fake_backend: FakeBackend) -> None:
    embeddings = E5Embeddings(fake_backend, expected_dimension=FAKE_DIMENSION)
    embeddings.embed_query("Wie lange dauert der Versand?")

    assert fake_backend.seen == [QUERY_PREFIX + "Wie lange dauert der Versand?"]


def test_praefixe_sind_verschieden(fake_backend: FakeBackend) -> None:
    """Asymmetrisches Vergessen ist der gefaehrlichste Fall (P-001).

    Waeren beide Praefixe gleich, faellt der Fehler beim Testen nicht auf,
    verschiebt aber die Score-Verteilung und macht den Schwellwert ungueltig.
    """
    assert QUERY_PREFIX != PASSAGE_PREFIX


# --- Dimensionspruefung (ADR-016) ------------------------------------------


def test_falsche_dimension_bricht_ab() -> None:
    backend = FakeBackend(dimension=8)
    embeddings = E5Embeddings(backend, expected_dimension=FAKE_DIMENSION)

    with pytest.raises(EmbeddingConfigError) as excinfo:
        embeddings.embed_documents(["irgendein Text"])

    assert "8" in str(excinfo.value)
    assert str(FAKE_DIMENSION) in str(excinfo.value)


def test_passende_dimension_laeuft_durch(fake_backend: FakeBackend) -> None:
    embeddings = E5Embeddings(fake_backend, expected_dimension=FAKE_DIMENSION)
    vektoren = embeddings.embed_documents(["Text"])

    assert len(vektoren[0]) == FAKE_DIMENSION


# --- Normpruefung (ADR-008) ------------------------------------------------


def test_nicht_normalisierte_vektoren_brechen_ab() -> None:
    """Positivtest fuer die Normpruefung.

    Ohne diesen Test waere die Pruefung ein totes Muster wie das PEM-Muster aus
    P-010: syntaktisch vorhanden, nie ausgeloest, niemand merkt es.

    Fehlt die Normalisierung, ist das Skalarprodukt kein Kosinus mehr. Die
    Score-Formel (1 + cos) / 2 verlaesst 0..1, und der Schwellwertvergleich aus
    ADR-003 wird sinnlos - ohne Absturz.
    """
    backend = FakeBackend(normalize=False)
    embeddings = E5Embeddings(backend, expected_dimension=FAKE_DIMENSION)

    with pytest.raises(EmbeddingConfigError) as excinfo:
        embeddings.embed_documents(["mehrere verschiedene Woerter hier drin"])

    meldung = str(excinfo.value)
    assert "normalisiert" in meldung
    assert "ADR-008" in meldung


def test_normalisierte_vektoren_laufen_durch(fake_backend: FakeBackend) -> None:
    embeddings = E5Embeddings(fake_backend, expected_dimension=FAKE_DIMENSION)
    embeddings.embed_documents(["mehrere verschiedene Woerter hier drin"])
    embeddings.embed_query("noch eine Frage")


def test_pruefung_laeuft_auch_ueber_embed_query() -> None:
    """Die Pruefung darf nicht nur am Ingest-Pfad haengen."""
    backend = FakeBackend(normalize=False)
    embeddings = E5Embeddings(backend, expected_dimension=FAKE_DIMENSION)

    with pytest.raises(EmbeddingConfigError):
        embeddings.embed_query("mehrere verschiedene Woerter hier drin")


# --- max_seq_length (ADR-017) ----------------------------------------------


def test_max_seq_length_wird_durchgereicht(fake_backend: FakeBackend) -> None:
    embeddings = E5Embeddings(fake_backend, expected_dimension=FAKE_DIMENSION, max_seq_length=512)

    assert embeddings.max_seq_length == 512


def test_ohne_max_seq_length_ist_es_none(fake_backend: FakeBackend) -> None:
    """None heisst 'nicht ermittelt', nicht 'keine Grenze'.

    Der Unterschied ist wichtig: Der Ingest meldet dann, dass nicht geprueft
    wurde, statt 'null Chunks am Limit' zu behaupten.
    """
    embeddings = E5Embeddings(fake_backend, expected_dimension=FAKE_DIMENSION)

    assert embeddings.max_seq_length is None
    assert embeddings.count_tokens("egal") is None
