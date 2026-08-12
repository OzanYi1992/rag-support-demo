"""Embedding-Kapselung.

Einzige Stelle, an der die e5-Praefixe gesetzt werden (ADR-009), und einzige
Stelle, an der das Embedding-Modell geladen wird.

Drei Eigenschaften des Modells werden hier gegen die Konfiguration geprueft,
statt sie zu glauben:

  Dimension       muss zu settings.embedding_dimension passen, sonst ist jeder
                  gebaute Index unbrauchbar (ADR-016).
  Normalisierung  muss aktiv sein, sonst ist das Skalarprodukt kein Kosinus und
                  die Score-Formel aus ADR-008 verlaesst 0..1.
  max_seq_length  wird vom Modell gelesen, nicht hart gesetzt (ADR-017).

Alle drei scheitern sonst still: Es kommen Zahlen heraus, die plausibel
aussehen. Genau deshalb wird geprueft und nicht angenommen.
"""

from __future__ import annotations

import math
from typing import Protocol

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import Settings

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

# Toleranz fuer die Normpruefung. Float32 und die Normalisierung im Modell
# liefern nicht exakt 1.0; 1e-3 ist weit genug fuer Rundungsfehler und eng
# genug, um einen unnormalisierten Vektor sicher zu fangen (dessen Norm liegt
# bei e5 typischerweise im zweistelligen Bereich).
NORM_TOLERANCE = 1e-3


class EmbeddingBackend(Protocol):
    """Was der Wrapper von einem Backend braucht.

    Als Protocol formuliert, damit Tests einen gefakten Embedder einsetzen
    koennen, ohne 470 MB Modellgewichte zu laden - conventions.md verbietet
    Netzwerkzugriff in Unit-Tests.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class EmbeddingConfigError(RuntimeError):
    """Modell und Konfiguration passen nicht zusammen."""


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


class E5Embeddings:
    """Setzt die e5-Praefixe und prueft die Voraussetzungen der Score-Formel.

    Die Praefixe stehen NICHT im gespeicherten Chunk-Text. Sie werden hier
    unmittelbar vor dem Aufruf des Modells vorangestellt und danach nie wieder
    sichtbar (ADR-009). `page_content` bleibt damit das, was es sein soll: der
    Inhalt des Dokuments - auch im LLM-Kontext und in der Quellenanzeige.
    """

    def __init__(
        self,
        backend: EmbeddingBackend,
        expected_dimension: int,
        max_seq_length: int | None = None,
    ) -> None:
        self._backend = backend
        self._expected_dimension = expected_dimension
        self._max_seq_length = max_seq_length
        self._checked = False

    @property
    def max_seq_length(self) -> int | None:
        """Tokengrenze des Modells, oder None wenn sie nicht ermittelbar war."""
        return self._max_seq_length

    @property
    def dimension(self) -> int:
        return self._expected_dimension

    def _check_vector(self, vector: list[float]) -> None:
        """Prueft Dimension und Norm des ersten Vektors, danach nie wieder.

        Beides sind Eigenschaften des Modells, nicht der Eingabe - eine Probe
        genuegt. Der Aufwand pro Aufruf waere sonst umsonst bezahlt.
        """
        if self._checked:
            return

        if len(vector) != self._expected_dimension:
            raise EmbeddingConfigError(
                f"Das Modell liefert {len(vector)}-dimensionale Vektoren, erwartet waren "
                f"{self._expected_dimension} (EMBEDDING_DIMENSION). Ein Index mit falscher "
                f"Dimension ist unbrauchbar, nicht nur schlechter. Siehe ADR-016."
            )

        norm = _vector_norm(vector)
        if abs(norm - 1.0) > NORM_TOLERANCE:
            raise EmbeddingConfigError(
                f"Die Vektoren sind nicht normalisiert (Norm {norm:.4f}, erwartet 1.0). "
                f"Ohne normalisierte Vektoren ist das Skalarprodukt kein Kosinus, und die "
                f"Score-Formel (1 + cos) / 2 verlaesst den Bereich 0..1 - der "
                f"Schwellwertvergleich aus ADR-003 wird damit sinnlos, ohne dass etwas "
                f"abstuerzt. Pruefe normalize_embeddings=True. Siehe ADR-008."
            )

        self._checked = True

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Bettet Passagen ein, mit vorangestelltem `passage: `."""
        vectors = self._backend.embed_documents([PASSAGE_PREFIX + t for t in texts])
        if vectors:
            self._check_vector(vectors[0])
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Bettet eine Suchanfrage ein, mit vorangestelltem `query: `."""
        vector = self._backend.embed_query(QUERY_PREFIX + text)
        self._check_vector(vector)
        return vector

    def count_tokens(self, text: str) -> int | None:
        """Tokenlaenge eines Textes, oder None wenn kein Tokenizer erreichbar ist.

        Wird vom Ingest gebraucht, um Chunks zu zaehlen, die an die Tokengrenze
        stossen (ADR-017). Oberhalb der Grenze kuerzt sentence-transformers
        stillschweigend - das Chunk-Ende verschwindet dann aus dem Vektor,
        bleibt aber im Sidecar und im LLM-Kontext.
        """
        tokenizer = _find_tokenizer(self._backend)
        if tokenizer is None:
            return None
        return len(tokenizer.encode(text))


def _sentence_transformer(backend: object) -> object | None:
    """Holt das sentence-transformers-Modell hinter dem Backend.

    In langchain-huggingface 1.2.2 liegt es unter `_client`. Der fuehrende
    Unterstrich ist ein Implementierungsdetail und kann sich aendern, deshalb
    wird auch das oeffentliche `client` geprueft.

    Bewusst defensiv: Findet sich nichts, entfallen Tokenzaehlung und
    Tokengrenze. Der Ingest laeuft dann weiter und meldet None - "nicht
    geprueft" ist etwas anderes als "nichts gefunden".
    """
    for name in ("_client", "client"):
        modell = getattr(backend, name, None)
        if modell is not None:
            return modell
    return None


def _find_tokenizer(backend: object) -> object | None:
    """Sucht den Tokenizer im sentence-transformers-Modell hinter dem Backend."""
    modell = _sentence_transformer(backend)
    if modell is None:
        return None
    tokenizer = getattr(modell, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return tokenizer
    return None


def _read_max_seq_length(backend: object) -> int | None:
    """Liest max_seq_length vom geladenen Modell.

    Nicht hart 512 setzen: Der Wert gehoert zum Modell, und ADR-016 laesst einen
    Modellwechsel ausdruecklich offen. Ein hart gesetzter Wert waere beim
    naechsten Wechsel still falsch.
    """
    modell = _sentence_transformer(backend)
    if modell is None:
        return None
    value = getattr(modell, "max_seq_length", None)
    return int(value) if isinstance(value, int) else None


def build_embeddings(settings: Settings) -> E5Embeddings:
    """Baut die Embedding-Kapselung aus der Konfiguration.

    Bewusst eine Funktion und kein Modul-Level-Objekt: Ein beim Import geladenes
    Modell wuerde jeden Import dieser Datei an einen 470-MB-Download binden.

    Nicht mandantenabhaengig - das Modell ist fuer alle Mandanten dasselbe. Der
    Index ist es nicht, siehe app/index_store.py.
    """
    backend = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        # normalize_embeddings ist keine Bequemlichkeit, sondern Voraussetzung
        # der Score-Formel aus ADR-008. Der Wrapper prueft es zusaetzlich zur
        # Laufzeit, weil ein stiller Wegfall hier nicht auffiele.
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.embedding_batch_size,
        },
    )
    return E5Embeddings(
        backend=backend,
        expected_dimension=settings.embedding_dimension,
        max_seq_length=_read_max_seq_length(backend),
    )
