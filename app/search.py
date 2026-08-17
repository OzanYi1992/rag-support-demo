"""Suche im Index eines Mandanten.

Reine Suche: kein Schwellwert, keine Eskalation, kein LLM. Das kommt in Phase 3.

Der Zweck dieser Schicht in Phase 2 ist Nachweisbarkeit. Ein Index, den niemand
abgefragt hat, ist unverifiziert - eine Ingestion, die nur schreibt, kann nicht
zeigen, dass sie richtig geschrieben hat.

Hier entsteht auch die Mandantengrenze, an der ein Leck tatsaechlich entstuende:
beim Aufloesen des Mandanten und beim Laden seines Index.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings, get_settings
from app.embeddings import E5Embeddings, get_embeddings
from app.index_store import load_index


@dataclass(frozen=True)
class SearchHit:
    """Ein Treffer mit normalisiertem Score und Herkunft."""

    text: str
    # relevance_score in 0..1, groesser ist besser (ADR-008). Rohe Distanzen
    # verlassen die Kapselung nicht.
    score: float
    source_file: str
    chunk_index: int
    tenant_slug: str


def _to_relevance_score(inner_product: float) -> float:
    """Rechnet das Skalarprodukt in einen relevance_score in 0..1 um.

    Auf normalisierten Vektoren ist das Skalarprodukt der Kosinus und liegt in
    [-1, 1]. `(1 + cos) / 2` bildet das auf [0, 1] ab, groesser ist besser.

    Die Normalisierung ist Voraussetzung, nicht Annahme: Der Embedding-Wrapper
    prueft die Vektornorm und bricht ab, wenn sie fehlt (ADR-008). Ohne diese
    Pruefung waere der Wertebereich hier still falsch.

    Das Klemmen faengt Rundungsfehler in der letzten Stelle ab, nicht mehr. Ein
    grob abweichender Wert kaeme gar nicht bis hierher.
    """
    return max(0.0, min(1.0, (1.0 + inner_product) / 2.0))


def search_tenant(
    tenant_id: str,
    query: str,
    k: int | None = None,
    settings: Settings | None = None,
    embeddings: E5Embeddings | None = None,
) -> list[SearchHit]:
    """Sucht im Index genau eines Mandanten.

    `tenant_id` ist Pflichtparameter an erster Position, ohne Default und ohne
    Optional (ADR-001). Der Index wird pro Aufruf aus der `tenant_id` aufgeloest
    und lebt nicht laenger als der Aufruf - es gibt kein geteiltes Store-Objekt,
    ueber das Mandant B die Chunks von Mandant A bekaeme.

    Der zurueckgegebene `tenant_slug` stammt aus dem Sidecar, nicht aus dem
    Parameter. Damit ist er ein Beleg und keine Wiederholung der Eingabe: Ein
    Treffer mit fremdem Slug faellt auf, statt sich zu tarnen.
    """
    aktive_settings = settings if settings is not None else get_settings()
    treffer_anzahl = k if k is not None else aktive_settings.retrieval_top_k

    geladen = load_index(tenant_id, Path(aktive_settings.index_dir))

    # get_embeddings statt build_embeddings: der Embedder wird prozessweit
    # geteilt (ADR-020). Ohne das laed jede Anfrage 0,47 GB neu.
    aktive_embeddings = embeddings if embeddings is not None else get_embeddings(aktive_settings)
    frage_vektor = np.array([aktive_embeddings.embed_query(query)], dtype=np.float32)

    # FAISS liefert weniger Treffer als angefragt, wenn der Index kleiner ist;
    # fehlende Plaetze kommen als Index -1 zurueck.
    scores, positionen = geladen.index.search(frage_vektor, treffer_anzahl)

    hits: list[SearchHit] = []
    for score, position in zip(scores[0], positionen[0], strict=True):
        if position < 0:
            continue
        chunk = geladen.chunks[int(position)]
        hits.append(
            SearchHit(
                text=chunk.text,
                score=_to_relevance_score(float(score)),
                source_file=chunk.source_file,
                chunk_index=chunk.chunk_index,
                tenant_slug=chunk.tenant_slug,
            )
        )
    return hits
