"""Ingestion je Mandant.

Liest die Markdown-Dokumente eines Mandanten, chunkt sie, bettet sie ein und
schreibt einen FAISS-Index nach `data/index/<tenant_id>/`.

`tenant_id` ist ueberall Pflichtparameter an erster Position, und es gibt kein
Index- oder Embedding-Objekt auf Modulebene (ADR-001). Jeder Lauf loest seinen
Mandanten selbst auf.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.embeddings import E5Embeddings, build_embeddings
from app.index_store import ChunkRecord, index_size_bytes, write_index
from app.tenants import list_tenants, load_tenant

DOCS_SUBDIR = "docs"
DOC_PATTERN = "*.md"


@dataclass(frozen=True)
class IngestReport:
    """Ergebnis eines Ingest-Laufs fuer genau einen Mandanten."""

    tenant_id: str
    files: int
    chunks: int
    duration_seconds: float
    index_bytes: int

    # Chunks, deren Tokenlaenge inklusive "passage: "-Praefix die Grenze des
    # Modells erreicht. Oberhalb kuerzt sentence-transformers stillschweigend:
    # Das Chunk-Ende verschwindet aus dem Vektor, bleibt aber im Sidecar und
    # damit im LLM-Kontext. Kein Abbruch - sichtbar machen (ADR-017).
    chunks_at_token_limit: int

    # None, wenn kein Tokenizer erreichbar war. Dann wurde nicht gezaehlt, und
    # das ist etwas anderes als "nichts gefunden".
    max_seq_length: int | None

    def summary(self) -> str:
        zeilen = [
            f"Mandant           {self.tenant_id}",
            f"Dateien           {self.files}",
            f"Chunks            {self.chunks}",
            f"Dauer             {self.duration_seconds:.2f} s",
            f"Index auf Platte  {self.index_bytes / 1024:.1f} KiB",
        ]
        if self.max_seq_length is None:
            zeilen.append("Tokengrenze       nicht ermittelbar, nicht geprueft")
        elif self.chunks_at_token_limit:
            zeilen.append(
                f"Tokengrenze       {self.chunks_at_token_limit} von {self.chunks} Chunks "
                f"erreichen {self.max_seq_length} Token und werden beim Einbetten "
                f"gekuerzt"
            )
        else:
            zeilen.append(f"Tokengrenze       kein Chunk erreicht {self.max_seq_length} Token")
        return "\n".join(zeilen)


def _read_documents(tenant_id: str, tenants_dir: Path) -> list[tuple[str, str]]:
    """Liest die Markdown-Dokumente eines Mandanten als (Dateiname, Text)."""
    docs_dir = Path(tenants_dir) / tenant_id / DOCS_SUBDIR
    if not docs_dir.is_dir():
        return []
    return [
        (p.name, p.read_text(encoding="utf-8"))
        for p in sorted(docs_dir.glob(DOC_PATTERN))
        if p.is_file()
    ]


def _split_documents(
    tenant_id: str,
    documents: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    """Zerlegt die Dokumente in Chunks mit Herkunftsangabe.

    Die Praefixe aus ADR-009 werden hier NICHT gesetzt. Der gespeicherte Text
    ist der Inhalt des Dokuments, sonst nichts - das Praefix lebt im
    Embedding-Wrapper.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    records: list[ChunkRecord] = []
    for source_file, text in documents:
        for i, stueck in enumerate(splitter.split_text(text)):
            records.append(
                ChunkRecord(
                    text=stueck,
                    source_file=source_file,
                    chunk_index=i,
                    tenant_slug=tenant_id,
                )
            )
    return records


def _count_chunks_at_token_limit(
    embeddings: E5Embeddings, chunks: list[ChunkRecord]
) -> tuple[int, int | None]:
    """Zaehlt Chunks, die inklusive Praefix an die Tokengrenze stossen."""
    grenze = embeddings.max_seq_length
    if grenze is None:
        return 0, None

    from app.embeddings import PASSAGE_PREFIX

    treffer = 0
    for chunk in chunks:
        laenge = embeddings.count_tokens(PASSAGE_PREFIX + chunk.text)
        if laenge is None:
            return 0, None
        if laenge >= grenze:
            treffer += 1
    return treffer, grenze


def ingest_tenant(
    tenant_id: str,
    settings: Settings | None = None,
    embeddings: E5Embeddings | None = None,
) -> IngestReport:
    """Baut den Index eines Mandanten neu.

    `tenant_id` ist Pflichtparameter ohne Default (ADR-001). `settings` und
    `embeddings` sind injizierbar, damit Tests ohne echtes Modell und ohne
    vollstaendige Konfiguration auskommen - beides hat keinen Mandantenbezug.
    """
    aktive_settings = settings if settings is not None else get_settings()

    # Wirft TenantNotFound bzw. InvalidTenantSlug, bevor irgendetwas geschrieben
    # wird. Ein Index fuer einen Mandanten, den es nicht gibt, darf nicht
    # entstehen.
    tenant = load_tenant(tenant_id, aktive_settings.tenants_dir)

    start = time.perf_counter()
    documents = _read_documents(tenant.slug, aktive_settings.tenants_dir)
    chunks = _split_documents(
        tenant.slug,
        documents,
        chunk_size=aktive_settings.chunk_size,
        chunk_overlap=aktive_settings.chunk_overlap,
    )

    if not chunks:
        raise ValueError(
            f"Mandant {tenant_id!r} hat keine Dokumente unter "
            f"{Path(aktive_settings.tenants_dir) / tenant_id / DOCS_SUBDIR}. "
            f"Ein leerer Index wuerde jede Frage eskalieren lassen."
        )

    aktive_embeddings = embeddings if embeddings is not None else build_embeddings(aktive_settings)

    am_limit, grenze = _count_chunks_at_token_limit(aktive_embeddings, chunks)
    vektoren = np.array(
        aktive_embeddings.embed_documents([c.text for c in chunks]),
        dtype=np.float32,
    )

    write_index(
        tenant_id=tenant.slug,
        index_dir=aktive_settings.index_dir,
        vectors=vektoren,
        chunks=chunks,
        model_name=aktive_settings.embedding_model,
        dimension=aktive_embeddings.dimension,
    )

    return IngestReport(
        tenant_id=tenant.slug,
        files=len(documents),
        chunks=len(chunks),
        duration_seconds=time.perf_counter() - start,
        index_bytes=index_size_bytes(tenant.slug, aktive_settings.index_dir),
        chunks_at_token_limit=am_limit,
        max_seq_length=grenze,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingest",
        description="Baut den Vektorindex eines Mandanten neu.",
    )
    gruppe = parser.add_mutually_exclusive_group(required=True)
    gruppe.add_argument("tenant_id", nargs="?", help="Slug des Mandanten")
    gruppe.add_argument("--all", action="store_true", help="alle Mandanten nacheinander")
    args = parser.parse_args(argv)

    settings = get_settings()
    slugs = list_tenants(settings.tenants_dir) if args.all else [args.tenant_id]

    if not slugs:
        print("Keine Mandanten gefunden.", file=sys.stderr)
        return 1

    # Das Modell einmal laden und wiederverwenden. Kein Modul-Level-Objekt: Es
    # entsteht hier, im Aufruf, und lebt nicht laenger als dieser Lauf.
    embeddings = build_embeddings(settings)

    for slug in slugs:
        report = ingest_tenant(slug, settings=settings, embeddings=embeddings)
        print(report.summary())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
