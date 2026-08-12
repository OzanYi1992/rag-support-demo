"""Persistenz des Vektorindex, ein Index je Mandant.

FAISS wird direkt angebunden, ohne LangChain-Vectorstore (ADR-015). Der Index
liegt als `index.faiss`, die Chunk-Texte und Metadaten daneben als
`sidecar.json`. Damit gibt es kein Pickle und keinen Schalter, der
Pickle-Deserialisierung erlaubt.

Zwei Eigenschaften, die hier durchgesetzt werden:

  Atomares Schreiben   Beide Dateien entstehen in einem temporaeren Verzeichnis
                       und werden erst am Ende an ihren Platz geschoben. Ein
                       Abbruch dazwischen hinterlaesst keinen halben Zustand.
  Konsistenzpruefung   Index und Sidecar tragen dieselbe Kennung. Beim Laden
                       wird sie geprueft, Abweichung ist ein lauter Fehler.

Beides adressiert denselben Fehler: ein Index, der zu seinem Sidecar nicht
passt, liefert Treffer mit falschem Text - plausibel und falsch.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

# Wird mitgeschrieben und beim Laden geprueft. Aendert sich das Ablageformat,
# wird die Zahl erhoeht - dann scheitert ein alter Index laut statt still.
FORMAT_VERSION = 1

INDEX_FILE = "index.faiss"
SIDECAR_FILE = "sidecar.json"


class IndexNotFound(FileNotFoundError):
    """Fuer diesen Mandanten existiert kein Index."""


class IndexInconsistent(RuntimeError):
    """Index und Sidecar passen nicht zusammen.

    Faellt an, wenn eine der beiden Dateien fehlt, wenn die Kennungen
    auseinanderlaufen oder wenn die Vektorzahl im Index nicht zur Chunk-Zahl im
    Sidecar passt. Immer ein Abbruch, nie eine Reparatur: Was hier nicht passt,
    liefert sonst Treffer mit dem Text eines anderen Chunks.
    """


@dataclass(frozen=True)
class ChunkRecord:
    """Ein Chunk, so wie er im Sidecar liegt."""

    text: str
    source_file: str
    chunk_index: int
    tenant_slug: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "tenant_slug": self.tenant_slug,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ChunkRecord:
        return ChunkRecord(
            text=data["text"],
            source_file=data["source_file"],
            chunk_index=int(data["chunk_index"]),
            tenant_slug=data["tenant_slug"],
        )


@dataclass(frozen=True)
class LoadedIndex:
    """Ein geladener Mandantenindex mit seinen Chunks."""

    tenant_id: str
    index: faiss.Index
    chunks: list[ChunkRecord]
    model_name: str
    dimension: int


def index_path(tenant_id: str, index_dir: Path) -> Path:
    """Verzeichnis des Index eines Mandanten.

    `tenant_id` ist Pflichtparameter. Es gibt keinen Weg, hier ohne Mandant
    hineinzukommen (ADR-001).
    """
    return Path(index_dir) / tenant_id


def write_index(
    tenant_id: str,
    index_dir: Path,
    vectors: np.ndarray,
    chunks: list[ChunkRecord],
    model_name: str,
    dimension: int,
) -> Path:
    """Schreibt Index und Sidecar atomar.

    Zuerst entsteht alles in einem temporaeren Verzeichnis neben dem Ziel, dann
    wird umbenannt. Bricht der Vorgang vorher ab, bleibt der alte Zustand
    stehen - es gibt keinen Moment, in dem ein Index ohne passendes Sidecar
    sichtbar waere.

    Das temporaere Verzeichnis liegt bewusst neben dem Ziel und nicht in /tmp:
    os.replace ist nur innerhalb desselben Dateisystems atomar.
    """
    if len(chunks) != vectors.shape[0]:
        raise IndexInconsistent(
            f"{len(chunks)} Chunks, aber {vectors.shape[0]} Vektoren fuer Mandant "
            f"{tenant_id!r}. Das darf nicht geschrieben werden."
        )
    if vectors.shape[1] != dimension:
        raise IndexInconsistent(
            f"Vektoren haben Dimension {vectors.shape[1]}, erwartet {dimension} "
            f"fuer Mandant {tenant_id!r}."
        )

    ziel = index_path(tenant_id, index_dir)
    ziel.parent.mkdir(parents=True, exist_ok=True)

    # IndexFlatIP: Skalarprodukt. Auf normalisierten Vektoren ist das der
    # Kosinus - Voraussetzung der Score-Formel aus ADR-008. Der Embedding-
    # Wrapper prueft die Normalisierung zusaetzlich zur Laufzeit.
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(vectors.astype(np.float32))

    tmp_dir = Path(tempfile.mkdtemp(prefix=f".{tenant_id}-", dir=ziel.parent))
    try:
        faiss.write_index(faiss_index, str(tmp_dir / INDEX_FILE))
        sidecar = {
            "format_version": FORMAT_VERSION,
            "tenant_slug": tenant_id,
            "model_name": model_name,
            "dimension": dimension,
            "chunk_count": len(chunks),
            "chunks": [c.as_dict() for c in chunks],
        }
        (tmp_dir / SIDECAR_FILE).write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if ziel.exists():
            shutil.rmtree(ziel)
        os.replace(tmp_dir, ziel)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return ziel


def load_index(tenant_id: str, index_dir: Path) -> LoadedIndex:
    """Laedt Index und Sidecar eines Mandanten und prueft ihre Uebereinstimmung.

    Kein Modul-Level-Cache (ADR-001). Ein Cache waere nur mit `tenant_id` im
    Schluessel zulaessig und ist bislang nicht noetig.
    """
    verzeichnis = index_path(tenant_id, index_dir)
    index_datei = verzeichnis / INDEX_FILE
    sidecar_datei = verzeichnis / SIDECAR_FILE

    if not verzeichnis.is_dir():
        raise IndexNotFound(
            f"Kein Index fuer Mandant {tenant_id!r} unter {verzeichnis}. "
            f"Erst 'python -m app.ingest {tenant_id}' ausfuehren."
        )

    # Fehlt eine der beiden Dateien, ist die Ablage kaputt - nicht leer. Der
    # Unterschied ist wichtig: Ein leerer Index waere ein gueltiger Zustand,
    # eine halbe Ablage nicht.
    for datei in (index_datei, sidecar_datei):
        if not datei.is_file():
            raise IndexInconsistent(
                f"{datei.name} fehlt im Index von Mandant {tenant_id!r}. Index und "
                f"Sidecar gehoeren zusammen; eine halbe Ablage wird nicht geladen. "
                f"Neu aufbauen mit 'python -m app.ingest {tenant_id}'."
            )

    sidecar = json.loads(sidecar_datei.read_text(encoding="utf-8"))

    if sidecar.get("format_version") != FORMAT_VERSION:
        raise IndexInconsistent(
            f"Index von {tenant_id!r} hat Formatversion {sidecar.get('format_version')}, "
            f"erwartet {FORMAT_VERSION}. Neu aufbauen."
        )
    if sidecar.get("tenant_slug") != tenant_id:
        raise IndexInconsistent(
            f"Das Sidecar unter {verzeichnis} gehoert zu Mandant "
            f"{sidecar.get('tenant_slug')!r}, geladen wurde {tenant_id!r}. "
            f"Das waere ein Mandantenleck und wird abgewiesen."
        )

    faiss_index = faiss.read_index(str(index_datei))
    chunks = [ChunkRecord.from_dict(c) for c in sidecar["chunks"]]

    if len(chunks) != sidecar.get("chunk_count"):
        raise IndexInconsistent(
            f"Sidecar von {tenant_id!r} meldet {sidecar.get('chunk_count')} Chunks, "
            f"enthaelt aber {len(chunks)}."
        )
    if faiss_index.ntotal != len(chunks):
        raise IndexInconsistent(
            f"Index von {tenant_id!r} enthaelt {faiss_index.ntotal} Vektoren, das "
            f"Sidecar {len(chunks)} Chunks. Treffer wuerden auf den falschen Text "
            f"zeigen."
        )
    if faiss_index.d != sidecar.get("dimension"):
        raise IndexInconsistent(
            f"Index von {tenant_id!r} hat Dimension {faiss_index.d}, das Sidecar "
            f"meldet {sidecar.get('dimension')}."
        )

    return LoadedIndex(
        tenant_id=tenant_id,
        index=faiss_index,
        chunks=chunks,
        model_name=sidecar["model_name"],
        dimension=int(sidecar["dimension"]),
    )


def index_size_bytes(tenant_id: str, index_dir: Path) -> int:
    """Groesse der Ablage auf Platte, fuer den IngestReport."""
    verzeichnis = index_path(tenant_id, index_dir)
    if not verzeichnis.is_dir():
        return 0
    return sum(f.stat().st_size for f in verzeichnis.iterdir() if f.is_file())
