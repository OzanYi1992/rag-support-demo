"""Tests der Indexablage.

Schwerpunkt ist die Konsistenz zwischen Index und Sidecar. Laufen die beiden
auseinander, liefert die Suche Treffer, deren Text zu einem anderen Chunk
gehoert - plausibel und falsch, ohne Absturz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.index_store import (
    INDEX_FILE,
    SIDECAR_FILE,
    ChunkRecord,
    IndexInconsistent,
    IndexNotFound,
    index_path,
    load_index,
    write_index,
)

DIM = 4


def _vektoren(n: int) -> np.ndarray:
    """n normalisierte Vektoren, jeder auf einer eigenen Achse."""
    werte = np.zeros((n, DIM), dtype=np.float32)
    for i in range(n):
        werte[i][i % DIM] = 1.0
    return werte


def _chunks(n: int, tenant_id: str = "demo-acme") -> list[ChunkRecord]:
    return [
        ChunkRecord(
            text=f"Chunk Nummer {i}",
            source_file="doku.md",
            chunk_index=i,
            tenant_slug=tenant_id,
        )
        for i in range(n)
    ]


def _schreibe(tmp_path: Path, n: int = 3, tenant_id: str = "demo-acme") -> Path:
    write_index(
        tenant_id=tenant_id,
        index_dir=tmp_path,
        vectors=_vektoren(n),
        chunks=_chunks(n, tenant_id),
        model_name="fake-model",
        dimension=DIM,
    )
    return index_path(tenant_id, tmp_path)


# --- Schreiben und Laden ---------------------------------------------------


def test_schreiben_und_laden(tmp_path: Path) -> None:
    _schreibe(tmp_path, n=3)
    geladen = load_index("demo-acme", tmp_path)

    assert geladen.tenant_id == "demo-acme"
    assert geladen.index.ntotal == 3
    assert len(geladen.chunks) == 3
    assert geladen.dimension == DIM
    assert geladen.chunks[1].text == "Chunk Nummer 1"


def test_kein_index_wirft_not_found(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFound):
        load_index("gibt-es-nicht", tmp_path)


def test_ueberschreiben_ersetzt_vollstaendig(tmp_path: Path) -> None:
    """Ein zweiter Lauf darf keine Chunks des ersten uebrig lassen."""
    _schreibe(tmp_path, n=5)
    _schreibe(tmp_path, n=2)

    geladen = load_index("demo-acme", tmp_path)
    assert geladen.index.ntotal == 2
    assert len(geladen.chunks) == 2


# --- Atomares Schreiben ----------------------------------------------------


def test_kein_temporaerer_rest_nach_dem_schreiben(tmp_path: Path) -> None:
    """Nach dem Lauf liegt genau das Zielverzeichnis da, kein Zwischenstand."""
    _schreibe(tmp_path, n=3)

    eintraege = sorted(p.name for p in tmp_path.iterdir())
    assert eintraege == ["demo-acme"]


def test_schreiben_bricht_bei_ungleicher_anzahl_ab(tmp_path: Path) -> None:
    """Inkonsistenz wird beim Schreiben gefangen, nicht erst beim Laden."""
    with pytest.raises(IndexInconsistent):
        write_index(
            tenant_id="demo-acme",
            index_dir=tmp_path,
            vectors=_vektoren(3),
            chunks=_chunks(2),
            model_name="fake-model",
            dimension=DIM,
        )
    assert not index_path("demo-acme", tmp_path).exists()


# --- Konsistenzpruefung beim Laden -----------------------------------------


def test_fehlendes_sidecar_ist_ein_lauter_fehler(tmp_path: Path) -> None:
    """Ein Index ohne Sidecar ist kaputt, nicht leer.

    Genau der Zustand, den nicht-atomares Schreiben hinterlassen wuerde.
    """
    verzeichnis = _schreibe(tmp_path, n=3)
    (verzeichnis / SIDECAR_FILE).unlink()

    with pytest.raises(IndexInconsistent) as excinfo:
        load_index("demo-acme", tmp_path)
    assert SIDECAR_FILE in str(excinfo.value)


def test_fehlender_index_ist_ein_lauter_fehler(tmp_path: Path) -> None:
    verzeichnis = _schreibe(tmp_path, n=3)
    (verzeichnis / INDEX_FILE).unlink()

    with pytest.raises(IndexInconsistent):
        load_index("demo-acme", tmp_path)


def test_abweichende_chunk_zahl_wird_gefangen(tmp_path: Path) -> None:
    verzeichnis = _schreibe(tmp_path, n=3)
    sidecar_datei = verzeichnis / SIDECAR_FILE
    daten = json.loads(sidecar_datei.read_text(encoding="utf-8"))
    daten["chunks"] = daten["chunks"][:2]
    sidecar_datei.write_text(json.dumps(daten), encoding="utf-8")

    with pytest.raises(IndexInconsistent):
        load_index("demo-acme", tmp_path)


def test_abweichende_dimension_wird_gefangen(tmp_path: Path) -> None:
    verzeichnis = _schreibe(tmp_path, n=3)
    sidecar_datei = verzeichnis / SIDECAR_FILE
    daten = json.loads(sidecar_datei.read_text(encoding="utf-8"))
    daten["dimension"] = DIM + 1
    sidecar_datei.write_text(json.dumps(daten), encoding="utf-8")

    with pytest.raises(IndexInconsistent):
        load_index("demo-acme", tmp_path)


def test_abweichende_formatversion_wird_gefangen(tmp_path: Path) -> None:
    verzeichnis = _schreibe(tmp_path, n=3)
    sidecar_datei = verzeichnis / SIDECAR_FILE
    daten = json.loads(sidecar_datei.read_text(encoding="utf-8"))
    daten["format_version"] = 999
    sidecar_datei.write_text(json.dumps(daten), encoding="utf-8")

    with pytest.raises(IndexInconsistent):
        load_index("demo-acme", tmp_path)


def test_fremder_tenant_slug_im_sidecar_wird_abgewiesen(tmp_path: Path) -> None:
    """Das waere ein Mandantenleck und muss laut scheitern.

    Ein Sidecar, das zu einem anderen Mandanten gehoert, liefert dessen Chunks
    aus - ohne Fehlermeldung, wenn niemand hinsieht.
    """
    verzeichnis = _schreibe(tmp_path, n=3)
    sidecar_datei = verzeichnis / SIDECAR_FILE
    daten = json.loads(sidecar_datei.read_text(encoding="utf-8"))
    daten["tenant_slug"] = "demo-nordwind"
    sidecar_datei.write_text(json.dumps(daten), encoding="utf-8")

    with pytest.raises(IndexInconsistent) as excinfo:
        load_index("demo-acme", tmp_path)
    assert "demo-nordwind" in str(excinfo.value)
