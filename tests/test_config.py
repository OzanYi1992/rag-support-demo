"""Tests der Konfiguration.

Schwerpunkt ist die Verankerung relativer Pfade. Der Fehler, den sie
verhindert, ist still und vollstaendig: Laeuft die Anwendung mit einem anderen
Arbeitsverzeichnis, zeigt `tenants` ins Leere, jedes url_token ergibt 404, und
nichts weist darauf hin. Genau der Fall, der im Container auftraete - dort
kommt die Konfiguration aus der Umgebung, es gibt kein `.env`, und der Start
scheitert deshalb nicht laut.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import PROJEKTWURZEL, Settings

# Pflichtangaben, damit Settings ohne .env baut. Der Wert ist ein Platzhalter -
# hier wird nie ein Modell gerufen.
PFLICHT = {
    "llm_provider": "openai",
    "openai_api_key": "platzhalter",
    "openai_model": "platzhalter",
}


def test_projektwurzel_kommt_aus_dem_dateiort() -> None:
    """Nicht aus os.getcwd(), sonst loeste die Behebung das Problem mit dem
    Mechanismus auf, der es verursacht."""
    assert PROJEKTWURZEL.is_absolute()
    assert (PROJEKTWURZEL / "app" / "config.py").is_file()
    assert (PROJEKTWURZEL / "pyproject.toml").is_file()


def test_relative_pfade_werden_an_der_paketwurzel_verankert() -> None:
    settings = Settings(**PFLICHT, tenants_dir=Path("tenants"), index_dir=Path("data/index"))
    assert settings.tenants_dir == PROJEKTWURZEL / "tenants"
    assert settings.index_dir == PROJEKTWURZEL / "data" / "index"
    assert settings.tenants_dir.is_absolute()
    assert settings.index_dir.is_absolute()


def test_pfade_sind_unabhaengig_vom_arbeitsverzeichnis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Der eigentliche Test: derselbe relative Wert, zwei Arbeitsverzeichnisse,
    dieselben absoluten Pfade."""
    im_projekt = Settings(**PFLICHT, tenants_dir=Path("tenants"), index_dir=Path("data/index"))

    monkeypatch.chdir(tmp_path)
    anderswo = Settings(**PFLICHT, tenants_dir=Path("tenants"), index_dir=Path("data/index"))

    assert anderswo.tenants_dir == im_projekt.tenants_dir
    assert anderswo.index_dir == im_projekt.index_dir


def test_gegenprobe_ohne_verankerung_waere_es_ein_anderer_pfad(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Gegenprobe nach conventions.md, Regel 2.

    Ohne diesen Test waere die Aussage 'die Pfade sind unabhaengig vom
    Arbeitsverzeichnis' nicht von 'die Pfade sind zufaellig gleich' zu
    unterscheiden. Hier wird gezeigt, dass die naive Aufloesung - relativ zum
    Arbeitsverzeichnis - tatsaechlich etwas anderes ergeben haette.
    """
    monkeypatch.chdir(tmp_path)
    settings = Settings(**PFLICHT, tenants_dir=Path("tenants"))

    naiv = (tmp_path / "tenants").resolve()
    assert settings.tenants_dir != naiv, (
        "Die Pfade sind im Test zufaellig gleich - die Gegenprobe traegt nicht."
    )
    assert settings.tenants_dir == PROJEKTWURZEL / "tenants"


def test_absolute_pfade_bleiben_unangetastet(tmp_path: Path) -> None:
    """Zweite Gegenprobe, und die wichtigere.

    Ein Validator, der stumpf ALLES auf die Paketwurzel zwingt, waere bei den
    Tests oben ebenfalls gruen - und wuerde jeden Test mit `tmp_path` sowie
    jede Bereitstellung mit absolutem Pfad kaputtmachen. Wer einen Pfad
    ausdruecklich setzt, meint ihn auch.
    """
    eigen = tmp_path / "woanders" / "tenants"
    settings = Settings(**PFLICHT, tenants_dir=eigen, index_dir=tmp_path / "idx")

    assert settings.tenants_dir == eigen
    assert settings.index_dir == tmp_path / "idx"
    assert PROJEKTWURZEL not in settings.tenants_dir.parents
