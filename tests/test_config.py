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


def test_embedding_device_nimmt_nur_cpu() -> None:
    """torch ist als CPU-Wheel gepinnt und enthaelt keinen CUDA-Code.

    Ohne diese Pruefung scheitert ein falsches Geraet erst beim Laden des
    Modells - im Container also lange nach dem Start, beim ersten Embedding.
    """
    with pytest.raises(ValueError, match="EMBEDDING_DEVICE"):
        Settings(**PFLICHT, embedding_device="cuda")


def test_embedding_device_cpu_bleibt_zulaessig() -> None:
    """Gegenprobe: Der zulaessige Wert muss durchkommen.

    Ohne sie waere ein Validator, der ALLES ablehnt, von einem, der das
    Richtige ablehnt, nicht zu unterscheiden.
    """
    assert Settings(**PFLICHT, embedding_device="cpu").embedding_device == "cpu"


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


# --- Sampling ---------------------------------------------------------------


def test_temperatur_hat_einen_standard_und_der_ist_eins() -> None:
    """1.0, weil die gpt-5-Familie nichts anderes annimmt.

    Der Standard ist bewusst NICHT 0.0. langchain-openai verwirft jeden anderen
    Wert als 1 fuer diese Modellfamilie still, und eine Einstellung, die nicht
    im Payload ankommt, ist eine Behauptung. Siehe
    `test_nur_temperatur_eins_erreicht_das_konfigurierte_modell`.
    """
    assert Settings(**PFLICHT).llm_temperature == 1.0


def test_temperatur_nimmt_nur_werte_bis_eins() -> None:
    """Obergrenze ist der Schnitt aller drei Provider aus ADR-006.

    OpenAI nimmt bis 2.0, Anthropic nur bis 1.0. Ein Wert, der beim
    Providerwechsel ungueltig wird, macht die Austauschbarkeit zur Behauptung.
    """
    with pytest.raises(ValueError):
        Settings(**PFLICHT, llm_temperature=1.5)
    with pytest.raises(ValueError):
        Settings(**PFLICHT, llm_temperature=-0.1)


def test_temperatur_nimmt_den_zulaessigen_bereich_an() -> None:
    """Gegenprobe: Ein Validator, der alles ablehnt, prueft nichts."""
    for wert in (0.0, 0.5, 1.0):
        assert Settings(**PFLICHT, llm_temperature=wert).llm_temperature == wert
