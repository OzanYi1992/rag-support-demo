"""Tests des Logs: Ereignisse auf stdout, das Token nicht im Zugriffsprotokoll.

Was hier gehalten wird:

1. Nach create_app ergibt jedes Ereignis genau eine Zeile auf stdout, mit der
   Uhrzeit in UTC und dem JSON des Ereignisses. Ohne Einrichtung ging es unter
   uvicorn verloren.
2. Das Zugriffsprotokoll von uvicorn traegt das url_token nicht im Klartext.
3. Die Stufe kommt aus LOG_LEVEL, und ein Tippfehler scheitert beim Start.

Das Zugriffsprotokoll schreibt uvicorn, nicht die Anwendung. Der Test baut die
Zeile deshalb so, wie uvicorn sie baut, und formatiert sie mit dessen eigenem
Formatierer.
"""

from __future__ import annotations

import io
import json
import logging
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from uvicorn.logging import AccessFormatter

from app.config import Settings
from app.embeddings import E5Embeddings
from app.main import _TokenMaske, create_app
from tests.conftest import FAKE_DIMENSION, FakeBackend

TOKEN = "geheim-token-7Q4Z-1234567890"

# Die Zeile eines Ereignisses: Uhrzeit in UTC mit Millisekunden, dann JSON.
ZEILE = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3})Z (\{.*\})")


def _settings(tmp_path: Path, **abweichend: object) -> Settings:
    return Settings(
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=tmp_path / "tenants",
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
        **abweichend,  # type: ignore[arg-type]
    )


def _client(settings: Settings) -> TestClient:
    embeddings = E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION)
    return TestClient(create_app(settings, embeddings=embeddings))


@pytest.fixture(autouse=True)
def _logger_zuruecksetzen() -> Iterator[None]:
    """Die Logger sind prozessweit. Was ein Test an Stufe und Filtern setzt,
    soll den naechsten nicht treffen."""
    ereignisse = logging.getLogger("rag.api")
    zugriff = logging.getLogger("uvicorn.access")
    stufe_ereignisse, stufe_zugriff = ereignisse.level, zugriff.level
    filter_zugriff = list(zugriff.filters)
    yield
    ereignisse.setLevel(stufe_ereignisse)
    zugriff.setLevel(stufe_zugriff)
    zugriff.filters[:] = filter_zugriff


def _zeilen(ausgabe: str) -> list[str]:
    return [zeile for zeile in ausgabe.splitlines() if zeile.strip()]


def test_ein_ereignis_ist_genau_eine_zeile_auf_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Ein unbekannter Pfad loest genau ein Ereignis aus, route_unbekannt. Auf
    stdout steht genau eine Zeile: die Uhrzeit in UTC, dann gueltiges JSON mit
    denselben Feldern wie bisher. Zweimal create_app ergibt keine zweite Zeile."""
    create_app(_settings(tmp_path))
    client = _client(_settings(tmp_path))
    capsys.readouterr()

    vorher = datetime.now(UTC)
    client.get("/nirgends")
    zeilen = _zeilen(capsys.readouterr().out)

    assert len(zeilen) == 1, zeilen
    treffer = ZEILE.fullmatch(zeilen[0])
    assert treffer, zeilen[0]
    zeit = datetime.strptime(treffer[1], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=UTC)
    assert vorher - timedelta(seconds=1) <= zeit <= datetime.now(UTC) + timedelta(seconds=1)
    daten = json.loads(treffer[2])
    assert daten == {"ereignis": "route_unbekannt", "tenant_id": None, "pfad": "/nirgends"}


def test_ohne_einrichtung_erreicht_kein_ereignis_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Gegenprobe zum Test oben: Ohne den Handler aus create_app schreibt der
    Logger nichts auf stdout. Der Test oben misst also die Einrichtung und nicht
    einen Handler, den es ohnehin gaebe."""
    client = _client(_settings(tmp_path))
    ereignisse = logging.getLogger("rag.api")
    gesichert = list(ereignisse.handlers)
    ereignisse.handlers[:] = []
    try:
        capsys.readouterr()
        client.get("/nirgends")
        assert _zeilen(capsys.readouterr().out) == []
    finally:
        ereignisse.handlers[:] = gesichert


def test_log_level_warning_unterdrueckt_die_ereignisse(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Die Stufe kommt aus LOG_LEVEL. Ereignisse sind INFO und erscheinen bei
    WARNING nicht."""
    client = _client(_settings(tmp_path, log_level="warning"))
    assert logging.getLogger("rag.api").level == logging.WARNING
    capsys.readouterr()
    client.get("/nirgends")
    assert _zeilen(capsys.readouterr().out) == []


@pytest.mark.parametrize(("eingabe", "stufe"), [("info", "INFO"), (" Warning ", "WARNING")])
def test_log_level_wird_gross_geschrieben(tmp_path: Path, eingabe: str, stufe: str) -> None:
    assert _settings(tmp_path, log_level=eingabe).log_level == stufe


def test_ein_tippfehler_in_log_level_scheitert_beim_start(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="LOG_LEVEL"):
        _settings(tmp_path, log_level="INFOO")


def _zugriffszeile(pfad: str) -> str:
    """Schreibt eine Zugriffszeile genau so, wie uvicorn sie schreibt, und gibt
    sie formatiert zurueck."""
    strom = io.StringIO()
    handler = logging.StreamHandler(strom)
    handler.setFormatter(
        AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s', use_colors=False)
    )
    zugriff = logging.getLogger("uvicorn.access")
    zugriff.setLevel(logging.INFO)
    zugriff.addHandler(handler)
    try:
        zugriff.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "POST", pfad, "1.1", 200)
    finally:
        zugriff.removeHandler(handler)
    return strom.getvalue()


def test_das_zugriffsprotokoll_maskiert_das_token(tmp_path: Path) -> None:
    """Nach create_app steht im Zugriffsprotokoll der Platzhalter statt des
    Tokens. Andere Pfade bleiben, wie sie sind.

    Gegenprobe im selben Test: Ohne den Filter steht das Token in der Zeile.
    """
    create_app(_settings(tmp_path))

    zeile = _zugriffszeile(f"/t/{TOKEN}/chat")
    assert TOKEN not in zeile
    assert '"POST /t/{token}/chat HTTP/1.1" 200 OK' in zeile
    assert "/t/{token}/doc/faq.md" in _zugriffszeile(f"/t/{TOKEN}/doc/faq.md")
    assert "/t/{token}/" in _zugriffszeile(f"/t/{TOKEN}/")
    assert "/static/app.js?v=abc" in _zugriffszeile("/static/app.js?v=abc")

    zugriff = logging.getLogger("uvicorn.access")
    masken = [f for f in zugriff.filters if isinstance(f, _TokenMaske)]
    assert len(masken) == 1
    zugriff.removeFilter(masken[0])
    assert TOKEN in _zugriffszeile(f"/t/{TOKEN}/chat")


def test_der_filter_kommt_nur_einmal_dazu(tmp_path: Path) -> None:
    """Mehrfach create_app, ein Filter und ein Handler."""
    for _ in range(3):
        create_app(_settings(tmp_path))
    zugriff = logging.getLogger("uvicorn.access")
    assert sum(isinstance(f, _TokenMaske) for f in zugriff.filters) == 1
    ereignisse = logging.getLogger("rag.api")
    assert sum(type(h).__name__ == "_StdoutHandler" for h in ereignisse.handlers) == 1
