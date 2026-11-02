"""Gegenproben zur Netzsperre in tests/conftest.py.

Jeder Test ohne die Marke integration darf weder einen Namen ausserhalb dieses
Rechners aufloesen noch sich dorthin verbinden. Die Tests hier rufen das
absichtlich auf und leeren danach die Liste der Versuche, sonst fielen sie
selbst. Der letzte Test zeigt in einem eigenen pytest-Lauf, dass ein Test, der
das Netz sucht und den Fehler schluckt, trotzdem faellt.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import Settings
from app.llm import LlmNichtErreichbar, build_llm
from app.prompts import antwortmodell_fuer
from tests.conftest import REPO_ROOT, NetzGesperrt, _ist_lokal


def test_eine_verbindung_nach_draussen_ist_gesperrt(kein_netz: list[str]) -> None:
    """192.0.2.1 ist ein Adressbereich nur fuer Dokumentation; ohne Sperre liefe
    die Verbindung ins Zeitlimit. `create_connection` loest auch eine Adresse
    zuerst auf und faellt schon dort; `connect` geht direkt."""
    with pytest.raises(NetzGesperrt):
        socket.create_connection(("192.0.2.1", 443), timeout=0.5)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(NetzGesperrt):
        s.settimeout(0.5)
        s.connect(("192.0.2.1", 443))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s, pytest.raises(NetzGesperrt):
        s.connect_ex(("192.0.2.1", 443))
    assert kein_netz == [
        "Namensaufloesung '192.0.2.1'",
        "Verbindung '192.0.2.1'",
        "Verbindung '192.0.2.1'",
    ]
    kein_netz.clear()


def test_der_name_des_anbieters_wird_nicht_aufgeloest(kein_netz: list[str]) -> None:
    with pytest.raises(NetzGesperrt):
        socket.getaddrinfo("api.openai.com", 443)
    assert kein_netz == ["Namensaufloesung 'api.openai.com'"]
    kein_netz.clear()


def test_der_weg_zum_anbieter_ist_gesperrt(kein_netz: list[str]) -> None:
    """Der echte Client, wie die Anwendung ihn baut: Der Aufruf kommt nicht beim
    Anbieter an, die Anwendung meldet ihn als nicht erreichbar."""
    settings = Settings(
        llm_provider="openai",
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        llm_max_retries=0,
    )
    llm = build_llm(settings, answer_model=antwortmodell_fuer("en"))
    with pytest.raises(LlmNichtErreichbar):
        llm.generate("system", "user")
    assert kein_netz and all("api.openai.com" in v for v in kein_netz), kein_netz
    kein_netz.clear()


def test_eine_verbindung_auf_diesem_rechner_bleibt_erlaubt(kein_netz: list[str]) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        with socket.create_connection(server.getsockname(), timeout=2):
            pass
    assert kein_netz == []


@pytest.mark.parametrize(
    ("host", "lokal"),
    [
        ("127.0.0.1", True),
        ("127.5.6.7", True),
        ("::1", True),
        ("localhost", True),
        ("LOCALHOST", True),
        (b"localhost", True),
        (None, True),
        ("", True),
        ("192.0.2.1", False),
        ("10.0.0.1", False),
        ("api.openai.com", False),
        ("testserver", False),
        (443, False),
    ],
)
def test_ist_lokal(host: object, lokal: bool) -> None:
    assert _ist_lokal(host) is lokal


_TEST_MIT_NETZ = """
import socket


def test_sucht_das_netz_und_schluckt_den_fehler():
    try:
        socket.create_connection(("192.0.2.1", 443), timeout=0.5)
    except OSError:
        pass


def test_ohne_netz():
    assert True
"""


def test_ein_test_der_das_netz_sucht_faellt(tmp_path: Path) -> None:
    """Gegenprobe in einem eigenen pytest-Lauf mit derselben conftest.py: Der
    Test, der das Netz sucht, faellt am Ende, obwohl er den Fehler gefangen
    hat; der andere besteht. pytest zaehlt den Aufruf als bestanden und den
    Abschluss als Fehler, daher "2 passed, 1 error"."""
    shutil.copy(REPO_ROOT / "tests" / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_innen.py").write_text(_TEST_MIT_NETZ, encoding="utf-8")
    lauf = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    ausgabe = lauf.stdout + lauf.stderr
    assert lauf.returncode == 1, ausgabe
    assert "2 passed, 1 error" in ausgabe, ausgabe
    assert "ERROR test_innen.py::test_sucht_das_netz_und_schluckt_den_fehler" in ausgabe, ausgabe
    assert "ERROR test_innen.py::test_ohne_netz" not in ausgabe, ausgabe
    assert "wollte das Netz erreichen: Namensaufloesung '192.0.2.1'" in ausgabe, ausgabe
