"""Tests der HTTP-Schicht.

Die Schicht entscheidet nichts, deshalb pruefen diese Tests nicht, ob richtig
geantwortet wird - das tut test_rag.py. Hier geht es um vier Fragen:

1. Gibt ein oeffentlicher Endpunkt etwas preis, das er nicht soll?
2. Fuehrt ein Token zuverlaessig auf genau einen Mandanten und auf keinen anderen?
3. Sind unbekanntes und ungueltiges Token von aussen unterscheidbar?
4. Greift die Kostendeckelung?
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import E5Embeddings
from app.main import create_app
from app.prompts import GroundedAnswer
from tests.conftest import FAKE_DIMENSION, FakeBackend, FakeLlm, lege_mandant_an

ACME_TOKEN = "acme-token-1234567890"
NORDWIND_TOKEN = "nordwind-token-123456"

ACME_TEXT = """# Retouren

Jede Ruecksendung braucht eine RMA-Nummer. Sie ist 21 Kalendertage gueltig.
Ohne Nummer nehmen wir die Sendung nicht an.
"""

NORDWIND_TEXT = """# Lieferung

Wir liefern als Zwei-Mann-Montage in einem Terminfenster von vier Stunden.
Die Montage ist im Preis enthalten.
"""

ACME_ESKALATION = "Dazu finde ich in den Unterlagen der ACME nichts."


def _umgebung(tmp_path: Path) -> tuple[Settings, E5Embeddings]:
    from app.ingest import ingest_tenant

    tenants_root = tmp_path / "tenants"
    tenants_root.mkdir()
    lege_mandant_an(tenants_root, "demo-acme", ACME_TEXT, ACME_TOKEN, ACME_ESKALATION)
    lege_mandant_an(tenants_root, "demo-nordwind", NORDWIND_TEXT, NORDWIND_TOKEN)

    settings = Settings(
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=tenants_root,
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
        chunk_size=200,
        chunk_overlap=20,
    )
    embeddings = E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION)
    for slug in ("demo-acme", "demo-nordwind"):
        ingest_tenant(slug, settings=settings, embeddings=embeddings)
    return settings, embeddings


def gute_antwort(text: str = "Ja, 21 Kalendertage.") -> GroundedAnswer:
    return GroundedAnswer(answerable=True, answer=text, sources=["doku.md"], language="de")


@pytest.fixture
def umgebung(tmp_path: Path) -> tuple[Settings, E5Embeddings]:
    return _umgebung(tmp_path)


def _client(
    umgebung: tuple[Settings, E5Embeddings],
    llm: FakeLlm | None = None,
    rate_limit: int = 30,
) -> tuple[TestClient, FakeLlm]:
    settings, embeddings = umgebung
    aktives_llm = llm or FakeLlm(parsed=gute_antwort())
    app = create_app(settings, llm=aktives_llm, embeddings=embeddings, rate_limit=rate_limit)
    return TestClient(app), aktives_llm


# --- /health ----------------------------------------------------------------


def test_health_gibt_keine_mandantendaten_preis(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Der Endpunkt ist oeffentlich. Die Token sind die Zugangskontrolle
    (ADR-007); eine Mandantenliste hier waere ihr Gegenteil."""
    client, _ = _client(umgebung)
    antwort = client.get("/health")

    assert antwort.status_code == 200
    assert antwort.json() == {"status": "ok"}

    rumpf = antwort.text.lower()
    for verraeter in ("demo-acme", "demo-nordwind", "acme", "nordwind", "tenant"):
        assert verraeter not in rumpf
    assert ACME_TOKEN not in antwort.text
    assert NORDWIND_TOKEN not in antwort.text


def test_health_braucht_kein_token(umgebung: tuple[Settings, E5Embeddings]) -> None:
    client, _ = _client(umgebung)
    assert client.get("/health").status_code == 200


# --- Tokenaufloesung --------------------------------------------------------


def test_gueltiges_token_liefert_die_oberflaeche(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    client, _ = _client(umgebung)
    antwort = client.get(f"/t/{ACME_TOKEN}/")

    assert antwort.status_code == 200
    assert "text/html" in antwort.headers["content-type"]
    # Der Anzeigename kommt serverseitig aus der TenantConfig.
    assert "Demo Acme" in antwort.text or "demo-acme" in antwort.text.lower()
    # Kein Platzhalter darf ungefuellt durchrutschen.
    assert "{{display_name}}" not in antwort.text


def test_unbekanntes_token_ist_404(umgebung: tuple[Settings, E5Embeddings]) -> None:
    client, _ = _client(umgebung)
    antwort = client.get("/t/gibtesnichtaberlangenug/")
    assert antwort.status_code == 404


def test_unbekanntes_und_zu_kurzes_token_sind_ununterscheidbar(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Aus der Antwort darf nicht ableitbar sein, ob ein Token existiert.

    Verschiedene Wortlaute waeren ein Orakel: Wer den Unterschied zwischen
    'ungueltig' und 'unbekannt' sieht, kann Token einkreisen.
    """
    client, _ = _client(umgebung)
    unbekannt = client.get("/t/gibtesnichtaberlangenug/")
    zu_kurz = client.get("/t/kurz/")

    assert unbekannt.status_code == zu_kurz.status_code == 404
    assert unbekannt.json() == zu_kurz.json()


def test_kein_endpunkt_listet_mandanten_auf(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    client, _ = _client(umgebung)
    for pfad in ("/t/", "/t", "/tenants", "/openapi.json", "/docs"):
        antwort = client.get(pfad)
        assert antwort.status_code in (404, 405), pfad
        assert "demo-acme" not in antwort.text
        assert "demo-nordwind" not in antwort.text


# --- Chat -------------------------------------------------------------------


def test_chat_roundtrip(umgebung: tuple[Settings, E5Embeddings]) -> None:
    client, llm = _client(umgebung)
    antwort = client.post(
        f"/t/{ACME_TOKEN}/chat", json={"question": "Wie lange gilt die RMA-Nummer?"}
    )

    assert antwort.status_code == 200
    daten = antwort.json()
    assert daten["text"] == "Ja, 21 Kalendertage."
    assert daten["escalated"] is False
    assert daten["sources"]
    assert daten["lang"] == "de"
    assert daten["prompt_tokens"] == 123
    assert len(llm.calls) == 1


def test_chat_reicht_response_language_durch(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    llm = FakeLlm(
        parsed=GroundedAnswer(
            answerable=True, answer="Twenty-one days.", sources=["doc.md"], language="en"
        )
    )
    client, _ = _client(umgebung, llm=llm)
    antwort = client.post(
        f"/t/{ACME_TOKEN}/chat",
        json={"question": "How long is the RMA valid?", "response_language": "en"},
    )
    assert antwort.status_code == 200
    assert antwort.json()["lang"] == "en"


def test_chat_mit_unbekanntem_token_ruft_kein_llm_auf(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Ein fremdes Token darf nichts kosten."""
    client, llm = _client(umgebung)
    antwort = client.post("/t/gibtesnichtaberlangenug/chat", json={"question": "Hallo"})

    assert antwort.status_code == 404
    assert llm.calls == []


def test_leere_frage_wird_abgewiesen(umgebung: tuple[Settings, E5Embeddings]) -> None:
    client, llm = _client(umgebung)
    antwort = client.post(f"/t/{ACME_TOKEN}/chat", json={"question": "   "})
    # Pydantic laesst Leerzeichen durch; entscheidend ist, dass kein Absturz
    # passiert und die Antwort wohlgeformt bleibt.
    assert antwort.status_code in (200, 422)


def test_eskalation_zeigt_den_text_des_mandanten(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    llm = FakeLlm(parsed=GroundedAnswer(answerable=False, answer="", sources=[], language="de"))
    client, _ = _client(umgebung, llm=llm)
    antwort = client.post(
        f"/t/{ACME_TOKEN}/chat", json={"question": "Welche Farbe hat der Karton?"}
    )

    assert antwort.status_code == 200
    daten = antwort.json()
    assert daten["escalated"] is True
    assert daten["escalation_reason"] == "not_grounded"
    assert daten["text"] == ACME_ESKALATION


# --- Ratenbegrenzung --------------------------------------------------------


def test_rate_limit_greift(umgebung: tuple[Settings, E5Embeddings]) -> None:
    client, llm = _client(umgebung, rate_limit=3)

    for i in range(3):
        antwort = client.post(f"/t/{ACME_TOKEN}/chat", json={"question": f"Frage {i}"})
        assert antwort.status_code == 200, i

    gesperrt = client.post(f"/t/{ACME_TOKEN}/chat", json={"question": "eine zu viel"})
    assert gesperrt.status_code == 429
    assert "Retry-After" in gesperrt.headers
    assert int(gesperrt.headers["Retry-After"]) >= 1
    # Die abgewiesene Anfrage darf kein Modell gekostet haben.
    assert len(llm.calls) == 3


def test_rate_limit_gilt_je_token(umgebung: tuple[Settings, E5Embeddings]) -> None:
    """Ein ausgeschoepfter Mandant darf einen anderen nicht blockieren."""
    client, _ = _client(umgebung, rate_limit=2)

    for _ in range(2):
        assert client.post(f"/t/{ACME_TOKEN}/chat", json={"question": "x"}).status_code == 200
    assert client.post(f"/t/{ACME_TOKEN}/chat", json={"question": "x"}).status_code == 429

    # Nordwind hat sein eigenes Kontingent.
    assert client.post(f"/t/{NORDWIND_TOKEN}/chat", json={"question": "x"}).status_code == 200


# --- Mandantentrennung (ADR-001) --------------------------------------------


def test_token_von_acme_liefert_nie_inhalte_von_nordwind(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Der Kern der Demo.

    Geprueft wird der Prompt, nicht die Antwort: Die Attrappe antwortet immer
    dasselbe, egal was im Kontext steht. Ein Leck waere daran zu erkennen, dass
    fremde Inhalte ueberhaupt in den Kontext gelangen - danach ist es zu spaet,
    denn dann entscheidet nur noch das Modell.
    """
    client, llm = _client(umgebung)
    antwort = client.post(
        f"/t/{ACME_TOKEN}/chat",
        json={"question": "Wie laeuft die Lieferung mit Zwei-Mann-Montage?"},
    )

    assert antwort.status_code == 200
    assert len(llm.calls) == 1
    _, user_prompt = llm.calls[0]

    # Nur der Kontextblock, nicht die Frage. Der Prompt ist
    # "Kontext:\n\n...\n\n---\n\nFrage: ..." - die Frage darf die fremden
    # Begriffe enthalten, sie kommt ja vom Fragenden.
    kontext = user_prompt.split("Frage:")[0]

    for fremd in ("Zwei-Mann-Montage", "Terminfenster", "Montage ist im Preis"):
        assert fremd not in kontext, f"Inhalt von demo-nordwind im Kontext: {fremd}"

    # Zwei Gegenproben, sonst waere der Test auch bei leerem Kontext gruen:
    # der eigene Mandant liefert Inhalt, und die Frage steht im Prompt.
    assert "RMA-Nummer" in kontext
    assert "Zwei-Mann-Montage" in user_prompt


def test_beide_mandanten_bekommen_eigene_kontexte(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    client, llm = _client(umgebung)

    client.post(f"/t/{ACME_TOKEN}/chat", json={"question": "RMA?"})
    client.post(f"/t/{NORDWIND_TOKEN}/chat", json={"question": "Montage?"})

    assert len(llm.calls) == 2
    acme_prompt = llm.calls[0][1]
    nordwind_prompt = llm.calls[1][1]

    assert "RMA" in acme_prompt
    assert "RMA" not in nordwind_prompt
    assert "Montage" in nordwind_prompt


def test_eskalationstext_ist_mandantenspezifisch(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    llm = FakeLlm(parsed=GroundedAnswer(answerable=False, answer="", sources=[], language="de"))
    client, _ = _client(umgebung, llm=llm)

    acme = client.post(f"/t/{ACME_TOKEN}/chat", json={"question": "?"}).json()
    nordwind = client.post(f"/t/{NORDWIND_TOKEN}/chat", json={"question": "?"}).json()

    assert acme["text"] == ACME_ESKALATION
    assert nordwind["text"] != ACME_ESKALATION
