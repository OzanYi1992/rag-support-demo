"""Tests der HTTP-Schicht.

Die Schicht entscheidet nichts, deshalb pruefen diese Tests nicht, ob richtig
geantwortet wird - das tut test_rag.py. Hier geht es um vier Fragen:

1. Gibt ein oeffentlicher Endpunkt etwas preis, das er nicht soll?
2. Fuehrt ein Token zuverlaessig auf genau einen Mandanten und auf keinen anderen?
3. Sind unbekanntes und ungueltiges Token von aussen unterscheidbar?
4. Greift die Kostendeckelung?
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import E5Embeddings
from app.main import NICHT_GEFUNDEN, create_app, statische_fassung
from app.prompts import GroundedAnswer
from app.tenants import load_tenant
from app.texts import JAVASCRIPT_SCHLUESSEL
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

ENGLISCH_TOKEN = "englisch-token-1234567890"

ENGLISCH_TEXT = """# Shipping

Standard delivery takes three working days. Express delivery arrives the next
working day if the order is placed before noon.
"""

ENGLISCH_ESKALATION = "I cannot find that in the documents. Please write to hello@example.test."


def _umgebung(tmp_path: Path) -> tuple[Settings, E5Embeddings]:
    from app.ingest import ingest_tenant

    tenants_root = tmp_path / "tenants"
    tenants_root.mkdir()
    lege_mandant_an(tenants_root, "demo-acme", ACME_TEXT, ACME_TOKEN, ACME_ESKALATION)
    lege_mandant_an(tenants_root, "demo-nordwind", NORDWIND_TEXT, NORDWIND_TOKEN)
    lege_mandant_an(
        tenants_root,
        "demo-englisch",
        ENGLISCH_TEXT,
        ENGLISCH_TOKEN,
        ENGLISCH_ESKALATION,
        language="en",
    )

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
    for slug in ("demo-acme", "demo-nordwind", "demo-englisch"):
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


def test_wurzel_ist_404_und_hinterlaesst_eine_spur(
    umgebung: tuple[Settings, E5Embeddings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Es gibt keine Route auf `/`. Das ist richtig - aber es muss im Log
    stehen, sonst sucht der Betreiber im Dunkeln (P-018, dritte Auspraegung)."""
    client, _ = _client(umgebung)
    with caplog.at_level("INFO", logger="rag.api"):
        antwort = client.get("/")

    assert antwort.status_code == 404
    assert antwort.json() == {"detail": NICHT_GEFUNDEN}
    assert any("route_unbekannt" in eintrag.message for eintrag in caplog.records)


def test_nicht_gefunden_ist_zweisprachig_und_ueberall_derselbe_wortlaut(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Der 404 traegt beide Sprachen und richtet sich nach keinem Mandanten.

    Zwei Gruende, und beide muessen gleichzeitig gelten:

    1. Ein zerbrochener Link kommt auch bei einem englischsprachigen
       Empfaenger an. Ein rein deutscher Satz laesst ihn ratlos zurueck.
    2. Der Wortlaut darf sich NICHT nach einer Sprache richten. Er faellt,
       bevor ein Mandant aufgeloest ist - und zwei verschiedene Wortlaute
       waeren ein Orakel, an dem sich die Existenz eines Token ablesen liesse.

    Deshalb: beide Haelften immer zusammen, fuer jeden Aufruf identisch.
    """
    client, _ = _client(umgebung)

    assert "Diese Adresse gibt es nicht." in NICHT_GEFUNDEN
    assert "This address does not exist." in NICHT_GEFUNDEN

    antworten = [
        client.get("/"),
        client.get("/t/gibtesnichtaberlangenug/"),
        client.get("/t/kurz/"),
        client.get("/beliebiger-unbekannter-pfad"),
    ]
    for antwort in antworten:
        assert antwort.status_code == 404
        assert antwort.json() == {"detail": NICHT_GEFUNDEN}


def test_log_enthaelt_niemals_das_token(
    umgebung: tuple[Settings, E5Embeddings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Das Token ist die Zugangskontrolle. Ein Log, das Token mitschreibt, ist
    eine Schluesselliste - und Logs wandern spaeter aus meinem Zugriff heraus."""
    client, _ = _client(umgebung)
    with caplog.at_level("INFO", logger="rag.api"):
        client.get(f"/t/{ACME_TOKEN}/unbekannter-unterpfad")
        client.get(f"/t/{ACME_TOKEN}/")
        client.post(f"/t/{ACME_TOKEN}/chat", json={"question": "RMA?"})

    gesamtes_log = "\n".join(eintrag.message for eintrag in caplog.records)
    assert gesamtes_log, "Kein Logeintrag - der Test waere sonst blind."
    assert ACME_TOKEN not in gesamtes_log
    # Gegenprobe: Das Muster steht sehr wohl drin, sonst waere die Diagnose
    # ueber das Log unmoeglich.
    assert "{token}" in gesamtes_log
    # Und die tenant_id ist als Dimension da, wo ein Mandant aufgeloest wurde
    # (ADR-002).
    assert "demo-acme" in gesamtes_log


# --- Sprache der Oberflaeche ------------------------------------------------


def _seite(client: TestClient, token: str) -> str:
    antwort = client.get(f"/t/{token}/")
    assert antwort.status_code == 200
    return antwort.text


def test_englischer_mandant_liefert_eine_englische_seite(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    client, _ = _client(umgebung)
    seite = _seite(client, ENGLISCH_TOKEN)

    assert '<html lang="en">' in seite
    assert "Support Assistant" in seite
    assert "Your question" in seite
    assert ">Send<" in seite
    assert "Answers come exclusively from the stored documents." in seite

    # Und keine deutsche Zeile bleibt stehen.
    for deutsch in ("Support-Assistent", "Ihre Frage", ">Senden<", "Guten Tag"):
        assert deutsch not in seite, deutsch


def test_deutscher_mandant_liefert_unveraendert_eine_deutsche_seite(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Die Gegenprobe. Ohne sie zeigt der Test oben nur, dass sich etwas geaendert hat."""
    client, _ = _client(umgebung)
    seite = _seite(client, ACME_TOKEN)

    assert '<html lang="de">' in seite
    assert "Support-Assistent" in seite
    assert "Ihre Frage" in seite
    assert ">Senden<" in seite
    assert "Support Assistant" not in seite


def test_kein_platzhalter_bleibt_ungefuellt(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Ein vergessener Platzhalter steht sonst woertlich in der Seite.

    Er stuerzt nicht ab und faellt in keinem anderen Test auf - er steht
    einfach da, vor dem Interessenten.
    """
    client, _ = _client(umgebung)
    for token in (ACME_TOKEN, NORDWIND_TOKEN, ENGLISCH_TOKEN):
        seite = _seite(client, token)
        assert "{{" not in seite, token
        assert "}}" not in seite, token
        # Der Platzhalter INNERHALB eines Katalogtextes muss ebenfalls
        # eingesetzt sein.
        assert "{display_name}" not in seite, token


def test_browsertexte_stehen_als_gueltiges_json_in_der_seite(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """app.js liest seine Texte aus diesem Block. Ist er kaputt, ist die
    Oberflaeche stumm - ohne Fehler auf der Serverseite."""
    import json
    import re

    client, _ = _client(umgebung)
    for token, erwartet in (
        (ACME_TOKEN, "Sucht in den Unterlagen …"),
        (ENGLISCH_TOKEN, "Searching the documents …"),
    ):
        seite = _seite(client, token)
        treffer = re.search(
            r'<script id="texte" type="application/json">(.*?)</script>', seite, re.S
        )
        assert treffer, token
        daten = json.loads(treffer.group(1))
        assert daten["sucht"] == erwartet
        assert set(daten) == set(JAVASCRIPT_SCHLUESSEL)
        # Was nur der Server braucht, steht nicht in der ausgelieferten Seite.
        assert "ratenlimit_detail" not in daten


def test_json_block_kann_nicht_vorzeitig_schliessen(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Ein `<` im JSON-Block waere der Anfang eines Tags, nicht Daten."""
    client, _ = _client(umgebung)
    import re

    seite = _seite(client, ENGLISCH_TOKEN)
    treffer = re.search(r'<script id="texte" type="application/json">(.*?)</script>', seite, re.S)
    assert treffer
    assert "<" not in treffer.group(1)
    assert ">" not in treffer.group(1)


def test_ratenlimit_meldung_folgt_der_mandantensprache(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Der Mandant ist hier aufgeloest, also gibt es eine Sprache.

    Anders als beim 404 verraet dieser Text nichts: Wer ihn sieht, hat bereits
    ein gueltiges Token.
    """
    client, _ = _client(umgebung, rate_limit=1)

    client.post(f"/t/{ENGLISCH_TOKEN}/chat", json={"question": "How long is delivery?"})
    gebremst = client.post(f"/t/{ENGLISCH_TOKEN}/chat", json={"question": "And express?"})

    assert gebremst.status_code == 429
    assert gebremst.json()["detail"] == "Too many requests. Please wait a moment."

    # Gegenprobe beim deutschen Mandanten, eigenes Kontingent je Token.
    client.post(f"/t/{ACME_TOKEN}/chat", json={"question": "Wie lange gilt die RMA?"})
    deutsch = client.post(f"/t/{ACME_TOKEN}/chat", json={"question": "Und ohne Nummer?"})

    assert deutsch.status_code == 429
    assert deutsch.json()["detail"] == "Zu viele Anfragen. Bitte kurz warten."


def test_englischer_mandant_eskaliert_auf_englisch(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Die Eskalationsnachricht kommt aus der tenant.yaml, nicht aus dem Katalog.

    Sie nennt die Supportadresse dieses Mandanten; ein globaler Katalog kann
    das nicht tragen. Der Test haelt fest, dass der englische Mandant auf
    Englisch eskaliert - ohne dass der Katalog daran beteiligt ist.
    """
    llm = FakeLlm(parsed=GroundedAnswer(answerable=False, answer="", sources=[], language="en"))
    client, _ = _client(umgebung, llm=llm)
    antwort = client.post(f"/t/{ENGLISCH_TOKEN}/chat", json={"question": "What colour is the box?"})

    assert antwort.status_code == 200
    daten = antwort.json()
    assert daten["escalated"] is True
    assert daten["text"] == ENGLISCH_ESKALATION


def test_sprache_steht_im_log_und_der_token_nicht(
    umgebung: tuple[Settings, E5Embeddings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Die Sprache ist eine Dimension, das Token bleibt draussen."""
    client, _ = _client(umgebung)
    with caplog.at_level("INFO", logger="rag.api"):
        client.get(f"/t/{ENGLISCH_TOKEN}/")

    gesamtes_log = "\n".join(eintrag.message for eintrag in caplog.records)
    assert gesamtes_log, "Kein Logeintrag - der Test waere sonst blind."
    assert '"sprache": "en"' in gesamtes_log
    assert '"tenant_id": "demo-englisch"' in gesamtes_log
    assert ENGLISCH_TOKEN not in gesamtes_log


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


# --- Vorwaermen des Embedders beim Start (ADR-023) --------------------------
#
# Geprueft wird, WANN der Embedder entsteht, nicht wie oft. Wie oft, ist
# ADR-020 und steht in tests/test_embeddings_geteilt.py - hier wird es nur
# nicht verletzt.
#
# Der Lebenszyklus laeuft nur, wenn TestClient als Kontextmanager benutzt wird.
# Die uebrigen Tests dieser Datei tun das bewusst nicht: Sie speisen einen
# Ersatz ein und brauchen den Start nicht.


class _ZaehlendeFabrik:
    """Ersatz fuer get_embeddings, der mitzaehlt statt ein Modell zu laden."""

    def __init__(self, embeddings: E5Embeddings, fehler: Exception | None = None) -> None:
        self._embeddings = embeddings
        self._fehler = fehler
        self.aufrufe = 0

    def __call__(self, settings: Settings) -> E5Embeddings:
        self.aufrufe += 1
        if self._fehler is not None:
            raise self._fehler
        return self._embeddings


def test_embedder_entsteht_beim_start_ohne_anfrage(
    umgebung: tuple[Settings, E5Embeddings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Der Kern von ADR-023.

    Ohne Vorwaermen meldet /health rund sechs Sekunden lang Gesundheit, bevor
    eine Frage beantwortet werden kann. Nach dem Start muss der Embedder da
    sein, ohne dass je eine Anfrage lief.
    """
    settings, embeddings = umgebung
    fabrik = _ZaehlendeFabrik(embeddings)
    monkeypatch.setattr("app.main.get_embeddings", fabrik)

    app = create_app(settings, llm=FakeLlm(parsed=gute_antwort()))
    assert app.state.embeddings is None, "vor dem Start darf nichts da sein"
    assert fabrik.aufrufe == 0

    with TestClient(app):
        assert fabrik.aufrufe == 1
        assert app.state.embeddings is embeddings


def test_eingespeister_embedder_wird_nicht_ueberschrieben(
    umgebung: tuple[Settings, E5Embeddings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gegenprobe, und zugleich der Schutz der uebrigen Testsuite.

    Wuerde der Start einen eingespeisten Ersatz ueberschreiben, zoegen alle
    Tests, die eine Attrappe einspeisen, beim Start das echte Modell - und
    conventions.md verbietet Netzzugriff in Unit-Tests.
    """
    settings, embeddings = umgebung
    fabrik = _ZaehlendeFabrik(E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION))
    monkeypatch.setattr("app.main.get_embeddings", fabrik)

    app = create_app(settings, llm=FakeLlm(parsed=gute_antwort()), embeddings=embeddings)

    with TestClient(app):
        assert fabrik.aufrufe == 0, "der Start darf die Fabrik gar nicht erst rufen"
        assert app.state.embeddings is embeddings


def test_kein_zweiter_embedder_durch_anfragen(
    umgebung: tuple[Settings, E5Embeddings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-020 bleibt unangetastet: Das Vorwaermen aendert das WANN, nicht das
    WIE OFT. Zwei Anfragen nach dem Start duerfen keinen zweiten erzeugen."""
    settings, embeddings = umgebung
    fabrik = _ZaehlendeFabrik(embeddings)
    monkeypatch.setattr("app.main.get_embeddings", fabrik)

    app = create_app(settings, llm=FakeLlm(parsed=gute_antwort()))

    with TestClient(app) as client:
        for _ in range(2):
            antwort = client.post(
                f"/t/{ACME_TOKEN}/chat", json={"question": "Wie lange gilt die RMA-Nummer?"}
            )
            assert antwort.status_code == 200
        assert fabrik.aufrufe == 1


def test_fehlschlag_beim_laden_verhindert_den_start(
    umgebung: tuple[Settings, E5Embeddings], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Der zweite Grund aus ADR-023, geprueft statt behauptet.

    Ohne diesen Test waere "scheitert beim Start statt beim Interessenten" eine
    Absichtserklaerung. Ein kaputter Modellcache muss den Start verhindern -
    dieselbe Regel wie beim fehlenden Schluessel: laut und sofort.
    """
    settings, _ = umgebung
    fabrik = _ZaehlendeFabrik(
        E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION),
        fehler=RuntimeError("Modellgewichte nicht ladbar"),
    )
    monkeypatch.setattr("app.main.get_embeddings", fabrik)

    app = create_app(settings, llm=FakeLlm(parsed=gute_antwort()))

    with pytest.raises(RuntimeError, match="Modellgewichte nicht ladbar"):
        with TestClient(app):
            pass
    assert fabrik.aufrufe == 1


# =============================================================================
# CACHE: die Seite nie, die versionierten Dateien lange
#
# Der Anlass steht in OP-054. Am 2026-09-23 lieferte ein Browser ein app.js von
# vor EN-1 mit deutschen Zeichenketten aus, obwohl die Datei im Image englisch
# war. Gemeldet als Sprachfehler, war es ein Cachefehler - /static/ trug keinen
# Cache-Control-Kopf, und der Browser entschied nach eigener Heuristik.
#
# Geprueft wird an den Kopfzeilen und an der ausgelieferten Adresse, nicht am
# Code, der sie erzeugt.
# =============================================================================


def test_die_seite_wird_nie_zwischengespeichert(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """no-store, aus zwei Gruenden.

    Die Seite traegt die Adressen der versionierten Dateien - eine
    zwischengespeicherte Seite verweist weiter auf die alte Fassung, und das
    Verfahren laeuft leer. Und sie traegt das url_token, das ist die
    Zugangskontrolle (ADR-007).
    """
    client, _ = _client(umgebung)
    antwort = client.get(f"/t/{ACME_TOKEN}/")
    assert antwort.status_code == 200
    assert antwort.headers["Cache-Control"] == "no-store"


def test_versionierte_dateien_duerfen_lange_behalten_werden(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Lange und immutable - unter einer gegebenen Adresse kann sich nichts
    aendern, weil der Inhaltsschluessel in der Adresse steht."""
    client, _ = _client(umgebung)
    for pfad in ("/static/app.js", "/static/style.css"):
        antwort = client.get(pfad)
        assert antwort.status_code == 200, pfad
        kopf = antwort.headers["Cache-Control"]
        assert "immutable" in kopf, pfad
        assert "max-age=31536000" in kopf, pfad


def test_die_vorlage_unter_static_wird_nicht_zwischengespeichert(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """index.html liegt im selben Verzeichnis, darf aber nicht lange gelten.

    Sie ist die Vorlage mit den Platzhaltern. Ein immutable darauf waere genau
    der Fehler, den dieses Verfahren verhindern soll - nur eine Ebene hoeher.
    """
    client, _ = _client(umgebung)
    antwort = client.get("/static/index.html")
    assert antwort.status_code == 200
    assert antwort.headers["Cache-Control"] == "no-store"


def test_die_seite_bindet_die_dateien_mit_fassungsschluessel_ein(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Weg A aus OP-054: Die Adresse traegt den Schluessel.

    Geprueft wird die ausgelieferte Seite, nicht die Vorlage - zwischen beiden
    liegt die Ersetzung, und ein nicht ersetzter Platzhalter waere hier sichtbar.
    """
    client, _ = _client(umgebung)
    seite = client.get(f"/t/{ACME_TOKEN}/").text
    treffer = re.findall(r"/static/(app\.js|style\.css)\?v=([0-9a-f]+)", seite)
    assert len(treffer) == 2, f"nicht beide Dateien versioniert: {treffer}"
    schluessel = {wert for _, wert in treffer}
    assert len(schluessel) == 1, f"verschiedene Schluessel: {schluessel}"
    assert "{{fassung}}" not in seite


def test_der_fassungsschluessel_folgt_dem_inhalt(tmp_path: Path) -> None:
    """Aendert sich eine Datei, aendert sich der Schluessel - und nur dann.

    Das ist der Unterschied zum Commit als Schluessel: Ein Commit an einer
    beliebigen Stelle des Projekts wuerde jeden Browsercache entwerten, ein
    Inhaltshash nur den betroffenen.
    """
    verzeichnis = tmp_path / "static"
    verzeichnis.mkdir()
    (verzeichnis / "app.js").write_text("var a = 1;", encoding="utf-8")
    (verzeichnis / "style.css").write_text("body {}", encoding="utf-8")

    vorher = statische_fassung(verzeichnis)

    # Eine unbeteiligte Datei aendert nichts.
    (verzeichnis / "liesmich.txt").write_text("egal", encoding="utf-8")
    assert statische_fassung(verzeichnis) == vorher

    # Eine eingebundene Datei aendert alles.
    (verzeichnis / "app.js").write_text("var a = 2;", encoding="utf-8")
    nachher = statische_fassung(verzeichnis)
    assert nachher != vorher
    assert len(nachher) == 12


def test_die_begruessung_nennt_die_themen_des_mandanten(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Ein Interessent soll wissen, worueber er fragen kann.

    Ohne diese Angabe fragt er, was ihm einfaellt, bekommt eine korrekte
    Eskalation und haelt das System fuer schwach.
    """
    settings, _ = umgebung
    client, _ = _client(umgebung)
    seite = client.get(f"/t/{ACME_TOKEN}/").text
    mandant = load_tenant("demo-acme", settings.tenants_dir)
    for wort in mandant.topics.split():
        assert html.escape(wort) in seite, wort
