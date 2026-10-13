"""Tests der Verweise auf ein Dokument ueber den Assistenten.

Die Bedingung, gegen die diese Datei gebaut ist: Bei einer Frage, fuer die
kein Fachdokument noetig ist, steht weder unter noch in der Antwort ein
Dokument. Unter der Antwort heisst `sources`, in der Antwort heisst der Text.

Der Pflichttest laeuft ueber `answer()` und nicht nur ueber die Funktion: Ein
Aufrufer, der die Entfernung vergisst, muss ihn rot machen.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from app.config import Settings
from app.embeddings import E5Embeddings
from app.prompts import GroundedAnswer
from app.rag import REASON_NOT_GROUNDED, answer
from app.tenants import TenantConfig
from app.texts import kurzformel_erkennen
from app.verweis import assistentenverweise_entfernen, ist_assistentendokument
from tests.conftest import FAKE_DIMENSION, FakeBackend, FakeLlm

ABOUT = "about-this-assistant.md"
ZWEITES = "how-this-assistant-works.md"

# Eine Frage nach dem Assistenten, die ueber Retrieval und Modell laeuft. Eine
# Frage aus den Listen der Kurzformeln, etwa "What can you help me with?",
# bekaeme eine feste Antwort und erreichte das Modell nie; dann prueften die
# Tests unten nichts.
FRAGE = "Which topics does this assistant cover?"

FACHTEXT = """# Shipping

Standard delivery takes three working days. Express delivery arrives the next
working day if the order is placed before noon.
"""

ABOUTTEXT = """# About This Assistant

This assistant is an AI system. It answers only from the stored documents and
can help with shipping, returns and payment.
"""

ZWEITTEXT = """# How this assistant works

A language model writes the answers from the stored documents.
"""

ESKALATION = "I cannot find anything about that in the documents."

TITEL_EN = "“About This Assistant”"
TITEL_DE = "„Über den Assistenten“"


def _lege_an(root: Path, slug: str, language: str = "en") -> None:
    ordner = root / slug / "docs"
    ordner.mkdir(parents=True)
    (ordner / "doku.md").write_text(FACHTEXT, encoding="utf-8")
    (ordner / ABOUT).write_text(ABOUTTEXT, encoding="utf-8")
    (ordner / ZWEITES).write_text(ZWEITTEXT, encoding="utf-8")
    (root / slug / "tenant.yaml").write_text(
        yaml.safe_dump(
            {
                "display_name": slug,
                "language": language,
                "topics": "shipping, returns and payment",
                "escalation_message": ESKALATION,
                "url_token": f"{slug}-token-1234567890",
                "assistant_documents": [ABOUT, ZWEITES],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def umgebung(tmp_path: Path) -> tuple[Settings, E5Embeddings]:
    from app.ingest import ingest_tenant

    root = tmp_path / "tenants"
    root.mkdir()
    _lege_an(root, "demo-assist")
    settings = Settings(
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=root,
        index_dir=tmp_path / "index",
        embedding_dimension=FAKE_DIMENSION,
        chunk_size=200,
        chunk_overlap=20,
    )
    embeddings = E5Embeddings(FakeBackend(), expected_dimension=FAKE_DIMENSION)
    ingest_tenant("demo-assist", settings=settings, embeddings=embeddings)
    return settings, embeddings


def test_die_frage_dieser_tests_ist_keine_kurzformel() -> None:
    """Gegenprobe zu allen Tests ueber `_antwort`: Die Frage laeuft ueber das Modell."""
    assert kurzformel_erkennen(FRAGE) is None
    assert kurzformel_erkennen("What can you help me with?") == "capabilities"


def _antwort(
    umgebung: tuple[Settings, E5Embeddings],
    text: str,
    sources: list[str] | None = None,
    sprache: str = "en",
    answerable: bool = True,
) -> object:
    settings, embeddings = umgebung
    llm = FakeLlm(
        parsed=GroundedAnswer(
            answerable=answerable,
            answer=text,
            sources=sources if sources is not None else [ABOUT],
            language=sprache,
        )
    )
    return answer(
        "demo-assist",
        FRAGE,
        settings=settings,
        embeddings=embeddings,
        llm=llm,
    )


# Jede Schreibweise der beiden Namen, die nach der Entfernung nicht mehr im
# Text stehen darf: mit und ohne .md, ohne Ruecksicht auf die Schreibung.
_NAME_IRGENDWIE = re.compile(
    r"about-this-assistant(?:\.md)?|how-this-assistant-works(?:\.md)?", re.IGNORECASE
)


def _sauber(text: str) -> None:
    """Regel 5: kein Name, keine leere Klammer, kein doppelter Leerraum, kein
    Satzzeichen ohne Wort davor."""
    assert not _NAME_IRGENDWIE.search(text), text
    assert not re.search(r"\[\s*\]|\(\s*\)", text), text
    assert "  " not in text, text
    assert not re.search(r"(^|\s)[.,;:!?]", text), text
    assert not re.search(r"[.,;:!?][.,;:]", text), text


# --- der Pflichttest --------------------------------------------------------

# (Antworttext des Modells, ausgelieferter Text, Zahl der Verweise)
FORMEN = [
    # Regel 2: Die Angabe nennt nur Assistentendokumente und faellt ganz weg.
    ("I can help with shipping. [Source: about-this-assistant.md]", "I can help with shipping.", 1),
    ("I can help with shipping. [about-this-assistant.md]", "I can help with shipping.", 1),
    ("I can help with shipping. Source: about-this-assistant.md.", "I can help with shipping.", 1),
    (
        "I can help with shipping. Sources: about-this-assistant.md; how-this-assistant-works.md.",
        "I can help with shipping.",
        2,
    ),
    # dieselbe Form am Textende, ohne Punkt
    (
        "I can help with shipping. Sources: about-this-assistant.md; how-this-assistant-works.md",
        "I can help with shipping.",
        2,
    ),
    ("I can help with shipping (Source: about-this-assistant.md).", "I can help with shipping.", 1),
    ("I can help with shipping (about-this-assistant.md).", "I can help with shipping.", 1),
    ("I can help with shipping. [Quelle: about-this-assistant.md]", "I can help with shipping.", 1),
    ("I can help with shipping. Quellen: about-this-assistant.md.", "I can help with shipping.", 1),
    ("I can help with shipping. [ABOUT-THIS-ASSISTANT.MD]", "I can help with shipping.", 1),
    ("I can help with shipping. [Source: about-this-assistant]", "I can help with shipping.", 1),
    ("[Source: about-this-assistant.md] I can help with shipping.", "I can help with shipping.", 1),
    ("I can help with shipping [about-this-assistant.md].", "I can help with shipping.", 1),
    (
        "I can help with shipping.\n\n[Source: about-this-assistant.md]\n\nAsk me anything.",
        "I can help with shipping.\n\nAsk me anything.",
        1,
    ),
    # Regel 3: Die Angabe nennt auch eine andere Datei. Nur der Name samt
    # Trennzeichen faellt weg.
    (
        "Delivery takes three days. Sources: doku.md; about-this-assistant.md.",
        "Delivery takes three days. Sources: doku.md.",
        1,
    ),
    (
        "Delivery takes three days. [Source: about-this-assistant.md; doku.md]",
        "Delivery takes three days. [Source: doku.md]",
        1,
    ),
    (
        "Delivery takes three days [doku.md, about-this-assistant.md].",
        "Delivery takes three days [doku.md].",
        1,
    ),
    # Regel 4: Der Name steht im Fliesstext und wird durch den Titel ersetzt.
    (
        "According to about-this-assistant.md, I can help with shipping.",
        f"According to {TITEL_EN}, I can help with shipping.",
        1,
    ),
    (
        "Source: about-this-assistant.md says that I answer from the documents.",
        f"Source: {TITEL_EN} says that I answer from the documents.",
        1,
    ),
]


@pytest.mark.parametrize(("modelltext", "erwartet", "anzahl"), FORMEN)
def test_kein_assistentendokument_im_ausgelieferten_text(
    umgebung: tuple[Settings, E5Embeddings], modelltext: str, erwartet: str, anzahl: int
) -> None:
    """Ueber answer(): Der Name steht in keiner Schreibweise im Text, und alles
    andere ist Zeichen fuer Zeichen das, was das Modell geschrieben hat."""
    a = _antwort(umgebung, modelltext)

    assert not _NAME_IRGENDWIE.search(a.text), a.text
    assert a.text == erwartet
    assert a.removed_references == anzahl
    _sauber(a.text)


def test_fremde_dateinamen_bleiben_unveraendert(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Nur die Namen aus assistant_documents. Ein Name, der einen von ihnen
    enthaelt, ist ein anderer Name."""
    texte = [
        "Delivery takes three days. [Source: doku.md]",
        "Delivery takes three days. Sources: doku.md; faq.md.",
        "See my-about-this-assistant.md, about-this-assistant-v2.md and about-this-assistant.mdx.",
        "Delivery takes three days. [Note] (see below)",
    ]
    for text in texte:
        a = _antwort(umgebung, text, sources=["doku.md"])
        assert a.text == text
        assert a.removed_references == 0


def test_eine_eskalierte_antwort_bleibt_unveraendert(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Regel 1: Eine Eskalation traegt den Text des Mandanten, und der wird
    nicht angefasst. Was das Modell geschrieben hat, wird nie ausgeliefert."""
    a = _antwort(umgebung, "Source: about-this-assistant.md says nothing.", answerable=False)
    assert a.escalated is True
    assert a.escalation_reason == REASON_NOT_GROUNDED
    assert a.text == ESKALATION
    assert a.removed_references == 0


def test_zweimal_angewendet_wie_einmal() -> None:
    """Regel 6, fuer jede Form des Pflichttests."""
    tenant = _tenant()
    for modelltext, _, _ in FORMEN:
        einmal, _ = assistentenverweise_entfernen("demo-assist", tenant, modelltext, "en")
        zweimal, n = assistentenverweise_entfernen("demo-assist", tenant, einmal, "en")
        assert zweimal == einmal, modelltext
        assert n == 0, modelltext


# --- Titel, Sprache und die Grenzen ------------------------------------------


def _tenant(language: str = "en") -> TenantConfig:
    return TenantConfig(
        slug="demo-assist",
        display_name="Assist",
        language=language,
        topics="shipping and returns" if language == "en" else "Versand und Retouren",
        escalation_message=ESKALATION if language == "en" else "Dazu finde ich nichts.",
        url_token="assist-token-1234567890",
        assistant_documents=[ABOUT, ZWEITES],
    )


def test_der_titel_folgt_der_antwortsprache() -> None:
    """de und en nach dem Feld language der Antwort, sonst die Mandantensprache."""
    text = "Laut about-this-assistant.md helfe ich bei Versand."
    de, _ = assistentenverweise_entfernen("demo-assist", _tenant("en"), text, "de")
    assert de == f"Laut {TITEL_DE} helfe ich bei Versand."

    en, _ = assistentenverweise_entfernen("demo-assist", _tenant("de"), text, "EN")
    assert en == f"Laut {TITEL_EN} helfe ich bei Versand."

    fremd, _ = assistentenverweise_entfernen("demo-assist", _tenant("de"), text, "fr")
    assert fremd == f"Laut {TITEL_DE} helfe ich bei Versand."
    fremd, _ = assistentenverweise_entfernen("demo-assist", _tenant("en"), text, "")
    assert fremd == f"Laut {TITEL_EN} helfe ich bei Versand."


def test_die_titel_stehen_mit_echten_zeichen_im_katalog() -> None:
    """Deutsche Anfuehrungszeichen und ein echtes Ü, englische typografische."""
    from app.texts import TEXTE

    assert TEXTE["de"].assistentendokument_titel == "„Über den Assistenten“"
    assert TEXTE["en"].assistentendokument_titel == "“About This Assistant”"


def test_waere_der_text_leer_bleibt_er_unveraendert() -> None:
    """Regel 5, letzter Satz: lieber ein Dateiname als eine leere Blase."""
    tenant = _tenant()
    for text in ("[Source: about-this-assistant.md]", "Source: about-this-assistant.md."):
        assert assistentenverweise_entfernen("demo-assist", tenant, text, "en") == (text, 0)


def test_ohne_assistentendokumente_bleibt_alles_wie_es_ist() -> None:
    """Ein Mandant ohne das Feld: Text unveraendert, null Verweise."""
    tenant = _tenant().model_copy(update={"assistant_documents": []})
    text = "I can help. [Source: about-this-assistant.md]"
    assert assistentenverweise_entfernen("demo-assist", tenant, text, "en") == (text, 0)


def test_die_funktion_prueft_den_mandanten() -> None:
    """Der Text eines Mandanten geht nie unter der tenant_id eines anderen durch."""
    with pytest.raises(ValueError):
        assistentenverweise_entfernen("demo-anderer", _tenant(), "x", "en")


def test_ist_assistentendokument() -> None:
    namen = [ABOUT]
    for name in ("about-this-assistant.md", "About-This-Assistant.MD", "about-this-assistant"):
        assert ist_assistentendokument(name, namen), name
    for name in ("faq.md", "my-about-this-assistant.md", "about-this-assistant-v2.md", ""):
        assert not ist_assistentendokument(name, namen), name
    assert not ist_assistentendokument("about-this-assistant.md", [])
