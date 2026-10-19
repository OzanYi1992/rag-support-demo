"""Tests der Verweise auf Dokumente des Mandanten im Antworttext.

Die Bedingung, gegen die diese Datei gebaut ist: Im Antworttext steht kein
Dateiname eines Dokuments des Mandanten. Die Quellen stehen unter der Antwort,
zum Aufklappen. Fuer das Dokument ueber den Assistenten gilt dazu: Es steht
weder unter noch in der Antwort.

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
from app.tenants import InvalidTenantSlug, TenantConfig
from app.texts import kurzformel_erkennen
from app.verweis import (
    Mandantendokumente,
    ist_assistentendokument,
    mandantendokumente,
    quellverweise_entfernen,
)
from tests.conftest import FAKE_DIMENSION, FakeBackend, FakeLlm

ABOUT = "about-this-assistant.md"
ZWEITES = "how-this-assistant-works.md"
FACH = "doku.md"
# Eine Datei, die es bei keinem Mandanten gibt: eine erfundene Quelle.
ERFUNDEN = "made-up.md"
# Ein Dokument des zweiten Mandanten.
FREMD = "retouren.md"

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

FREMDTEXT = """# Retouren

Jede Ruecksendung braucht eine RMA-Nummer.
"""

ESKALATION = "I cannot find anything about that in the documents."

TITEL_EN = "“About This Assistant”"
TITEL_DE = "„Über den Assistenten“"
FACHTITEL_EN = "“Shipping”"
FACHTITEL_DE = "„Shipping“"


def _lege_an(
    root: Path, slug: str, dateien: dict[str, str], assistent: list[str], language: str = "en"
) -> None:
    ordner = root / slug / "docs"
    ordner.mkdir(parents=True)
    for name, text in dateien.items():
        (ordner / name).write_text(text, encoding="utf-8")
    (root / slug / "tenant.yaml").write_text(
        yaml.safe_dump(
            {
                "display_name": slug,
                "language": language,
                "topics": "shipping, returns and payment"
                if language == "en"
                else "Versand und Retouren",
                "escalation_message": ESKALATION
                if language == "en"
                else "Dazu finde ich in den Unterlagen nichts.",
                "url_token": f"{slug}-token-1234567890",
                "assistant_documents": assistent,
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
    _lege_an(
        root,
        "demo-assist",
        {FACH: FACHTEXT, ABOUT: ABOUTTEXT, ZWEITES: ZWEITTEXT},
        [ABOUT, ZWEITES],
    )
    _lege_an(root, "demo-fremd", {FREMD: FREMDTEXT}, [], language="de")
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
    ingest_tenant("demo-fremd", settings=settings, embeddings=embeddings)
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
    slug: str = "demo-assist",
) -> object:
    settings, embeddings = umgebung
    llm = FakeLlm(
        parsed=GroundedAnswer(
            answerable=answerable,
            answer=text,
            sources=sources if sources is not None else [FACH],
            language=sprache,
        )
    )
    return answer(
        slug,
        FRAGE,
        settings=settings,
        embeddings=embeddings,
        llm=llm,
    )


# Jede Schreibweise der Namen, die nach der Entfernung nicht mehr im Text
# stehen darf: die Assistentendokumente mit und ohne .md, das Fachdokument mit
# .md, ohne Ruecksicht auf die Schreibung.
_NAME_IRGENDWIE = re.compile(
    r"about-this-assistant(?:\.md)?|how-this-assistant-works(?:\.md)?|doku\.md", re.IGNORECASE
)


def _sauber(text: str) -> None:
    """Regel 5: kein Name, keine leere Klammer, kein doppelter Leerraum, kein
    Satzzeichen ohne Wort davor, kein Backtick, der zu einem Namen gehoerte."""
    assert not _NAME_IRGENDWIE.search(text), text
    assert not re.search(r"\[\s*\]|\(\s*\)", text), text
    assert "  " not in text, text
    assert not re.search(r"(^|\s)[.,;:!?]", text), text
    assert not re.search(r"[.,;:!?][.,;:]", text), text
    assert "``" not in text and "`“" not in text and "”`" not in text, text


# --- der Pflichttest --------------------------------------------------------

# (Antworttext des Modells, ausgelieferter Text, Zahl der Namen)
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
    # Regel 3: Die Angabe nennt auch eine Datei, die kein Dokument des
    # Mandanten ist. Nur die Namen des Mandanten samt Trennzeichen fallen weg.
    (
        "Delivery takes three days. Sources: made-up.md; about-this-assistant.md.",
        "Delivery takes three days. Sources: made-up.md.",
        1,
    ),
    (
        "Delivery takes three days. [Source: about-this-assistant.md; made-up.md]",
        "Delivery takes three days. [Source: made-up.md]",
        1,
    ),
    (
        "Delivery takes three days [made-up.md, about-this-assistant.md].",
        "Delivery takes three days [made-up.md].",
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
    # --- jedes Dokument des Mandanten, in den Formen aus den Antworttexten ---
    ("Delivery takes three days. [doku.md]", "Delivery takes three days.", 1),
    ("Delivery takes three days. [Source: doku.md]", "Delivery takes three days.", 1),
    ("Delivery takes three days (doku.md).", "Delivery takes three days.", 1),
    ("Delivery takes three days. Quelle: doku.md.", "Delivery takes three days.", 1),
    (
        "Delivery takes three days. Quelle: doku.md. Express is faster.",
        "Delivery takes three days. Express is faster.",
        1,
    ),
    ("Delivery takes three days. Source: doku.md", "Delivery takes three days.", 1),
    (
        "Delivery takes three days. [doku.md] Express is faster. [doku.md]",
        "Delivery takes three days. Express is faster.",
        2,
    ),
    # mehrere Namen, getrennt durch Komma, "and" oder "und"
    (
        "Delivery takes three days. Sources: doku.md, about-this-assistant.md",
        "Delivery takes three days.",
        2,
    ),
    (
        "Delivery takes three days. Sources: doku.md and about-this-assistant.md.",
        "Delivery takes three days.",
        2,
    ),
    (
        "Delivery takes three days. Quellen: doku.md und how-this-assistant-works.md.",
        "Delivery takes three days.",
        2,
    ),
    (
        "Delivery takes three days. "
        "Sources: doku.md, how-this-assistant-works.md, and about-this-assistant.md.",
        "Delivery takes three days.",
        3,
    ),
    (
        "Delivery takes three days. [doku.md, about-this-assistant.md]",
        "Delivery takes three days.",
        2,
    ),
    (
        "Delivery takes three days. [doku.md; about-this-assistant.md]",
        "Delivery takes three days.",
        2,
    ),
    (
        "Delivery takes three days. [doku.md and how-this-assistant-works.md]",
        "Delivery takes three days.",
        2,
    ),
    # angrenzende Klammern
    (
        "Delivery takes three days. [doku.md][about-this-assistant.md]",
        "Delivery takes three days.",
        2,
    ),
    (
        "Delivery takes three days. [doku.md] [about-this-assistant.md] Ask me anything.",
        "Delivery takes three days. Ask me anything.",
        2,
    ),
    # Backticks gehoeren zum Namen
    ("Delivery takes three days. Source: `doku.md`.", "Delivery takes three days.", 1),
    ("Delivery takes three days. [`doku.md`]", "Delivery takes three days.", 1),
    (
        "According to `doku.md`, delivery takes three days.",
        f"According to {FACHTITEL_EN}, delivery takes three days.",
        1,
    ),
    # Fliesstext: der Titel aus der ersten Zeile mit "# "
    (
        "According to doku.md, delivery takes three days.",
        f"According to {FACHTITEL_EN}, delivery takes three days.",
        1,
    ),
    (
        "According to DOKU.MD, delivery takes three days.",
        f"According to {FACHTITEL_EN}, delivery takes three days.",
        1,
    ),
    # Regel 3 mit einer erfundenen Quelle neben dem Fachdokument
    (
        "Delivery takes three days. [Source: doku.md; made-up.md]",
        "Delivery takes three days. [Source: made-up.md]",
        1,
    ),
    (
        "Delivery takes three days. Sources: made-up.md and doku.md.",
        "Delivery takes three days. Sources: made-up.md.",
        1,
    ),
]


@pytest.mark.parametrize(("modelltext", "erwartet", "anzahl"), FORMEN)
def test_kein_dokumentname_im_ausgelieferten_text(
    umgebung: tuple[Settings, E5Embeddings], modelltext: str, erwartet: str, anzahl: int
) -> None:
    """Ueber answer(): Kein Name eines Dokuments des Mandanten steht im Text, und
    alles andere ist Zeichen fuer Zeichen das, was das Modell geschrieben hat."""
    a = _antwort(umgebung, modelltext)

    assert not _NAME_IRGENDWIE.search(a.text), a.text
    assert a.text == erwartet
    assert a.removed_references == anzahl
    _sauber(a.text)


def test_andere_namen_bleiben_unveraendert(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Nur die Namen der Dokumente dieses Mandanten. Ein Name, der einen von
    ihnen enthaelt, ist ein anderer Name, und der Stamm eines Fachdokuments ohne
    .md ist ein Wort."""
    texte = [
        "Delivery takes three days. [Source: made-up.md]",
        "Delivery takes three days. Sources: made-up.md; faq.md.",
        "See my-about-this-assistant.md, about-this-assistant-v2.md and about-this-assistant.mdx.",
        "See my-doku.md, doku-v2.md and doku.mdx.",
        "Die Doku sagt, die Lieferung dauert drei Tage. (Doku)",
        "Delivery takes three days. [Note] (see below)",
    ]
    for text in texte:
        a = _antwort(umgebung, text)
        assert a.text == text
        assert a.removed_references == 0


def test_der_name_eines_anderen_mandanten_bleibt_stehen(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Die Dokumentliste ist die DIESES Mandanten. retouren.md gehoert zu
    demo-fremd: in einer Antwort von demo-assist ist es ein anderer Name und
    bleibt sichtbar, in einer von demo-fremd faellt es weg."""
    text = "Delivery takes three days. [retouren.md]"
    a = _antwort(umgebung, text)
    assert (a.text, a.removed_references) == (text, 0)

    b = _antwort(umgebung, text, sources=[FREMD], sprache="de", slug="demo-fremd")
    assert (b.text, b.removed_references) == ("Delivery takes three days.", 1)


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


# --- die Funktion direkt -----------------------------------------------------


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


def _dokumente(tenant_id: str = "demo-assist") -> Mandantendokumente:
    return Mandantendokumente(
        tenant_id=tenant_id,
        titel={
            FACH: "Shipping",
            ABOUT: "About This Assistant",
            ZWEITES: "How this assistant works",
        },
    )


def _entfernen(
    text: str, sprache: str = "en", tenant: TenantConfig | None = None
) -> tuple[str, int]:
    return quellverweise_entfernen("demo-assist", tenant or _tenant(), text, sprache, _dokumente())


def test_zweimal_angewendet_wie_einmal() -> None:
    """Regel 6, fuer jede Form des Pflichttests."""
    for modelltext, _, _ in FORMEN:
        einmal, _ = _entfernen(modelltext)
        zweimal, n = _entfernen(einmal)
        assert zweimal == einmal, modelltext
        assert n == 0, modelltext


def test_der_titel_folgt_der_antwortsprache() -> None:
    """de und en nach dem Feld language der Antwort, sonst die Mandantensprache.
    Der Titel eines Fachdokuments bleibt der aus der Datei, die
    Anfuehrungszeichen folgen der Antwortsprache."""
    text = "Laut about-this-assistant.md helfe ich bei Versand."
    de, _ = _entfernen(text, "de", _tenant("en"))
    assert de == f"Laut {TITEL_DE} helfe ich bei Versand."

    en, _ = _entfernen(text, "EN", _tenant("de"))
    assert en == f"Laut {TITEL_EN} helfe ich bei Versand."

    fremd, _ = _entfernen(text, "fr", _tenant("de"))
    assert fremd == f"Laut {TITEL_DE} helfe ich bei Versand."
    fremd, _ = _entfernen(text, "", _tenant("en"))
    assert fremd == f"Laut {TITEL_EN} helfe ich bei Versand."

    fach = "Laut doku.md dauert die Lieferung drei Tage."
    assert _entfernen(fach, "de") == (f"Laut {FACHTITEL_DE} dauert die Lieferung drei Tage.", 1)
    assert _entfernen(fach, "en") == (f"Laut {FACHTITEL_EN} dauert die Lieferung drei Tage.", 1)


def test_die_titel_stehen_mit_echten_zeichen_im_katalog() -> None:
    """Deutsche Anfuehrungszeichen U+201E und U+201C, englische U+201C und
    U+201D, und ein echtes Ü."""
    from app.texts import TEXTE

    assert TEXTE["de"].assistentendokument_titel == "„Über den Assistenten“"
    assert TEXTE["en"].assistentendokument_titel == "“About This Assistant”"
    assert TEXTE["de"].dokumenttitel_im_text == "„{titel}“"
    assert TEXTE["en"].dokumenttitel_im_text == "“{titel}”"


def test_waere_der_text_leer_bleibt_er_unveraendert() -> None:
    """Regel 5, letzter Satz: lieber ein Dateiname als eine leere Blase."""
    for text in (
        "[Source: about-this-assistant.md]",
        "Source: about-this-assistant.md.",
        "[doku.md]",
        "Sources: doku.md and about-this-assistant.md",
    ):
        assert _entfernen(text) == (text, 0)


def test_ohne_dokumente_bleibt_alles_wie_es_ist() -> None:
    """Ein Mandant ohne Assistentendokumente und mit leerer Dokumentliste: Text
    unveraendert, null Namen."""
    tenant = _tenant().model_copy(update={"assistant_documents": []})
    leer = Mandantendokumente(tenant_id="demo-assist", titel={})
    text = "I can help. [Source: about-this-assistant.md] [doku.md]"
    assert quellverweise_entfernen("demo-assist", tenant, text, "en", leer) == (text, 0)


def test_die_funktion_prueft_den_mandanten() -> None:
    """Der Text eines Mandanten geht nie unter der tenant_id eines anderen
    durch, und die Dokumentliste eines anderen nie in seinen Text."""
    with pytest.raises(ValueError):
        quellverweise_entfernen("demo-anderer", _tenant(), "x", "en", _dokumente("demo-anderer"))
    with pytest.raises(ValueError):
        quellverweise_entfernen("demo-assist", _tenant(), "x", "en", _dokumente("demo-anderer"))
    assert _entfernen("x") == ("x", 0)


def test_ist_assistentendokument() -> None:
    namen = [ABOUT]
    for name in ("about-this-assistant.md", "About-This-Assistant.MD", "about-this-assistant"):
        assert ist_assistentendokument(name, namen), name
    for name in ("faq.md", "my-about-this-assistant.md", "about-this-assistant-v2.md", ""):
        assert not ist_assistentendokument(name, namen), name
    assert not ist_assistentendokument("about-this-assistant.md", [])


# --- die Dokumentliste ---------------------------------------------------------


def test_mandantendokumente_liest_namen_und_titel(tmp_path: Path) -> None:
    """Nur die *.md im Ordner dieses Mandanten. Titel ist die erste Zeile mit
    "# ", ohne diese Zeichen; fehlt sie, der Name ohne .md."""
    ordner = tmp_path / "demo-assist" / "docs"
    ordner.mkdir(parents=True)
    (ordner / "mit-titel.md").write_text("Vorspann\n# Where We Ship  \n\n# Zweiter\n", "utf-8")
    (ordner / "ohne-titel.md").write_text("Nur Text\n## Unterpunkt\n", "utf-8")
    (ordner / "notiz.txt").write_text("# Keine Markdowndatei\n", "utf-8")
    (ordner / "unterordner.md").mkdir()
    anderer = tmp_path / "demo-anderer" / "docs"
    anderer.mkdir(parents=True)
    (anderer / "fremd.md").write_text("# Fremd\n", "utf-8")

    dokumente = mandantendokumente("demo-assist", tmp_path)
    assert dokumente.tenant_id == "demo-assist"
    assert dokumente.titel == {"mit-titel.md": "Where We Ship", "ohne-titel.md": "ohne-titel"}

    assert mandantendokumente("demo-ohne-ordner", tmp_path).titel == {}


def test_mandantendokumente_prueft_den_slug(tmp_path: Path) -> None:
    """Der Ordner wird aus der tenant_id aufgeloest und nie aus einem Pfad, der
    hinausfuehrt."""
    for falsch in ("../demo-assist", "demo/assist", "Demo-Assist", ""):
        with pytest.raises(InvalidTenantSlug):
            mandantendokumente(falsch, tmp_path)
