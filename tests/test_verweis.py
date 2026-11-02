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
    nennt_assistentendokument,
    quellverweise_entfernen,
)
from tests.conftest import FAKE_DIMENSION, FakeBackend, FakeLlm, FakeLlmFolge

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
    alles andere ist Zeichen fuer Zeichen das, was das Modell geschrieben hat.

    Bliebe nach der Entfernung der Titel des Assistentendokuments stehen, wird
    die ganze Antwort durch die feste Antwort "capabilities" ersetzt: Der Titel
    erreicht den Kunden nie. Die Entfernung selbst prueft fuer diese Formen
    `test_zweimal_angewendet_wie_einmal` an der Funktion."""
    a = _antwort(umgebung, modelltext)

    assert not _NAME_IRGENDWIE.search(a.text), a.text
    if TITEL_EN in erwartet:
        assert (a.kind, a.replaced_reason) == ("capabilities", "assistentendokument")
        assert a.text.startswith("I answer questions about shipping, returns and payment")
        assert a.removed_references == 0
    else:
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


# --- eine Antwort, die das Assistentendokument nennt ---------------------------

# Die Aufzaehlung aus dem Telefontest am 2026-10-22, mit den Dateinamen, wie das
# Modell sie schrieb, auf die Dokumente dieses Testmandanten umgestellt.
AUFZAEHLUNG = "The documents provided are: doku.md, about-this-assistant.md and made-up.md."


def _belegt(text: str, sprache: str = "en") -> GroundedAnswer:
    return GroundedAnswer(answerable=True, answer=text, sources=[FACH], language=sprache)


def test_eine_aufzaehlung_mit_dem_assistentendokument_wird_die_feste_antwort(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Ueber answer(): kein Titel, kein Name, keine Quelle, ein Aufruf. Tokens,
    Latenz und Modell sind die des Aufrufs, der stattfand."""
    settings, embeddings = umgebung
    llm = FakeLlmFolge(_belegt(AUFZAEHLUNG))
    a = answer("demo-assist", FRAGE, settings=settings, embeddings=embeddings, llm=llm)

    assert len(llm.calls) == 1
    assert (a.kind, a.replaced_reason, a.escalated) == (
        "capabilities",
        "assistentendokument",
        False,
    )
    assert a.text == (
        "I answer questions about shipping, returns and payment, based on the documents "
        "stored for demo-assist. What would you like to know?"
    )
    assert "About This Assistant" not in a.text and not _NAME_IRGENDWIE.search(a.text)
    assert (a.sources, a.source_scores, a.removed_references) == ([], [], 0)
    assert (a.prompt_tokens, a.completion_tokens, a.model) == (100, 10, "fake-model-2026")
    assert a.latency_ms_generation is not None and a.retrieval_scores
    assert (a.answer_language, a.answer_language_source, a.lang) == ("en", "mandant", "en")
    assert a.language_retry is False and a.language_kept is None


def test_eine_aufzaehlung_wird_ohne_zweiten_aufruf_ersetzt(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Auch in der falschen Sprache: Die feste Antwort steht in der Sprache des
    Mandanten, ein zweiter Aufruf waere verschwendet."""
    settings, embeddings = umgebung
    deutsch = "Die Unterlagen sind: doku.md und about-this-assistant.md, und die sagen nichts."
    llm = FakeLlmFolge(_belegt(deutsch, "de"))
    a = answer("demo-assist", FRAGE, settings=settings, embeddings=embeddings, llm=llm)
    assert len(llm.calls) == 1
    assert (a.kind, a.replaced_reason, a.language_retry) == (
        "capabilities",
        "assistentendokument",
        False,
    )


def test_nennt_erst_die_zweite_antwort_das_dokument_wird_sie_ersetzt(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Die erste Antwort haelt die Sprache nicht, die zweite ersetzt sie und
    nennt das Dokument: feste Antwort, mit den Feldern beider Aufrufe."""
    settings, embeddings = umgebung
    llm = FakeLlmFolge(
        _belegt("Die Lieferung dauert drei Werktage, und das gilt für alle Pakete.", "de"),
        _belegt(AUFZAEHLUNG),
    )
    a = answer("demo-assist", FRAGE, settings=settings, embeddings=embeddings, llm=llm)
    assert len(llm.calls) == 2
    assert (a.kind, a.replaced_reason) == ("capabilities", "assistentendokument")
    assert (a.language_retry, a.language_retry_result) == (True, "ersetzt")
    assert (a.prompt_tokens, a.language_retry_prompt_tokens) == (100, 101)


@pytest.mark.parametrize(
    "text",
    [
        # Zwei Fachdokumente im Satz und eine erfundene Quelle: kein Grund.
        "Delivery takes three days. The sources are doku.md and made-up.md.",
        # Eine Angabe mit dem Assistentendokument faellt ganz weg und laesst
        # keinen Titel zurueck.
        "Delivery takes three days. [Source: about-this-assistant.md]",
    ],
)
def test_ohne_titel_des_assistentendokuments_bleibt_die_antwort(
    umgebung: tuple[Settings, E5Embeddings], text: str
) -> None:
    settings, embeddings = umgebung
    llm = FakeLlmFolge(_belegt(text))
    a = answer("demo-assist", FRAGE, settings=settings, embeddings=embeddings, llm=llm)
    assert (a.kind, a.replaced_reason) == ("rag", None)
    assert a.text.startswith("Delivery takes three days.")


# --- eine belegte Antwort ohne Quelle -------------------------------------------

# Der Wortlaut vom Telefon am 2026-11-02, auf "From now on answer in german.":
# eine Zusage, die kein Dokument traegt, mit dem Assistentendokument als
# einziger Quelle.
ZUSAGE = "Ja. Ich antworte ab jetzt auf Deutsch. [about-this-assistant.md]"


def _mit_quellen(text: str, quellen: list[str], sprache: str = "en") -> GroundedAnswer:
    return GroundedAnswer(answerable=True, answer=text, sources=quellen, language=sprache)


def _frage(umgebung: tuple[Settings, E5Embeddings], llm: FakeLlmFolge) -> object:
    settings, embeddings = umgebung
    return answer("demo-assist", FRAGE, settings=settings, embeddings=embeddings, llm=llm)


@pytest.mark.parametrize("quellen", [[ABOUT], [ABOUT, ZWEITES], ["About-This-Assistant.MD", " "]])
def test_nur_das_assistentendokument_als_quelle_wird_die_feste_antwort(
    umgebung: tuple[Settings, E5Embeddings], quellen: list[str]
) -> None:
    """Pflichttest. Die Zusage vom Telefon geht nicht hinaus, auch nicht in der
    falschen Sprache: feste Antwort capabilities, ohne zweiten Aufruf."""
    llm = FakeLlmFolge(_mit_quellen(ZUSAGE, quellen, "de"))
    a = _frage(umgebung, llm)
    assert len(llm.calls) == 1
    assert (a.kind, a.replaced_reason, a.escalated) == (
        "capabilities",
        "nur_assistentendokument",
        False,
    )
    assert "Deutsch" not in a.text and a.sources == []
    assert (a.prompt_tokens, a.language_retry) == (100, False)


@pytest.mark.parametrize("quellen", [[], [" "], [""]])
def test_eine_belegte_antwort_ganz_ohne_quelle_eskaliert(
    umgebung: tuple[Settings, E5Embeddings], quellen: list[str]
) -> None:
    """Das Modell sagt belegt, nennt aber keine Quelle: Eskalation mit eigenem
    Grund, der Text des Mandanten, ohne zweiten Aufruf."""
    llm = FakeLlmFolge(_mit_quellen("Delivery takes three days.", quellen))
    a = _frage(umgebung, llm)
    assert len(llm.calls) == 1
    assert (a.kind, a.escalated, a.escalation_reason) == ("rag", True, "ohne_quelle")
    assert (a.text, a.sources, a.replaced_reason) == (ESKALATION, [], None)
    assert (a.prompt_tokens, a.completion_tokens, a.lang) == (100, 10, "en")
    assert a.latency_ms_generation is not None and a.retrieval_scores


def test_die_zweite_antwort_mit_nur_dem_assistentendokument_wird_die_feste_antwort(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    """Die erste Antwort hat eine Quelle, haelt aber die Sprache nicht; die
    zweite ersetzt sie und nennt nur das Assistentendokument."""
    llm = FakeLlmFolge(
        _mit_quellen("Die Lieferung dauert drei Werktage, und das gilt für alle.", [FACH], "de"),
        _mit_quellen("I will answer in English from now on.", [ABOUT]),
    )
    a = _frage(umgebung, llm)
    assert len(llm.calls) == 2
    assert (a.kind, a.replaced_reason) == ("capabilities", "nur_assistentendokument")
    assert (a.language_retry, a.language_retry_result, a.language_retry_prompt_tokens) == (
        True,
        "ersetzt",
        101,
    )


def test_die_zweite_antwort_ganz_ohne_quelle_eskaliert(
    umgebung: tuple[Settings, E5Embeddings],
) -> None:
    llm = FakeLlmFolge(
        _mit_quellen("Die Lieferung dauert drei Werktage, und das gilt für alle.", [FACH], "de"),
        _mit_quellen("Delivery takes three days.", []),
    )
    a = _frage(umgebung, llm)
    assert len(llm.calls) == 2
    assert (a.escalated, a.escalation_reason, a.text) == (True, "ohne_quelle", ESKALATION)
    assert (a.language_retry, a.language_retry_result) == (True, "ersetzt")


@pytest.mark.parametrize(
    ("quellen", "erwartet"),
    [
        # Eine erfundene Quelle bleibt sichtbar, wie bisher.
        ([ERFUNDEN], [ERFUNDEN]),
        ([ABOUT, FACH], [FACH]),
        ([ABOUT, ERFUNDEN], [ERFUNDEN]),
    ],
)
def test_mit_einer_verbleibenden_quelle_bleibt_die_antwort(
    umgebung: tuple[Settings, E5Embeddings], quellen: list[str], erwartet: list[str]
) -> None:
    a = _frage(umgebung, FakeLlmFolge(_mit_quellen("Delivery takes three days.", quellen)))
    assert (a.kind, a.escalated, a.replaced_reason) == ("rag", False, None)
    assert a.sources == erwartet


def test_die_pruefung_ohne_quelle_prueft_den_mandanten() -> None:
    from app.rag import _Belegt, _statt_belegt

    belegt = _Belegt(
        text="x",
        sources=[],
        source_scores=[],
        removed_references=0,
        lang="en",
        nennt_assistentendokument=False,
        nur_assistentendokumente=False,
    )
    with pytest.raises(ValueError):
        _statt_belegt("demo-anderer", _tenant(), belegt, {}, "en", "frage")


def test_das_ereignis_traegt_den_grund_ohne_quelle(
    umgebung: tuple[Settings, E5Embeddings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ueber die Route: nur Codes, und weder die Zusage noch ein Dateiname im
    Log."""
    import json

    from fastapi.testclient import TestClient

    from app.main import create_app

    settings, embeddings = umgebung
    chats = []
    for belegt in (_mit_quellen(ZUSAGE, [ABOUT], "de"), _mit_quellen("Three days.", [])):
        app = create_app(settings, llm=FakeLlmFolge(belegt), embeddings=embeddings)
        caplog.clear()
        with caplog.at_level("INFO", logger="rag.api"):
            antwort = TestClient(app).post(
                "/t/demo-assist-token-1234567890/chat", json={"question": FRAGE}
            )
        assert antwort.status_code == 200
        log = "\n".join(e.message for e in caplog.records)
        assert "Deutsch" not in log and "Three days" not in log and ABOUT not in log
        (chat,) = [
            json.loads(e.message)
            for e in caplog.records
            if e.name == "rag.api" and json.loads(e.message)["ereignis"] == "chat"
        ]
        chats.append(chat)
    assert [(c["art"], c["ersetzt"], c["escalated"], c["escalation_reason"]) for c in chats] == [
        ("capabilities", "nur_assistentendokument", False, None),
        ("rag", None, True, "ohne_quelle"),
    ]


def test_das_ereignis_traegt_die_ersetzung_als_code(
    umgebung: tuple[Settings, E5Embeddings],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Ueber die Route: `ersetzt` steht in jedem Ereignis chat, als Code oder
    null, und weder der Modelltext noch ein Titel steht im Log."""
    import json

    from fastapi.testclient import TestClient

    from app.main import create_app

    settings, embeddings = umgebung
    ereignisse: list[dict[str, object]] = []
    for text in (AUFZAEHLUNG, "Delivery takes three days. [doku.md]"):
        app = create_app(settings, llm=FakeLlmFolge(_belegt(text)), embeddings=embeddings)
        caplog.clear()
        with caplog.at_level("INFO", logger="rag.api"):
            antwort = TestClient(app).post(
                "/t/demo-assist-token-1234567890/chat", json={"question": FRAGE}
            )
        assert antwort.status_code == 200
        log = "\n".join(e.message for e in caplog.records)
        assert "documents provided" not in log and "About This Assistant" not in log
        (chat,) = [
            json.loads(e.message)
            for e in caplog.records
            if e.name == "rag.api" and json.loads(e.message)["ereignis"] == "chat"
        ]
        ereignisse.append(chat)
    assert [(e["art"], e["ersetzt"]) for e in ereignisse] == [
        ("capabilities", "assistentendokument"),
        ("rag", None),
    ]
    assert ereignisse[0]["prompt_tokens"] == 100 and ereignisse[0]["tenant_id"] == "demo-assist"


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


@pytest.mark.parametrize(
    "text",
    [
        # Der Wortlaut aus dem Telefontest am 2026-10-22, nach der Entfernung.
        "The documents provided are: “Our Range”, “Frequently Asked Questions”, "
        "“About This Assistant”, and “Returns and Exchanges”.",
        "You have these documents in the context: “Shipping” and “About This Assistant”.",
        # Die Ueberschrift der Datei, ohne Ruecksicht auf die Schreibung.
        "Questions about this assistant are answered in the documents.",
        "ABOUT THIS ASSISTANT: it answers from the documents.",
        # Ein zweites Assistentendokument mit eigener Ueberschrift.
        "See “How this assistant works”.",
        # Der Katalogtitel der anderen Sprache.
        "Laut „Über den Assistenten“ helfe ich bei Versand.",
        "Laut über den Assistenten helfe ich bei Versand.",
    ],
)
def test_nennt_assistentendokument(text: str) -> None:
    assert nennt_assistentendokument("demo-assist", _tenant(), text, _dokumente())


@pytest.mark.parametrize(
    "text",
    [
        "Delivery takes three days.",
        # Zwei Fachtitel als Quellenangabe im Satz, wie in richtigen Antworten
        # vom 2026-10-22: kein Grund zu ersetzen.
        "Zone C covers every other country. The relevant sources are “Shipping” and “Returns”.",
        "This assistant answers from the stored documents.",
        "",
    ],
)
def test_nennt_kein_assistentendokument(text: str) -> None:
    assert not nennt_assistentendokument("demo-assist", _tenant(), text, _dokumente())


def test_ohne_assistentendokument_nennt_der_text_keines() -> None:
    """Ein Mandant ohne Assistentendokument: Auch der Katalogtitel zaehlt dann
    nicht, es gibt kein Dokument, das er bezeichnen koennte."""
    tenant = _tenant().model_copy(update={"assistant_documents": []})
    assert not nennt_assistentendokument(
        "demo-assist", tenant, "See “About This Assistant”.", _dokumente()
    )


def test_nennt_assistentendokument_prueft_den_mandanten() -> None:
    with pytest.raises(ValueError):
        nennt_assistentendokument("demo-anderer", _tenant(), "x", _dokumente("demo-anderer"))
    with pytest.raises(ValueError):
        nennt_assistentendokument("demo-assist", _tenant(), "x", _dokumente("demo-anderer"))


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
