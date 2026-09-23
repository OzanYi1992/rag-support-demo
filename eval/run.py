"""Messwerkzeug der Phase 5.

Faehrt den Goldsatz eines Mandanten gegen den aktuellen Stand und schreibt das
Ergebnis als JSON. Wertet NICHT aus - es legt Zahlen vor.

    python -m eval.run demo-acme
    python -m eval.run --all
    python -m eval.run demo-acme --retrieval-only
    python -m eval.run --all --top-k 4 --retrieval-only --lauf B

Zwei Dinge, die kein Zufall sind:

RANG UEBER DIE VOLLSTAENDIGE RANGLISTE. Der Rang der erwarteten Quelldatei wird
ueber ALLE Chunks bestimmt, nicht ueber die Top-k. Bei k=4 laege das englische
Zieldokument der cross_lingual-Fragen (Rang 14 bis 20 laut OP-019) unter "nicht
gefunden" - und damit waere genau das Signal weg, das das Problem sichtbar
gemacht hat. Eine Hit-Rate haette nur "nein" gesagt.

AGGREGATION JE KATEGORIE, NIE INSGESAMT. Ein Mittel ueber alle Fragen mittelt
den cross_lingual-Ausfall mit den direkten Treffern weg und sieht brauchbar aus,
waehrend das Verkaufsargument nicht traegt.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml

from app.config import PROJEKTWURZEL, Settings
from app.embeddings import get_embeddings
from app.escalation import (
    REASON_BELOW_THRESHOLD,
    REASON_FLAT,
    REASON_NO_HITS,
)
from app.rag import REASON_NOT_GROUNDED, REASON_UNPARSEABLE, answer
from app.search import search_tenant
from app.tenants import load_tenant
from app.texts import VERFUEGBARE_SPRACHEN, passt_zur_sprache

EVAL_DIR = PROJEKTWURZEL / "eval"
ERGEBNIS_DIR = EVAL_DIR / "results"

# Welches Tor hat gegriffen. ADR-019 haelt die Gruende getrennt, damit genau
# diese Auswertung moeglich ist.
TOR_RETRIEVAL = {REASON_NO_HITS, REASON_BELOW_THRESHOLD, REASON_FLAT}
TOR_GROUNDEDNESS = {REASON_NOT_GROUNDED}
TOR_SONSTIGES = {REASON_UNPARSEABLE}

# Welche Metrik eine Ergebnisdatei enthaelt. Steht im Kopf JEDER Datei, weil
# ein Hinweis, der nicht dort steht wo gelesen wird, keiner ist (P-018).
#
# Aeltere Laeufe massen nur den Rang der QUELLDATEI. Neuere messen zusaetzlich
# den Rang des CHUNKS mit der erwarteten Textstelle. Die Zahlen sehen gleich
# aus und sind es nicht: "Datei unter den Top-k" ist etwas anderes als
# "Antwort im Kontext". Ohne dieses Feld vergleicht jemand Zahlen, die nicht
# vergleichbar sind.
# erwartete_quelle traegt entweder einen Dateinamen oder einen dieser beiden
# Platzhalter. Ein blankes null ist NICHT zulaessig, weil es zwei verschiedene
# Dinge bedeutet haette und nichts am Zeichen die beiden unterscheidet: "es gibt
# hier keine richtige Quelle" und "eine gaebe es, sie wurde nur nie bestimmt".
# Ein Leser sieht bei null keinen Anlass zur Rueckfrage - die Mehrdeutigkeit ist
# stillschweigend. Siehe pitfalls.md, P-019.
QUELLE_NICHT_ANWENDBAR = "nicht_anwendbar"  # nicht_abgedeckt: es GIBT keine
QUELLE_NICHT_BESTIMMT = "nicht_bestimmt"  # eine gaebe es, sie fehlt im Goldsatz
QUELLE_PLATZHALTER = {QUELLE_NICHT_ANWENDBAR, QUELLE_NICHT_BESTIMMT}

METRIK_NUR_DATEI = "nur_datei"
METRIK_DATEI_UND_CHUNK = "datei_und_chunk"
METRIK_UNBESTIMMT = "unbestimmt"
METRIKEN = {METRIK_NUR_DATEI, METRIK_DATEI_UND_CHUNK, METRIK_UNBESTIMMT}

# Cloud-Laeufe. Die Raenge kommen auch dann aus der LOKALEN vollstaendigen
# Rangliste - die HTTP-Schnittstelle liefert nur die Top-k. Damit das keine
# stille Annahme bleibt, werden die Top-k-Scores der Cloud je Frage mit den
# lokalen verglichen. Stimmen sie, ist der Index im Image derselbe wie der
# lokale; bisher war genau das eine unausgesprochene Voraussetzung jeder
# Cloud-Aussage. Die Toleranz faengt Rundung und unterschiedliche
# CPU-Befehlssaetze ab, nicht einen anderen Index.
SCORE_TOLERANZ = 1e-3

# Abweichungsklasse je Frage. Wird AUS DEN DATEN abgeleitet, nie von Hand
# gesetzt: Sie ergibt sich daraus, ob der Chunk mit der erwarteten Textstelle
# im Kontext lag.
#
# In den Rohdaten sehen alle Abweichungen gleich aus - fuenfmal not_grounded.
# Dass sie zwei verschiedene Ursachen haben, war bisher nur in Prosa
# festgehalten. Wer die Datei oeffnet, hatte keinen Anlass zu trennen.
ABW_KEINE = "keine_abweichung"
ABW_NICHT_IM_KONTEXT = "antwort_lag_nicht_im_kontext"
ABW_IM_KONTEXT = "antwort_lag_im_kontext_trotzdem_eskaliert"
ABW_UNBESTIMMT = "unbestimmt_keine_textstelle_im_goldsatz"
ABW_NICHT_BEWERTBAR = "nicht_bewertbar_ohne_llm"

HINWEIS_CROSS_LINGUAL = (
    "Deutsche Frage, englische Quelle. Die deutschen Zwillingsdokumente sind "
    "Ablenker in der Anfragesprache - genau darin wirkt Language Bias. Der Rang "
    "zaehlt ueber die vollstaendige Rangliste, nicht ueber die Top-k."
)
HINWEIS_CROSS_LINGUAL_UMGEKEHRT = (
    "Englische Frage, deutscher Korpus. NICHT symmetrisch zu cross_lingual: Hier "
    "gibt es ueberhaupt keine englischen Ablenker, weil der ganze Korpus deutsch "
    "ist. Ein gutes Ergebnis ist deshalb KEINE Entwarnung zur Sprachgrenze - es "
    "belegt nur, dass ein einsprachiger Korpus keine Konkurrenz in der falschen "
    "Sprache hat."
)


# Befunde der Kontrollfrage. Sie laeuft NUR, wenn die kundennah formulierte
# Frage ihr Ziel verfehlt hat, und beantwortet dann genau eine Frage: Lag es am
# Retrieval oder an der Distanz zwischen Kundensprache und Dokumentsprache?
KONTROLLE_RETRIEVAL = "retrieval"  # auch die Dokumentformulierung findet nichts
KONTROLLE_SPRACHABSTAND = "sprachabstand"  # nur die Dokumentformulierung findet


@dataclass
class Frageergebnis:
    id: str
    kategorie: str
    frage: str
    erwartete_quelle: str | None
    erwartete_textstelle: str | None
    erwartet_eskalation: bool

    # ZWEI RAENGE, und die Unterscheidung ist der Kern. rang sagt "richtige
    # Datei unter den Top-k", rang_chunk sagt "Antwort im Kontext". acme-06 hat
    # gezeigt, dass das auseinanderfaellt: Datei auf Rang 3, Antwort in einem
    # Chunk, der nie geliefert wurde. Eine Hit-Rate auf Dateiebene sagt dann
    # etwas anderes, als sie zu sagen scheint.
    rang: int | None = None  # Datei, ueber die VOLLSTAENDIGE Rangliste
    in_top_k: bool = False
    rang_chunk: int | None = None  # Chunk mit der erwarteten Textstelle
    antwort_im_kontext: bool | None = None
    top_k_scores: list[float] = field(default_factory=list)
    top_k_quellen: list[str] = field(default_factory=list)
    fremde_chunks: list[str] = field(default_factory=list)

    eskaliert: bool | None = None
    tor: str | None = None
    grund: str | None = None
    eskalation_wie_erwartet: bool | None = None
    abweichungsklasse: str | None = None

    latenz_retrieval_ms: int | None = None
    latenz_generierung_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    antworttext: str | None = None
    modell: str | None = None

    # --- Kontrollfrage ---------------------------------------------------
    # Die Dokumentformulierung derselben Frage, auf dieselbe Textstelle.
    #
    # Wozu: Eine kundennah formulierte Frage kann aus zwei Gruenden scheitern -
    # weil das Retrieval die Stelle nicht hergibt, oder weil sie schlecht
    # formuliert ist. Ohne Kontrollfall sind die beiden nicht zu trennen, und
    # ein roter Goldsatzeintrag sagt dann nichts darueber aus, WAS nicht
    # funktioniert.
    #
    # Findet die Dokumentformulierung ebenfalls nichts, liegt es am Retrieval.
    # Findet sie, liegt es an der Distanz zwischen Kundensprache und
    # Dokumentsprache - und das ist ein Befund ueber das System, kein Fehler im
    # Goldsatz. Genau diese Unterscheidung ist der Grund, warum die
    # Kontrollfrage existiert.
    #
    # Sie laeuft nur bei einem Fehlschlag und nur im Retrieval: kein
    # Modellaufruf, keine Kosten, keine Verzerrung der Latenzzahlen.
    kontrollfrage: str | None = None
    kontroll_rang: int | None = None
    kontroll_rang_chunk: int | None = None
    kontroll_in_top_k: bool | None = None
    kontroll_antwort_im_kontext: bool | None = None
    kontroll_befund: str | None = None

    # --- Antwortsprache ---------------------------------------------------
    # `lang` ist die Angabe des MODELLS ueber die von ihm benutzte Sprache.
    # `sprache_folgt_frage` ist die Nachpruefung an den Texten - die beiden
    # koennen auseinanderfallen, und dann ist die Selbstauskunft falsch.
    #
    # Warum das hier steht: Am 2026-09-23 beantwortete der englische Mandant
    # drei von elf englischen Fragen auf Deutsch. Aufgefallen ist das nur, weil
    # jemand die Antworttexte gelesen hat - keine Zahl im Bericht zeigte es.
    # Eine Eigenschaft, die man nur durch Lesen entdeckt, ist nicht gemessen.
    lang: str | None = None
    sprache_der_frage: str | None = None
    sprache_der_antwort: str | None = None
    sprache_folgt_frage: bool | None = None

    # Nur in Cloud-Laeufen: Stimmen die Top-k-Scores der Cloud mit den lokalen?
    scores_wie_lokal: bool | None = None
    score_abweichung_max: float | None = None


def _abweichungsklasse(erg: Frageergebnis) -> str:
    """Leitet die Klasse aus den vorhandenen Daten ab.

    Nie von Hand gesetzt. Ohne LLM-Lauf gibt es keine Abweichung zu
    klassifizieren; ohne erwartete Textstelle im Goldsatz fehlt die Grundlage,
    und dann wird das ausgewiesen statt geraten.
    """
    if erg.eskaliert is None:
        return ABW_NICHT_BEWERTBAR
    if erg.eskalation_wie_erwartet:
        return ABW_KEINE
    if erg.antwort_im_kontext is None:
        return ABW_UNBESTIMMT
    return ABW_IM_KONTEXT if erg.antwort_im_kontext else ABW_NICHT_IM_KONTEXT


def pruefe_goldsatz(gold: dict[str, Any], quelle: str) -> None:
    """Lehnt ein blankes null bei erwartete_quelle ab.

    Der Wert muss sagen, WELCHE Art von Abwesenheit gemeint ist. Sonst steht
    dasselbe Zeichen fuer "es gibt keine richtige Quelle" und "sie wurde nie
    bestimmt", und die Auswertung behandelt beides gleich, obwohl das eine ein
    Entwurfsmerkmal und das andere eine Luecke ist.
    """
    fehler = [
        f["id"]
        for f in gold.get("fragen", [])
        if "erwartete_quelle" not in f or f["erwartete_quelle"] is None
    ]
    if fehler:
        raise ValueError(
            f"{quelle}: erwartete_quelle fehlt oder ist null bei {fehler}. "
            f"Erlaubt sind ein Dateiname, {QUELLE_NICHT_ANWENDBAR!r} "
            f"(es gibt keine richtige Quelle) oder {QUELLE_NICHT_BESTIMMT!r} "
            f"(eine gaebe es, sie ist nicht hinterlegt)."
        )


def _tor_von(grund: str | None) -> str | None:
    if grund is None:
        return None
    if grund in TOR_RETRIEVAL:
        return "retrieval"
    if grund in TOR_GROUNDEDNESS:
        return "groundedness"
    return "sonstiges"


def _chunkzahl(settings: Settings, slug: str) -> int:
    sidecar = Path(settings.index_dir) / slug / "sidecar.json"
    return int(json.loads(sidecar.read_text(encoding="utf-8"))["chunk_count"])


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJEKTWURZEL,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unbekannt"


def _preise() -> dict[str, Any]:
    pfad = EVAL_DIR / "preise.yaml"
    if not pfad.is_file():
        return {}
    return yaml.safe_load(pfad.read_text(encoding="utf-8")) or {}


def _kosten(
    preise: dict[str, Any], modell: str | None, ein: int | None, aus: int | None
) -> float | None:
    """None, solange keine Preise hinterlegt sind. Bricht bewusst nicht ab."""
    if not modell or ein is None or aus is None:
        return None
    eintrag = (preise.get("modelle") or {}).get(modell) or {}
    p_ein, p_aus = eintrag.get("eingabe_je_1m"), eintrag.get("ausgabe_je_1m")
    if p_ein is None or p_aus is None:
        return None
    return round(ein / 1_000_000 * p_ein + aus / 1_000_000 * p_aus, 6)


@dataclass(frozen=True)
class CloudZiel:
    """Deployte Instanz, gegen die der Generierungsschritt laeuft."""

    base_url: str
    image_digest: str
    client: httpx.Client

    # Name der Revision, die zum Messzeitpunkt Verkehr getragen hat.
    #
    # Warum getrennt vom Digest: Der Digest sagt, was DEPLOYT wurde. Antworten
    # kann in der Luecke eines Rollouts eine andere Revision - am 2026-09-23
    # sind auf diese Weise drei Messwerte am Vorgaenger entstanden, alle echt
    # gemessen und alle am falschen Gegenstand. Ohne diesen Namen ist einem
    # Ergebnis hinterher nicht anzusehen, WER geantwortet hat.
    revision: str | None = None


def cloud_antwort(ziel: CloudZiel, url_token: str, frage: str) -> dict[str, Any]:
    """Stellt eine Frage an die deployte Instanz.

    Das url_token steht nur im Pfad dieses einen Aufrufs. Es wird weder
    zurueckgegeben noch gespeichert - auch nicht in einer Fehlermeldung. Es ist
    die Zugangskontrolle (ADR-007), und Ergebnisdateien liegen im oeffentlichen
    Repository. Deshalb kein raise_for_status(): dessen Meldung enthaelt die URL.
    """
    antwort = ziel.client.post(
        f"{ziel.base_url.rstrip('/')}/t/{url_token}/chat",
        json={"question": frage},
        timeout=180.0,
    )
    if antwort.status_code != 200:
        raise RuntimeError(
            f"Cloud antwortete mit HTTP {antwort.status_code}. "
            f"Die URL wird nicht ausgegeben, sie enthaelt das url_token."
        )
    daten: dict[str, Any] = antwort.json()
    return daten


def vergleiche_scores(lokal: list[float], cloud: list[float]) -> tuple[bool, float | None]:
    """Gleich, wenn Laenge und jeder Score innerhalb der Toleranz stimmen.

    Gibt zusaetzlich die groesste Abweichung zurueck, damit ein "gleich" nicht
    nur behauptet, sondern beziffert ist.
    """
    if len(lokal) != len(cloud):
        return False, None
    if not lokal:
        return True, 0.0
    abweichung = max(abs(a - b) for a, b in zip(lokal, cloud, strict=True))
    return abweichung <= SCORE_TOLERANZ, round(abweichung, 6)


def _hat_ziel_verfehlt(erg: Frageergebnis) -> bool:
    """Verfehlt heisst: Datei nicht in den Top-k ODER Antwort nicht im Kontext.

    Beides zaehlt, weil beides fuer den Interessenten dasselbe bedeutet - er
    bekommt die Auskunft nicht. Die Datei auf Rang 3 zu haben nuetzt nichts,
    wenn der Chunk mit der Zahl nie geliefert wurde.
    """
    if erg.erwartete_quelle and erg.erwartete_quelle not in QUELLE_PLATZHALTER:
        if not erg.in_top_k:
            return True
    if erg.erwartete_textstelle and not erg.antwort_im_kontext:
        return True
    return False


def _kontrolle_fahren(
    slug: str,
    erg: Frageergebnis,
    settings: Settings,
    top_k: int,
    gesamtzahl: int,
    embedder: Any,  # noqa: ANN401
) -> None:
    """Faehrt die Kontrollfrage, wenn die eigentliche Frage ihr Ziel verfehlt hat.

    Bewusst NUR dann. Eine Kontrollfrage, die immer mitlaeuft, verdoppelt die
    Retrievalzeit jedes Laufs und liefert in dem Fall, der interessiert, keine
    zusaetzliche Information - denn wenn die Kundenfrage trifft, ist nichts zu
    erklaeren.

    `embedder` ist der bereits aufgeloeste Embedder des Aufrufers. Ihn hier neu
    zu holen waere zwar zulaessig, wuerde die Messung aber um einen Aufruf
    verschieben, den es im Normalfall nicht gibt.

    `Any` fuer `embedder`: Der Typ ist E5Embeddings, aber eval/ importiert ihn
    sonst nirgends und ein Import nur fuer eine Annotation zoege das
    Embeddingmodul in jede Nutzung dieses Moduls.
    """
    if not erg.kontrollfrage or not _hat_ziel_verfehlt(erg):
        return

    alle = search_tenant(
        slug, erg.kontrollfrage, k=gesamtzahl, settings=settings, embeddings=embedder
    )

    if erg.erwartete_quelle and erg.erwartete_quelle not in QUELLE_PLATZHALTER:
        for platz, treffer in enumerate(alle, start=1):
            if treffer.source_file == erg.erwartete_quelle:
                erg.kontroll_rang = platz
                break
        erg.kontroll_in_top_k = erg.kontroll_rang is not None and erg.kontroll_rang <= top_k

    if erg.erwartete_textstelle:
        for platz, treffer in enumerate(alle, start=1):
            if erg.erwartete_textstelle in treffer.text:
                erg.kontroll_rang_chunk = platz
                break
        erg.kontroll_antwort_im_kontext = (
            erg.kontroll_rang_chunk is not None and erg.kontroll_rang_chunk <= top_k
        )

    # Die Kontrollfrage gilt als getroffen, wenn sie dieselbe Huerde nimmt, an
    # der die Kundenfrage gescheitert ist. Derselbe Massstab fuer beide, sonst
    # vergleicht der Befund zwei verschiedene Dinge.
    kontroll_ergebnis = Frageergebnis(
        id=erg.id,
        kategorie=erg.kategorie,
        frage=erg.kontrollfrage,
        erwartete_quelle=erg.erwartete_quelle,
        erwartete_textstelle=erg.erwartete_textstelle,
        erwartet_eskalation=erg.erwartet_eskalation,
        rang=erg.kontroll_rang,
        in_top_k=bool(erg.kontroll_in_top_k),
        rang_chunk=erg.kontroll_rang_chunk,
        antwort_im_kontext=erg.kontroll_antwort_im_kontext,
    )
    erg.kontroll_befund = (
        KONTROLLE_RETRIEVAL if _hat_ziel_verfehlt(kontroll_ergebnis) else KONTROLLE_SPRACHABSTAND
    )


def eine_frage(
    slug: str,
    eintrag: dict[str, Any],
    settings: Settings,
    top_k: int,
    gesamtzahl: int,
    nur_retrieval: bool,
    cloud: CloudZiel | None = None,
    url_token: str | None = None,
) -> Frageergebnis:
    erg = Frageergebnis(
        id=eintrag["id"],
        kategorie=eintrag["kategorie"],
        frage=eintrag["frage"],
        erwartete_quelle=eintrag.get("erwartete_quelle"),
        erwartete_textstelle=eintrag.get("erwartete_textstelle"),
        erwartet_eskalation=bool(eintrag.get("erwartet_eskalation")),
    )
    embedder = get_embeddings(settings)

    # Vollstaendige Rangliste. Der Rang der erwarteten Quelldatei ist der Rang
    # ihres BESTEN Chunks.
    begonnen = time.monotonic()
    alle = search_tenant(slug, erg.frage, k=gesamtzahl, settings=settings, embeddings=embedder)
    erg.latenz_retrieval_ms = int((time.monotonic() - begonnen) * 1000)

    if erg.erwartete_quelle and erg.erwartete_quelle not in QUELLE_PLATZHALTER:
        for platz, treffer in enumerate(alle, start=1):
            if treffer.source_file == erg.erwartete_quelle:
                erg.rang = platz
                break
        erg.in_top_k = erg.rang is not None and erg.rang <= top_k

    if erg.erwartete_textstelle:
        for platz, treffer in enumerate(alle, start=1):
            if erg.erwartete_textstelle in treffer.text:
                erg.rang_chunk = platz
                break
        erg.antwort_im_kontext = erg.rang_chunk is not None and erg.rang_chunk <= top_k

    erg.kontrollfrage = eintrag.get("kontrollfrage")
    _kontrolle_fahren(slug, erg, settings, top_k, gesamtzahl, embedder)

    top = alle[:top_k]
    lokale_scores = [t.score for t in top]
    erg.top_k_scores = [round(s, 4) for s in lokale_scores]
    erg.top_k_quellen = [t.source_file for t in top]
    erg.fremde_chunks = sorted({t.tenant_slug for t in alle if t.tenant_slug != slug})

    if nur_retrieval:
        erg.abweichungsklasse = _abweichungsklasse(erg)
        return erg

    if cloud is not None:
        # Nur die Generierung laeuft gegen die deployte Instanz. Die Raenge oben
        # bleiben lokal - die Schnittstelle liefert keine vollstaendige Rangliste.
        if url_token is None:
            raise ValueError("Cloud-Lauf ohne url_token.")
        daten = cloud_antwort(cloud, url_token, erg.frage)
        erg.eskaliert = daten["escalated"]
        erg.grund = daten["escalation_reason"]
        erg.latenz_retrieval_ms = daten["latency_ms_retrieval"]
        erg.latenz_generierung_ms = daten["latency_ms_generation"]
        erg.prompt_tokens = daten["prompt_tokens"]
        erg.completion_tokens = daten["completion_tokens"]
        erg.antworttext = daten["text"]
        erg.modell = daten["model"]
        erg.scores_wie_lokal, erg.score_abweichung_max = vergleiche_scores(
            lokale_scores, daten["retrieval_scores"]
        )
    else:
        antwort = answer(slug, erg.frage, settings=settings, embeddings=embedder)
        erg.eskaliert = antwort.escalated
        erg.grund = antwort.escalation_reason
        erg.latenz_retrieval_ms = antwort.latency_ms_retrieval
        erg.latenz_generierung_ms = antwort.latency_ms_generation
        erg.prompt_tokens = antwort.prompt_tokens
        erg.completion_tokens = antwort.completion_tokens
        erg.antworttext = antwort.text
        erg.modell = antwort.model
        erg.lang = antwort.lang

    erg.tor = _tor_von(erg.grund)
    erg.eskalation_wie_erwartet = erg.eskaliert == erg.erwartet_eskalation
    erg.abweichungsklasse = _abweichungsklasse(erg)
    _sprache_pruefen(erg)
    return erg


def _rate_sprache(text: str) -> str | None:
    """Raet die Sprache eines Textes ueber die Markerlisten aus app/texts.py.

    Gibt None zurueck, wenn sich keine Sprache entscheiden laesst - bei einem
    kurzen Text ohne Funktionswoerter etwa. Das ist kein Mangel: Eine geratene
    Sprache waere schlechter als keine, weil sie in die Auswertung einginge.
    """
    passende = [s for s in VERFUEGBARE_SPRACHEN if passt_zur_sprache(text, s) is None]
    return passende[0] if len(passende) == 1 else None


def _sprache_pruefen(erg: Frageergebnis) -> None:
    """Haelt fest, ob die Antwort der Sprache der Frage folgt.

    Nur fuer nicht eskalierte Antworten. Der Eskalationstext ist der des
    Mandanten und folgt per Konstruktion dessen Sprache, nicht der Frage - ihn
    mitzuzaehlen wuerde die Zahl verwaessern.
    """
    if erg.eskaliert is not False or not erg.antworttext:
        return
    erg.sprache_der_frage = _rate_sprache(erg.frage)
    erg.sprache_der_antwort = _rate_sprache(erg.antworttext)
    if erg.sprache_der_frage and erg.sprache_der_antwort:
        erg.sprache_folgt_frage = erg.sprache_der_frage == erg.sprache_der_antwort


def aggregiere(
    ergebnisse: list[Frageergebnis], top_k: int, preise: dict[str, Any], modell: str | None
) -> dict[str, Any]:
    """Je Kategorie getrennt. Bewusst kein Gesamtwert ueber alle Fragen."""
    je_kategorie: dict[str, list[Frageergebnis]] = {}
    for e in ergebnisse:
        je_kategorie.setdefault(e.kategorie, []).append(e)

    aus: dict[str, Any] = {}
    for kategorie, gruppe in sorted(je_kategorie.items()):
        mit_ziel = [
            e for e in gruppe if e.erwartete_quelle and e.erwartete_quelle not in QUELLE_PLATZHALTER
        ]
        treffer = [e for e in mit_ziel if e.in_top_k]
        mit_stelle = [e for e in gruppe if e.erwartete_textstelle]
        im_kontext = [e for e in mit_stelle if e.antwort_im_kontext]
        mrr = (
            round(statistics.fmean(1 / e.rang if e.rang else 0.0 for e in mit_ziel), 4)
            if mit_ziel
            else None
        )
        latenzen = [
            (e.latenz_retrieval_ms or 0) + (e.latenz_generierung_ms or 0)
            for e in gruppe
            if e.latenz_retrieval_ms is not None
        ]
        kosten = [
            k
            for k in (_kosten(preise, modell, e.prompt_tokens, e.completion_tokens) for e in gruppe)
            if k is not None
        ]
        eskaliert = [e for e in gruppe if e.eskaliert]

        eintrag: dict[str, Any] = {
            "fragen": len(gruppe),
            "hit_rate_at_k": (round(len(treffer) / len(mit_ziel), 4) if mit_ziel else None),
            "treffer_in_top_k": f"{len(treffer)}/{len(mit_ziel)}" if mit_ziel else None,
            "mrr": mrr,
            "raenge_datei": [e.rang for e in mit_ziel],
            "antwort_im_kontext": (f"{len(im_kontext)}/{len(mit_stelle)}" if mit_stelle else None),
            "antwort_im_kontext_rate": (
                round(len(im_kontext) / len(mit_stelle), 4) if mit_stelle else None
            ),
            "raenge_chunk": [e.rang_chunk for e in mit_stelle],
            # Kontrollfragen: Nur die gescheiterten Fragen haben einen Befund.
            # Getrennt ausgewiesen, weil die beiden Befunde verschiedene Dinge
            # bedeuten - "retrieval" ist ein Mangel, "sprachabstand" ist eine
            # Eigenschaft, und beide in eine Zahl zu werfen verdeckt genau das.
            "kontrolle_retrieval": sum(
                1 for e in gruppe if e.kontroll_befund == KONTROLLE_RETRIEVAL
            ),
            "kontrolle_sprachabstand": sum(
                1 for e in gruppe if e.kontroll_befund == KONTROLLE_SPRACHABSTAND
            ),
            "kontrolle_nicht_gefahren": sum(
                1 for e in gruppe if e.kontrollfrage and e.kontroll_befund is None
            ),
            "kontrollfrage_fehlt": sorted(
                e.id
                for e in gruppe
                if not e.kontrollfrage
                and _hat_ziel_verfehlt(e)
                and e.erwartete_quelle
                and e.erwartete_quelle not in QUELLE_PLATZHALTER
            ),
            "sprache_folgt_frage_nicht": sorted(
                e.id for e in gruppe if e.sprache_folgt_frage is False
            ),
            "eskaliert": len(eskaliert),
            "eskaliert_retrieval": sum(1 for e in eskaliert if e.tor == "retrieval"),
            "eskaliert_groundedness": sum(1 for e in eskaliert if e.tor == "groundedness"),
            "eskaliert_sonstiges": sum(1 for e in eskaliert if e.tor == "sonstiges"),
            "eskalation_wie_erwartet": sum(1 for e in gruppe if e.eskalation_wie_erwartet)
            if any(e.eskalation_wie_erwartet is not None for e in gruppe)
            else None,
            "abweichungsklassen": {
                k: sum(1 for e in gruppe if e.abweichungsklasse == k)
                for k in (
                    ABW_KEINE,
                    ABW_NICHT_IM_KONTEXT,
                    ABW_IM_KONTEXT,
                    ABW_UNBESTIMMT,
                    ABW_NICHT_BEWERTBAR,
                )
                if any(e.abweichungsklasse == k for e in gruppe)
            },
            "latenz_p50_ms": round(statistics.median(latenzen)) if latenzen else None,
            "latenz_p95_ms": (
                round(sorted(latenzen)[max(0, int(len(latenzen) * 0.95) - 1)]) if latenzen else None
            ),
            "kosten_je_frage_usd": (round(statistics.fmean(kosten), 6) if kosten else None),
        }
        if kategorie == "cross_lingual":
            eintrag["hinweis"] = HINWEIS_CROSS_LINGUAL
        if kategorie == "cross_lingual_umgekehrt":
            eintrag["hinweis"] = HINWEIS_CROSS_LINGUAL_UMGEKEHRT
        aus[kategorie] = eintrag
    return aus


def fahre(
    slug: str,
    top_k_erzwungen: int | None,
    nur_retrieval: bool,
    lauf: str,
    baseline: bool,
    cloud: CloudZiel | None = None,
) -> dict[str, Any]:
    settings = Settings()
    tenant = load_tenant(slug, settings.tenants_dir)
    gold = yaml.safe_load((EVAL_DIR / slug / "gold.yaml").read_text(encoding="utf-8"))

    if top_k_erzwungen is not None:
        top_k, herkunft = top_k_erzwungen, "erzwungen ueber --top-k"
    elif tenant.retrieval_top_k is not None:
        top_k, herkunft = tenant.retrieval_top_k, "tenant.yaml (Mandanten-Override)"
    else:
        top_k, herkunft = settings.retrieval_top_k, "Settings-Default"

    gesamt = _chunkzahl(settings, slug)
    preise = _preise()

    # Das url_token verlaesst diese Funktion nur in Richtung cloud_antwort().
    # Es steht weder im Kopf noch in einem Frageergebnis.
    url_token = tenant.url_token if cloud is not None else None
    ergebnisse = [
        eine_frage(slug, f, settings, top_k, gesamt, nur_retrieval, cloud, url_token)
        for f in gold["fragen"]
    ]

    bedeutung = (
        "rang = Rang der Quelldatei, rang_chunk = Rang des Chunks mit "
        "der erwarteten Textstelle. hit_rate_at_k zaehlt Dateitreffer, "
        "antwort_im_kontext_rate zaehlt Chunktreffer. Ergebnisse mit "
        "metrik=nur_datei sind mit antwort_im_kontext_rate NICHT "
        "vergleichbar."
    )
    if cloud is not None and not nur_retrieval:
        # Dasselbe Feld darf nicht stillschweigend zwei Bedeutungen tragen
        # (P-019). Was metrik in DIESEM Lauf heisst, steht deshalb hier.
        bedeutung += (
            " IN DIESEM LAUF: rang und rang_chunk sind LOKAL bestimmt, ueber die "
            "vollstaendige Rangliste des lokalen Index. Eskalation, Antworttext, "
            "Token und Latenzen stammen aus der deployten Instanz. Dass beide "
            "denselben Index benutzen, belegt rangliste_stimmt_ueberein: Je Frage "
            "wurden die Top-k-Scores der Cloud mit den lokalen verglichen."
        )
        verglichen = [e for e in ergebnisse if e.scores_wie_lokal is not None]
        gleich = sum(1 for e in verglichen if e.scores_wie_lokal)
        abweichungen = [
            e.score_abweichung_max for e in verglichen if e.score_abweichung_max is not None
        ]
        modell = ", ".join(sorted({e.modell for e in ergebnisse if e.modell})) or None
        zielfelder: dict[str, Any] = {
            "ziel": "cloud",
            "retrieval_quelle": "lokal",
            "antwort_quelle": "cloud",
            "fqdn": urlparse(cloud.base_url).hostname,
            "image_digest": cloud.image_digest,
            "antwortende_revision": cloud.revision,
            "rangliste_stimmt_ueberein": f"{gleich}/{len(verglichen)}",
            "score_abweichung_max": max(abweichungen) if abweichungen else None,
            "score_toleranz": SCORE_TOLERANZ,
        }
    else:
        modell = None if nur_retrieval else settings.model_name
        zielfelder = {
            "ziel": "lokal",
            "retrieval_quelle": "lokal",
            "antwort_quelle": None if nur_retrieval else "lokal",
        }

    return {
        "kopf": {
            "zeitstempel": datetime.now(UTC).isoformat(timespec="seconds"),
            "lauf": lauf,
            "baseline": baseline,
            "mandant": slug,
            "modus": "nur_retrieval" if nur_retrieval else "retrieval_und_llm",
            **zielfelder,
            "metrik": METRIK_DATEI_UND_CHUNK,
            "metrik_bedeutung": bedeutung,
            "git_commit": _git_commit(),
            "embedding_modell": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "llm_modell": modell,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": top_k,
            "top_k_herkunft": herkunft,
            "eskalationsstrategie": settings.escalation_strategy,
            "retrieval_score_threshold": settings.retrieval_score_threshold,
            "chunks_im_index": gesamt,
            "preise_hinterlegt": bool(preise.get("stand")),
        },
        "aggregiert_je_kategorie": aggregiere(ergebnisse, top_k, preise, modell),
        "fragen": [asdict(e) for e in ergebnisse],
    }


def pruefe_kopf(bericht: dict[str, Any]) -> None:
    """Verweigert das Schreiben einer Datei ohne gueltige Metrikangabe.

    Der Sinn ist nicht Formalismus. Eine Ergebnisdatei ohne Metrikangabe ist
    spaeter nicht mehr einzuordnen, und genau daraus entsteht der Vergleich von
    Zahlen, die nicht vergleichbar sind. Lieber kein Ergebnis als ein
    uneinordenbares.
    """
    kopf = bericht.get("kopf", {})
    metrik = kopf.get("metrik")
    if metrik not in METRIKEN:
        raise ValueError(
            f"Ergebniskopf ohne gueltige Metrikangabe (metrik={metrik!r}). "
            f"Erlaubt: {sorted(METRIKEN)}. Datei wird nicht geschrieben."
        )
    if not kopf.get("metrik_bedeutung"):
        raise ValueError(
            "Ergebniskopf ohne metrik_bedeutung. Ein Kuerzel ohne Erklaerung "
            "ist in vier Wochen so wenig wert wie gar keine Angabe."
        )


def _zielzeile(kopf: dict[str, Any]) -> str:
    """Macht einen Cloud-Lauf schon in der Konsolenausgabe als solchen kenntlich."""
    if kopf.get("ziel") != "cloud":
        return "ZIEL: lokal"
    return (
        f"ZIEL: cloud ({kopf['fqdn']})   Revision: {kopf.get('antwortende_revision')}   "
        f"Raenge: lokal   "
        f"Top-k-Scores wie lokal: {kopf['rangliste_stimmt_ueberein']} "
        f"(max. Abweichung {kopf['score_abweichung_max']})"
    )


def tabelle(bericht: dict[str, Any]) -> str:
    kopf = bericht["kopf"]
    z = [
        f"{'=' * 78}",
        f"{kopf['mandant']}   Lauf {kopf['lauf']}   {kopf['modus']}",
        f"top_k={kopf['top_k']} ({kopf['top_k_herkunft']})   "
        f"chunks={kopf['chunks_im_index']}   strategie={kopf['eskalationsstrategie']}",
        f"embedding={kopf['embedding_modell']}   llm={kopf['llm_modell']}   "
        f"commit={kopf['git_commit']}",
        f"METRIK: {kopf['metrik']}",
        _zielzeile(kopf),
        f"{'=' * 78}",
        "",
        f"{'Kategorie':<26}{'Hit@k':>8}{'MRR':>8}{'Eskal.':>8}{'  davon Tor':<18}{'p50 ms':>8}",
        "-" * 78,
    ]
    for kategorie, a in bericht["aggregiert_je_kategorie"].items():
        hit = a["treffer_in_top_k"] or "–"
        mrr = f"{a['mrr']:.3f}" if a["mrr"] is not None else "–"
        tore = (
            f"R{a['eskaliert_retrieval']} "
            f"G{a['eskaliert_groundedness']} "
            f"S{a['eskaliert_sonstiges']}"
        )
        p50 = a["latenz_p50_ms"] if a["latenz_p50_ms"] is not None else "–"
        antw = a["antwort_im_kontext"] or "–"
        z.append(f"{kategorie:<26}{hit:>9}{antw:>9}{mrr:>7}{a['eskaliert']:>8}  {tore:<10}{p50:>6}")
    fremd = sorted({t for f in bericht["fragen"] for t in f["fremde_chunks"]})
    z += ["", f"Fremde Mandanten in irgendeiner Trefferliste: {fremd or 'keine'}"]

    # Kontrollfragen. Ein Befund, der nirgends auftaucht, ist keiner.
    befunde = [
        (f["id"], f["kontroll_befund"], f["frage"], f.get("kontrollfrage"))
        for f in bericht["fragen"]
        if f.get("kontroll_befund")
    ]
    if befunde:
        z += ["", "Kontrollfragen (nur bei verfehltem Ziel gefahren):"]
        for fid, befund, frage, kontrolle in befunde:
            deutung = (
                "Retrieval gibt die Stelle nicht her"
                if befund == KONTROLLE_RETRIEVAL
                else "Abstand Kundensprache/Dokumentsprache - Befund, kein Goldsatzfehler"
            )
            z.append(f"  {fid}: {befund} - {deutung}")
            z.append(f"      gefragt:    {frage}")
            z.append(f"      Kontrolle:  {kontrolle}")

    falsche_sprache = sorted(
        {
            fid
            for a in bericht["aggregiert_je_kategorie"].values()
            for fid in a.get("sprache_folgt_frage_nicht", [])
        }
    )
    if falsche_sprache:
        z += [
            "",
            f"ACHTUNG: Antwortsprache folgt der Frage NICHT: {', '.join(falsche_sprache)}",
            "  Vor einem Interessenten ist das der sichtbarste Fehler ueberhaupt.",
        ]

    fehlt = sorted(
        {
            fid
            for a in bericht["aggregiert_je_kategorie"].values()
            for fid in a.get("kontrollfrage_fehlt", [])
        }
    )
    if fehlt:
        z += [
            "",
            f"ACHTUNG: verfehlt, aber ohne Kontrollfrage: {', '.join(fehlt)}",
            "  Ohne Kontrollfall ist nicht zu sagen, ob es am Retrieval liegt",
            "  oder an der Formulierung. Der rote Eintrag ist nicht deutbar.",
        ]
    if not kopf["preise_hinterlegt"]:
        z.append("Kosten: keine Preise in eval/preise.yaml hinterlegt, Spalte leer.")
    return "\n".join(z)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m eval.run")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("slug", nargs="?", help="Mandant")
    g.add_argument("--all", action="store_true")
    p.add_argument(
        "--retrieval-only",
        action="store_true",
        help="ohne LLM - Raenge, Hit-Rate und MRR brauchen keines",
    )
    p.add_argument("--top-k", type=int, default=None, help="erzwingt k statt des Mandantenwerts")
    p.add_argument("--lauf", default="A", help="Kennzeichnung im Ergebniskopf")
    p.add_argument(
        "--baseline",
        action="store_true",
        help="als Bezugspunkt kennzeichnen (nur solche committen)",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="deployte Instanz; nur der Generierungsschritt laeuft dort, die Raenge bleiben lokal",
    )
    p.add_argument(
        "--image-digest",
        default=None,
        help="Pflicht zu --base-url: Digest des deployten Images, landet im Ergebniskopf",
    )
    p.add_argument(
        "--revision",
        default=None,
        help=(
            "Name der Revision, die zum Messzeitpunkt Verkehr traegt. Pflicht zu "
            "--base-url: Der Digest sagt, was deployt wurde, nicht wer geantwortet hat"
        ),
    )
    args = p.parse_args(argv)

    if args.base_url and not args.revision:
        p.error(
            "--base-url verlangt --revision: Ein Cloud-Ergebnis ohne die antwortende "
            "Revision ist nicht einzuordnen. In der Luecke eines Rollouts antwortet "
            "der Vorgaenger, und das Ergebnis sieht genauso aus."
        )
    if args.base_url and not args.image_digest:
        p.error(
            "--base-url verlangt --image-digest: "
            "Ein Cloud-Ergebnis ohne Image ist nicht einzuordnen."
        )
    if args.base_url and args.retrieval_only:
        p.error(
            "--base-url und --retrieval-only schliessen sich aus: "
            "Die Cloud liefert nur die Antwort."
        )

    # httpx protokolliert jede Anfrage samt URL auf INFO, und die URL enthaelt das
    # url_token. Deshalb nie unterhalb von WARNING, egal wer Logging konfiguriert.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    slugs = (
        sorted(d.name for d in EVAL_DIR.iterdir() if d.is_dir() and (d / "gold.yaml").is_file())
        if args.all
        else [args.slug]
    )

    ERGEBNIS_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        cloud = (
            CloudZiel(args.base_url, args.image_digest, client, args.revision)
            if args.base_url
            else None
        )
        for slug in slugs:
            bericht = fahre(slug, args.top_k, args.retrieval_only, args.lauf, args.baseline, cloud)
            pruefe_kopf(bericht)
            stempel = bericht["kopf"]["zeitstempel"].replace(":", "").replace("-", "")
            datei = ERGEBNIS_DIR / f"{stempel}-{slug}-lauf{args.lauf}.json"
            datei.write_text(json.dumps(bericht, ensure_ascii=False, indent=2), encoding="utf-8")
            print(tabelle(bericht))
            print(f"\ngeschrieben: {datei.relative_to(PROJEKTWURZEL)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
