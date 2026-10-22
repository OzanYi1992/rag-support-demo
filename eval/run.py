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
import copy
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
from app.llm import LlmNichtErreichbar
from app.prompts import Runde
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

    # --- Mehrfachlauf ------------------------------------------------------
    # Seit dem 2026-09-24 faehrt jede Frage n-mal. Grund: Die LLM-seitigen Zahlen
    # aus EN-3 und EN-6 waren Stichproben von eins, und bei einem Ausgang, der
    # wuerfelt, ist eine Stichprobe von eins von einem Zufall nicht zu
    # unterscheiden (P-028, P-030).
    #
    # Die retrievalseitigen Felder oben stammen aus dem ERSTEN Lauf. Dass das
    # zulaessig ist, behauptet dieser Lauf nicht, sondern belegt es:
    # retrieval_identisch vergleicht die Scorelisten aller Laeufe.
    laeufe: int = 1
    retrieval_identisch: bool | None = None
    score_abweichung_intern: float | None = None

    # LLM-Seite als Verteilung statt als Einzelwert. Die Skalarfelder oben
    # tragen weiter den ersten Lauf - damit bleibt ein Bericht mit laeufe=1
    # lesbar wie bisher.
    eskaliert_anzahl: int | None = None
    eskalation_wie_erwartet_anzahl: int | None = None
    stabil: bool | None = None

    # Antwortsprache als STANDARDGROESSE, nicht als Sonderpruefung. Seit dem
    # 2026-09-24: Zwei englische Regeln ueber Laenge und Fokus haben die Sprache
    # in 14 von 20 Faellen gekippt, ohne inhaltlich etwas mit Sprache zu tun zu
    # haben (P-030). Eine Groesse, die so leicht kippt, gehoert in jede Messung.
    sprachen_der_antworten: list[str] = field(default_factory=list)
    sprache_folgt_frage_anzahl: int | None = None
    sprache_bewertbar_anzahl: int | None = None
    antwortlaengen: list[int] = field(default_factory=list)

    # Jeder Lauf als eigener Eintrag, mit seinem Antworttext. Die Einzelfelder
    # oben tragen nur den ersten Lauf; bei einer Frage, die in zwei von fuenf
    # Laeufen antwortet, stand so oft gerade die Antwort nicht in der Datei, die
    # man lesen muesste. Die Felder oben bleiben, wie sie sind, damit aeltere
    # Ergebnisdateien vergleichbar bleiben.
    #
    # Jeder Eintrag traegt Text aus dem Korpus des Mandanten. Vor einem Commit
    # gilt deshalb dasselbe wie fuer `antworttext`: Bei einem Mandanten mit
    # echten Inhalten gehoert die Datei nicht ins oeffentliche Repo.
    laeufe_einzeln: list[dict[str, Any]] = field(default_factory=list)

    # --- Gespraech ---------------------------------------------------------
    # Nur in den Laeufen mit Verlauf belegt. `verlauf_runden` ist die Zahl der
    # frueheren Runden, `erwartete_sprache` die Sprache, in der die Antwort
    # kommen soll, und `erwartete_art` die feste Antwort, die eine Folgefrage
    # nach dem Assistenten bekommen soll. `arm` traegt in der Gegenprobe A oder B.
    verlauf_runden: int = 0
    erwartete_sprache: str | None = None
    erwartete_art: str | None = None
    arm: str | None = None

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


# Pause zwischen zwei Cloudaufrufen, in Sekunden.
#
# Die Anwendung begrenzt auf 30 Anfragen je 60 Sekunden und Token
# (app/main.py, RateLimiter). Ein Lauf mit fuenf Wiederholungen je Frage stellt
# 70 Anfragen an denselben Mandanten - ohne Pause laeuft er nach etwa der Haelfte
# in ein 429 und stirbt mitten in der Messung.
#
# 2.2 Sekunden ergeben rund 27 Anfragen je Minute und damit Luft nach unten. Die
# Pause verfaelscht keine Messung: Latenzen kommen aus den Feldern der Antwort,
# nicht aus der Wanduhr.
CLOUD_PAUSE_S = 2.2

# Ein Ausfall des Anbieters bei einem lokalen Lauf, als LlmNichtErreichbar.
#
# Die Anwendung wiederholt bewusst nicht; der Nutzer sieht dann eine eigene
# Meldung. Fuer das Messwerkzeug ist ein Ausfall aber kein Messergebnis: Die
# Anfrage hat keine Antwort erzeugt, die zu werten waere. Ohne Wiederholung
# beendet ein einzelner 503 einen Lauf von mehreren hundert Aufrufen, und weil
# die Datei erst am Ende geschrieben wird, ist alles davor verloren. So
# geschehen am 2026-10-13, kurz vor dem Ende einer Gegenprobe.
#
# Wiederholt wird der ganze Lauf, Umformung und Antwort, und jeder Lauf zaehlt
# seine Wiederholungen in `ausfall_wiederholt`.
AUSFALL_VERSUCHE = 3
AUSFALL_PAUSE_S = 15.0


def cloud_antwort(
    ziel: CloudZiel, url_token: str, frage: str, verlauf: list[Runde] | None = None
) -> dict[str, Any]:
    """Stellt eine Frage an die deployte Instanz, mit Verlauf, wenn es einen gibt.

    Das url_token steht nur im Pfad dieses einen Aufrufs. Es wird weder
    zurueckgegeben noch gespeichert - auch nicht in einer Fehlermeldung. Es ist
    die Zugangskontrolle (ADR-007), und Ergebnisdateien liegen im oeffentlichen
    Repository. Deshalb kein raise_for_status(): dessen Meldung enthaelt die URL.
    """
    adresse = f"{ziel.base_url.rstrip('/')}/t/{url_token}/chat"
    rumpf: dict[str, Any] = {"question": frage}
    if verlauf:
        rumpf["history"] = [runde.model_dump() for runde in verlauf]
    for versuch in (1, 2):
        antwort = ziel.client.post(adresse, json=rumpf, timeout=180.0)
        if antwort.status_code == 200:
            daten: dict[str, Any] = antwort.json()
            return daten
        # Ein 429 ist kein Messergebnis, sondern ein Taktproblem: Die Anfrage hat
        # das Modell nie erreicht. Der Limiter sagt selbst, wie lange zu warten
        # ist - also wird gewartet und einmal wiederholt, statt den Lauf zu
        # verlieren. Eine Wiederholung verfaelscht nichts, weil nichts gemessen
        # wurde.
        if antwort.status_code == 429 and versuch == 1:
            warte = int(antwort.headers.get("Retry-After", "5")) + 1
            print(f"    HTTP 429, warte {warte} s und wiederhole einmal", file=sys.stderr)
            time.sleep(warte)
            continue
        raise RuntimeError(
            f"Cloud antwortete mit HTTP {antwort.status_code} in Versuch {versuch}. "
            f"Die URL wird nicht ausgegeben, sie enthaelt das url_token."
        )
    raise RuntimeError("Cloud antwortete zweimal nicht mit 200.")


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


def _retrieval_teil(
    slug: str,
    eintrag: dict[str, Any],
    settings: Settings,
    top_k: int,
    gesamtzahl: int,
) -> tuple[Frageergebnis, Any, list[float]]:
    """Raenge, Kontrollfrage und Top-k einer Frage, ohne Modell.

    Gibt das Ergebnis, den Embedder und die lokalen Top-k-Scores zurueck.
    """
    erg = Frageergebnis(
        id=eintrag["id"],
        kategorie=eintrag["kategorie"],
        frage=eintrag["frage"],
        erwartete_quelle=eintrag.get("erwartete_quelle"),
        erwartete_textstelle=eintrag.get("erwartete_textstelle"),
        erwartet_eskalation=bool(eintrag.get("erwartet_eskalation")),
        erwartete_sprache=eintrag.get("erwartete_sprache"),
        erwartete_art=eintrag.get("erwartete_art"),
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
    return erg, embedder, lokale_scores


# Die Felder zur Antwortsprache je Lauf, unter den Namen des Ereignisses chat.
# Sie sind die Entscheidung und das Urteil des Servers. Der Massstab fuer die
# Antwortsprache in eval.run bleibt die Markerschaetzung, sprache_der_antwort:
# Mit dem Erkenner entscheidet der Server ueber den zweiten Aufruf, und mit
# demselben Werkzeug gemessen bestaetigte er sich selbst.
SPRACHFELDER = (
    "antwortsprache",
    "antwortsprache_quelle",
    "sprache_gehalten",
    "zweiter_aufruf",
    "zweiter_aufruf_ergebnis",
    "zweiter_aufruf_ms",
    "zweiter_aufruf_prompt_tokens",
    "zweiter_aufruf_completion_tokens",
)

# Woher die Felder kommen: die Namen in Answer und in der Antwort der Route.
_SPRACHFELDER_IN_ANSWER = {
    "antwortsprache": "answer_language",
    "antwortsprache_quelle": "answer_language_source",
    "sprache_gehalten": "language_kept",
    "zweiter_aufruf": "language_retry",
    "zweiter_aufruf_ergebnis": "language_retry_result",
    "zweiter_aufruf_ms": "latency_ms_language_retry",
    "zweiter_aufruf_prompt_tokens": "language_retry_prompt_tokens",
    "zweiter_aufruf_completion_tokens": "language_retry_completion_tokens",
}


@dataclass
class _Sammler:
    """Was die Laeufe einer Frage ueber alle Laeufe sammeln."""

    eskalationen: list[bool] = field(default_factory=list)
    sprachen: list[str] = field(default_factory=list)
    bewertbar: int = 0
    folgt: int = 0
    scorelisten: list[list[float]] = field(default_factory=list)


def _ein_aufruf(
    slug: str,
    frage: str,
    settings: Settings,
    embedder: Any,  # noqa: ANN401
    cloud: CloudZiel | None,
    url_token: str | None,
    verlauf: list[Runde] | None,
    response_language: str | None = None,
    sprachpruefung: bool = True,
) -> dict[str, Any]:
    """Ein Lauf, lokal ueber answer() oder gegen die deployte Instanz.

    `response_language` gibt die Antwortsprache vor, nur lokal; ohne sie legt
    der Server die Sprache selbst fest. `sprachpruefung=False` laesst den
    zweiten Aufruf weg, nur lokal: Gegen eine Instanz entscheidet die Instanz.

    Gibt die Felder in einer Form zurueck, die fuer beide Wege gleich ist. Eine
    Instanz von vor einem Feld liefert es nicht; dann steht None.
    """
    if cloud is not None:
        # Nur die Generierung laeuft gegen die deployte Instanz. Die Raenge
        # bleiben lokal - die Schnittstelle liefert keine vollstaendige
        # Rangliste.
        if url_token is None:
            raise ValueError("Cloud-Lauf ohne url_token.")
        if response_language is not None:
            raise ValueError("Eine vorgegebene Antwortsprache gibt es nur lokal.")
        if not sprachpruefung:
            raise ValueError("Ohne Sprachpruefung gibt es nur lokal.")
        daten = cloud_antwort(cloud, url_token, frage, verlauf)
        return {
            "eskaliert": daten["escalated"],
            "grund": daten["escalation_reason"],
            "antworttext": daten["text"],
            "lang": None,
            "kind": daten.get("kind"),
            "scores": list(daten["retrieval_scores"]),
            "latenz_retrieval_ms": daten["latency_ms_retrieval"],
            "latenz_generierung_ms": daten["latency_ms_generation"],
            "prompt_tokens": daten["prompt_tokens"],
            "completion_tokens": daten["completion_tokens"],
            "modell": daten["model"],
            "sources": daten.get("sources"),
            "removed_references": daten.get("removed_references"),
            "eigenstaendige_frage": daten.get("standalone_question"),
            "umformung": daten.get("rewrite_status"),
            "umformung_grund": daten.get("rewrite_reason"),
            "umformung_ms": daten.get("latency_ms_rewrite"),
            "umformung_prompt_tokens": daten.get("rewrite_prompt_tokens"),
            "umformung_completion_tokens": daten.get("rewrite_completion_tokens"),
            **{feld: daten.get(name) for feld, name in _SPRACHFELDER_IN_ANSWER.items()},
        }
    for versuch in range(1, AUSFALL_VERSUCHE + 1):
        try:
            antwort = answer(
                slug,
                frage,
                response_language=response_language,
                settings=settings,
                embeddings=embedder,
                history=verlauf,
                sprachpruefung=sprachpruefung,
            )
            break
        except LlmNichtErreichbar as fehler:
            if versuch == AUSFALL_VERSUCHE:
                raise
            print(
                f"    Anbieter nicht erreichbar ({fehler.grund}, {fehler.aufruf}), "
                f"warte {AUSFALL_PAUSE_S:.0f} s und wiederhole",
                file=sys.stderr,
            )
            time.sleep(AUSFALL_PAUSE_S)
    return {
        "response_language": response_language,
        "ausfall_wiederholt": versuch - 1,
        "eskaliert": antwort.escalated,
        "grund": antwort.escalation_reason,
        "antworttext": antwort.text,
        "lang": antwort.lang,
        "kind": antwort.kind,
        "scores": list(antwort.retrieval_scores),
        "latenz_retrieval_ms": antwort.latency_ms_retrieval,
        "latenz_generierung_ms": antwort.latency_ms_generation,
        "prompt_tokens": antwort.prompt_tokens,
        "completion_tokens": antwort.completion_tokens,
        "modell": antwort.model,
        "sources": list(antwort.sources),
        "removed_references": antwort.removed_references,
        "eigenstaendige_frage": antwort.standalone_question,
        "umformung": antwort.rewrite_status,
        "umformung_grund": antwort.rewrite_reason,
        "umformung_ms": antwort.latency_ms_rewrite,
        "umformung_prompt_tokens": antwort.rewrite_prompt_tokens,
        "umformung_completion_tokens": antwort.rewrite_completion_tokens,
        **{feld: getattr(antwort, name) for feld, name in _SPRACHFELDER_IN_ANSWER.items()},
    }


def _lauf_eintragen(
    slug: str,
    erg: Frageergebnis,
    sammler: _Sammler,
    lauf_nr: int,
    d: dict[str, Any],
    settings: Settings,
    embedder: Any,  # noqa: ANN401
    top_k: int,
    verlauf: list[Runde] | None,
) -> None:
    """Traegt einen Lauf in das Ergebnis und in den Sammler ein."""
    eskaliert = d["eskaliert"]
    antworttext = d["antworttext"]
    sammler.eskalationen.append(bool(eskaliert))
    sammler.scorelisten.append(d["scores"])

    # Die Antwortsprache je Lauf. Eskalationstexte bleiben aussen vor: Sie
    # sind der Text des Mandanten und folgen dessen Sprache, nicht der Frage.
    gesprochen = None
    if not eskaliert and antworttext:
        erg.antwortlaengen.append(len(antworttext))
        gesprochen = _rate_sprache(antworttext)
        if gesprochen:
            sammler.sprachen.append(gesprochen)
        gefragt = _rate_sprache(erg.frage)
        if gesprochen and gefragt:
            sammler.bewertbar += 1
            sammler.folgt += int(gesprochen == gefragt)

    eigenstaendig = d["eigenstaendige_frage"]
    eintrag: dict[str, Any] = {
        "lauf": lauf_nr + 1,
        "kind": d["kind"],
        "eskaliert": bool(eskaliert),
        "grund": d["grund"],
        "lang": d["lang"],
        "sprache_der_antwort": gesprochen,
        "antworttext": antworttext,
        "sources": d["sources"],
        "removed_references": d["removed_references"],
        "prompt_tokens": d["prompt_tokens"],
        "completion_tokens": d["completion_tokens"],
        "eigenstaendige_frage": eigenstaendig,
        "sprache_der_eigenstaendigen_frage": _rate_sprache(eigenstaendig)
        if eigenstaendig
        else None,
        "umformung": d["umformung"],
        "umformung_grund": d["umformung_grund"],
        "umformung_ms": d["umformung_ms"],
        "umformung_prompt_tokens": d["umformung_prompt_tokens"],
        "umformung_completion_tokens": d["umformung_completion_tokens"],
        "ausfall_wiederholt": d.get("ausfall_wiederholt", 0),
        # Die Latenz JE LAUF. Die Einzelfelder der Frage tragen nur den ersten
        # Lauf; ohne diese beiden liesse sich die Latenz nicht je Fassung oder
        # nur fuer Laeufe mit zweitem Aufruf auswerten. Beide beschreiben den
        # ersten Aufruf, die Umformung und den zweiten Aufruf tragen
        # umformung_ms und zweiter_aufruf_ms.
        "latenz_retrieval_ms": d["latenz_retrieval_ms"],
        "latenz_generierung_ms": d["latenz_generierung_ms"],
        **{feld: d.get(feld) for feld in SPRACHFELDER},
    }

    # Mit Verlauf sucht jeder Lauf mit seiner eigenen Frage. Ob die erwartete
    # Stelle im Kontext lag, wird deshalb je Lauf mit genau dieser Frage
    # nachgesehen, nicht mit der getippten.
    if verlauf is not None:
        stelle = None
        if erg.erwartete_textstelle and d["kind"] == "rag":
            suchfrage = eigenstaendig or erg.frage
            treffer = search_tenant(
                slug, suchfrage, k=top_k, settings=settings, embeddings=embedder
            )
            stelle = any(erg.erwartete_textstelle in t.text for t in treffer)
        eintrag["stelle_im_kontext"] = stelle
    erg.laeufe_einzeln.append(eintrag)

    # Der ERSTE Lauf fuellt die Einzelfelder. Damit bleibt ein Bericht mit
    # laeufe=1 genau der von vorher.
    if lauf_nr == 0:
        erg.eskaliert = eskaliert
        erg.grund = d["grund"]
        erg.antworttext = antworttext
        erg.lang = d["lang"]
        erg.latenz_retrieval_ms = d["latenz_retrieval_ms"]
        erg.latenz_generierung_ms = d["latenz_generierung_ms"]
        erg.prompt_tokens = d["prompt_tokens"]
        erg.completion_tokens = d["completion_tokens"]
        erg.modell = d["modell"]


def _abschliessen(erg: Frageergebnis, sammler: _Sammler, mit_verlauf: bool) -> None:
    """Verteilungen ueber alle Laeufe und die abgeleiteten Felder."""
    erg.eskaliert_anzahl = sum(sammler.eskalationen)
    erg.eskalation_wie_erwartet_anzahl = sum(
        1 for e in sammler.eskalationen if e == erg.erwartet_eskalation
    )
    erg.stabil = len(set(sammler.eskalationen)) == 1
    erg.sprachen_der_antworten = sammler.sprachen
    erg.sprache_bewertbar_anzahl = sammler.bewertbar
    erg.sprache_folgt_frage_anzahl = sammler.folgt

    # Retrieval ueber alle Laeufe: eine Scoreliste oder ein Befund. Mit Verlauf
    # nicht anwendbar, weil jeder Lauf mit seiner eigenen eigenstaendigen Frage
    # sucht; dort steht None statt eines falschen Alarms.
    if mit_verlauf:
        erg.retrieval_identisch = None
        erg.score_abweichung_intern = None
    else:
        listen = sammler.scorelisten
        erg.retrieval_identisch = all(liste == listen[0] for liste in listen)
        erg.score_abweichung_intern = max(
            (abs(a - b) for liste in listen[1:] for a, b in zip(liste, listen[0], strict=False)),
            default=0.0,
        )

    erg.tor = _tor_von(erg.grund)
    erg.eskalation_wie_erwartet = erg.eskaliert == erg.erwartet_eskalation
    erg.abweichungsklasse = _abweichungsklasse(erg)
    _sprache_pruefen(erg)


def eine_frage(
    slug: str,
    eintrag: dict[str, Any],
    settings: Settings,
    top_k: int,
    gesamtzahl: int,
    nur_retrieval: bool,
    cloud: CloudZiel | None = None,
    url_token: str | None = None,
    laeufe: int = 1,
    verlauf: list[Runde] | None = None,
    sprachpruefung: bool = True,
) -> Frageergebnis:
    erg, embedder, lokale_scores = _retrieval_teil(slug, eintrag, settings, top_k, gesamtzahl)

    if nur_retrieval:
        erg.abweichungsklasse = _abweichungsklasse(erg)
        return erg

    # --- Die Generierung, n-mal --------------------------------------------
    # Retrieval steht oben und ist deterministisch. Belegt wird das hier: Jeder
    # Lauf vergleicht seine Top-k-Scores mit denen des ersten Laufs, und
    # retrieval_identisch traegt das Ergebnis. Bei einem Cloudlauf liefert die
    # Instanz ihre Scores mit, dann wird zusaetzlich gegen lokal verglichen.
    erg.laeufe = laeufe
    erg.verlauf_runden = len(verlauf or [])
    sammler = _Sammler()
    for lauf_nr in range(laeufe):
        # Vor JEDEM Cloudaufruf, auch dem ersten einer Frage. Nur zwischen den
        # Wiederholungen zu pausieren waere ein Taktfehler: Fuenf Aufrufe in neun
        # Sekunden und dann sofort die naechste Frage ergeben rund 33 Anfragen je
        # Minute - drei ueber dem Limit, und der Lauf stirbt nach etwa der Haelfte.
        if cloud is not None:
            time.sleep(CLOUD_PAUSE_S)
        d = _ein_aufruf(
            slug,
            erg.frage,
            settings,
            embedder,
            cloud,
            url_token,
            verlauf,
            sprachpruefung=sprachpruefung,
        )
        if cloud is not None and lauf_nr == 0:
            erg.scores_wie_lokal, erg.score_abweichung_max = vergleiche_scores(
                lokale_scores, d["scores"]
            )
        _lauf_eintragen(slug, erg, sammler, lauf_nr, d, settings, embedder, top_k, verlauf)

    _abschliessen(erg, sammler, mit_verlauf=verlauf is not None)
    return erg


def gegenprobe_frage(
    slug: str,
    eintrag: dict[str, Any],
    settings: Settings,
    top_k: int,
    gesamtzahl: int,
    laeufe: int,
    vorrunde: Runde,
    arme: tuple[str, ...] = ("A", "B"),
    sprachpruefung: bool = True,
) -> tuple[Frageergebnis, Frageergebnis]:
    """Eine Goldfrage in zwei Fassungen, verschraenkt: A ohne Verlauf, B als
    zweite Runde nach einer fremden ersten.

    Je Lauf einmal A und einmal B, die Reihenfolge wechselt von Lauf zu Lauf.
    So verschiebt eine Aenderung beim Anbieter waehrend der Messung beide
    Fassungen gleich und erscheint nicht als Wirkung des Verlaufs.

    `arme` sind die Fassungen, die gefahren werden. Mit nur B laeuft A nicht:
    Dann ist eine fruehere Messung der Bezug fuer A, und die Fassung A dieses
    Ergebnisses bleibt ohne Laeufe.

    Die Raenge stammen aus der getippten Frage und gelten fuer A. B sucht je
    Lauf mit seiner eigenstaendigen Frage; ist sie zeichengleich mit der
    getippten, ist der Antwortprompt von B zeichengleich mit dem von A.
    """
    erg_a, embedder, _ = _retrieval_teil(slug, eintrag, settings, top_k, gesamtzahl)
    erg_b = copy.deepcopy(erg_a)
    erg_a.arm, erg_b.arm = "A", "B"
    erg_a.laeufe = erg_b.laeufe = laeufe
    erg_b.verlauf_runden = 1
    sammler_a, sammler_b = _Sammler(), _Sammler()
    fassungen = {
        "A": (erg_a, sammler_a, None),
        "B": (erg_b, sammler_b, [vorrunde]),
    }
    for lauf_nr in range(laeufe):
        for arm in ("A", "B") if lauf_nr % 2 == 0 else ("B", "A"):
            if arm not in arme:
                continue
            erg, sammler, verlauf = fassungen[arm]
            d = _ein_aufruf(
                slug,
                erg.frage,
                settings,
                embedder,
                None,
                None,
                verlauf,
                sprachpruefung=sprachpruefung,
            )
            _lauf_eintragen(slug, erg, sammler, lauf_nr, d, settings, embedder, top_k, verlauf)
    if "A" in arme:
        _abschliessen(erg_a, sammler_a, mit_verlauf=False)
    if "B" in arme:
        _abschliessen(erg_b, sammler_b, mit_verlauf=True)
    return erg_a, erg_b


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


def latenz_p95(latenzen: list[int]) -> int | None:
    """Das 95. Perzentil nach der Methode des naechsten Rangs.

    Rang = 0,95 * n, aufgerundet; der Wert an diesem Rang der sortierten Liste.
    Bei zwei Werten ist das der groessere, bei einem Wert dieser Wert. Gerechnet
    in ganzen Zahlen, damit keine Gleitkommarundung den Rang verschieben kann.

    Die fruehere Formel nahm `int(n * 0,95) - 1` als Index und lieferte bei zwei
    Werten den kleineren: ein p95 unter dem p50.
    """
    if not latenzen:
        return None
    rang = (95 * len(latenzen) + 99) // 100
    return round(sorted(latenzen)[rang - 1])


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
            # --- Antwortsprache: Standardgroesse, nicht Sonderpruefung ------
            # Gezaehlt werden LAEUFE, nicht Fragen. Bei n=5 und einer Frage, die
            # in einem von fuenf Laeufen die Sprache wechselt, wuerde eine
            # Fragenzaehlung entweder "eine Frage falsch" oder "eine Frage
            # richtig" sagen - beides waere irrefuehrend.
            "sprache_laeufe_bewertbar": sum(e.sprache_bewertbar_anzahl or 0 for e in gruppe),
            "sprache_folgt_frage_laeufe": sum(e.sprache_folgt_frage_anzahl or 0 for e in gruppe),
            "sprache_folgt_frage_rate": (
                round(
                    sum(e.sprache_folgt_frage_anzahl or 0 for e in gruppe)
                    / sum(e.sprache_bewertbar_anzahl or 0 for e in gruppe),
                    4,
                )
                if sum(e.sprache_bewertbar_anzahl or 0 for e in gruppe)
                else None
            ),
            "sprache_folgt_frage_nicht": sorted(
                e.id
                for e in gruppe
                if (e.sprache_bewertbar_anzahl or 0) > (e.sprache_folgt_frage_anzahl or 0)
            ),
            # --- Stabilitaet ueber die Laeufe -------------------------------
            # Eine Frage ist stabil, wenn alle ihre Laeufe dieselbe
            # Eskalationsentscheidung hatten. Bei laeufe=1 ist das trivial wahr
            # und sagt nichts - deshalb steht die Laufzahl im Kopf.
            "instabile_fragen": sorted(e.id for e in gruppe if e.stabil is False),
            "retrieval_nicht_identisch": sorted(
                e.id for e in gruppe if e.retrieval_identisch is False
            ),
            # `eskaliert` zaehlt FRAGEN, deren erster Lauf eskalierte - so war es
            # immer, und alte Berichte bleiben damit vergleichbar. Die drei
            # Zeilen darunter zaehlen LAEUFE und sind das, was bei n>1 zaehlt:
            # Eine Frage, die in drei von fuenf Laeufen eskaliert, ist in einer
            # Fragenzaehlung entweder ganz drin oder ganz draussen.
            "eskaliert": len(eskaliert),
            "laeufe_gesamt": sum(e.laeufe for e in gruppe),
            "eskaliert_laeufe": sum(e.eskaliert_anzahl or 0 for e in gruppe),
            "eskalation_wie_erwartet_laeufe": sum(
                e.eskalation_wie_erwartet_anzahl or 0 for e in gruppe
            ),
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
            "latenz_p95_ms": latenz_p95(latenzen),
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
    laeufe: int = 1,
    sprachpruefung: bool = True,
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
        eine_frage(
            slug,
            f,
            settings,
            top_k,
            gesamt,
            nur_retrieval,
            cloud,
            url_token,
            laeufe,
            sprachpruefung=sprachpruefung,
        )
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
            # Die Anzahl Laeufe JE FRAGE. Ohne diese Zahl ist keine LLM-seitige
            # Angabe in diesem Bericht einzuordnen: Bei n=1 ist jede davon eine
            # Stichprobe von eins, und genau das hat die Zahlen aus EN-3 und
            # EN-6 entwertet (P-028, P-030).
            "laeufe_je_frage": laeufe,
            # Ob der Server einen zweiten Aufruf machen durfte, wenn die Antwort
            # die festgelegte Sprache nicht hielt. null, wo eval.run das nicht
            # bestimmt: ohne Modell, und gegen eine deployte Instanz, die
            # selbst entscheidet.
            "sprachpruefung": None if nur_retrieval or cloud is not None else sprachpruefung,
            "laeufe_begruendung": (
                "n>1 seit 2026-09-24. Die Generierung wuerfelt: Dieselbe Frage "
                "lieferte bei identischem Retrieval verschiedene Antwortsprachen "
                "und verschiedene Eskalationsentscheidungen. Eine Stichprobe von "
                "eins ist davon nicht zu unterscheiden."
            ),
            "sampling_temperature": settings.llm_temperature,
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


# --- Gespraech: Verlaufsfaelle und Gegenprobe --------------------------------
#
# Die Faelle stehen je Mandant in eval/<mandant>/verlauf.yaml. Die Antworten im
# Verlauf sind eingefroren: echte Antworten, einmal erzeugt und in die Datei
# geschrieben, damit jeder Lauf denselben Verlauf sieht. `answer: null` heisst
# "noch nicht eingefroren", `answer: ""` heisst "die Runde ist eskaliert".

ARTEN_FUER_VERLAUF = ("capabilities", "identity")


def pruefe_verlauf(daten: dict[str, Any], quelle: str, eingefroren: bool) -> None:
    """Lehnt eine Falldatei ab, die nicht messbar ist.

    Geprueft wird jedes Pflichtfeld und, mit `eingefroren`, dass jede Antwort im
    Verlauf eingefroren ist. Jede Runde muss die Grenzen der Route einhalten,
    sonst misst die Datei einen Verlauf, den es im Betrieb nicht geben kann.
    """
    fehler: list[str] = []
    for f in daten.get("faelle", []):
        fid = f.get("id", "?")
        for feld in ("id", "kategorie", "verlauf", "folgefrage", "begruendung"):
            if not f.get(feld):
                fehler.append(f"{fid}: {feld} fehlt")
        if f.get("erwartete_quelle") is None:
            fehler.append(f"{fid}: erwartete_quelle fehlt oder ist null")
        if not isinstance(f.get("erwartet_eskalation"), bool):
            fehler.append(f"{fid}: erwartet_eskalation ist kein Wahrheitswert")
        if f.get("erwartete_sprache") not in VERFUEGBARE_SPRACHEN:
            fehler.append(f"{fid}: erwartete_sprache {f.get('erwartete_sprache')!r} unbekannt")
        if not isinstance(f.get("laeufe"), int) or f["laeufe"] < 1:
            fehler.append(f"{fid}: laeufe fehlt oder ist kleiner als 1")
        if f.get("erwartete_art") not in (None, *ARTEN_FUER_VERLAUF):
            fehler.append(f"{fid}: erwartete_art {f.get('erwartete_art')!r} unbekannt")
        if len(f.get("verlauf") or []) > 3:
            fehler.append(f"{fid}: mehr als drei Runden")
        for runde in f.get("verlauf") or []:
            if eingefroren and runde.get("answer") is None:
                fehler.append(f"{fid}: Antwort auf {runde.get('question')!r} nicht eingefroren")
            anweisung = runde.get("eingeschleuste_anweisung")
            if anweisung and runde.get("answer") is not None:
                if not runde["answer"].endswith(anweisung):
                    fehler.append(f"{fid}: die eingeschleuste Anweisung steht nicht am Ende")
            try:
                Runde(question=runde.get("question"), answer=runde.get("answer") or "")
            except ValueError as grund:
                fehler.append(f"{fid}: Runde ungueltig: {grund}")
    vorrunde = daten.get("gegenprobe_vorrunde")
    if vorrunde is not None and eingefroren and vorrunde.get("answer") is None:
        fehler.append("gegenprobe_vorrunde: Antwort nicht eingefroren")
    if fehler:
        raise ValueError(f"{quelle}: " + "; ".join(fehler))


def _runden(liste: list[dict[str, Any]]) -> list[Runde]:
    return [Runde(question=r["question"], answer=r.get("answer") or "") for r in liste]


def lade_verlauf(slug: str, eingefroren: bool = True) -> dict[str, Any]:
    pfad = EVAL_DIR / slug / "verlauf.yaml"
    daten: dict[str, Any] = yaml.safe_load(pfad.read_text(encoding="utf-8"))
    pruefe_verlauf(daten, str(pfad), eingefroren)
    return daten


def _top_k_und_herkunft(settings: Settings, slug: str) -> tuple[int, str]:
    tenant = load_tenant(slug, settings.tenants_dir)
    if tenant.retrieval_top_k is not None:
        return tenant.retrieval_top_k, "tenant.yaml (Mandanten-Override)"
    return settings.retrieval_top_k, "Settings-Default"


def _kopf_lokal(
    settings: Settings,
    slug: str,
    lauf: str,
    baseline: bool,
    modus: str,
    laeufe: object,
    top_k: int,
    herkunft: str,
    gesamt: int,
    preise: dict[str, Any],
    bedeutung: str,
) -> dict[str, Any]:
    """Der Ergebniskopf eines lokalen Laufs mit Modell, fuer die Gespraechsmodi."""
    return {
        "zeitstempel": datetime.now(UTC).isoformat(timespec="seconds"),
        "lauf": lauf,
        "baseline": baseline,
        "laeufe_je_frage": laeufe,
        "sampling_temperature": settings.llm_temperature,
        "mandant": slug,
        "modus": modus,
        "ziel": "lokal",
        "retrieval_quelle": "lokal",
        "antwort_quelle": "lokal",
        "metrik": METRIK_DATEI_UND_CHUNK,
        "metrik_bedeutung": bedeutung,
        "git_commit": _git_commit(),
        "embedding_modell": settings.embedding_model,
        "embedding_dimension": settings.embedding_dimension,
        "llm_modell": settings.model_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": top_k,
        "top_k_herkunft": herkunft,
        "eskalationsstrategie": settings.escalation_strategy,
        "retrieval_score_threshold": settings.retrieval_score_threshold,
        "chunks_im_index": gesamt,
        "preise_hinterlegt": bool(preise.get("stand")),
    }


def _gleich_getippt(ergebnisse: list[Frageergebnis]) -> str:
    """Wie oft die eigenstaendige Frage zeichengleich mit der getippten war."""
    umgeformt = [
        (e.frage, lauf["eigenstaendige_frage"])
        for e in ergebnisse
        for lauf in e.laeufe_einzeln
        if lauf.get("eigenstaendige_frage") is not None
    ]
    gleich = sum(1 for getippt, eigen in umgeformt if getippt == eigen)
    return f"{gleich}/{len(umgeformt)}"


BEDEUTUNG_VERLAUF = (
    "Folgefragen mit eingefrorenem Verlauf. rang und rang_chunk sind die der "
    "GETIPPTEN Folgefrage und sagen deshalb wenig; je Lauf steht in "
    "laeufe_einzeln, mit welcher eigenstaendigen Frage gesucht wurde und ob die "
    "erwartete Stelle mit ihr im Kontext lag (stelle_im_kontext)."
)
BEDEUTUNG_KONTROLLE = (
    "Dieselben Folgefragen ohne Verlauf, also das Verhalten ohne Gespraech. "
    "rang = Rang der Quelldatei, rang_chunk = Rang des Chunks mit der "
    "erwarteten Textstelle."
)
BEDEUTUNG_NUR_B = (
    " IN DIESEM LAUF NUR FASSUNG B: fragen und aggregiert_je_kategorie sind leer, "
    "der Bezug fuer A ist eine fruehere Ergebnisdatei."
)
BEDEUTUNG_GEGENPROBE = (
    "Goldfragen verschraenkt in zwei Fassungen: A ohne Verlauf, B als zweite "
    "Runde nach einer fremden ersten. Raenge aus der getippten Frage, sie gelten "
    "fuer A. b_eigenstaendig_gleich_getippt zaehlt die Laeufe von B, deren "
    "eigenstaendige Frage zeichengleich mit der getippten war."
)


def fahre_verlauf(
    slug: str,
    lauf: str,
    baseline: bool,
    mit_verlauf: bool,
    laeufe_kontrolle: int,
    sprachpruefung: bool = True,
) -> dict[str, Any]:
    """Die Verlaufsfaelle eines Mandanten, mit Verlauf oder als Kontrolle ohne.

    Mit Verlauf nimmt jeder Fall seine Laufzahl aus der Datei. Die Kontrolle
    stellt dieselben Folgefragen ohne Verlauf, je `laeufe_kontrolle` mal.
    """
    settings = Settings()
    daten = lade_verlauf(slug, eingefroren=mit_verlauf)
    top_k, herkunft = _top_k_und_herkunft(settings, slug)
    gesamt = _chunkzahl(settings, slug)
    preise = _preise()

    ergebnisse: list[Frageergebnis] = []
    laeufe_je_fall: dict[str, int] = {}
    for fall in daten["faelle"]:
        eintrag = {**fall, "frage": fall["folgefrage"]}
        n = fall["laeufe"] if mit_verlauf else laeufe_kontrolle
        laeufe_je_fall[fall["id"]] = n
        verlauf = _runden(fall["verlauf"]) if mit_verlauf else None
        ergebnisse.append(
            eine_frage(
                slug,
                eintrag,
                settings,
                top_k,
                gesamt,
                False,
                laeufe=n,
                verlauf=verlauf,
                sprachpruefung=sprachpruefung,
            )
        )

    kopf = _kopf_lokal(
        settings,
        slug,
        lauf,
        baseline,
        "verlauf" if mit_verlauf else "verlauf_kontrolle",
        "je_fall",
        top_k,
        herkunft,
        gesamt,
        preise,
        BEDEUTUNG_VERLAUF if mit_verlauf else BEDEUTUNG_KONTROLLE,
    )
    kopf["laeufe_je_fall"] = laeufe_je_fall
    kopf["sprachpruefung"] = sprachpruefung
    # YAML liest `stand: 2026-10-13` als Datum; die Ergebnisdatei ist JSON.
    kopf["falldatei_stand"] = str(daten["stand"]) if daten.get("stand") is not None else None
    if mit_verlauf:
        kopf["eigenstaendig_gleich_getippt"] = _gleich_getippt(ergebnisse)
    return {
        "kopf": kopf,
        "aggregiert_je_kategorie": aggregiere(ergebnisse, top_k, preise, settings.model_name),
        "fragen": [asdict(e) for e in ergebnisse],
    }


def fahre_gegenprobe(
    slug: str,
    lauf: str,
    baseline: bool,
    laeufe: int,
    nur_b: bool = False,
    sprachpruefung: bool = True,
) -> dict[str, Any]:
    """Die Goldfragen verschraenkt, A ohne und B mit fremder erster Runde.

    Mit `nur_b` laeuft nur Fassung B, etwa nach einer Aenderung, die nur den Weg
    mit Verlauf betrifft. Dann ist eine fruehere Messung der Bezug fuer A.
    """
    settings = Settings()
    gold = yaml.safe_load((EVAL_DIR / slug / "gold.yaml").read_text(encoding="utf-8"))
    daten = lade_verlauf(slug, eingefroren=True)
    vorrunde_roh = daten.get("gegenprobe_vorrunde")
    if vorrunde_roh is None:
        raise ValueError(f"eval/{slug}/verlauf.yaml hat keine gegenprobe_vorrunde.")
    vorrunde = _runden([vorrunde_roh])[0]
    top_k, herkunft = _top_k_und_herkunft(settings, slug)
    gesamt = _chunkzahl(settings, slug)
    preise = _preise()

    arme = ("B",) if nur_b else ("A", "B")
    arm_a: list[Frageergebnis] = []
    arm_b: list[Frageergebnis] = []
    for f in gold["fragen"]:
        a, b = gegenprobe_frage(
            slug, f, settings, top_k, gesamt, laeufe, vorrunde, arme, sprachpruefung
        )
        if "A" in arme:
            arm_a.append(a)
        arm_b.append(b)

    kopf = _kopf_lokal(
        settings,
        slug,
        lauf,
        baseline,
        "gegenprobe",
        laeufe,
        top_k,
        herkunft,
        gesamt,
        preise,
        BEDEUTUNG_GEGENPROBE + (BEDEUTUNG_NUR_B if nur_b else ""),
    )
    kopf["fassungen"] = list(arme)
    kopf["sprachpruefung"] = sprachpruefung
    kopf["vorrunde_frage"] = vorrunde.question
    kopf["b_eigenstaendig_gleich_getippt"] = _gleich_getippt(arm_b)
    return {
        "kopf": kopf,
        "aggregiert_je_kategorie": aggregiere(arm_a, top_k, preise, settings.model_name),
        "aggregiert_je_kategorie_b": aggregiere(arm_b, top_k, preise, settings.model_name),
        "fragen": [asdict(e) for e in arm_a],
        "fragen_b": [asdict(e) for e in arm_b],
    }


# --- Sprache: drei Wege zur Antwortsprache -----------------------------------
#
# Die Faelle stehen je Mandant in eval/<mandant>/sprache.yaml, ohne Verlauf.
# Jede Frage laeuft in zwei von drei Fassungen, verschraenkt wie die
# Gegenprobe:
#
#   A  ohne response_language und ohne Sprachpruefung: so, wie die Seite fragte,
#      bevor der Server die Sprache pruefte;
#   B  mit response_language gleich der erwarteten Sprache;
#   O  ohne response_language, mit Sprachpruefung und hoechstens einem zweiten
#      Aufruf: der Weg der Seite heute.
#
# Die Dateien tragen dazu die erwartete Sprache je Goldfrage; gold.yaml bleibt,
# wie sie ist.

FASSUNGEN_SPRACHE = ("A", "B", "O")

# Ueberschrift der zweiten Tabelle in der Ausgabe.
UEBERSCHRIFT_FASSUNG = {
    "A": "ohne Vorgabe und ohne Sprachpruefung",
    "B": "mit vorgegebener Sprache",
    "O": "ohne Vorgabe, mit Sprachpruefung",
}

BEDEUTUNG_SPRACHE = (
    "Fragen ohne Verlauf in zwei Fassungen, verschraenkt; welche, steht in fassungen. "
    "A: ohne response_language und mit sprachpruefung=False, so wie die Seite fragte, "
    "bevor der Server die Sprache pruefte. B: mit response_language gleich der "
    "erwarteten Sprache. O: ohne response_language und mit Sprachpruefung, also "
    "hoechstens einem zweiten Aufruf, wenn die belegte Antwort die festgelegte Sprache "
    "nicht haelt; der Weg der Seite. Die erste Fassung steht unter fragen und "
    "aggregiert_je_kategorie, die zweite unter fragen_<buchstabe> und "
    "aggregiert_je_kategorie_<buchstabe>, klein geschrieben. sprache_der_antwort ist "
    "die Markerschaetzung von eval.run und der Massstab fuer die Antwortsprache; "
    "antwortsprache, antwortsprache_quelle und sprache_gehalten sind Entscheidung und "
    "Urteil des Servers, lang die Angabe des Modells. Die Felder ohne zweiter_aufruf "
    "im Namen beschreiben den ersten Aufruf, ausser antworttext, sources, "
    "removed_references und lang, die zur ausgelieferten Antwort gehoeren. "
    "stelle_im_kontext je Lauf ist die der Frage; ohne Verlauf sucht jeder Lauf mit "
    "derselben Frage."
)


def _fassung_aufruf(fassung: str, sprache: str) -> tuple[str | None, bool]:
    """Was eine Fassung an answer() gibt: vorgegebene Sprache und Sprachpruefung.

    Bei B ist die Pruefung ohne Wirkung: Eine vorgegebene Sprache prueft der
    Server nicht.
    """
    if fassung == "A":
        return None, False
    if fassung == "B":
        return sprache, True
    if fassung == "O":
        return None, True
    raise ValueError(f"Unbekannte Fassung: {fassung!r}")


def pruefe_sprache(daten: dict[str, Any], quelle: str, gold_ids: set[str]) -> None:
    """Lehnt eine Sprachdatei ab, die nicht messbar ist.

    `gold_ids` sind die IDs aus gold.yaml desselben Mandanten. Eine Ausnahme im
    Abschnitt goldsatz, die es dort nicht gibt, wuerde sonst still nichts
    bewirken.
    """
    fehler: list[str] = []
    for f in daten.get("faelle", []):
        fid = f.get("id", "?")
        for feld in ("id", "kategorie", "frage", "begruendung"):
            if not f.get(feld):
                fehler.append(f"{fid}: {feld} fehlt")
        if f.get("erwartete_sprache") not in VERFUEGBARE_SPRACHEN:
            fehler.append(f"{fid}: erwartete_sprache {f.get('erwartete_sprache')!r} unbekannt")
        if f.get("erwartete_quelle") is None:
            fehler.append(f"{fid}: erwartete_quelle fehlt oder ist null")
        if not isinstance(f.get("erwartet_eskalation"), bool):
            fehler.append(f"{fid}: erwartet_eskalation ist kein Wahrheitswert")
        if not isinstance(f.get("laeufe"), int) or f["laeufe"] < 1:
            fehler.append(f"{fid}: laeufe fehlt oder ist kleiner als 1")
    gold = daten.get("goldsatz")
    if gold is not None:
        if not isinstance(gold.get("laeufe"), int) or gold["laeufe"] < 1:
            fehler.append("goldsatz: laeufe fehlt oder ist kleiner als 1")
        if gold.get("erwartete_sprache") not in VERFUEGBARE_SPRACHEN:
            fehler.append("goldsatz: erwartete_sprache unbekannt")
        for sprache, ids in (gold.get("abweichend") or {}).items():
            if sprache not in VERFUEGBARE_SPRACHEN:
                fehler.append(f"goldsatz: abweichende Sprache {sprache!r} unbekannt")
            for gid in ids:
                if gid not in gold_ids:
                    fehler.append(f"goldsatz: {gid} steht nicht in gold.yaml")
    if fehler:
        raise ValueError(f"{quelle}: " + "; ".join(fehler))


def lade_sprache(slug: str) -> dict[str, Any]:
    pfad = EVAL_DIR / slug / "sprache.yaml"
    daten: dict[str, Any] = yaml.safe_load(pfad.read_text(encoding="utf-8"))
    gold = yaml.safe_load((EVAL_DIR / slug / "gold.yaml").read_text(encoding="utf-8"))
    pruefe_sprache(daten, str(pfad), {f["id"] for f in gold["fragen"]})
    return daten


def goldsprachen(daten: dict[str, Any]) -> dict[str, str]:
    """Die erwartete Sprache je Goldfrage-ID, aus dem Abschnitt goldsatz."""
    gold = daten["goldsatz"]
    zuordnung: dict[str, str] = {}
    for sprache, ids in (gold.get("abweichend") or {}).items():
        for gid in ids:
            zuordnung[gid] = sprache
    return zuordnung


def sprachprobe_frage(
    slug: str,
    eintrag: dict[str, Any],
    settings: Settings,
    top_k: int,
    gesamtzahl: int,
    laeufe: int,
    sprache: str,
    fassungen: tuple[str, str] = ("A", "B"),
) -> tuple[Frageergebnis, Frageergebnis]:
    """Eine Frage in zwei Fassungen, verschraenkt, Standard A und B. Je Lauf
    einmal jede, die Reihenfolge wechselt von Lauf zu Lauf. Was die Fassungen
    an answer() geben, steht in `_fassung_aufruf`.

    Ohne Verlauf sucht jeder Lauf mit derselben Frage. Die Stelle im Kontext ist
    deshalb je Lauf die der Frage und steht so in jedem Eintrag.
    """
    erste, zweite = fassungen
    erg_1, embedder, _ = _retrieval_teil(slug, eintrag, settings, top_k, gesamtzahl)
    erg_1.erwartete_sprache = sprache
    erg_2 = copy.deepcopy(erg_1)
    erg_1.arm, erg_2.arm = erste, zweite
    erg_1.laeufe = erg_2.laeufe = laeufe
    aufbau = {erste: (erg_1, _Sammler()), zweite: (erg_2, _Sammler())}
    for lauf_nr in range(laeufe):
        for fassung in (erste, zweite) if lauf_nr % 2 == 0 else (zweite, erste):
            erg, sammler = aufbau[fassung]
            vorgabe, pruefung = _fassung_aufruf(fassung, sprache)
            d = _ein_aufruf(
                slug, erg.frage, settings, embedder, None, None, None, vorgabe, pruefung
            )
            _lauf_eintragen(slug, erg, sammler, lauf_nr, d, settings, embedder, top_k, None)
            gesprochen = erg.laeufe_einzeln[-1]["sprache_der_antwort"]
            erg.laeufe_einzeln[-1].update(
                {
                    "response_language": vorgabe,
                    "sprachpruefung": pruefung,
                    "erwartete_sprache": sprache,
                    "sprache_wie_erwartet": None if gesprochen is None else gesprochen == sprache,
                    "stelle_im_kontext": erg.antwort_im_kontext,
                }
            )
    for erg, sammler in aufbau.values():
        _abschliessen(erg, sammler, mit_verlauf=False)
    return erg_1, erg_2


def fahre_sprache(
    slug: str,
    lauf: str,
    baseline: bool,
    gold: bool,
    fassungen: tuple[str, str] = ("A", "B"),
) -> dict[str, Any]:
    """Die Sprachfaelle eines Mandanten, oder mit `gold` seinen Goldsatz, je in
    zwei Fassungen, Standard A und B."""
    settings = Settings()
    daten = lade_sprache(slug)
    top_k, herkunft = _top_k_und_herkunft(settings, slug)
    gesamt = _chunkzahl(settings, slug)
    preise = _preise()

    if gold:
        if daten.get("goldsatz") is None:
            raise ValueError(f"eval/{slug}/sprache.yaml hat keinen Abschnitt goldsatz.")
        goldfragen = yaml.safe_load((EVAL_DIR / slug / "gold.yaml").read_text(encoding="utf-8"))
        abweichend = goldsprachen(daten)
        standard = daten["goldsatz"]["erwartete_sprache"]
        laeufe = int(daten["goldsatz"]["laeufe"])
        auftraege = [(f, abweichend.get(f["id"], standard), laeufe) for f in goldfragen["fragen"]]
    else:
        auftraege = [(f, f["erwartete_sprache"], int(f["laeufe"])) for f in daten["faelle"]]

    arm_1: list[Frageergebnis] = []
    arm_2: list[Frageergebnis] = []
    laeufe_je_frage: dict[str, int] = {}
    for eintrag, sprache, n in auftraege:
        laeufe_je_frage[eintrag["id"]] = n
        a, b = sprachprobe_frage(slug, eintrag, settings, top_k, gesamt, n, sprache, fassungen)
        arm_1.append(a)
        arm_2.append(b)

    kopf = _kopf_lokal(
        settings,
        slug,
        lauf,
        baseline,
        "sprache_gold" if gold else "sprache",
        "je_frage",
        top_k,
        herkunft,
        gesamt,
        preise,
        BEDEUTUNG_SPRACHE,
    )
    kopf["laeufe_je_fall"] = laeufe_je_frage
    kopf["fassungen"] = list(fassungen)
    kopf["falldatei_stand"] = str(daten["stand"]) if daten.get("stand") is not None else None
    zweite = fassungen[1].lower()
    return {
        "kopf": kopf,
        "aggregiert_je_kategorie": aggregiere(arm_1, top_k, preise, settings.model_name),
        f"aggregiert_je_kategorie_{zweite}": aggregiere(arm_2, top_k, preise, settings.model_name),
        "fragen": [asdict(e) for e in arm_1],
        f"fragen_{zweite}": [asdict(e) for e in arm_2],
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
    # Dieselbe Begruendung wie fuer die Metrik, nur eine Ebene spaeter gelernt:
    # Eine LLM-seitige Zahl ohne die Laufzahl ist nicht einzuordnen. Die Zahlen
    # aus EN-3 und EN-6 waren Stichproben von eins, und niemand konnte es der
    # Datei ansehen.
    if kopf.get("modus") != "nur_retrieval" and not kopf.get("laeufe_je_frage"):
        raise ValueError(
            "Ergebniskopf ohne laeufe_je_frage. Eine LLM-seitige Zahl ohne die "
            "Anzahl Laeufe ist nicht einzuordnen - und eine Stichprobe von eins "
            "sieht genauso aus wie eine Eigenschaft. Datei wird nicht geschrieben."
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
        f"LAEUFE JE FRAGE: {kopf.get('laeufe_je_frage')}   "
        f"temperature={kopf.get('sampling_temperature')}",
        f"METRIK: {kopf['metrik']}",
        _zielzeile(kopf),
        f"{'=' * 78}",
        "",
        f"{'Kategorie':<22}{'Hit@k':>6}{'Kontext':>8}{'MRR':>6}"
        f"{'Eskal.':>7}{'wieErw':>8}{' Tor':<10}{'Sprache':>8}{'p50':>6}",
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
        # Die Antwortsprache steht ab dem 2026-09-24 in der Tabelle, nicht in
        # einer Sonderzeile darunter. Gezaehlt werden LAEUFE: "18/20" heisst
        # achtzehn von zwanzig bewertbaren Laeufen dieser Kategorie folgten der
        # Sprache der Frage.
        if a.get("sprache_laeufe_bewertbar"):
            sprache = f"{a['sprache_folgt_frage_laeufe']}/{a['sprache_laeufe_bewertbar']}"
        else:
            sprache = "–"
        # Eskalation als LAEUFE, gleiche Zaehlweise wie die Sprachspalte. "3/10"
        # heisst drei von zehn Laeufen dieser Kategorie eskalierten.
        eskal = f"{a['eskaliert_laeufe']}/{a['laeufe_gesamt']}"

        # "wieErw" macht die Zeile ueberhaupt lesbar, ergaenzt am 2026-09-24.
        # Grund: Die Kategorie fremder_mandant bedeutet je Mandant das GEGENTEIL.
        # Bei demo-fellgate muss eskaliert werden, die Antwort liegt nur im
        # fremden Korpus. Bei demo-acme und demo-nordwind gerade nicht - dort ist
        # es eine fremd KLINGENDE Frage, die der eigene Korpus deckt. Ohne diese
        # Spalte liest sich "9/10 eskaliert" bei Acme wie ein Erfolg und ist fast
        # durchweg ein Fehlschlag. Siehe OP-058.
        # Gegaengelt am vorhandenen Feld eskalation_wie_erwartet: Es ist None,
        # wenn keine Frage eine Eskalationsentscheidung hatte - also im reinen
        # Retrievallauf. Dort waere "0/4" kein Ergebnis, sondern ein Fehlschluss.
        wie_erwartet = (
            f"{a['eskalation_wie_erwartet_laeufe']}/{a['laeufe_gesamt']}"
            if a.get("eskalation_wie_erwartet") is not None
            else "–"
        )
        z.append(
            f"{kategorie:<22}{hit:>6}{antw:>8}{mrr:>6}"
            f"{eskal:>7}{wie_erwartet:>8} {tore:<9}{sprache:>8}{p50:>6}"
        )
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
        # Genannt wird jede Frage, bei der MINDESTENS EIN Lauf die Sprache
        # gewechselt hat. Das ist strenger als "die Mehrheit war falsch", und
        # zwar mit Absicht: Ein Interessent klickt einmal.
        z += [
            "",
            f"ACHTUNG: Antwortsprache folgt der Frage NICHT: {', '.join(falsche_sprache)}",
            "  Genannt ist jede Frage mit mindestens einem abweichenden Lauf.",
            "  Vor einem Interessenten ist das der sichtbarste Fehler ueberhaupt.",
        ]
        for f in bericht["fragen"]:
            if f["id"] in falsche_sprache:
                z.append(
                    f"  {f['id']}: {f.get('sprache_folgt_frage_anzahl')}/"
                    f"{f.get('sprache_bewertbar_anzahl')} Laeufe folgten, "
                    f"Sprachen {f.get('sprachen_der_antworten')}"
                )

    instabil = sorted(
        {
            fid
            for a in bericht["aggregiert_je_kategorie"].values()
            for fid in a.get("instabile_fragen", [])
        }
    )
    if instabil:
        # Kein ACHTUNG: Eine schwankende Eskalationsentscheidung ist ein Befund
        # und nicht zwingend ein Mangel. Sie muss nur sichtbar sein, weil sie bei
        # n=1 wie eine Eigenschaft aussieht.
        z += ["", f"Uneinheitliche Eskalationsentscheidung ueber die Laeufe: {', '.join(instabil)}"]
        for f in bericht["fragen"]:
            if f["id"] in instabil:
                z.append(
                    f"  {f['id']}: {f.get('eskaliert_anzahl')} von {f.get('laeufe')} Laeufen "
                    f"eskaliert, erwartet war {f.get('erwartet_eskalation')}"
                )

    retrieval_wackelt = sorted(
        {
            fid
            for a in bericht["aggregiert_je_kategorie"].values()
            for fid in a.get("retrieval_nicht_identisch", [])
        }
    )
    if retrieval_wackelt:
        # Das waere ein echter Befund: Retrieval ist deterministisch, und wenn
        # es das nicht ist, taugt keine Zahl in diesem Bericht.
        z += [
            "",
            f"ACHTUNG: Retrieval NICHT identisch ueber die Laeufe: {', '.join(retrieval_wackelt)}",
            "  Das darf nicht vorkommen. Bis zur Klaerung traegt dieser Bericht nichts.",
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
        "--laeufe",
        type=int,
        default=5,
        help=(
            "Laeufe JE FRAGE, Standard 5. Die Generierung wuerfelt; eine "
            "Stichprobe von eins ist von einer Eigenschaft nicht zu unterscheiden"
        ),
    )
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
    p.add_argument(
        "--verlauf",
        action="store_true",
        help="Verlaufsfaelle aus eval/<mandant>/verlauf.yaml, mit Verlauf und der Laufzahl je Fall",
    )
    p.add_argument(
        "--kontrolle",
        action="store_true",
        help="mit --verlauf: dieselben Folgefragen ohne Verlauf, --laeufe je Frage",
    )
    p.add_argument(
        "--gegenprobe",
        action="store_true",
        help="Goldfragen verschraenkt, A ohne Verlauf und B nach einer fremden ersten Runde",
    )
    p.add_argument(
        "--sprache",
        action="store_true",
        help="Sprachfaelle aus eval/<mandant>/sprache.yaml, in zwei Fassungen, siehe --fassungen",
    )
    p.add_argument(
        "--fassungen",
        default=None,
        help=(
            "mit --sprache: genau zwei der Fassungen A, B und O, durch Komma getrennt, "
            "Standard A,B. A ohne Vorgabe und ohne Sprachpruefung, B mit vorgegebener "
            "Sprache, O ohne Vorgabe mit Sprachpruefung"
        ),
    )
    p.add_argument(
        "--ohne-sprachpruefung",
        action="store_true",
        help=(
            "Goldsatz, --verlauf und --gegenprobe ohne zweiten Aufruf zur Antwortsprache; "
            "die Sprache wird trotzdem festgelegt und beurteilt"
        ),
    )
    p.add_argument(
        "--gold",
        action="store_true",
        help="mit --sprache: der Goldsatz, mit der erwarteten Sprache aus sprache.yaml",
    )
    p.add_argument(
        "--nur-b",
        action="store_true",
        help="mit --gegenprobe: nur Fassung B; der Bezug fuer A ist eine fruehere Datei",
    )
    args = p.parse_args(argv)

    gespraech = args.verlauf or args.gegenprobe or args.sprache
    if args.kontrolle and not args.verlauf:
        p.error("--kontrolle gilt nur zusammen mit --verlauf.")
    if args.nur_b and not args.gegenprobe:
        p.error("--nur-b gilt nur zusammen mit --gegenprobe.")
    if args.gold and not args.sprache:
        p.error("--gold gilt nur zusammen mit --sprache.")
    if args.sprache and (args.verlauf or args.gegenprobe):
        p.error("--sprache ist ein eigener Lauf, ohne --verlauf und --gegenprobe.")
    if args.verlauf and args.gegenprobe:
        p.error("--verlauf und --gegenprobe sind zwei getrennte Laeufe.")
    if args.fassungen is not None and not args.sprache:
        p.error("--fassungen gilt nur zusammen mit --sprache.")
    fassungen = tuple((args.fassungen or "A,B").split(","))
    if (
        len(fassungen) != 2
        or len(set(fassungen)) != 2
        or not set(fassungen) <= set(FASSUNGEN_SPRACHE)
    ):
        p.error("--fassungen verlangt genau zwei verschiedene der Buchstaben A, B und O, etwa A,O.")
    if args.ohne_sprachpruefung and (args.sprache or args.base_url or args.retrieval_only):
        p.error(
            "--ohne-sprachpruefung gilt fuer Goldsatz, --verlauf und --gegenprobe. Bei "
            "--sprache sagt es die Fassung, gegen eine Instanz entscheidet die Instanz, "
            "und --retrieval-only ruft kein Modell."
        )
    if gespraech and (args.retrieval_only or args.base_url or args.top_k is not None):
        p.error(
            "--verlauf, --gegenprobe und --sprache laufen lokal mit Modell und mit dem "
            "top_k des Mandanten; --retrieval-only, --base-url und --top-k passen nicht dazu."
        )

    if args.laeufe < 1:
        p.error("--laeufe muss mindestens 1 sein.")
    if args.laeufe == 1 and not args.retrieval_only:
        # Kein Fehler, aber ein Hinweis: Ein LLM-Lauf mit n=1 ist zulaessig und
        # manchmal gewollt. Er darf nur nicht unbemerkt als Basis dienen.
        print(
            "HINWEIS: --laeufe 1 bei einem LLM-Lauf. Jede LLM-seitige Zahl "
            "dieses Berichts ist damit eine Stichprobe von eins.",
            file=sys.stderr,
        )

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

    datei_noetig = (
        "sprache.yaml" if args.sprache else ("verlauf.yaml" if gespraech else "gold.yaml")
    )
    slugs = (
        sorted(d.name for d in EVAL_DIR.iterdir() if d.is_dir() and (d / datei_noetig).is_file())
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
            pruefung = not args.ohne_sprachpruefung
            if args.verlauf:
                bericht = fahre_verlauf(
                    slug, args.lauf, args.baseline, not args.kontrolle, args.laeufe, pruefung
                )
            elif args.sprache:
                bericht = fahre_sprache(
                    slug,
                    args.lauf,
                    args.baseline,
                    args.gold,
                    (fassungen[0], fassungen[1]),
                )
            elif args.gegenprobe:
                bericht = fahre_gegenprobe(
                    slug,
                    args.lauf,
                    args.baseline,
                    args.laeufe,
                    nur_b=args.nur_b,
                    sprachpruefung=pruefung,
                )
            else:
                bericht = fahre(
                    slug,
                    args.top_k,
                    args.retrieval_only,
                    args.lauf,
                    args.baseline,
                    cloud,
                    # Ohne LLM gibt es nichts zu wiederholen: Retrieval ist
                    # deterministisch, und n Laeufe davon kosten Zeit ohne
                    # Erkenntnis.
                    1 if args.retrieval_only else args.laeufe,
                    sprachpruefung=pruefung,
                )
            pruefe_kopf(bericht)
            stempel = bericht["kopf"]["zeitstempel"].replace(":", "").replace("-", "")
            datei = ERGEBNIS_DIR / f"{stempel}-{slug}-lauf{args.lauf}.json"
            datei.write_text(json.dumps(bericht, ensure_ascii=False, indent=2), encoding="utf-8")
            if bericht["kopf"].get("fassungen") != ["B"]:
                print(tabelle(bericht))
            zweite = next((k for k in bericht if k.startswith("aggregiert_je_kategorie_")), None)
            if zweite is not None:
                buchstabe = zweite.removeprefix("aggregiert_je_kategorie_")
                titel = (
                    UEBERSCHRIFT_FASSUNG[buchstabe.upper()]
                    if args.sprache
                    else "nach der fremden ersten Runde"
                )
                print(f"\nFassung {buchstabe.upper()}, {titel}:")
                print(
                    tabelle(
                        {
                            **bericht,
                            "aggregiert_je_kategorie": bericht[zweite],
                            "fragen": bericht[f"fragen_{buchstabe}"],
                        }
                    )
                )
            print(f"\ngeschrieben: {datei.relative_to(PROJEKTWURZEL)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
