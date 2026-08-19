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
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

EVAL_DIR = PROJEKTWURZEL / "eval"
ERGEBNIS_DIR = EVAL_DIR / "results"

# Welches Tor hat gegriffen. ADR-019 haelt die Gruende getrennt, damit genau
# diese Auswertung moeglich ist.
TOR_RETRIEVAL = {REASON_NO_HITS, REASON_BELOW_THRESHOLD, REASON_FLAT}
TOR_GROUNDEDNESS = {REASON_NOT_GROUNDED}
TOR_SONSTIGES = {REASON_UNPARSEABLE}

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

    latenz_retrieval_ms: int | None = None
    latenz_generierung_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    antworttext: str | None = None


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


def eine_frage(
    slug: str,
    eintrag: dict[str, Any],
    settings: Settings,
    top_k: int,
    gesamtzahl: int,
    nur_retrieval: bool,
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

    if erg.erwartete_quelle:
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

    top = alle[:top_k]
    erg.top_k_scores = [round(t.score, 4) for t in top]
    erg.top_k_quellen = [t.source_file for t in top]
    erg.fremde_chunks = sorted({t.tenant_slug for t in alle if t.tenant_slug != slug})

    if nur_retrieval:
        return erg

    antwort = answer(slug, erg.frage, settings=settings, embeddings=embedder)
    erg.eskaliert = antwort.escalated
    erg.grund = antwort.escalation_reason
    erg.tor = _tor_von(antwort.escalation_reason)
    erg.eskalation_wie_erwartet = antwort.escalated == erg.erwartet_eskalation
    erg.latenz_retrieval_ms = antwort.latency_ms_retrieval
    erg.latenz_generierung_ms = antwort.latency_ms_generation
    erg.prompt_tokens = antwort.prompt_tokens
    erg.completion_tokens = antwort.completion_tokens
    erg.antworttext = antwort.text
    return erg


def aggregiere(
    ergebnisse: list[Frageergebnis], top_k: int, preise: dict[str, Any], modell: str | None
) -> dict[str, Any]:
    """Je Kategorie getrennt. Bewusst kein Gesamtwert ueber alle Fragen."""
    je_kategorie: dict[str, list[Frageergebnis]] = {}
    for e in ergebnisse:
        je_kategorie.setdefault(e.kategorie, []).append(e)

    aus: dict[str, Any] = {}
    for kategorie, gruppe in sorted(je_kategorie.items()):
        mit_ziel = [e for e in gruppe if e.erwartete_quelle]
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
            "eskaliert": len(eskaliert),
            "eskaliert_retrieval": sum(1 for e in eskaliert if e.tor == "retrieval"),
            "eskaliert_groundedness": sum(1 for e in eskaliert if e.tor == "groundedness"),
            "eskaliert_sonstiges": sum(1 for e in eskaliert if e.tor == "sonstiges"),
            "eskalation_wie_erwartet": sum(1 for e in gruppe if e.eskalation_wie_erwartet)
            if any(e.eskalation_wie_erwartet is not None for e in gruppe)
            else None,
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
    slug: str, top_k_erzwungen: int | None, nur_retrieval: bool, lauf: str, baseline: bool
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

    ergebnisse = [
        eine_frage(slug, f, settings, top_k, gesamt, nur_retrieval) for f in gold["fragen"]
    ]

    return {
        "kopf": {
            "zeitstempel": datetime.now(UTC).isoformat(timespec="seconds"),
            "lauf": lauf,
            "baseline": baseline,
            "mandant": slug,
            "modus": "nur_retrieval" if nur_retrieval else "retrieval_und_llm",
            "git_commit": _git_commit(),
            "embedding_modell": settings.embedding_model,
            "embedding_dimension": settings.embedding_dimension,
            "llm_modell": None if nur_retrieval else settings.model_name,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": top_k,
            "top_k_herkunft": herkunft,
            "eskalationsstrategie": settings.escalation_strategy,
            "retrieval_score_threshold": settings.retrieval_score_threshold,
            "chunks_im_index": gesamt,
            "preise_hinterlegt": bool(preise.get("stand")),
        },
        "aggregiert_je_kategorie": aggregiere(
            ergebnisse,
            top_k,
            preise,
            None if nur_retrieval else settings.model_name,
        ),
        "fragen": [asdict(e) for e in ergebnisse],
    }


def tabelle(bericht: dict[str, Any]) -> str:
    kopf = bericht["kopf"]
    z = [
        f"{'=' * 78}",
        f"{kopf['mandant']}   Lauf {kopf['lauf']}   {kopf['modus']}",
        f"top_k={kopf['top_k']} ({kopf['top_k_herkunft']})   "
        f"chunks={kopf['chunks_im_index']}   strategie={kopf['eskalationsstrategie']}",
        f"embedding={kopf['embedding_modell']}   llm={kopf['llm_modell']}   "
        f"commit={kopf['git_commit']}",
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
    args = p.parse_args(argv)

    slugs = (
        sorted(d.name for d in EVAL_DIR.iterdir() if d.is_dir() and (d / "gold.yaml").is_file())
        if args.all
        else [args.slug]
    )

    ERGEBNIS_DIR.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        bericht = fahre(slug, args.top_k, args.retrieval_only, args.lauf, args.baseline)
        stempel = bericht["kopf"]["zeitstempel"].replace(":", "").replace("-", "")
        ziel = ERGEBNIS_DIR / f"{stempel}-{slug}-lauf{args.lauf}.json"
        ziel.write_text(json.dumps(bericht, ensure_ascii=False, indent=2), encoding="utf-8")
        print(tabelle(bericht))
        print(f"\ngeschrieben: {ziel.relative_to(PROJEKTWURZEL)}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
