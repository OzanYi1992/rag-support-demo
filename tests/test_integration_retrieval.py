"""Integrationstests mit dem echten Embedding-Modell.

Standardmaessig ausgeschlossen (`-m 'not integration'` in pyproject). Explizit
fahren mit:

    .venv/bin/pytest -m integration

Beim ersten Lauf werden rund 470 MB Modellgewichte geladen. Danach liegen sie im
HuggingFace-Cache.

Diese Tests pruefen, was der gefakte Embedder nicht kann: Semantik. Sie sind der
Grund, warum es sie ueberhaupt gibt - die Unit-Tests belegen den Mechanismus, hier
wird belegt, dass das Retrieval etwas taugt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.embeddings import build_embeddings
from app.ingest import ingest_tenant
from app.search import search_tenant

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parent.parent

# Merkmale, die es im jeweils anderen Mandanten weder gibt noch geben koennte.
ACME_MERKMAL = "Brauche ich eine RMA-Nummer fuer eine Ruecksendung?"
NORDWIND_MERKMAL = "Wie laeuft die Zwei-Mann-Montage mit Terminfenster ab?"

# Frage ohne ein einziges gemeinsames Wort mit technical-support-en.md.
# Geprueft, nicht behauptet: siehe test_deutsche_frage_teilt_kein_wort.
DEUTSCHE_FRAGE_AUF_ENGLISCHES_DOKUMENT = (
    "Wie schnell bekomme ich Hilfe, wenn eine Anlage stillsteht?"
)


@pytest.fixture(scope="module")
def echte_umgebung(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Ingestiert beide Demo-Mandanten mit dem echten Modell."""
    tenants_root = REPO_ROOT / "tenants"
    for slug in ("demo-acme", "demo-nordwind"):
        docs = tenants_root / slug / "docs"
        if not docs.is_dir() or not list(docs.glob("*.md")):
            pytest.skip(
                f"Keine Dokumente unter {docs}. Die Demo-Inhalte muessen vorhanden sein, "
                f"bevor der Integrationstest etwas zeigen kann."
            )

    settings = Settings(
        openai_api_key="platzhalter",
        openai_model="platzhalter",
        tenants_dir=tenants_root,
        index_dir=tmp_path_factory.mktemp("index"),
    )
    embeddings = build_embeddings(settings)
    for slug in ("demo-acme", "demo-nordwind"):
        ingest_tenant(slug, settings=settings, embeddings=embeddings)
    return settings


def test_deutsche_frage_teilt_kein_wort_mit_dem_englischen_dokument() -> None:
    """Vorbedingung des naechsten Tests, hier geprueft statt angenommen.

    Ohne diese Pruefung koennte der Treffer auch durch Wortueberschneidung
    zustande kommen - etwa ueber Produktbezeichnungen, die in beiden Sprachen
    gleich sind. Dann waere er kein Beleg fuer semantisches Retrieval.
    """
    import re

    dokument = REPO_ROOT / "tenants" / "demo-acme" / "docs" / "technical-support-en.md"
    if not dokument.is_file():
        pytest.skip(f"{dokument} fehlt.")

    woerter_en = set(re.findall(r"[a-z]+", dokument.read_text(encoding="utf-8").lower()))
    woerter_de = set(re.findall(r"[a-zäöüß]+", DEUTSCHE_FRAGE_AUF_ENGLISCHES_DOKUMENT.lower()))

    assert not (woerter_de & woerter_en), (
        f"Frage und Dokument teilen Woerter: {sorted(woerter_de & woerter_en)}. "
        f"Der Treffer waere dann kein Beleg fuer semantisches Retrieval."
    )


@pytest.mark.xfail(
    reason=(
        "Language Bias: e5-small findet das englische Dokument bei deutscher Frage "
        "nur mit Wortbruecke. Gemessen am 2026-08-10 - Raenge 15/18/19/20 von 20. "
        "e5-large wurde gegengemessen und verworfen (1,7 GB, Ingest Faktor 8,4, "
        "engere Score-Spanne, zwei von drei sauberen Fragen weiterhin offen). "
        "Die Ursache ist benannt und muss nicht gesucht werden; offen ist die "
        "Abhilfe. Siehe knowledge/open-points.md, OP-019, OP-020, OP-021. "
        "Der Test bleibt bewusst stehen: Er schlaegt als xpass an, sobald eine "
        "Abhilfe wirkt."
    ),
    strict=False,
)
def test_deutsche_frage_findet_englisches_dokument(echte_umgebung: Settings) -> None:
    """Der cross-linguale Fall. Erwartet fehlschlagend, siehe Marker.

    Eine deutsche Frage soll ein englisches Quelldokument finden, ohne ein
    einziges Wort mit ihm zu teilen. Das gelingt derzeit nicht.

    Der Test wird nicht geloescht und nicht weichgeklopft. Ein uebersprungener
    Test mit dokumentiertem Grund ist ehrlicher als beides - und dieser hier
    meldet sich von selbst, wenn OP-020 oder OP-021 greifen.
    """
    treffer = search_tenant(
        "demo-acme", DEUTSCHE_FRAGE_AUF_ENGLISCHES_DOKUMENT, k=5, settings=echte_umgebung
    )

    assert treffer
    quellen = {t.source_file for t in treffer}
    assert "technical-support-en.md" in quellen, (
        f"Erwartet wurde ein Treffer im englischen Dokument, gefunden: {sorted(quellen)}"
    )


@pytest.mark.parametrize(
    ("slug", "fremd", "eigene_frage", "fremde_frage"),
    [
        ("demo-acme", "demo-nordwind", ACME_MERKMAL, NORDWIND_MERKMAL),
        ("demo-nordwind", "demo-acme", NORDWIND_MERKMAL, ACME_MERKMAL),
    ],
)
def test_isolation_in_beide_richtungen(
    echte_umgebung: Settings, slug: str, fremd: str, eigene_frage: str, fremde_frage: str
) -> None:
    """Frage nach dem Merkmal des anderen Mandanten - zwei Nachweise.

    Erstens: kein Treffer traegt den fremden Slug. Das ist die harte Grenze aus
    ADR-001.

    Zweitens: die Scores liegen niedriger als bei der eigenen Frage. Das zeigt,
    dass die Trennung nicht nur strukturell haelt, sondern dass der Index auch
    inhaltlich nichts Passendes hergibt - die Grundlage dafuer, dass ADR-003
    spaeter eskaliert statt zu erfinden.
    """
    fremde_treffer = search_tenant(slug, fremde_frage, k=5, settings=echte_umgebung)
    eigene_treffer = search_tenant(slug, eigene_frage, k=5, settings=echte_umgebung)

    assert fremde_treffer and eigene_treffer
    assert all(t.tenant_slug == slug for t in fremde_treffer)
    assert not any(t.tenant_slug == fremd for t in fremde_treffer)

    assert fremde_treffer[0].score < eigene_treffer[0].score, (
        f"Das fremde Merkmal erreicht Score {fremde_treffer[0].score:.4f}, das eigene "
        f"{eigene_treffer[0].score:.4f}. Ohne Abstand koennte spaeter kein Schwellwert "
        f"zwischen beiden trennen (ADR-003, OP-004)."
    )


def test_gemeinsames_thema_wird_unterschiedlich_beantwortet(echte_umgebung: Settings) -> None:
    """Lieferzeiten gibt es bei beiden - die Antworten kommen aus je eigenen Quellen.

    Zeigt, dass die Trennung nicht durch Themenferne entsteht, sondern durch den
    Mechanismus. Waeren die Mandanten nur inhaltlich weit auseinander, bewiese
    der Isolationstest wenig.
    """
    frage = "Wie lange dauert die Lieferung?"

    acme = search_tenant("demo-acme", frage, k=3, settings=echte_umgebung)
    nordwind = search_tenant("demo-nordwind", frage, k=3, settings=echte_umgebung)

    assert acme and nordwind
    assert all(t.tenant_slug == "demo-acme" for t in acme)
    assert all(t.tenant_slug == "demo-nordwind" for t in nordwind)
    assert {t.text for t in acme}.isdisjoint({t.text for t in nordwind})


def test_dimension_und_norm_des_echten_modells(echte_umgebung: Settings) -> None:
    """Belegt ADR-016 und die Voraussetzung aus ADR-008 am echten Modell."""
    embeddings = build_embeddings(echte_umgebung)
    vektor = embeddings.embed_query("Eine beliebige Frage.")

    assert len(vektor) == 384
    norm = sum(v * v for v in vektor) ** 0.5
    assert abs(norm - 1.0) < 1e-3
    assert embeddings.max_seq_length == 512
