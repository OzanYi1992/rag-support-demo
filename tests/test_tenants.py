"""Tests des Mandantenmodells.

Schwerpunkt sind die Faelle, die STILL scheitern wuerden: ein Traversal, der
durchrutscht, ein doppeltes url_token, das den ersten Treffer liefert, ein
vergessenes public_image_allowed, das als True gilt. Alle drei enden damit, dass
ein Mandant Daten eines anderen zu sehen bekommt oder veroeffentlicht wird -
ohne Absturz und ohne Fehlermeldung.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.tenants import (
    MIN_TOKEN_LENGTH,
    AmbiguousUrlToken,
    InvalidTenantSlug,
    TenantConfig,
    TenantNotFound,
    generate_url_token,
    list_tenants,
    load_tenant,
    resolve_token,
    tenants_for_public_image,
)

# --- Die beiden Demo-Mandanten laden ---------------------------------------


def test_load_both_demo_tenants(demo_tenants_dir: Path) -> None:
    acme = load_tenant("demo-acme", demo_tenants_dir)
    nordwind = load_tenant("demo-nordwind", demo_tenants_dir)

    assert acme.slug == "demo-acme"
    assert nordwind.slug == "demo-nordwind"
    assert acme.display_name == "ACME Elektronikhandel"
    assert acme.languages == ["de", "en"]
    assert nordwind.languages == ["de"]
    assert acme.retrieval_top_k == 6
    assert nordwind.retrieval_top_k is None
    assert acme.escalation_message.strip()
    assert nordwind.escalation_message.strip()


def test_slug_kommt_aus_dem_verzeichnis(demo_tenants_dir: Path) -> None:
    """Der Slug stammt aus dem Verzeichnisnamen, nicht aus der Datei.

    Sonst koennten Verzeichnis und Inhalt auseinanderlaufen und ein Mandant den
    Namen eines anderen tragen.
    """
    assert load_tenant("demo-acme", demo_tenants_dir).slug == "demo-acme"


def test_list_tenants(demo_tenants_dir: Path) -> None:
    assert list_tenants(demo_tenants_dir) == ["demo-acme", "demo-nordwind"]


# --- Slug-Validierung ------------------------------------------------------


@pytest.mark.parametrize(
    "slug",
    [
        "AB",  # zu kurz und Grossbuchstaben
        "Gross",  # Grossbuchstaben
        "mit_unterstrich",  # Unterstrich nicht erlaubt
        "",  # leer
        "ab",  # zwei Zeichen, unter der Mindestlaenge
        "x" * 41,  # ueber der Hoechstlaenge
        "mit punkt.yaml",  # Leerzeichen und Punkt
    ],
)
def test_invalid_slug_rejected(slug: str, demo_tenants_dir: Path) -> None:
    with pytest.raises(InvalidTenantSlug):
        load_tenant(slug, demo_tenants_dir)


@pytest.mark.parametrize(
    "slug",
    [
        "../etc",
        "a/../../b",
        "demo-acme/../..",
        "../../../../etc/passwd",
        "./demo-acme",
        "demo-acme/",
    ],
)
def test_traversal_rejected(slug: str, demo_tenants_dir: Path) -> None:
    """Traversal scheitert bereits an der Form des Slugs.

    Wichtig ist nicht nur, DASS abgewiesen wird, sondern dass es an der
    Slug-Pruefung scheitert und nicht erst daran, dass zufaellig keine Datei
    gefunden wurde - deshalb InvalidTenantSlug und nicht TenantNotFound.
    """
    with pytest.raises(InvalidTenantSlug):
        load_tenant(slug, demo_tenants_dir)


def test_unknown_slug_raises(demo_tenants_dir: Path) -> None:
    """Gueltige Form, aber kein Mandant: TenantNotFound, nicht InvalidTenantSlug."""
    with pytest.raises(TenantNotFound):
        load_tenant("gibt-es-nicht", demo_tenants_dir)


# --- url_token -------------------------------------------------------------


def test_resolve_token_findet_richtigen_mandanten(demo_tenants_dir: Path) -> None:
    tenant = resolve_token("demo-acme-oeffentlich-7f3a91c4e2", demo_tenants_dir)
    assert tenant.slug == "demo-acme"


def test_unknown_token_rejected(demo_tenants_dir: Path) -> None:
    with pytest.raises(TenantNotFound):
        resolve_token("dieses-token-gibt-es-nicht-0000", demo_tenants_dir)


def test_leeres_token_wird_abgewiesen(demo_tenants_dir: Path) -> None:
    with pytest.raises(TenantNotFound):
        resolve_token("", demo_tenants_dir)


def test_duplicate_url_token_raises(tmp_tenants_dir_mit_doppeltem_token: Path) -> None:
    """Zwei Mandanten mit demselben Token: Abbruch statt stillem Erst-Treffer.

    Der kopierte-tenant.yaml-Fehler. Ein zurueckgegebener Erst-Treffer hiesse,
    dass ein Interessent den Mandanten eines anderen sieht.
    """
    with pytest.raises(AmbiguousUrlToken) as excinfo:
        resolve_token("geteiltes-token-1234567890", tmp_tenants_dir_mit_doppeltem_token)

    meldung = str(excinfo.value)
    assert "erster-mandant" in meldung
    assert "zweiter-mandant" in meldung
    # Der Token selbst gehoert nicht in die Meldung (ADR-002).
    assert "geteiltes-token-1234567890" not in meldung


def test_url_token_min_length() -> None:
    zu_kurz = "x" * (MIN_TOKEN_LENGTH - 1)
    with pytest.raises(ValidationError):
        TenantConfig(
            slug="test-mandant",
            display_name="Test",
            languages=["de"],
            escalation_message="nichts gefunden",
            url_token=zu_kurz,
        )


def test_generate_url_token_ist_lang_genug_und_verschieden() -> None:
    a = generate_url_token()
    b = generate_url_token()
    assert len(a) >= MIN_TOKEN_LENGTH
    assert a != b


# --- public_image_allowed (ADR-014) ----------------------------------------


def test_public_image_defaults_false(tmp_tenants_dir: Path) -> None:
    """Fehlt das Feld in der YAML, ist es False - nicht True.

    Ein vergessenes Flag muss in die sichere Richtung wirken, sonst landet ein
    Interessentenmandant in einem oeffentlichen Image.
    """
    assert load_tenant("ohne-flag", tmp_tenants_dir).public_image_allowed is False


def test_absent_flag_not_in_public_list(tmp_tenants_dir: Path) -> None:
    assert tenants_for_public_image(tmp_tenants_dir) == []


def test_demo_tenants_in_public_list(demo_tenants_dir: Path) -> None:
    """Die beiden Demo-Mandanten setzen das Flag ausdruecklich auf True."""
    assert tenants_for_public_image(demo_tenants_dir) == ["demo-acme", "demo-nordwind"]
