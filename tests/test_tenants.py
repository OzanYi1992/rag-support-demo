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
import yaml
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
from app.texts import VERFUEGBARE_SPRACHEN, passt_zur_sprache
from tests.conftest import lege_mandant_an

# --- Die beiden Demo-Mandanten laden ---------------------------------------


def test_load_both_demo_tenants(demo_tenants_dir: Path) -> None:
    acme = load_tenant("demo-acme", demo_tenants_dir)
    nordwind = load_tenant("demo-nordwind", demo_tenants_dir)

    assert acme.slug == "demo-acme"
    assert nordwind.slug == "demo-nordwind"
    assert acme.display_name == "ACME Elektronikhandel"
    assert acme.language == "de"
    assert nordwind.language == "de"
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
    """Sortiert, vollstaendig, und genau diese drei.

    Bewusst eine feste Liste und kein "mindestens": Ein Mandant, der hier
    unbemerkt dazukommt, ist ein Mandant, der unbemerkt in ein oeffentliches
    Image geraten koennte.
    """
    assert list_tenants(demo_tenants_dir) == ["demo-acme", "demo-fellgate", "demo-nordwind"]


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
            topics="Versand und Retouren",
            escalation_message="nichts gefunden",
            url_token=zu_kurz,
        )


def test_generate_url_token_ist_lang_genug_und_verschieden() -> None:
    a = generate_url_token()
    b = generate_url_token()
    assert len(a) >= MIN_TOKEN_LENGTH
    assert a != b


# --- language ---------------------------------------------------------------


def test_language_default_ist_de(tmp_path: Path) -> None:
    """Fehlt das Feld, bleibt der Mandant deutsch.

    Der Default entscheidet ueber das Verhalten der beiden bestehenden
    Mandanten. Waere er "en", schaltete ein vergessenes Feld sie um.
    """
    slug = "ohne-sprache"
    tenant_dir = tmp_path / slug
    tenant_dir.mkdir()
    (tenant_dir / "tenant.yaml").write_text(
        yaml.safe_dump(
            {
                "display_name": "Ohne Sprache",
                "topics": "Versand und Retouren",
                "escalation_message": "Dazu finde ich nichts.",
                "url_token": "ohne-sprache-token-1234",
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    assert load_tenant(slug, tmp_path).language == "de"


def test_language_en_wird_geladen(tmp_path: Path) -> None:
    lege_mandant_an(tmp_path, "demo-englisch", "Doc", "englisch-token-1234567890", language="en")
    assert load_tenant("demo-englisch", tmp_path).language == "en"


def test_dritte_sprache_scheitert_beim_laden(tmp_path: Path) -> None:
    """Eine Sprache ohne Textkatalog wird beim LADEN abgewiesen, nicht spaeter.

    Wuerde sie durchgelassen, liefe die Oberflaeche in einer Sprache und der
    Mandant in einer anderen - und aufgefallen waere es erst, wenn jemand die
    Seite aufruft.
    """
    # Eigene Texte statt der Fixture-Defaults: Die gibt es fuer "fr" bewusst
    # nicht, und der Test will die Ablehnung beim LADEN pruefen, nicht ein
    # Scheitern schon beim Anlegen der Datei. Das gilt fuer escalation_message
    # und seit dem 2026-09-24 genauso fuer topics.
    lege_mandant_an(
        tmp_path,
        "demo-franzoesisch",
        "Doc",
        "franz-token-1234567890",
        escalation_message="Je ne trouve rien a ce sujet.",
        language="fr",
        topics="expedition, retours et paiement",
    )

    with pytest.raises(ValidationError) as fehler:
        load_tenant("demo-franzoesisch", tmp_path)

    meldung = str(fehler.value)
    # Die Meldung muss sagen, WAS fehlt, nicht nur DASS der Wert falsch ist.
    assert "fr" in meldung
    assert "app/texts.py" in meldung


def test_language_haelt_sich_an_den_textkatalog() -> None:
    """Schema und Katalog koennen nicht auseinanderlaufen.

    Die erlaubten Werte werden aus dem Katalog abgeleitet und nicht hier
    zweitgefuehrt. Dieser Test haelt die Ableitung fest: Kaeme eine Sprache
    hinzu, ohne dass der Katalog sie kennt, faellt es hier auf.
    """
    # Beide Mandantentexte muessen der geprueften Sprache folgen, sonst weist
    # der Sprachvalidator sie ab - und dieser Test prueft die Ableitung der
    # zulaessigen Sprachen, nicht den Sprachvalidator.
    texte_je_sprache = {
        "de": ("Versand und Retouren", "nichts gefunden"),
        "en": ("shipping and returns", "nothing found"),
    }
    for sprache in VERFUEGBARE_SPRACHEN:
        themen, eskalation = texte_je_sprache[sprache]
        mandant = TenantConfig(
            slug="test-mandant",
            display_name="Test",
            language=sprache,
            topics=themen,
            escalation_message=eskalation,
            url_token="test-token-1234567890",
        )
        assert mandant.language == sprache


def test_eskalationsnachricht_in_falscher_sprache_scheitert_beim_laden(
    tmp_path: Path,
) -> None:
    """Der Fall, der ohne diese Pruefung jeden Test bestehen wuerde.

    Ein englischer Mandant mit deutscher Eskalationsnachricht faellt sonst erst
    vor einem Interessenten auf - und zwar bei der nicht gedeckten Frage, also
    genau dort, wo das System sich von seiner guten Seite zeigen soll.
    """
    lege_mandant_an(
        tmp_path,
        "demo-falschsprachig",
        "Doc",
        "falschsprachig-token-12345",
        escalation_message=(
            "Dazu finde ich in den Unterlagen leider nichts. Bitte wenden Sie sich an den Support."
        ),
        language="en",
    )

    with pytest.raises(ValidationError) as fehler:
        load_tenant("demo-falschsprachig", tmp_path)

    meldung = str(fehler.value)
    assert "escalation_message" in meldung
    assert "demo-falschsprachig" in meldung


def test_eskalationsnachricht_in_richtiger_sprache_laedt(tmp_path: Path) -> None:
    """Gegenprobe. Ohne sie zeigt der Test oben nur, dass irgendetwas scheitert."""
    lege_mandant_an(
        tmp_path,
        "demo-richtigsprachig",
        "Doc",
        "richtigsprachig-token-1234",
        escalation_message=(
            "I cannot find anything about that in the documents. Please write to our support desk."
        ),
        language="en",
    )

    mandant = load_tenant("demo-richtigsprachig", tmp_path)
    assert mandant.language == "en"


def test_bestehende_mandanten_bestehen_die_sprachpruefung(demo_tenants_dir: Path) -> None:
    """Die Pruefung darf die gepflegten Mandanten nicht abweisen.

    Weist sie hier etwas ab, ist die Heuristik zu scharf - nicht der Mandant
    falsch. Der Test laeuft gegen das echte tenants/-Verzeichnis und deckt
    damit jeden Mandanten ab, der spaeter dazukommt.
    """
    for slug in list_tenants(demo_tenants_dir):
        mandant = load_tenant(slug, demo_tenants_dir)
        assert mandant.escalation_message.strip(), slug


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
    """Alle drei Demo-Mandanten setzen das Flag ausdruecklich auf True.

    Der Vergleich mit list_tenants() darunter ist der eigentliche Punkt: Er
    haelt fest, dass HEUTE jeder vorhandene Mandant freigegeben ist. Kommt ein
    Interessentenmandant dazu, faellt dieser Test - und zwar bevor das Image
    gebaut wird.
    """
    assert tenants_for_public_image(demo_tenants_dir) == [
        "demo-acme",
        "demo-fellgate",
        "demo-nordwind",
    ]
    assert tenants_for_public_image(demo_tenants_dir) == list_tenants(demo_tenants_dir)


# --- topics ------------------------------------------------------------------


def test_topics_ist_pflicht():
    """Ein Mandant ohne Themenangabe soll sich nicht laden lassen.

    Kein Standardwert, und das ist Absicht: Ein leerer Default liesse eine
    Luecke in der Begruessung, ein generischer behauptete eine Abdeckung, die
    dieser Korpus vielleicht nicht hat.
    """
    with pytest.raises(ValidationError) as fehler:
        TenantConfig(
            slug="ohne-themen",
            display_name="Ohne Themen",
            language="de",
            escalation_message="nichts gefunden",
            url_token="test-token-1234567890",
        )
    assert "topics" in str(fehler.value)


def test_topics_in_falscher_sprache_scheitert_beim_laden(tmp_path: Path):
    """Dieselbe Pruefung wie fuer die Eskalationsnachricht, und aus demselben Grund.

    Der Unterschied: `topics` steht in der Begruessung, also im ERSTEN Satz, den
    ein Interessent liest. Ein deutscher Themensatz bei einem englischen
    Mandanten faellt damit sofort auf - aber erst vor dem Interessenten, wenn ihn
    hier niemand abweist.
    """
    lege_mandant_an(
        tmp_path,
        "demo-englisch",
        "Doc",
        "englisch-token-1234567890",
        language="en",
        topics="Versand, Retouren und Zahlung sowie die Lieferung",
    )
    with pytest.raises(ValidationError) as fehler:
        load_tenant("demo-englisch", tmp_path)
    meldung = str(fehler.value)
    assert "topics" in meldung
    assert "en" in meldung


def test_alle_demomandanten_tragen_themen_in_ihrer_sprache(demo_tenants_dir: Path):
    """Gegenprobe am echten Verzeichnis: Die drei Mandanten laden.

    Ohne diesen Test koennte der Sprachvalidator oben alles ablehnen und der
    vorige Test waere trotzdem gruen.
    """
    for slug in ("demo-acme", "demo-nordwind", "demo-fellgate"):
        mandant = load_tenant(slug, demo_tenants_dir)
        assert mandant.topics.strip(), slug
        assert passt_zur_sprache(mandant.topics, mandant.language) is None, slug
