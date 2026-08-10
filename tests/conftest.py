"""Fixtures fuer die Mandantentests.

Keine Fixture setzt Umgebungsvariablen, und kein Test baut `Settings()`. Die
Mandantenfunktionen bekommen ihr Wurzelverzeichnis injiziert - dadurch laufen
diese Tests ohne `.env` und ohne LLM-Schluessel, mit denen sie nichts zu tun
haben.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def demo_tenants_dir() -> Path:
    """Das echte tenants/-Verzeichnis des Repos mit den beiden Demo-Mandanten."""
    return REPO_ROOT / "tenants"


@pytest.fixture
def tmp_tenants_dir(tmp_path: Path) -> Path:
    """Ein Wurzelverzeichnis mit einem Mandanten OHNE public_image_allowed.

    Bewusst nicht einer der Demo-Mandanten: Der Default-Test soll das Modell
    pruefen, nicht die YAML-Datei. Wuerde er gegen demo-acme laufen, pruefte er
    nur, dass dort `true` steht.
    """
    slug = "ohne-flag"
    tenant_dir = tmp_path / slug
    tenant_dir.mkdir()
    (tenant_dir / "tenant.yaml").write_text(
        yaml.safe_dump(
            {
                "display_name": "Ohne Flag",
                "languages": ["de"],
                "escalation_message": "Dazu finde ich nichts.",
                "url_token": "ohne-flag-token-1234567890",
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def tmp_tenants_dir_mit_doppeltem_token(tmp_path: Path) -> Path:
    """Zwei Mandanten mit demselben url_token - der kopierte-Datei-Fehler."""
    token = "geteiltes-token-1234567890"
    for slug in ("erster-mandant", "zweiter-mandant"):
        tenant_dir = tmp_path / slug
        tenant_dir.mkdir()
        (tenant_dir / "tenant.yaml").write_text(
            yaml.safe_dump(
                {
                    "display_name": slug,
                    "languages": ["de"],
                    "escalation_message": "Dazu finde ich nichts.",
                    "url_token": token,
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
    return tmp_path
