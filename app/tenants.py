"""Mandantenmodell.

Der `slug` ist intern: Verzeichnisnamen, Telemetrie, Logs. Oeffentlich adressiert
wird ein Mandant ausschliesslich ueber `url_token` (ADR-007).

Alle Funktionen nehmen `tenant_id` bzw. `url_token` als Pflichtparameter an erster
Position. Es gibt keinen Modul-Level-Zustand ueber Mandanten (ADR-001).
"""

from __future__ import annotations

import re
import secrets
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.config import get_settings

# Kein Punkt, kein Schraegstrich, keine Grossbuchstaben. Damit ist "../" nicht
# darstellbar - Traversal scheitert bereits an der Form, nicht erst am Pfad.
SLUG_PATTERN = re.compile(r"^[a-z0-9-]{3,40}$")

TENANT_FILE = "tenant.yaml"

# Ein url_token unter dieser Laenge ist ratbar. Siehe ADR-007: Der Token ist
# Zugangsschutz durch Unratbarkeit, und Unratbarkeit ist eine Laengenfrage.
MIN_TOKEN_LENGTH = 16


class InvalidTenantSlug(ValueError):
    """Der Slug verletzt SLUG_PATTERN. Auch der Traversal-Fall landet hier."""


class TenantNotFound(LookupError):
    """Kein Mandant unter diesem Slug bzw. zu diesem url_token."""


class AmbiguousUrlToken(LookupError):
    """Mehr als ein Mandant traegt dasselbe url_token.

    Das ist kein Randfall, sondern der naheliegendste Fehler beim Anlegen eines
    Mandanten: eine kopierte tenant.yaml, in der der Token stehen geblieben ist.
    Wuerde hier der erste Treffer zurueckgegeben, bekaeme ein Interessent den
    Mandanten eines anderen zu sehen - ohne Fehlermeldung, ohne Absturz. Dieselbe
    Klasse stiller Fehler wie die Score-Umkehr aus ADR-008.
    """


class TenantConfig(BaseModel):
    """Konfiguration eines Mandanten, gelesen aus tenants/<slug>/tenant.yaml."""

    slug: str
    display_name: str
    languages: list[str]
    system_prompt_extra: str = ""
    escalation_message: str
    model_override: str | None = None
    retrieval_top_k: int | None = None

    # Oeffentliche Adresse des Mandanten (ADR-007). Erscheint nie im Log; dort
    # steht der slug (ADR-002).
    url_token: str = Field(min_length=MIN_TOKEN_LENGTH)

    # ADR-014: Nur Mandanten mit erfundenen Inhalten duerfen in ein oeffentliches
    # Container-Image. Default False, weil ein vergessenes Flag in die sichere
    # Richtung wirken muss. Ausgewertet wird es im Docker-Build in Phase 7 - ueber
    # tenants_for_public_image(), damit die Pruefung nicht im Dockerfile
    # nachgebaut wird.
    public_image_allowed: bool = False


def generate_url_token() -> str:
    """Erzeugt ein neues url_token.

    `secrets.token_urlsafe(24)` liefert 32 Zeichen aus 192 Bit Entropie. Nicht
    `random` - das ist vorhersagbar, und der Token ist ein Zugangsmerkmal.
    """
    return secrets.token_urlsafe(24)


def _validate_slug(tenant_id: str) -> str:
    """Prueft den Slug gegen SLUG_PATTERN und gibt ihn unveraendert zurueck."""
    if not SLUG_PATTERN.match(tenant_id):
        raise InvalidTenantSlug(
            f"Ungueltiger Mandanten-Slug: {tenant_id!r}. "
            f"Erlaubt ist {SLUG_PATTERN.pattern} (Kleinbuchstaben, Ziffern, Bindestrich)."
        )
    return tenant_id


def _resolve_tenants_dir(tenants_dir: Path | None) -> Path:
    """Loest das Wurzelverzeichnis auf, ohne es auf Modulebene festzuhalten.

    Ist `tenants_dir` None, kommt der Wert aus den Settings - aber erst hier, beim
    Aufruf. Ein Modul-Level-`get_settings()` wuerde jeden Import dieser Datei an
    eine vollstaendige Konfiguration binden, auch in Tests, die nur das
    Mandantenmodell pruefen.
    """
    if tenants_dir is not None:
        return Path(tenants_dir)
    return Path(get_settings().tenants_dir)


def _tenant_path(tenant_id: str, tenants_dir: Path) -> Path:
    """Liefert den Pfad zur tenant.yaml eines Mandanten.

    Zwei Ebenen Schutz gegen Traversal, weil in diesem Projekt bisher keine
    einzelne Regel gereicht hat:

    1. `_validate_slug` laesst weder Punkt noch Schraegstrich zu.
    2. Der aufgeloeste Pfad muss unterhalb des Wurzelverzeichnisses liegen. Das
       faengt zusaetzlich den Fall ab, dass ein Mandantenverzeichnis ein Symlink
       nach draussen ist - was Ebene 1 nicht sehen kann.
    """
    root = tenants_dir.resolve()
    candidate = (root / tenant_id / TENANT_FILE).resolve()
    if not candidate.is_relative_to(root):
        raise InvalidTenantSlug(
            f"Mandantenpfad fuer {tenant_id!r} zeigt aus {root} heraus und wird abgewiesen."
        )
    return candidate


def load_tenant(tenant_id: str, tenants_dir: Path | None = None) -> TenantConfig:
    """Laedt die Konfiguration eines Mandanten.

    `tenant_id` ist Pflichtparameter an erster Position, ohne Default und ohne
    Optional (ADR-001). `tenants_dir` ist das Wurzelverzeichnis und hat keinen
    Mandantenbezug; es ist injizierbar, damit Tests ohne Settings auskommen.
    """
    _validate_slug(tenant_id)
    root = _resolve_tenants_dir(tenants_dir)
    path = _tenant_path(tenant_id, root)

    if not path.is_file():
        raise TenantNotFound(f"Kein Mandant unter dem Slug {tenant_id!r}.")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise TenantNotFound(f"tenant.yaml von {tenant_id!r} enthaelt kein Mapping.")

    # Der Slug kommt aus dem Verzeichnisnamen, nicht aus der Datei. Sonst koennten
    # Verzeichnis und Inhalt auseinanderlaufen, und ein Mandant traege den Namen
    # eines anderen.
    raw["slug"] = tenant_id
    return TenantConfig(**raw)


def list_tenants(tenants_dir: Path | None = None) -> list[str]:
    """Listet die Slugs aller Mandanten mit einer tenant.yaml, sortiert."""
    root = _resolve_tenants_dir(tenants_dir)
    if not root.is_dir():
        return []

    slugs: list[str] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        if not SLUG_PATTERN.match(entry.name):
            continue
        if (entry / TENANT_FILE).is_file():
            slugs.append(entry.name)
    return sorted(slugs)


def resolve_token(url_token: str, tenants_dir: Path | None = None) -> TenantConfig:
    """Loest ein oeffentliches url_token auf den zugehoerigen Mandanten auf.

    Prueft ALLE Mandanten und bricht bei einer Kollision ab, statt den ersten
    Treffer zu liefern - siehe AmbiguousUrlToken.

    Bekannte Eigenschaft: Jeder Aufruf liest alle Mandantenverzeichnisse. Bei
    einer Handvoll Mandanten ist das unkritisch. Ein Cache waere nach ADR-001 nur
    mit tenant_id im Schluessel zulaessig und wuerde das Kollisionsproblem
    ohnehin nicht loesen.
    """
    root = _resolve_tenants_dir(tenants_dir)

    treffer: list[TenantConfig] = []
    for slug in list_tenants(root):
        tenant = load_tenant(slug, root)
        # compare_digest statt ==, weil der Token ein Zugangsmerkmal ist.
        if secrets.compare_digest(tenant.url_token, url_token):
            treffer.append(tenant)

    if len(treffer) > 1:
        kollidierend = ", ".join(sorted(t.slug for t in treffer))
        # Der Token steht bewusst NICHT in der Meldung (ADR-002), die Slugs schon.
        raise AmbiguousUrlToken(
            f"Mehrere Mandanten teilen dasselbe url_token: {kollidierend}. "
            f"Vermutlich eine kopierte {TENANT_FILE}. Jeder Mandant braucht ein eigenes Token."
        )
    if not treffer:
        raise TenantNotFound("Kein Mandant zu diesem url_token.")
    return treffer[0]


def tenants_for_public_image(tenants_dir: Path | None = None) -> list[str]:
    """Slugs der Mandanten, die in ein oeffentliches Container-Image duerfen.

    Nach ADR-014 nur Mandanten mit erfundenen Inhalten. Der Docker-Build in
    Phase 7 wertet diese Liste aus und bricht ab, wenn ein Mandant ohne Flag in
    ein oeffentliches Ziel geraet. Die Pruefung gehoert hierher und nicht ins
    Dockerfile, damit sie testbar ist.
    """
    root = _resolve_tenants_dir(tenants_dir)
    return sorted(
        slug for slug in list_tenants(root) if load_tenant(slug, root).public_image_allowed
    )
