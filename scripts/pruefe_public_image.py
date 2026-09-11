"""Build-Gate fuer das oeffentliche Container-Image.

Geprueft wird, **was im Image landet** - nicht, was im Build-Kontext liegt. Eine
Kontextpruefung braeche den Bau ab, sobald ein Mandant mit echten Inhalten lokal
liegt, und das ist der Regelfall dieses Arbeitsbaums. Der volle Kontext dient
ausschliesslich dazu, die Allowlist zu BERECHNEN.

Danach wird Mengengleichheit geprueft, in beide Richtungen. Beide Richtungen
fangen einen anderen Fehler, und keine ersetzt die andere:

1. **Kopiert, aber nicht auf der Allowlist.** Ein Mandant mit echten Inhalten ist
   in ein oeffentliches Image geraten. Das ist der Fall, den das Sichtbarkeitsfeld
   in der Mandantenkonfiguration strukturell verhindern soll.

2. **Auf der Allowlist, aber nicht kopiert.** Das Wurzelverzeichnis zeigt ins
   Leere. Ohne diese Richtung startet die Anwendung sauber, findet keinen
   einzigen Mandanten, und jede Anfrage endet in 404 - ohne Hinweis worauf. Der
   Fehler ist still und vollstaendig; genau deshalb wird er hier laut gemacht.

3. **Kopiert, aber ohne Freigabe in der eigenen Konfiguration.** Diese Pruefung
   liest die Flags im Image noch einmal, unabhaengig von der aufgezeichneten
   Liste. Sie traegt auch dann, wenn die Allowlist-Datei selbst falsch ist.

Dieselbe Pruefung laeuft im Build, von aussen gegen das fertige Image und gegen
den laufenden Container. Deshalb liegt sie hier und nicht im Dockerfile: Was nur
im Dockerfile steht, ist im Betrieb nicht mehr nachvollziehbar - und nicht
testbar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.tenants import list_tenants, tenants_for_public_image

# Pfade im Image. Sie sind hier Voreinstellung, damit die Pruefung von aussen
# ohne Argumente laeuft:
#
#     docker run --rm IMAGE python scripts/pruefe_public_image.py --pruefen
#
# Das Dockerfile setzt dieselben Pfade. Die Kopplung ist beabsichtigt und steht
# an beiden Stellen im Kommentar - eine stille Kopplung waere das Problem, eine
# benannte ist es nicht.
IMAGE_TENANTS_DIR = Path("/app/tenants")
IMAGE_ALLOWLIST = Path("/app/erlaubte-mandanten.txt")


class PruefungFehlgeschlagen(RuntimeError):
    """Das Image entspricht nicht der Allowlist. Traegt die Abweichungen mit."""

    def __init__(self, abweichungen: list[str]) -> None:
        super().__init__("\n".join(abweichungen))
        self.abweichungen = abweichungen


def allowlist_berechnen(tenants_dir: Path) -> list[str]:
    """Slugs, die in ein oeffentliches Image duerfen, aus dem vollen Kontext.

    Reine Funktion ueber dem Mandantenmodell. Ob eine leere Liste ein Fehler ist,
    entscheidet der Aufrufer und nicht diese Funktion - im Build ist sie einer,
    in einem Test ueber einem Verzeichnis ohne freigegebene Mandanten nicht.
    """
    return tenants_for_public_image(tenants_dir)


def allowlist_schreiben(tenants_dir: Path, ziel: Path) -> list[str]:
    """Berechnet die Allowlist und legt sie als Datei ab."""
    slugs = allowlist_berechnen(tenants_dir)
    ziel.parent.mkdir(parents=True, exist_ok=True)
    ziel.write_text("".join(f"{slug}\n" for slug in slugs), encoding="utf-8")
    return slugs


def allowlist_lesen(pfad: Path) -> list[str]:
    """Liest eine abgelegte Allowlist. Leerzeilen werden verworfen."""
    if not pfad.is_file():
        raise PruefungFehlgeschlagen(
            [
                f"Allowlist {pfad} fehlt. Ohne sie ist nicht pruefbar, welche "
                f"Mandanten im Image stehen duerfen - und ein nicht pruefbares "
                f"Image gilt als nicht freigegeben."
            ]
        )
    return [
        zeile.strip() for zeile in pfad.read_text(encoding="utf-8").splitlines() if zeile.strip()
    ]


def abweichungen(tenants_dir: Path, allowlist: list[str]) -> list[str]:
    """Vergleicht das Wurzelverzeichnis mit der Allowlist, in beide Richtungen.

    Leere Rueckgabe heisst: keine Abweichung. Jeder Eintrag benennt genau eine.
    """
    vorhanden = set(list_tenants(tenants_dir))
    erlaubt = set(allowlist)
    mit_freigabe = set(tenants_for_public_image(tenants_dir))

    befunde: list[str] = []

    for slug in sorted(vorhanden - erlaubt):
        befunde.append(
            f"{slug}: liegt im Image, steht aber nicht auf der Allowlist. "
            f"Ein Mandant mit echten Inhalten gehoert nicht in ein oeffentliches Image."
        )

    for slug in sorted(erlaubt - vorhanden):
        befunde.append(
            f"{slug}: steht auf der Allowlist, liegt aber nicht unter {tenants_dir}. "
            f"Entweder wurde er nicht kopiert, oder das Wurzelverzeichnis zeigt "
            f"woanders hin. Beides laesst die Anwendung sauber starten und keinen "
            f"Mandanten finden."
        )

    for slug in sorted(vorhanden - mit_freigabe):
        befunde.append(
            f"{slug}: liegt im Image, seine Konfiguration gibt ihn aber nicht fuer "
            f"ein oeffentliches Image frei."
        )

    return befunde


def pruefen(tenants_dir: Path, allowlist: list[str]) -> None:
    """Wirft PruefungFehlgeschlagen, wenn das Image nicht der Allowlist entspricht."""
    if not allowlist:
        raise PruefungFehlgeschlagen(
            [
                "Die Allowlist ist leer. Ein Image ohne freigegebenen Mandanten ist "
                "keine Demo, sondern ein stiller Fehlschlag: Die Anwendung startet, "
                "und jede Anfrage endet in 404."
            ]
        )
    befunde = abweichungen(tenants_dir, allowlist)
    if befunde:
        raise PruefungFehlgeschlagen(befunde)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python scripts/pruefe_public_image.py",
        description="Prueft, welche Mandanten in einem oeffentlichen Image liegen duerfen.",
    )
    modus = p.add_mutually_exclusive_group(required=True)
    modus.add_argument(
        "--allowlist-schreiben",
        metavar="PFAD",
        type=Path,
        help="berechnet die Allowlist aus --tenants-dir und legt sie unter PFAD ab",
    )
    modus.add_argument(
        "--pruefen",
        action="store_true",
        help="vergleicht --tenants-dir mit --allowlist, in beide Richtungen",
    )
    p.add_argument(
        "--tenants-dir",
        type=Path,
        default=IMAGE_TENANTS_DIR,
        help=f"Wurzelverzeichnis der Mandanten (Vorgabe: {IMAGE_TENANTS_DIR})",
    )
    p.add_argument(
        "--allowlist",
        type=Path,
        default=IMAGE_ALLOWLIST,
        help=f"abgelegte Allowlist (Vorgabe: {IMAGE_ALLOWLIST})",
    )
    args = p.parse_args(argv)

    if args.allowlist_schreiben is not None:
        slugs = allowlist_schreiben(args.tenants_dir, args.allowlist_schreiben)
        if not slugs:
            print(
                f"Kein Mandant unter {args.tenants_dir} ist fuer ein oeffentliches "
                f"Image freigegeben. Ein Image ohne Mandanten waere keine Demo.",
                file=sys.stderr,
            )
            return 1
        print(f"Allowlist nach {args.allowlist_schreiben}: {', '.join(slugs)}")
        return 0

    try:
        pruefen(args.tenants_dir, allowlist_lesen(args.allowlist))
    except PruefungFehlgeschlagen as fehler:
        print("Das Image entspricht nicht der Allowlist:", file=sys.stderr)
        for befund in fehler.abweichungen:
            print(f"  - {befund}", file=sys.stderr)
        return 1

    geprueft = ", ".join(sorted(list_tenants(args.tenants_dir)))
    print(f"Allowlist und Image stimmen ueberein: {geprueft}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
