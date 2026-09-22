"""Tests des Build-Gates fuer das oeffentliche Container-Image.

Das Gate prueft, WAS IM IMAGE LANDET, nicht was im Build-Kontext liegt. Der
Unterschied ist keine Feinheit: Eine Kontextpruefung braeche den Bau ab, sobald
ein Mandant mit echten Inhalten lokal liegt - und das ist der Regelfall dieses
Arbeitsbaums. Dieser Fall hat deshalb einen eigenen Test, der GRUEN sein muss.

Nach conventions.md braucht jede Pruefung einen Positivtest, der treffen MUSS,
und eine Gegenprobe. Hier ist beides doppelt vorhanden, weil das Gate zwei
Richtungen hat, die verschiedene Fehler fangen.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.pruefe_public_image import (
    PruefungFehlgeschlagen,
    abweichungen,
    allowlist_berechnen,
    allowlist_lesen,
    allowlist_schreiben,
    main,
    pruefen,
)


def _mandant(root: Path, slug: str, *, freigegeben: bool) -> None:
    """Legt einen Mandanten mit oder ohne Freigabe fuer oeffentliche Images an."""
    verzeichnis = root / slug
    (verzeichnis / "docs").mkdir(parents=True)
    daten: dict[str, object] = {
        "display_name": slug,
        "language": "de",
        "escalation_message": "Dazu finde ich nichts.",
        "url_token": f"{slug}-token-1234567890",
    }
    if freigegeben:
        daten["public_image_allowed"] = True
    (verzeichnis / "tenant.yaml").write_text(
        yaml.safe_dump(daten, allow_unicode=True), encoding="utf-8"
    )
    (verzeichnis / "docs" / "doku.md").write_text("Ein Satz.", encoding="utf-8")


# --- Die Allowlist berechnen ------------------------------------------------


def test_allowlist_ist_leer_ohne_freigabe(tmp_tenants_dir: Path) -> None:
    """Der Mandant aus der Fixture traegt das Flag nicht - also darf er nicht.

    Positivtest fuer die Voreinstellung: Ein vergessenes Flag wirkt in die
    sichere Richtung.
    """
    assert allowlist_berechnen(tmp_tenants_dir) == []


def test_allowlist_enthaelt_alle_demo_mandanten(demo_tenants_dir: Path) -> None:
    """Gegenprobe: Ueber dem echten tenants/ muss die Berechnung treffen.

    Ohne diesen Test waere die leere Liste oben von einem Werkzeugfehler nicht
    zu unterscheiden.

    Die Liste ist fest und nicht "mindestens zwei". Eine Berechnung, die einen
    Mandanten zu viel liefert, ist genauso falsch wie eine, die einen zu wenig
    liefert - nur teurer, weil sie ins Image fuehrt.
    """
    assert allowlist_berechnen(demo_tenants_dir) == ["demo-acme", "demo-fellgate", "demo-nordwind"]


def test_allowlist_schreiben_und_lesen_ergibt_dasselbe(tmp_path: Path) -> None:
    kontext = tmp_path / "kontext"
    kontext.mkdir()
    _mandant(kontext, "demo-eins", freigegeben=True)
    _mandant(kontext, "demo-zwei", freigegeben=True)
    ziel = tmp_path / "erlaubte-mandanten.txt"

    geschrieben = allowlist_schreiben(kontext, ziel)

    assert geschrieben == ["demo-eins", "demo-zwei"]
    assert allowlist_lesen(ziel) == ["demo-eins", "demo-zwei"]


def test_fehlende_allowlist_datei_ist_ein_abbruch(tmp_path: Path) -> None:
    """Eine nicht pruefbare Ablage gilt als nicht freigegeben, nicht als in Ordnung."""
    with pytest.raises(PruefungFehlgeschlagen):
        allowlist_lesen(tmp_path / "gibt-es-nicht.txt")


# --- Der Regelfall, den eine Kontextpruefung zerstoert haette ---------------


def test_fremder_mandant_im_kontext_stoert_den_bau_nicht(tmp_path: Path) -> None:
    """DER Test dieser Datei.

    Im Build-Kontext liegt ein Mandant ohne Freigabe - der Normalzustand,
    sobald ein Interessent angelegt ist. Kopiert wird nur die Allowlist. Das
    Gate muss schweigen.

    Waere das Gate eine Kontextpruefung, schluege dieser Test fehl, und der
    Demo-Bau waere genau dann unmoeglich, wenn das Projekt benutzt wird.
    """
    kontext = tmp_path / "kontext"
    kontext.mkdir()
    _mandant(kontext, "demo-eins", freigegeben=True)
    _mandant(kontext, "echter-interessent", freigegeben=False)

    allowlist = allowlist_berechnen(kontext)
    assert allowlist == ["demo-eins"]

    # Was tatsaechlich ins Image kopiert wird: nur die Allowlist.
    image = tmp_path / "image"
    image.mkdir()
    _mandant(image, "demo-eins", freigegeben=True)

    assert abweichungen(image, allowlist) == []
    pruefen(image, allowlist)  # wirft nicht


# --- Richtung 1: kopiert, aber nicht erlaubt --------------------------------


def test_mandant_zuviel_im_image_bricht_ab(tmp_path: Path) -> None:
    """Ein Mandant mit echten Inhalten ist ins oeffentliche Image geraten."""
    image = tmp_path / "image"
    image.mkdir()
    _mandant(image, "demo-eins", freigegeben=True)
    _mandant(image, "echter-interessent", freigegeben=False)

    befunde = abweichungen(image, ["demo-eins"])

    assert any("echter-interessent" in b for b in befunde)
    with pytest.raises(PruefungFehlgeschlagen):
        pruefen(image, ["demo-eins"])


def test_kopierter_mandant_ohne_freigabe_faellt_auch_ohne_allowlist_auf(
    tmp_path: Path,
) -> None:
    """Dritte Pruefung: die Flags im Image werden unabhaengig nachgelesen.

    Selbst wenn die Allowlist-Datei den Mandanten faelschlich fuehrt, traegt
    diese Richtung noch.
    """
    image = tmp_path / "image"
    image.mkdir()
    _mandant(image, "falsch-gelistet", freigegeben=False)

    befunde = abweichungen(image, ["falsch-gelistet"])

    assert any("gibt ihn aber nicht fuer" in b for b in befunde)


# --- Richtung 2: erlaubt, aber nicht kopiert (die Pfadfalle) ----------------


def test_fehlender_mandant_im_image_bricht_ab(tmp_path: Path) -> None:
    """Das Wurzelverzeichnis zeigt ins Leere.

    Ohne diese Richtung startet die Anwendung sauber und findet keinen
    Mandanten - der stille Fehler, den config.py fuer den Container voraussagt.
    """
    image = tmp_path / "image"
    image.mkdir()
    _mandant(image, "demo-eins", freigegeben=True)

    befunde = abweichungen(image, ["demo-eins", "demo-zwei"])

    assert any("demo-zwei" in b for b in befunde)
    with pytest.raises(PruefungFehlgeschlagen):
        pruefen(image, ["demo-eins", "demo-zwei"])


def test_leeres_wurzelverzeichnis_bricht_ab(tmp_path: Path) -> None:
    """Der Vollfall der Pfadfalle: gar kein Mandant gefunden."""
    leer = tmp_path / "zeigt-ins-leere"
    leer.mkdir()

    with pytest.raises(PruefungFehlgeschlagen):
        pruefen(leer, ["demo-eins", "demo-zwei"])


def test_leere_allowlist_bricht_ab(tmp_path: Path) -> None:
    """Ein Image ohne freigegebenen Mandanten ist keine Demo."""
    image = tmp_path / "image"
    image.mkdir()

    with pytest.raises(PruefungFehlgeschlagen):
        pruefen(image, [])


# --- Die Kommandozeile, wie der Build sie aufruft ---------------------------


def test_cli_schreiben_dann_pruefen_ist_gruen(tmp_path: Path) -> None:
    kontext = tmp_path / "kontext"
    kontext.mkdir()
    _mandant(kontext, "demo-eins", freigegeben=True)
    _mandant(kontext, "echter-interessent", freigegeben=False)
    liste = tmp_path / "erlaubte-mandanten.txt"

    assert (
        main(
            [
                "--allowlist-schreiben",
                str(liste),
                "--tenants-dir",
                str(kontext),
            ]
        )
        == 0
    )

    image = tmp_path / "image"
    image.mkdir()
    _mandant(image, "demo-eins", freigegeben=True)

    assert main(["--pruefen", "--tenants-dir", str(image), "--allowlist", str(liste)]) == 0


def test_cli_meldet_abweichung_mit_exitcode(tmp_path: Path) -> None:
    """Exit-Code != 0, sonst laeuft der Docker-Build ueber den Befund hinweg."""
    liste = tmp_path / "erlaubte-mandanten.txt"
    liste.write_text("demo-eins\n", encoding="utf-8")

    image = tmp_path / "image"
    image.mkdir()
    _mandant(image, "demo-eins", freigegeben=True)
    _mandant(image, "echter-interessent", freigegeben=False)

    assert main(["--pruefen", "--tenants-dir", str(image), "--allowlist", str(liste)]) == 1


def test_cli_schreiben_ohne_freigegebenen_mandanten_meldet_fehler(tmp_path: Path) -> None:
    kontext = tmp_path / "kontext"
    kontext.mkdir()
    _mandant(kontext, "echter-interessent", freigegeben=False)

    assert (
        main(
            [
                "--allowlist-schreiben",
                str(tmp_path / "liste.txt"),
                "--tenants-dir",
                str(kontext),
            ]
        )
        == 1
    )


# --- Freigegeben heisst nichts, wenn die Datei nicht mitkommt --------------


def test_freigegebene_mandanten_sind_von_git_verfolgt(demo_tenants_dir: Path) -> None:
    """Jeder freigegebene Mandant muss auch wirklich im Repo liegen.

    Der Fehler, den dieser Test faengt, ist vollstaendig still. `.gitignore`
    schliesst `tenants/*` aus und nimmt die Demo-Mandanten einzeln wieder auf.
    Wird ein neuer Mandant dort vergessen, passiert LOKAL nichts Auffaelliges:
    Er liegt im Arbeitsbaum, `list_tenants()` findet ihn, das Build-Gate
    berechnet seine Allowlist aus dem Kontext und meldet gruen.

    Erst ein Bau aus einem frischen Checkout haette ihn nicht. Und auch dann
    bleibt es still: Das Gate berechnet die Allowlist wieder aus dem Kontext -
    diesmal ohne ihn - findet Uebereinstimmung und meldet erneut gruen. Das
    Image waere einen Mandanten zu klein, ohne eine einzige Fehlermeldung, und
    der Link darauf liefe ins Leere.

    Das Gate kann diesen Fall nicht sehen. Es vergleicht den Kontext mit sich
    selbst. Nur git weiss, was tatsaechlich mitkommt.
    """
    wurzel = demo_tenants_dir.parent
    if not (wurzel / ".git").exists():
        pytest.skip("Kein git-Arbeitsbaum - im Container gibt es keinen.")

    lauf = subprocess.run(
        ["git", "ls-files", "--", "tenants/"],
        cwd=wurzel,
        capture_output=True,
        text=True,
        check=False,
    )
    if lauf.returncode != 0:
        pytest.skip(f"git ls-files nicht ausfuehrbar: {lauf.stderr.strip()}")

    verfolgt = {zeile.split("/")[1] for zeile in lauf.stdout.splitlines() if "/" in zeile}
    # Gegenprobe: Faende git gar nichts, waere die Pruefung blind und jede
    # Aussage darunter wertlos.
    assert verfolgt, "git ls-files liefert nichts unter tenants/ - der Test waere blind."

    freigegeben = set(allowlist_berechnen(demo_tenants_dir))
    fehlend = sorted(freigegeben - verfolgt)
    assert not fehlend, (
        f"Diese Mandanten sind fuer ein oeffentliches Image freigegeben, werden aber "
        f"nicht von git verfolgt: {fehlend}. Sie fehlen in einem Bau aus einem frischen "
        f"Checkout, und zwar ohne Fehlermeldung. In .gitignore namentlich aufnehmen."
    )
