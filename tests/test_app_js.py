"""Statische Pruefungen an static/app.js.

Es gibt kein JavaScript-Werkzeug in diesem Projekt, und es soll auch keins
dazukommen. Was sich am Quelltext pruefen laesst, wird deshalb hier geprueft:
dass keine Schnittstelle vorkommt, ueber die Text zu Markup wird, und dass der
Rahmen fuer Quelldokumente seine Sandbox ohne Ausnahme bekommt.

Gesucht wird im Code, nicht in den Kommentaren. Der Kopfkommentar nennt
innerHTML ausdruecklich, als das, was nicht benutzt wird.
"""

from __future__ import annotations

import re

from tests.conftest import REPO_ROOT

APP_JS = REPO_ROOT / "static" / "app.js"

# Jede dieser Schnittstellen macht aus einer Zeichenkette Markup oder laedt
# Markup in einen Rahmen.
VERBOTEN = ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "srcdoc")


def _ohne_kommentare(quelle: str) -> str:
    """Entfernt Block- und Zeilenkommentare.

    Bewusst einfach. Das traegt, solange keine Zeichenkette in app.js `//`
    oder `/*` enthaelt; `test_die_kommentarentfernung_hat_ihre_voraussetzung`
    prueft das.
    """
    ohne_bloecke = re.sub(r"/\*.*?\*/", "", quelle, flags=re.S)
    return re.sub(r"(^|\s)//.*$", r"\1", ohne_bloecke, flags=re.M)


def _verbotene(quelle: str) -> list[str]:
    code = _ohne_kommentare(quelle)
    return [name for name in VERBOTEN if name in code]


def test_app_js_nutzt_keine_verbotene_schnittstelle() -> None:
    """Kein innerHTML, outerHTML, insertAdjacentHTML, document.write, srcdoc.

    Die Gegenproben stehen im selben Test: Die Rohdatei nennt innerHTML im
    Kommentar, die Suche entfernt Kommentare also wirklich. Und dieselbe
    Pruefung schlaegt an echtem Code an, an einem Kommentar aber nicht.
    """
    roh = APP_JS.read_text(encoding="utf-8")
    assert "innerHTML" in roh
    assert _verbotene("x.innerHTML = y;") == ["innerHTML"]
    assert _verbotene("// innerHTML") == []
    assert _verbotene("/* innerHTML */") == []

    assert _verbotene(roh) == []


def test_app_js_setzt_ein_leeres_sandbox_attribut() -> None:
    """Der Rahmen bekommt sandbox ohne jede Ausnahme, und zwar vor der Adresse.

    Faengt eine aufgeweichte Sandbox (ein allow-Token), eine fehlende und eine,
    die erst nach der Adresse gesetzt wird: Die Flags greifen erst, wenn der
    Rahmen navigiert.
    """
    code = _ohne_kommentare(APP_JS.read_text(encoding="utf-8"))
    assert 'setAttribute("sandbox", "")' in code
    assert "allow-" not in code
    assert 'removeAttribute("sandbox")' not in code
    assert code.index('setAttribute("sandbox", "")') < code.index(".src =")


def test_die_kommentarentfernung_hat_ihre_voraussetzung() -> None:
    """Keine Zeichenkette in app.js enthaelt `//` oder `/*`.

    Sonst schnitte die Kommentarentfernung oben echten Code ab, und eine
    verbotene Schnittstelle dahinter bliebe ungesehen.
    """
    roh = APP_JS.read_text(encoding="utf-8")
    zeichenketten = re.findall(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'', roh)
    assert zeichenketten, "keine Zeichenkette gefunden - die Suche waere blind"
    for kette in zeichenketten:
        assert "//" not in kette and "/*" not in kette, kette
