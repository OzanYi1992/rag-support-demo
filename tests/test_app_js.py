"""Statische Pruefungen an static/app.js, dazu eine an static/style.css.

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
STYLE_CSS = REPO_ROOT / "static" / "style.css"

# Jede dieser Schnittstellen macht aus einer Zeichenkette Markup oder laedt
# Markup in einen Rahmen. Dazu jeder Speicher im Browser: Der Verlauf eines
# Gespraechs liegt nur in einer Variablen der Seite und darf das Neuladen nicht
# ueberleben.
VERBOTEN = (
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "document.write",
    "srcdoc",
    "localStorage",
    "sessionStorage",
    "indexedDB",
    "document.cookie",
)


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
    assert _verbotene('x = localStorage.getItem("v");') == ["localStorage"]
    assert _verbotene("// localStorage") == []
    assert _verbotene("document.cookie = 'a=b';") == ["document.cookie"]

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


def test_app_js_zeigt_bei_503_den_eigenen_text() -> None:
    """Bei 503 der Text fuer einen Ausfall, jeder andere Fehler wie bisher.

    Die Pruefung auf 503 steht vor der allgemeinen auf antwort.ok. Stuende sie
    dahinter, liefe ein 503 in anfrage_fehlgeschlagen und der eigene Text kaeme
    nie an. Ein 502 hat keinen eigenen Zweig und landet bei
    anfrage_fehlgeschlagen.
    """
    code = _ohne_kommentare(APP_JS.read_text(encoding="utf-8"))
    assert "antwort.status === 503" in code
    assert "texte.anbieter_nicht_erreichbar" in code
    assert code.index("antwort.status === 503") < code.index("!antwort.ok")
    assert "502" not in code
    assert "throw new Error(texte.anfrage_fehlgeschlagen)" in code


def test_app_js_schickt_hoechstens_drei_runden_und_nur_runden_ueber_die_unterlagen() -> None:
    """Der Verlauf geht mit der Frage hinaus, hoechstens drei Runden. Hinein
    kommt nur eine Antwort der Art "rag", und eine eskalierte mit leerer
    Antwort. Die Antwort ist der ganze Text, nicht die gekuerzte Anzeige, auf
    2 000 Zeichen begrenzt. Statisch geprueft; das Verhalten zeigt der Browser.
    """
    code = _ohne_kommentare(APP_JS.read_text(encoding="utf-8"))
    assert "var MAX_RUNDEN = 3;" in code
    assert "var MAX_ANTWORT = 2000;" in code
    assert "gespraech.slice(-MAX_RUNDEN)" in code
    assert "body: JSON.stringify({ question: frage, history: runden })" in code
    assert 'if (daten.kind === "rag")' in code
    assert 'answer: daten.escalated ? "" : fuerDenVerlauf(daten.text)' in code
    # Die Runde kommt erst nach der Antwort hinein, im Zweig fuer Status 200.
    assert code.index("return antwort.json()") < code.index("gespraech.push(")
    assert code.index("gespraech.push(") < code.index(".catch(")


def test_app_js_zeigt_verstanden_als_ueber_textcontent() -> None:
    """Die eigenstaendige Frage kommt vom Modell und geht nur ueber textContent
    in die Seite, mit einer Ersetzungsfunktion statt einer Zeichenkette."""
    code = _ohne_kommentare(APP_JS.read_text(encoding="utf-8"))
    assert 'texte.verstanden_als.replace("{frage}", function () {' in code
    assert "zeile.textContent = texte.verstanden_als" in code


def test_neues_gespraech_leert_ohne_netz() -> None:
    """Der Knopf leert den Verlauf und entfernt alle Blasen bis auf die erste,
    die Begruessung. Im Zweig steht kein Netzaufruf und kein Neuladen."""
    code = _ohne_kommentare(APP_JS.read_text(encoding="utf-8"))
    anfang = code.index('neu.addEventListener("click"')
    zweig = code[anfang : code.index("});", anfang)]
    assert "gespraech = [];" in zweig
    assert "verlauf.children.length > 1" in zweig
    assert "fetch(" not in zweig and "location" not in zweig


def _css_regel(quelle: str, auswahl: str) -> str:
    """Der Rumpf der ersten Regel mit genau diesem Selektor, ohne Kommentare."""
    ohne = re.sub(r"/\*.*?\*/", "", quelle, flags=re.S)
    treffer = re.search(rf"(?m)^{re.escape(auswahl)}\s*\{{([^}}]*)\}}", ohne)
    assert treffer, f"keine Regel {auswahl!r}"
    return treffer.group(1)


def test_die_kopfzeile_bleibt_beim_rollen_stehen() -> None:
    """Titel und "Neues Gespraech" bleiben oben sichtbar, mit dem Hintergrund
    der Seite aus dem Farbtoken, hell wie dunkel. Statisch geprueft; Hoehe und
    Klickbarkeit nach dem Rollen zeigt der Browser.

    Gegenprobe: Die Suche findet die Regel auch ohne sticky, sie ist also nicht
    blind.
    """
    regel = _css_regel(STYLE_CSS.read_text(encoding="utf-8"), "header")
    assert "position: sticky;" in regel
    assert "top: 0;" in regel
    assert "background: var(--grund);" in regel
    assert "padding: 1.25rem 0 1rem;" in regel
    assert "position: sticky" not in _css_regel("header {\n  padding: 0;\n}", "header")
