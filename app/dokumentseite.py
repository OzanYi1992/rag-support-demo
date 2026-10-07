"""Quelldokumente als eigenstaendige HTML-Seite.

Aus dem Markdown eines Quelldokuments wird hier eine vollstaendige Seite. Die
Route in app/main.py liefert sie aus, und die Chatseite zeigt sie in einem
Rahmen mit leerem sandbox-Attribut. Das HTML eines Dokuments wird nie Teil der
Chatseite.

Dokumentinhalt ist Fremdtext. Was hier entsteht, ist deshalb auf drei Ebenen
abgesichert, und jede traegt fuer sich:

1. Der Renderer maskiert. HTML im Dokument wird Text, Bilder, Links und
   Linkdefinitionen bleiben Text. Siehe `_parser()`.
2. Die Antwort traegt `sandbox` und `default-src 'none'` in der
   Content-Security-Policy. Damit laeuft kein Skript und nichts wird
   nachgeladen, auch wenn jemand die Adresse direkt oeffnet statt im Rahmen.
   Siehe `INHALTSRICHTLINIE`.
3. Der Rahmen in der Chatseite traegt ein leeres sandbox-Attribut.

Fiele die erste Ebene aus, etwa weil jemand HTML im Renderer einschaltet,
liefe ein eingeschleustes <script> trotzdem nicht.

Einen Mandanten loest dieses Modul nicht auf. Es bekommt Text, der bereits
hinter der Mandantengrenze gelesen wurde, und haelt nichts davon fest: kein
Parser, kein Zwischenspeicher, kein Dokument auf Modulebene. Die Konstanten
hier sind das CSS, sein Hash und die Richtlinie.
"""

from __future__ import annotations

import base64
import hashlib
import html

from markdown_it import MarkdownIt
from markdown_it.rules_core import StateCore
from markdown_it.token import Token

# Das CSS der Dokumentseite. Es steht als <style>-Block in der Seite und ist
# ueber seinen Hash in der Richtlinie erlaubt. Ein Stylesheet unter /static/
# braeuchte 'self' in der Richtlinie, und ob 'self' bei einer Seite mit
# undurchsichtigem Ursprung greift, ist nicht belegt.
#
# Die Farben sind die Palette aus static/style.css, hell und dunkel, damit das
# Dokument im Rahmen aussieht wie der Chat darum herum. Sie stehen damit an zwei
# Stellen, und ein Test haelt beide gleich.
#
# Breite Tabellen rollen fuer sich seitlich, nicht die ganze Seite. Auf dem
# Telefon ist der Rahmen gut 300 Pixel breit, und eine Tabelle mit fuenf
# Spalten passt dort nicht hinein.
_STIL = """\
:root {
  color-scheme: light dark;
  --grund: #ffffff;
  --grund-gedaempft: #f4f5f7;
  --text: #1b1d21;
  --text-leise: #5c6370;
  --linie: #dfe2e7;
}

@media (prefers-color-scheme: dark) {
  :root {
    --grund: #16181c;
    --grund-gedaempft: #202329;
    --text: #e8eaed;
    --text-leise: #9aa2ae;
    --linie: #313640;
  }
}

* { box-sizing: border-box; }

body {
  margin: 0;
  padding: .75rem 1rem 1.25rem;
  background: var(--grund);
  color: var(--text);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  overflow-wrap: break-word;
}

h1, h2, h3, h4, h5, h6 { margin: 1.1rem 0 .5rem; line-height: 1.3; }
h1 { margin-top: .25rem; font-size: 1.2rem; }
h2 { font-size: 1.05rem; }
h3, h4, h5, h6 { font-size: 1rem; }

p, ul, ol, table, pre, blockquote { margin: 0 0 .75rem; }
ul, ol { padding-left: 1.25rem; }
li { margin: .2rem 0; }

table {
  display: block;
  max-width: 100%;
  overflow-x: auto;
  border-collapse: collapse;
  font-size: .9rem;
}
th, td {
  padding: .35rem .55rem;
  border: 1px solid var(--linie);
  text-align: left;
  vertical-align: top;
}
th { background: var(--grund-gedaempft); font-weight: 600; }

code {
  padding: 0 .2em;
  border-radius: .2rem;
  background: var(--grund-gedaempft);
  font: .9em ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
pre {
  padding: .6rem .75rem;
  overflow-x: auto;
  border-radius: .4rem;
  background: var(--grund-gedaempft);
}
pre code { padding: 0; background: none; }

blockquote {
  padding-left: .75rem;
  border-left: 3px solid var(--linie);
  color: var(--text-leise);
}
hr { border: 0; border-top: 1px solid var(--linie); }
"""

# Der Hash steht in der Richtlinie. Er gilt fuer genau diesen Text zwischen
# <style> und </style>, Zeichen fuer Zeichen. Wer das CSS aendert, aendert den
# Hash mit, und zwar von selbst, weil er beim Import berechnet wird.
STIL_HASH = "sha256-" + base64.b64encode(hashlib.sha256(_STIL.encode("utf-8")).digest()).decode(
    "ascii"
)

# Die Content-Security-Policy der Dokumentantwort, je Direktive:
#
# sandbox             Keine Skripte, keine Popups, kein Formularversand und ein
#                     eigener undurchsichtiger Ursprung. Das gilt fuer die
#                     Antwort selbst, also auch, wenn jemand die Adresse direkt
#                     oeffnet statt im Rahmen. Wirkt nur als Kopfzeile.
# default-src 'none'  Kein Skript, kein Bild, keine Schrift, kein Rahmen, keine
#                     Verbindung. Nichts von fremden Hosts, selbst wenn der
#                     Renderer ein <img> durchliesse.
# style-src <hash>    Nur der eigene <style>-Block. style-Attribute bleiben
#                     gesperrt; ausgerichtete Tabellenspalten erscheinen
#                     deshalb ohne Ausrichtung.
# frame-ancestors     Nur Seiten des eigenen Ursprungs duerfen die Seite
#   'self'            einbetten. Wirkt nur als Kopfzeile.
# base-uri 'none'     Kein <base>, das Adressen umlenkt. default-src deckt das
#                     nicht ab.
INHALTSRICHTLINIE = "; ".join(
    (
        "sandbox",
        "default-src 'none'",
        f"style-src '{STIL_HASH}'",
        "frame-ancestors 'self'",
        "base-uri 'none'",
    )
)


def _naechstes_nichtleeres(kinder: list[Token], ab: int) -> int | None:
    """Index des naechsten Kindes ab `ab`, das kein leerer Text ist."""
    for i in range(ab, len(kinder)):
        if not (kinder[i].type == "text" and kinder[i].content == ""):
            return i
    return None


def _fette_erste_zeile(state: StateCore) -> None:
    """Macht in der FAQ aus Frage und Antwort zwei Zeilen.

    Die FAQ steht im Korpus in der Form

        **Frage?**
        Antwort ...

    Im Markdown ist der Zeilenumbruch dazwischen ein weicher Umbruch, und der
    erscheint als Leerzeichen: Frage und Antwort laufen in einer Zeile
    zusammen. Diese Regel macht genau diesen einen Umbruch zu einem harten,
    also zu <br>: den weichen Umbruch direkt nach einem Fettdruck, der einen
    Absatz eroeffnet. Alle anderen weichen Umbrueche bleiben, wie sie sind. Der
    Korpus ist bei rund 80 Zeichen hart umbrochen, und jeden Umbruch zu einem
    harten zu machen, zerhackte jeden Absatz.

    Die Regel greift in jedem Absatz, auch im Absatz eines Listenpunkts. Ein
    `- **Begriff**` mit eingeruecktem Text in der naechsten Zeile bekommt also
    ebenfalls ein <br>.

    Vorsicht bei Aenderungen: markdown-it legt vor einem eroeffnenden Fettdruck
    und hinter seinem Ende LEERE Text-Token an. Eine Pruefung auf das erste
    oder das naechste Kind trifft deshalb nie. Die Suche unten ueberspringt
    sie, und ein Test laeuft gegen die echte FAQ, weil ein Beispiel im Test
    diese Eigenheit nicht zeigen muss.
    """
    tokens = state.tokens
    for i, block in enumerate(tokens):
        if block.type != "inline" or i == 0 or tokens[i - 1].type != "paragraph_open":
            continue
        kinder = block.children or []
        anfang = _naechstes_nichtleeres(kinder, 0)
        if anfang is None or kinder[anfang].type != "strong_open":
            continue

        tiefe = 0
        ende: int | None = None
        for j in range(anfang, len(kinder)):
            if kinder[j].type == "strong_open":
                tiefe += 1
            elif kinder[j].type == "strong_close":
                tiefe -= 1
                if tiefe == 0:
                    ende = j
                    break
        if ende is None:
            continue

        danach = _naechstes_nichtleeres(kinder, ende + 1)
        if danach is not None and kinder[danach].type == "softbreak":
            kinder[danach].type = "hardbreak"


def _parser() -> MarkdownIt:
    """Der Renderer, je Aufruf neu gebaut.

    `js-default` ist die Voreinstellung, die markdown-it-py fuer Fremdtext
    empfiehlt: HTML im Dokument wird nicht ausgewertet (`html: False`), dazu
    sind linkify und typographer aus. Zusaetzlich abgeschaltet:

    * `image`, weil ein Bild von einer fremden Adresse nachgeladen wuerde.
    * `link` und `autolink`, weil ein Link aus dem Rahmen herausfuehrt. Ob
      Kundendokumente Links bekommen, ist nicht entschieden.
    * `reference`, weil eine Definition wie `[r]: https://...` sonst still aus
      der Ausgabe verschwindet, wenn `link` aus ist. Ohne die Regel bleibt sie
      als Text stehen. Was im Dokument steht, steht dann auch in der Anzeige.

    `html_block` und `html_inline` bleiben bewusst aktiv. Solange `html` aus
    ist, werten sie nichts aus. Der eine Schalter ist `html`, und ein Test
    faengt es, wenn er umgelegt wird.

    Kein Parser auf Modulebene: Zur Threadsicherheit sagt die Bibliothek
    nichts, FastAPI fuehrt die Route im Threadpool aus, und der Bau ist billig,
    gemessen am 2026-10-06 rund 0,09 ms.
    """
    md = MarkdownIt("js-default").disable(["image", "link", "autolink", "reference"])
    md.core.ruler.after("inline", "fette_erste_zeile", _fette_erste_zeile)
    return md


def markdown_rendern(tenant_id: str, text: str) -> str:
    """Rendert das Markdown eines Quelldokuments zu einem HTML-Fragment.

    `tenant_id` steht an erster Stelle, obwohl hier kein Mandant aufgeloest
    wird. Die Funktion bekommt Dokumentinhalt, und jede Funktion, die
    Dokumente beruehrt, nimmt den Mandanten als Pflichtparameter, auch wenn
    sie ihn nur traegt.
    """
    return _parser().render(text)


def dokumentseite_bauen(tenant_id: str, dateiname: str, sprache: str, text: str) -> str:
    """Baut die ganze Seite eines Quelldokuments.

    Die Seite entsteht durch Verketten fester Bruchstuecke, und der Inhalt
    kommt zuletzt. Ueber Inhalt und Dateinamen laeuft KEIN Ersetzen. Die
    Oberflaeche entsteht aus einer Vorlage mit Platzhaltern; dieselbe Technik
    hier liesse ein `{{titel}}` im Dokument von einem spaeteren Schritt ersetzen.

    Der Dateiname steht maskiert im <title>. Eine Ueberschrift mit dem Namen
    gibt es nicht: Jedes Dokument beginnt mit einer eigenen, und der Knopf
    ueber dem Rahmen traegt den Namen schon.

    Kein <script>, kein <link>, kein src, kein href, keine fremde Adresse. Die
    Seite traegt auch keinen Katalogtext: Sie zeigt ein Dokument, nicht die
    Oberflaeche.
    """
    fragment = markdown_rendern(tenant_id, text)
    return (
        "<!doctype html>\n"
        f'<html lang="{html.escape(sprache)}">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="robots" content="noindex, nofollow">\n'
        f"<title>{html.escape(dateiname)}</title>\n"
        f"<style>{_STIL}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{fragment}"
        "</body>\n"
        "</html>\n"
    )
