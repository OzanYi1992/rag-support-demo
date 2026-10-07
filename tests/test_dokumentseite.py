"""Tests der Dokumentseite, ohne HTTP.

Dokumentinhalt ist Fremdtext, und hier wird er zum ersten Mal Markup. Die
Tests pruefen deshalb zweierlei: dass fremdes Markup NICHT als Markup ankommt,
und dass der Renderer trotzdem rendert. Ohne das Zweite waere eine Ausgabe, die
alles verschluckt, von einer sicheren nicht zu unterscheiden. Jeder Test
traegt seine Gegenprobe deshalb selbst.

Wo es auf den echten Korpus ankommt, lesen die Tests ihn, und zwar nur lesend.
Eine Regel, die am Beispiel im Test gruen ist, kann an der echten Datei
wirkungslos sein.
"""

from __future__ import annotations

import re

from markdown_it import MarkdownIt

from app import dokumentseite
from app.dokumentseite import dokumentseite_bauen, markdown_rendern
from tests.conftest import REPO_ROOT

MANDANT = "mandant-a"
KORPUS = REPO_ROOT / "tenants"

# Die Tags, die diese Konfiguration ueberhaupt erzeugen kann. Links, Bilder und
# rohes HTML sind abgeschaltet; was hier fehlt, darf in keiner Ausgabe stehen.
ERLAUBTE_TAGS = frozenset(
    {
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "strong",
        "em",
        "s",
        "code",
        "pre",
        "blockquote",
        "hr",
        "br",
        "table",
        "thead",
        "tbody",
        "tr",
        "th",
        "td",
    }
)

_TAG = re.compile(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)([^>]*)>")


def _gegenprobe_rendert(ausgabe: str) -> None:
    """Fettdruck und Tabelle kommen als Markup an. Sonst ist der Test blind."""
    assert "<strong>fett</strong>" in ausgabe
    assert "<table>" in ausgabe


GEGENPROBE = "**fett**\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"


def test_fremdes_markup_kommt_nicht_als_markup_an() -> None:
    """Eingeschaltetes HTML, eine wieder aktive Bildregel, ein umgangener Link.

    Die Gegenprobe steht vorn und laeuft fuer sich: Schaltet jemand HTML im
    Renderer ein, bleibt sie gruen, und der Test faellt an der Zusicherung zu
    <script>. So zeigt der Fehlschlag auf die Ursache.
    """
    _gegenprobe_rendert(markdown_rendern(MANDANT, GEGENPROBE))

    ausgabe = markdown_rendern(
        MANDANT,
        "<script>alert(1)</script>\n\n"
        '<img src=x onerror="alert(1)">\n\n'
        '<iframe src="https://fremd.example/"></iframe>\n\n'
        "[klick](javascript:alert(1))\n\n"
        "![bild](https://fremd.example/x.png)\n",
    )
    assert "<script" not in ausgabe
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in ausgabe
    assert "<img" not in ausgabe
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in ausgabe
    assert "<iframe" not in ausgabe
    assert "&lt;iframe" in ausgabe
    assert "<a " not in ausgabe
    # Was nicht Markup wird, bleibt sichtbar stehen.
    assert "[klick](javascript:alert(1))" in ausgabe
    assert "![bild](https://fremd.example/x.png)" in ausgabe


def test_fremder_link_wird_kein_a_element() -> None:
    """Direkter Link, Autolink, Mailadresse und Referenzlink: kein <a>.

    Der Linktext bleibt sichtbar. Ein Link fuehrte aus dem Rahmen heraus, und
    ob Kundendokumente Links bekommen, ist nicht entschieden.
    """
    ausgabe = markdown_rendern(
        MANDANT,
        "[fremd](https://fremd.example/a)\n\n"
        "<https://fremd.example/b>\n\n"
        "<a@fremd.example>\n\n"
        "[fremd][r]\n\n"
        "[r]: https://fremd.example/c\n\n" + GEGENPROBE,
    )
    assert "<a" not in ausgabe
    assert "href" not in ausgabe
    assert "mailto" not in ausgabe
    assert "[fremd](https://fremd.example/a)" in ausgabe
    assert "&lt;https://fremd.example/b&gt;" in ausgabe
    assert "&lt;a@fremd.example&gt;" in ausgabe
    assert "[fremd][r]" in ausgabe
    _gegenprobe_rendert(ausgabe)


def test_referenzdefinition_bleibt_als_text_stehen() -> None:
    """Ist `reference` aktiv, verschwindet eine Definition still aus der Ausgabe.

    Die Gegenprobe zeigt das an derselben Eingabe: Mit der Regel ist die Zeile
    weg. Ohne diesen Nachweis waere nicht zu sehen, ob der Test ueberhaupt
    etwas fangen kann.
    """
    eingabe = "Text davor.\n\n[r]: https://fremd.example/c\n"
    assert "[r]: https://fremd.example/c" in markdown_rendern(MANDANT, eingabe)

    mit_regel = MarkdownIt("js-default").disable(["image", "link", "autolink"])
    assert "fremd.example" not in mit_regel.render(eingabe)


def test_fette_erste_zeile_wird_zum_umbruch_und_sonst_nichts() -> None:
    """Die FAQ-Form bekommt ein <br>, und nichts sonst.

    Faengt eine fehlende Regel und eine, die zu weit greift. Der Korpus ist
    hart umbrochen; ein <br> an jeder Zeile zerhackte jeden Absatz.
    """
    assert markdown_rendern(MANDANT, "**Frage?**\nAntwort.\n") == (
        "<p><strong>Frage?</strong><br>\nAntwort.</p>\n"
    )

    # Die Regel greift auch im Absatz eines Listenpunkts. Im Korpus kommt diese
    # Form nicht vor; festgehalten ist, was die Regel dort tut.
    assert markdown_rendern(MANDANT, "- **Begriff**\n  eingerueckter Text\n") == (
        "<ul>\n<li><strong>Begriff</strong><br>\neingerueckter Text</li>\n</ul>\n"
    )

    for eingabe in (
        "**Hinweis:** Text, der auf derselben Zeile weiterlaeuft,\nund eine zweite Zeile.\n",
        "Ein Satz mit **Fettdruck** in der Mitte,\nund eine zweite Zeile.\n",
        "- **Zone A** — Festland,\n  zweite Zeile.\n",
        "1. **Aufmass.** Wir messen\n   vor Ort.\n",
        "**Voraussetzungen:**\n\n- erstens\n- zweitens\n",
        "Erste Zeile\n**fett, aber nicht am Anfang des Absatzes**\nweiter.\n",
    ):
        assert "<br>" not in markdown_rendern(MANDANT, eingabe), eingabe


def test_echte_tabelle_aus_where_we_ship() -> None:
    """Die echte Datei: fuenf Spalten, drei Zeilen, ein Sonderzeichen.

    Faengt eine abgeschaltete Tabellenregel, verlorene Zellen und
    Kodierungsschaeden.
    """
    text = (KORPUS / "demo-fellgate" / "docs" / "where-we-ship.md").read_text(encoding="utf-8")
    ausgabe = markdown_rendern("demo-fellgate", text)

    assert ausgabe.count("<table>") == 1
    assert re.findall(r"<th>(.*?)</th>", ausgabe) == [
        "Zone",
        "Where it applies",
        "Standard delivery",
        "Shipping below 60 euro",
        "Shipping from 60 euro",
    ]
    assert len(re.findall(r"<tr>\n<td>", ausgabe)) == 3
    assert "<td>Mainland Denmark, Finland and Norway</td>" in ausgabe
    assert "Øresund" in ausgabe

    tabelle = ausgabe[ausgabe.index("<table>") : ausgabe.index("</table>")]
    assert not any(rest for _, _, rest in _TAG.findall(tabelle)), "Attribut in der Tabelle"


def test_ein_dokument_je_sprache() -> None:
    """Ein deutsches und ein englisches Dokument, beide echt.

    Die Zahl 17 ist die Zahl der Eintraege in der FAQ. Faengt eine Regel, die am
    Beispiel gruen und an der echten FAQ wirkungslos ist: Um den Fettdruck
    liegen leere Text-Token, und eine Pruefung auf das naechste Kind traefe nie.
    """
    deutsch = markdown_rendern(
        "demo-acme",
        (KORPUS / "demo-acme" / "docs" / "versand-und-lieferzeiten.md").read_text(encoding="utf-8"),
    )
    assert "<h1>Versand und Lieferzeiten</h1>" in deutsch
    assert "<table>" in deutsch

    faq = markdown_rendern(
        "demo-fellgate",
        (KORPUS / "demo-fellgate" / "docs" / "faq.md").read_text(encoding="utf-8"),
    )
    assert "<h1>" in faq
    assert "<strong>" in faq
    assert faq.count("</strong><br>") == 17


def test_der_ganze_korpus_rendert_ohne_ein_einziges_attribut() -> None:
    """Jedes Dokument jedes Demomandanten: nur bekannte Tags, kein Attribut.

    Faengt style, href, src oder class an jeder Stelle des echten Korpus. Die
    Mindestzahlen verhindern eine Suche, die blind ist, weil sie nichts findet.
    """
    mandanten = sorted(p.name for p in KORPUS.glob("demo-*") if p.is_dir())
    assert len(mandanten) >= 3, mandanten

    for slug in mandanten:
        dateien = sorted((KORPUS / slug / "docs").glob("*.md"))
        assert dateien, f"{slug} hat kein Dokument"
        for datei in dateien:
            ausgabe = markdown_rendern(slug, datei.read_text(encoding="utf-8"))
            tags = _TAG.findall(ausgabe)
            assert tags, f"{slug}/{datei.name}: kein Tag"
            for _, name, rest in tags:
                assert name in ERLAUBTE_TAGS, f"{slug}/{datei.name}: <{name}>"
                assert rest == "", f"{slug}/{datei.name}: <{name}{rest}>"


def test_dokumentseite_laedt_nichts_nach() -> None:
    """Kein Skript, kein Stylesheet, keine Schrift, kein Bild von irgendwoher."""
    seite = dokumentseite_bauen(MANDANT, "doku.md", "de", "# Titel\n\nText mit **fett**.\n")

    for verboten in ("src=", "href=", "<script", "<link", "url(", "@import"):
        assert verboten not in seite.lower(), verboten

    # Gegenprobe: Stil und Inhalt stehen in der Seite.
    assert "<style>" in seite
    assert "<h1>Titel</h1>" in seite
    assert "<strong>fett</strong>" in seite
    assert '<html lang="de">' in seite


def test_platzhalter_loesen_nichts_aus() -> None:
    """Platzhalter im Dokument und im Dateinamen kommen woertlich an.

    Faengt einen Seitenbau durch Ersetzen nacheinander: Dort wuerde ein
    `{{titel}}` im Dokument vom naechsten Schritt ersetzt.
    """
    text = "Hier steht {{inhalt}}, {{titel}} und {{lang}}.\n"
    fragment = markdown_rendern(MANDANT, text)
    seite = dokumentseite_bauen(MANDANT, "{{inhalt}}.md", "de", text)

    assert "<title>{{inhalt}}.md</title>" in seite
    assert seite.count(fragment) == 1
    assert "{{titel}}" in fragment and "{{lang}}" in fragment
    assert seite.count("{{titel}}") == 1
    assert seite.count("{{lang}}") == 1
    assert seite.count("{{inhalt}}") == 2


def _farbvariablen(css: str) -> tuple[dict[str, str], dict[str, str]]:
    """Die Variablen aus dem hellen :root und aus dem dunklen Block."""
    hell, dunkel = css.split("@media (prefers-color-scheme: dark)")
    muster = re.compile(r"(--[a-z-]+):\s*([^;]+);")
    return dict(muster.findall(hell)), dict(muster.findall(dunkel.split("}")[0]))


def test_palette_ist_an_beiden_stellen_gleich() -> None:
    """Die Palette steht in static/style.css und in der Dokumentseite.

    Laeuft sie auseinander, sieht das Dokument im Rahmen anders aus als der
    Chat darum herum, hell oder dunkel. Jede Variable der Dokumentseite muss es
    auch im Chat geben, mit demselben Wert. Die Mindestzahl ist die Gegenprobe
    gegen einen Vergleich, der nichts vergleicht.
    """
    stil_hell, stil_dunkel = _farbvariablen(dokumentseite._STIL)
    css_hell, css_dunkel = _farbvariablen(
        (REPO_ROOT / "static" / "style.css").read_text(encoding="utf-8")
    )

    for name, stil, css in (("hell", stil_hell, css_hell), ("dunkel", stil_dunkel, css_dunkel)):
        gemeinsam = set(stil) & set(css)
        assert len(gemeinsam) >= 3, (name, sorted(gemeinsam))
        assert set(stil) <= set(css), (name, sorted(set(stil) - set(css)))
        for variable in sorted(gemeinsam):
            assert stil[variable] == css[variable], (name, variable, stil[variable], css[variable])
