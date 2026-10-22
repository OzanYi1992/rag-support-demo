"""Erkennt die Sprache einer Frage oder einer Antwort, ohne Netz.

Wofuer: Der Server legt die Antwortsprache fest, nicht das Modell. Er erkennt
die Sprache der getippten Frage und prueft, ob die Antwort sie haelt. Beides
laeuft ueber `sprache_bestimmen()`.

Erkannt wird in zwei Stufen, und die erste ist der Grund, warum es zwei sind:

1. `rank()` ueber alle Sprachen des Modells. Steht vorne keine Sprache des
   Katalogs, ist der Text nicht erkennbar. Ein Erkenner, der nur zwischen den
   Sprachen des Katalogs waehlen darf, haelt eine norwegische Frage oft sicher
   fuer Deutsch: "Leverer dere til Norge?" kommt so auf de mit 0,887.
2. Sonst zaehlt der Anteil dieser Sprache an der Summe der Katalogsprachen. Ab
   `SCHWELLE` gilt sie, darunter ist der Text nicht erkennbar. Der Anteil ist
   derselbe Wert, den der Erkenner nach `set_languages()` als Konfidenz meldete;
   ein zweites, eingeschraenktes Objekt braucht es deshalb nicht.

"Nicht erkennbar" ist ein Ergebnis und kein Fehler: Dann entscheidet der
Aufrufer, etwa mit der Sprache des Mandanten.
"""

from __future__ import annotations

import threading

from py3langid.langid import MODEL_FILE, LanguageIdentifier

from app.texts import VERFUEGBARE_SPRACHEN

# Ab diesem Anteil an den Katalogsprachen gilt die vorderste Sprache als
# erkannt. Darunter ist der Text nicht erkennbar, und der Aufrufer faellt auf
# eine andere Sprache zurueck.
SCHWELLE = 0.7


# --- Ein Erkenner je Prozess -------------------------------------------------
#
# Wie der Embedder entsteht der Erkenner einmal je Prozess und wird ueber
# Anfragen und Mandanten hinweg geteilt. Er liegt auf Modulebene, und das ist
# hier zulaessig, weil zwei Eigenschaften gelten, die ein Retriever oder Index
# nie hat:
#
# 1. Mandantenunabhaengig. Derselbe Text ergibt fuer jeden Mandanten dieselbe
#    Rangfolge. Es gibt keinen Mandantenbezug, den das Objekt verlieren koennte.
# 2. Zustandslos. Das Objekt haelt die Modelltabellen, sonst nichts. In
#    py3langid 0.4.0 legen `rank()` und `classify()` ihre Werte je Aufruf neu
#    an (`LanguageIdentifier._decide`); nur `set_languages()` setzt Tabellen
#    neu. Die Anwendung ruft `set_languages()` nie. Damit ist das Objekt auch
#    ueber die Threads der Route hinweg sicher geteilt.
#
# Faellt eine der beiden Eigenschaften weg, etwa durch einen Cache ueber
# erkannte Texte, faellt die Erlaubnis. Durchgesetzt wird das in
# tests/test_sprache.py, nicht durch diesen Kommentar.
#
# Bewusst nicht die Funktionen auf Modulebene von py3langid: Sie teilen ein
# Objekt, auf dem jeder Aufrufer `set_languages()` rufen kann.

_geteilter_erkenner: LanguageIdentifier | None = None
_erkenner_sperre = threading.Lock()


def erkenner() -> LanguageIdentifier:
    """Liefert den Erkenner dieses Prozesses und laedt ihn beim ersten Bedarf.

    Das Laden dauert rund 0,4 s. Die Anwendung waermt den Erkenner beim Start
    vor, damit die erste Frage es nicht bezahlt. Die Sperre sorgt dafuer, dass
    zwei gleichzeitige erste Aufrufe ihn nicht zweimal laden.
    """
    global _geteilter_erkenner
    if _geteilter_erkenner is None:
        with _erkenner_sperre:
            if _geteilter_erkenner is None:
                _geteilter_erkenner = LanguageIdentifier.from_model_file(
                    MODEL_FILE, norm_probs=True
                )
    return _geteilter_erkenner


def _erkenner_leeren() -> None:
    """Nur fuer Tests. Im Betrieb gibt es keinen Grund, den Erkenner neu zu laden."""
    global _geteilter_erkenner
    with _erkenner_sperre:
        _geteilter_erkenner = None


def katalogverteilung(text: str) -> tuple[str, dict[str, float]]:
    """Die vorderste Sprache des Modells und der Anteil jeder Katalogsprache.

    Die Anteile summieren sich zu 1 und sind das, was ein auf den Katalog
    eingeschraenkter Erkenner als Wahrscheinlichkeit meldete. Die vorderste
    Sprache kommt aus der Rangfolge ueber ALLE Sprachen des Modells und kann
    ausserhalb des Katalogs liegen.
    """
    rangfolge = erkenner().rank(text)
    werte = dict(rangfolge)
    summe = sum(werte[sprache] for sprache in VERFUEGBARE_SPRACHEN)
    return rangfolge[0][0], {s: werte[s] / summe for s in VERFUEGBARE_SPRACHEN}


def sprache_bestimmen(text: str) -> str | None:
    """Die Sprache eines Textes aus dem Katalog, oder None, wenn sie nicht
    erkennbar ist.

    Nicht erkennbar heisst: Vorne steht eine Sprache ausserhalb des Katalogs,
    oder der Anteil der vordersten an den Katalogsprachen liegt unter
    `SCHWELLE`. Ein leerer Text ist nicht erkennbar, ohne den Erkenner zu
    fragen.

    Keine tenant_id: Die Funktion beruehrt nur den uebergebenen Text und haelt
    nichts davon fest, wie `normalisieren()` und `passt_zur_sprache()`.
    """
    if not text.strip():
        return None
    vorne, anteile = katalogverteilung(text)
    if vorne not in anteile:
        return None
    return vorne if anteile[vorne] >= SCHWELLE else None
