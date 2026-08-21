"""Entscheidung ueber Eskalation auf der Retrieval-Seite.

ADR-003 sagt: eskalieren statt erfinden. Das steht nicht zur Debatte. Offen ist
nach OP-018 allein, WORAN Eskalation gemessen wird - die Score-Spanne ueber 20
inhaltlich verschiedene Chunks betrug 0,0295, und die Modellkarte nennt das
erwartetes Verhalten. Ein absoluter Schwellwert hat auf dieser Skala wenig
Trennschaerfe.

Deshalb ist die Entscheidung hier eine austauschbare Strategie und kein fester
Vergleich (ADR-018). Welche gewinnt, entscheidet Phase 5 mit dem Goldsatz.

Dies ist das ERSTE von zwei Toren. Das zweite ist Groundedness in app/rag.py:
Retrieval kann hoch bewerten und die Antwort trotzdem nicht enthalten, weil der
Score Aehnlichkeit misst und nicht Abdeckung.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings
from app.search import SearchHit

# Gruende, die in der Answer und spaeter in der Telemetrie auftauchen. Als
# Konstanten, damit Phase 5 sie auswerten kann, ohne Zeichenketten zu raten.
REASON_NO_HITS = "retrieval_no_hits"
REASON_BELOW_THRESHOLD = "retrieval_below_threshold"
REASON_FLAT = "retrieval_flat"


class EscalationConfigError(ValueError):
    """Einer Strategie fehlt der Wert, den sie braucht.

    Wird BEIM ERZEUGEN geworfen, nicht beim Vergleich. Eine Defensivpruefung am
    Vergleich - "if schwellwert is not None and score < schwellwert" - liesse die
    Eskalation stillschweigend nie feuern. Das waere die ADR-008-Inversion in
    anderer Gestalt: Das System antwortet genau dann, wenn es schweigen muesste,
    und niemand merkt es. Siehe open-points.md, OP-013.
    """


@dataclass(frozen=True)
class EscalationDecision:
    """Ergebnis der Retrieval-seitigen Pruefung."""

    escalate: bool
    reason: str | None
    # Kennzahl, auf der die Entscheidung beruht. Geht in die Telemetrie (ADR-002),
    # damit Phase 5 auswerten kann, WARUM eskaliert wurde und nicht nur DASS.
    metric_name: str
    metric_value: float


class EscalationStrategy(Protocol):
    """Was jede Strategie koennen muss."""

    def should_escalate(self, hits: list[SearchHit]) -> EscalationDecision: ...


class DegenerateOnly:
    """Eskaliert ausschliesslich bei leerer Trefferliste. VOREINSTELLUNG.

    **Dies ist kein Schwellwert und soll keiner sein.**

    STRUKTURELL INERT: Diese Strategie greift auf dem regulaeren Weg praktisch
    nie, weil eine leere Trefferliste dort unerreichbar ist - ein vorhandener
    Index liefert immer Treffer, auch schlechte. Das ist der gewuenschte
    Zustand, nicht ein Fehler: Mit praktisch abgeschaltetem Retrieval-Tor
    misst man zum ersten Mal, wie oft das Groundedness-Tor ALLEIN eskaliert.
    Ein von Anfang an kalibriertes Retrieval-Tor haette diese Frage fuer immer
    verdeckt, weil es zuerst gegriffen haette.

    Wer in einer Auswertung sieht, dass das Retrieval-Tor nie greift, darf
    daraus NICHT schliessen, dass es funktioniert.

    Vor Phase 5 gibt es fuer keine der kalibrierten Strategien einen belastbaren
    Wert. Eine Zahl aus drei Fragen auf einem Miniaturkorpus waere keine Messung,
    sondern eine Anpassung an n=3 - und im ADR saehe sie aus wie ein belegter
    Wert. Nach P-005 ist ein geschaetzter Schwellwert wertlos.

    Der Ausweg liegt im zweiten Tor: Groundedness braucht keine Kalibrierung, es
    ist ein semantisches Urteil des Modells. Bis Phase 5 traegt es die Last
    allein.

    Nebeneffekt, der Phase 5 hilft: Mit praktisch abgeschaltetem Retrieval-Tor
    misst Phase 5 erstmals, wie oft Groundedness ALLEIN greift. Das ist die Zahl,
    die entscheidet, ob das Retrieval-Tor ueberhaupt gebraucht wird oder nur
    doppelt sichert. Ein von Anfang an kalibriertes Retrieval-Tor wuerde diese
    Frage fuer immer verdecken - es griffe zuerst, und niemand erfuehre, ob
    Groundedness gereicht haette.

    Siehe ADR-018 und open-points.md, OP-018.
    """

    def should_escalate(self, hits: list[SearchHit]) -> EscalationDecision:
        return EscalationDecision(
            escalate=not hits,
            reason=REASON_NO_HITS if not hits else None,
            metric_name="hit_count",
            metric_value=float(len(hits)),
        )


class AbsoluteThreshold:
    """Bester Score unter dem Schwellwert -> eskalieren.

    LAEUFT IN DER VOREINSTELLUNG NICHT. Das ist Absicht und keine
    Unfertigkeit - siehe ADR-018. Voreinstellung ist DegenerateOnly.

    Wer diese Strategie aktiviert, aendert das Messverfahren: Ab dann greift
    das Retrieval-Tor, und alle Zahlen zur Frage, wie oft das Groundedness-Tor
    ALLEIN eskaliert, sind hinfaellig. Der Goldsatz ist dann neu zu fahren
    (eval/run.py), sonst werden Ergebnisse aus zwei verschiedenen Verfahren
    nebeneinandergestellt.

    Die urspruengliche Annahme aus ADR-003. Nicht Voreinstellung, bis Phase 5
    gemessen hat: Auf einer Skala, deren Werte sich um drei Hundertstel draengen,
    muesste der Wert auf zwei Nachkommastellen sitzen und waere gegen jede
    Aenderung an Korpus oder Chunking instabil (OP-018).
    """

    def __init__(self, threshold: float | None) -> None:
        if threshold is None:
            raise EscalationConfigError(
                "ESCALATION_STRATEGY ist 'absolute_threshold', aber "
                "RETRIEVAL_SCORE_THRESHOLD ist nicht gesetzt. Der Wert hat bewusst "
                "keinen Default - ein geschaetzter Schwellwert ist wertlos "
                "(pitfalls.md, P-005; open-points.md, OP-004). Entweder den Wert "
                "setzen oder ESCALATION_STRATEGY auf 'degenerate_only' lassen."
            )
        self._threshold = threshold

    def should_escalate(self, hits: list[SearchHit]) -> EscalationDecision:
        if not hits:
            return EscalationDecision(True, REASON_NO_HITS, "best_score", 0.0)

        # Groesser ist besser (ADR-008). Die Normierung auf 0..1 passiert in
        # app/search.py; rohe Distanzen kommen hier nie an.
        bester = max(h.score for h in hits)
        return EscalationDecision(
            escalate=bester < self._threshold,
            reason=REASON_BELOW_THRESHOLD if bester < self._threshold else None,
            metric_name="best_score",
            metric_value=bester,
        )


class RelativeMargin:
    """Abstand zwischen bestem Treffer und dem Median der Trefferliste.

    LAEUFT IN DER VOREINSTELLUNG NICHT. Das ist Absicht und keine
    Unfertigkeit - siehe ADR-018. Voreinstellung ist DegenerateOnly.

    Wer diese Strategie aktiviert, aendert das Messverfahren: Ab dann greift
    das Retrieval-Tor, und alle Zahlen zur Frage, wie oft das Groundedness-Tor
    ALLEIN eskaliert, sind hinfaellig. Der Goldsatz ist dann neu zu fahren
    (eval/run.py), sonst werden Ergebnisse aus zwei verschiedenen Verfahren
    nebeneinandergestellt.

    Ein deutlicher Vorsprung des besten Treffers spricht fuer einen echten Fund,
    ein flaches Feld fuer gleichmaessiges Rauschen. Das ist gegen die enge
    Score-Verteilung robuster als ein absoluter Vergleich, weil es die Lage des
    Bandes herausrechnet und nur seine Form betrachtet.

    Nicht Voreinstellung, bis Phase 5 gemessen hat.
    """

    def __init__(self, min_margin: float | None) -> None:
        if min_margin is None:
            raise EscalationConfigError(
                "ESCALATION_STRATEGY ist 'relative_margin', aber "
                "ESCALATION_MIN_MARGIN ist nicht gesetzt. Der Wert hat bewusst "
                "keinen Default - er ist wie der Schwellwert modellspezifisch und "
                "gehoert kalibriert (pitfalls.md, P-005; open-points.md, OP-018). "
                "Entweder den Wert setzen oder ESCALATION_STRATEGY auf "
                "'degenerate_only' lassen."
            )
        self._min_margin = min_margin

    def should_escalate(self, hits: list[SearchHit]) -> EscalationDecision:
        if not hits:
            return EscalationDecision(True, REASON_NO_HITS, "margin_best_median", 0.0)

        scores = [h.score for h in hits]
        abstand = max(scores) - statistics.median(scores)
        return EscalationDecision(
            escalate=abstand < self._min_margin,
            reason=REASON_FLAT if abstand < self._min_margin else None,
            metric_name="margin_best_median",
            metric_value=abstand,
        )


def build_escalation_strategy(settings: Settings) -> EscalationStrategy:
    """Baut die konfigurierte Strategie und prueft dabei ihre Konfiguration.

    Bewusst eine Funktion und kein Modul-Level-Objekt (ADR-001). Die Strategie
    ist zwar mandantenunabhaengig, aber ein beim Import gebautes Objekt liesse
    einen Konfigurationsfehler erst beim Import auffallen - und damit auch in
    Tests, die mit Eskalation nichts zu tun haben.
    """
    name = settings.escalation_strategy
    if name == "degenerate_only":
        return DegenerateOnly()
    if name == "absolute_threshold":
        return AbsoluteThreshold(settings.retrieval_score_threshold)
    if name == "relative_margin":
        return RelativeMargin(settings.escalation_min_margin)
    raise EscalationConfigError(
        f"Unbekannte ESCALATION_STRATEGY: {name!r}. Erlaubt sind "
        f"'degenerate_only', 'absolute_threshold', 'relative_margin'."
    )
