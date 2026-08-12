"""Tests der Eskalationsstrategien.

Schwerpunkt ist, dass eine unkalibrierte Strategie BEIM ERZEUGEN scheitert und
nicht beim Vergleich. Eine Defensivpruefung am Vergleich liesse die Eskalation
stillschweigend nie feuern - das System antwortete genau dann, wenn es schweigen
muesste (OP-013).
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.escalation import (
    REASON_BELOW_THRESHOLD,
    REASON_FLAT,
    REASON_NO_HITS,
    AbsoluteThreshold,
    DegenerateOnly,
    EscalationConfigError,
    RelativeMargin,
    build_escalation_strategy,
)
from app.search import SearchHit


def hits(*scores: float) -> list[SearchHit]:
    return [
        SearchHit(
            text=f"Chunk {i}",
            score=s,
            source_file="doku.md",
            chunk_index=i,
            tenant_slug="demo-acme",
        )
        for i, s in enumerate(scores)
    ]


# --- DegenerateOnly --------------------------------------------------------


def test_degenerate_only_eskaliert_bei_leerer_liste() -> None:
    entscheidung = DegenerateOnly().should_escalate([])

    assert entscheidung.escalate is True
    assert entscheidung.reason == REASON_NO_HITS
    assert entscheidung.metric_name == "hit_count"
    assert entscheidung.metric_value == 0.0


def test_degenerate_only_eskaliert_sonst_nie() -> None:
    """Auch bei durchweg schlechten Scores. Das ist Absicht, nicht Nachlaessigkeit.

    Bis Phase 5 traegt das Groundedness-Tor die Last allein (ADR-018).
    """
    entscheidung = DegenerateOnly().should_escalate(hits(0.51, 0.50, 0.50))

    assert entscheidung.escalate is False
    assert entscheidung.reason is None
    assert entscheidung.metric_value == 3.0


def test_degenerate_only_braucht_keine_konfiguration() -> None:
    """Der Kern der Voreinstellung: sie kann nicht unkalibriert scheitern."""
    DegenerateOnly()


# --- AbsoluteThreshold -----------------------------------------------------


def test_absolute_threshold_ohne_wert_wirft_beim_erzeugen() -> None:
    with pytest.raises(EscalationConfigError) as excinfo:
        AbsoluteThreshold(None)

    meldung = str(excinfo.value)
    assert "RETRIEVAL_SCORE_THRESHOLD" in meldung
    assert "P-005" in meldung


def test_absolute_threshold_eskaliert_unter_dem_wert() -> None:
    entscheidung = AbsoluteThreshold(0.90).should_escalate(hits(0.88, 0.87))

    assert entscheidung.escalate is True
    assert entscheidung.reason == REASON_BELOW_THRESHOLD
    assert entscheidung.metric_name == "best_score"
    assert entscheidung.metric_value == pytest.approx(0.88)


def test_absolute_threshold_eskaliert_nicht_darueber() -> None:
    entscheidung = AbsoluteThreshold(0.90).should_escalate(hits(0.95, 0.80))

    assert entscheidung.escalate is False
    assert entscheidung.metric_value == pytest.approx(0.95)


def test_absolute_threshold_eskaliert_bei_leerer_liste() -> None:
    assert AbsoluteThreshold(0.90).should_escalate([]).escalate is True


# --- RelativeMargin --------------------------------------------------------


def test_relative_margin_ohne_wert_wirft_beim_erzeugen() -> None:
    with pytest.raises(EscalationConfigError) as excinfo:
        RelativeMargin(None)

    assert "ESCALATION_MIN_MARGIN" in str(excinfo.value)


def test_relative_margin_eskaliert_bei_flachem_feld() -> None:
    """Gleichmaessiges Rauschen: kein Treffer hebt sich ab."""
    entscheidung = RelativeMargin(0.05).should_escalate(hits(0.91, 0.90, 0.90, 0.89))

    assert entscheidung.escalate is True
    assert entscheidung.reason == REASON_FLAT
    assert entscheidung.metric_name == "margin_best_median"


def test_relative_margin_eskaliert_nicht_bei_klarem_vorsprung() -> None:
    entscheidung = RelativeMargin(0.05).should_escalate(hits(0.95, 0.70, 0.68, 0.65))

    assert entscheidung.escalate is False
    assert entscheidung.metric_value > 0.05


def test_relative_margin_eskaliert_bei_leerer_liste() -> None:
    assert RelativeMargin(0.05).should_escalate([]).escalate is True


# --- Factory ---------------------------------------------------------------


def _settings(**kwargs: object) -> Settings:
    return Settings(openai_api_key="x", openai_model="x", **kwargs)


def test_factory_liefert_degenerate_only_als_voreinstellung() -> None:
    strategie = build_escalation_strategy(_settings())

    assert isinstance(strategie, DegenerateOnly)


def test_factory_baut_absolute_threshold() -> None:
    strategie = build_escalation_strategy(
        _settings(escalation_strategy="absolute_threshold", retrieval_score_threshold=0.9)
    )

    assert isinstance(strategie, AbsoluteThreshold)


def test_factory_baut_relative_margin() -> None:
    strategie = build_escalation_strategy(
        _settings(escalation_strategy="relative_margin", escalation_min_margin=0.05)
    )

    assert isinstance(strategie, RelativeMargin)


def test_factory_wirft_bei_fehlendem_wert() -> None:
    """Der Fehler faellt beim Bauen an, nicht bei der ersten Frage."""
    with pytest.raises(EscalationConfigError):
        build_escalation_strategy(_settings(escalation_strategy="absolute_threshold"))
