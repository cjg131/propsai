from __future__ import annotations

from app.utils.kelly import (
    american_to_decimal,
    american_to_implied_probability,
    kelly_criterion,
    calculate_ev,
    calculate_edge,
    confidence_to_tier,
    kelly_to_units,
)


def test_american_to_decimal_negative():
    assert american_to_decimal(-110) == (100 / 110) + 1


def test_american_to_decimal_positive():
    assert american_to_decimal(150) == 2.5


def test_implied_probability_negative():
    prob = american_to_implied_probability(-110)
    assert abs(prob - 0.5238) < 0.001


def test_implied_probability_positive():
    prob = american_to_implied_probability(200)
    assert abs(prob - 0.3333) < 0.001


def test_kelly_criterion_positive_edge():
    # 60% win prob, -110 odds (1.909 decimal), half Kelly
    result = kelly_criterion(0.60, 1.909, fraction=0.5)
    assert result > 0


def test_kelly_criterion_no_edge():
    # 50% win prob, -110 odds — no edge
    result = kelly_criterion(0.50, 1.909, fraction=0.5)
    # Should be near zero or zero (slight negative edge at -110)
    assert result <= 0.01


def test_kelly_criterion_negative_edge():
    result = kelly_criterion(0.40, 1.909, fraction=0.5)
    assert result == 0.0


def test_calculate_ev_positive():
    ev = calculate_ev(0.60, -110, stake=100)
    assert ev > 0


def test_calculate_ev_negative():
    ev = calculate_ev(0.40, -110, stake=100)
    assert ev < 0


def test_calculate_edge_positive():
    edge = calculate_edge(0.60, -110)
    assert edge > 0


def test_calculate_edge_negative():
    edge = calculate_edge(0.45, -110)
    assert edge < 0


def test_confidence_to_tier():
    assert confidence_to_tier(0.80) == 5
    assert confidence_to_tier(0.70) == 4
    assert confidence_to_tier(0.60) == 3
    assert confidence_to_tier(0.55) == 2
    assert confidence_to_tier(0.45) == 1


def test_kelly_to_units():
    units = kelly_to_units(0.05, bankroll=1000, unit_size=10)
    assert units == 5.0
