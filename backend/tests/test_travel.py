from __future__ import annotations

from app.utils.travel import (
    haversine_distance,
    get_travel_distance,
    get_timezone_change,
    calculate_fatigue_score,
)


def test_haversine_distance_same_point():
    assert haversine_distance(40.0, -74.0, 40.0, -74.0) == 0.0


def test_haversine_distance_known():
    # NYC to LA is roughly 2450 miles
    dist = haversine_distance(40.7128, -74.0060, 34.0522, -118.2437)
    assert 2400 < dist < 2500


def test_get_travel_distance_cross_country():
    dist = get_travel_distance("BOS", "LAL")
    assert dist > 2000


def test_get_travel_distance_same_city():
    dist = get_travel_distance("LAL", "LAC")
    assert dist < 50


def test_get_travel_distance_unknown_team():
    dist = get_travel_distance("UNKNOWN", "LAL")
    assert dist == 0.0


def test_timezone_change_east_to_west():
    tz = get_timezone_change("BOS", "LAL")
    assert tz == 3


def test_timezone_change_same_tz():
    tz = get_timezone_change("BOS", "NYK")
    assert tz == 0


def test_fatigue_score_fresh():
    score = calculate_fatigue_score(
        travel_distance=0, timezone_change=0,
        is_back_to_back=False, days_rest=3,
    )
    assert score == 0.0  # Well-rested at home


def test_fatigue_score_exhausted():
    score = calculate_fatigue_score(
        travel_distance=2500, timezone_change=3,
        is_back_to_back=True, days_rest=0,
    )
    assert score >= 0.8


def test_fatigue_score_moderate():
    score = calculate_fatigue_score(
        travel_distance=800, timezone_change=1,
        is_back_to_back=False, days_rest=1,
    )
    assert 0.1 < score < 0.5
