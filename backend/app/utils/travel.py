from __future__ import annotations

"""Travel distance and fatigue calculations for NBA teams."""

import math

NBA_ARENAS = {
    "ATL": {"lat": 33.757, "lon": -84.396, "tz": "America/New_York"},
    "BOS": {"lat": 42.366, "lon": -71.062, "tz": "America/New_York"},
    "BKN": {"lat": 40.683, "lon": -73.976, "tz": "America/New_York"},
    "CHA": {"lat": 35.225, "lon": -80.839, "tz": "America/New_York"},
    "CHI": {"lat": 41.881, "lon": -87.674, "tz": "America/Chicago"},
    "CLE": {"lat": 41.496, "lon": -81.688, "tz": "America/New_York"},
    "DAL": {"lat": 32.790, "lon": -96.810, "tz": "America/Chicago"},
    "DEN": {"lat": 39.749, "lon": -105.008, "tz": "America/Denver"},
    "DET": {"lat": 42.341, "lon": -83.055, "tz": "America/New_York"},
    "GSW": {"lat": 37.768, "lon": -122.388, "tz": "America/Los_Angeles"},
    "HOU": {"lat": 29.751, "lon": -95.362, "tz": "America/Chicago"},
    "IND": {"lat": 39.764, "lon": -86.156, "tz": "America/Indiana/Indianapolis"},
    "LAC": {"lat": 33.944, "lon": -118.341, "tz": "America/Los_Angeles"},
    "LAL": {"lat": 34.043, "lon": -118.267, "tz": "America/Los_Angeles"},
    "MEM": {"lat": 35.138, "lon": -90.051, "tz": "America/Chicago"},
    "MIA": {"lat": 25.781, "lon": -80.187, "tz": "America/New_York"},
    "MIL": {"lat": 43.045, "lon": -87.917, "tz": "America/Chicago"},
    "MIN": {"lat": 44.980, "lon": -93.276, "tz": "America/Chicago"},
    "NOP": {"lat": 29.949, "lon": -90.082, "tz": "America/Chicago"},
    "NYK": {"lat": 40.751, "lon": -73.994, "tz": "America/New_York"},
    "OKC": {"lat": 35.463, "lon": -97.515, "tz": "America/Chicago"},
    "ORL": {"lat": 28.539, "lon": -81.384, "tz": "America/New_York"},
    "PHI": {"lat": 39.901, "lon": -75.172, "tz": "America/New_York"},
    "PHX": {"lat": 33.446, "lon": -112.071, "tz": "America/Phoenix"},
    "POR": {"lat": 45.532, "lon": -122.667, "tz": "America/Los_Angeles"},
    "SAC": {"lat": 38.580, "lon": -121.500, "tz": "America/Los_Angeles"},
    "SAS": {"lat": 29.427, "lon": -98.438, "tz": "America/Chicago"},
    "TOR": {"lat": 43.643, "lon": -79.379, "tz": "America/Toronto"},
    "UTA": {"lat": 40.768, "lon": -111.901, "tz": "America/Denver"},
    "WAS": {"lat": 38.898, "lon": -77.021, "tz": "America/New_York"},
}

TIMEZONE_OFFSETS = {
    "America/New_York": -5,
    "America/Chicago": -6,
    "America/Denver": -7,
    "America/Los_Angeles": -8,
    "America/Phoenix": -7,
    "America/Indiana/Indianapolis": -5,
    "America/Toronto": -5,
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in miles between two lat/lon points."""
    R = 3959  # Earth radius in miles

    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def get_travel_distance(from_team: str, to_team: str) -> float:
    """Get travel distance in miles between two team arenas."""
    if from_team not in NBA_ARENAS or to_team not in NBA_ARENAS:
        return 0.0

    from_arena = NBA_ARENAS[from_team]
    to_arena = NBA_ARENAS[to_team]

    return round(
        haversine_distance(
            from_arena["lat"], from_arena["lon"],
            to_arena["lat"], to_arena["lon"],
        ),
        1,
    )


def get_timezone_change(from_team: str, to_team: str) -> int:
    """Get timezone change in hours between two team arenas."""
    if from_team not in NBA_ARENAS or to_team not in NBA_ARENAS:
        return 0

    from_tz = NBA_ARENAS[from_team]["tz"]
    to_tz = NBA_ARENAS[to_team]["tz"]

    from_offset = TIMEZONE_OFFSETS.get(from_tz, 0)
    to_offset = TIMEZONE_OFFSETS.get(to_tz, 0)

    return abs(to_offset - from_offset)


def calculate_fatigue_score(
    travel_distance: float,
    timezone_change: int,
    is_back_to_back: bool,
    days_rest: int,
) -> float:
    """
    Calculate a fatigue score from 0 (fresh) to 1 (exhausted).

    Factors:
    - Travel distance (>1500 miles = significant)
    - Timezone changes (each hour = more fatigue)
    - Back-to-back games
    - Days of rest
    """
    score = 0.0

    # Travel distance factor (0-0.3)
    if travel_distance > 2000:
        score += 0.3
    elif travel_distance > 1000:
        score += 0.2
    elif travel_distance > 500:
        score += 0.1

    # Timezone factor (0-0.2)
    score += min(timezone_change * 0.1, 0.2)

    # Back-to-back factor (0-0.3)
    if is_back_to_back:
        score += 0.3

    # Rest days factor (0-0.2)
    if days_rest == 0:
        score += 0.2
    elif days_rest == 1:
        score += 0.1
    elif days_rest >= 3:
        score -= 0.1  # Well-rested bonus

    return max(0.0, min(1.0, score))
