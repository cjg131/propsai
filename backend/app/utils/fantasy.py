from __future__ import annotations
"""Fantasy scoring calculations for DraftKings, FanDuel, and Yahoo."""


def draftkings_score(
    points: int = 0,
    rebounds: int = 0,
    assists: int = 0,
    steals: int = 0,
    blocks: int = 0,
    turnovers: int = 0,
    three_pointers_made: int = 0,
) -> float:
    """Calculate DraftKings NBA fantasy score."""
    score = (
        points * 1.0
        + rebounds * 1.25
        + assists * 1.5
        + steals * 2.0
        + blocks * 2.0
        + turnovers * -0.5
        + three_pointers_made * 0.5
    )

    # Double-double bonus
    stats_over_10 = sum(
        1 for s in [points, rebounds, assists, steals, blocks] if s >= 10
    )
    if stats_over_10 >= 2:
        score += 1.5
    # Triple-double bonus
    if stats_over_10 >= 3:
        score += 3.0

    return round(score, 1)


def fanduel_score(
    points: int = 0,
    rebounds: int = 0,
    assists: int = 0,
    steals: int = 0,
    blocks: int = 0,
    turnovers: int = 0,
) -> float:
    """Calculate FanDuel NBA fantasy score."""
    score = (
        points * 1.0
        + rebounds * 1.2
        + assists * 1.5
        + steals * 3.0
        + blocks * 3.0
        + turnovers * -1.0
    )
    return round(score, 1)


def yahoo_score(
    points: int = 0,
    rebounds: int = 0,
    assists: int = 0,
    steals: int = 0,
    blocks: int = 0,
    turnovers: int = 0,
    three_pointers_made: int = 0,
    field_goals_made: int = 0,
    field_goals_attempted: int = 0,
    free_throws_made: int = 0,
    free_throws_attempted: int = 0,
) -> float:
    """Calculate Yahoo NBA fantasy score."""
    score = (
        points * 1.0
        + rebounds * 1.2
        + assists * 1.5
        + steals * 3.0
        + blocks * 3.0
        + turnovers * -1.0
        + three_pointers_made * 0.5
        + (field_goals_made - field_goals_attempted) * 0.0  # FG% not directly scored
        + (free_throws_made - free_throws_attempted) * 0.0
    )
    return round(score, 1)


def calculate_fantasy_score(
    format: str,
    points: int = 0,
    rebounds: int = 0,
    assists: int = 0,
    steals: int = 0,
    blocks: int = 0,
    turnovers: int = 0,
    three_pointers_made: int = 0,
    field_goals_made: int = 0,
    field_goals_attempted: int = 0,
    free_throws_made: int = 0,
    free_throws_attempted: int = 0,
) -> float:
    """Calculate fantasy score for the given format."""
    if format == "draftkings":
        return draftkings_score(
            points, rebounds, assists, steals, blocks, turnovers, three_pointers_made
        )
    elif format == "fanduel":
        return fanduel_score(points, rebounds, assists, steals, blocks, turnovers)
    elif format == "yahoo":
        return yahoo_score(
            points, rebounds, assists, steals, blocks, turnovers,
            three_pointers_made, field_goals_made, field_goals_attempted,
            free_throws_made, free_throws_attempted,
        )
    else:
        raise ValueError(f"Unknown fantasy format: {format}")
