from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class BookOdds(BaseModel):
    sportsbook: str
    line: float
    over_odds: int  # American odds
    under_odds: int
    last_updated: datetime | None = None


class OddsComparison(BaseModel):
    player_id: str
    player_name: str
    team: str
    opponent: str
    game_id: str
    prop_type: str
    books: list[BookOdds] = []
    best_over_book: str | None = None
    best_over_odds: int | None = None
    best_under_book: str | None = None
    best_under_odds: int | None = None
    opening_line: float | None = None
    current_consensus_line: float | None = None


class OddsComparisonResponse(BaseModel):
    comparisons: list[OddsComparison] = []
    total: int = 0
