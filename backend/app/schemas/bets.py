from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class BetCreate(BaseModel):
    player_id: str
    player_name: str
    prop_type: str
    line: float
    bet_type: str  # "over" or "under"
    odds: int  # American odds
    sportsbook: str
    stake: float
    prediction_id: str | None = None
    notes: str | None = None


class BetDetail(BaseModel):
    id: str
    player_id: str
    player_name: str
    team: str
    opponent: str
    game_date: datetime
    prop_type: str
    line: float
    bet_type: str
    odds: int
    sportsbook: str
    stake: float
    status: str  # "pending", "won", "lost", "push"
    actual_value: float | None = None
    profit: float | None = None
    prediction_id: str | None = None
    confidence_tier: int | None = None
    notes: str | None = None
    created_at: datetime | None = None
    resolved_at: datetime | None = None


class BetResponse(BaseModel):
    id: str
    message: str | None = None
    bet: BetDetail | None = None


class BetListResponse(BaseModel):
    bets: list[BetDetail] = []
    total: int = 0


class BetSummaryResponse(BaseModel):
    total_bets: int = 0
    wins: int = 0
    losses: int = 0
    pending: int = 0
    pushes: int = 0
    win_rate: float = 0.0
    roi: float = 0.0
    total_wagered: float = 0.0
    total_profit: float = 0.0
    current_streak: int = 0
    streak_type: str = "none"  # "win", "loss", "none"
    best_prop_type: str | None = None
    worst_prop_type: str | None = None
