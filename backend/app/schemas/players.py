from __future__ import annotations
from pydantic import BaseModel
from datetime import datetime


class PlayerStats(BaseModel):
    season_avg_points: float = 0.0
    season_avg_rebounds: float = 0.0
    season_avg_assists: float = 0.0
    season_avg_threes: float = 0.0
    season_avg_steals: float = 0.0
    season_avg_blocks: float = 0.0
    season_avg_turnovers: float = 0.0
    season_avg_minutes: float = 0.0
    last5_avg_points: float = 0.0
    last5_avg_rebounds: float = 0.0
    last5_avg_assists: float = 0.0
    last10_avg_points: float = 0.0
    last10_avg_rebounds: float = 0.0
    last10_avg_assists: float = 0.0
    home_avg_points: float = 0.0
    away_avg_points: float = 0.0
    usage_rate: float = 0.0
    games_played: int = 0


class InjuryInfo(BaseModel):
    status: str  # "healthy", "questionable", "doubtful", "out"
    description: str | None = None
    last_updated: datetime | None = None
    source: str | None = None


class PlayerDetail(BaseModel):
    id: str
    name: str
    team: str
    team_id: str
    position: str
    jersey_number: str | None = None
    headshot_url: str | None = None
    stats: PlayerStats | None = None
    injury: InjuryInfo | None = None
    is_starter: bool = True
    is_recently_traded: bool = False
    is_rookie: bool = False


class PlayerResponse(BaseModel):
    id: str
    message: str | None = None
    player: PlayerDetail | None = None


class PlayerListResponse(BaseModel):
    players: list[PlayerDetail] = []
    total: int = 0


class ScoutingReportResponse(BaseModel):
    player_id: str
    report: str
    generated_at: datetime | None = None
    model_used: str = "gpt-4"
    stats_summary: PlayerStats | None = None
    matchup_analysis: str | None = None
    injury_impact: str | None = None
    prop_recommendations: list[dict] = []
