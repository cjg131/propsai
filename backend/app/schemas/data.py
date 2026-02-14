from __future__ import annotations
from pydantic import BaseModel
from datetime import datetime


class DataStatusResponse(BaseModel):
    last_refresh: datetime | None = None
    total_players: int = 0
    total_games: int = 0
    total_seasons: int = 0
    api_quota_used: int = 0
    api_quota_limit: int = 0
    model_last_trained: datetime | None = None


class DataLoadResponse(BaseModel):
    status: str
    message: str
    seasons_requested: int = 0
    seasons_loaded: int = 0
    games_loaded: int = 0
    players_loaded: int = 0
