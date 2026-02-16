from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ModelContribution(BaseModel):
    model_name: str
    prediction: float
    confidence: float = 0.0
    weight: float = 0.0


class FeatureImportance(BaseModel):
    feature_name: str
    importance: float
    direction: str  # "positive" or "negative"


class PredictionDetail(BaseModel):
    id: str
    player_id: str
    player_name: str
    team: str
    opponent: str
    game_id: str
    prop_type: str  # "points", "rebounds", "assists", "threes", etc.
    line: float
    predicted_value: float
    prediction_range_low: float
    prediction_range_high: float
    over_probability: float
    under_probability: float
    confidence_score: float
    confidence_tier: int  # 1-5 stars
    edge_pct: float
    expected_value: float
    recommended_bet: str  # "over" or "under"
    kelly_bet_size: float
    best_book: str
    best_odds: int  # American odds
    ensemble_agreement: float  # 0-1, how much models agree
    model_contributions: list[ModelContribution] = []
    feature_importances: list[FeatureImportance] = []
    line_edge_signal: str | None = None  # "strong_over", "moderate_over", etc.
    avg_vs_line_pct: float | None = None  # L10 avg vs line deviation %
    pct_games_over_line: float | None = None  # % of last 20 games over line
    l10_avg: float | None = None  # Last 10 game average for this stat
    created_at: datetime | None = None


class GameInfo(BaseModel):
    game_id: str
    home_team: str  # abbreviation
    away_team: str  # abbreviation
    home_team_name: str
    away_team_name: str
    game_date: str
    pick_count: int = 0


class PredictionResponse(BaseModel):
    id: str
    message: str | None = None
    prediction: PredictionDetail | None = None


class PredictionListResponse(BaseModel):
    predictions: list[PredictionDetail]
    games: list[GameInfo] = []
    total: int
    filters_applied: dict = {}
