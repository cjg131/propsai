from __future__ import annotations
from pydantic import BaseModel


class AppSettingsResponse(BaseModel):
    bankroll: float = 1000.0
    unit_size: float = 10.0
    active_preset: str = "balanced"
    fantasy_format: str = "draftkings"
    preferred_books: list[str] = []


class AppSettingsUpdate(BaseModel):
    bankroll: float | None = None
    unit_size: float | None = None
    active_preset: str | None = None
    fantasy_format: str | None = None
    preferred_books: list[str] | None = None


class ModelPresetResponse(BaseModel):
    id: str
    name: str
    description: str
    kelly_fraction: float
    min_confidence: float
    is_builtin: bool = False
    model_weights: dict[str, float] | None = None


class ModelPresetCreate(BaseModel):
    name: str
    description: str = ""
    kelly_fraction: float = 0.5
    min_confidence: float = 0.55
    model_weights: dict[str, float] | None = None
