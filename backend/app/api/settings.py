from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.logging_config import get_logger
from app.schemas.settings import (
    AppSettingsResponse,
    AppSettingsUpdate,
    ModelPresetCreate,
    ModelPresetResponse,
)
from app.services.supabase_client import get_supabase

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=AppSettingsResponse)
async def get_settings():
    """Get current app settings."""
    sb = get_supabase()
    try:
        result = sb.table("app_settings").select("*").eq("id", "default").single().execute()
        row = result.data
        if row:
            return AppSettingsResponse(
                bankroll=row.get("bankroll", 1000.0),
                unit_size=row.get("unit_size", 10.0),
                active_preset=row.get("active_preset", "balanced"),
                fantasy_format=row.get("fantasy_format", "draftkings"),
                preferred_books=row.get("preferred_books", []),
            )
    except Exception as e:
        logger.warning("Error reading settings, using defaults", error=str(e))
    return AppSettingsResponse()


@router.put("/")
async def update_settings(settings: AppSettingsUpdate):
    """Update app settings."""
    sb = get_supabase()
    try:
        update_data = {k: v for k, v in settings.dict().items() if v is not None}
        if not update_data:
            return {"status": "ok", "message": "No changes"}
        sb.table("app_settings").update(update_data).eq("id", "default").execute()
        return {"status": "ok", "updated": list(update_data.keys())}
    except Exception as e:
        logger.error("Error updating settings", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/presets", response_model=list[ModelPresetResponse])
async def get_presets():
    """Get all model presets."""
    sb = get_supabase()
    try:
        result = sb.table("model_presets").select("*").order("is_builtin", desc=True).execute()
        return [
            ModelPresetResponse(
                id=row["id"],
                name=row["name"],
                description=row.get("description", ""),
                kelly_fraction=row.get("kelly_fraction", 0.5),
                min_confidence=row.get("min_confidence", 0.55),
                is_builtin=row.get("is_builtin", False),
                model_weights=row.get("model_weights"),
            )
            for row in (result.data or [])
        ]
    except Exception as e:
        logger.warning("Error reading presets, using defaults", error=str(e))
        return [
            ModelPresetResponse(id="conservative", name="Conservative",
                description="Quarter Kelly, high confidence threshold.",
                kelly_fraction=0.25, min_confidence=0.7, is_builtin=True),
            ModelPresetResponse(id="balanced", name="Balanced",
                description="Half Kelly, standard confidence.",
                kelly_fraction=0.5, min_confidence=0.55, is_builtin=True),
            ModelPresetResponse(id="aggressive", name="Aggressive",
                description="Full Kelly, lower confidence threshold.",
                kelly_fraction=1.0, min_confidence=0.4, is_builtin=True),
        ]


@router.post("/presets", response_model=ModelPresetResponse)
async def create_preset(preset: ModelPresetCreate):
    """Create a custom model preset."""
    sb = get_supabase()
    try:
        result = sb.table("model_presets").insert({
            "name": preset.name,
            "description": preset.description,
            "kelly_fraction": preset.kelly_fraction,
            "min_confidence": preset.min_confidence,
            "model_weights": preset.model_weights or {},
            "is_builtin": False,
        }).execute()
        if result.data:
            row = result.data[0]
            return ModelPresetResponse(
                id=row["id"], name=row["name"],
                description=row.get("description", ""),
                kelly_fraction=row.get("kelly_fraction", 0.5),
                min_confidence=row.get("min_confidence", 0.55),
                is_builtin=False,
                model_weights=row.get("model_weights"),
            )
        raise HTTPException(status_code=500, detail="Failed to create preset")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating preset", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
