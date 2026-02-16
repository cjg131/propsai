from __future__ import annotations

import csv
import io
import json as json_lib

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from app.schemas.data import DataLoadResponse, DataStatusResponse
from app.services.data_manager import get_data_manager

router = APIRouter()


@router.get("/status", response_model=DataStatusResponse)
async def get_data_status():
    """Get current data pipeline status."""
    manager = get_data_manager()
    status = await manager.get_data_status()
    return DataStatusResponse(**status)


@router.post("/refresh")
async def refresh_data(background_tasks: BackgroundTasks):
    """Manually refresh all data from APIs."""
    manager = get_data_manager()
    result = await manager.refresh_today_data()
    return {"status": "completed", "result": result}


@router.post("/load-historical", response_model=DataLoadResponse)
async def load_historical_data(seasons: int = 10):
    """Load historical NBA data into the database."""
    return DataLoadResponse(
        status="pending",
        message=f"Run: poetry run python scripts/seed_historical.py --seasons {seasons}",
        seasons_requested=seasons,
    )


@router.post("/retrain")
async def retrain_models():
    """Retrain all ensemble models with latest data."""
    manager = get_data_manager()
    result = await manager.retrain_models()
    return result


@router.post("/prefetch")
async def prefetch_bdl(background_tasks: BackgroundTasks):
    """
    Background pre-fetch all BDL season data with gentle rate limiting.
    Runs slowly (~10-15 min) to avoid 429s. Stores to file cache.
    Call this once, then predictions run instantly from cache.
    """
    import asyncio

    from app.services.balldontlie import get_balldontlie

    bdl = get_balldontlie()
    loop = asyncio.get_event_loop()

    def _run():
        return bdl.background_prefetch()

    background_tasks.add_task(loop.run_in_executor, None, _run)
    return {"status": "started", "message": "Background pre-fetch started. Check logs for progress."}


@router.get("/news")
async def get_news_sentiment():
    """
    Fetch recent NBA news articles with per-player sentiment scores.
    Returns articles from RSS feeds + NewsAPI, plus player sentiment map.
    """
    import asyncio

    from app.services.news_sentiment import get_news_sentiment as get_news_svc
    from app.services.supabase_client import get_supabase

    loop = asyncio.get_event_loop()
    news_svc = get_news_svc()

    # Get known player names from today's predictions
    sb = get_supabase()
    known_players: set[str] = set()
    try:
        from datetime import date
        today_str = date.today().isoformat()
        preds = sb.table("predictions").select(
            "*, players(name)"
        ).gte("created_at", today_str).execute()
        for p in (preds.data or []):
            player_info = p.get("players") or {}
            name = player_info.get("name", "")
            if name:
                known_players.add(name.lower())
    except Exception:
        pass

    # Fallback: if no predictions, use a broad set
    if not known_players:
        try:
            players = sb.table("players").select("name").limit(300).execute()
            for p in (players.data or []):
                name = p.get("name", "")
                if name:
                    known_players.add(name.lower())
        except Exception:
            pass

    # Fetch articles
    rss_articles = await loop.run_in_executor(None, news_svc.fetch_rss_articles, 48)
    newsapi_articles = await loop.run_in_executor(None, news_svc.fetch_newsapi_articles, 48)
    all_articles = rss_articles + newsapi_articles

    # Build per-player sentiment
    player_sentiment = await loop.run_in_executor(
        None, news_svc.build_player_sentiment, known_players, 48
    )

    # Filter to only players with mentions
    mentioned = {
        k: v for k, v in player_sentiment.items()
        if v.get("news_volume", 0) > 0
    }

    return {
        "articles": all_articles[:100],
        "total_articles": len(all_articles),
        "player_sentiment": mentioned,
        "players_mentioned": len(mentioned),
        "sources": {
            "rss": len(rss_articles),
            "newsapi": len(newsapi_articles),
        },
    }


@router.get("/correlations")
async def get_player_correlations():
    """
    Compute and return player prop correlations from BDL cached data.
    Used by the parlay builder to show data-backed correlated parlays.
    """
    import asyncio
    import json
    from datetime import date
    from pathlib import Path

    from app.services.correlation_engine import build_all_player_correlations, find_correlated_parlays
    from app.services.supabase_client import get_supabase

    loop = asyncio.get_event_loop()
    cache_dir = Path(__file__).parent.parent / "cache"

    # Load BDL cached game logs
    all_game_logs: dict[int, list[dict]] = {}
    bdl_id_to_name: dict[int, str] = {}

    for season in [2025, 2024]:
        path = cache_dir / f"bdl_season_{season}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for row in data:
            player = row.get("player", {})
            pid = player.get("id")
            if not pid:
                continue
            name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip().lower()
            bdl_id_to_name[pid] = name
            all_game_logs.setdefault(pid, []).append(row)

    # Compute correlations
    player_correlations = await loop.run_in_executor(
        None, build_all_player_correlations, all_game_logs, bdl_id_to_name
    )

    # Get today's predictions for correlated parlay suggestions
    sb = get_supabase()
    today_str = date.today().isoformat()
    predictions = []
    try:
        preds = sb.table("predictions").select(
            "*, players(name)"
        ).gte("created_at", today_str).execute()
        for p in (preds.data or []):
            player_info = p.get("players") or {}
            predictions.append({
                **p,
                "player_name": player_info.get("name", ""),
            })
    except Exception:
        pass

    # Find correlated parlays
    correlated_parlays = find_correlated_parlays(
        player_correlations, predictions, min_correlation=0.25, min_confidence=50.0
    )

    return {
        "players_with_correlations": len(player_correlations),
        "correlated_parlays": correlated_parlays[:20],
        "total_suggestions": len(correlated_parlays),
    }


@router.get("/export/{format}")
async def export_data(format: str):
    """Export predictions and bets as a downloadable CSV or JSON file."""
    from datetime import date

    from app.services.supabase_client import get_supabase

    sb = get_supabase()
    today_str = date.today().isoformat()

    # Fetch predictions with player names
    try:
        preds = sb.table("predictions").select(
            "*, players(name, team_id)"
        ).order("created_at", desc=True).limit(2000).execute()
        predictions = preds.data or []
    except Exception:
        predictions = []

    # Fetch bets
    try:
        bets_result = sb.table("bets").select("*").order(
            "created_at", desc=True
        ).limit(1000).execute()
        bets = bets_result.data or []
    except Exception:
        bets = []

    if format == "csv":
        output = io.StringIO()
        if predictions:
            # Flatten predictions for CSV
            rows = []
            for p in predictions:
                player_info = p.get("players") or {}
                rows.append({
                    "player": player_info.get("name", ""),
                    "team": player_info.get("team_id", ""),
                    "prop_type": p.get("prop_type", ""),
                    "line": p.get("line", ""),
                    "predicted_value": p.get("predicted_value", ""),
                    "recommended_bet": p.get("recommended_bet", ""),
                    "confidence_score": p.get("confidence_score", ""),
                    "edge_pct": p.get("edge_pct", ""),
                    "expected_value": p.get("expected_value", ""),
                    "best_book": p.get("best_book", ""),
                    "best_odds": p.get("best_odds", ""),
                    "kelly_bet_size": p.get("kelly_bet_size", ""),
                    "over_probability": p.get("over_probability", ""),
                    "created_at": p.get("created_at", ""),
                })
            if rows:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=propsai_export_{today_str}.csv"},
        )

    elif format == "json":
        export_data_dict = {
            "exported_at": today_str,
            "predictions": predictions,
            "bets": bets,
            "total_predictions": len(predictions),
            "total_bets": len(bets),
        }
        content = json_lib.dumps(export_data_dict, indent=2, default=str)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=propsai_export_{today_str}.json"},
        )

    else:
        return {"status": "error", "message": f"Unsupported format: {format}. Use 'csv' or 'json'."}
