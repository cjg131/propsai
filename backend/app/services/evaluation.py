"""
Prediction Evaluation & Bayesian Updating Service.

After games finish, this service:
1. Fetches actual box scores from BallDontLie
2. Compares predictions vs actuals
3. Computes hit rates, RMSE, calibration metrics
4. Stores results in Supabase for the Performance page
5. Adjusts model confidence calibration over time
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from app.logging_config import get_logger

logger = get_logger(__name__)


def _parse_min(m: str) -> float:
    if not m or m in ("0", "00", ""):
        return 0.0
    try:
        if ":" in str(m):
            p = str(m).split(":")
            return float(p[0]) + float(p[1]) / 60
        return float(m)
    except (ValueError, IndexError):
        return 0.0


# Map our prop types to BDL stat keys
PROP_TO_BDL = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "turnover",
}


async def evaluate_predictions(target_date: date | None = None) -> dict:
    """
    Evaluate yesterday's predictions against actual results.

    Fetches actual box scores from BDL, matches them to predictions
    in Supabase, and computes accuracy metrics.
    """
    import asyncio

    from app.services.balldontlie import get_balldontlie
    from app.services.supabase_client import get_supabase

    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    date_str = target_date.isoformat()
    sb = get_supabase()
    bdl = get_balldontlie()
    loop = asyncio.get_event_loop()

    logger.info(f"Evaluating predictions for {date_str}...")

    # 1. Fetch predictions for that date
    try:
        pred_result = sb.table("predictions").select("*").gte(
            "created_at", date_str
        ).lt(
            "created_at", (target_date + timedelta(days=1)).isoformat()
        ).execute()
        predictions = pred_result.data or []
    except Exception as e:
        logger.error(f"Failed to fetch predictions: {e}")
        return {"status": "error", "message": str(e)}

    if not predictions:
        return {"status": "no_predictions", "date": date_str}

    logger.info(f"Found {len(predictions)} predictions for {date_str}")

    # 2. Fetch actual box scores from BDL for that date
    actual_stats = await loop.run_in_executor(
        None, bdl.get_team_game_logs_by_date, date_str
    )

    # Filter DNP games
    actual_stats = [s for s in actual_stats if _parse_min(s.get("min", "0")) > 0]
    logger.info(f"Fetched {len(actual_stats)} actual box scores for {date_str}")

    if not actual_stats:
        return {"status": "no_actuals", "date": date_str}

    # 3. Build player name → actual stats lookup
    actual_by_name: dict[str, dict] = {}
    for s in actual_stats:
        p = s.get("player", {})
        name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip().lower()
        if name:
            actual_by_name[name] = s

    # 4. Fetch player names from Supabase
    player_ids = list(set(p.get("player_id") for p in predictions if p.get("player_id")))
    player_names: dict[str, str] = {}
    if player_ids:
        try:
            pr = sb.table("players").select("id, name").in_("id", player_ids).execute()
            for p in (pr.data or []):
                player_names[p["id"]] = p["name"].lower()
        except Exception:
            pass

    # 5. Match predictions to actuals and compute metrics
    results = []
    hits = 0
    misses = 0
    total_error = 0.0
    total_sq_error = 0.0
    by_prop: dict[str, dict] = defaultdict(lambda: {"hits": 0, "misses": 0, "errors": []})
    by_confidence: dict[str, dict] = defaultdict(lambda: {"hits": 0, "total": 0})

    for pred in predictions:
        pid = pred.get("player_id", "")
        pname = player_names.get(pid, "")
        prop_type = pred.get("prop_type", "")
        predicted = pred.get("predicted_value", 0)
        line = pred.get("line", 0)
        bet = pred.get("recommended_bet", "")
        confidence = pred.get("confidence_score", 50)
        edge = pred.get("edge_pct", 0)

        actual_data = actual_by_name.get(pname)
        if not actual_data:
            continue

        bdl_key = PROP_TO_BDL.get(prop_type)
        if not bdl_key:
            continue

        actual_value = actual_data.get(bdl_key, 0) or 0

        # Did the bet hit?
        if bet == "over":
            hit = actual_value > line
        elif bet == "under":
            hit = actual_value < line
        else:
            hit = False

        error = abs(predicted - actual_value)
        total_error += error
        total_sq_error += error ** 2

        if hit:
            hits += 1
        else:
            misses += 1

        # By prop type
        prop_stats = by_prop[prop_type]
        if hit:
            prop_stats["hits"] += 1
        else:
            prop_stats["misses"] += 1
        prop_stats["errors"].append(error)

        # By confidence tier
        if confidence >= 75:
            tier = "high"
        elif confidence >= 60:
            tier = "medium"
        else:
            tier = "low"
        by_confidence[tier]["total"] += 1
        if hit:
            by_confidence[tier]["hits"] += 1

        results.append({
            "player_name": pname,
            "prop_type": prop_type,
            "predicted": round(predicted, 1),
            "actual": actual_value,
            "line": line,
            "bet": bet,
            "hit": hit,
            "error": round(error, 1),
            "confidence": round(confidence, 1),
            "edge": round(edge, 1),
        })

    total = hits + misses
    if total == 0:
        return {"status": "no_matches", "date": date_str, "predictions": len(predictions), "actuals": len(actual_stats)}

    import math
    rmse = math.sqrt(total_sq_error / total)
    mae = total_error / total
    hit_rate = hits / total

    # Prop-level metrics
    prop_metrics = {}
    for pt, stats in by_prop.items():
        pt_total = stats["hits"] + stats["misses"]
        pt_rmse = math.sqrt(sum(e**2 for e in stats["errors"]) / max(len(stats["errors"]), 1))
        prop_metrics[pt] = {
            "hit_rate": round(stats["hits"] / max(pt_total, 1), 3),
            "total": pt_total,
            "rmse": round(pt_rmse, 2),
        }

    # Confidence calibration
    calibration = {}
    for tier, stats in by_confidence.items():
        calibration[tier] = {
            "hit_rate": round(stats["hits"] / max(stats["total"], 1), 3),
            "total": stats["total"],
        }

    summary = {
        "status": "completed",
        "date": date_str,
        "total_evaluated": total,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hit_rate, 3),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "by_prop_type": prop_metrics,
        "calibration_by_confidence": calibration,
        "sample_results": results[:10],
    }

    logger.info(
        f"Evaluation for {date_str}: {total} predictions, "
        f"{hit_rate:.1%} hit rate, RMSE={rmse:.2f}, MAE={mae:.2f}"
    )

    # 6. Store evaluation results in Supabase
    try:
        sb.table("prediction_evaluations").upsert({
            "evaluation_date": date_str,
            "total_predictions": total,
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hit_rate, 4),
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "by_prop_type": prop_metrics,
            "calibration": calibration,
        }).execute()
    except Exception as e:
        # Table might not exist yet — that's fine
        logger.debug(f"Could not store evaluation: {e}")

    return summary
