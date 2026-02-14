"""
Paper Trading Service — Virtual Bet Tracking.

Automatically places virtual bets on high-confidence predictions,
tracks them in Supabase, and resolves them when game results come in.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timedelta
from typing import Optional

from app.logging_config import get_logger
from app.services.supabase_client import get_supabase

logger = get_logger(__name__)

# Kelly fraction for paper trading (half-Kelly)
PAPER_KELLY_FRACTION = 0.5
DEFAULT_STARTING_BANKROLL = 1000.0
MIN_CONFIDENCE = 60.0


async def get_paper_trading_status() -> dict:
    """Get current paper trading session status and summary."""
    sb = get_supabase()

    try:
        # Get all paper trades
        result = sb.table("paper_trades").select("*").order(
            "created_at", desc=True
        ).execute()
        trades = result.data or []
    except Exception as e:
        logger.error(f"Failed to load paper trades: {e}")
        return {
            "status": "error",
            "message": str(e),
            "has_table": False,
        }

    if not trades:
        return {
            "status": "ok",
            "has_table": True,
            "total_trades": 0,
            "pending": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "hit_rate": 0,
            "total_profit": 0,
            "current_bankroll": DEFAULT_STARTING_BANKROLL,
            "starting_bankroll": DEFAULT_STARTING_BANKROLL,
            "roi": 0,
            "today_trades": 0,
            "today_pending": 0,
            "recent_trades": [],
        }

    wins = sum(1 for t in trades if t.get("result") == "win")
    losses = sum(1 for t in trades if t.get("result") == "loss")
    pushes = sum(1 for t in trades if t.get("result") == "push")
    pending = sum(1 for t in trades if t.get("result") == "pending")
    resolved = wins + losses
    hit_rate = wins / resolved if resolved > 0 else 0
    total_profit = sum(float(t.get("profit", 0) or 0) for t in trades)
    total_wagered = sum(float(t.get("stake", 0) or 0) for t in trades if t.get("result") != "pending")
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    # Current bankroll = starting + total profit
    current_bankroll = DEFAULT_STARTING_BANKROLL + total_profit

    # Today's trades
    today_str = date.today().isoformat()
    today_trades = [t for t in trades if str(t.get("game_date", "")) == today_str]

    # Streak
    resolved_trades = [t for t in trades if t.get("result") in ("win", "loss")]
    resolved_trades.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    streak = 0
    streak_type = "none"
    if resolved_trades:
        streak_type = resolved_trades[0].get("result", "none")
        for t in resolved_trades:
            if t.get("result") == streak_type:
                streak += 1
            else:
                break

    return {
        "status": "ok",
        "has_table": True,
        "total_trades": len(trades),
        "pending": pending,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": round(hit_rate, 4),
        "total_profit": round(total_profit, 2),
        "total_wagered": round(total_wagered, 2),
        "current_bankroll": round(current_bankroll, 2),
        "starting_bankroll": DEFAULT_STARTING_BANKROLL,
        "roi": round(roi, 2),
        "today_trades": len(today_trades),
        "today_pending": sum(1 for t in today_trades if t.get("result") == "pending"),
        "streak": streak,
        "streak_type": streak_type,
        "recent_trades": trades[:50],
    }


async def place_paper_trades(
    min_confidence: float = MIN_CONFIDENCE,
    target_date: Optional[date] = None,
) -> dict:
    """
    Auto-place virtual bets on high-confidence predictions.
    Uses predictions already in the DB from /api/predictions/generate.
    If target_date is provided, places bets on that date's predictions.
    """
    sb = get_supabase()

    # Get current bankroll
    status = await get_paper_trading_status()
    if status.get("status") == "error":
        return status
    current_bankroll = status.get("current_bankroll", DEFAULT_STARTING_BANKROLL)

    if current_bankroll <= 0:
        return {"status": "error", "message": "Paper bankroll is depleted"}

    # Load predictions for the target date
    # Predictions may have been generated late at night (UTC), so their
    # created_at can be the next calendar day.  We use a wide window:
    # from the target date at midnight to the next day at 23:59 UTC,
    # which covers predictions generated the evening before or early
    # the next morning.  If no target_date, mirror the "today" endpoint
    # logic (gte today).
    bet_date = target_date or date.today()
    bet_date_str = bet_date.isoformat()
    window_start = bet_date_str  # e.g. 2026-02-12
    window_end = (bet_date + timedelta(days=2)).isoformat()  # e.g. 2026-02-14
    try:
        pred_result = sb.table("predictions").select(
            "*"
        ).gte("created_at", window_start).lt("created_at", window_end).execute()
        predictions = pred_result.data or []
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        return {"status": "error", "message": f"Failed to load predictions: {e}"}

    if not predictions:
        return {"status": "ok", "message": f"No predictions found for {bet_date_str}", "trades_placed": 0}

    # Deduplicate: keep only the most recent batch of predictions.
    # All predictions in a single generate run share the same created_at
    # (within seconds).  Pick the latest created_at date and keep only those.
    latest_date = max(p.get("created_at", "") for p in predictions)[:10]
    predictions = [p for p in predictions if (p.get("created_at", "") or "")[:10] == latest_date]

    # Build player name + team lookup (predictions table doesn't store player_name)
    player_ids = list(set(str(p.get("player_id", "")) for p in predictions if p.get("player_id")))
    players_map: dict[str, dict] = {}
    teams_map: dict[str, str] = {}
    try:
        pr = sb.table("players").select("id, name, team_id").execute()
        for p in (pr.data or []):
            players_map[str(p["id"])] = p
        tr = sb.table("teams").select("id, abbreviation").execute()
        for t in (tr.data or []):
            teams_map[str(t["id"])] = t["abbreviation"]
    except Exception as e:
        logger.warning(f"Failed to load player/team lookups: {e}")

    # Check which predictions already have paper trades
    try:
        existing = sb.table("paper_trades").select("prediction_id").execute()
        existing_ids = set(str(t.get("prediction_id")) for t in (existing.data or []))
    except Exception:
        existing_ids = set()

    session_id = f"session-{bet_date_str}-{uuid.uuid4().hex[:8]}"
    trades_to_insert = []
    running_bankroll = current_bankroll

    for pred in predictions:
        pred_id = pred.get("id", "")
        if str(pred_id) in existing_ids:
            continue

        confidence = float(pred.get("confidence_score", 0) or 0)
        if confidence < min_confidence:
            continue

        recommended_bet = pred.get("recommended_bet", "")
        if not recommended_bet:
            continue

        line = float(pred.get("line", 0) or 0)
        if line <= 0:
            continue

        predicted_value = float(pred.get("predicted_value", 0) or 0)
        over_prob = float(pred.get("over_probability", 0.5) or 0.5)

        # Kelly sizing
        bet_prob = over_prob if recommended_bet == "over" else (1.0 - over_prob)
        decimal_odds = 1.909  # -110
        b = decimal_odds - 1
        kelly = (b * bet_prob - (1 - bet_prob)) / b
        kelly = max(kelly, 0) * PAPER_KELLY_FRACTION
        if kelly <= 0:
            continue

        stake = min(running_bankroll * kelly, running_bankroll * 0.03)
        stake = min(stake, DEFAULT_STARTING_BANKROLL * 0.10)
        stake = round(stake, 2)
        if stake < 1:
            continue

        pid = str(pred.get("player_id", ""))
        player_data = players_map.get(pid, {})
        player_name = player_data.get("name", "Unknown")
        team_id = str(player_data.get("team_id", ""))
        team_abbr = teams_map.get(team_id, "")

        trades_to_insert.append({
            "prediction_id": str(pred_id),
            "player_name": player_name,
            "player_id": pid,
            "team": team_abbr,
            "opponent": pred.get("opponent", ""),
            "game_date": bet_date_str,
            "prop_type": pred.get("prop_type", ""),
            "line": line,
            "predicted_value": predicted_value,
            "recommended_bet": recommended_bet,
            "confidence_score": confidence,
            "odds": -110,
            "stake": stake,
            "result": "pending",
            "profit": 0,
            "bankroll_after": round(running_bankroll, 2),
            "session_id": session_id,
        })

    if not trades_to_insert:
        return {"status": "ok", "message": "No new qualifying trades", "trades_placed": 0}

    # Insert trades
    try:
        sb.table("paper_trades").insert(trades_to_insert).execute()
    except Exception as e:
        logger.error(f"Failed to insert paper trades: {e}")
        return {"status": "error", "message": f"Failed to insert: {e}"}

    logger.info(f"Paper trading: placed {len(trades_to_insert)} virtual bets")
    return {
        "status": "ok",
        "trades_placed": len(trades_to_insert),
        "session_id": session_id,
        "total_staked": round(sum(t["stake"] for t in trades_to_insert), 2),
    }


async def resolve_paper_trades(target_date: Optional[date] = None) -> dict:
    """
    Resolve pending paper trades by comparing to actual game results.
    Uses BDL cached data for actuals.
    """
    import json
    from pathlib import Path

    sb = get_supabase()
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    target_str = target_date.isoformat()

    # Load pending trades for this date
    try:
        result = sb.table("paper_trades").select("*").eq(
            "game_date", target_str
        ).eq("result", "pending").execute()
        pending_trades = result.data or []
    except Exception as e:
        return {"status": "error", "message": f"Failed to load pending trades: {e}"}

    if not pending_trades:
        return {"status": "ok", "message": "No pending trades to resolve", "resolved": 0}

    # Load BDL data for actuals
    cache_dir = Path(__file__).parent.parent / "cache"
    bdl_key_map = {
        "points": "pts",
        "rebounds": "reb",
        "assists": "ast",
        "threes": "fg3m",
        "steals": "stl",
        "blocks": "blk",
        "turnovers": "turnover",
    }

    def _extract_actuals(rows: list[dict], target: str) -> dict[str, dict[str, float]]:
        """Build player -> stat dict from BDL rows for a given date."""
        result: dict[str, dict[str, float]] = {}
        for row in rows:
            game = row.get("game", {})
            if game.get("date") != target:
                continue
            player = row.get("player", {})
            name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
            key = name.lower()
            result[key] = {
                "pts": row.get("pts", 0) or 0,
                "reb": row.get("reb", 0) or 0,
                "ast": row.get("ast", 0) or 0,
                "fg3m": row.get("fg3m", 0) or 0,
                "stl": row.get("stl", 0) or 0,
                "blk": row.get("blk", 0) or 0,
                "turnover": row.get("turnover", 0) or 0,
            }
        return result

    # Try 1: Check BDL file cache
    actuals: dict[str, dict[str, float]] = {}
    for season in [2025, 2024]:
        path = cache_dir / f"bdl_season_{season}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        actuals.update(_extract_actuals(data, target_str))

    # Try 2: If cache didn't have this date, fetch live from BDL API
    if not actuals:
        logger.info(f"Paper trading: no cached actuals for {target_str}, fetching live from BDL API...")
        try:
            from app.services.balldontlie import get_balldontlie
            bdl = get_balldontlie()
            live_rows = bdl.get_team_game_logs_by_date(target_str)
            if live_rows:
                logger.info(f"Paper trading: fetched {len(live_rows)} live box scores for {target_str}")
                actuals = _extract_actuals(live_rows, target_str)
        except Exception as e:
            logger.warning(f"Paper trading: live BDL fetch failed: {e}")

    resolved_count = 0
    wins = 0
    losses = 0
    total_profit = 0.0

    for trade in pending_trades:
        player_name = trade.get("player_name", "")
        prop_type = trade.get("prop_type", "")
        line = float(trade.get("line", 0))
        recommended_bet = trade.get("recommended_bet", "")
        stake = float(trade.get("stake", 0))

        bdl_key = bdl_key_map.get(prop_type)
        if not bdl_key:
            continue

        player_key = player_name.lower()
        player_actuals = actuals.get(player_key)
        if not player_actuals:
            continue

        actual_value = player_actuals.get(bdl_key, 0)

        # Determine result
        if actual_value == line:
            result_str = "push"
            profit = 0.0
        elif recommended_bet == "over":
            if actual_value > line:
                result_str = "win"
                profit = round(stake * 0.909, 2)
            else:
                result_str = "loss"
                profit = round(-stake, 2)
        else:  # under
            if actual_value < line:
                result_str = "win"
                profit = round(stake * 0.909, 2)
            else:
                result_str = "loss"
                profit = round(-stake, 2)

        # Update trade
        try:
            sb.table("paper_trades").update({
                "actual_value": actual_value,
                "result": result_str,
                "profit": profit,
                "resolved_at": datetime.utcnow().isoformat(),
            }).eq("id", trade["id"]).execute()

            resolved_count += 1
            total_profit += profit
            if result_str == "win":
                wins += 1
            elif result_str == "loss":
                losses += 1
        except Exception as e:
            logger.error(f"Failed to update trade {trade['id']}: {e}")

    return {
        "status": "ok",
        "resolved": resolved_count,
        "wins": wins,
        "losses": losses,
        "profit": round(total_profit, 2),
        "date": target_str,
    }


async def reset_paper_trading() -> dict:
    """Reset all paper trades (start fresh)."""
    sb = get_supabase()
    try:
        # Delete all paper trades
        sb.table("paper_trades").delete().neq("id", 0).execute()
        return {"status": "ok", "message": "Paper trading reset successfully"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
