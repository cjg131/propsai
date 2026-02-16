"""
Line Movement Tracking & CLV (Closing Line Value) Service.

Tracks how prop lines move from open → close, computes CLV%,
detects steam moves, and provides line movement as ML features.

CLV is the #1 metric sharp bettors use to prove long-term edge.
If your picks consistently beat the closing line, you have real edge.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta

from app.logging_config import get_logger

logger = get_logger(__name__)


# ── No-Vig Fair Odds Engine ──────────────────────────────────────────

def american_to_implied(odds: int | float) -> float:
    """Convert American odds to implied probability (0-1)."""
    odds = float(odds)
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def implied_to_american(prob: float) -> int:
    """Convert implied probability (0-1) to American odds."""
    if prob <= 0 or prob >= 1:
        return -110
    if prob >= 0.5:
        return round(-100 * prob / (1 - prob))
    else:
        return round(100 * (1 - prob) / prob)


def remove_vig(over_odds: int | float, under_odds: int | float) -> dict:
    """
    Remove the sportsbook's vig (hold) to get true fair probabilities.

    Example: Over -110 / Under -110
      Implied: 52.38% + 52.38% = 104.76% (4.76% hold)
      No-vig:  50.0% / 50.0% (fair)

    Example: Over -130 / Under +110
      Implied: 56.52% + 47.62% = 104.14%
      No-vig:  54.27% / 45.73%

    Returns: {
        "over_implied": 0.5238,
        "under_implied": 0.5238,
        "total_implied": 1.0476,
        "hold_pct": 4.76,
        "fair_over_prob": 0.50,
        "fair_under_prob": 0.50,
        "fair_over_odds": -100,
        "fair_under_odds": +100,
    }
    """
    over_imp = american_to_implied(over_odds)
    under_imp = american_to_implied(under_odds)
    total = over_imp + under_imp
    hold = (total - 1.0) * 100

    # Remove vig proportionally (multiplicative method)
    if total > 0:
        fair_over = over_imp / total
        fair_under = under_imp / total
    else:
        fair_over = 0.5
        fair_under = 0.5

    return {
        "over_implied": round(over_imp, 4),
        "under_implied": round(under_imp, 4),
        "total_implied": round(total, 4),
        "hold_pct": round(hold, 2),
        "fair_over_prob": round(fair_over, 4),
        "fair_under_prob": round(fair_under, 4),
        "fair_over_odds": implied_to_american(fair_over),
        "fair_under_odds": implied_to_american(fair_under),
    }


def compute_true_ev(
    model_prob: float,
    fair_prob: float,
    american_odds: int | float,
) -> dict:
    """
    Compute true Expected Value accounting for vig.

    model_prob: Our model's probability of the bet winning
    fair_prob: No-vig fair probability from the market
    american_odds: The actual odds we'd bet at (with vig)

    Returns: {
        "true_ev_pct": 5.2,       # True EV after accounting for vig
        "edge_vs_fair": 3.1,      # Our edge vs the fair line (no-vig)
        "edge_vs_market": 7.8,    # Our edge vs the market (with vig)
        "kelly_fraction": 0.052,  # Optimal Kelly bet size
    }
    """
    if american_odds > 0:
        decimal_odds = 1 + american_odds / 100.0
    else:
        decimal_odds = 1 + 100.0 / abs(american_odds)

    # True EV = (prob × payout) - (1 - prob) × stake
    ev = model_prob * (decimal_odds - 1) - (1 - model_prob)
    edge_vs_fair = (model_prob - fair_prob) * 100
    market_implied = american_to_implied(american_odds)
    edge_vs_market = (model_prob - market_implied) * 100

    # Kelly criterion: f* = (bp - q) / b where b = decimal_odds - 1
    b = decimal_odds - 1
    if b > 0:
        kelly = (b * model_prob - (1 - model_prob)) / b
        kelly = max(kelly, 0)
    else:
        kelly = 0

    return {
        "true_ev_pct": round(ev * 100, 2),
        "edge_vs_fair": round(edge_vs_fair, 2),
        "edge_vs_market": round(edge_vs_market, 2),
        "kelly_fraction": round(kelly, 4),
    }


# ── CLV Tracking ─────────────────────────────────────────────────────

def compute_clv(
    opening_line: float,
    closing_line: float,
    bet_direction: str,
) -> dict:
    """
    Compute Closing Line Value.

    If we bet Over 23.5 and the line closes at 22.5, the market moved
    toward our position → positive CLV.

    Returns: {
        "clv_points": 1.0,    # Line moved 1 point in our favor
        "clv_pct": 4.26,      # CLV as percentage
        "market_agrees": True, # Market moved toward our pick
    }
    """
    if bet_direction == "over":
        # For overs, line dropping = market agrees with us
        clv_points = opening_line - closing_line
    elif bet_direction == "under":
        # For unders, line rising = market agrees with us
        clv_points = closing_line - opening_line
    else:
        clv_points = 0

    clv_pct = (clv_points / max(opening_line, 0.5)) * 100

    return {
        "clv_points": round(clv_points, 2),
        "clv_pct": round(clv_pct, 2),
        "market_agrees": clv_points > 0,
    }


# ── Line Movement Features for ML ────────────────────────────────────

def compute_line_movement_features(snapshots: list[dict]) -> dict:
    """
    From a series of timestamped line snapshots, compute ML features:
    - Total movement (open to latest)
    - Movement velocity (points per hour)
    - Direction consistency (did it move steadily or oscillate?)
    - Steam move flag (>1 point in <30 min)

    snapshots: [{"line": 23.5, "timestamp": "2025-01-15T10:00:00", "over_odds": -110, "under_odds": -110}, ...]
    """
    if len(snapshots) < 2:
        return {
            "line_movement": 0,
            "movement_velocity": 0,
            "movement_direction": 0,
            "is_steam_move": 0,
            "vig_change": 0,
        }

    # Sort by timestamp
    sorted_snaps = sorted(snapshots, key=lambda s: s.get("timestamp", ""))
    first = sorted_snaps[0]
    last = sorted_snaps[-1]

    open_line = first.get("line", 0)
    close_line = last.get("line", 0)
    total_movement = close_line - open_line

    # Time span
    try:
        t0 = datetime.fromisoformat(first["timestamp"].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))
        hours = max((t1 - t0).total_seconds() / 3600, 0.01)
    except (ValueError, KeyError):
        hours = 1.0

    velocity = total_movement / hours

    # Direction consistency: count how many consecutive moves are in the same direction
    moves = []
    for i in range(1, len(sorted_snaps)):
        diff = sorted_snaps[i].get("line", 0) - sorted_snaps[i - 1].get("line", 0)
        if diff != 0:
            moves.append(1 if diff > 0 else -1)

    if moves:
        direction = sum(moves) / len(moves)  # -1 to +1
    else:
        direction = 0

    # Steam move detection: >0.5 point move between any two consecutive snapshots
    # within 30 minutes
    is_steam = 0
    for i in range(1, len(sorted_snaps)):
        diff = abs(sorted_snaps[i].get("line", 0) - sorted_snaps[i - 1].get("line", 0))
        if diff >= 0.5:
            try:
                t_prev = datetime.fromisoformat(sorted_snaps[i - 1]["timestamp"].replace("Z", "+00:00"))
                t_curr = datetime.fromisoformat(sorted_snaps[i]["timestamp"].replace("Z", "+00:00"))
                mins = (t_curr - t_prev).total_seconds() / 60
                if mins <= 60 and diff >= 0.5:
                    is_steam = 1
                    break
            except (ValueError, KeyError):
                pass

    # Vig change (hold increasing = books getting more confident)
    open_vig = remove_vig(
        first.get("over_odds", -110), first.get("under_odds", -110)
    ).get("hold_pct", 4.5)
    close_vig = remove_vig(
        last.get("over_odds", -110), last.get("under_odds", -110)
    ).get("hold_pct", 4.5)

    return {
        "line_movement": round(total_movement, 2),
        "movement_velocity": round(velocity, 3),
        "movement_direction": round(direction, 2),
        "is_steam_move": is_steam,
        "vig_change": round(close_vig - open_vig, 2),
    }


# ── Supabase Integration ─────────────────────────────────────────────

async def snapshot_current_lines() -> int:
    """
    Take a timestamped snapshot of all current prop lines.
    Stores in line_snapshots table for CLV tracking and movement analysis.
    Returns count of snapshots stored.
    """
    from app.services.supabase_client import get_supabase
    sb = get_supabase()

    try:
        # Fetch current prop lines
        result = sb.table("prop_lines").select("*").execute()
        lines = result.data or []

        if not lines:
            return 0

        now = datetime.utcnow().isoformat()
        snapshots = []
        for line in lines:
            snapshots.append({
                "player_id": line.get("player_id"),
                "game_id": line.get("game_id"),
                "prop_type": line.get("prop_type"),
                "sportsbook": line.get("sportsbook"),
                "line": line.get("line"),
                "over_odds": line.get("over_odds"),
                "under_odds": line.get("under_odds"),
                "snapshot_time": now,
            })

        # Batch insert
        count = 0
        for i in range(0, len(snapshots), 200):
            chunk = snapshots[i:i + 200]
            try:
                sb.table("line_snapshots").insert(chunk).execute()
                count += len(chunk)
            except Exception as e:
                logger.warning(f"Line snapshot insert error: {e}")

        logger.info(f"Stored {count} line snapshots at {now}")
        return count

    except Exception as e:
        logger.error(f"Failed to snapshot lines: {e}")
        return 0


async def compute_clv_for_date(target_date: date | None = None) -> dict:
    """
    Compute CLV for all predictions on a given date.
    Compares the line at prediction time vs the last snapshot before game time.
    """
    from app.services.supabase_client import get_supabase
    sb = get_supabase()

    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    date_str = target_date.isoformat()
    next_date = (target_date + timedelta(days=1)).isoformat()

    try:
        # Get predictions for that date
        preds = sb.table("predictions").select("*").gte(
            "created_at", date_str
        ).lt("created_at", next_date).execute()

        if not preds.data:
            return {"status": "no_predictions", "date": date_str}

        # Get all line snapshots for that date
        snaps = sb.table("line_snapshots").select("*").gte(
            "snapshot_time", date_str
        ).lt("snapshot_time", next_date).execute()

        snap_data = snaps.data or []

        # Group snapshots by player_id|prop_type|sportsbook
        snap_groups: dict[str, list[dict]] = defaultdict(list)
        for s in snap_data:
            key = f"{s.get('player_id')}|{s.get('prop_type')}|{s.get('sportsbook')}"
            snap_groups[key].append(s)

        results = []
        total_clv = 0.0
        clv_count = 0

        for pred in preds.data:
            pid = pred.get("player_id", "")
            prop_type = pred.get("prop_type", "")
            opening_line = pred.get("line", 0)
            bet = pred.get("recommended_bet", "")

            # Find the latest snapshot for this player+prop
            best_closing = None
            for book in ["DraftKings", "FanDuel"]:
                key = f"{pid}|{prop_type}|{book}"
                group = snap_groups.get(key, [])
                if group:
                    latest = max(group, key=lambda s: s.get("snapshot_time", ""))
                    if best_closing is None or latest.get("snapshot_time", "") > best_closing.get("snapshot_time", ""):
                        best_closing = latest

            if best_closing and bet:
                closing_line = best_closing.get("line", opening_line)
                clv = compute_clv(opening_line, closing_line, bet)
                total_clv += clv["clv_pct"]
                clv_count += 1
                results.append({
                    "player_id": pid,
                    "prop_type": prop_type,
                    "opening_line": opening_line,
                    "closing_line": closing_line,
                    "bet": bet,
                    **clv,
                })

        avg_clv = total_clv / max(clv_count, 1)
        positive_clv = sum(1 for r in results if r.get("clv_pct", 0) > 0)

        return {
            "status": "completed",
            "date": date_str,
            "total_evaluated": clv_count,
            "avg_clv_pct": round(avg_clv, 2),
            "positive_clv_count": positive_clv,
            "positive_clv_pct": round(positive_clv / max(clv_count, 1) * 100, 1),
            "results": results[:20],
        }

    except Exception as e:
        logger.error(f"CLV computation failed: {e}")
        return {"status": "error", "message": str(e)}
