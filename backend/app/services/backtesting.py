"""
Backtesting Engine — Historical Simulation.

Replays the SmartPredictor model against historical game data to measure
real predictive accuracy and simulated betting performance.

Uses BDL cached game logs (real per-game box scores) as the data source.

Approach:
  1. Load BDL cached game logs for the 2025 season
  2. For each game date, build the feature set the model would have had
     at that point in time (only using data BEFORE that date)
  3. Generate predictions using the trained SmartPredictor
  4. Compare predictions to actual box scores
  5. Simulate betting with configurable Kelly sizing
  6. Return equity curve, hit rates, ROI, and per-bet log
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from collections import defaultdict
from pathlib import Path

from app.logging_config import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"

# Map prop types to BDL stat keys
PROP_TO_BDL = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "turnover",
}

# Map prop types to PROP_STAT_MAP keys (for _build_feature_row)
PROP_TO_PG_KEY = {
    "points": "pts_pg",
    "rebounds": "reb_pg",
    "assists": "ast_pg",
    "threes": "three_pm_pg",
    "steals": "stl_pg",
    "blocks": "blk_pg",
    "turnovers": "tov_pg",
}


def _parse_min(m) -> float:
    if not m or m in ("0", "00", ""):
        return 0.0
    try:
        if ":" in str(m):
            p = str(m).split(":")
            return float(p[0]) + float(p[1]) / 60
        return float(m)
    except (ValueError, IndexError):
        return 0.0


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    mean = _avg(vals)
    return math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))


def _load_bdl_season(season: int) -> list[dict]:
    """Load BDL cached game logs for a season."""
    path = CACHE_DIR / f"bdl_season_{season}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _build_backtest_player(
    prior_games: list[dict],
    prop_type: str,
    current_game: dict,
    player_name: str,
) -> dict:
    """
    Build a player feature dict for backtesting, using only data
    available BEFORE the target game date.
    prior_games: list of BDL box score dicts (already filtered to before current game)
    """
    bdl_key = PROP_TO_BDL.get(prop_type, "pts")

    if not prior_games:
        return {}

    def stat_vals(key: str) -> list[float]:
        return [float(g.get(key, 0) or 0) for g in prior_games]

    def min_vals() -> list[float]:
        return [_parse_min(g.get("min", "0")) for g in prior_games]

    avg_stat = _avg(stat_vals(bdl_key))
    mins = min_vals()
    avg_min = _avg(mins)

    # Rolling averages
    def rolling(key: str, n: int) -> float:
        vals = stat_vals(key)
        return _avg(vals[-n:]) if vals else 0.0

    def rolling_min(n: int) -> float:
        return _avg(mins[-n:]) if mins else 0.0

    last3 = rolling(bdl_key, 3)
    last5 = rolling(bdl_key, 5)
    last10 = rolling(bdl_key, 10)
    std_stat = _std(stat_vals(bdl_key))

    last3_min = rolling_min(3)
    last5_min = rolling_min(5)

    # Home/away splits
    is_home = current_game.get("is_home", False)
    home_games = [g for g in prior_games if g.get("is_home")]
    away_games = [g for g in prior_games if not g.get("is_home")]
    home_avg = _avg([float(g.get(bdl_key, 0) or 0) for g in home_games]) if home_games else avg_stat
    away_avg = _avg([float(g.get(bdl_key, 0) or 0) for g in away_games]) if away_games else avg_stat

    # Matchup history
    opp_team_id = current_game.get("opp_team_id")
    vs_opp = [g for g in prior_games if g.get("opp_team_id") == opp_team_id] if opp_team_id else []
    vs_opp_avg = _avg([float(g.get(bdl_key, 0) or 0) for g in vs_opp]) if vs_opp else avg_stat

    # Rest days
    rest_days = 2
    if prior_games:
        last_date = prior_games[-1].get("game_date", "")
        cur_date = current_game.get("game_date", "")
        if last_date and cur_date:
            try:
                ld = date.fromisoformat(str(last_date))
                cd = date.fromisoformat(str(cur_date))
                rest_days = (cd - ld).days
            except (ValueError, TypeError):
                pass

    return {
        "pts_pg": _avg(stat_vals("pts")),
        "reb_pg": _avg(stat_vals("reb")),
        "ast_pg": _avg(stat_vals("ast")),
        "three_pm_pg": _avg(stat_vals("fg3m")),
        "stl_pg": _avg(stat_vals("stl")),
        "blk_pg": _avg(stat_vals("blk")),
        "tov_pg": _avg(stat_vals("turnover")),
        "mpg": avg_min,
        "games_played": len(prior_games),
        "starter_pct": 0.8,
        "usage_rate": 0,
        "per": 0,
        "ts_pct": 0,
        "plus_minus": _avg(stat_vals("plus_minus")),
        "fpts_dk": 0,
        f"last3_{bdl_key}": last3,
        f"last5_{bdl_key}": last5,
        f"last10_{bdl_key}": last10,
        f"trend_{bdl_key}": last5 - avg_stat,
        f"std_{bdl_key}": std_stat,
        f"home_avg_{bdl_key}": home_avg,
        f"away_avg_{bdl_key}": away_avg,
        "is_home": is_home,
        f"vs_opp_avg_{bdl_key}": vs_opp_avg,
        "vs_opp_games": len(vs_opp),
        "last3_min": last3_min,
        "last5_min": last5_min,
        "trend_min": last3_min - avg_min,
        "team_pace": 100,
        "team_win_pct": 0.5,
        "opp_pace": 100,
        "pace_factor": 1.0,
        "rest_days": rest_days,
        "is_b2b": 1 if rest_days <= 1 else 0,
        "spread": 0,
        "over_under": 220,
        "name": player_name,
    }


async def run_backtest(
    start_date: date | None = None,
    end_date: date | None = None,
    prop_types: list[str] | None = None,
    min_confidence: float = 60.0,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.5,
    min_games_played: int = 10,
    progress_callback=None,
) -> dict:
    """
    Run a full historical backtest simulation using BDL cached game logs.
    """
    from app.services.smart_predictor import get_smart_predictor

    predictor = get_smart_predictor()

    if prop_types is None:
        prop_types = list(PROP_TO_BDL.keys())

    if start_date is None:
        start_date = date(2025, 10, 21)  # 2025-26 NBA season start
    if end_date is None:
        end_date = date.today() - timedelta(days=1)

    # 1. Load BDL cached game logs — combine all available seasons
    raw_logs = []
    for season_year in [2024, 2025]:
        raw_logs.extend(_load_bdl_season(season_year))
    if not raw_logs:
        return {"status": "error", "message": "No BDL cached game logs found. Run data prefetch first."}

    logger.info(f"Backtest: loaded {len(raw_logs)} raw BDL box scores")

    # 2. Parse into per-player game logs sorted by date
    player_games: dict[int, list[dict]] = defaultdict(list)
    for row in raw_logs:
        player = row.get("player", {})
        game = row.get("game", {})
        team = row.get("team", {})
        game_date_str = game.get("date", "")
        if not game_date_str:
            continue

        minutes = _parse_min(row.get("min", "0"))
        if minutes < 1:
            continue

        pid = player.get("id")
        if not pid:
            continue

        home_team_id = game.get("home_team_id")
        visitor_team_id = game.get("visitor_team_id")
        team_id = team.get("id")
        is_home = (team_id == home_team_id)
        opp_team_id = visitor_team_id if is_home else home_team_id

        entry = {
            "game_date": game_date_str,
            "game_id": game.get("id"),
            "player_id": pid,
            "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
            "team_id": team_id,
            "team_abbr": team.get("abbreviation", ""),
            "is_home": is_home,
            "opp_team_id": opp_team_id,
            "min": minutes,
            "pts": row.get("pts", 0) or 0,
            "reb": row.get("reb", 0) or 0,
            "ast": row.get("ast", 0) or 0,
            "fg3m": row.get("fg3m", 0) or 0,
            "stl": row.get("stl", 0) or 0,
            "blk": row.get("blk", 0) or 0,
            "turnover": row.get("turnover", 0) or 0,
            "plus_minus": row.get("plus_minus", 0) or 0,
        }
        player_games[pid].append(entry)

    for pid in player_games:
        player_games[pid].sort(key=lambda g: g["game_date"])

    logger.info(f"Backtest: indexed {len(player_games)} players")

    # 3. Get all unique game dates in range
    all_dates = set()
    for pid, games in player_games.items():
        for g in games:
            d = g["game_date"]
            if start_date.isoformat() <= d <= end_date.isoformat():
                all_dates.add(d)

    game_dates = sorted(all_dates)
    total_dates = len(game_dates)

    if total_dates == 0:
        return {"status": "error", "message": f"No games found between {start_date} and {end_date}"}

    if progress_callback:
        await progress_callback(15, f"Indexed {len(player_games)} players, starting simulation...")

    # 5. Run the simulation date by date
    current_bankroll = bankroll
    peak_bankroll = bankroll
    max_drawdown = 0.0

    equity_curve = [{"date": start_date.isoformat(), "bankroll": bankroll, "bet_num": 0}]
    bet_log = []
    daily_results = []

    total_bets = 0
    total_hits = 0
    total_misses = 0
    total_profit = 0.0
    by_prop: dict[str, dict] = defaultdict(lambda: {"hits": 0, "misses": 0, "profit": 0.0, "bets": 0})
    by_confidence: dict[str, dict] = defaultdict(lambda: {"hits": 0, "total": 0})

    for date_idx, game_date_str in enumerate(game_dates):
        if current_bankroll <= 0:
            break

        pct = 15 + int((date_idx / max(total_dates, 1)) * 80)
        if progress_callback and date_idx % 5 == 0:
            await progress_callback(pct, f"Simulating {game_date_str} ({date_idx+1}/{total_dates})...")

        day_bets = 0
        day_hits = 0
        day_profit = 0.0
        max_daily_bets = 20  # Cap daily bets to simulate realistic bettor

        # For each player, find games on this date
        for pid, games_list in player_games.items():
            if day_bets >= max_daily_bets:
                break
            # Find entries for this date
            for gi, game_entry in enumerate(games_list):
                if game_entry["game_date"] != game_date_str:
                    continue

                # Skip if player didn't play meaningful minutes
                if game_entry.get("min", 0) < 5:
                    continue

                # Get prior games (everything before this game index)
                prior_games = games_list[:gi]
                if len(prior_games) < min_games_played:
                    continue

                player_name = game_entry.get("player_name", "Unknown")

                # For each prop type, generate prediction and compare
                for prop_type in prop_types:
                    bdl_key = PROP_TO_BDL[prop_type]
                    actual_value = game_entry.get(bdl_key, 0) or 0

                    # Build player features from prior games only
                    player_dict = _build_backtest_player(
                        prior_games, prop_type, game_entry, player_name
                    )
                    if not player_dict:
                        continue

                    # Use recent rolling average as the line (simulates sportsbook line)
                    recent_vals = [float(g.get(bdl_key, 0) or 0) for g in prior_games[-10:]]
                    if not recent_vals:
                        continue
                    line = round(_avg(recent_vals) * 2) / 2  # Round to nearest 0.5

                    if line <= 0:
                        continue

                    # Generate prediction
                    try:
                        result = predictor.predict_prop(player_dict, prop_type, line)
                    except Exception:
                        continue

                    predicted = result.get("predicted_value", 0)
                    confidence = result.get("confidence_score", 0)
                    over_prob = result.get("over_probability", 0.5)

                    # Only bet if confidence meets threshold
                    if confidence < min_confidence:
                        continue

                    # Determine bet direction
                    if predicted > line:
                        bet_direction = "over"
                        bet_prob = over_prob
                    else:
                        bet_direction = "under"
                        bet_prob = 1.0 - over_prob

                    # Check if bet hit
                    if bet_direction == "over":
                        hit = actual_value > line
                    else:
                        hit = actual_value < line

                    # Push (exact line)
                    if actual_value == line:
                        continue

                    # Kelly sizing
                    decimal_odds = 1.909  # -110 standard
                    b = decimal_odds - 1
                    kelly = (b * bet_prob - (1 - bet_prob)) / b
                    kelly = max(kelly, 0) * kelly_fraction

                    if kelly <= 0:
                        continue

                    stake = min(current_bankroll * kelly, current_bankroll * 0.03)  # Cap at 3%
                    stake = min(stake, bankroll * 0.10)  # Never bet more than 10% of starting bankroll
                    stake = round(stake, 2)

                    if stake < 1:
                        continue

                    # Compute profit
                    if hit:
                        profit = round(stake * b, 2)
                        total_hits += 1
                    else:
                        profit = round(-stake, 2)
                        total_misses += 1

                    current_bankroll += profit
                    total_profit += profit
                    total_bets += 1
                    day_bets += 1
                    day_profit += profit
                    if hit:
                        day_hits += 1

                    # Track peak and drawdown
                    if current_bankroll > peak_bankroll:
                        peak_bankroll = current_bankroll
                    dd = (peak_bankroll - current_bankroll) / peak_bankroll if peak_bankroll > 0 else 0
                    if dd > max_drawdown:
                        max_drawdown = dd

                    # By prop type
                    prop_stats = by_prop[prop_type]
                    prop_stats["bets"] += 1
                    prop_stats["profit"] += profit
                    if hit:
                        prop_stats["hits"] += 1
                    else:
                        prop_stats["misses"] += 1

                    # By confidence tier
                    if confidence >= 80:
                        tier = "5_star"
                    elif confidence >= 60:
                        tier = "4_star"
                    else:
                        tier = "3_star"
                    by_confidence[tier]["total"] += 1
                    if hit:
                        by_confidence[tier]["hits"] += 1

                    bet_log.append({
                        "date": game_date_str,
                        "player": player_name,
                        "prop_type": prop_type,
                        "line": line,
                        "predicted": round(predicted, 1),
                        "actual": actual_value,
                        "bet": bet_direction,
                        "hit": hit,
                        "confidence": round(confidence, 1),
                        "stake": stake,
                        "profit": profit,
                        "bankroll": round(current_bankroll, 2),
                    })

        # Daily equity point
        if day_bets > 0:
            equity_curve.append({
                "date": game_date_str,
                "bankroll": round(current_bankroll, 2),
                "bet_num": total_bets,
            })
            daily_results.append({
                "date": game_date_str,
                "bets": day_bets,
                "hits": day_hits,
                "profit": round(day_profit, 2),
                "bankroll": round(current_bankroll, 2),
            })

    if progress_callback:
        await progress_callback(98, "Computing final statistics...")

    # 6. Compute summary statistics
    total_resolved = total_hits + total_misses
    hit_rate = total_hits / total_resolved if total_resolved > 0 else 0
    total_wagered = sum(abs(b["stake"]) for b in bet_log)
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    # Prop breakdown
    prop_breakdown = {}
    for pt, stats in by_prop.items():
        pt_total = stats["hits"] + stats["misses"]
        prop_breakdown[pt] = {
            "bets": pt_total,
            "hits": stats["hits"],
            "misses": stats["misses"],
            "hit_rate": round(stats["hits"] / max(pt_total, 1), 3),
            "profit": round(stats["profit"], 2),
        }

    # Confidence calibration
    calibration = {}
    for tier, stats in by_confidence.items():
        calibration[tier] = {
            "hit_rate": round(stats["hits"] / max(stats["total"], 1), 3),
            "total": stats["total"],
        }

    # Streak calculation
    streak = 0
    streak_type = "none"
    if bet_log:
        streak_type = "win" if bet_log[-1]["hit"] else "loss"
        for b in reversed(bet_log):
            if (b["hit"] and streak_type == "win") or (not b["hit"] and streak_type == "loss"):
                streak += 1
            else:
                break

    if progress_callback:
        await progress_callback(100, "Backtest complete!")

    return {
        "status": "completed",
        "config": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "prop_types": prop_types,
            "min_confidence": min_confidence,
            "starting_bankroll": bankroll,
            "kelly_fraction": kelly_fraction,
        },
        "summary": {
            "total_bets": total_bets,
            "wins": total_hits,
            "losses": total_misses,
            "hit_rate": round(hit_rate, 4),
            "total_wagered": round(total_wagered, 2),
            "total_profit": round(total_profit, 2),
            "roi": round(roi, 2),
            "final_bankroll": round(current_bankroll, 2),
            "max_drawdown": round(max_drawdown * 100, 2),
            "game_dates_simulated": total_dates,
            "current_streak": streak,
            "streak_type": streak_type,
        },
        "equity_curve": equity_curve,
        "daily_results": daily_results,
        "by_prop_type": prop_breakdown,
        "calibration": calibration,
        "bet_log": bet_log[-200:],  # Last 200 bets for display
        "total_bet_log_size": len(bet_log),
    }
