"""
Smart Prediction Engine v4 — Full Feature Set.

Uses BallDontLie game logs + SportsDataIO team stats + The Odds API lines
to build 40+ features per player-prop, then runs an ensemble of ML models.

Feature groups:
  1. Rolling averages (last 3/5/10 games) from BDL game logs
  2. Recent form / trend (hot/cold detection)
  3. Home/away splits from actual game logs
  4. Matchup history (player vs specific opponent)
  5. Travel distance + timezone fatigue scoring
  6. Rest / schedule load (B2B, 3-in-4, games in last 7/14 days)
  7. Game script / blowout risk (spread, O/U → minutes impact)
  8. Injury impact on teammates (usage redistribution)
  9. Consistency (std dev — lower = more predictable)
  10. Pace-adjusted projections (combined pace × opponent defense)
  11. Real sportsbook line as anchor
"""
from __future__ import annotations

import statistics as stats_module
from datetime import date as date_cls
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from xgboost import XGBRegressor

from app.logging_config import get_logger

logger = get_logger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "models" / "artifacts"

# Prop type -> BDL game log stat key
PROP_STAT_MAP = {
    "points": "pts_pg",
    "rebounds": "reb_pg",
    "assists": "ast_pg",
    "threes": "three_pm_pg",
    "steals": "stl_pg",
    "blocks": "blk_pg",
    "turnovers": "tov_pg",
}

# BDL rolling-average key for each prop
PROP_BDL_KEY = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "tov",
}

# Opponent defense column for each prop (from SportsDataIO team features)
PROP_OPP_DEF = {
    "points": "opp_pts_allowed",
    "rebounds": "opp_reb_allowed",
    "assists": "opp_ast_allowed",
    "threes": "opp_3pm_allowed",
    "steals": "opp_stl_allowed",
    "blocks": "opp_blk_allowed",
    "turnovers": "opp_pts_allowed",
}

# Focused feature columns — only features that vary meaningfully during training.
# Dead features (always 0 or constant) have been removed to reduce noise.
FEATURE_COLS = [
    # ── Core stat averages ──
    "avg_stat",                 # season average for this prop
    "mpg",                      # minutes per game
    "games_played",             # sample size
    # ── Rolling averages (the most predictive features) ──
    "last3_stat",               # last 3 games avg
    "last5_stat",               # last 5 games avg
    "last10_stat",              # last 10 games avg
    "trend_stat",               # last5 - season avg (hot/cold)
    # ── Consistency / variance ──
    "std_stat",                 # std dev (lower = more predictable)
    "cv_stat",                  # coefficient of variation (std/mean)
    "max_last10",               # max value in last 10 games
    "min_last10",               # min value in last 10 games
    "range_last10",             # max - min in last 10 (volatility)
    "pct_above_avg",            # % of recent games above season avg
    # ── Home/away splits ──
    "home_avg_stat",
    "away_avg_stat",
    "home_away_diff",
    "split_for_game",           # home avg if home, else away avg
    # ── Matchup history ──
    "vs_opp_avg_stat",
    "vs_opp_games",
    # ── Minutes context ──
    "last3_min",
    "last5_min",
    "trend_min",                # recent minutes trend
    # ── Derived rate features ──
    "stat_per_min",             # avg_stat / mpg (per-minute production)
    "last5_stat_per_min",       # last5 stat / last5 min
    "min_x_rate",               # last5_min × stat_per_min (projected)
    # ── Schedule / fatigue ──
    "is_home",
    "is_b2b",
    "rest_days",
    "games_last_7",
    # ── Travel ──
    "travel_distance",
    "fatigue_score",
    # ── Momentum features ──
    "last3_vs_last10",          # last3 - last10 (short-term momentum)
    "last5_vs_season",          # last5 - season avg (medium-term form)
    "streak_direction",         # +1 if last3 > last10, -1 if below
    # ── Cross-stat context (other stats affect this one) ──
    "pts_pg",                   # always include overall scoring context
    "reb_pg",
    "ast_pg",
    # ── Opponent / game context (the features that beat Vegas) ──
    "opp_stat_allowed",         # what opponent allows per game for this prop
    "opp_pts_allowed",          # opponent defensive rating (pts allowed/game)
    "pace_factor",              # combined team+opp pace relative to league avg
    "usage_rate",               # player's share of team possessions
    "spread",                   # game spread (negative = favored)
    "over_under",               # game total
    "starter_pct",              # % of games started (role indicator)
    "opp_reb_allowed",          # opponent rebounds allowed/game
    "opp_ast_allowed",          # opponent assists allowed/game
    "opp_3pm_allowed",          # opponent 3PM allowed/game
]


def _build_feature_row(player: dict, prop_type: str, line: float | None) -> dict:
    """Build focused feature row for one player + prop type."""
    stat_col = PROP_STAT_MAP.get(prop_type, "pts_pg")
    bdl_key = PROP_BDL_KEY.get(prop_type, "pts")
    avg_stat = player.get(stat_col, 0)

    mpg = player.get("mpg", 0)

    # Rolling averages from BDL game logs
    last3 = player.get(f"last3_{bdl_key}", avg_stat)
    last5 = player.get(f"last5_{bdl_key}", avg_stat)
    last10 = player.get(f"last10_{bdl_key}", avg_stat)
    trend = player.get(f"trend_{bdl_key}", 0)
    std_stat = player.get(f"std_{bdl_key}", 0)

    # Home/away splits
    is_home = player.get("is_home", False)
    home_avg = player.get(f"home_avg_{bdl_key}", avg_stat)
    away_avg = player.get(f"away_avg_{bdl_key}", avg_stat)

    # Matchup history
    vs_opp_avg = player.get(f"vs_opp_avg_{bdl_key}", avg_stat)
    vs_opp_games = player.get("vs_opp_games", 0)

    # Minutes rolling
    last3_min = player.get("last3_min", mpg)
    last5_min = player.get("last5_min", mpg)
    trend_min = player.get("trend_min", 0)

    # Derived rate features
    stat_per_min = avg_stat / max(mpg, 1) if mpg > 0 else 0
    last5_stat_per_min = last5 / max(last5_min, 1) if last5_min > 0 else stat_per_min

    # Variance features
    cv_stat = std_stat / max(avg_stat, 0.1) if avg_stat > 0 else 0
    max_last10 = player.get("max_last10", last10 + std_stat)
    min_last10 = player.get("min_last10", max(last10 - std_stat, 0))
    range_last10 = max_last10 - min_last10
    pct_above_avg = player.get("pct_above_avg", 0.5)

    return {
        # Core
        "avg_stat": avg_stat,
        "mpg": mpg,
        "games_played": player.get("games_played", 0),
        # Rolling averages
        "last3_stat": last3,
        "last5_stat": last5,
        "last10_stat": last10,
        "trend_stat": trend,
        # Consistency / variance
        "std_stat": std_stat,
        "cv_stat": cv_stat,
        "max_last10": max_last10,
        "min_last10": min_last10,
        "range_last10": range_last10,
        "pct_above_avg": pct_above_avg,
        # Home/away
        "home_avg_stat": home_avg,
        "away_avg_stat": away_avg,
        "home_away_diff": home_avg - away_avg,
        "split_for_game": home_avg if is_home else away_avg,
        # Matchup
        "vs_opp_avg_stat": vs_opp_avg,
        "vs_opp_games": vs_opp_games,
        # Minutes
        "last3_min": last3_min,
        "last5_min": last5_min,
        "trend_min": trend_min,
        # Derived rates
        "stat_per_min": stat_per_min,
        "last5_stat_per_min": last5_stat_per_min,
        "min_x_rate": last5_min * stat_per_min,
        # Schedule
        "is_home": 1.0 if is_home else 0.0,
        "is_b2b": 1.0 if player.get("is_b2b") else 0.0,
        "rest_days": player.get("rest_days", 2),
        "games_last_7": player.get("games_last_7", 3),
        # Travel
        "travel_distance": player.get("travel_distance", 0),
        "fatigue_score": player.get("fatigue_score", 0),
        # Momentum
        "last3_vs_last10": last3 - last10,
        "last5_vs_season": last5 - avg_stat,
        "streak_direction": 1.0 if last3 > last10 else (-1.0 if last3 < last10 else 0.0),
        # Cross-stat context
        "pts_pg": player.get("pts_pg", 0),
        "reb_pg": player.get("reb_pg", 0),
        "ast_pg": player.get("ast_pg", 0),
        # Opponent / game context
        "opp_stat_allowed": player.get(PROP_OPP_DEF.get(prop_type, "opp_pts_allowed"), 0),
        "opp_pts_allowed": player.get("opp_pts_allowed", 0),
        "pace_factor": player.get("pace_factor", 1.0),
        "usage_rate": player.get("usage_rate", 0),
        "spread": player.get("spread", 0),
        "over_under": player.get("over_under", 220),
        "starter_pct": player.get("starter_pct", 0),
        "opp_reb_allowed": player.get("opp_reb_allowed", 0),
        "opp_ast_allowed": player.get("opp_ast_allowed", 0),
        "opp_3pm_allowed": player.get("opp_3pm_allowed", 0),
    }


class SmartPredictor:
    """
    Ensemble predictor using 40+ enriched features.
    Models: XGBoost, Random Forest, Gradient Boosting, Bayesian Ridge
    Meta-model: Ridge regression stacking
    """

    def __init__(self):
        self.is_trained = False
        self.prop_models: dict[str, dict] = {}

    def train_all_props(self, enriched_players: dict[str, dict]) -> dict:
        """
        Train models using game-by-game data from BallDontLie.

        For each game a player played, we compute rolling features from
        games BEFORE that game, then use the actual game outcome as target.
        This gives us ~10,000+ training rows instead of ~400.
        """
        from app.services.balldontlie import get_balldontlie
        from app.utils.travel import calculate_fatigue_score, get_timezone_change, get_travel_distance

        bdl = get_balldontlie()

        # ── Fetch multi-season box scores for richer training ──
        logger.info("Fetching multi-season box scores for game-by-game training...")
        season_stats = bdl.get_multi_season_stats()

        if len(season_stats) < 1000:
            logger.warning(f"Only {len(season_stats)} season stats, falling back to cross-sectional training")
            return self._train_cross_sectional(enriched_players)

        logger.info(f"Using {len(season_stats)} multi-season box scores for game-by-game training")

        # ── Parse minutes helper ──
        def parse_min(m: str) -> float:
            if not m or m in ("0", "00", ""):
                return 0.0
            try:
                if ":" in str(m):
                    p = str(m).split(":")
                    return float(p[0]) + float(p[1]) / 60
                return float(m)
            except (ValueError, IndexError):
                return 0.0

        # ── Filter out DNP games and organize by player ──
        from collections import Counter, defaultdict
        player_games: dict[int, list[dict]] = defaultdict(list)
        for row in season_stats:
            if parse_min(row.get("min", "0")) <= 0:
                continue
            pid = row.get("player", {}).get("id")
            if pid:
                player_games[pid].append(row)

        # Sort each player's games by date
        for pid in player_games:
            player_games[pid].sort(key=lambda g: g.get("game", {}).get("date", ""))

        logger.info(f"Season data: {sum(len(v) for v in player_games.values())} games from {len(player_games)} players")

        # ── Build enriched player lookup for team/opponent context ──
        player_name_to_enriched: dict[str, dict] = {}
        for pid_str, pf in enriched_players.items():
            name = (pf.get("name") or "").strip().lower()
            if name:
                player_name_to_enriched[name] = pf

        # ── BDL team ID → abbreviation mapping ──
        team_id_to_abbr: dict[int, str] = {}
        for row in season_stats[:500]:
            t = row.get("team", {})
            if t.get("id") and t.get("abbreviation"):
                team_id_to_abbr[t["id"]] = t["abbreviation"]

        # ── Build game-by-game training dataset ──
        # For each game index i, use games [0..i-1] as history to compute
        # rolling features, and game i's actual stats as the target.
        STAT_KEYS = {
            "points": "pts", "rebounds": "reb", "assists": "ast",
            "threes": "fg3m", "steals": "stl", "blocks": "blk",
            "turnovers": "turnover",
        }

        # ── Pre-compute per-player arrays for ALL stat types (vectorized) ──
        # This avoids recomputing rolling averages from scratch for every game.

        # Build team lookup for enriched data
        team_abbr_to_enriched: dict[str, dict] = {}
        for ep in enriched_players.values():
            t = ep.get("team", "")
            if t and t not in team_abbr_to_enriched:
                team_abbr_to_enriched[t] = ep

        # ── Compute per-team per-season defensive stats from BDL data ──
        # For each game, we know what the opponent scored. By aggregating
        # across all games for a team in a season, we get "points allowed per game" etc.
        # This lets us have real opponent defense features during training.
        team_season_allowed: dict[tuple, dict] = defaultdict(lambda: {
            "pts": 0, "reb": 0, "ast": 0, "fg3m": 0, "stl": 0, "blk": 0, "turnover": 0,
            "games": 0, "wins": 0, "total_pts_scored": 0,
        })
        # Also track team pace (possessions proxy = FGA + 0.44*FTA - ORB + TOV)
        # We'll use a simpler proxy: total points scored + allowed / 2 per game
        # Group box scores by game_id to get team totals per game
        game_team_totals: dict[tuple, dict] = defaultdict(lambda: {
            "pts": 0, "reb": 0, "ast": 0, "fg3m": 0, "stl": 0, "blk": 0, "turnover": 0,
        })
        game_info_map: dict = {}  # game_id -> {date, home_team_id, visitor_team_id, home_score, visitor_score}
        for row in season_stats:
            game = row.get("game", {})
            team = row.get("team", {})
            gid = game.get("id")
            tid = team.get("id")
            if not gid or not tid:
                continue
            key = (gid, tid)
            for sk in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
                game_team_totals[key][sk] += (row.get(sk, 0) or 0)
            if gid not in game_info_map:
                game_info_map[gid] = {
                    "date": game.get("date", "")[:10],
                    "home_team_id": game.get("home_team_id"),
                    "visitor_team_id": game.get("visitor_team_id"),
                    "home_score": game.get("home_team_score", 0) or 0,
                    "visitor_score": game.get("visitor_team_score", 0) or 0,
                }

        # Now compute what each team ALLOWED per game per season
        for gid, ginfo in game_info_map.items():
            d = ginfo["date"]
            try:
                yr, mo = int(d[:4]), int(d[5:7])
                season = yr if mo >= 10 else yr - 1
            except (ValueError, IndexError):
                continue
            htid = ginfo["home_team_id"]
            vtid = ginfo["visitor_team_id"]
            if not htid or not vtid:
                continue
            # Home team allowed = visitor team's totals
            v_totals = game_team_totals.get((gid, vtid), {})
            h_totals = game_team_totals.get((gid, htid), {})
            if v_totals:
                ts = team_season_allowed[(htid, season)]
                for sk in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
                    ts[sk] += v_totals.get(sk, 0)
                ts["games"] += 1
                ts["total_pts_scored"] += sum(s.get("pts", 0) for s in [h_totals] if s)
                if ginfo["home_score"] > ginfo["visitor_score"]:
                    ts["wins"] += 1
            if h_totals:
                ts = team_season_allowed[(vtid, season)]
                for sk in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
                    ts[sk] += h_totals.get(sk, 0)
                ts["games"] += 1
                ts["total_pts_scored"] += sum(s.get("pts", 0) for s in [v_totals] if s)
                if ginfo["visitor_score"] > ginfo["home_score"]:
                    ts["wins"] += 1

        logger.info(f"Computed opponent defense stats for {len(team_season_allowed)} team-seasons")

        # Pre-extract all stat arrays per player (once, shared across prop types)
        player_precomputed: dict[int, dict] = {}
        for pid, games in player_games.items():
            if len(games) < 3:
                continue
            p = games[0].get("player", {})
            pname = f"{p.get('first_name', '')} {p.get('last_name', '')}".lower()
            enriched = player_name_to_enriched.get(pname, {})
            team_id = games[0].get("team", {}).get("id")
            team_abbr = team_id_to_abbr.get(team_id, "")

            # Extract arrays once
            all_mins = [parse_min(g.get("min", "0")) for g in games]
            all_dates = [g.get("game", {}).get("date", "")[:10] for g in games]
            # Detect season for each game: Oct-Dec = that year, Jan-Sep = year-1
            all_seasons = []
            for d in all_dates:
                try:
                    yr, mo = int(d[:4]), int(d[5:7])
                    all_seasons.append(yr if mo >= 10 else yr - 1)
                except (ValueError, IndexError):
                    all_seasons.append(0)
            all_home = [
                g.get("team", {}).get("id") == g.get("game", {}).get("home_team_id")
                for g in games
            ]
            all_opp_ids = [
                g.get("game", {}).get("visitor_team_id") if h
                else g.get("game", {}).get("home_team_id")
                for g, h in zip(games, all_home)
            ]

            # Per-stat arrays
            stat_arrays = {}
            for stype, skey in STAT_KEYS.items():
                stat_arrays[stype] = [g.get(skey, 0) or 0 for g in games]

            # Game IDs for looking up team totals
            all_game_ids = [g.get("game", {}).get("id") for g in games]
            # Starter detection: minutes > 20 in majority of games = likely starter
            starter_count = sum(1 for m in all_mins if m >= 20)
            starter_pct = starter_count / max(len(all_mins), 1)
            # Usage proxy: pts / (team_pts_in_game) — approximate from scoring share
            # We'll compute this per-game later

            player_precomputed[pid] = {
                "enriched": enriched,
                "team_abbr": team_abbr,
                "team_id": team_id,
                "mins": all_mins,
                "dates": all_dates,
                "is_home": all_home,
                "opp_ids": all_opp_ids,
                "game_ids": all_game_ids,
                "stats": stat_arrays,
                "seasons": all_seasons,
                "n_games": len(games),
                "starter_pct": starter_pct,
            }

        logger.info(f"Pre-computed arrays for {len(player_precomputed)} players")

        metrics = {}

        for prop_type, stat_col in PROP_STAT_MAP.items():
            rows = []
            targets = []
            row_seasons = []

            for pid, pc in player_precomputed.items():
                enriched = pc["enriched"]
                team_abbr = pc["team_abbr"]
                stat_vals = pc["stats"][prop_type]
                mins_vals = pc["mins"]
                n_total = pc["n_games"]

                # Cumulative sums for O(1) rolling averages
                # RESET at season boundaries so features match inference (current-season only)
                seasons = pc["seasons"]
                cum_stat = [0.0]
                cum_min = [0.0]
                season_start_idx = [0] * n_total  # index where current season starts
                cur_season = seasons[0] if seasons else 0
                cur_start = 0
                for j in range(n_total):
                    if seasons[j] != cur_season:
                        cur_season = seasons[j]
                        cur_start = j
                        # Reset cumulative sums at season boundary
                        # Include this game's stats in the reset
                        cum_stat.append(stat_vals[j])
                        cum_min.append(mins_vals[j])
                    else:
                        cum_stat.append(cum_stat[-1] + stat_vals[j])
                        cum_min.append(cum_min[-1] + mins_vals[j])
                    season_start_idx[j] = cur_start

                # Home/away cumulative sums (also reset at season boundaries)
                home_cum = [0.0]
                home_count_cum = [0]
                away_cum = [0.0]
                away_count_cum = [0]
                prev_s = seasons[0] if seasons else 0
                for j in range(n_total):
                    if seasons[j] != prev_s:
                        prev_s = seasons[j]
                        # Reset home/away at season boundary, include this game
                        if pc["is_home"][j]:
                            home_cum.append(stat_vals[j])
                            home_count_cum.append(1)
                            away_cum.append(0.0)
                            away_count_cum.append(0)
                        else:
                            home_cum.append(0.0)
                            home_count_cum.append(0)
                            away_cum.append(stat_vals[j])
                            away_count_cum.append(1)
                    elif pc["is_home"][j]:
                        home_cum.append(home_cum[-1] + stat_vals[j])
                        home_count_cum.append(home_count_cum[-1] + 1)
                        away_cum.append(away_cum[-1])
                        away_count_cum.append(away_count_cum[-1])
                    else:
                        away_cum.append(away_cum[-1] + stat_vals[j])
                        away_count_cum.append(away_count_cum[-1] + 1)
                        home_cum.append(home_cum[-1])
                        home_count_cum.append(home_count_cum[-1])

                # Cross-stat cumulative sums for pts/reb/ast context
                # (season-scoped, same reset logic as cum_stat)
                pts_vals = pc["stats"]["points"]
                reb_vals = pc["stats"]["rebounds"]
                ast_vals = pc["stats"]["assists"]
                cum_pts = [0.0]
                cum_reb = [0.0]
                cum_ast = [0.0]
                cs_season = seasons[0] if seasons else 0
                for j in range(n_total):
                    if seasons[j] != cs_season:
                        cs_season = seasons[j]
                        cum_pts.append(pts_vals[j])
                        cum_reb.append(reb_vals[j])
                        cum_ast.append(ast_vals[j])
                    else:
                        cum_pts.append(cum_pts[-1] + pts_vals[j])
                        cum_reb.append(cum_reb[-1] + reb_vals[j])
                        cum_ast.append(cum_ast[-1] + ast_vals[j])

                # Matchup history: group by opponent
                opp_stats: dict[int, list[float]] = defaultdict(list)

                enriched.get("team_pace", 100)
                enriched.get("team_win_pct", 0.5)

                for i in range(3, n_total):
                    actual_stat = stat_vals[i]
                    actual_min = mins_vals[i]
                    if actual_min <= 0:
                        continue

                    # Number of games in CURRENT season before this game
                    s_start = season_start_idx[i]
                    n_season = i - s_start  # games in this season before game i
                    if n_season < 3:
                        continue  # need at least 3 season games for features

                    # Rolling averages from cumulative sums (season-scoped)
                    season_avg = cum_stat[i] / n_season
                    last3 = (cum_stat[i] - cum_stat[max(i-3, s_start)]) / min(3, n_season)
                    last5 = (cum_stat[i] - cum_stat[max(i-5, s_start)]) / min(5, n_season)
                    last10 = (cum_stat[i] - cum_stat[max(i-10, s_start)]) / min(10, n_season)
                    trend = last5 - season_avg

                    mpg = cum_min[i] / n_season
                    last3_min = (cum_min[i] - cum_min[max(i-3, s_start)]) / min(3, n_season)
                    last5_min = (cum_min[i] - cum_min[max(i-5, s_start)]) / min(5, n_season)
                    trend_min = last5_min - mpg

                    # Std dev (use last 20 games max for speed)
                    window = stat_vals[max(i-20, 0):i]
                    std_stat = stats_module.stdev(window) if len(window) >= 5 else 0.0

                    # Home/away splits
                    hc = home_count_cum[i]
                    ac = away_count_cum[i]
                    home_avg = home_cum[i] / max(hc, 1)
                    away_avg = away_cum[i] / max(ac, 1)

                    is_home = pc["is_home"][i]
                    opp_id = pc["opp_ids"][i]

                    # Matchup history (accumulated as we go)
                    vs_list = opp_stats.get(opp_id, [])
                    vs_avg = sum(vs_list) / len(vs_list) if vs_list else season_avg
                    vs_count = len(vs_list)

                    # Update matchup history for future games
                    if opp_id:
                        opp_stats[opp_id].append(actual_stat)

                    # Rest
                    try:
                        gd = date_cls.fromisoformat(pc["dates"][i])
                        pd_ = date_cls.fromisoformat(pc["dates"][i-1])
                        days_rest = (gd - pd_).days
                    except (ValueError, TypeError):
                        days_rest = 2
                    is_b2b = 1 if days_rest <= 1 else 0

                    # Games in last 7 days (approximate from recent games)
                    games_last_7 = 0
                    for k in range(i-1, max(i-5, -1), -1):
                        try:
                            dk = date_cls.fromisoformat(pc["dates"][k])
                            if (gd - dk).days <= 7:
                                games_last_7 += 1
                            else:
                                break
                        except (ValueError, TypeError):
                            break

                    # Travel
                    opp_abbr = team_id_to_abbr.get(opp_id, "")
                    if is_home:
                        travel_dist, tz_change = 0.0, 0
                    else:
                        travel_dist = get_travel_distance(team_abbr, opp_abbr)
                        tz_change = get_timezone_change(team_abbr, opp_abbr)
                    fatigue = calculate_fatigue_score(travel_dist, tz_change, is_b2b, days_rest)

                    # Opponent context — from BDL-computed team season stats
                    game_season = seasons[i]
                    opp_ts = team_season_allowed.get((opp_id, game_season), {})
                    opp_games = max(opp_ts.get("games", 0), 1)
                    opp_pts_allowed = opp_ts.get("pts", 0) / opp_games
                    opp_reb_allowed = opp_ts.get("reb", 0) / opp_games
                    opp_ast_allowed = opp_ts.get("ast", 0) / opp_games
                    opp_3pm_allowed = opp_ts.get("fg3m", 0) / opp_games
                    # Prop-specific opponent defense
                    opp_stat_key = PROP_BDL_KEY.get(prop_type, "pts")
                    opp_stat_allowed = opp_ts.get(opp_stat_key, 0) / opp_games

                    # Team's own season stats for pace/spread proxy
                    own_ts = team_season_allowed.get((pc["team_id"], game_season), {})
                    own_games = max(own_ts.get("games", 0), 1)
                    own_pts_scored = own_ts.get("total_pts_scored", 0) / own_games
                    opp_pts_scored = opp_ts.get("total_pts_scored", 0) / opp_games
                    # Pace proxy: average of both teams' scoring rates
                    pace_factor = (own_pts_scored + opp_pts_scored) / 220.0 if (own_pts_scored + opp_pts_scored) > 0 else 1.0
                    # Spread proxy from win%
                    own_win_pct = own_ts.get("wins", 0) / own_games
                    opp_win_pct = opp_ts.get("wins", 0) / opp_games
                    spread = (opp_win_pct - own_win_pct) * 15  # rough conversion
                    if not is_home:
                        spread += 3  # away team disadvantage
                    else:
                        spread -= 3
                    over_under = own_pts_scored + opp_pts_allowed  # expected total
                    # Derived rate features
                    stat_per_min = season_avg / max(mpg, 1) if mpg > 0 else 0
                    last5_spm = last5 / max(last5_min, 1) if last5_min > 0 else stat_per_min

                    # Variance features from recent window
                    window_vals = stat_vals[max(i-10, s_start):i]
                    if len(window_vals) >= 3:
                        max_l10 = max(window_vals)
                        min_l10 = min(window_vals)
                        pct_above = sum(1 for v in window_vals if v > season_avg) / len(window_vals)
                    else:
                        max_l10 = last10 + std_stat
                        min_l10 = max(last10 - std_stat, 0)
                        pct_above = 0.5
                    range_l10 = max_l10 - min_l10
                    cv_stat = std_stat / max(season_avg, 0.1) if season_avg > 0 else 0

                    # Cross-stat context: compute season avgs for pts/reb/ast
                    pts_season = cum_pts[i] / max(n_season, 1)
                    reb_season = cum_reb[i] / max(n_season, 1)
                    ast_season = cum_ast[i] / max(n_season, 1)

                    # Usage proxy: player's pts share of team scoring
                    usage_rate = (pts_season / max(own_pts_scored, 1)) * 100 if own_pts_scored > 0 else 20.0

                    feat = {
                        # Core
                        "avg_stat": season_avg,
                        "mpg": mpg,
                        "games_played": n_season,
                        # Rolling averages
                        "last3_stat": last3,
                        "last5_stat": last5,
                        "last10_stat": last10,
                        "trend_stat": trend,
                        # Consistency / variance
                        "std_stat": std_stat,
                        "cv_stat": cv_stat,
                        "max_last10": max_l10,
                        "min_last10": min_l10,
                        "range_last10": range_l10,
                        "pct_above_avg": pct_above,
                        # Home/away
                        "home_avg_stat": home_avg,
                        "away_avg_stat": away_avg,
                        "home_away_diff": home_avg - away_avg,
                        "split_for_game": home_avg if is_home else away_avg,
                        # Matchup
                        "vs_opp_avg_stat": vs_avg,
                        "vs_opp_games": vs_count,
                        # Minutes
                        "last3_min": last3_min,
                        "last5_min": last5_min,
                        "trend_min": trend_min,
                        # Derived rates
                        "stat_per_min": stat_per_min,
                        "last5_stat_per_min": last5_spm,
                        "min_x_rate": last5_min * stat_per_min,
                        # Schedule
                        "is_home": 1.0 if is_home else 0.0,
                        "is_b2b": float(is_b2b),
                        "rest_days": days_rest,
                        "games_last_7": min(games_last_7, 4),
                        # Travel
                        "travel_distance": travel_dist,
                        "fatigue_score": fatigue,
                        # Momentum
                        "last3_vs_last10": last3 - last10,
                        "last5_vs_season": last5 - season_avg,
                        "streak_direction": 1.0 if last3 > last10 else (-1.0 if last3 < last10 else 0.0),
                        # Cross-stat context
                        "pts_pg": pts_season,
                        "reb_pg": reb_season,
                        "ast_pg": ast_season,
                        # Opponent / game context
                        "opp_stat_allowed": opp_stat_allowed,
                        "opp_pts_allowed": opp_pts_allowed,
                        "pace_factor": pace_factor,
                        "usage_rate": usage_rate,
                        "spread": round(spread, 1),
                        "over_under": round(over_under, 1),
                        "starter_pct": pc["starter_pct"],
                        "opp_reb_allowed": opp_reb_allowed,
                        "opp_ast_allowed": opp_ast_allowed,
                        "opp_3pm_allowed": opp_3pm_allowed,
                    }

                    rows.append(feat)
                    targets.append(actual_stat)
                    row_seasons.append(seasons[i])

            if len(rows) < 30:
                logger.warning(f"Not enough game-by-game data for {prop_type}: {len(rows)} rows")
                continue

            row_seasons_arr = row_seasons

            # Cap training rows to keep training time reasonable (~40k max)
            # Keep all rows — more recent data is naturally at the end
            MAX_TRAIN_ROWS = 40000
            if len(rows) > MAX_TRAIN_ROWS:
                # Sample uniformly but keep the last 20k rows (most recent) intact
                import random as _rng
                _rng.seed(42)
                keep_recent = min(20000, len(rows))
                older = rows[:-keep_recent]
                recent = rows[-keep_recent:]
                older_targets = targets[:-keep_recent]
                recent_targets = targets[-keep_recent:]
                sample_size = MAX_TRAIN_ROWS - keep_recent
                indices = sorted(_rng.sample(range(len(older)), min(sample_size, len(older))))
                older_seasons = row_seasons_arr[:-keep_recent]
                recent_seasons = row_seasons_arr[-keep_recent:]
                rows = [older[i] for i in indices] + recent
                targets = [older_targets[i] for i in indices] + recent_targets
                row_seasons_arr = [older_seasons[i] for i in indices] + recent_seasons
                logger.info(f"Sampled {prop_type} from {len(rows)+len(older)-len(indices)} to {len(rows)} rows (kept {keep_recent} recent)")

            # Season-based sample weights: current season=1.0, decay 0.2 per older season (min 0.2)
            max_season = max(row_seasons_arr) if row_seasons_arr else 2025
            season_weights = np.array(
                [max(0.2, 1.0 - 0.2 * (max_season - s)) for s in row_seasons_arr],
                dtype=float,
            )
            logger.info(f"Training {prop_type} on {len(rows)} game-by-game rows (season weights: {dict(sorted(Counter(season_weights).items()))})...")
            X = pd.DataFrame(rows)[FEATURE_COLS].fillna(0)
            y = np.array(targets, dtype=float)

            # Train/test split for honest evaluation
            split = int(len(rows) * 0.85)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y[:split], y[split:]
            w_train, _w_test = season_weights[:split], season_weights[split:]

            models = {
                "xgboost": XGBRegressor(
                    n_estimators=600, max_depth=4, learning_rate=0.03,
                    subsample=0.75, colsample_bytree=0.6, random_state=42,
                    verbosity=0, reg_alpha=0.5, reg_lambda=2.0,
                    min_child_weight=10, gamma=0.1,
                ),
                "random_forest": RandomForestRegressor(
                    n_estimators=400, max_depth=8, min_samples_leaf=15,
                    max_features=0.6, random_state=42, n_jobs=-1,
                ),
                "gradient_boosting": GradientBoostingRegressor(
                    n_estimators=400, max_depth=4, learning_rate=0.03,
                    subsample=0.75, min_samples_leaf=15, random_state=42,
                    max_features=0.7,
                ),
                "bayesian_ridge": BayesianRidge(),
            }

            base_preds_train = {}
            base_preds_test = {}
            model_metrics = {}

            for name, model in models.items():
                try:
                    if name == "bayesian_ridge":
                        model.fit(X_train, y_train)
                    else:
                        model.fit(X_train, y_train, sample_weight=w_train)
                    train_preds = model.predict(X_train)
                    test_preds = model.predict(X_test)
                    train_rmse = float(np.sqrt(np.mean((train_preds - y_train) ** 2)))
                    test_rmse = float(np.sqrt(np.mean((test_preds - y_test) ** 2)))
                    base_preds_train[name] = train_preds
                    base_preds_test[name] = test_preds
                    model_metrics[name] = test_rmse
                    logger.info(f"  {name} {prop_type}: train_rmse={train_rmse:.2f} test_rmse={test_rmse:.2f}")
                except Exception as e:
                    logger.warning(f"Error training {name} for {prop_type}: {e}")

            # Meta-model stacking (positive=True prevents pathological negative coefficients
            # from multicollinearity between correlated base models)
            if len(base_preds_train) >= 2:
                meta_X_train = pd.DataFrame(base_preds_train)
                meta_X_test = pd.DataFrame(base_preds_test)
                meta_model = Ridge(alpha=1.0, positive=True)
                meta_model.fit(meta_X_train, y_train, sample_weight=w_train)
                meta_test_preds = meta_model.predict(meta_X_test)
                meta_rmse = float(np.sqrt(np.mean((meta_test_preds - y_test) ** 2)))
                model_metrics["meta_ensemble"] = meta_rmse
            else:
                meta_model = None

            # Now retrain on ALL data for production
            for name, model in models.items():
                try:
                    if name == "bayesian_ridge":
                        model.fit(X, y)
                    else:
                        model.fit(X, y, sample_weight=season_weights)
                except Exception:
                    pass
            if meta_model and len(base_preds_train) >= 2:
                all_base = {}
                for name, model in models.items():
                    try:
                        all_base[name] = model.predict(X)
                    except Exception:
                        pass
                if len(all_base) >= 2:
                    meta_model.fit(pd.DataFrame(all_base), y, sample_weight=season_weights)

            valid = {k: v for k, v in model_metrics.items() if k != "meta_ensemble" and v > 0}
            total_inv = sum(1.0 / v for v in valid.values()) if valid else 1
            weights = {k: (1.0 / v) / total_inv for k, v in valid.items()}

            self.prop_models[prop_type] = {
                "models": models,
                "meta_model": meta_model,
                "weights": weights,
                "feature_cols": FEATURE_COLS,
            }

            metrics[prop_type] = model_metrics
            logger.info(f"Trained {prop_type}: {model_metrics}")

        self.is_trained = len(self.prop_models) > 0
        logger.info(f"Game-by-game training complete: {list(self.prop_models.keys())} ({len(self.prop_models)} prop types)")
        if self.prop_models:
            self._save()
        return metrics

    def _train_cross_sectional(self, enriched_players: dict[str, dict]) -> dict:
        """Fallback: train on player averages (one row per player)."""
        metrics = {}
        for prop_type, stat_col in PROP_STAT_MAP.items():
            rows, targets = [], []
            for pid, pf in enriched_players.items():
                avg = pf.get(stat_col, 0)
                if avg <= 0.1 or pf.get("games_played", 0) < 5:
                    continue
                rows.append(_build_feature_row(pf, prop_type, None))
                targets.append(avg)
            if len(rows) < 20:
                continue
            X = pd.DataFrame(rows)[FEATURE_COLS].fillna(0)
            y = np.array(targets)
            noise_scale = np.std(y) * 0.15
            y_noisy = y + np.random.normal(0, noise_scale, len(y))
            models = {
                "xgboost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.06, subsample=0.8, colsample_bytree=0.7, random_state=42, verbosity=0),
                "random_forest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1),
                "gradient_boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.08, subsample=0.8, random_state=42),
                "bayesian_ridge": BayesianRidge(),
            }
            base_preds, model_metrics = {}, {}
            for name, model in models.items():
                try:
                    model.fit(X, y_noisy)
                    preds = model.predict(X)
                    base_preds[name] = preds
                    model_metrics[name] = float(np.sqrt(np.mean((preds - y) ** 2)))
                except Exception:
                    pass
            meta_model = None
            if len(base_preds) >= 2:
                meta_model = Ridge(alpha=1.0, positive=True)
                meta_model.fit(pd.DataFrame(base_preds), y)
                meta_preds = meta_model.predict(pd.DataFrame(base_preds))
                model_metrics["meta_ensemble"] = float(np.sqrt(np.mean((meta_preds - y) ** 2)))
            valid = {k: v for k, v in model_metrics.items() if k != "meta_ensemble" and v > 0}
            total_inv = sum(1.0 / v for v in valid.values()) if valid else 1
            weights = {k: (1.0 / v) / total_inv for k, v in valid.items()}
            self.prop_models[prop_type] = {"models": models, "meta_model": meta_model, "weights": weights, "feature_cols": FEATURE_COLS}
            metrics[prop_type] = model_metrics
        self.is_trained = True
        self._save()
        return metrics

    def predict_prop(self, player: dict, prop_type: str, line: float | None) -> dict:
        """
        Predict a single prop for a single player using full feature set.
        """
        prop_data = self.prop_models.get(prop_type)
        if not prop_data:
            logger.debug(f"No model for {prop_type}, available: {list(self.prop_models.keys())}")
            stat_col = PROP_STAT_MAP.get(prop_type, "pts_pg")
            return self._fallback_prediction(player.get(stat_col, 0), line, player)

        feat = _build_feature_row(player, prop_type, line)
        X = pd.DataFrame([feat])[prop_data["feature_cols"]].fillna(0)

        base_preds = {}
        for name, model in prop_data["models"].items():
            try:
                pred = model.predict(X)[0]
                base_preds[name] = pred
            except Exception:
                pass

        if not base_preds:
            stat_col = PROP_STAT_MAP.get(prop_type, "pts_pg")
            return self._fallback_prediction(player.get(stat_col, 0), line, player)

        # Always use inverse-RMSE weighted average of base models.
        # Meta-model stacking consistently underperforms due to multicollinearity
        # between highly correlated base model predictions.
        weights = prop_data["weights"]
        total_w = sum(weights.get(k, 0) for k in base_preds)
        if total_w > 0:
            predicted_value = sum(
                base_preds[k] * weights.get(k, 0) for k in base_preds
            ) / total_w
        else:
            predicted_value = float(np.mean(list(base_preds.values())))

        predicted_value = round(max(predicted_value, 0.1), 1)
        actual_line = line if line is not None else round(predicted_value * 2) / 2

        # Use player-specific std dev from game logs if available, else defaults
        bdl_key = PROP_BDL_KEY.get(prop_type, "pts")
        player_std = player.get(f"std_{bdl_key}", 0)
        DEFAULT_STD = {
            "points": 7.5, "rebounds": 3.0, "assists": 2.5,
            "threes": 1.2, "steals": 0.8, "blocks": 0.7, "turnovers": 1.0,
        }
        game_std = player_std if player_std > 0 else DEFAULT_STD.get(prop_type, 3.0)

        pred_values = list(base_preds.values())
        model_std = float(np.std(pred_values)) if len(pred_values) > 1 else 0.0
        pred_std = max(game_std, model_std * 3, 0.5)

        diff = predicted_value - actual_line
        z_score = diff / pred_std
        over_prob = round(min(max(0.5 + 0.5 * np.tanh(z_score * 0.4), 0.05), 0.95), 3)

        # Ensemble agreement
        if len(pred_values) >= 2:
            agreement = 1.0 - min(np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-6), 1.0)
        else:
            agreement = 0.5

        # Confidence scoring — now uses game log depth + consistency
        confidence = self._compute_confidence(
            player, agreement, diff, actual_line, line is not None
        )
        confidence_tier = min(int(confidence / 20) + 1, 5)

        contributions = []
        for name, pred in base_preds.items():
            contributions.append({
                "model_name": name,
                "prediction": round(float(pred), 1),
                "weight": round(prop_data["weights"].get(name, 0), 3),
            })

        return {
            "predicted_value": predicted_value,
            "over_probability": over_prob,
            "under_probability": round(1.0 - over_prob, 3),
            "confidence_score": confidence,
            "confidence_tier": confidence_tier,
            "ensemble_agreement": round(agreement, 3),
            "contributions": contributions,
            "prediction_std": round(pred_std, 2),
        }

    def _compute_confidence(
        self, player: dict, agreement: float, diff: float,
        actual_line: float, has_real_line: bool,
    ) -> float:
        """
        Confidence scoring that rewards:
        - More game log data (rolling averages are reliable)
        - Low variance (consistent player)
        - Strong matchup history
        - Fresh rest / no fatigue
        - Model agreement
        """
        c = 30.0

        # Real sportsbook line
        if has_real_line:
            c += 8

        # Game log depth
        log_count = player.get("game_log_count", 0)
        if log_count >= 40:
            c += 10
        elif log_count >= 20:
            c += 6
        elif log_count >= 10:
            c += 3

        # Minutes / role
        mpg = player.get("mpg", 0)
        if mpg > 32:
            c += 6
        elif mpg > 25:
            c += 3

        starter = player.get("starter_pct", 0)
        if starter > 0.8:
            c += 4

        # Consistency (low std = more predictable)
        std_pts = player.get("std_pts", 10)
        if std_pts < 5:
            c += 6
        elif std_pts < 8:
            c += 3

        # Matchup history
        vs_games = player.get("vs_opp_games", 0)
        if vs_games >= 3:
            c += 5
        elif vs_games >= 1:
            c += 2

        # Model agreement
        c += agreement * 12

        # Edge size (bigger edge = more confidence, but cap it)
        c += min(abs(diff) / max(actual_line, 1) * 25, 12)

        # Fatigue penalty
        fatigue = player.get("fatigue_score", 0)
        if fatigue > 0.5:
            c -= 5
        elif fatigue > 0.3:
            c -= 2

        # Blowout risk penalty (starters may sit)
        blowout = player.get("blowout_risk", 0)
        if blowout > 0.5:
            c -= 4

        return round(min(max(c, 15), 98), 1)

    def _fallback_prediction(self, avg: float, line: float | None, player: dict | None = None) -> dict:
        """Simple fallback when no model is available."""
        actual_line = line if line is not None else round(avg * 2) / 2
        diff = avg - actual_line
        over_prob = round(min(max(0.5 + diff * 0.1, 0.1), 0.9), 3)
        return {
            "predicted_value": round(avg, 1),
            "over_probability": over_prob,
            "under_probability": round(1.0 - over_prob, 3),
            "confidence_score": 35.0,
            "confidence_tier": 2,
            "ensemble_agreement": 0.5,
            "contributions": [],
            "prediction_std": abs(avg * 0.15),
        }

    def _save(self):
        save_dir = ARTIFACTS_DIR / "smart"
        save_dir.mkdir(parents=True, exist_ok=True)
        for prop_type, data in self.prop_models.items():
            joblib.dump({
                "models": {k: v for k, v in data["models"].items()},
                "meta_model": data["meta_model"],
                "weights": data["weights"],
                "feature_cols": data["feature_cols"],
            }, str(save_dir / f"{prop_type}.joblib"))
        logger.info(f"Smart models saved to {save_dir}")

    def load(self) -> bool:
        load_dir = ARTIFACTS_DIR / "smart"
        if not load_dir.exists():
            return False

        for prop_type in PROP_STAT_MAP:
            path = load_dir / f"{prop_type}.joblib"
            if path.exists():
                try:
                    data = joblib.load(str(path))
                    self.prop_models[prop_type] = data
                except Exception as e:
                    logger.warning(f"Error loading {prop_type}: {e}")

        self.is_trained = len(self.prop_models) > 0
        if self.is_trained:
            logger.info(f"Loaded smart models for {list(self.prop_models.keys())}")
        return self.is_trained


_predictor: SmartPredictor | None = None


def get_smart_predictor() -> SmartPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SmartPredictor()
        _predictor.load()
    return _predictor
