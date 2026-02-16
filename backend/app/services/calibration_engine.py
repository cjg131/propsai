"""
Calibration Engine — Walk-Forward Prediction Accuracy Measurement.

Core question: How accurately does our model predict what a player will do?

For every player, every game day, every prop type (PTS/REB/AST/3PM/STL/BLK/TO):
  1. Build features using ONLY data available before that game
  2. Generate a predicted stat value
  3. Compare predicted value to actual result
  4. Measure: MAE, RMSE, bias, % within 1/2/3 of actual

Lines are irrelevant — we measure raw prediction accuracy against reality.
If the model can accurately predict actuals, beating any line becomes trivial.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

from app.logging_config import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).parent.parent / "cache"
CALIBRATION_DIR = Path(__file__).parent.parent / "calibration"

PROP_TO_BDL = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "steals": "stl",
    "blocks": "blk",
    "turnovers": "turnover",
}

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


def _load_bdl_seasons(seasons: list[int] | None = None) -> list[dict]:
    if seasons is None:
        seasons = [2021, 2022, 2023, 2024, 2025]
    all_data = []
    for s in seasons:
        path = CACHE_DIR / f"bdl_season_{s}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            all_data.extend(data)
            logger.info(f"Calibration: loaded {len(data)} rows for season {s}")
    return all_data


def _build_calibration_features(
    prior_games: list[dict],
    current_game: dict,
    prop_type: str,
) -> dict | None:
    """
    Build a feature dict using ONLY data before the current game.
    Returns None if insufficient data.
    """
    if len(prior_games) < 10:
        return None

    bdl_key = PROP_TO_BDL.get(prop_type, "pts")

    def stat_vals(key: str) -> list[float]:
        return [float(g.get(key, 0) or 0) for g in prior_games]

    def min_vals() -> list[float]:
        return [g.get("min_parsed", 0) for g in prior_games]

    all_stat = stat_vals(bdl_key)
    mins = min_vals()
    avg_stat = _avg(all_stat)
    avg_min = _avg(mins)

    if avg_min < 5:
        return None

    last3 = _avg(all_stat[-3:]) if len(all_stat) >= 3 else avg_stat
    last5 = _avg(all_stat[-5:]) if len(all_stat) >= 5 else avg_stat
    last10 = _avg(all_stat[-10:]) if len(all_stat) >= 10 else avg_stat
    std_stat = _std(all_stat[-20:]) if len(all_stat) >= 5 else 0.0

    last3_min = _avg(mins[-3:]) if len(mins) >= 3 else avg_min
    last5_min = _avg(mins[-5:]) if len(mins) >= 5 else avg_min

    is_home = current_game.get("is_home", False)
    home_stats = [float(g.get(bdl_key, 0) or 0) for g in prior_games if g.get("is_home")]
    away_stats = [float(g.get(bdl_key, 0) or 0) for g in prior_games if not g.get("is_home")]
    home_avg = _avg(home_stats) if home_stats else avg_stat
    away_avg = _avg(away_stats) if away_stats else avg_stat

    opp_team_id = current_game.get("opp_team_id")
    vs_opp = [g for g in prior_games if g.get("opp_team_id") == opp_team_id] if opp_team_id else []
    vs_opp_avg = _avg([float(g.get(bdl_key, 0) or 0) for g in vs_opp]) if vs_opp else avg_stat

    rest_days = 2
    if prior_games:
        last_date = prior_games[-1].get("game_date", "")
        cur_date = current_game.get("game_date", "")
        if last_date and cur_date:
            try:
                ld = date.fromisoformat(str(last_date)[:10])
                cd = date.fromisoformat(str(cur_date)[:10])
                rest_days = (cd - ld).days
            except (ValueError, TypeError):
                pass

    is_b2b = 1 if rest_days <= 1 else 0

    games_last_7 = 0
    cur_date_str = current_game.get("game_date", "")
    if cur_date_str:
        try:
            cd = date.fromisoformat(str(cur_date_str)[:10])
            for g in reversed(prior_games[-7:]):
                gd = date.fromisoformat(str(g.get("game_date", ""))[:10])
                if (cd - gd).days <= 7:
                    games_last_7 += 1
                else:
                    break
        except (ValueError, TypeError):
            pass

    # Derived rate features
    stat_per_min = avg_stat / max(avg_min, 1) if avg_min > 0 else 0
    last5_stat_per_min = last5 / max(last5_min, 1) if last5_min > 0 else stat_per_min

    # Variance features from last 10 games
    window = all_stat[-10:]
    if len(window) >= 3:
        max_last10 = max(window)
        min_last10 = min(window)
        pct_above_avg = sum(1 for v in window if v > avg_stat) / len(window)
    else:
        max_last10 = last10 + std_stat
        min_last10 = max(last10 - std_stat, 0)
        pct_above_avg = 0.5
    range_last10 = max_last10 - min_last10
    cv_stat = std_stat / max(avg_stat, 0.1) if avg_stat > 0 else 0

    # Use season avg as a dummy line for the model
    dummy_line = round(avg_stat * 2) / 2
    if dummy_line <= 0:
        dummy_line = 0.5

    return {
        # These keys match what _build_feature_row produces for inference
        PROP_TO_PG_KEY.get(prop_type, "pts_pg"): avg_stat,
        "pts_pg": _avg(stat_vals("pts")),
        "reb_pg": _avg(stat_vals("reb")),
        "ast_pg": _avg(stat_vals("ast")),
        "mpg": avg_min,
        "games_played": len(prior_games),
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
        "rest_days": rest_days,
        "is_b2b": is_b2b,
        "games_last_7": games_last_7,
        "travel_distance": 0,
        "fatigue_score": 0.15 if is_b2b else 0.0,
        # New variance/derived features
        "max_last10": max_last10,
        "min_last10": min_last10,
        "range_last10": range_last10,
        "pct_above_avg": pct_above_avg,
        "cv_stat": cv_stat,
        "stat_per_min": stat_per_min,
        "last5_stat_per_min": last5_stat_per_min,
        "game_log_count": len(prior_games),
        "name": current_game.get("player_name", ""),
        "_dummy_line": dummy_line,
        "_avg_stat": avg_stat,
        "_last10": last10,
    }


async def run_calibration(
    seasons: list[int] | None = None,
    prop_types: list[str] | None = None,
    min_games_history: int = 15,
    min_minutes: float = 12.0,
    sample_pct: float = 1.0,
    progress_callback=None,
) -> dict:
    """
    Run walk-forward calibration: predict every player/game/prop,
    compare predicted value to actual result.

    No lines involved — pure prediction accuracy vs reality.
    """
    import random

    from app.services.smart_predictor import get_smart_predictor

    predictor = get_smart_predictor()
    if not predictor.is_trained:
        return {"status": "error", "message": "Model not trained. Run retrain first."}

    if prop_types is None:
        prop_types = list(PROP_TO_BDL.keys())

    if progress_callback:
        await progress_callback(2, "Loading historical game data...")

    raw_logs = _load_bdl_seasons(seasons)
    if not raw_logs:
        return {"status": "error", "message": "No BDL cached data found."}

    logger.info(f"Calibration: loaded {len(raw_logs)} total box scores")

    # Parse into per-player game logs
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

        player_games[pid].append({
            "game_date": game_date_str[:10],
            "game_id": game.get("id"),
            "player_id": pid,
            "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
            "team_id": team_id,
            "team_abbr": team.get("abbreviation", ""),
            "is_home": is_home,
            "opp_team_id": opp_team_id,
            "min_parsed": minutes,
            "pts": row.get("pts", 0) or 0,
            "reb": row.get("reb", 0) or 0,
            "ast": row.get("ast", 0) or 0,
            "fg3m": row.get("fg3m", 0) or 0,
            "stl": row.get("stl", 0) or 0,
            "blk": row.get("blk", 0) or 0,
            "turnover": row.get("turnover", 0) or 0,
            "plus_minus": row.get("plus_minus", 0) or 0,
        })

    for pid in player_games:
        player_games[pid].sort(key=lambda g: g["game_date"])

    total_players = len(player_games)
    total_games = sum(len(v) for v in player_games.values())
    logger.info(f"Calibration: {total_players} players, {total_games} games")
    print(f"\n{'='*60}")
    print(f"CALIBRATION: {total_players} players, {total_games:,} games")
    print(f"{'='*60}")

    if progress_callback:
        await progress_callback(8, f"Indexed {total_players} players, {total_games:,} games. Starting evaluation...")

    # Sample players if requested
    player_ids = list(player_games.keys())
    if sample_pct < 1.0:
        random.seed(42)
        player_ids = random.sample(player_ids, max(1, int(len(player_ids) * sample_pct)))

    # ── Walk-forward evaluation ──
    results: list[dict] = []
    processed = 0
    skipped = 0

    import time as _time
    _t0 = _time.time()

    for pi, pid in enumerate(player_ids):
        games = player_games[pid]

        if pi % 50 == 0:
            elapsed = _time.time() - _t0
            pct = int((pi / max(len(player_ids), 1)) * 100)
            # Running MAE so far
            if results:
                running_mae = sum(r["abs_error"] for r in results) / len(results)
                running_baseline_mae = sum(abs(r["baseline_error"]) for r in results) / len(results)
                improvement = ((running_baseline_mae - running_mae) / max(running_baseline_mae, 0.01)) * 100
                print(f"  [{pct:3d}%] Player {pi+1}/{len(player_ids)} | {processed:,} preds | MAE={running_mae:.3f} | Baseline={running_baseline_mae:.3f} | +{improvement:.1f}% better | {elapsed:.0f}s")
            else:
                print(f"  [{pct:3d}%] Player {pi+1}/{len(player_ids)} | starting... | {elapsed:.0f}s")

        if progress_callback and pi % 100 == 0:
            pct = 8 + int((pi / max(len(player_ids), 1)) * 85)
            await progress_callback(pct, f"Player {pi+1}/{len(player_ids)} · {processed:,} predictions evaluated")

        for gi in range(min_games_history, len(games)):
            game = games[gi]
            if game.get("min_parsed", 0) < min_minutes:
                continue

            prior = games[:gi]

            for prop_type in prop_types:
                bdl_key = PROP_TO_BDL[prop_type]
                actual_value = float(game.get(bdl_key, 0) or 0)

                features = _build_calibration_features(prior, game, prop_type)
                if features is None:
                    skipped += 1
                    continue

                dummy_line = features.pop("_dummy_line")
                avg_stat = features.pop("_avg_stat")
                last10 = features.pop("_last10")

                try:
                    pred_result = predictor.predict_prop(features, prop_type, dummy_line)
                except Exception:
                    skipped += 1
                    continue

                predicted = pred_result.get("predicted_value", 0)
                error = predicted - actual_value

                results.append({
                    "prop_type": prop_type,
                    "predicted": round(predicted, 2),
                    "actual": actual_value,
                    "error": round(error, 2),
                    "abs_error": round(abs(error), 2),
                    "player_name": game.get("player_name", ""),
                    "game_date": game.get("game_date", ""),
                    "is_home": game.get("is_home", False),
                    "rest_days": features.get("rest_days", 2),
                    "mpg": round(features.get("mpg", 0), 1),
                    "games_played": len(prior),
                    "avg_stat": round(avg_stat, 2),
                    "last10": round(last10, 2),
                    # Also store a simple baseline: last-10 avg as prediction
                    "baseline_error": round(last10 - actual_value, 2),
                })
                processed += 1

    if progress_callback:
        await progress_callback(95, f"Computing diagnostics from {processed:,} predictions...")

    elapsed = _time.time() - _t0
    print(f"\n{'='*60}")
    print(f"CALIBRATION COMPLETE: {processed:,} predictions in {elapsed:.0f}s")
    if results:
        final_mae = sum(r["abs_error"] for r in results) / len(results)
        final_baseline = sum(abs(r["baseline_error"]) for r in results) / len(results)
        improvement = ((final_baseline - final_mae) / max(final_baseline, 0.01)) * 100
        print(f"  Model MAE:    {final_mae:.3f}")
        print(f"  Baseline MAE: {final_baseline:.3f}")
        print(f"  Improvement:  {improvement:+.1f}%")
    print(f"{'='*60}\n")
    logger.info(f"Calibration: {processed:,} predictions, {skipped:,} skipped")

    report = _compute_diagnostics(results)

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_DIR / "calibration_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    # Save sample of raw results
    with open(CALIBRATION_DIR / "calibration_results.json", "w") as f:
        json.dump(results[-10000:], f, indent=2, default=str)

    if progress_callback:
        await progress_callback(100, f"Done! {processed:,} predictions evaluated.")

    return report


def _compute_diagnostics(results: list[dict]) -> dict:
    """Compute comprehensive accuracy diagnostics — predicted vs actual."""

    total = len(results)
    if total == 0:
        return {"status": "error", "message": "No predictions generated"}

    # ── Overall accuracy ──
    all_errors = [r["error"] for r in results]
    all_abs = [r["abs_error"] for r in results]
    baseline_abs = [abs(r["baseline_error"]) for r in results]

    overall_mae = _avg(all_abs)
    overall_rmse = math.sqrt(_avg([e**2 for e in all_errors]))
    overall_bias = _avg(all_errors)
    baseline_mae = _avg(baseline_abs)

    within_1 = sum(1 for e in all_abs if e <= 1.0) / total
    within_2 = sum(1 for e in all_abs if e <= 2.0) / total
    within_3 = sum(1 for e in all_abs if e <= 3.0) / total
    within_5 = sum(1 for e in all_abs if e <= 5.0) / total

    # ── Per-prop accuracy ──
    prop_results: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        prop_results[r["prop_type"]].append(r)

    by_prop = {}
    for prop_type, preds in prop_results.items():
        errs = [r["error"] for r in preds]
        abs_errs = [r["abs_error"] for r in preds]
        base_abs = [abs(r["baseline_error"]) for r in preds]
        sorted_abs = sorted(abs_errs)

        mae = _avg(abs_errs)
        rmse = math.sqrt(_avg([e**2 for e in errs]))
        bias = _avg(errs)
        base_mae = _avg(base_abs)
        median_ae = sorted_abs[len(sorted_abs) // 2] if sorted_abs else 0

        w1 = sum(1 for e in abs_errs if e <= 1.0) / len(preds)
        w2 = sum(1 for e in abs_errs if e <= 2.0) / len(preds)
        w3 = sum(1 for e in abs_errs if e <= 3.0) / len(preds)
        w5 = sum(1 for e in abs_errs if e <= 5.0) / len(preds)

        by_prop[prop_type] = {
            "count": len(preds),
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "median_ae": round(median_ae, 3),
            "bias": round(bias, 3),
            "bias_direction": "over-predicts" if bias > 0.1 else ("under-predicts" if bias < -0.1 else "neutral"),
            "within_1": round(w1, 4),
            "within_2": round(w2, 4),
            "within_3": round(w3, 4),
            "within_5": round(w5, 4),
            "baseline_mae": round(base_mae, 3),
            "improvement_vs_baseline": round((base_mae - mae) / base_mae * 100, 1) if base_mae > 0 else 0,
        }

    # ── MPG bucket analysis ──
    mpg_buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        mpg = r.get("mpg", 0)
        if mpg >= 30:
            mpg_buckets["30+ mpg (stars)"].append(r)
        elif mpg >= 24:
            mpg_buckets["24-30 mpg (starters)"].append(r)
        elif mpg >= 18:
            mpg_buckets["18-24 mpg (rotation)"].append(r)
        else:
            mpg_buckets["<18 mpg (bench)"].append(r)

    mpg_analysis = {}
    for bucket, preds in mpg_buckets.items():
        abs_errs = [r["abs_error"] for r in preds]
        mpg_analysis[bucket] = {
            "mae": round(_avg(abs_errs), 3),
            "count": len(preds),
        }

    # ── Games played analysis ──
    gp_buckets: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        gp = r.get("games_played", 0)
        if gp >= 50:
            gp_buckets["50+ games"].append(r)
        elif gp >= 30:
            gp_buckets["30-50 games"].append(r)
        elif gp >= 20:
            gp_buckets["20-30 games"].append(r)
        else:
            gp_buckets["10-20 games"].append(r)

    gp_analysis = {}
    for bucket, preds in gp_buckets.items():
        abs_errs = [r["abs_error"] for r in preds]
        gp_analysis[bucket] = {
            "mae": round(_avg(abs_errs), 3),
            "count": len(preds),
        }

    # ── Home/away + B2B splits ──
    home = [r for r in results if r.get("is_home")]
    away = [r for r in results if not r.get("is_home")]
    b2b = [r for r in results if r.get("rest_days", 2) <= 1]
    rested = [r for r in results if r.get("rest_days", 2) >= 2]

    # ── Worst predictions ──
    worst = sorted(results, key=lambda r: r["abs_error"], reverse=True)[:20]

    # ── Best predicted players (lowest MAE, min 20 predictions) ──
    player_errors: dict[str, list[float]] = defaultdict(list)
    for r in results:
        player_errors[r["player_name"]].append(r["abs_error"])

    best_players = []
    worst_players = []
    for name, errs in player_errors.items():
        if len(errs) >= 20:
            entry = {"player": name, "mae": round(_avg(errs), 3), "predictions": len(errs)}
            best_players.append(entry)
            worst_players.append(entry)

    best_players.sort(key=lambda x: x["mae"])
    worst_players.sort(key=lambda x: -x["mae"])

    return {
        "status": "completed",
        "total_predictions": total,
        "overall": {
            "mae": round(overall_mae, 3),
            "rmse": round(overall_rmse, 3),
            "bias": round(overall_bias, 3),
            "bias_direction": "over-predicts" if overall_bias > 0.1 else ("under-predicts" if overall_bias < -0.1 else "neutral"),
            "within_1": round(within_1, 4),
            "within_2": round(within_2, 4),
            "within_3": round(within_3, 4),
            "within_5": round(within_5, 4),
            "baseline_mae": round(baseline_mae, 3),
            "improvement_vs_baseline_pct": round((baseline_mae - overall_mae) / baseline_mae * 100, 1) if baseline_mae > 0 else 0,
        },
        "by_prop_type": by_prop,
        "mpg_analysis": mpg_analysis,
        "games_played_analysis": gp_analysis,
        "splits": {
            "home_mae": round(_avg([r["abs_error"] for r in home]), 3) if home else 0,
            "away_mae": round(_avg([r["abs_error"] for r in away]), 3) if away else 0,
            "home_count": len(home),
            "away_count": len(away),
            "b2b_mae": round(_avg([r["abs_error"] for r in b2b]), 3) if b2b else 0,
            "rested_mae": round(_avg([r["abs_error"] for r in rested]), 3) if rested else 0,
            "b2b_count": len(b2b),
            "rested_count": len(rested),
        },
        "best_predicted_players": best_players[:15],
        "worst_predicted_players": worst_players[:15],
        "worst_individual_predictions": worst[:15],
    }
