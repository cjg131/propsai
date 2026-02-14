#!/usr/bin/env python3
"""
Run calibration directly in the terminal with live output.
Usage: poetry run python -u run_calibration.py [--sample 0.2]

FAST version: precomputes cumulative sums per player so each
prediction uses O(1) rolling averages instead of rebuilding from scratch.
"""
from __future__ import annotations

import sys
import os
import json
import time
import math
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
import builtins
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

PROP_TO_BDL = {
    "points": "pts", "rebounds": "reb", "assists": "ast",
    "threes": "fg3m", "steals": "stl", "blocks": "blk", "turnovers": "turnover",
}

PROP_TO_PG_KEY = {
    "points": "pts_pg", "rebounds": "reb_pg", "assists": "ast_pg",
    "threes": "three_pm_pg", "steals": "stl_pg", "blocks": "blk_pg",
    "turnovers": "tov_pg",
}

CACHE_DIR = Path(__file__).parent / "app" / "cache"
CALIBRATION_DIR = Path(__file__).parent / "app" / "calibration"


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=1.0, help="Fraction of players to evaluate (0.0-1.0)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  CALIBRATION — Prediction Accuracy Test (FAST)")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading trained model...")
    from app.services.smart_predictor import get_smart_predictor
    predictor = get_smart_predictor()
    if not predictor.is_trained:
        print("ERROR: Model not trained. Run retrain first.")
        return
    print(f"  Model loaded: {len(predictor.prop_models)} prop types")

    # Load data
    print("\n[2/4] Loading historical game data...")
    all_rows = []
    for f in sorted(CACHE_DIR.glob("bdl_season_*.json")):
        print(f"  Loading {f.name}...", end=" ")
        with open(f) as fh:
            data = json.load(fh)
            print(f"{len(data):,} rows")
            all_rows.extend(data)
    if not all_rows:
        print("ERROR: No cached BDL data found.")
        return
    print(f"  Total: {len(all_rows):,} box scores")

    # Parse into per-player game logs
    print("\n[3/4] Indexing player games...")
    player_games: dict[int, list[dict]] = defaultdict(list)
    for row in all_rows:
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
            "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
            "is_home": is_home,
            "opp_team_id": opp_team_id,
            "min_parsed": minutes,
            "pts": float(row.get("pts", 0) or 0),
            "reb": float(row.get("reb", 0) or 0),
            "ast": float(row.get("ast", 0) or 0),
            "fg3m": float(row.get("fg3m", 0) or 0),
            "stl": float(row.get("stl", 0) or 0),
            "blk": float(row.get("blk", 0) or 0),
            "turnover": float(row.get("turnover", 0) or 0),
        })

    for pid in player_games:
        player_games[pid].sort(key=lambda g: g["game_date"])

    total_players = len(player_games)
    total_games = sum(len(v) for v in player_games.values())
    print(f"  {total_players} players, {total_games:,} games")

    # Sample if requested
    player_ids = list(player_games.keys())
    if args.sample < 1.0:
        import random
        random.seed(42)
        player_ids = random.sample(player_ids, max(1, int(len(player_ids) * args.sample)))
        print(f"  Sampled {len(player_ids)} players ({args.sample*100:.0f}%)")

    # Run evaluation — BATCH mode for speed
    # Instead of calling predict_prop() one-at-a-time (slow: creates DataFrame + runs 4 models each call),
    # we build ALL features for a prop type, then run models once in batch.
    import numpy as np
    import pandas as pd
    from app.services.smart_predictor import FEATURE_COLS

    prop_types = list(PROP_TO_BDL.keys())
    MIN_HISTORY = 15
    MIN_MINUTES = 12.0

    print(f"\n[4/4] Evaluating {len(player_ids)} players × 7 props (BATCH mode)...")
    print()

    results = []
    processed = 0
    t0 = time.time()

    for prop_type in prop_types:
        bdl_key = PROP_TO_BDL[prop_type]
        prop_data = predictor.prop_models.get(prop_type)
        if not prop_data:
            print(f"  Skipping {prop_type}: no model")
            continue

        feature_cols = prop_data["feature_cols"]
        models = prop_data["models"]
        weights = prop_data["weights"]

        # Build ALL feature rows for this prop type across all players
        all_rows = []
        all_meta = []  # (actual, last10, player_name, game_date, is_home, mpg)

        pt_start = time.time()
        for pi, pid in enumerate(player_ids):
            games = player_games[pid]
            n = len(games)
            if n < MIN_HISTORY + 1:
                continue

            # Precompute arrays for this player
            vals = [g[bdl_key] for g in games]
            min_v = [g["min_parsed"] for g in games]
            pts_v = [g["pts"] for g in games]
            reb_v = [g["reb"] for g in games]
            ast_v = [g["ast"] for g in games]
            home_f = [1 if g["is_home"] else 0 for g in games]

            # Prefix sums
            cum = [0.0] * (n + 1)
            cum_m = [0.0] * (n + 1)
            cum_p = [0.0] * (n + 1)
            cum_r = [0.0] * (n + 1)
            cum_a = [0.0] * (n + 1)
            cum_h = [0.0] * (n + 1)  # home stat sum
            cum_hc = [0] * (n + 1)    # home game count
            for j in range(n):
                cum[j+1] = cum[j] + vals[j]
                cum_m[j+1] = cum_m[j] + min_v[j]
                cum_p[j+1] = cum_p[j] + pts_v[j]
                cum_r[j+1] = cum_r[j] + reb_v[j]
                cum_a[j+1] = cum_a[j] + ast_v[j]
                cum_h[j+1] = cum_h[j] + (vals[j] if home_f[j] else 0)
                cum_hc[j+1] = cum_hc[j] + home_f[j]

            # Matchup accumulator
            opp_stats = defaultdict(list)

            for i in range(MIN_HISTORY, n):
                if min_v[i] < MIN_MINUTES:
                    # Still update matchup history
                    opp_id = games[i].get("opp_team_id")
                    if opp_id:
                        opp_stats[opp_id].append(vals[i])
                    continue

                np_ = i  # n_prior
                actual = vals[i]

                # Rolling avgs (O(1))
                sa = cum[i] / np_
                l3 = (cum[i] - cum[max(i-3,0)]) / min(3, np_)
                l5 = (cum[i] - cum[max(i-5,0)]) / min(5, np_)
                l10 = (cum[i] - cum[max(i-10,0)]) / min(10, np_)
                trend = l5 - sa

                mpg = cum_m[i] / np_
                l3m = (cum_m[i] - cum_m[max(i-3,0)]) / min(3, np_)
                l5m = (cum_m[i] - cum_m[max(i-5,0)]) / min(5, np_)

                # Std dev (last 20)
                w = vals[max(i-20,0):i]
                std_s = statistics.stdev(w) if len(w) >= 5 else 0.0

                # Home/away
                hc = cum_hc[i]
                ac = np_ - hc
                h_avg = cum_h[i] / max(hc, 1)
                a_avg = (cum[i] - cum_h[i]) / max(ac, 1)
                is_home = games[i]["is_home"]

                # Matchup
                opp_id = games[i].get("opp_team_id")
                vs_l = opp_stats.get(opp_id, [])
                vs_avg = sum(vs_l) / len(vs_l) if vs_l else sa
                vs_n = len(vs_l)

                # Update matchup for future
                if opp_id:
                    opp_stats[opp_id].append(actual)

                # Variance
                w10 = vals[max(i-10,0):i]
                if len(w10) >= 3:
                    mx = max(w10); mn = min(w10)
                    pa = sum(1 for v in w10 if v > sa) / len(w10)
                else:
                    mx = l10 + std_s; mn = max(l10 - std_s, 0); pa = 0.5
                cv = std_s / max(sa, 0.1) if sa > 0 else 0

                # Rates
                spm = sa / max(mpg, 1) if mpg > 0 else 0
                l5spm = l5 / max(l5m, 1) if l5m > 0 else spm

                # Rest
                try:
                    cd = date.fromisoformat(games[i]["game_date"][:10])
                    pd2 = date.fromisoformat(games[i-1]["game_date"][:10])
                    rd = (cd - pd2).days
                except (ValueError, TypeError):
                    rd = 2
                b2b = 1 if rd <= 1 else 0

                gl7 = 0
                try:
                    for k in range(i-1, max(i-5,-1), -1):
                        dk = date.fromisoformat(games[k]["game_date"][:10])
                        if (cd - dk).days <= 7: gl7 += 1
                        else: break
                except (ValueError, TypeError):
                    pass

                row = {
                    "avg_stat": sa, "mpg": mpg, "games_played": np_,
                    "last3_stat": l3, "last5_stat": l5, "last10_stat": l10, "trend_stat": trend,
                    "std_stat": std_s, "cv_stat": cv,
                    "max_last10": mx, "min_last10": mn, "range_last10": mx - mn, "pct_above_avg": pa,
                    "home_avg_stat": h_avg, "away_avg_stat": a_avg,
                    "home_away_diff": h_avg - a_avg,
                    "split_for_game": h_avg if is_home else a_avg,
                    "vs_opp_avg_stat": vs_avg, "vs_opp_games": vs_n,
                    "last3_min": l3m, "last5_min": l5m, "trend_min": l3m - mpg,
                    "stat_per_min": spm, "last5_stat_per_min": l5spm, "min_x_rate": l5m * spm,
                    "is_home": 1.0 if is_home else 0.0, "is_b2b": float(b2b),
                    "rest_days": rd, "games_last_7": gl7,
                    "travel_distance": 0, "fatigue_score": 0.15 if b2b else 0.0,
                    "last3_vs_last10": l3 - l10, "last5_vs_season": l5 - sa,
                    "streak_direction": 1.0 if l3 > l10 else (-1.0 if l3 < l10 else 0.0),
                    "pts_pg": cum_p[i] / np_, "reb_pg": cum_r[i] / np_, "ast_pg": cum_a[i] / np_,
                }
                all_rows.append(row)
                all_meta.append((actual, l10, games[i].get("player_name",""), games[i]["game_date"], is_home, mpg))

        if not all_rows:
            continue

        # BATCH predict: build one DataFrame, run each model once
        X = pd.DataFrame(all_rows)[feature_cols].fillna(0)

        # Get predictions from each model
        model_preds = {}
        for name, model in models.items():
            try:
                model_preds[name] = model.predict(X)
            except Exception:
                pass

        if not model_preds:
            continue

        # Weighted average
        total_w = sum(weights.get(k, 0) for k in model_preds)
        if total_w > 0:
            predicted_arr = sum(model_preds[k] * weights.get(k, 0) for k in model_preds) / total_w
        else:
            predicted_arr = np.mean(list(model_preds.values()), axis=0)

        predicted_arr = np.maximum(predicted_arr, 0.1).round(1)

        # Collect results
        for idx in range(len(all_rows)):
            actual, l10, pname, gdate, is_h, mpg_v = all_meta[idx]
            pred = float(predicted_arr[idx])
            err = pred - actual
            results.append({
                "prop_type": prop_type,
                "predicted": round(pred, 2),
                "actual": actual,
                "error": round(err, 2),
                "abs_error": round(abs(err), 2),
                "player_name": pname,
                "game_date": gdate,
                "is_home": is_h,
                "mpg": round(mpg_v, 1),
                "baseline_error": round(l10 - actual, 2),
            })
            processed += 1

        pt_elapsed = time.time() - pt_start
        mae = sum(r["abs_error"] for r in results if r["prop_type"] == prop_type) / max(len([r for r in results if r["prop_type"] == prop_type]), 1)
        base = sum(abs(r["baseline_error"]) for r in results if r["prop_type"] == prop_type) / max(len([r for r in results if r["prop_type"] == prop_type]), 1)
        imp = ((base - mae) / max(base, 0.01)) * 100
        print(f"  {prop_type:<12} {len(all_rows):>7,} preds | MAE={mae:.3f} vs Base={base:.3f} ({imp:+.1f}%) | {pt_elapsed:.1f}s")

    elapsed = time.time() - t0

    # ── Final Results ──
    print()
    print("=" * 60)
    print(f"  RESULTS: {processed:,} predictions in {elapsed:.0f}s")
    print("=" * 60)

    if not results:
        print("  No predictions generated!")
        return

    # Overall
    all_mae = sum(r["abs_error"] for r in results) / len(results)
    all_base = sum(abs(r["baseline_error"]) for r in results) / len(results)
    all_bias = sum(r["error"] for r in results) / len(results)
    within_1 = sum(1 for r in results if r["abs_error"] <= 1) / len(results) * 100
    within_2 = sum(1 for r in results if r["abs_error"] <= 2) / len(results) * 100
    within_5 = sum(1 for r in results if r["abs_error"] <= 5) / len(results) * 100
    imp = ((all_base - all_mae) / max(all_base, 0.01)) * 100

    print(f"\n  Overall:")
    print(f"    Model MAE:    {all_mae:.3f}")
    print(f"    Baseline MAE: {all_base:.3f}")
    print(f"    Improvement:  {imp:+.1f}%")
    print(f"    Bias:         {all_bias:+.3f} ({'over-predicts' if all_bias > 0 else 'under-predicts'})")
    print(f"    Within 1:     {within_1:.1f}%")
    print(f"    Within 2:     {within_2:.1f}%")
    print(f"    Within 5:     {within_5:.1f}%")

    # Per prop type
    print(f"\n  By Prop Type:")
    print(f"    {'Prop':<12} {'MAE':>6} {'Base':>6} {'Imp%':>6} {'Bias':>7} {'W/in 2':>7}")
    print(f"    {'-'*50}")
    for pt in prop_types:
        pr = [r for r in results if r["prop_type"] == pt]
        if not pr:
            continue
        mae = sum(r["abs_error"] for r in pr) / len(pr)
        base = sum(abs(r["baseline_error"]) for r in pr) / len(pr)
        bias = sum(r["error"] for r in pr) / len(pr)
        w2 = sum(1 for r in pr if r["abs_error"] <= 2) / len(pr) * 100
        imp_pt = ((base - mae) / max(base, 0.01)) * 100
        print(f"    {pt:<12} {mae:>6.2f} {base:>6.2f} {imp_pt:>+5.1f}% {bias:>+6.2f} {w2:>6.1f}%")

    # Save report
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "total_predictions": processed,
        "overall": {
            "mae": round(all_mae, 3),
            "baseline_mae": round(all_base, 3),
            "improvement_pct": round(imp, 1),
            "bias": round(all_bias, 3),
            "within_1_pct": round(within_1, 1),
            "within_2_pct": round(within_2, 1),
            "within_5_pct": round(within_5, 1),
        },
    }
    with open(CALIBRATION_DIR / "calibration_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {CALIBRATION_DIR / 'calibration_report.json'}")
    print()


if __name__ == "__main__":
    main()
