#!/usr/bin/env python3
"""
Train a line-aware over/under classification model.

Instead of predicting raw stat values, this model predicts P(over | features, line).
It learns when Vegas lines are mispriced by combining player features with
line-relative features.

Training data: historical prop lines (The Odds API) + actual box scores (BDL).
Output: Calibrated probability that the player goes OVER the line.

Usage:
    poetry run python -u train_line_model.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss, roc_auc_score,
)
from xgboost import XGBClassifier

# ── Config ──
CACHE_DIR = Path(__file__).parent / "app" / "cache"
HIST_DIR = CACHE_DIR / "historical_odds"
MODEL_DIR = Path(__file__).parent / "app" / "models" / "artifacts" / "line_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

PROP_BDL_KEY = {
    "points": "pts", "rebounds": "reb", "assists": "ast",
    "threes": "fg3m", "steals": "stl", "blocks": "blk",
    "turnovers": "turnover",
}


def parse_min(m) -> float:
    if not m or m in ("0", "00", ""):
        return 0.0
    try:
        if ":" in str(m):
            p = str(m).split(":")
            return float(p[0]) + float(p[1]) / 60
        return float(m)
    except (ValueError, IndexError):
        return 0.0


# ── Feature columns for the line model ──
# These combine player performance features with line-relative features
LINE_FEATURE_COLS = [
    # Player performance (from BDL history)
    "avg_stat",
    "last3_stat",
    "last5_stat",
    "last10_stat",
    "trend_stat",
    "std_stat",
    "cv_stat",
    "mpg",
    "games_played",
    # Consistency
    "max_last10",
    "min_last10",
    "range_last10",
    "pct_above_avg",
    # Home/away
    "home_avg_stat",
    "away_avg_stat",
    "split_for_game",
    # Matchup
    "vs_opp_avg_stat",
    "vs_opp_games",
    # Minutes
    "last3_min",
    "last5_min",
    # Rates
    "stat_per_min",
    "last5_stat_per_min",
    # Schedule
    "is_home",
    "is_b2b",
    "rest_days",
    # Momentum
    "last3_vs_last10",
    "last5_vs_season",
    "streak_direction",
    # Opponent defense
    "opp_stat_allowed",
    "opp_pts_allowed",
    "pace_factor",
    "usage_rate",
    "starter_pct",
    # ── LINE-RELATIVE FEATURES (the key innovation) ──
    "line",                    # the actual sportsbook line
    "avg_minus_line",          # season avg - line (positive = avg above line)
    "last5_minus_line",        # recent form vs line
    "last10_minus_line",       # medium-term vs line
    "split_minus_line",        # home/away split vs line
    "vs_opp_minus_line",       # matchup history vs line
    "line_vs_avg_pct",         # (line - avg) / avg (how far line deviates from avg)
    "pct_games_over_line",     # % of recent games where player exceeded this line
    "consecutive_overs",       # streak of consecutive games over the line
    "line_difficulty",         # line / (avg + 0.1) — how hard the line is relative to avg
]


def main():
    print("=" * 60, flush=True)
    print("  LINE-AWARE OVER/UNDER CLASSIFICATION MODEL", flush=True)
    print("=" * 60, flush=True)

    # 1. Load BDL data
    print("\n[1/5] Loading BDL box scores...", flush=True)
    all_rows = []
    for f in sorted(CACHE_DIR.glob("bdl_season_*.json")):
        data = json.loads(f.read_text())
        all_rows.extend(data)
        print(f"  {f.name}: {len(data):,} rows", flush=True)
    print(f"  Total: {len(all_rows):,} rows", flush=True)

    # Index by player
    player_games: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        p = row.get("player", {})
        g = row.get("game", {})
        t = row.get("team", {})
        name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip().lower()
        gd = g.get("date", "")[:10]
        mins = parse_min(row.get("min", "0"))
        if mins <= 0 or not name or not gd:
            continue

        home_team_id = g.get("home_team_id")
        visitor_team_id = g.get("visitor_team_id")
        team_id = t.get("id")
        is_home = (team_id == home_team_id)
        opp_team_id = visitor_team_id if is_home else home_team_id

        player_games[name].append({
            "game_date": gd,
            "team_id": team_id,
            "is_home": is_home,
            "opp_team_id": opp_team_id,
            "min": mins,
            "pts": row.get("pts", 0) or 0,
            "reb": row.get("reb", 0) or 0,
            "ast": row.get("ast", 0) or 0,
            "fg3m": row.get("fg3m", 0) or 0,
            "stl": row.get("stl", 0) or 0,
            "blk": row.get("blk", 0) or 0,
            "turnover": row.get("turnover", 0) or 0,
        })

    for name in player_games:
        player_games[name].sort(key=lambda g: g["game_date"])

    print(f"  {len(player_games)} players indexed", flush=True)

    # 2. Compute team defense stats
    print("\n[2/5] Computing team defensive stats...", flush=True)
    game_team_totals = defaultdict(lambda: defaultdict(int))
    game_info = {}
    for row in all_rows:
        game = row.get("game", {})
        team = row.get("team", {})
        gid = game.get("id")
        tid = team.get("id")
        if not gid or not tid:
            continue
        gd = game.get("date", "")[:10]
        try:
            yr, mo = int(gd[:4]), int(gd[5:7])
            season = yr if mo >= 10 else yr - 1
        except (ValueError, IndexError):
            continue
        if season != 2025:
            continue
        for sk in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
            game_team_totals[(gid, tid)][sk] += (row.get(sk, 0) or 0)
        if gid not in game_info:
            game_info[gid] = {
                "date": gd,
                "home_team_id": game.get("home_team_id"),
                "visitor_team_id": game.get("visitor_team_id"),
                "home_score": game.get("home_team_score", 0) or 0,
                "visitor_score": game.get("visitor_team_score", 0) or 0,
            }

    team_allowed = defaultdict(lambda: {"pts": 0, "reb": 0, "ast": 0, "fg3m": 0,
                                         "stl": 0, "blk": 0, "turnover": 0,
                                         "games": 0, "wins": 0, "pts_scored": 0})
    for gid, gi in game_info.items():
        htid = gi["home_team_id"]
        vtid = gi["visitor_team_id"]
        if not htid or not vtid:
            continue
        v_tot = game_team_totals.get((gid, vtid), {})
        h_tot = game_team_totals.get((gid, htid), {})
        if v_tot:
            ts = team_allowed[htid]
            for sk in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
                ts[sk] += v_tot.get(sk, 0)
            ts["games"] += 1
            ts["pts_scored"] += h_tot.get("pts", 0)
            if gi["home_score"] > gi["visitor_score"]:
                ts["wins"] += 1
        if h_tot:
            ts = team_allowed[vtid]
            for sk in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
                ts[sk] += h_tot.get(sk, 0)
            ts["games"] += 1
            ts["pts_scored"] += v_tot.get("pts", 0)
            if gi["visitor_score"] > gi["home_score"]:
                ts["wins"] += 1

    print(f"  {len(team_allowed)} teams", flush=True)

    # 3. Load historical odds and build training data
    print("\n[3/5] Building training dataset from historical lines + actuals...", flush=True)
    rows = []
    targets = []
    skipped_no_history = 0
    skipped_no_actual = 0
    matched = 0

    for f in sorted(HIST_DIR.glob("props_*.json")):
        data = json.loads(f.read_text())
        game_date = data.get("date", "")
        if not game_date:
            continue

        for prop in data.get("props", []):
            prop_type = prop.get("prop_type", "")
            bdl_key = PROP_BDL_KEY.get(prop_type)
            if not bdl_key:
                continue

            name = prop.get("player", "").strip().lower()
            line = prop.get("line", 0)
            if line <= 0 or not name:
                continue

            # Get player's games
            games = player_games.get(name, [])
            prior = [g for g in games if g["game_date"] < game_date and g["min"] > 0]
            if len(prior) < 10:
                skipped_no_history += 1
                continue

            # Get actual result
            actual_games = [g for g in games if g["game_date"][:10] == game_date and g["min"] > 0]
            if not actual_games:
                skipped_no_actual += 1
                continue

            actual = actual_games[0]
            actual_value = actual[bdl_key]

            # Skip pushes
            if actual_value == line:
                continue

            target = 1 if actual_value > line else 0

            # Build features
            n = len(prior)
            vals = [g[bdl_key] for g in prior]
            mins_arr = [g["min"] for g in prior]

            avg_stat = sum(vals) / n
            mpg = sum(mins_arr) / n
            l3 = sum(vals[-3:]) / min(3, n)
            l5 = sum(vals[-5:]) / min(5, n)
            l10 = sum(vals[-10:]) / min(10, n)
            trend = l5 - avg_stat
            std_s = statistics.stdev(vals[-20:]) if len(vals) >= 5 else 0.0
            cv = std_s / max(avg_stat, 0.1) if avg_stat > 0 else 0

            l3m = sum(mins_arr[-3:]) / min(3, n)
            l5m = sum(mins_arr[-5:]) / min(5, n)

            is_home = actual.get("is_home", False)
            home_vals = [g[bdl_key] for g in prior if g.get("is_home")]
            away_vals = [g[bdl_key] for g in prior if not g.get("is_home")]
            h_avg = sum(home_vals) / max(len(home_vals), 1) if home_vals else avg_stat
            a_avg = sum(away_vals) / max(len(away_vals), 1) if away_vals else avg_stat
            split = h_avg if is_home else a_avg

            opp_id = actual.get("opp_team_id")
            vs_vals = [g[bdl_key] for g in prior if g.get("opp_team_id") == opp_id]
            vs_avg = sum(vs_vals) / len(vs_vals) if vs_vals else avg_stat

            window = vals[-10:]
            mx = max(window) if window else l10
            mn = min(window) if window else 0
            pa = sum(1 for v in window if v > avg_stat) / max(len(window), 1)

            spm = avg_stat / max(mpg, 1) if mpg > 0 else 0
            l5spm = l5 / max(l5m, 1) if l5m > 0 else spm

            try:
                gd = date.fromisoformat(game_date)
                pd_ = date.fromisoformat(prior[-1]["game_date"])
                rd = (gd - pd_).days
            except (ValueError, TypeError):
                rd = 2
            b2b = 1 if rd <= 1 else 0

            # Opponent defense
            opp_ts = team_allowed.get(opp_id, {})
            own_ts = team_allowed.get(actual.get("team_id"), {})
            opp_g = max(opp_ts.get("games", 0), 1)
            own_g = max(own_ts.get("games", 0), 1)
            opp_stat_allowed = opp_ts.get(bdl_key, 0) / opp_g if opp_ts else 0
            opp_pts_allowed = opp_ts.get("pts", 0) / opp_g if opp_ts else 112.0
            own_pts_scored = own_ts.get("pts_scored", 0) / own_g if own_ts else 112.0
            opp_pts_scored = opp_ts.get("pts_scored", 0) / opp_g if opp_ts else 112.0
            pace_factor = (own_pts_scored + opp_pts_scored) / 220.0 if (own_pts_scored + opp_pts_scored) > 0 else 1.0
            usage_rate = (sum(g["pts"] for g in prior) / n / max(own_pts_scored, 1)) * 100 if own_pts_scored > 0 else 20.0
            starter_count = sum(1 for g in prior if g["min"] >= 20)
            starter_pct = starter_count / n

            # ── LINE-RELATIVE FEATURES ──
            avg_minus_line = avg_stat - line
            last5_minus_line = l5 - line
            last10_minus_line = l10 - line
            split_minus_line = split - line
            vs_opp_minus_line = vs_avg - line
            line_vs_avg_pct = (line - avg_stat) / max(avg_stat, 0.1)
            pct_games_over_line = sum(1 for v in vals[-20:] if v > line) / min(20, len(vals))

            # Consecutive overs streak
            consec = 0
            for v in reversed(vals):
                if v > line:
                    consec += 1
                else:
                    break

            line_difficulty = line / max(avg_stat + 0.1, 0.1)

            feat = {
                "avg_stat": avg_stat,
                "last3_stat": l3,
                "last5_stat": l5,
                "last10_stat": l10,
                "trend_stat": trend,
                "std_stat": std_s,
                "cv_stat": cv,
                "mpg": mpg,
                "games_played": n,
                "max_last10": mx,
                "min_last10": mn,
                "range_last10": mx - mn,
                "pct_above_avg": pa,
                "home_avg_stat": h_avg,
                "away_avg_stat": a_avg,
                "split_for_game": split,
                "vs_opp_avg_stat": vs_avg,
                "vs_opp_games": len(vs_vals),
                "last3_min": l3m,
                "last5_min": l5m,
                "stat_per_min": spm,
                "last5_stat_per_min": l5spm,
                "is_home": 1.0 if is_home else 0.0,
                "is_b2b": float(b2b),
                "rest_days": rd,
                "last3_vs_last10": l3 - l10,
                "last5_vs_season": l5 - avg_stat,
                "streak_direction": 1.0 if l3 > l10 else (-1.0 if l3 < l10 else 0.0),
                "opp_stat_allowed": opp_stat_allowed,
                "opp_pts_allowed": opp_pts_allowed,
                "pace_factor": pace_factor,
                "usage_rate": usage_rate,
                "starter_pct": starter_pct,
                # Line-relative
                "line": line,
                "avg_minus_line": avg_minus_line,
                "last5_minus_line": last5_minus_line,
                "last10_minus_line": last10_minus_line,
                "split_minus_line": split_minus_line,
                "vs_opp_minus_line": vs_opp_minus_line,
                "line_vs_avg_pct": line_vs_avg_pct,
                "pct_games_over_line": pct_games_over_line,
                "consecutive_overs": consec,
                "line_difficulty": line_difficulty,
            }

            rows.append(feat)
            targets.append(target)
            matched += 1

    print(f"  Matched: {matched:,} prop lines with actuals", flush=True)
    print(f"  Skipped (no history): {skipped_no_history:,}", flush=True)
    print(f"  Skipped (no actual): {skipped_no_actual:,}", flush=True)
    print(f"  Over rate: {sum(targets)/len(targets)*100:.1f}%", flush=True)

    # 4. Train model
    print("\n[4/5] Training XGBoost classifier...", flush=True)
    X = pd.DataFrame(rows)[LINE_FEATURE_COLS].fillna(0)
    y = np.array(targets)

    # Time-based split: last 20% for test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  Train: {len(X_train):,} rows, Test: {len(X_test):,} rows", flush=True)
    print(f"  Train over rate: {y_train.mean()*100:.1f}%, Test over rate: {y_test.mean()*100:.1f}%", flush=True)

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        min_child_weight=10,
        gamma=0.1,
        random_state=42,
        verbosity=0,
        use_label_encoder=False,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # Calibrate probabilities
    print("  Calibrating probabilities (isotonic regression)...", flush=True)
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    # Evaluate
    raw_probs = model.predict_proba(X_test)[:, 1]
    cal_probs = calibrated.predict_proba(X_test)[:, 1]

    print(f"\n  Raw XGBoost:", flush=True)
    print(f"    Accuracy:    {accuracy_score(y_test, (raw_probs > 0.5).astype(int))*100:.1f}%", flush=True)
    print(f"    AUC:         {roc_auc_score(y_test, raw_probs):.4f}", flush=True)
    print(f"    Brier:       {brier_score_loss(y_test, raw_probs):.4f}", flush=True)
    print(f"    Log loss:    {log_loss(y_test, raw_probs):.4f}", flush=True)

    print(f"\n  Calibrated:", flush=True)
    print(f"    Accuracy:    {accuracy_score(y_test, (cal_probs > 0.5).astype(int))*100:.1f}%", flush=True)
    print(f"    AUC:         {roc_auc_score(y_test, cal_probs):.4f}", flush=True)
    print(f"    Brier:       {brier_score_loss(y_test, cal_probs):.4f}", flush=True)
    print(f"    Log loss:    {log_loss(y_test, cal_probs):.4f}", flush=True)

    # Analyze by confidence bucket
    print(f"\n  Calibration by predicted probability:", flush=True)
    print(f"  {'P(over) bin':<15} {'Count':>6} {'Actual%':>8} {'Profit@-110':>12}", flush=True)
    print(f"  {'-'*45}", flush=True)

    for lo, hi in [(0.0, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65), (0.65, 1.0)]:
        mask = (cal_probs >= lo) & (cal_probs < hi)
        if mask.sum() == 0:
            continue
        actual_rate = y_test[mask].mean()
        count = mask.sum()

        # Simulate profit: bet over when P>0.55, under when P<0.45
        profit = 0
        for p, actual in zip(cal_probs[mask], y_test[mask]):
            if p >= 0.55:  # bet over
                profit += 0.909 if actual == 1 else -1  # -110 odds
            elif p <= 0.45:  # bet under
                profit += 0.909 if actual == 0 else -1

        print(f"  {lo:.2f}-{hi:.2f}       {count:>6} {actual_rate*100:>7.1f}% ${profit:>+10.1f}", flush=True)

    # Feature importance
    print(f"\n  Top 15 features by importance:", flush=True)
    importances = model.feature_importances_
    feat_imp = sorted(zip(LINE_FEATURE_COLS, importances), key=lambda x: -x[1])
    for fname, imp in feat_imp[:15]:
        print(f"    {fname:<25} {imp:.4f}", flush=True)

    # 5. Save model
    print("\n[5/5] Saving model...", flush=True)
    joblib.dump(calibrated, MODEL_DIR / "line_classifier.joblib")
    joblib.dump(LINE_FEATURE_COLS, MODEL_DIR / "line_feature_cols.joblib")
    # Also save the raw model for feature importance analysis
    joblib.dump(model, MODEL_DIR / "line_classifier_raw.joblib")
    print(f"  Saved to {MODEL_DIR}", flush=True)

    # Simulate a realistic betting strategy on test set
    print(f"\n{'='*60}", flush=True)
    print(f"  SIMULATED BETTING (test set, -110 odds)", flush=True)
    print(f"{'='*60}", flush=True)

    for threshold in [0.53, 0.55, 0.57, 0.60]:
        bets = 0
        wins = 0
        profit = 0
        for p, actual in zip(cal_probs, y_test):
            if p >= threshold:  # bet over
                bets += 1
                if actual == 1:
                    wins += 1
                    profit += 100 / 110  # win at -110
                else:
                    profit -= 1
            elif p <= (1 - threshold):  # bet under
                bets += 1
                if actual == 0:
                    wins += 1
                    profit += 100 / 110
                else:
                    profit -= 1

        if bets > 0:
            hit = wins / bets * 100
            roi = profit / bets * 100
            print(f"  Threshold {threshold:.0%}: {bets} bets, {wins}W-{bets-wins}L ({hit:.1f}%), ROI={roi:+.1f}%, P&L=${profit:+.1f}", flush=True)

    print(f"\n  (Need >52.4% hit rate at -110 to be profitable)", flush=True)


if __name__ == "__main__":
    main()
