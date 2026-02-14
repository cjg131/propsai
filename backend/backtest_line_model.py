#!/usr/bin/env python3
"""
Backtest the line-aware classification model against real sportsbook lines.

Uses the trained line_classifier to predict P(over) and simulates betting
with Kelly criterion sizing over 30 days of real historical lines.

Usage:
    poetry run python -u backtest_line_model.py [--threshold 0.53] [--kelly-frac 0.25]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent / "app" / "cache"
HIST_DIR = CACHE_DIR / "historical_odds"
MODEL_DIR = Path(__file__).parent / "app" / "models" / "artifacts" / "line_model"

PROP_BDL_KEY = {
    "points": "pts", "rebounds": "reb", "assists": "ast",
    "threes": "fg3m",
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


def build_line_features(prior, actual, bdl_key, line, team_allowed):
    """Build feature dict for the line model."""
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
        gd = date.fromisoformat(actual["game_date"][:10])
        pd_ = date.fromisoformat(prior[-1]["game_date"][:10])
        rd = (gd - pd_).days
    except (ValueError, TypeError):
        rd = 2
    b2b = 1 if rd <= 1 else 0

    opp_ts = team_allowed.get(opp_id, {})
    own_ts = team_allowed.get(actual.get("team_id"), {})
    opp_g = max(opp_ts.get("games", 0), 1)
    own_g = max(own_ts.get("games", 0), 1)
    opp_stat_allowed = opp_ts.get(bdl_key, 0) / opp_g if opp_ts else 0
    opp_pts_allowed = opp_ts.get("pts", 0) / opp_g if opp_ts else 112.0
    own_pts_scored = own_ts.get("pts_scored", 0) / own_g if own_ts else 112.0
    opp_pts_scored = opp_ts.get("pts_scored", 0) / opp_g if opp_ts else 112.0
    pace_factor = (own_pts_scored + opp_pts_scored) / 220.0 if (own_pts_scored + opp_pts_scored) > 0 else 1.0
    pts_pg = sum(g["pts"] for g in prior) / n
    usage_rate = (pts_pg / max(own_pts_scored, 1)) * 100 if own_pts_scored > 0 else 20.0
    starter_pct = sum(1 for g in prior if g["min"] >= 20) / n

    # Line-relative
    consec = 0
    for v in reversed(vals):
        if v > line:
            consec += 1
        else:
            break

    return {
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
        "line": line,
        "avg_minus_line": avg_stat - line,
        "last5_minus_line": l5 - line,
        "last10_minus_line": l10 - line,
        "split_minus_line": split - line,
        "vs_opp_minus_line": vs_avg - line,
        "line_vs_avg_pct": (line - avg_stat) / max(avg_stat, 0.1),
        "pct_games_over_line": sum(1 for v in vals[-20:] if v > line) / min(20, len(vals)),
        "consecutive_overs": consec,
        "line_difficulty": line / max(avg_stat + 0.1, 0.1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.53,
                        help="Min P(over) to bet over, or max P(over) to bet under")
    parser.add_argument("--kelly-frac", type=float, default=0.25,
                        help="Fraction of Kelly to use (0.25 = quarter Kelly)")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--max-daily-bets", type=int, default=20)
    parser.add_argument("--prop-types", type=str, default="points,rebounds,assists,threes")
    args = parser.parse_args()

    prop_types = [p.strip() for p in args.prop_types.split(",")]

    # Load model
    print("[1/4] Loading line model...", flush=True)
    model = joblib.load(MODEL_DIR / "line_classifier.joblib")
    feature_cols = joblib.load(MODEL_DIR / "line_feature_cols.joblib")
    print(f"  Loaded classifier with {len(feature_cols)} features", flush=True)

    # Load BDL
    print("[2/4] Loading BDL data...", flush=True)
    all_rows = []
    for f in sorted(CACHE_DIR.glob("bdl_season_*.json")):
        all_rows.extend(json.loads(f.read_text()))
    print(f"  {len(all_rows):,} rows", flush=True)

    # Index players
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
            "game_date": gd, "team_id": team_id,
            "is_home": is_home, "opp_team_id": opp_team_id,
            "min": mins,
            "pts": row.get("pts", 0) or 0, "reb": row.get("reb", 0) or 0,
            "ast": row.get("ast", 0) or 0, "fg3m": row.get("fg3m", 0) or 0,
            "stl": row.get("stl", 0) or 0, "blk": row.get("blk", 0) or 0,
            "turnover": row.get("turnover", 0) or 0,
        })
    for name in player_games:
        player_games[name].sort(key=lambda g: g["game_date"])

    # Team defense
    print("[3/4] Computing team defense...", flush=True)
    game_team_totals = defaultdict(lambda: defaultdict(int))
    game_info_map = {}
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
        if gid not in game_info_map:
            game_info_map[gid] = {
                "date": gd,
                "home_team_id": game.get("home_team_id"),
                "visitor_team_id": game.get("visitor_team_id"),
                "home_score": game.get("home_team_score", 0) or 0,
                "visitor_score": game.get("visitor_team_score", 0) or 0,
            }
    team_allowed = defaultdict(lambda: {"pts": 0, "reb": 0, "ast": 0, "fg3m": 0,
                                         "stl": 0, "blk": 0, "turnover": 0,
                                         "games": 0, "wins": 0, "pts_scored": 0})
    for gid, gi in game_info_map.items():
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

    # Load historical odds — ONLY out-of-sample dates (after training cutoff)
    # Training used first 80% of 23k samples → cutoff at Jan 29
    OOS_START = "2026-01-30"
    print(f"[4/4] Running backtest (out-of-sample only, >= {OOS_START})...\n", flush=True)
    hist_files = sorted(HIST_DIR.glob("props_*.json"))

    bankroll = args.bankroll
    start_bankroll = bankroll
    all_bets = []
    daily_results = []

    for f in hist_files:
        data = json.loads(f.read_text())
        game_date = data.get("date", "")
        props = data.get("props", [])
        if not props or game_date < OOS_START:
            continue

        # Score all props for this date
        candidates = []
        for prop in props:
            prop_type = prop.get("prop_type", "")
            if prop_type not in prop_types:
                continue
            bdl_key = PROP_BDL_KEY.get(prop_type)
            if not bdl_key:
                continue

            name = prop.get("player", "").strip().lower()
            line = prop.get("line", 0)
            over_odds = prop.get("over_odds", -110)
            under_odds = prop.get("under_odds", -110)
            if line <= 0 or not name:
                continue

            games = player_games.get(name, [])
            prior = [g for g in games if g["game_date"] < game_date and g["min"] > 0]
            if len(prior) < 10:
                continue

            actual_games = [g for g in games if g["game_date"][:10] == game_date and g["min"] > 0]
            if not actual_games:
                continue
            actual = actual_games[0]
            actual_value = actual[bdl_key]
            if actual_value == line:
                continue  # push

            # Build features and predict
            feat = build_line_features(prior, actual, bdl_key, line, team_allowed)
            X = pd.DataFrame([feat])[feature_cols].fillna(0)
            p_over = model.predict_proba(X)[0, 1]

            candidates.append({
                "name": prop.get("player", ""),
                "prop_type": prop_type,
                "line": line,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "actual": actual_value,
                "p_over": p_over,
                "book": prop.get("book", ""),
            })

        # Filter and rank by edge
        bettable = []
        for c in candidates:
            p = c["p_over"]
            if p >= args.threshold:
                # Bet over
                odds = c["over_odds"]
                if odds > 0:
                    dec = 1 + odds / 100
                else:
                    dec = 1 + 100 / abs(odds)
                implied = 1 / dec
                edge = p - implied
                if edge > 0:
                    bettable.append({**c, "direction": "OVER", "odds": odds,
                                     "dec_odds": dec, "edge": edge, "model_prob": p})
            elif p <= (1 - args.threshold):
                # Bet under
                odds = c["under_odds"]
                if odds > 0:
                    dec = 1 + odds / 100
                else:
                    dec = 1 + 100 / abs(odds)
                implied = 1 / dec
                p_under = 1 - p
                edge = p_under - implied
                if edge > 0:
                    bettable.append({**c, "direction": "UNDER", "odds": odds,
                                     "dec_odds": dec, "edge": edge, "model_prob": p_under})

        # Sort by edge, take top N
        bettable.sort(key=lambda x: -x["edge"])
        bettable = bettable[:args.max_daily_bets]

        day_bets = []
        for b in bettable:
            # Kelly sizing
            dec = b["dec_odds"]
            p_win = b["model_prob"]
            b_val = dec - 1
            kelly = (p_win * b_val - (1 - p_win)) / b_val if b_val > 0 else 0
            kelly = max(0, kelly) * args.kelly_frac
            kelly = min(kelly, 0.03)  # cap at 3%
            stake = round(bankroll * kelly, 2)
            if stake < 1:
                continue

            # Resolve
            if b["direction"] == "OVER":
                hit = b["actual"] > b["line"]
            else:
                hit = b["actual"] < b["line"]

            profit = round(stake * (dec - 1), 2) if hit else -stake
            bankroll = round(bankroll + profit, 2)

            day_bets.append({
                "date": game_date,
                "player": b["name"],
                "prop": b["prop_type"],
                "direction": b["direction"],
                "line": b["line"],
                "actual": b["actual"],
                "p_over": round(b["p_over"], 3),
                "edge": round(b["edge"] * 100, 1),
                "odds": b["odds"],
                "hit": hit,
                "stake": stake,
                "profit": profit,
                "bankroll": bankroll,
            })
            all_bets.append(day_bets[-1])

        if day_bets:
            wins = sum(1 for b in day_bets if b["hit"])
            losses = len(day_bets) - wins
            day_profit = sum(b["profit"] for b in day_bets)
            daily_results.append({
                "date": game_date, "bets": len(day_bets),
                "wins": wins, "losses": losses,
                "profit": round(day_profit, 2), "bankroll": bankroll,
            })
            pct = wins / len(day_bets) * 100
            print(f"  {game_date}: {wins}W-{losses}L ({pct:5.1f}%) "
                  f"P&L=${day_profit:+8.2f}  BR=${bankroll:8.2f}", flush=True)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS — LINE-AWARE MODEL vs REAL SPORTSBOOK LINES")
    print(f"{'='*70}")

    if not all_bets:
        print("  No bets placed!")
        return

    total_bets = len(all_bets)
    total_wins = sum(1 for b in all_bets if b["hit"])
    total_losses = total_bets - total_wins
    total_profit = sum(b["profit"] for b in all_bets)
    total_staked = sum(b["stake"] for b in all_bets)
    hit_rate = total_wins / total_bets * 100

    print(f"\n  Period:     {min(b['date'] for b in all_bets)} to {max(b['date'] for b in all_bets)}")
    print(f"  Days:       {len(daily_results)}")
    print(f"  Threshold:  {args.threshold:.0%} P(over/under)")
    print(f"  Kelly:      {args.kelly_frac:.0%} Kelly")
    print(f"  Bets:       {total_bets} ({total_wins}W - {total_losses}L)")
    print(f"  Hit Rate:   {hit_rate:.1f}%")
    print(f"  Total Staked: ${total_staked:.2f}")
    print(f"  Profit:     ${total_profit:+.2f}")
    print(f"  ROI:        {total_profit/max(total_staked,1)*100:+.1f}%")
    print(f"  Bankroll:   ${start_bankroll:.0f} → ${bankroll:.2f}")

    # By prop type
    print(f"\n  {'Prop':<12} {'Bets':>5} {'W':>4} {'L':>4} {'Hit%':>6} {'Profit':>10}")
    print(f"  {'-'*45}")
    for pt in prop_types:
        pt_bets = [b for b in all_bets if b["prop"] == pt]
        if not pt_bets:
            continue
        w = sum(1 for b in pt_bets if b["hit"])
        l_ = len(pt_bets) - w
        p = sum(b["profit"] for b in pt_bets)
        print(f"  {pt:<12} {len(pt_bets):>5} {w:>4} {l_:>4} {w/len(pt_bets)*100:>5.1f}% ${p:>+9.2f}")

    # By direction
    print(f"\n  {'Direction':<12} {'Bets':>5} {'W':>4} {'L':>4} {'Hit%':>6} {'Profit':>10}")
    print(f"  {'-'*45}")
    for d in ["OVER", "UNDER"]:
        d_bets = [b for b in all_bets if b["direction"] == d]
        if not d_bets:
            continue
        w = sum(1 for b in d_bets if b["hit"])
        l_ = len(d_bets) - w
        p = sum(b["profit"] for b in d_bets)
        print(f"  {d:<12} {len(d_bets):>5} {w:>4} {l_:>4} {w/len(d_bets)*100:>5.1f}% ${p:>+9.2f}")

    # Profitable days
    if daily_results:
        profit_days = sum(1 for d in daily_results if d["profit"] > 0)
        best = max(daily_results, key=lambda d: d["profit"])
        worst = min(daily_results, key=lambda d: d["profit"])
        print(f"\n  Profitable: {profit_days}/{len(daily_results)} days ({profit_days/len(daily_results)*100:.0f}%)")
        print(f"  Best day:   {best['date']} ({best['wins']}W-{best['losses']}L, ${best['profit']:+.2f})")
        print(f"  Worst day:  {worst['date']} ({worst['wins']}W-{worst['losses']}L, ${worst['profit']:+.2f})")

    # Show some example bets
    print(f"\n  Sample bets (last 10):")
    print(f"  {'Date':<12} {'Player':<22} {'Prop':<8} {'Dir':<6} {'Line':>5} {'Act':>4} {'P(ov)':>6} {'Edge':>5} {'Hit':>4} {'P&L':>8}")
    print(f"  {'-'*85}")
    for b in all_bets[-10:]:
        hit_str = "✓" if b["hit"] else "✗"
        print(f"  {b['date']:<12} {b['player'][:21]:<22} {b['prop']:<8} {b['direction']:<6} "
              f"{b['line']:>5.1f} {b['actual']:>4} {b['p_over']:>5.3f} {b['edge']:>4.1f}% "
              f"{hit_str:>4} ${b['profit']:>+7.2f}")


if __name__ == "__main__":
    main()
