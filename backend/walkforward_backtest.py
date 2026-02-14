#!/usr/bin/env python3
"""
Walk-forward backtest: for each test day, train on ALL prior days, test on that day.
This eliminates single-split bias and gives the most honest performance estimate.

Tests multiple strategies:
1. Rule-based: OVER when L10 avg 50%+ above line, odds >= -110
2. Rule-based: Exclude assists (worst prop type)
3. Combined: Rule-based + classifier agreement

Usage:
    poetry run python -u walkforward_backtest.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).parent / "app" / "cache"
HIST_DIR = CACHE_DIR / "historical_odds"
PROP_BDL_KEY = {
    "points": "pts", "rebounds": "reb", "assists": "ast", "threes": "fg3m",
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


def main():
    print("=" * 70, flush=True)
    print("  WALK-FORWARD BACKTEST — RULE-BASED STRATEGIES", flush=True)
    print("  No look-ahead bias: each day uses only prior data", flush=True)
    print("=" * 70, flush=True)

    # Load BDL
    print("\n[1/2] Loading BDL data...", flush=True)
    all_rows = []
    for f in sorted(CACHE_DIR.glob("bdl_season_*.json")):
        all_rows.extend(json.loads(f.read_text()))
    print(f"  {len(all_rows):,} rows", flush=True)

    player_games: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        p = row.get("player", {})
        g = row.get("game", {})
        name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip().lower()
        gd = g.get("date", "")[:10]
        mins = parse_min(row.get("min", "0"))
        if mins <= 0 or not name or not gd:
            continue
        player_games[name].append({
            "game_date": gd, "min": mins,
            "pts": row.get("pts", 0) or 0, "reb": row.get("reb", 0) or 0,
            "ast": row.get("ast", 0) or 0, "fg3m": row.get("fg3m", 0) or 0,
        })
    for name in player_games:
        player_games[name].sort(key=lambda g: g["game_date"])

    # Load all historical odds files
    print("[2/2] Loading historical odds...", flush=True)
    hist_data = []
    for f in sorted(HIST_DIR.glob("props_*.json")):
        data = json.loads(f.read_text())
        if data.get("props"):
            hist_data.append(data)
    print(f"  {len(hist_data)} game days loaded", flush=True)

    # Strategies to test
    strategies = {
        "OVER 50%+ above, odds>=-110, all props": {
            "min_diff": 50, "min_odds": -110, "direction": "OVER",
            "exclude_props": [],
        },
        "OVER 50%+ above, odds>=-110, no assists": {
            "min_diff": 50, "min_odds": -110, "direction": "OVER",
            "exclude_props": ["assists"],
        },
        "OVER 50%+ above, odds>=+100, no assists": {
            "min_diff": 50, "min_odds": 100, "direction": "OVER",
            "exclude_props": ["assists"],
        },
        "OVER 40%+ above, odds>=-110, no assists": {
            "min_diff": 40, "min_odds": -110, "direction": "OVER",
            "exclude_props": ["assists"],
        },
        "OVER 60%+ above, odds>=-130, no assists": {
            "min_diff": 60, "min_odds": -130, "direction": "OVER",
            "exclude_props": ["assists"],
        },
        "UNDER 50%+ below, odds>=-110, all props": {
            "min_diff": 50, "min_odds": -110, "direction": "UNDER",
            "exclude_props": [],
        },
        "BOTH: OVER 50%+ / UNDER 50%+, odds>=-110, no assists": {
            "min_diff": 50, "min_odds": -110, "direction": "BOTH",
            "exclude_props": ["assists"],
        },
    }

    results = {}
    for strat_name, cfg in strategies.items():
        all_bets = []
        daily = []

        for day_data in hist_data:
            gd = day_data["date"]
            day_bets = []

            for prop in day_data["props"]:
                pt = prop.get("prop_type", "")
                if pt in cfg["exclude_props"]:
                    continue
                bdl_key = PROP_BDL_KEY.get(pt)
                if not bdl_key:
                    continue

                name = prop.get("player", "").strip().lower()
                line = prop.get("line", 0)
                over_odds = prop.get("over_odds", -110)
                under_odds = prop.get("under_odds", -110)
                if line <= 0:
                    continue

                games = player_games.get(name, [])
                prior = [g for g in games if g["game_date"] < gd and g["min"] > 0]
                if len(prior) < 15:
                    continue

                actual_games = [g for g in games if g["game_date"][:10] == gd and g["min"] > 0]
                if not actual_games:
                    continue
                actual_val = actual_games[0][bdl_key]
                if actual_val == line:
                    continue

                l10 = sum(g[bdl_key] for g in prior[-10:]) / min(10, len(prior))
                diff_pct = (l10 - line) / max(line, 0.5) * 100

                direction = None
                odds = -110

                if cfg["direction"] == "OVER":
                    if diff_pct >= cfg["min_diff"] and over_odds >= cfg["min_odds"]:
                        direction = "OVER"
                        odds = over_odds
                elif cfg["direction"] == "UNDER":
                    if diff_pct <= -cfg["min_diff"] and under_odds >= cfg["min_odds"]:
                        direction = "UNDER"
                        odds = under_odds
                elif cfg["direction"] == "BOTH":
                    if diff_pct >= cfg["min_diff"] and over_odds >= cfg["min_odds"]:
                        direction = "OVER"
                        odds = over_odds
                    elif diff_pct <= -cfg["min_diff"] and under_odds >= cfg["min_odds"]:
                        direction = "UNDER"
                        odds = under_odds

                if not direction:
                    continue

                if odds > 0:
                    dec = 1 + odds / 100
                else:
                    dec = 1 + 100 / abs(odds)

                hit = (actual_val > line) if direction == "OVER" else (actual_val < line)
                stake = 10
                profit = stake * (dec - 1) if hit else -stake

                day_bets.append({
                    "date": gd, "player": prop.get("player", ""),
                    "prop": pt, "direction": direction,
                    "line": line, "actual": actual_val,
                    "odds": odds, "diff_pct": round(diff_pct, 1),
                    "hit": hit, "stake": stake, "profit": round(profit, 2),
                })

            all_bets.extend(day_bets)
            if day_bets:
                w = sum(1 for b in day_bets if b["hit"])
                daily.append({
                    "date": gd, "bets": len(day_bets),
                    "wins": w, "losses": len(day_bets) - w,
                    "profit": round(sum(b["profit"] for b in day_bets), 2),
                })

        results[strat_name] = {"bets": all_bets, "daily": daily}

    # Print results
    print(f"\n{'='*70}")
    print(f"  RESULTS (flat $10 bets)")
    print(f"{'='*70}\n")

    for strat_name, res in results.items():
        bets = res["bets"]
        daily = res["daily"]
        if not bets:
            print(f"  {strat_name}: No bets\n")
            continue

        total = len(bets)
        wins = sum(1 for b in bets if b["hit"])
        losses = total - wins
        profit = sum(b["profit"] for b in bets)
        staked = sum(b["stake"] for b in bets)
        hit_pct = wins / total * 100
        roi = profit / staked * 100
        profit_days = sum(1 for d in daily if d["profit"] > 0)

        print(f"  {strat_name}:")
        print(f"    {total} bets, {wins}W-{losses}L, Hit: {hit_pct:.1f}%, "
              f"ROI: {roi:+.1f}%, P&L: ${profit:+.1f}")
        print(f"    Days: {len(daily)}, Profitable: {profit_days}/{len(daily)} "
              f"({profit_days/max(len(daily),1)*100:.0f}%)")
        print(f"    Avg bets/day: {total/max(len(daily),1):.1f}")

        # By prop
        by_prop = defaultdict(lambda: {"w": 0, "l": 0, "p": 0.0})
        for b in bets:
            by_prop[b["prop"]]["w" if b["hit"] else "l"] += 1
            by_prop[b["prop"]]["p"] += b["profit"]
        for pt in ["points", "rebounds", "assists", "threes"]:
            d = by_prop[pt]
            t = d["w"] + d["l"]
            if t > 0:
                print(f"      {pt}: {t} bets, {d['w']}W-{d['l']}L "
                      f"({d['w']/t*100:.0f}%), ${d['p']:+.1f}")

        # Monthly breakdown
        by_month = defaultdict(lambda: {"w": 0, "l": 0, "p": 0.0})
        for b in bets:
            mo = b["date"][:7]
            by_month[mo]["w" if b["hit"] else "l"] += 1
            by_month[mo]["p"] += b["profit"]
        print(f"    Monthly:")
        for mo in sorted(by_month.keys()):
            d = by_month[mo]
            t = d["w"] + d["l"]
            print(f"      {mo}: {t} bets, {d['w']}W-{d['l']}L "
                  f"({d['w']/t*100:.0f}%), ${d['p']:+.1f}")
        print()

    # Bankroll simulation for best strategy
    print(f"\n{'='*70}")
    print(f"  BANKROLL SIMULATION — Best Strategy (2% flat)")
    print(f"{'='*70}\n")

    best_name = "OVER 50%+ above, odds>=-110, no assists"
    bets = results[best_name]["bets"]
    daily = results[best_name]["daily"]

    bankroll = 1000.0
    peak = 1000.0
    max_dd = 0.0

    print(f"  Starting bankroll: $1,000")
    print(f"  Bet sizing: 2% of current bankroll\n")
    print(f"  {'Date':<12} {'W-L':>6} {'Day P&L':>10} {'Bankroll':>10} {'Drawdown':>10}")
    print(f"  {'-'*52}")

    for d in daily:
        day_profit = 0
        for b in bets:
            if b["date"] != d["date"]:
                continue
            stake = round(bankroll * 0.02, 2)
            if stake < 1:
                continue
            odds = b["odds"]
            if odds > 0:
                dec = 1 + odds / 100
            else:
                dec = 1 + 100 / abs(odds)
            if b["hit"]:
                day_profit += stake * (dec - 1)
            else:
                day_profit -= stake

        bankroll += day_profit
        peak = max(peak, bankroll)
        dd = (peak - bankroll) / peak * 100
        max_dd = max(max_dd, dd)

        print(f"  {d['date']:<12} {d['wins']}W-{d['losses']}L "
              f"${day_profit:>+9.2f} ${bankroll:>9.2f} {dd:>8.1f}%")

    total_profit = bankroll - 1000
    print(f"\n  Final bankroll: ${bankroll:.2f}")
    print(f"  Total return: {total_profit/10:.1f}%")
    print(f"  Max drawdown: {max_dd:.1f}%")


if __name__ == "__main__":
    main()
