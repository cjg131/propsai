#!/usr/bin/env python3
"""
Paper trade a single day's games using the trained model.
Usage: poetry run python -u run_paper_trade.py [--date 2026-02-12]

Fetches actual box scores from BDL (live or cache), builds features
from prior games, runs predictions, and simulates bets.
"""
from __future__ import annotations

import sys
import os
import json
import time
import math
import statistics
import argparse
from collections import defaultdict
from datetime import date, timedelta
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="Game date (YYYY-MM-DD), default=yesterday")
    parser.add_argument("--min-confidence", type=float, default=60.0, help="Min confidence to bet")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Starting bankroll")
    parser.add_argument("--real-lines", type=str, default=None, help="Path to JSON file with real sportsbook lines")
    args = parser.parse_args()

    # Load real sportsbook lines if provided
    real_lines = {}
    if args.real_lines:
        with open(args.real_lines) as f:
            real_lines = json.load(f)
        print(f"  Using {len(real_lines)} REAL sportsbook lines from {args.real_lines}")

    target_date = args.date or (date.today() - timedelta(days=1)).isoformat()

    print("=" * 60)
    print(f"  PAPER TRADE — {target_date}")
    print(f"  Bankroll: ${args.bankroll:.0f} | Min confidence: {args.min_confidence}")
    print("=" * 60)

    # 1. Load model
    print("\n[1/5] Loading trained model...")
    from app.services.smart_predictor import get_smart_predictor
    predictor = get_smart_predictor()
    print(f"  Model loaded: {list(predictor.prop_models.keys())}")

    # 2. Load ALL historical game data (for building features)
    print("\n[2/5] Loading historical game data...")
    cache_dir = Path(__file__).parent / "app" / "cache"
    all_rows = []
    for f in sorted(cache_dir.glob("bdl_season_*.json")):
        data = json.load(open(f))
        all_rows.extend(data)
        print(f"  {f.name}: {len(data):,} rows")
    print(f"  Total: {len(all_rows):,} box scores")

    # 3. Get actuals for target date
    print(f"\n[3/5] Getting box scores for {target_date}...")

    # Check cache first
    target_rows = []
    for row in all_rows:
        g = row.get("game", {})
        if g.get("date") == target_date:
            target_rows.append(row)

    if target_rows:
        print(f"  Found {len(target_rows)} box scores in cache")
    else:
        # Fetch live from BDL
        print(f"  Not in cache — fetching live from BDL API...")
        from app.services.balldontlie import get_balldontlie
        bdl = get_balldontlie()
        target_rows = bdl.get_team_game_logs_by_date(target_date)
        print(f"  Fetched {len(target_rows)} box scores from BDL API")

    if not target_rows:
        print(f"\n  ERROR: No box scores found for {target_date}")
        print("  (Games may not have been played or data not yet available)")
        return

    # 4. Index ALL player games for feature building
    print(f"\n[4/5] Building player histories...")
    player_games = defaultdict(list)
    for row in all_rows:
        player = row.get("player", {})
        game = row.get("game", {})
        team = row.get("team", {})
        game_date_str = game.get("date", "")
        if not game_date_str:
            continue

        minutes = parse_min(row.get("min", "0"))
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
        }
        player_games[pid].append(entry)

    # Also index the live-fetched target rows if they weren't in cache
    for row in target_rows:
        player = row.get("player", {})
        game = row.get("game", {})
        team = row.get("team", {})
        pid = player.get("id")
        if not pid:
            continue
        # Check if already indexed
        existing_dates = set(g["game_date"] for g in player_games.get(pid, []))
        if target_date in existing_dates:
            continue

        home_team_id = game.get("home_team_id")
        visitor_team_id = game.get("visitor_team_id")
        team_id = team.get("id")
        is_home = (team_id == home_team_id)
        opp_team_id = visitor_team_id if is_home else home_team_id

        entry = {
            "game_date": target_date,
            "player_id": pid,
            "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
            "team_id": team_id,
            "team_abbr": team.get("abbreviation", ""),
            "is_home": is_home,
            "opp_team_id": opp_team_id,
            "min_parsed": parse_min(row.get("min", "0")),
            "pts": row.get("pts", 0) or 0,
            "reb": row.get("reb", 0) or 0,
            "ast": row.get("ast", 0) or 0,
            "fg3m": row.get("fg3m", 0) or 0,
            "stl": row.get("stl", 0) or 0,
            "blk": row.get("blk", 0) or 0,
            "turnover": row.get("turnover", 0) or 0,
        }
        player_games[pid].append(entry)

    # Sort all games by date
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g["game_date"])

    # Build actuals lookup from target rows
    actuals = {}
    for row in target_rows:
        player = row.get("player", {})
        name = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        minutes = parse_min(row.get("min", "0"))
        if minutes < 5:
            continue
        actuals[name.lower()] = {
            "name": name,
            "team": row.get("team", {}).get("abbreviation", ""),
            "min": minutes,
            "pts": row.get("pts", 0) or 0,
            "reb": row.get("reb", 0) or 0,
            "ast": row.get("ast", 0) or 0,
            "fg3m": row.get("fg3m", 0) or 0,
            "stl": row.get("stl", 0) or 0,
            "blk": row.get("blk", 0) or 0,
            "turnover": row.get("turnover", 0) or 0,
        }

    print(f"  {len(player_games)} players indexed, {len(actuals)} played on {target_date}")

    # 4b. Compute per-team season stats from BDL data (opponent defense)
    print(f"  Computing team defensive stats from historical data...")
    from collections import Counter
    game_team_pts = defaultdict(lambda: defaultdict(int))  # (game_id, team_id) -> {stat: total}
    game_info = {}
    for row in all_rows:
        game = row.get("game", {})
        team = row.get("team", {})
        gid = game.get("id")
        tid = team.get("id")
        if not gid or not tid:
            continue
        gd = game.get("date", "")[:10]
        if gd >= target_date:
            continue  # only use pre-game data
        for sk in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
            game_team_pts[(gid, tid)][sk] += (row.get(sk, 0) or 0)
        if gid not in game_info:
            game_info[gid] = {
                "date": gd,
                "home_team_id": game.get("home_team_id"),
                "visitor_team_id": game.get("visitor_team_id"),
                "home_score": game.get("home_team_score", 0) or 0,
                "visitor_score": game.get("visitor_team_score", 0) or 0,
            }

    # Aggregate: what each team allowed per game this season
    team_allowed = defaultdict(lambda: {"pts": 0, "reb": 0, "ast": 0, "fg3m": 0,
                                         "stl": 0, "blk": 0, "turnover": 0,
                                         "games": 0, "wins": 0, "pts_scored": 0})
    for gid, gi in game_info.items():
        d = gi["date"]
        try:
            yr, mo = int(d[:4]), int(d[5:7])
            season = yr if mo >= 10 else yr - 1
        except (ValueError, IndexError):
            continue
        # Only use current season
        if season != 2025:
            continue
        htid = gi["home_team_id"]
        vtid = gi["visitor_team_id"]
        if not htid or not vtid:
            continue
        v_tot = game_team_pts.get((gid, vtid), {})
        h_tot = game_team_pts.get((gid, htid), {})
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

    print(f"  Computed defensive stats for {len(team_allowed)} teams")

    # 5. Generate predictions and simulate bets
    print(f"\n[5/5] Running predictions and simulating bets...")
    print()

    MIN_HISTORY = 15
    bets = []
    all_preds = []
    bankroll = args.bankroll

    for player_key, actual_stats in actuals.items():
        # Find this player's ID
        pid = None
        for p_id, games in player_games.items():
            for g in games:
                if g["player_name"].lower() == player_key and g["game_date"] == target_date:
                    pid = p_id
                    break
            if pid:
                break

        if not pid:
            continue

        games = player_games[pid]
        # Find index of target game
        target_idx = None
        for i, g in enumerate(games):
            if g["game_date"] == target_date:
                target_idx = i
                break

        if target_idx is None or target_idx < MIN_HISTORY:
            continue

        prior = games[:target_idx]
        current = games[target_idx]
        n = len(prior)

        for prop_type, bdl_key in PROP_TO_BDL.items():
            actual_value = actual_stats.get(bdl_key, 0)

            # Build features from prior games
            vals = [g[bdl_key] for g in prior]
            mins = [g["min_parsed"] for g in prior]

            sa = sum(vals) / n
            mpg = sum(mins) / n
            l3 = sum(vals[-3:]) / min(3, n)
            l5 = sum(vals[-5:]) / min(5, n)
            l10 = sum(vals[-10:]) / min(10, n)
            trend = l5 - sa

            l3m = sum(mins[-3:]) / min(3, n)
            l5m = sum(mins[-5:]) / min(5, n)

            # Std dev
            std_s = statistics.stdev(vals[-20:]) if len(vals) >= 5 else 0.0

            # Home/away
            h_vals = [vals[j] for j in range(n) if prior[j]["is_home"]]
            a_vals = [vals[j] for j in range(n) if not prior[j]["is_home"]]
            h_avg = sum(h_vals) / len(h_vals) if h_vals else sa
            a_avg = sum(a_vals) / len(a_vals) if a_vals else sa
            is_home = current["is_home"]

            # Matchup
            opp_id = current.get("opp_team_id")
            vs_vals = [vals[j] for j in range(n) if prior[j].get("opp_team_id") == opp_id] if opp_id else []
            vs_avg = sum(vs_vals) / len(vs_vals) if vs_vals else sa

            # Variance
            w10 = vals[-10:]
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
                cd = date.fromisoformat(current["game_date"][:10])
                pd2 = date.fromisoformat(prior[-1]["game_date"][:10])
                rd = (cd - pd2).days
            except (ValueError, TypeError):
                rd = 2
            b2b = 1 if rd <= 1 else 0

            gl7 = 0
            try:
                for k in range(len(prior) - 1, max(len(prior) - 5, -1), -1):
                    dk = date.fromisoformat(prior[k]["game_date"][:10])
                    if (cd - dk).days <= 7:
                        gl7 += 1
                    else:
                        break
            except (ValueError, TypeError):
                pass

            pts_pg = sum(g["pts"] for g in prior) / n
            reb_pg = sum(g["reb"] for g in prior) / n
            ast_pg = sum(g["ast"] for g in prior) / n

            # Use real sportsbook line if available, otherwise synthetic
            line_key = f"{player_key}|{prop_type}"
            if real_lines:
                line_info = real_lines.get(line_key)
                if not line_info:
                    continue  # Skip props without real lines
                line = line_info["line"]
                bet_odds = line_info.get("odds", -110)
                book = line_info.get("book", "")
            else:
                line = round(l10 * 2) / 2
                if line <= 0:
                    line = 0.5
                bet_odds = -110
                book = "synthetic"

            # Opponent defense from BDL-computed team stats
            opp_id = current.get("opp_team_id")
            own_tid = current.get("team_id")
            opp_ts = team_allowed.get(opp_id, {})
            own_ts = team_allowed.get(own_tid, {})
            opp_g = max(opp_ts.get("games", 0), 1)
            own_g = max(own_ts.get("games", 0), 1)
            opp_pts_allowed = opp_ts.get("pts", 0) / opp_g if opp_ts else 112.0
            opp_reb_allowed = opp_ts.get("reb", 0) / opp_g if opp_ts else 44.0
            opp_ast_allowed = opp_ts.get("ast", 0) / opp_g if opp_ts else 25.0
            opp_3pm_allowed = opp_ts.get("fg3m", 0) / opp_g if opp_ts else 12.0
            opp_stat_allowed = opp_ts.get(bdl_key, 0) / opp_g if opp_ts else 0

            own_pts_scored = own_ts.get("pts_scored", 0) / own_g if own_ts else 112.0
            opp_pts_scored = opp_ts.get("pts_scored", 0) / opp_g if opp_ts else 112.0

            # Pace proxy
            pace_factor = (own_pts_scored + opp_pts_scored) / 220.0 if (own_pts_scored + opp_pts_scored) > 0 else 1.0
            # Spread proxy from win%
            own_win_pct = own_ts.get("wins", 0) / own_g if own_ts else 0.5
            opp_win_pct = opp_ts.get("wins", 0) / opp_g if opp_ts else 0.5
            spread = (opp_win_pct - own_win_pct) * 15
            if not is_home:
                spread += 3
            else:
                spread -= 3
            over_under = own_pts_scored + opp_pts_allowed

            # Usage proxy
            usage_rate = (pts_pg / max(own_pts_scored, 1)) * 100 if own_pts_scored > 0 else 20.0
            # Starter proxy
            starter_count = sum(1 for g in prior if g["min_parsed"] >= 20)
            starter_pct = starter_count / n

            player_dict = {
                "pts_pg": pts_pg, "reb_pg": reb_pg, "ast_pg": ast_pg,
                "mpg": mpg, "games_played": n,
                f"last3_{bdl_key}": l3, f"last5_{bdl_key}": l5,
                f"last10_{bdl_key}": l10, f"trend_{bdl_key}": trend,
                f"std_{bdl_key}": std_s,
                f"home_avg_{bdl_key}": h_avg, f"away_avg_{bdl_key}": a_avg,
                "is_home": is_home,
                f"vs_opp_avg_{bdl_key}": vs_avg, "vs_opp_games": len(vs_vals),
                "last3_min": l3m, "last5_min": l5m, "trend_min": l3m - mpg,
                "rest_days": rd, "is_b2b": b2b, "games_last_7": gl7,
                "travel_distance": 0, "fatigue_score": 0.15 if b2b else 0.0,
                "game_log_count": n,
                # Variance features
                "max_last10": mx, "min_last10": mn, "range_last10": mx - mn,
                "pct_above_avg": pa, "cv_stat": cv,
                "stat_per_min": spm, "last5_stat_per_min": l5spm,
                # NEW: Opponent / game context features
                "opp_stat_allowed": opp_stat_allowed,
                "opp_pts_allowed": opp_pts_allowed,
                "opp_reb_allowed": opp_reb_allowed,
                "opp_ast_allowed": opp_ast_allowed,
                "opp_3pm_allowed": opp_3pm_allowed,
                "pace_factor": pace_factor,
                "usage_rate": usage_rate,
                "spread": spread,
                "over_under": over_under,
                "starter_pct": starter_pct,
            }

            try:
                result = predictor.predict_prop(player_dict, prop_type, line)
            except Exception:
                continue

            predicted = result.get("predicted_value", 0)
            confidence = result.get("confidence_score", 0)
            over_prob = result.get("over_probability", 0.5)
            rec_bet = result.get("recommended_bet", "over")

            all_preds.append({
                "player": actual_stats["name"],
                "team": actual_stats["team"],
                "prop": prop_type,
                "line": line,
                "predicted": predicted,
                "actual": actual_value,
                "confidence": confidence,
                "rec_bet": rec_bet,
                "over_prob": over_prob,
            })

            # Only bet if confidence meets threshold
            if confidence < args.min_confidence:
                continue

            # Check if bet hits
            if rec_bet == "over":
                hit = actual_value > line
            else:
                hit = actual_value < line

            # Push
            if actual_value == line:
                continue

            # Convert American odds to decimal
            if bet_odds > 0:
                decimal_odds = 1 + bet_odds / 100
            else:
                decimal_odds = 1 + 100 / abs(bet_odds)
            b = decimal_odds - 1

            # Kelly sizing
            bet_prob = over_prob if rec_bet == "over" else (1.0 - over_prob)
            kelly = (b * bet_prob - (1 - bet_prob)) / b
            kelly = max(kelly, 0) * 0.5  # half-Kelly
            if kelly <= 0:
                continue

            stake = min(bankroll * kelly, bankroll * 0.03)
            stake = min(stake, args.bankroll * 0.10)
            stake = round(stake, 2)
            if stake < 1:
                continue

            profit = round(stake * b, 2) if hit else round(-stake, 2)
            bankroll += profit

            bets.append({
                "player": actual_stats["name"],
                "team": actual_stats["team"],
                "prop": prop_type,
                "line": line,
                "predicted": round(predicted, 1),
                "actual": actual_value,
                "bet": rec_bet.upper(),
                "hit": hit,
                "confidence": round(confidence, 1),
                "stake": stake,
                "profit": profit,
            })

    # Results
    print("=" * 80)
    print(f"  PAPER TRADE RESULTS — {target_date}")
    print("=" * 80)

    if not bets:
        print("\n  No qualifying bets found.")
        print(f"  Total predictions generated: {len(all_preds)}")
        return

    wins = sum(1 for b in bets if b["hit"])
    losses = len(bets) - wins
    hit_rate = wins / len(bets)
    total_profit = sum(b["profit"] for b in bets)
    total_staked = sum(b["stake"] for b in bets)
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    print(f"\n  Bets:       {len(bets)} ({wins}W - {losses}L)")
    print(f"  Hit Rate:   {hit_rate:.1%}")
    print(f"  Profit:     ${total_profit:+.2f}")
    print(f"  ROI:        {roi:+.1f}%")
    print(f"  Bankroll:   ${args.bankroll:.0f} → ${bankroll:.2f}")
    print()

    # By prop type
    print(f"  {'Prop':<12} {'Bets':>5} {'W':>4} {'L':>4} {'Hit%':>7} {'Profit':>9}")
    print(f"  {'-'*47}")
    for prop_type in PROP_TO_BDL:
        pb = [b for b in bets if b["prop"] == prop_type]
        if not pb:
            continue
        pw = sum(1 for b in pb if b["hit"])
        pl = len(pb) - pw
        phr = pw / len(pb)
        pp = sum(b["profit"] for b in pb)
        print(f"  {prop_type:<12} {len(pb):>5} {pw:>4} {pl:>4} {phr:>6.1%} ${pp:>+8.2f}")

    # Show individual bets
    print(f"\n  {'Player':<22} {'Prop':<10} {'Bet':<6} {'Line':>6} {'Pred':>6} {'Actual':>6} {'Hit':>4} {'Conf':>5} {'P&L':>8}")
    print(f"  {'-'*85}")
    for b in sorted(bets, key=lambda x: -abs(x["profit"])):
        hit_str = "✓" if b["hit"] else "✗"
        print(f"  {b['player']:<22} {b['prop']:<10} {b['bet']:<6} {b['line']:>6.1f} {b['predicted']:>6.1f} {b['actual']:>6} {hit_str:>4} {b['confidence']:>5.1f} ${b['profit']:>+7.2f}")

    print()


if __name__ == "__main__":
    main()
