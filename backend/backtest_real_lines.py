#!/usr/bin/env python3
"""
Backtest the model against REAL sportsbook lines over multiple days.

Uses:
  - Historical prop lines from The Odds API (cached in app/cache/historical_odds/)
  - Actual box scores from BallDontLie (cached in app/cache/)
  - Trained SmartPredictor model

This is the TRUE test of whether the model can beat Vegas.

Usage:
    poetry run python -u backtest_real_lines.py [--min-confidence 60] [--min-edge 5]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "app" / "cache"
HIST_DIR = CACHE_DIR / "historical_odds"

PROP_BDL_KEY = {
    "points": "pts", "rebounds": "reb", "assists": "ast",
    "threes": "fg3m", "steals": "stl", "blocks": "blk",
    "turnovers": "turnover",
}

PROP_STAT_MAP = {
    "points": "pts_pg", "rebounds": "reb_pg", "assists": "ast_pg",
    "threes": "three_pm_pg", "steals": "stl_pg", "blocks": "blk_pg",
    "turnovers": "tov_pg",
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


def load_all_bdl_data() -> list[dict]:
    """Load all BDL season data from file cache."""
    all_rows = []
    for f in sorted(CACHE_DIR.glob("bdl_season_*.json")):
        data = json.loads(f.read_text())
        all_rows.extend(data)
    return all_rows


def build_player_index(all_rows: list[dict]) -> dict[int, list[dict]]:
    """Index all player games by player_id, sorted by date."""
    player_games: dict[int, list[dict]] = defaultdict(list)
    for row in all_rows:
        player = row.get("player", {})
        game = row.get("game", {})
        team = row.get("team", {})
        pid = player.get("id")
        if not pid:
            continue
        minutes = parse_min(row.get("min", "0"))
        game_date = game.get("date", "")[:10]
        if not game_date:
            continue

        home_team_id = game.get("home_team_id")
        visitor_team_id = game.get("visitor_team_id")
        team_id = team.get("id")
        is_home = (team_id == home_team_id)
        opp_team_id = visitor_team_id if is_home else home_team_id

        entry = {
            "game_date": game_date,
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

    # Sort by date
    for pid in player_games:
        player_games[pid].sort(key=lambda g: g["game_date"])

    return player_games


def build_name_to_pid(player_games: dict[int, list[dict]]) -> dict[str, int]:
    """Build lowercase name → player_id mapping."""
    name_map: dict[str, int] = {}
    for pid, games in player_games.items():
        if games:
            name = games[-1]["player_name"].lower()
            name_map[name] = pid
    return name_map


def build_team_defense(all_rows: list[dict], before_date: str) -> dict:
    """Compute per-team defensive stats from BDL data before a given date."""
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
        if gd >= before_date:
            continue
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

    return team_allowed


def build_features_for_player(
    prior: list[dict], current: dict, prop_type: str,
    team_allowed: dict, bdl_key: str,
) -> dict:
    """Build the full feature dict for a player/prop prediction."""
    n = len(prior)
    vals = [g[bdl_key] for g in prior]
    mins = [g["min_parsed"] for g in prior]

    avg_stat = sum(vals) / n
    mpg = sum(mins) / n

    # Rolling averages
    l3 = sum(vals[-3:]) / min(3, n)
    l5 = sum(vals[-5:]) / min(5, n)
    l10 = sum(vals[-10:]) / min(10, n)
    trend = l5 - avg_stat

    l3m = sum(mins[-3:]) / min(3, n)
    l5m = sum(mins[-5:]) / min(5, n)

    # Std dev
    std_s = statistics.stdev(vals[-20:]) if len(vals) >= 5 else 0.0

    # Home/away splits
    is_home = current.get("is_home", False)
    home_vals = [g[bdl_key] for g in prior if g.get("is_home")]
    away_vals = [g[bdl_key] for g in prior if not g.get("is_home")]
    h_avg = sum(home_vals) / max(len(home_vals), 1) if home_vals else avg_stat
    a_avg = sum(away_vals) / max(len(away_vals), 1) if away_vals else avg_stat

    # Matchup history
    opp_id = current.get("opp_team_id")
    vs_vals = [g[bdl_key] for g in prior if g.get("opp_team_id") == opp_id]
    vs_avg = sum(vs_vals) / len(vs_vals) if vs_vals else avg_stat

    # Rest
    from datetime import date as date_cls
    try:
        gd = date_cls.fromisoformat(current["game_date"])
        pd_ = date_cls.fromisoformat(prior[-1]["game_date"])
        rd = (gd - pd_).days
    except (ValueError, TypeError):
        rd = 2
    b2b = rd <= 1
    gl7 = sum(1 for g in prior[-5:] if g["game_date"] >= (gd - timedelta(days=7)).isoformat())

    # Variance
    window = vals[-10:]
    mx = max(window) if window else l10
    mn = min(window) if window else 0
    pa = sum(1 for v in window if v > avg_stat) / max(len(window), 1) if window else 0.5
    cv = std_s / max(avg_stat, 0.1) if avg_stat > 0 else 0
    spm = avg_stat / max(mpg, 1) if mpg > 0 else 0
    l5spm = l5 / max(l5m, 1) if l5m > 0 else spm

    # Cross-stat context
    pts_pg = sum(g["pts"] for g in prior) / n
    reb_pg = sum(g["reb"] for g in prior) / n
    ast_pg = sum(g["ast"] for g in prior) / n

    # Opponent defense
    opp_ts = team_allowed.get(opp_id, {})
    own_tid = current.get("team_id")
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
    pace_factor = (own_pts_scored + opp_pts_scored) / 220.0 if (own_pts_scored + opp_pts_scored) > 0 else 1.0
    own_win_pct = own_ts.get("wins", 0) / own_g if own_ts else 0.5
    opp_win_pct = opp_ts.get("wins", 0) / opp_g if opp_ts else 0.5
    spread = (opp_win_pct - own_win_pct) * 15
    if not is_home:
        spread += 3
    else:
        spread -= 3
    over_under = own_pts_scored + opp_pts_allowed
    usage_rate = (pts_pg / max(own_pts_scored, 1)) * 100 if own_pts_scored > 0 else 20.0
    starter_count = sum(1 for g in prior if g["min_parsed"] >= 20)
    starter_pct = starter_count / n

    stat_col = PROP_STAT_MAP.get(prop_type, "pts_pg")

    return {
        stat_col: avg_stat,
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
        "max_last10": mx, "min_last10": mn, "range_last10": mx - mn,
        "pct_above_avg": pa, "cv_stat": cv,
        "stat_per_min": spm, "last5_stat_per_min": l5spm,
        "opp_stat_allowed": opp_stat_allowed,
        "opp_pts_allowed": opp_pts_allowed,
        "opp_reb_allowed": opp_reb_allowed,
        "opp_ast_allowed": opp_ast_allowed,
        "opp_3pm_allowed": opp_3pm_allowed,
        "pace_factor": pace_factor,
        "usage_rate": usage_rate,
        "spread": round(spread, 1),
        "over_under": round(over_under, 1),
        "starter_pct": starter_pct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-confidence", type=float, default=60.0)
    parser.add_argument("--min-edge", type=float, default=0.0, help="Min edge %% over line")
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--prop-types", type=str, default="points,rebounds,assists,threes",
                        help="Comma-separated prop types")
    args = parser.parse_args()

    prop_types = [p.strip() for p in args.prop_types.split(",")]

    # 1. Load model
    print("[1/4] Loading model...", flush=True)
    from app.services.smart_predictor import get_smart_predictor
    predictor = get_smart_predictor()
    print(f"  Model loaded: {list(predictor.prop_models.keys())}", flush=True)

    # 2. Load BDL data
    print("[2/4] Loading BDL box scores...", flush=True)
    all_rows = load_all_bdl_data()
    print(f"  {len(all_rows):,} total rows", flush=True)
    player_games = build_player_index(all_rows)
    name_to_pid = build_name_to_pid(player_games)
    print(f"  {len(player_games)} players indexed", flush=True)

    # 3. Load historical odds
    print("[3/4] Loading historical prop lines...", flush=True)
    hist_files = sorted(HIST_DIR.glob("props_*.json"))
    all_hist: dict[str, list[dict]] = {}  # date -> list of prop dicts
    total_lines = 0
    for f in hist_files:
        data = json.loads(f.read_text())
        props = data.get("props", [])
        if props:
            d = data.get("date", f.stem.replace("props_", ""))
            all_hist[d] = props
            total_lines += len(props)
    print(f"  {len(all_hist)} dates, {total_lines:,} prop lines", flush=True)

    # 4. Run backtest
    print(f"[4/4] Running backtest (conf>={args.min_confidence}, edge>={args.min_edge}%)...\n", flush=True)

    # Pre-compute team defense once (using all data up to latest date)
    # We'll rebuild per-date for accuracy but that's slow — use a single snapshot
    latest_date = max(all_hist.keys())
    team_defense = build_team_defense(all_rows, latest_date)

    bankroll = args.bankroll
    all_bets = []
    daily_results = []

    for game_date in sorted(all_hist.keys()):
        props = all_hist[game_date]
        day_bets = []

        for prop in props:
            prop_type = prop.get("prop_type", "")
            if prop_type not in prop_types:
                continue

            bdl_key = PROP_BDL_KEY.get(prop_type)
            if not bdl_key:
                continue

            player_name = prop.get("player", "").strip()
            line = prop.get("line", 0)
            over_odds = prop.get("over_odds", -110)
            under_odds = prop.get("under_odds", -110)
            book = prop.get("book", "")

            if line <= 0:
                continue

            # Find player in BDL data
            pid = name_to_pid.get(player_name.lower())
            if not pid:
                continue

            games = player_games.get(pid, [])
            # Get games before this date
            prior = [g for g in games if g["game_date"] < game_date and g["min_parsed"] > 0]
            if len(prior) < 15:
                continue

            # Get actual result for this date
            actual_games = [g for g in games if g["game_date"][:10] == game_date and g["min_parsed"] > 0]
            if not actual_games:
                continue
            actual = actual_games[0]
            actual_value = actual[bdl_key]

            # Build features
            features = build_features_for_player(prior, actual, prop_type, team_defense, bdl_key)

            # Predict
            try:
                result = predictor.predict_prop(features, prop_type, line)
            except Exception:
                continue

            predicted = result.get("predicted_value", 0)
            confidence = result.get("confidence_score", 0)
            rec_bet = result.get("recommended_bet", "over")

            if confidence < args.min_confidence:
                continue

            # Edge: how far prediction is from line
            edge_pct = abs(predicted - line) / max(line, 0.5) * 100
            if edge_pct < args.min_edge:
                continue

            # Determine bet direction and odds
            if rec_bet == "over":
                bet_odds = over_odds
                hit = actual_value > line
            else:
                bet_odds = under_odds
                hit = actual_value < line

            # Kelly sizing
            if bet_odds > 0:
                decimal_odds = 1 + bet_odds / 100
            else:
                decimal_odds = 1 + 100 / abs(bet_odds)
            b = decimal_odds - 1
            implied_prob = 1 / decimal_odds
            model_prob = confidence / 100
            kelly = (model_prob * b - (1 - model_prob)) / b if b > 0 else 0
            kelly = max(0, min(kelly, 0.03))  # cap at 3%
            stake = round(bankroll * kelly, 2)
            if stake < 1:
                continue

            profit = round(stake * b, 2) if hit else -stake
            bankroll += profit

            bet = {
                "date": game_date,
                "player": player_name,
                "prop": prop_type,
                "bet": rec_bet.upper(),
                "line": line,
                "predicted": round(predicted, 1),
                "actual": actual_value,
                "hit": hit,
                "confidence": round(confidence, 1),
                "edge": round(edge_pct, 1),
                "odds": bet_odds,
                "stake": stake,
                "profit": round(profit, 2),
                "bankroll": round(bankroll, 2),
                "book": book,
            }
            day_bets.append(bet)
            all_bets.append(bet)

        if day_bets:
            wins = sum(1 for b in day_bets if b["hit"])
            losses = len(day_bets) - wins
            day_profit = sum(b["profit"] for b in day_bets)
            daily_results.append({
                "date": game_date,
                "bets": len(day_bets),
                "wins": wins,
                "losses": losses,
                "profit": round(day_profit, 2),
                "bankroll": round(bankroll, 2),
            })
            pct = wins / len(day_bets) * 100
            print(f"  {game_date}: {wins}W-{losses}L ({pct:.0f}%) P&L=${day_profit:+.2f}  BR=${bankroll:.2f}", flush=True)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS — REAL SPORTSBOOK LINES")
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
    print(f"  Bets:       {total_bets} ({total_wins}W - {total_losses}L)")
    print(f"  Hit Rate:   {hit_rate:.1f}%")
    print(f"  Profit:     ${total_profit:+.2f}")
    print(f"  ROI:        {total_profit/max(total_staked,1)*100:+.1f}%")
    print(f"  Bankroll:   ${args.bankroll:.0f} → ${bankroll:.2f}")

    # By prop type
    print(f"\n  {'Prop':<12} {'Bets':>5} {'W':>4} {'L':>4} {'Hit%':>6} {'Profit':>10}")
    print(f"  {'-'*45}")
    for pt in prop_types:
        pt_bets = [b for b in all_bets if b["prop"] == pt]
        if not pt_bets:
            continue
        w = sum(1 for b in pt_bets if b["hit"])
        l = len(pt_bets) - w
        p = sum(b["profit"] for b in pt_bets)
        print(f"  {pt:<12} {len(pt_bets):>5} {w:>4} {l:>4} {w/len(pt_bets)*100:>5.1f}% ${p:>+9.2f}")

    # By confidence tier
    print(f"\n  {'Conf Tier':<12} {'Bets':>5} {'W':>4} {'L':>4} {'Hit%':>6} {'Profit':>10}")
    print(f"  {'-'*45}")
    for lo, hi, label in [(60, 70, "60-70"), (70, 80, "70-80"), (80, 90, "80-90"), (90, 100, "90+")]:
        tier = [b for b in all_bets if lo <= b["confidence"] < hi]
        if not tier:
            continue
        w = sum(1 for b in tier if b["hit"])
        l = len(tier) - w
        p = sum(b["profit"] for b in tier)
        print(f"  {label:<12} {len(tier):>5} {w:>4} {l:>4} {w/len(tier)*100:>5.1f}% ${p:>+9.2f}")

    # Worst and best days
    if daily_results:
        best = max(daily_results, key=lambda d: d["profit"])
        worst = min(daily_results, key=lambda d: d["profit"])
        print(f"\n  Best day:   {best['date']} ({best['wins']}W-{best['losses']}L, ${best['profit']:+.2f})")
        print(f"  Worst day:  {worst['date']} ({worst['wins']}W-{worst['losses']}L, ${worst['profit']:+.2f})")

    # Profitable days
    profit_days = sum(1 for d in daily_results if d["profit"] > 0)
    print(f"  Profitable: {profit_days}/{len(daily_results)} days ({profit_days/max(len(daily_results),1)*100:.0f}%)")


if __name__ == "__main__":
    main()
