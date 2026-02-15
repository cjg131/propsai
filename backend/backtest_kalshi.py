"""
Backtest our model against settled Kalshi NBA player prop markets.

For each settled market:
  1. Look up the player's BDL game logs (only games BEFORE the market date)
  2. Compute rolling features from those logs
  3. Run our model's predict_prop to get our over/under probability
  4. Compare to Kalshi's implied probability (from last_price)
  5. Simulate buying YES or NO when we see an edge, calculate P&L

Usage:
    poetry run python -u backtest_kalshi.py [--min-edge 5] [--min-volume 50]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.services.smart_predictor import SmartPredictor, PROP_BDL_KEY
from app.logging_config import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path(__file__).parent / "app" / "cache" / "kalshi_history"
BDL_CACHE = Path(__file__).parent / "app" / "cache"

# Map Kalshi prop types to our model's prop types
KALSHI_TO_MODEL_PROP = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "blocks": "blocks",
    "steals": "steals",
}

# Prop type -> BDL box score key
PROP_TO_BDL_STAT = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "blocks": "blk",
    "steals": "stl",
    "threes": "fg3m",
}


def parse_minutes(min_str) -> float:
    if not min_str or min_str in ("0", "00", ""):
        return 0.0
    try:
        if ":" in str(min_str):
            parts = str(min_str).split(":")
            return float(parts[0]) + float(parts[1]) / 60
        return float(min_str)
    except (ValueError, IndexError):
        return 0.0


def load_bdl_season_data() -> list[dict]:
    """Load BDL 2025 season box scores from cache."""
    path = BDL_CACHE / "bdl_season_2025.json"
    if not path.exists():
        print(f"ERROR: BDL cache not found at {path}")
        print("Run the backend server first to populate the cache.")
        sys.exit(1)
    data = json.loads(path.read_text())
    print(f"Loaded {len(data)} BDL box scores for 2025 season")
    return data


def build_player_index(bdl_data: list[dict]) -> dict[str, list[dict]]:
    """
    Index BDL data by player name (lowercase).
    Returns {name: [game_logs sorted by date]}.
    """
    player_games: dict[str, list[dict]] = defaultdict(list)
    for row in bdl_data:
        if parse_minutes(row.get("min", "0")) <= 0:
            continue
        p = row.get("player", {})
        name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip().lower()
        if name:
            player_games[name].append(row)

    # Sort by date
    for name in player_games:
        player_games[name].sort(key=lambda g: g.get("game", {}).get("date", ""))

    print(f"Indexed {len(player_games)} players from BDL data")
    return dict(player_games)


def compute_features_at_date(
    game_logs: list[dict],
    before_date: str,
    prop_type: str,
) -> dict | None:
    """
    Compute rolling features using only games BEFORE the given date.
    Returns a player dict compatible with SmartPredictor.predict_prop.
    """
    bdl_key = PROP_TO_BDL_STAT.get(prop_type, "pts")

    # Filter to games before the market date
    filtered = [
        g for g in game_logs
        if g.get("game", {}).get("date", "")[:10] < before_date
    ]

    if len(filtered) < 3:
        return None

    # Extract stat arrays
    stat_vals = [g.get(bdl_key, 0) or 0 for g in filtered]
    pts_arr = [g.get("pts", 0) or 0 for g in filtered]
    reb_arr = [g.get("reb", 0) or 0 for g in filtered]
    ast_arr = [g.get("ast", 0) or 0 for g in filtered]
    stl_arr = [g.get("stl", 0) or 0 for g in filtered]
    blk_arr = [g.get("blk", 0) or 0 for g in filtered]
    fg3m_arr = [g.get("fg3m", 0) or 0 for g in filtered]
    mins = [parse_minutes(g.get("min", "0")) for g in filtered]

    n = len(filtered)
    season_avg = sum(stat_vals) / n
    mpg = sum(mins) / n

    def rolling(arr, window):
        if n >= window:
            return sum(arr[-window:]) / window
        return sum(arr) / n

    std_stat = statistics.stdev(stat_vals) if n >= 5 else 0.0

    # Home/away splits
    home_stats, away_stats = [], []
    for g in filtered:
        game = g.get("game", {})
        team_id = g.get("team", {}).get("id")
        is_home = team_id == game.get("home_team_id")
        val = g.get(bdl_key, 0) or 0
        if is_home:
            home_stats.append(val)
        else:
            away_stats.append(val)

    home_avg = sum(home_stats) / max(len(home_stats), 1)
    away_avg = sum(away_stats) / max(len(away_stats), 1)

    # Max/min in last 10
    last10_vals = stat_vals[-10:] if n >= 10 else stat_vals
    max_last10 = max(last10_vals)
    min_last10 = min(last10_vals)

    # Pct above average
    above = sum(1 for v in stat_vals[-10:] if v > season_avg)
    pct_above = above / min(10, n)

    features = {
        # Core stat averages (SmartPredictor expects these keys)
        "pts_pg": sum(pts_arr) / n,
        "reb_pg": sum(reb_arr) / n,
        "ast_pg": sum(ast_arr) / n,
        "stl_pg": sum(stl_arr) / n,
        "blk_pg": sum(blk_arr) / n,
        "three_pm_pg": sum(fg3m_arr) / n,
        "mpg": mpg,
        "games_played": n,
        "game_log_count": n,

        # Rolling averages
        f"last3_{bdl_key}": rolling(stat_vals, 3),
        f"last5_{bdl_key}": rolling(stat_vals, 5),
        f"last10_{bdl_key}": rolling(stat_vals, 10),
        f"trend_{bdl_key}": rolling(stat_vals, 5) - season_avg,
        f"std_{bdl_key}": std_stat,

        # Home/away
        f"home_avg_{bdl_key}": home_avg,
        f"away_avg_{bdl_key}": away_avg,

        # Matchup (no opponent info from Kalshi, use season avg)
        f"vs_opp_avg_{bdl_key}": season_avg,
        "vs_opp_games": 0,

        # Minutes
        "last3_min": rolling(mins, 3),
        "last5_min": rolling(mins, 5),
        "trend_min": rolling(mins, 5) - mpg,

        # Variance
        "max_last10": max_last10,
        "min_last10": min_last10,
        "pct_above_avg": pct_above,
        "std_pts": statistics.stdev(pts_arr) if n >= 5 else 0.0,

        # Schedule (defaults — we don't know exact game context from Kalshi)
        "is_home": False,
        "is_b2b": False,
        "rest_days": 2,
        "games_last_7": 3,

        # Travel (defaults)
        "travel_distance": 0,
        "fatigue_score": 0,

        # Opponent context (league averages as defaults)
        "opp_pts_allowed": 112,
        "opp_reb_allowed": 44,
        "opp_ast_allowed": 25,
        "opp_3pm_allowed": 12,
        "opp_stl_allowed": 7.5,
        "opp_blk_allowed": 5,
        "pace_factor": 1.0,
        "usage_rate": 0,
        "spread": 0,
        "over_under": 220,
        "starter_pct": 0.8 if mpg > 25 else 0.3,
    }

    return features


def load_settled_markets() -> list[dict]:
    """Load all cached settled Kalshi markets."""
    all_markets = []
    for path in sorted(CACHE_DIR.glob("*_parsed.json")):
        data = json.loads(path.read_text())
        all_markets.extend(data)
    print(f"Loaded {len(all_markets)} settled Kalshi markets from cache")
    return all_markets


def normalize_name(name: str) -> str:
    """Normalize player names for matching across data sources."""
    # Unicode -> ASCII (Dončić -> Doncic, Jokić -> Jokic, Vučević -> Vucevic)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.strip().lower()
    name = name.replace(".", "").replace("'", "").replace("'", "")
    # Remove common suffixes
    for suffix in [" iii", " ii", " iv", " jr", " sr"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
            break
    return name


def run_backtest(
    min_edge: float = 5.0,
    min_volume: int = 50,
    bet_size: float = 10.0,
    min_price: int = 10,
    max_price: int = 90,
):
    """
    Run the backtest.

    Args:
        min_edge: Minimum edge (percentage points) to trigger a bet.
        min_volume: Minimum market volume to include.
        bet_size: Dollar amount per bet.
        min_price: Minimum last_price (cents) to include — filters illiquid extremes.
        max_price: Maximum last_price (cents) to include.
    """
    # Load data
    bdl_data = load_bdl_season_data()
    player_index_raw = build_player_index(bdl_data)
    # Build normalized name index
    player_index: dict[str, list[dict]] = {}
    for raw_name, logs in player_index_raw.items():
        norm = normalize_name(raw_name)
        player_index[norm] = logs
    markets = load_settled_markets()

    # Load our trained model
    predictor = SmartPredictor()
    if not predictor.load():
        print("WARNING: No trained model found. Using fallback predictions.")

    valid_prop_types = set(KALSHI_TO_MODEL_PROP.keys())

    # Stats tracking
    total_markets = 0
    matched_player = 0
    had_features = 0
    had_edge = 0
    bets_placed = 0
    bets_won = 0
    total_wagered = 0.0
    total_pnl = 0.0

    by_prop: dict[str, dict] = defaultdict(lambda: {
        "bets": 0, "wins": 0, "wagered": 0.0, "pnl": 0.0, "edges": [],
    })
    by_direction: dict[str, dict] = defaultdict(lambda: {
        "bets": 0, "wins": 0, "wagered": 0.0, "pnl": 0.0,
    })
    bet_details: list[dict] = []

    for market in markets:
        prop_type = market.get("prop_type", "")
        if prop_type not in valid_prop_types:
            continue

        total_markets += 1

        volume = market.get("volume", 0)
        if volume < min_volume:
            continue

        player_name = normalize_name(market.get("player_name", ""))
        if not player_name:
            continue

        game_logs = player_index.get(player_name)
        if not game_logs:
            continue
        matched_player += 1

        close_time = market.get("close_time", "")[:10]
        if not close_time:
            continue

        model_prop = KALSHI_TO_MODEL_PROP[prop_type]
        features = compute_features_at_date(game_logs, close_time, model_prop)
        if not features:
            continue
        had_features += 1

        line = market.get("line")
        if line is None:
            continue

        # Run our model
        prediction = predictor.predict_prop(features, model_prop, line)
        our_over_prob = prediction["over_probability"]
        our_under_prob = prediction["under_probability"]

        # Kalshi's implied probability from last_price (cents, 0-100)
        last_price = market.get("last_price", 0)
        if last_price < min_price or last_price > max_price:
            continue

        kalshi_yes_prob = last_price / 100.0   # YES = over
        kalshi_no_prob = 1.0 - kalshi_yes_prob  # NO = under

        # Calculate edge
        over_edge = (our_over_prob - kalshi_yes_prob) * 100
        under_edge = (our_under_prob - kalshi_no_prob) * 100

        bet_direction = None
        edge_pct = 0.0
        buy_price = 0

        if over_edge >= min_edge:
            bet_direction = "YES"
            edge_pct = over_edge
            buy_price = last_price
        elif under_edge >= min_edge:
            bet_direction = "NO"
            edge_pct = under_edge
            buy_price = 100 - last_price

        if not bet_direction:
            continue

        had_edge += 1

        # Determine outcome
        result = market.get("result")
        sv = market.get("settlement_value", 0) or 0

        # For binary markets: result is "yes" (sv=100) or "no" (sv=0)
        # For scalar/structured: sv is 0-100 representing payout in cents
        if result == "yes":
            yes_payout_cents = 100
        elif result == "no":
            yes_payout_cents = 0
        elif result == "scalar":
            yes_payout_cents = sv  # e.g. 22 means YES holders get 22¢
        else:
            continue

        no_payout_cents = 100 - yes_payout_cents

        # P&L calculation
        # Buying YES at buy_price cents: payout = yes_payout_cents per contract
        # Buying NO at (100 - last_price) cents: payout = no_payout_cents per contract
        num_contracts = max(1, int(bet_size / (buy_price / 100)))
        cost = num_contracts * buy_price / 100  # dollars

        if bet_direction == "YES":
            payout = num_contracts * yes_payout_cents / 100  # dollars
        else:
            payout = num_contracts * no_payout_cents / 100

        pnl = payout - cost
        won = pnl > 0

        bets_placed += 1
        if won:
            bets_won += 1
        total_wagered += cost
        total_pnl += pnl

        ps = by_prop[prop_type]
        ps["bets"] += 1
        if won:
            ps["wins"] += 1
        ps["wagered"] += cost
        ps["pnl"] += pnl
        ps["edges"].append(edge_pct)

        ds = by_direction[bet_direction]
        ds["bets"] += 1
        if won:
            ds["wins"] += 1
        ds["wagered"] += cost
        ds["pnl"] += pnl

        bet_details.append({
            "ticker": market.get("ticker", ""),
            "player": market.get("player_name", ""),
            "prop": prop_type,
            "line": line,
            "direction": bet_direction,
            "buy_price": buy_price,
            "our_prob": round(our_over_prob if bet_direction == "YES" else our_under_prob, 3),
            "kalshi_prob": round(kalshi_yes_prob if bet_direction == "YES" else kalshi_no_prob, 3),
            "edge": round(edge_pct, 1),
            "result": "WIN" if won else "LOSS",
            "pnl": round(pnl, 2),
            "date": close_time,
            "volume": volume,
        })

    # ── Print Results ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("KALSHI BACKTEST RESULTS")
    print("=" * 70)
    print(f"Settings: min_edge={min_edge}%, min_volume={min_volume}, bet_size=${bet_size:.2f}, "
          f"price_range={min_price}-{max_price}¢")
    print()

    print("── Pipeline ──")
    print(f"  Total settled markets (valid props): {total_markets}")
    print(f"  Matched to BDL player:               {matched_player}")
    print(f"  Had enough features (≥3 games):       {had_features}")
    print(f"  Had edge ≥ {min_edge}%:                  {had_edge}")
    print(f"  Bets placed:                          {bets_placed}")
    print()

    if bets_placed == 0:
        print("No bets placed. Try lowering --min-edge or --min-volume.")
        return

    win_rate = bets_won / bets_placed * 100
    roi = total_pnl / total_wagered * 100 if total_wagered > 0 else 0

    print("── Overall ──")
    print(f"  Record:    {bets_won}W - {bets_placed - bets_won}L ({win_rate:.1f}%)")
    print(f"  Wagered:   ${total_wagered:,.2f}")
    print(f"  P&L:       ${total_pnl:+,.2f}")
    print(f"  ROI:       {roi:+.1f}%")
    print()

    print("── By Prop Type ──")
    for prop, ps in sorted(by_prop.items()):
        if ps["bets"] == 0:
            continue
        wr = ps["wins"] / ps["bets"] * 100
        r = ps["pnl"] / ps["wagered"] * 100 if ps["wagered"] > 0 else 0
        avg_e = sum(ps["edges"]) / len(ps["edges"]) if ps["edges"] else 0
        print(f"  {prop:12s}  {ps['wins']}W-{ps['bets']-ps['wins']}L ({wr:.1f}%)  "
              f"P&L: ${ps['pnl']:+,.2f}  ROI: {r:+.1f}%  Avg Edge: {avg_e:.1f}%")
    print()

    print("── By Direction ──")
    for d, ds in sorted(by_direction.items()):
        if ds["bets"] == 0:
            continue
        wr = ds["wins"] / ds["bets"] * 100
        r = ds["pnl"] / ds["wagered"] * 100 if ds["wagered"] > 0 else 0
        print(f"  {d:4s}  {ds['wins']}W-{ds['bets']-ds['wins']}L ({wr:.1f}%)  "
              f"P&L: ${ds['pnl']:+,.2f}  ROI: {r:+.1f}%")
    print()

    # Top 10 best and worst bets
    sorted_bets = sorted(bet_details, key=lambda b: b["pnl"], reverse=True)
    print("── Top 10 Best Bets ──")
    for b in sorted_bets[:10]:
        print(f"  {b['date']} {b['player']:25s} {b['prop']:10s} L={b['line']:5.1f} "
              f"{b['direction']:3s}@{b['buy_price']:2d}¢  edge={b['edge']:+.1f}%  "
              f"{b['result']}  ${b['pnl']:+.2f}")
    print()

    print("── Top 10 Worst Bets ──")
    for b in sorted_bets[-10:]:
        print(f"  {b['date']} {b['player']:25s} {b['prop']:10s} L={b['line']:5.1f} "
              f"{b['direction']:3s}@{b['buy_price']:2d}¢  edge={b['edge']:+.1f}%  "
              f"{b['result']}  ${b['pnl']:+.2f}")
    print()

    # Save full results to JSON
    results_path = CACHE_DIR / "backtest_results.json"
    results = {
        "settings": {
            "min_edge": min_edge,
            "min_volume": min_volume,
            "bet_size": bet_size,
        },
        "summary": {
            "total_markets": total_markets,
            "matched_player": matched_player,
            "had_features": had_features,
            "had_edge": had_edge,
            "bets_placed": bets_placed,
            "bets_won": bets_won,
            "win_rate": round(win_rate, 1),
            "total_wagered": round(total_wagered, 2),
            "total_pnl": round(total_pnl, 2),
            "roi": round(roi, 1),
        },
        "by_prop": {
            k: {
                "bets": v["bets"],
                "wins": v["wins"],
                "wagered": round(v["wagered"], 2),
                "pnl": round(v["pnl"], 2),
                "roi": round(v["pnl"] / v["wagered"] * 100, 1) if v["wagered"] > 0 else 0,
                "avg_edge": round(sum(v["edges"]) / len(v["edges"]), 1) if v["edges"] else 0,
            }
            for k, v in by_prop.items()
        },
        "by_direction": {
            k: {
                "bets": v["bets"],
                "wins": v["wins"],
                "wagered": round(v["wagered"], 2),
                "pnl": round(v["pnl"], 2),
                "roi": round(v["pnl"] / v["wagered"] * 100, 1) if v["wagered"] > 0 else 0,
            }
            for k, v in by_direction.items()
        },
        "bets": bet_details,
    }
    results_path.write_text(json.dumps(results, indent=2))
    print(f"Full results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest model vs Kalshi settled markets")
    parser.add_argument("--min-edge", type=float, default=5.0,
                        help="Minimum edge in pct points to trigger a bet (default: 5)")
    parser.add_argument("--min-volume", type=int, default=50,
                        help="Minimum market volume (default: 50)")
    parser.add_argument("--bet-size", type=float, default=10.0,
                        help="Dollar amount per bet (default: 10)")
    parser.add_argument("--min-price", type=int, default=10,
                        help="Minimum last_price in cents (default: 10)")
    parser.add_argument("--max-price", type=int, default=90,
                        help="Maximum last_price in cents (default: 90)")
    args = parser.parse_args()

    run_backtest(
        min_edge=args.min_edge,
        min_volume=args.min_volume,
        bet_size=args.bet_size,
        min_price=args.min_price,
        max_price=args.max_price,
    )
