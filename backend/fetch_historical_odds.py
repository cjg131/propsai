#!/usr/bin/env python3
"""
Fetch historical player prop lines from The Odds API and cache to disk.

Usage:
    poetry run python -u fetch_historical_odds.py [--days 45] [--start-date 2025-12-28]

This fetches NBA player prop lines (points, rebounds, assists, threes) for
each game date, caches them as JSON files in app/cache/historical_odds/,
and tracks API usage to stay within budget.

Cost per game night: 1 (events) + N_games * 40 (4 prop markets * 10 credits)
With ~10 games/night: ~401 credits/night
Budget: 20,000 credits → ~49 nights
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import httpx

CACHE_DIR = Path(__file__).parent / "app" / "cache" / "historical_odds"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
PROP_MARKETS = "player_points,player_rebounds,player_assists,player_threes"
REGION = "us"

# Track usage
total_used = 0
total_remaining = 20000


def get_api_key() -> str:
    # Try env var first, then .env file
    key = os.environ.get("THE_ODDS_API_HISTORICAL_KEY", "")
    if not key:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("THE_ODDS_API_HISTORICAL_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    return key


def fetch_events(api_key: str, date_str: str) -> tuple[list[dict], dict]:
    """Fetch NBA events for a given date. Returns (events, headers)."""
    # Use 6pm ET (23:00 UTC) on the game date to catch evening games
    timestamp = f"{date_str}T23:00:00Z"
    url = f"{BASE_URL}/historical/sports/{SPORT}/events"
    params = {
        "apiKey": api_key,
        "date": timestamp,
    }
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    headers = {
        "used": int(resp.headers.get("x-requests-used", 0)),
        "remaining": int(resp.headers.get("x-requests-remaining", 0)),
        "last": int(resp.headers.get("x-requests-last", 0)),
    }
    return data.get("data", []), headers


def fetch_event_props(api_key: str, event_id: str, date_str: str) -> tuple[dict, dict]:
    """Fetch player props for a single event. Returns (data, headers)."""
    timestamp = f"{date_str}T23:00:00Z"
    url = f"{BASE_URL}/historical/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "date": timestamp,
        "regions": REGION,
        "markets": PROP_MARKETS,
        "oddsFormat": "american",
    }
    resp = httpx.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    headers = {
        "used": int(resp.headers.get("x-requests-used", 0)),
        "remaining": int(resp.headers.get("x-requests-remaining", 0)),
        "last": int(resp.headers.get("x-requests-last", 0)),
    }
    return data.get("data", {}), headers


def parse_props_from_event(event_data: dict) -> list[dict]:
    """Extract player prop lines from event odds data."""
    props = []
    home_team = event_data.get("home_team", "")
    away_team = event_data.get("away_team", "")
    commence_time = event_data.get("commence_time", "")
    event_id = event_data.get("id", "")

    for bookmaker in event_data.get("bookmakers", []):
        book_key = bookmaker.get("key", "")
        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")
            # Map Odds API market keys to our prop types
            prop_type_map = {
                "player_points": "points",
                "player_rebounds": "rebounds",
                "player_assists": "assists",
                "player_threes": "threes",
            }
            prop_type = prop_type_map.get(market_key)
            if not prop_type:
                continue

            # Group outcomes by player (Over/Under pairs)
            player_lines: dict[str, dict] = {}
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description", "")
                if not player_name:
                    continue
                direction = outcome.get("name", "").lower()  # "over" or "under"
                line = outcome.get("point", 0)
                odds = outcome.get("price", -110)

                if player_name not in player_lines:
                    player_lines[player_name] = {
                        "player": player_name,
                        "prop_type": prop_type,
                        "line": line,
                        "book": book_key,
                        "home_team": home_team,
                        "away_team": away_team,
                        "commence_time": commence_time,
                        "event_id": event_id,
                    }
                if direction == "over":
                    player_lines[player_name]["over_odds"] = odds
                elif direction == "under":
                    player_lines[player_name]["under_odds"] = odds

            props.extend(player_lines.values())

    return props


def get_game_dates(start_date: date, num_days: int) -> list[date]:
    """Generate list of dates to fetch, skipping already-cached dates."""
    dates = []
    for i in range(num_days):
        d = start_date + timedelta(days=i)
        if d >= date.today():
            break
        cache_file = CACHE_DIR / f"props_{d.isoformat()}.json"
        if cache_file.exists():
            continue
        dates.append(d)
    return dates


def dedupe_props(props: list[dict]) -> list[dict]:
    """Keep best line per player+prop (prefer DraftKings, then FanDuel)."""
    best: dict[str, dict] = {}
    book_priority = {"draftkings": 0, "fanduel": 1, "betmgm": 2}
    for p in props:
        key = f"{p['player'].lower()}|{p['prop_type']}"
        existing = best.get(key)
        if not existing:
            best[key] = p
        else:
            # Prefer higher-priority book
            ep = book_priority.get(existing.get("book", ""), 99)
            np_ = book_priority.get(p.get("book", ""), 99)
            if np_ < ep:
                best[key] = p
    return list(best.values())


def main():
    parser = argparse.ArgumentParser(description="Fetch historical NBA player prop lines")
    parser.add_argument("--days", type=int, default=45, help="Number of days to fetch")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date (YYYY-MM-DD). Default: --days ago from yesterday")
    parser.add_argument("--max-credits", type=int, default=15000,
                        help="Max credits to spend (leave buffer)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without fetching")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print("ERROR: No THE_ODDS_API_HISTORICAL_KEY found in env or .env")
        sys.exit(1)

    yesterday = date.today() - timedelta(days=1)
    if args.start_date:
        start = date.fromisoformat(args.start_date)
    else:
        start = yesterday - timedelta(days=args.days)

    dates_to_fetch = get_game_dates(start, args.days + 30)  # extra buffer for skipped dates
    # Limit to requested number of days
    dates_to_fetch = dates_to_fetch[:args.days]

    print(f"Historical Odds Fetcher")
    print(f"  Date range: {start} to {yesterday}")
    print(f"  Dates to fetch: {len(dates_to_fetch)} (already cached dates skipped)")
    print(f"  Max credits: {args.max_credits}")
    print(f"  Estimated cost: ~{len(dates_to_fetch) * 401} credits")
    print(f"  Cache dir: {CACHE_DIR}")
    print()

    if args.dry_run:
        for d in dates_to_fetch:
            print(f"  Would fetch: {d}")
        return

    credits_used = 0
    total_props = 0
    dates_fetched = 0

    for di, game_date in enumerate(dates_to_fetch):
        date_str = game_date.isoformat()
        cache_file = CACHE_DIR / f"props_{date_str}.json"

        if cache_file.exists():
            print(f"  [{di+1}/{len(dates_to_fetch)}] {date_str} — cached, skipping")
            continue

        # Check budget
        if credits_used + 500 > args.max_credits:
            print(f"\n  Budget limit reached ({credits_used} credits used). Stopping.")
            break

        print(f"  [{di+1}/{len(dates_to_fetch)}] {date_str} — fetching events...", end="", flush=True)

        try:
            events, hdrs = fetch_events(api_key, date_str)
            credits_used += hdrs.get("last", 1)
            remaining = hdrs.get("remaining", 0)
        except Exception as e:
            print(f" ERROR: {e}")
            time.sleep(2)
            continue

        # Filter to events that commenced on this date
        day_events = []
        for ev in events:
            ct = ev.get("commence_time", "")
            # Games starting between 4pm ET (21:00 UTC) on game_date and 4am ET next day
            if ct.startswith(date_str) or ct.startswith((game_date + timedelta(days=1)).isoformat()[:10]):
                day_events.append(ev)

        if not day_events:
            print(f" no games")
            # Save empty file so we don't re-fetch
            cache_file.write_text(json.dumps({"date": date_str, "props": [], "events": 0}))
            continue

        print(f" {len(day_events)} games", end="", flush=True)

        all_props = []
        for ev in day_events:
            eid = ev["id"]
            try:
                event_data, hdrs = fetch_event_props(api_key, eid, date_str)
                credits_used += hdrs.get("last", 40)
                remaining = hdrs.get("remaining", 0)
                props = parse_props_from_event(event_data)
                all_props.extend(props)
                time.sleep(0.3)  # gentle rate limiting
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    print(f" RATE LIMITED, waiting 10s...", end="", flush=True)
                    time.sleep(10)
                    try:
                        event_data, hdrs = fetch_event_props(api_key, eid, date_str)
                        credits_used += hdrs.get("last", 40)
                        props = parse_props_from_event(event_data)
                        all_props.extend(props)
                    except Exception:
                        pass
                else:
                    print(f" err:{e.response.status_code}", end="", flush=True)
            except Exception as e:
                print(f" err:{e}", end="", flush=True)

        # Dedupe: keep best line per player+prop
        deduped = dedupe_props(all_props)

        # Save to cache
        cache_data = {
            "date": date_str,
            "events": len(day_events),
            "total_props_raw": len(all_props),
            "props": deduped,
        }
        cache_file.write_text(json.dumps(cache_data, indent=1))
        total_props += len(deduped)
        dates_fetched += 1

        print(f" → {len(deduped)} props (credits: {credits_used}/{args.max_credits}, remaining: {remaining})")

        # Rate limit between dates
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"  Done! Fetched {dates_fetched} dates, {total_props} total prop lines")
    print(f"  Credits used: {credits_used}")
    print(f"  Cache: {CACHE_DIR}")

    # Show summary of cached data
    cached_files = sorted(CACHE_DIR.glob("props_*.json"))
    if cached_files:
        total_cached = 0
        for f in cached_files:
            d = json.loads(f.read_text())
            total_cached += len(d.get("props", []))
        print(f"  Total cached: {len(cached_files)} dates, {total_cached} prop lines")


if __name__ == "__main__":
    main()
