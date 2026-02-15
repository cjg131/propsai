"""
Fetch all settled NBA player prop markets from Kalshi and cache locally.

Usage:
    poetry run python -u fetch_kalshi_history.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.kalshi_api import (
    KalshiClient,
    NBA_PLAYER_PROP_SERIES,
    _parse_player_prop_market,
)

CACHE_DIR = Path(__file__).parent / "app" / "cache" / "kalshi_history"


MAX_RETRIES = 5
BASE_DELAY = 3.0  # seconds


async def fetch_with_retry(client: KalshiClient, **kwargs) -> dict:
    """Fetch with exponential backoff on 429."""
    for attempt in range(MAX_RETRIES):
        try:
            return await client.get_markets(**kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                print(f"  Rate limited, waiting {delay:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                await asyncio.sleep(delay)
            else:
                raise
    return {"markets": []}  # Should not reach here


async def fetch_and_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    client = KalshiClient()

    total_fetched = 0
    total_parsed = 0

    for series in NBA_PLAYER_PROP_SERIES:
        # Skip if already cached
        raw_path = CACHE_DIR / f"{series}_raw.json"
        if raw_path.exists():
            existing = json.loads(raw_path.read_text())
            print(f"\n{series}: already cached ({len(existing)} markets), skipping")
            total_fetched += len(existing)
            parsed_path = CACHE_DIR / f"{series}_parsed.json"
            if parsed_path.exists():
                total_parsed += len(json.loads(parsed_path.read_text()))
            continue

        print(f"\n{'='*60}")
        print(f"Fetching settled markets for {series}...")

        raw_markets: list[dict] = []
        cursor = None
        page = 0

        while True:
            page += 1
            try:
                data = await fetch_with_retry(
                    client,
                    status="settled",
                    series_ticker=series,
                    limit=200,
                    cursor=cursor,
                )
            except Exception as e:
                print(f"  Error on page {page}: {e}")
                break

            markets = data.get("markets", [])
            raw_markets.extend(markets)
            cursor = data.get("cursor")

            print(f"  Page {page}: {len(markets)} markets (total so far: {len(raw_markets)})")

            if not cursor or not markets:
                break

            # Small delay between pages to avoid rate limits
            await asyncio.sleep(0.5)

        if not raw_markets:
            print(f"  No settled markets found for {series}")
            continue

        # Parse all markets
        parsed = []
        for m in raw_markets:
            p = _parse_player_prop_market(m, series_ticker=series)
            if p:
                parsed.append(p)

        # Collect date range
        dates = set()
        for p in parsed:
            ct = p.get("close_time", "")[:10]
            if ct:
                dates.add(ct)
        sorted_dates = sorted(dates)

        total_vol = sum(p.get("volume", 0) for p in parsed)
        total_fetched += len(raw_markets)
        total_parsed += len(parsed)

        print(f"  Raw: {len(raw_markets)} | Parsed: {len(parsed)}")
        if sorted_dates:
            print(f"  Date range: {sorted_dates[0]} to {sorted_dates[-1]} ({len(sorted_dates)} days)")
        print(f"  Total volume: {total_vol:,}")

        # Save raw markets (full API response for maximum flexibility)
        raw_path = CACHE_DIR / f"{series}_raw.json"
        with open(raw_path, "w") as f:
            json.dump(raw_markets, f, indent=2)
        print(f"  Saved raw: {raw_path.name} ({len(raw_markets)} markets)")

        # Save parsed markets
        parsed_path = CACHE_DIR / f"{series}_parsed.json"
        with open(parsed_path, "w") as f:
            json.dump(parsed, f, indent=2)
        print(f"  Saved parsed: {parsed_path.name} ({len(parsed)} markets)")

    await client.close()

    print(f"\n{'='*60}")
    print(f"DONE: {total_fetched} raw markets fetched, {total_parsed} parsed")
    print(f"Cache directory: {CACHE_DIR}")

    # Also save a summary file
    summary = {
        "total_raw": total_fetched,
        "total_parsed": total_parsed,
        "series_fetched": NBA_PLAYER_PROP_SERIES,
        "cache_dir": str(CACHE_DIR),
    }
    with open(CACHE_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(fetch_and_cache())
