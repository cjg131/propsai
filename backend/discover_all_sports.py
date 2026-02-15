"""Discover ALL Kalshi sports series and map to Odds API sports."""
import asyncio
from app.services.kalshi_api import KalshiClient


async def main():
    client = KalshiClient()

    # Get ALL series
    all_series = []
    cursor = None
    for page in range(20):
        try:
            data = await client._get("/series", params={"limit": 200, "cursor": cursor})
            series = data.get("series", [])
            all_series.extend(series)
            cursor = data.get("cursor")
            if not cursor or not series:
                break
            await asyncio.sleep(0.3)
        except Exception as e:
            print(f"Error page {page}: {e}")
            break

    print(f"Total series: {len(all_series)}")

    # Filter to Sports category
    sports_series = [s for s in all_series if s.get("category", "").lower() == "sports"]
    print(f"Sports series: {len(sports_series)}")

    # Group by type (game, spread, total, etc.)
    by_type = {}
    for s in sports_series:
        ticker = s.get("ticker", "")
        title = s.get("title", "")
        t_lower = ticker.lower()

        if "game" in t_lower or "match" in t_lower or "winner" in t_lower:
            mtype = "GAME/MATCH"
        elif "spread" in t_lower:
            mtype = "SPREAD"
        elif "total" in t_lower:
            mtype = "TOTAL"
        elif "round" in t_lower:
            mtype = "ROUNDS"
        elif "btts" in t_lower:
            mtype = "BTTS"
        elif "goal" in t_lower:
            mtype = "GOAL"
        else:
            mtype = "OTHER"

        if mtype not in by_type:
            by_type[mtype] = []
        by_type[mtype].append(s)

    for mtype, series_list in sorted(by_type.items()):
        print(f"\n=== {mtype} ({len(series_list)}) ===")
        for s in sorted(series_list, key=lambda x: x.get("ticker", "")):
            ticker = s.get("ticker", "")
            title = s.get("title", "")
            print(f"  {ticker:45s} | {title}")


asyncio.run(main())
