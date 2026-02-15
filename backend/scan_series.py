"""Quick script to discover all Kalshi series tickers."""
import asyncio
from app.services.kalshi_api import KalshiClient


async def main():
    client = KalshiClient()
    series = {}
    cursor = None
    total_markets = 0

    for page in range(20):
        data = await client.get_markets(status="open", limit=200, cursor=cursor)
        markets = data.get("markets", [])
        total_markets += len(markets)
        for m in markets:
            st = m.get("series_ticker", "")
            if st not in series:
                series[st] = {"n": 0, "ex": m.get("title", "")[:90]}
            series[st]["n"] += 1
        cursor = data.get("cursor")
        if not cursor or not markets:
            break
        await asyncio.sleep(0.3)

    print(f"\nTotal markets scanned: {total_markets}")
    print(f"Unique series: {len(series)}\n")
    for s, info in sorted(series.items(), key=lambda x: -x[1]["n"]):
        print(f"  {s:45s} {info['n']:4d} mkts | {info['ex']}")


asyncio.run(main())
