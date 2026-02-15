"""Find ALL Kalshi series tickers for sports markets by scanning events."""
import asyncio
from app.services.kalshi_api import KalshiClient


async def main():
    client = KalshiClient()

    # Search for sports-related events
    sports_keywords = [
        "NBA", "NFL", "NHL", "MLB", "NCAA", "NCAAB", "MMA", "UFC",
        "soccer", "EPL", "LaLiga", "SerieA", "Bundesliga", "Ligue1",
        "Champions", "tennis", "ATP", "WTA",
        "GAME", "WINNER", "SPREAD", "TOTAL",
    ]

    # Method 1: Search events by category
    print("=== Searching by event categories ===")
    for keyword in ["basketball", "football", "hockey", "soccer", "mma", "tennis", "sports"]:
        try:
            data = await client._get("/events", params={
                "status": "open",
                "series_ticker": "",
                "limit": 50,
                "with_nested_markets": "true",
            })
            # Just get first page to see structure
            events = data.get("events", [])
            if events:
                print(f"\nEvents found: {len(events)}")
                for e in events[:3]:
                    print(f"  event_ticker={e.get('event_ticker','')} series={e.get('series_ticker','')} title={e.get('title','')[:80]}")
                    for m in e.get("markets", [])[:2]:
                        print(f"    market: {m.get('ticker','')} | {m.get('title','')[:60]} yes={m.get('yes_ask',0)} no={m.get('no_ask',0)}")
            break
        except Exception as ex:
            print(f"  Error: {ex}")

    # Method 2: Try known series prefixes from the screenshot
    print("\n=== Trying known series prefixes ===")
    prefixes = [
        "KXNCAAB", "KXNCAABGAME", "KXNCAABSPREAD", "KXNCAABTOTAL",
        "KXNBA", "KXNBAGAME", "KXNBASPREAD", "KXNBATOTAL",
        "KXNHL", "KXNHLGAME",
        "KXNFL", "KXNFLGAME",
        "KXMLB", "KXMLBGAME",
        "KXEPL", "KXLALIGA", "KXUCL", "KXUFC",
        "KXMVE", "KXMVESPORTS",
        # Try event-based
        "KXCBB", "KXCBBALL",
        "KXBASKETBALL", "KXHOCKEY", "KXSOCCER",
    ]

    found_series = {}
    for prefix in prefixes:
        try:
            data = await client.get_markets(
                status="open",
                series_ticker=prefix,
                limit=5,
            )
            markets = data.get("markets", [])
            if markets:
                found_series[prefix] = len(markets)
                m = markets[0]
                print(f"  ✅ {prefix}: {len(markets)}+ markets | ex: {m.get('title','')[:80]}")
            await asyncio.sleep(0.2)
        except Exception as ex:
            pass

    # Method 3: Get events list and find series tickers
    print("\n=== Fetching events to find series ===")
    try:
        data = await client._get("/events", params={"status": "open", "limit": 200})
        events = data.get("events", [])
        series_map = {}
        for e in events:
            st = e.get("series_ticker", "")
            if st and st not in series_map:
                series_map[st] = {
                    "title": e.get("title", "")[:80],
                    "category": e.get("category", ""),
                    "count": 0,
                }
            if st:
                series_map[st]["count"] += 1

        print(f"Found {len(series_map)} unique series from {len(events)} events")
        for st, info in sorted(series_map.items(), key=lambda x: -x[1]["count"]):
            print(f"  {st:45s} ({info['count']:3d} events) cat={info['category']:15s} | {info['title']}")
    except Exception as ex:
        print(f"Events fetch error: {ex}")

    # Method 4: Try the Kalshi series endpoint directly
    print("\n=== Trying /series endpoint ===")
    try:
        data = await client._get("/series", params={"limit": 200})
        series_list = data.get("series", [])
        print(f"Found {len(series_list)} series")
        sports_series = []
        for s in series_list:
            ticker = s.get("ticker", "")
            title = s.get("title", "")
            cat = s.get("category", "")
            t_lower = (title + ticker + cat).lower()
            if any(kw in t_lower for kw in ["sport", "game", "nba", "nfl", "nhl", "mlb", "ncaa", "soccer", "epl", "mma", "ufc", "tennis", "winner", "basket", "hockey", "football"]):
                sports_series.append(s)
                print(f"  {ticker:45s} cat={cat:15s} | {title[:80]}")
    except Exception as ex:
        print(f"Series endpoint error: {ex}")


asyncio.run(main())
