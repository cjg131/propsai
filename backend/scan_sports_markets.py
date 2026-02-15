"""Discover all sports-related Kalshi markets by title/ticker patterns."""
import asyncio
from app.services.kalshi_api import KalshiClient

SPORTS_KEYWORDS = [
    "nba", "nfl", "nhl", "mlb", "ncaa", "mma", "ufc", "soccer", "epl",
    "premier league", "la liga", "serie a", "bundesliga", "ligue 1",
    "champions league", "tennis", "atp", "wta",
    "points scored", "win", "game", "match", "bout", "fight",
    "spread", "total", "over", "under", "moneyline",
    "celtics", "lakers", "warriors", "knicks", "nets", "76ers", "heat",
    "bucks", "nuggets", "thunder", "cavaliers", "rockets", "mavericks",
    "chiefs", "eagles", "bills", "ravens", "lions", "49ers",
    "yankees", "dodgers", "braves", "astros", "phillies",
    "oilers", "panthers", "rangers", "bruins", "avalanche",
]


async def main():
    client = KalshiClient()
    sports_markets = []
    non_sports = []
    cursor = None
    total = 0

    for page in range(30):
        data = await client.get_markets(status="open", limit=200, cursor=cursor)
        markets = data.get("markets", [])
        total += len(markets)
        for m in markets:
            title = (m.get("title", "") or "").lower()
            ticker = (m.get("ticker", "") or "").lower()
            series = (m.get("series_ticker", "") or "").lower()
            combined = f"{title} {ticker} {series}"
            is_sport = any(kw in combined for kw in SPORTS_KEYWORDS)
            if is_sport:
                sports_markets.append(m)
            else:
                non_sports.append(m)
        cursor = data.get("cursor")
        if not cursor or not markets:
            break
        await asyncio.sleep(0.3)

    print(f"\nTotal markets: {total}")
    print(f"Sports markets: {len(sports_markets)}")
    print(f"Non-sports: {len(non_sports)}")

    # Categorize sports markets
    categories = {}
    for m in sports_markets:
        title = m.get("title", "")
        ticker = m.get("ticker", "")
        series = m.get("series_ticker", "")
        vol = m.get("volume", 0) or 0
        yes_ask = m.get("yes_ask", 0) or 0
        no_ask = m.get("no_ask", 0) or 0

        # Try to categorize
        t_lower = title.lower()
        if "parlay" in t_lower or "multi" in t_lower or ("," in title and "and" in t_lower):
            cat = "PARLAY"
        elif "points scored" in t_lower or "total" in t_lower:
            cat = "TOTAL/POINTS"
        elif "win" in t_lower or "beat" in t_lower:
            cat = "MONEYLINE/WIN"
        elif "spread" in t_lower:
            cat = "SPREAD"
        elif "over" in t_lower or "under" in t_lower:
            cat = "OVER/UNDER"
        else:
            cat = "OTHER"

        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "ticker": ticker,
            "title": title[:100],
            "vol": vol,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "series": series,
        })

    print("\n--- SPORTS MARKET CATEGORIES ---")
    for cat, mkts in sorted(categories.items(), key=lambda x: -len(x[1])):
        liquid = [m for m in mkts if m["vol"] >= 5 and m["yes_ask"] > 0 and m["no_ask"] > 0]
        print(f"\n{cat}: {len(mkts)} total, {len(liquid)} liquid")
        for m in sorted(liquid, key=lambda x: -x["vol"])[:5]:
            print(f"  [{m['ticker'][:30]:30s}] vol={m['vol']:5d} yes={m['yes_ask']:2d}c no={m['no_ask']:2d}c | {m['title']}")

    # Show non-sports categories too
    non_cats = {}
    for m in non_sports:
        title = m.get("title", "")
        t_lower = title.lower()
        if "temperature" in t_lower or "weather" in t_lower or "rain" in t_lower or "snow" in t_lower:
            cat = "WEATHER"
        elif "bitcoin" in t_lower or "crypto" in t_lower or "ethereum" in t_lower:
            cat = "CRYPTO"
        elif "stock" in t_lower or "s&p" in t_lower or "nasdaq" in t_lower or "dow" in t_lower:
            cat = "STOCKS"
        elif "election" in t_lower or "trump" in t_lower or "biden" in t_lower or "congress" in t_lower:
            cat = "POLITICS"
        else:
            cat = "OTHER"
        if cat not in non_cats:
            non_cats[cat] = 0
        non_cats[cat] += 1

    print("\n--- NON-SPORTS CATEGORIES ---")
    for cat, n in sorted(non_cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")


asyncio.run(main())
