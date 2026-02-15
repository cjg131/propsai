"""Debug why so many single-game markets fail to match Odds API events."""
import asyncio
from app.services.kalshi_api import KalshiClient
from app.services.kalshi_scanner import SINGLE_GAME_SERIES
from app.services.cross_market_sports import CrossMarketScanner
from app.config import get_settings


async def main():
    settings = get_settings()
    client = KalshiClient()
    scanner = CrossMarketScanner(odds_api_key=settings.the_odds_api_key)

    # Pick a few series with known markets
    test_series = ["KXNCAABGAME", "KXNHLGAME", "KXATPMATCH", "KXUCLGAME", "KXEPLGAME", "KXSIXNATIONSMATCH"]

    for series_ticker in test_series:
        info = SINGLE_GAME_SERIES.get(series_ticker, {})
        odds_sport = info.get("odds_sport", "")
        if not odds_sport:
            continue

        # Get Kalshi markets
        data = await client.get_markets(status="open", series_ticker=series_ticker, limit=20)
        markets = data.get("markets", [])
        if not markets:
            continue

        # Get Odds API events
        events = await scanner.get_odds(odds_sport, markets="h2h,spreads,totals")
        if not events:
            print(f"\n{series_ticker} ({odds_sport}): {len(markets)} Kalshi markets, 0 Odds API events")
            continue

        print(f"\n=== {series_ticker} ({odds_sport}): {len(markets)} Kalshi, {len(events)} Odds events ===")

        # Show Kalshi titles
        print("  Kalshi titles:")
        seen_titles = set()
        for m in markets[:10]:
            title = m.get("title", "")
            if title not in seen_titles:
                seen_titles.add(title)
                print(f"    {title}")

        # Show Odds API teams
        print("  Odds API events:")
        for e in events[:8]:
            home = e.get("home_team", "")
            away = e.get("away_team", "")
            print(f"    {away} @ {home}")

        # Try matching
        matched = 0
        unmatched_titles = set()
        for m in markets:
            title = m.get("title", "").lower()
            found = False
            for e in events:
                home = e.get("home_team", "")
                away = e.get("away_team", "")
                home_last = home.split()[-1].lower() if home else ""
                away_last = away.split()[-1].lower() if away else ""

                home_match = (home_last in title or home.lower() in title or
                              any(w.lower() in title for w in home.split() if len(w) > 3))
                away_match = (away_last in title or away.lower() in title or
                              any(w.lower() in title for w in away.split() if len(w) > 3))

                if home_match and away_match:
                    found = True
                    break

            if found:
                matched += 1
            else:
                t = m.get("title", "")
                if t not in unmatched_titles:
                    unmatched_titles.add(t)

        print(f"  Matched: {matched}/{len(markets)}")
        if unmatched_titles:
            print(f"  Unmatched titles ({len(unmatched_titles)}):")
            for t in list(unmatched_titles)[:5]:
                print(f"    ❌ {t}")

        await asyncio.sleep(0.5)

    await scanner.close()


asyncio.run(main())
