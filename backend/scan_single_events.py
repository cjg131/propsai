"""Discover single-event sports markets on Kalshi and try to match them to Odds API."""
import asyncio
from app.services.kalshi_api import KalshiClient
from app.services.cross_market_sports import CrossMarketScanner, MONITORED_SPORTS
from app.config import get_settings


async def main():
    settings = get_settings()
    client = KalshiClient()
    scanner = CrossMarketScanner(odds_api_key=settings.the_odds_api_key)

    # Step 1: Get ALL open Kalshi markets
    all_markets = []
    cursor = None
    for page in range(30):
        data = await client.get_markets(status="open", limit=200, cursor=cursor)
        markets = data.get("markets", [])
        all_markets.extend(markets)
        cursor = data.get("cursor")
        if not cursor or not markets:
            break
        await asyncio.sleep(0.3)

    print(f"Total Kalshi markets: {len(all_markets)}")

    # Step 2: Filter to sports-looking single-event markets (not multi-leg parlays)
    single_events = []
    for m in all_markets:
        title = (m.get("title", "") or "").lower()
        yes_ask = m.get("yes_ask", 0) or 0
        no_ask = m.get("no_ask", 0) or 0
        vol = m.get("volume", 0) or 0

        # Skip illiquid
        if yes_ask <= 2 or no_ask <= 2 or vol < 5:
            continue

        # Count commas — parlays have many legs separated by commas
        comma_count = title.count(",")

        # Single-event: 0 or 1 commas (e.g., "Team A wins" or "Over 220.5 points scored")
        if comma_count <= 1:
            single_events.append(m)

    print(f"Single-event liquid markets: {len(single_events)}")

    # Show examples
    for m in sorted(single_events, key=lambda x: -(x.get("volume", 0) or 0))[:20]:
        title = m.get("title", "")[:100]
        ticker = m.get("ticker", "")[:35]
        vol = m.get("volume", 0) or 0
        ya = m.get("yes_ask", 0) or 0
        na = m.get("no_ask", 0) or 0
        print(f"  [{ticker:35s}] vol={vol:5d} yes={ya:2d}c no={na:2d}c | {title}")

    # Step 3: Get sharp odds for a few sports and try matching
    print("\n--- MATCHING AGAINST SHARP ODDS ---")
    for sport in ["basketball_nba", "basketball_ncaab", "icehockey_nhl"]:
        events = await scanner.get_odds(sport, markets="h2h,totals")
        if not events:
            print(f"\n{sport}: no events")
            continue

        print(f"\n{sport}: {len(events)} events")
        for event in events[:3]:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            consensus = scanner.extract_sharp_consensus(event)
            sharp = consensus.get("sharp_consensus", {})
            n_sharp = consensus.get("sharp_books_found", 0)

            if not sharp:
                continue

            print(f"  {away} @ {home} (sharp books: {n_sharp})")
            for key, prob in sorted(sharp.items()):
                print(f"    {key}: {prob:.1%}")

            # Try to match against Kalshi markets
            mispricings = scanner.find_mispricings(consensus, single_events, min_edge=0.02)
            if mispricings:
                for mp in mispricings:
                    print(f"    *** EDGE: {mp['ticker']} {mp['side']} edge={mp['edge']:.1%} (sharp={mp['sharp_prob']:.1%} vs kalshi={mp['kalshi_implied']:.1%})")
            else:
                # Show why no match — check if any Kalshi title contains team names
                home_lower = home.lower()
                away_lower = away.lower()
                home_last = home.split()[-1].lower() if home else ""
                away_last = away.split()[-1].lower() if away else ""
                matched = []
                for m in single_events:
                    t = m.get("title", "").lower()
                    if home_last in t or away_last in t or home_lower in t or away_lower in t:
                        matched.append(m)
                if matched:
                    print(f"    Kalshi has {len(matched)} markets with team names but no edge match:")
                    for mm in matched[:3]:
                        print(f"      {mm.get('title','')[:100]}")
                else:
                    print(f"    No Kalshi markets found for {home_last}/{away_last}")

        await asyncio.sleep(0.5)

    await scanner.close()


asyncio.run(main())
