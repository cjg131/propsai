"""List all active Odds API sports."""
import asyncio
from app.services.cross_market_sports import CrossMarketScanner
from app.config import get_settings


async def main():
    settings = get_settings()
    scanner = CrossMarketScanner(odds_api_key=settings.the_odds_api_key)
    sports = await scanner.get_sports_list()
    for s in sports:
        if s.get("active"):
            print(f'{s["key"]:45s} | {s.get("group",""):25s} | {s["title"]}')
    await scanner.close()


asyncio.run(main())
