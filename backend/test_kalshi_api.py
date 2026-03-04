import asyncio
from app.services.kalshi_scanner import KalshiScanner
import json

async def main():
    scanner = KalshiScanner()
    await scanner.kalshi.login()
    markets = await scanner.kalshi.get_markets(limit=1, status="active")
    market = markets.get("markets", [])[0]
    print(json.dumps(market, indent=2))
    await scanner.kalshi.close()

if __name__ == "__main__":
    asyncio.run(main())
