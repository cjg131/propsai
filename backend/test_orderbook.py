import asyncio
from app.api.kalshi import get_kalshi_client
import json

async def main():
    client = get_kalshi_client()
    markets = await client.get_markets(limit=1, status="active")
    market = markets.get("markets", [])[0]
    ticker = market["ticker"]
    
    ob = await client.get_orderbook(ticker)
    print(json.dumps(ob, indent=2))
    
if __name__ == "__main__":
    asyncio.run(main())
