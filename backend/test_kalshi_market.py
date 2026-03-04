import asyncio
import os
import json
from app.api.kalshi import KalshiAPIClient

async def test():
    client = KalshiAPIClient()
    await client.login()
    markets = await client._get("/markets", params={"limit": 1, "status": "active"})
    market = markets.get("markets", [])[0]
    print(json.dumps(market, indent=2))
    
    ticker = market["ticker"]
    orderbook = await client._get(f"/markets/{ticker}/orderbook")
    print("\nOrderbook:")
    print(json.dumps(orderbook, indent=2))
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(test())
