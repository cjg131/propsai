import asyncio
import os
import json
from app.services.kalshi_api import KalshiAPIClient

async def main():
    client = KalshiAPIClient(email=os.environ.get("KALSHI_EMAIL"), password=os.environ.get("KALSHI_PASSWORD"))
    await client.login()
    markets = await client.get_markets(limit=1, status="active")
    print(json.dumps(markets, indent=2))
    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
