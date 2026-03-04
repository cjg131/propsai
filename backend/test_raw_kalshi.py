import asyncio
import os
import json
import httpx

async def main():
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://trading-api.kalshi.com/trade-api/v2/markets", params={"limit": 1, "status": "active"})
        if resp.status_code == 200:
            print(json.dumps(resp.json(), indent=2))
        else:
            print(f"Error: {resp.status_code}")

if __name__ == "__main__":
    asyncio.run(main())
