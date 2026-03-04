import asyncio
from backend.app.services.kalshi_api import KalshiAPI

async def main():
    client = KalshiAPI()
    await client.initialize()
    bal = await client.get_balance()
    print("Balance:", bal)
    
if __name__ == "__main__":
    asyncio.run(main())
