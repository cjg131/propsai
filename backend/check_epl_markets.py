"""Check what EPL single-game markets look like on Kalshi."""
import asyncio
from app.services.kalshi_api import KalshiClient


async def main():
    client = KalshiClient()

    for series in ["KXEPLGAME", "KXNCAABGAME", "KXNHLGAME", "KXUCLGAME"]:
        print(f"\n=== {series} ===")
        data = await client.get_markets(status="open", series_ticker=series, limit=20)
        for m in data.get("markets", [])[:10]:
            ya = m.get("yes_ask", 0) or 0
            na = m.get("no_ask", 0) or 0
            vol = m.get("volume", 0) or 0
            t = m.get("ticker", "")
            title = m.get("title", "")
            print(f"  ya={ya:2d}c na={na:2d}c vol={vol:4d} | {t[:45]:45s} | {title}")
        await asyncio.sleep(0.3)


asyncio.run(main())
