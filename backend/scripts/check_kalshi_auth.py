"""
Quick Kalshi auth check for the configured API key + private key pair.

Usage:
    cd backend
    python3 scripts/check_kalshi_auth.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.kalshi_api import KalshiClient


OFFICIAL_BASE_URLS = [
    "https://api.elections.kalshi.com/trade-api/v2",
    "https://demo-api.kalshi.co/trade-api/v2",
]


def _sign_request(private_key, timestamp_ms: str, method: str, path: str) -> str:
    message = (timestamp_ms + method + path).encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


async def _probe_base_url(base_url: str, api_key_id: str, private_key) -> tuple[int | None, str]:
    path = "/portfolio/balance"
    timestamp_ms = str(int(time.time() * 1000))
    signature = _sign_request(private_key, timestamp_ms, "GET", "/trade-api/v2" + path)
    headers = {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": signature,
    }
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=20.0) as client:
            resp = await client.get(path, headers=headers)
            return resp.status_code, resp.text[:300]
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def main_async() -> int:
    settings = get_settings()
    print("Kalshi auth check")
    print("=" * 60)
    print(f"API key id present: {bool(settings.kalshi_api_key_id)}")
    print(f"Private key path: {settings.kalshi_private_key_path}")
    print(f"Configured base URL: {settings.kalshi_base_url}")

    client = KalshiClient()
    try:
        print(f"Private key loaded: {bool(client.private_key)}")
        if not settings.kalshi_api_key_id:
            print("Result: missing KALSHI_API_KEY_ID")
            return 1
        if not client.private_key:
            print("Result: private key failed to load")
            return 1

        print("\nOfficial host probes")
        print("-" * 60)
        for base_url in OFFICIAL_BASE_URLS:
            status_code, body = await _probe_base_url(base_url, settings.kalshi_api_key_id, client.private_key)
            print(f"{base_url}")
            print(f"  status: {status_code}")
            print(f"  body: {body}")

        try:
            balance = await client.get_balance()
            print("Result: authenticated successfully")
            print(json.dumps(balance, indent=2))
            return 0
        except Exception as exc:
            print(f"Result: authentication failed: {type(exc).__name__}: {exc}")
            return 2
    finally:
        await client.close()


def main() -> None:
    raise SystemExit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
