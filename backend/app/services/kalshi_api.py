from __future__ import annotations

import asyncio
import base64
import re
import time
from pathlib import Path
from typing import Any

import httpx
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# NBA player prop series tickers on Kalshi (discovered from /series endpoint)
# Individual stat props
NBA_PLAYER_PROP_SERIES = [
    "KXNBAPTS",    # Player Points
    "KXNBAREB",    # Player Rebounds
    "KXNBAAST",    # Player Assists
    "KXNBABLK",    # Player Blocks
    "KXNBASTL",    # Player Steals
    "KXNBA3D",     # Triple Double
    "KXNBAPR",     # Points + Rebounds
    "KXNBAPRA",    # Points + Rebounds + Assists
    "KXNBARA",     # Rebounds + Assists
]

# Game-level NBA series (spreads, totals, etc.)
NBA_GAME_SERIES = [
    "KXNBAGAME",           # Game winner
    "KXNBATOTAL",          # Total points
    "KXNBATEAMTOTAL",      # Team total
    "KXNBA1HTOTAL",        # 1st half total
    "KXNBA2HTOTAL",        # 2nd half total
    "KXNBA1QSPREAD",       # 1st quarter spread
    "KXNBA1QTOTAL",        # 1st quarter total
    "KXMVENBASINGLEGAME",  # MVE single game
    "KXMVENBAMULTIGAMEEXTENDED",  # MVE multi game
]

# All NBA series combined
NBA_SERIES_PREFIXES = NBA_PLAYER_PROP_SERIES + NBA_GAME_SERIES

PROP_TYPE_MAP = {
    "points": ["pts", "points", "point"],
    "rebounds": ["reb", "rebounds", "rebound"],
    "assists": ["ast", "assists", "assist"],
    "threes": ["3pm", "three", "threes", "3-pointer", "3-pointers", "three-pointer"],
    "blocks": ["blk", "blocks", "block"],
    "steals": ["stl", "steals", "steal"],
    "triple_double": ["triple double", "triple-double"],
}

# Map series ticker to prop type for reliable classification
SERIES_TO_PROP_TYPE = {
    "KXNBAPTS": "points",
    "KXNBAREB": "rebounds",
    "KXNBAAST": "assists",
    "KXNBABLK": "blocks",
    "KXNBASTL": "steals",
    "KXNBA3D": "triple_double",
    "KXNBAPR": "points_rebounds",
    "KXNBAPRA": "points_rebounds_assists",
    "KXNBARA": "rebounds_assists",
}


def _load_private_key(key_path: str) -> rsa.RSAPrivateKey:
    """Load RSA private key from file."""
    path = Path(key_path)
    if not path.is_absolute():
        # Relative to backend directory
        path = Path(__file__).parent.parent.parent / path
    with open(path, "rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None,
            backend=default_backend(),
        )
    return private_key  # type: ignore[return-value]


def _sign_request(private_key: rsa.RSAPrivateKey, timestamp_ms: str, method: str, path: str) -> str:
    """Create RSA-PSS signature for Kalshi API auth."""
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


class KalshiClient:
    """Client for the Kalshi prediction market API."""

    # Kalshi rate limit: ~10 req/s sustained.  Keep headroom.
    _REQUEST_DELAY = 0.15  # seconds between requests (~6 req/s)
    _MAX_CONCURRENT = 2    # max parallel in-flight requests

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key_id = settings.kalshi_api_key_id
        self.private_key: rsa.RSAPrivateKey | None = None
        self._http = httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)
        self._semaphore = asyncio.Semaphore(self._MAX_CONCURRENT)
        self._last_request_time: float = 0.0

        if self.api_key_id and settings.kalshi_private_key_path:
            try:
                self.private_key = _load_private_key(settings.kalshi_private_key_path)
                logger.info("Kalshi API key loaded", key_id=self.api_key_id[:8] + "...")
            except Exception as e:
                logger.warning("Failed to load Kalshi private key", error=str(e))

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate authentication headers for a request."""
        if not self.private_key or not self.api_key_id:
            return {}
        timestamp_ms = str(int(time.time() * 1000))
        # Kalshi requires the full path including /trade-api/v2 prefix for signing
        full_path = "/trade-api/v2" + path
        signature = _sign_request(self.private_key, timestamp_ms, method, full_path)
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": signature,
        }

    async def _throttle(self) -> None:
        """Enforce minimum delay between requests to avoid 429s."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self._REQUEST_DELAY:
            await asyncio.sleep(self._REQUEST_DELAY - elapsed)
        self._last_request_time = time.monotonic()

    async def _get(self, path: str, params: dict[str, Any] | None = None, auth: bool = False) -> dict[str, Any]:
        """Make a GET request to the Kalshi API."""
        async with self._semaphore:
            for attempt in range(3):
                await self._throttle()
                headers = self._auth_headers("GET", path.split("?")[0]) if auth else {}
                resp = await self._http.get(path, params=params, headers=headers)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"Kalshi API rate limited (429), backing off for {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
            # If we exhausted retries
            resp.raise_for_status()
            return resp.json()

    async def _post(self, path: str, json_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make an authenticated POST request."""
        async with self._semaphore:
            for attempt in range(3):
                await self._throttle()
                headers = self._auth_headers("POST", path)
                resp = await self._http.post(path, json=json_data, headers=headers)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"Kalshi API rate limited (429), backing off for {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
            resp.raise_for_status()
            return resp.json()

    async def _delete(self, path: str) -> dict[str, Any]:
        """Make an authenticated DELETE request."""
        async with self._semaphore:
            for attempt in range(3):
                await self._throttle()
                headers = self._auth_headers("DELETE", path)
                resp = await self._http.delete(path, headers=headers)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 30))
                    logger.warning(f"Kalshi API rate limited (429), backing off for {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
            resp.raise_for_status()
            return resp.json()

    # ── Public endpoints (no auth) ──────────────────────────────────

    async def get_markets(
        self,
        status: str = "open",
        series_ticker: str | None = None,
        limit: int = 200,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Fetch markets with optional filters."""
        params: dict[str, Any] = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
        return await self._get("/markets", params=params)

    async def get_event(self, event_ticker: str) -> dict[str, Any]:
        """Fetch a single event by ticker."""
        return await self._get(f"/events/{event_ticker}")

    async def get_series(self, series_ticker: str) -> dict[str, Any]:
        """Fetch series info."""
        return await self._get(f"/series/{series_ticker}")

    async def get_orderbook(self, ticker: str) -> dict[str, Any]:
        """Fetch orderbook for a market."""
        return await self._get(f"/orderbook/{ticker}")

    # ── Authenticated endpoints ─────────────────────────────────────

    async def get_balance(self) -> dict[str, Any]:
        """Get account balance (requires auth)."""
        return await self._get("/portfolio/balance", auth=True)

    async def get_positions(self, limit: int = 100) -> dict[str, Any]:
        """Get current positions (requires auth)."""
        return await self._get("/portfolio/positions", params={"limit": limit}, auth=True)

    async def place_order(
        self,
        ticker: str,
        side: str,
        action: str = "buy",
        count: int = 1,
        type: str = "limit",
        yes_price: int | None = None,
        no_price: int | None = None,
        expiration_ts: int | None = None,
    ) -> dict[str, Any]:
        """
        Place an order on Kalshi.
        - ticker: market ticker
        - side: "yes" or "no"
        - action: "buy" or "sell"
        - count: number of contracts
        - type: "limit" or "market"
        - yes_price: limit price in cents (1-99) for yes side
        - no_price: limit price in cents (1-99) for no side
        - expiration_ts: unix timestamp for order expiry (optional)
        """
        payload: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": type,
        }
        if yes_price is not None:
            payload["yes_price"] = yes_price
        if no_price is not None:
            payload["no_price"] = no_price
        if expiration_ts is not None:
            payload["expiration_ts"] = expiration_ts
        return await self._post("/portfolio/orders", json_data=payload)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel a resting order."""
        return await self._delete(f"/portfolio/orders/{order_id}")

    async def get_order(self, order_id: str) -> dict[str, Any]:
        """Get details of a specific order."""
        return await self._get(f"/portfolio/orders/{order_id}", auth=True)

    async def get_orders(
        self,
        ticker: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get orders, optionally filtered."""
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        return await self._get("/portfolio/orders", params=params, auth=True)

    async def get_fills(
        self,
        ticker: str | None = None,
        order_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        """Get fill history."""
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        return await self._get("/portfolio/fills", params=params, auth=True)

    async def get_market(self, ticker: str) -> dict[str, Any]:
        """Fetch a single market by ticker."""
        return await self._get(f"/markets/{ticker}")

    # ── NBA-specific helpers ────────────────────────────────────────

    async def _fetch_series_markets(
        self,
        series_list: list[str],
        status: str = "open",
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Fetch all markets for given series tickers.
        Returns list of (series_ticker, raw_market) tuples.
        """
        results: list[tuple[str, dict[str, Any]]] = []
        seen_tickers: set[str] = set()

        for series in series_list:
            try:
                cursor = None
                while True:
                    data = await self.get_markets(
                        status=status,
                        series_ticker=series,
                        limit=200,
                        cursor=cursor,
                    )
                    markets = data.get("markets", [])
                    for m in markets:
                        if m["ticker"] not in seen_tickers:
                            seen_tickers.add(m["ticker"])
                            results.append((series, m))
                    cursor = data.get("cursor")
                    if not cursor or not markets:
                        break
            except httpx.HTTPStatusError:
                continue
            except Exception as e:
                logger.warning("Error fetching Kalshi series", series=series, status=status, error=str(e))
                continue

        return results

    async def get_nba_player_prop_markets(self) -> list[dict[str, Any]]:
        """
        Fetch all open NBA player prop markets.
        Returns parsed market data with player name, prop type, line, prices.
        """
        raw = await self._fetch_series_markets(NBA_SERIES_PREFIXES, status="open")

        parsed = []
        for series, m in raw:
            parsed_market = _parse_player_prop_market(m, series_ticker=series)
            if parsed_market:
                parsed.append(parsed_market)

        logger.info("Fetched Kalshi NBA markets", total_raw=len(raw), parsed=len(parsed))
        return parsed

    async def get_settled_nba_player_prop_markets(
        self,
        series_list: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch all settled NBA player prop markets.
        Returns parsed market data with settlement results.
        """
        target = series_list or NBA_PLAYER_PROP_SERIES
        raw = await self._fetch_series_markets(target, status="settled")

        parsed = []
        for series, m in raw:
            parsed_market = _parse_player_prop_market(m, series_ticker=series)
            if parsed_market:
                parsed.append(parsed_market)

        logger.info("Fetched settled Kalshi markets", total_raw=len(raw), parsed=len(parsed))
        return parsed

    async def get_trades(
        self,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        limit: int = 1000,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        """Fetch trades, optionally filtered by ticker and time range."""
        params: dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        if cursor:
            params["cursor"] = cursor
        return await self._get("/markets/trades", params=params)

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()


def _parse_player_prop_market(market: dict[str, Any], series_ticker: str | None = None) -> dict[str, Any] | None:
    """
    Parse a raw Kalshi market into a structured player prop format.
    Returns None if the market doesn't look like a player prop.
    """
    title = market.get("title", "")
    subtitle = market.get("subtitle", "")
    ticker = market.get("ticker", "")

    # Try to determine prop type from series ticker first (most reliable)
    prop_type = None
    if series_ticker:
        prop_type = SERIES_TO_PROP_TYPE.get(series_ticker)

    # Fallback: infer from title/subtitle keywords
    if not prop_type:
        for ptype, keywords in PROP_TYPE_MAP.items():
            if any(kw in title.lower() or kw in subtitle.lower() for kw in keywords):
                prop_type = ptype
                break

    if not prop_type:
        return None

    # Extract line/strike — Kalshi uses floor_strike for "N+ stat" markets
    line = None
    floor_strike = market.get("floor_strike")
    if floor_strike is not None:
        try:
            line = float(floor_strike)
        except (ValueError, TypeError):
            pass
    if line is None:
        functional_strike = market.get("functional_strike")
        if functional_strike is not None:
            try:
                line = float(functional_strike)
            except (ValueError, TypeError):
                pass
    if line is None:
        cap_strike = market.get("cap_strike")
        if cap_strike is not None:
            try:
                line = float(cap_strike)
            except (ValueError, TypeError):
                pass
    if line is None:
        # Parse from title: "Player: N+ stat" or "Will Player score N+ stat"
        match = re.search(r"(\d+(?:\.\d+)?)\+", title)
        if match:
            line = float(match.group(1))

    # Extract player name from title
    player_name = _extract_player_name(title)

    yes_bid = market.get("yes_bid", 0) or 0
    yes_ask = market.get("yes_ask", 0) or 0
    no_bid = market.get("no_bid", 0) or 0
    no_ask = market.get("no_ask", 0) or 0
    last_price = market.get("last_price", 0) or 0
    volume = market.get("volume", 0) or 0

    # Implied probability from yes_ask (cost to buy YES)
    implied_prob_yes = yes_ask / 100.0 if yes_ask else (last_price / 100.0 if last_price else 0)
    implied_prob_no = no_ask / 100.0 if no_ask else ((100 - last_price) / 100.0 if last_price else 0)

    return {
        "ticker": ticker,
        "event_ticker": market.get("event_ticker", ""),
        "series_ticker": series_ticker or "",
        "title": title,
        "subtitle": subtitle,
        "player_name": player_name,
        "prop_type": prop_type,
        "line": line,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "no_bid": no_bid,
        "no_ask": no_ask,
        "last_price": last_price,
        "implied_prob_over": round(implied_prob_yes, 4),
        "implied_prob_under": round(implied_prob_no, 4),
        "volume": volume,
        "volume_24h": market.get("volume_24h", 0) or 0,
        "open_interest": market.get("open_interest", 0) or 0,
        "status": market.get("status", ""),
        "close_time": market.get("close_time", ""),
        "settlement_ts": market.get("settlement_ts", ""),
        "result": market.get("result"),
        "settlement_value": market.get("settlement_value"),
        "strike_type": market.get("strike_type", ""),
    }


def _extract_player_name(title: str) -> str:
    """
    Try to extract a player name from a Kalshi market title.
    Actual Kalshi formats:
    - "Chet Holmgren: 3+ blocks"
    - "LeBron James: Triple Double"
    - "Jaren Jackson Jr.: 30+ points"
    """
    # Primary pattern: "Player Name: N+ stat" or "Player Name: Triple Double"
    match = re.match(r"^(.+?):\s", title)
    if match:
        return match.group(1).strip()

    # Fallback: "Will <Name> score/record/have..."
    match = re.match(r"(?:Will\s+)?(.+?)\s+(?:score|record|have|get|make|hit)\b", title, re.IGNORECASE)
    if match:
        name = match.group(1).strip()
        name = re.sub(r"^(the|a)\s+", "", name, flags=re.IGNORECASE)
        return name

    # Fallback: "<Name> Over/Under X.X"
    match = re.match(r"(.+?)\s+(?:over|under)\s+\d", title, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    return ""


# Singleton client
_client: KalshiClient | None = None


def get_kalshi_client() -> KalshiClient:
    """Get or create the singleton Kalshi client."""
    global _client
    if _client is None:
        _client = KalshiClient()
    return _client
