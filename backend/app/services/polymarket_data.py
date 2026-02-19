"""
Polymarket Cross-Reference Service.

Fetches prices from Polymarket's free public API (no key needed) for
markets that overlap with Kalshi. When Polymarket and Kalshi prices
diverge significantly, that's a cross-market edge signal.

Endpoints:
  - Gamma API: https://gamma-api.polymarket.com/events (market discovery)
  - CLOB API: https://clob.polymarket.com/price (current price)
"""
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Map Kalshi market categories to Polymarket keyword filters.
# Polymarket /markets has sports futures + political/macro — NOT crypto price brackets.
# These keywords match what actually exists on Polymarket's active market feed.
CATEGORY_SEARCH: dict[str, list[str]] = {
    "crypto": ["bitcoin", "ethereum", "crypto", "btc", "eth", "megaeth", "solana"],
    "finance": ["stock", "s&p", "nasdaq", "market cap", "ipo"],
    "econ": ["federal reserve", "inflation", "recession", "gdp", "unemployment", "tariff", "trade"],
    "sports": ["nba finals", "stanley cup", "world cup", "super bowl", "championship"],
}


class PolymarketData:
    """Fetches cross-reference prices from Polymarket."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, dict] = {}
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 120.0  # 2 minute cache

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=10)
        return self._client

    async def search_events(self, query: str, limit: int = 10) -> list[dict]:
        """Search Polymarket events by keyword."""
        client = await self._get_client()
        try:
            resp = await client.get(
                f"{GAMMA_BASE}/events",
                params={"title": query, "active": "true", "closed": "false", "limit": limit},
            )
            resp.raise_for_status()
            events = resp.json()
            return events if isinstance(events, list) else []
        except Exception as e:
            logger.debug(f"Polymarket search failed: {e}")
            return []

    async def get_market_price(self, token_id: str) -> float | None:
        """Get current YES price for a Polymarket market token."""
        client = await self._get_client()
        try:
            resp = await client.get(
                f"{CLOB_BASE}/price",
                params={"token_id": token_id, "side": "buy"},
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("price", 0))
        except Exception as e:
            logger.debug(f"Polymarket price failed: {e}")
            return None

    async def _fetch_active_markets(self, limit: int = 200) -> list[dict]:
        """Fetch active Polymarket markets via /markets endpoint (works reliably)."""
        client = await self._get_client()
        try:
            resp = await client.get(
                f"{GAMMA_BASE}/markets",
                params={"active": "true", "closed": "false", "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("markets", data.get("results", []))
        except Exception as e:
            logger.debug(f"Polymarket markets fetch failed: {e}")
            return []

    async def get_cross_market_signals(
        self, category: str = "crypto"
    ) -> dict[str, dict[str, Any]]:
        """
        Fetch Polymarket prices for markets matching a category.

        Uses /markets endpoint (bulk fetch + local keyword filter) since
        the /events search API returns stale/wrong results.

        Returns:
            {slug: {"title": str, "poly_price": float, "question": str}}
        """
        now = time.time()
        cache_key = f"signals_{category}"
        if cache_key in self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache[cache_key]

        keywords = CATEGORY_SEARCH.get(category, [category])
        all_markets = await self._fetch_active_markets(limit=200)
        matched: dict[str, dict[str, Any]] = {}

        for mkt in all_markets:
            question = (mkt.get("question") or mkt.get("title") or "").lower()
            if not question:
                continue

            # Check if any keyword matches the question
            if not any(kw.lower() in question for kw in keywords):
                continue

            # Parse outcomePrices — JSON string like '["0.65", "0.35"]'
            outcome_prices_raw = mkt.get("outcomePrices", "[]")
            try:
                outcome_prices = json.loads(outcome_prices_raw) if isinstance(outcome_prices_raw, str) else outcome_prices_raw
                yes_price = float(outcome_prices[0]) if outcome_prices else None
            except (json.JSONDecodeError, IndexError, ValueError, TypeError):
                yes_price = None

            # Skip resolved/illiquid markets
            if yes_price is None or yes_price <= 0.02 or yes_price >= 0.98:
                continue

            slug = mkt.get("slug") or mkt.get("id") or question[:40]
            # clobTokenIds is a JSON string like '["123","456"]' — parse it
            clob_raw = mkt.get("clobTokenIds", "[]")
            try:
                clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else (clob_raw or [])
                token_id = clob_ids[0] if clob_ids else ""
            except (json.JSONDecodeError, IndexError, TypeError):
                token_id = ""
            matched[str(slug)] = {
                "title": mkt.get("title") or question,
                "question": mkt.get("question") or question,
                "poly_price": round(yes_price * 100, 1),
                "token_id": token_id,
            }

        self._cache[cache_key] = matched
        self._cache_ts = now

        logger.info(f"Polymarket: found {len(matched)} {category} markets with prices")
        return matched

    async def get_edge_signal(
        self, kalshi_title: str, kalshi_price_cents: int, category: str = "crypto"
    ) -> dict[str, Any] | None:
        """
        Check if Polymarket has a matching market with a divergent price.

        Returns edge signal if divergence > 5%, else None.
        """
        poly_markets = await self.get_cross_market_signals(category)

        # Simple keyword matching between Kalshi title and Polymarket titles
        kalshi_lower = kalshi_title.lower()
        best_match = None
        best_overlap = 0

        for slug, data in poly_markets.items():
            poly_lower = data["question"].lower()
            # Count overlapping significant words
            kalshi_words = set(w for w in kalshi_lower.split() if len(w) > 3)
            poly_words = set(w for w in poly_lower.split() if len(w) > 3)
            overlap = len(kalshi_words & poly_words)

            if overlap > best_overlap and overlap >= 2:
                best_overlap = overlap
                best_match = data

        if not best_match:
            return None

        poly_price = best_match["poly_price"]
        divergence = (poly_price - kalshi_price_cents) / 100.0

        if abs(divergence) < 0.05:
            return None  # Less than 5% divergence — not significant

        return {
            "poly_title": best_match["question"],
            "poly_price_cents": poly_price,
            "kalshi_price_cents": kalshi_price_cents,
            "divergence": round(divergence, 3),
            "direction": "poly_higher" if divergence > 0 else "poly_lower",
            "confidence_boost": min(abs(divergence) * 0.5, 0.15),  # Max 15% boost
        }


# Singleton
_poly: PolymarketData | None = None


def get_polymarket_data() -> PolymarketData:
    global _poly
    if _poly is None:
        _poly = PolymarketData()
    return _poly
