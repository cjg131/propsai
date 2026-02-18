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

import time
from typing import Any

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Map Kalshi market categories to Polymarket search terms
CATEGORY_SEARCH: dict[str, list[str]] = {
    "crypto": ["bitcoin", "btc", "ethereum", "eth", "solana", "crypto"],
    "finance": ["s&p 500", "sp500", "nasdaq", "stock market"],
    "econ": ["cpi", "inflation", "fed rate", "unemployment", "gdp"],
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
                params={"title": query, "active": "true", "limit": limit},
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

    async def get_cross_market_signals(
        self, category: str = "crypto"
    ) -> dict[str, dict[str, Any]]:
        """
        Fetch Polymarket prices for markets matching a category.

        Returns:
            {event_slug: {"title": str, "poly_price": float, "token_id": str, "markets": [...]}}
        """
        now = time.time()
        cache_key = f"signals_{category}"
        if cache_key in self._cache and (now - self._cache_ts) < self._cache_ttl:
            return self._cache[cache_key]

        search_terms = CATEGORY_SEARCH.get(category, [category])
        all_events: dict[str, dict[str, Any]] = {}

        for term in search_terms[:3]:  # Limit searches
            events = await self.search_events(term, limit=5)
            for event in events:
                slug = event.get("slug", "")
                if not slug or slug in all_events:
                    continue

                title = event.get("title", "")
                markets = event.get("markets", [])

                for mkt in markets[:3]:  # Limit markets per event
                    token_ids = mkt.get("clobTokenIds", [])
                    if not token_ids:
                        continue

                    # Get price for YES token (first token)
                    price = await self.get_market_price(token_ids[0])
                    if price is not None and price > 0:
                        all_events[slug] = {
                            "title": title,
                            "question": mkt.get("question", title),
                            "poly_price": round(price * 100, 1),  # Convert to cents
                            "token_id": token_ids[0],
                        }
                        break  # One price per event is enough

        self._cache[cache_key] = all_events
        self._cache_ts = now

        logger.info(f"Polymarket: found {len(all_events)} {category} markets with prices")
        return all_events

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
