"""
Polymarket Trading Client.

Handles order placement, position management, and API credential generation
for Polymarket's CLOB (Central Limit Order Book) on Polygon.

Uses py-clob-client SDK for order signing and execution.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from decimal import Decimal
from typing import Any

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

CLOB_BASE = "https://clob.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"
CHAIN_ID = 137  # Polygon mainnet


class PolymarketTrader:
    """Authenticated Polymarket trading client."""

    def __init__(self) -> None:
        self._private_key: str = os.getenv("POLYMARKET_PRIVATE_KEY", "")
        self._api_key: str = os.getenv("POLYMARKET_API_KEY", "")
        self._api_secret: str = os.getenv("POLYMARKET_API_SECRET", "")
        self._api_passphrase: str = os.getenv("POLYMARKET_API_PASSPHRASE", "")
        self._clob_client: Any = None
        self._http: httpx.AsyncClient | None = None
        self._initialized = False
        self._positions_cache: dict[str, Any] = {}
        self._positions_cache_ts: float = 0.0

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=15)
        return self._http

    def _get_clob_client(self) -> Any:
        """Lazy-init the py-clob-client SDK."""
        if self._clob_client is not None:
            return self._clob_client

        try:
            from py_clob_client.client import ClobClient

            if self._api_key and self._api_secret and self._api_passphrase:
                # Use existing L2 credentials — build ApiCreds object
                try:
                    from py_clob_client.clob_types import ApiCreds
                    creds_obj = ApiCreds(
                        api_key=self._api_key,
                        api_secret=self._api_secret,
                        api_passphrase=self._api_passphrase,
                    )
                except ImportError:
                    creds_obj = {
                        "apiKey": self._api_key,
                        "secret": self._api_secret,
                        "passphrase": self._api_passphrase,
                    }
                self._clob_client = ClobClient(
                    host=CLOB_BASE,
                    chain_id=CHAIN_ID,
                    key=self._private_key,
                    creds=creds_obj,
                )
            elif self._private_key:
                # Will derive/create credentials from private key
                self._clob_client = ClobClient(
                    host=CLOB_BASE,
                    chain_id=CHAIN_ID,
                    key=self._private_key,
                )
            else:
                logger.error("No Polymarket credentials configured")
                return None

            return self._clob_client
        except ImportError:
            logger.error("py-clob-client not installed. Run: pip install py-clob-client")
            return None

    async def initialize(self) -> bool:
        """Initialize client and generate API credentials if needed."""
        if self._initialized:
            return True

        if not self._private_key:
            logger.error("POLYMARKET_PRIVATE_KEY not set")
            return False

        client = self._get_clob_client()
        if client is None:
            return False

        # If we don't have L2 creds, derive them
        if not self._api_key:
            try:
                creds = await asyncio.to_thread(client.create_or_derive_api_creds)
                # py-clob-client >= 0.30 returns ApiCreds object, not dict
                if hasattr(creds, "api_key"):
                    self._api_key = creds.api_key
                    self._api_secret = creds.api_secret
                    self._api_passphrase = creds.api_passphrase
                elif isinstance(creds, dict):
                    self._api_key = creds.get("apiKey", "")
                    self._api_secret = creds.get("secret", "")
                    self._api_passphrase = creds.get("passphrase", "")

                # Re-init client with derived creds object
                from py_clob_client.client import ClobClient
                self._clob_client = ClobClient(
                    host=CLOB_BASE,
                    chain_id=CHAIN_ID,
                    key=self._private_key,
                    creds=creds,  # Pass the ApiCreds object directly
                )

                logger.info("Polymarket API credentials derived successfully")
            except Exception as e:
                logger.error(f"Failed to derive Polymarket API credentials: {e}")
                return False

        self._initialized = True
        logger.info("Polymarket trader initialized")
        return True

    # ── Market Data ──────────────────────────────────────────────

    async def get_market(self, condition_id: str) -> dict[str, Any] | None:
        """Get market details by condition ID."""
        http = await self._get_http()
        try:
            resp = await http.get(f"{GAMMA_BASE}/markets/{condition_id}")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"Failed to get market {condition_id}: {e}")
            return None

    async def get_orderbook(self, token_id: str) -> dict[str, Any] | None:
        """Get order book for a token."""
        http = await self._get_http()
        try:
            resp = await http.get(f"{CLOB_BASE}/book", params={"token_id": token_id})
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug(f"Failed to get orderbook: {e}")
            return None

    async def get_price(self, token_id: str, side: str = "buy") -> float | None:
        """Get current price for a token."""
        http = await self._get_http()
        try:
            resp = await http.get(
                f"{CLOB_BASE}/price",
                params={"token_id": token_id, "side": side},
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("price", 0))
        except Exception as e:
            logger.debug(f"Failed to get price: {e}")
            return None

    async def get_midpoint(self, token_id: str) -> float | None:
        """Get midpoint price for a token."""
        http = await self._get_http()
        try:
            resp = await http.get(
                f"{CLOB_BASE}/midpoint",
                params={"token_id": token_id},
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data.get("mid", 0))
        except Exception as e:
            logger.debug(f"Failed to get midpoint: {e}")
            return None

    async def search_markets(
        self, query: str = "", limit: int = 100, active: bool = True
    ) -> list[dict[str, Any]]:
        """Search Polymarket markets."""
        http = await self._get_http()
        try:
            params: dict[str, Any] = {
                "active": str(active).lower(),
                "closed": "false",
                "limit": limit,
            }
            if query:
                params["title"] = query

            resp = await http.get(f"{GAMMA_BASE}/markets", params=params)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else data.get("markets", [])
        except Exception as e:
            logger.debug(f"Market search failed: {e}")
            return []

    # ── Order Execution ──────────────────────────────────────────

    async def place_limit_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        price: float,  # 0.0 to 1.0
        size: float,  # number of contracts
    ) -> dict[str, Any] | None:
        """Place a limit order on Polymarket.

        Args:
            token_id: The CLOB token ID for the outcome
            side: "BUY" or "SELL"
            price: Price per contract (0.0 to 1.0)
            size: Number of contracts

        Returns:
            Order response dict or None on failure
        """
        if not await self.initialize():
            return None

        try:
            from py_clob_client.order import OrderArgs

            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=side.upper(),
            )

            signed_order = await asyncio.to_thread(
                self._clob_client.create_order, order_args, "GTC"
            )
            response = await asyncio.to_thread(
                self._clob_client.post_order, signed_order
            )

            logger.info(
                "Polymarket limit order placed",
                side=side,
                price=price,
                size=size,
                token_id=token_id[:20],
                response=str(response)[:100],
            )
            return response if isinstance(response, dict) else {"result": str(response)}

        except Exception as e:
            logger.error(f"Failed to place limit order: {e}")
            return None

    async def place_market_order(
        self,
        token_id: str,
        side: str,  # "BUY" or "SELL"
        amount: float,  # USDC amount to spend
    ) -> dict[str, Any] | None:
        """Place a market order on Polymarket.

        Args:
            token_id: The CLOB token ID for the outcome
            side: "BUY" or "SELL"
            amount: Amount in USDC to spend

        Returns:
            Order response dict or None on failure
        """
        if not await self.initialize():
            return None

        try:
            from py_clob_client.order import MarketOrderArgs

            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount,
                side=side.upper(),
            )

            signed_order = await asyncio.to_thread(
                self._clob_client.create_market_order, order_args
            )
            response = await asyncio.to_thread(
                self._clob_client.post_order, signed_order
            )

            logger.info(
                "Polymarket market order placed",
                side=side,
                amount=amount,
                token_id=token_id[:20],
                response=str(response)[:100],
            )
            return response if isinstance(response, dict) else {"result": str(response)}

        except Exception as e:
            logger.error(f"Failed to place market order: {e}")
            return None

    # ── Position Management ──────────────────────────────────────

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get current open positions."""
        if not await self.initialize():
            return []

        try:
            positions = await asyncio.to_thread(self._clob_client.get_positions)
            return positions if isinstance(positions, list) else []
        except Exception as e:
            logger.debug(f"Failed to get positions: {e}")
            return []

    async def get_open_orders(self) -> list[dict[str, Any]]:
        """Get current open orders."""
        if not await self.initialize():
            return []

        try:
            orders = await asyncio.to_thread(self._clob_client.get_orders)
            return orders if isinstance(orders, list) else []
        except Exception as e:
            logger.debug(f"Failed to get open orders: {e}")
            return []

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if not await self.initialize():
            return False

        try:
            await asyncio.to_thread(self._clob_client.cancel, order_id)
            logger.info(f"Cancelled order {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders."""
        if not await self.initialize():
            return False

        try:
            await asyncio.to_thread(self._clob_client.cancel_all)
            logger.info("Cancelled all open orders")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders: {e}")
            return False

    # ── Balance ──────────────────────────────────────────────────

    async def get_balance(self) -> float | None:
        """Get USDC balance on Polymarket."""
        if not await self.initialize():
            return None

        try:
            balance = await asyncio.to_thread(self._clob_client.get_balance_allowance)
            # May return dict or numeric depending on SDK version
            if isinstance(balance, dict):
                return float(balance.get("balance", balance.get("allowance", 0)))
            return float(balance) if balance is not None else None
        except Exception as e:
            logger.debug(f"Failed to get balance: {e}")
            return None


# ── Singleton ────────────────────────────────────────────────────

_trader: PolymarketTrader | None = None


def get_polymarket_trader() -> PolymarketTrader:
    global _trader
    if _trader is None:
        _trader = PolymarketTrader()
    return _trader
