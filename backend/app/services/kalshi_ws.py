"""
Kalshi Orderbook WebSocket Client.

Maintains persistent WebSocket connections to Kalshi for real-time
orderbook updates on subscribed tickers. Provides:
  - Real-time bid/ask/last price updates
  - Orderbook depth snapshots
  - Trade tape (last trades)
  - Stale price detection (price hasn't moved despite external data changes)

Usage:
    ws = KalshiWebSocket()
    await ws.connect()
    await ws.subscribe(["TICKER1", "TICKER2"])
    snapshot = ws.get_snapshot("TICKER1")
    await ws.close()
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any, Callable

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from app.config import get_settings
from app.logging_config import get_logger
from app.services.kalshi_api import _load_private_key, _sign_request

logger = get_logger(__name__)

WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

# How long before we consider a price "stale" (no updates)
STALE_THRESHOLD_SECONDS = 300  # 5 minutes


class OrderbookSnapshot:
    """In-memory snapshot of a single market's orderbook state."""

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker
        self.yes_bid: int = 0
        self.yes_ask: int = 0
        self.no_bid: int = 0
        self.no_ask: int = 0
        self.last_price: int = 0
        self.last_trade_side: str = ""
        self.last_trade_count: int = 0
        self.volume: int = 0
        self.open_interest: int = 0
        self.last_update: float = 0.0  # Unix timestamp
        self.update_count: int = 0
        self.price_history: list[dict] = []  # Last N price changes

    @property
    def spread(self) -> int:
        """Yes bid-ask spread in cents."""
        if self.yes_bid > 0 and self.yes_ask > 0:
            return self.yes_ask - self.yes_bid
        return 99

    @property
    def mid_price(self) -> float:
        """Midpoint of yes bid/ask."""
        if self.yes_bid > 0 and self.yes_ask > 0:
            return (self.yes_bid + self.yes_ask) / 2.0
        return 0.0

    @property
    def is_stale(self) -> bool:
        """True if no updates received for STALE_THRESHOLD_SECONDS."""
        if self.last_update == 0:
            return True
        return (time.time() - self.last_update) > STALE_THRESHOLD_SECONDS

    @property
    def seconds_since_update(self) -> float:
        if self.last_update == 0:
            return float("inf")
        return time.time() - self.last_update

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "yes_bid": self.yes_bid,
            "yes_ask": self.yes_ask,
            "no_bid": self.no_bid,
            "no_ask": self.no_ask,
            "last_price": self.last_price,
            "spread": self.spread,
            "mid_price": self.mid_price,
            "is_stale": self.is_stale,
            "seconds_since_update": round(self.seconds_since_update, 1),
            "update_count": self.update_count,
            "volume": self.volume,
        }


class KalshiWebSocket:
    """Persistent WebSocket connection to Kalshi for real-time orderbook data."""

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key_id = settings.kalshi_api_key_id
        self._private_key = None
        if self.api_key_id and settings.kalshi_private_key_path:
            try:
                self._private_key = _load_private_key(settings.kalshi_private_key_path)
            except Exception as e:
                logger.warning("WS: Failed to load private key", error=str(e))

        self._ws: Any = None
        self._connected = False
        self._subscribed_tickers: set[str] = set()
        self._snapshots: dict[str, OrderbookSnapshot] = {}
        self._callbacks: list[Callable] = []
        self._recv_task: asyncio.Task | None = None
        self._heartbeat_task: asyncio.Task | None = None
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._should_run = False
        self._msg_count = 0

    # ── Connection management ────────────────────────────────────────

    async def connect(self) -> bool:
        """Establish WebSocket connection with auth."""
        self._should_run = True

        try:
            # Build auth headers
            headers = {}
            if self._private_key and self.api_key_id:
                timestamp_ms = str(int(time.time() * 1000))
                # For WS, sign the GET /trade-api/ws/v2 path
                signature = _sign_request(
                    self._private_key, timestamp_ms, "GET", "/trade-api/ws/v2"
                )
                headers = {
                    "KALSHI-ACCESS-KEY": self.api_key_id,
                    "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
                    "KALSHI-ACCESS-SIGNATURE": signature,
                }

            self._ws = await ws_connect(
                WS_URL,
                additional_headers=headers,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=5,
            )
            self._connected = True
            self._reconnect_delay = 1.0
            logger.info("Kalshi WS connected")

            # Start receiver and heartbeat
            self._recv_task = asyncio.create_task(self._receive_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Re-subscribe to any previously subscribed tickers
            if self._subscribed_tickers:
                await self._send_subscribe(list(self._subscribed_tickers))

            return True

        except Exception as e:
            logger.warning("Kalshi WS connect failed", error=str(e))
            self._connected = False
            return False

    async def close(self) -> None:
        """Close WebSocket connection."""
        self._should_run = False
        self._connected = False

        if self._recv_task:
            self._recv_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        logger.info("Kalshi WS closed")

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff."""
        if not self._should_run:
            return

        self._connected = False
        logger.info("Kalshi WS reconnecting", delay=self._reconnect_delay)
        await asyncio.sleep(self._reconnect_delay)
        self._reconnect_delay = min(
            self._reconnect_delay * 2, self._max_reconnect_delay
        )

        try:
            await self.connect()
        except Exception as e:
            logger.warning("Kalshi WS reconnect failed", error=str(e))
            if self._should_run:
                asyncio.create_task(self._reconnect())

    # ── Subscription management ──────────────────────────────────────

    async def subscribe(self, tickers: list[str]) -> None:
        """Subscribe to orderbook updates for given tickers."""
        new_tickers = [t for t in tickers if t not in self._subscribed_tickers]
        if not new_tickers:
            return

        for t in new_tickers:
            self._subscribed_tickers.add(t)
            if t not in self._snapshots:
                self._snapshots[t] = OrderbookSnapshot(t)

        if self._connected:
            await self._send_subscribe(new_tickers)

    async def unsubscribe(self, tickers: list[str]) -> None:
        """Unsubscribe from orderbook updates."""
        for t in tickers:
            self._subscribed_tickers.discard(t)
            self._snapshots.pop(t, None)

        if self._connected:
            await self._send_unsubscribe(tickers)

    async def _send_subscribe(self, tickers: list[str]) -> None:
        """Send subscription message to Kalshi WS."""
        if not self._ws:
            return

        # Kalshi WS uses channel-based subscriptions
        # Subscribe to orderbook_delta and ticker channels
        for ticker in tickers:
            try:
                msg = {
                    "id": self._msg_count,
                    "cmd": "subscribe",
                    "params": {
                        "channels": ["orderbook_delta", "ticker"],
                        "market_tickers": [ticker],
                    },
                }
                self._msg_count += 1
                await self._ws.send(json.dumps(msg))
                logger.debug("WS subscribed", ticker=ticker)
            except Exception as e:
                logger.warning("WS subscribe failed", ticker=ticker, error=str(e))

    async def _send_unsubscribe(self, tickers: list[str]) -> None:
        """Send unsubscribe message."""
        if not self._ws:
            return

        for ticker in tickers:
            try:
                msg = {
                    "id": self._msg_count,
                    "cmd": "unsubscribe",
                    "params": {
                        "channels": ["orderbook_delta", "ticker"],
                        "market_tickers": [ticker],
                    },
                }
                self._msg_count += 1
                await self._ws.send(json.dumps(msg))
            except Exception:
                pass

    # ── Message handling ─────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """Main receive loop — processes incoming WS messages."""
        try:
            async for raw_msg in self._ws:
                try:
                    msg = json.loads(raw_msg)
                    await self._handle_message(msg)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug("WS message handling error", error=str(e))

        except ConnectionClosed:
            logger.info("Kalshi WS connection closed")
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.warning("Kalshi WS receive error", error=str(e))

        # Reconnect if we should still be running
        if self._should_run:
            asyncio.create_task(self._reconnect())

    async def _handle_message(self, msg: dict) -> None:
        """Route incoming message to appropriate handler."""
        msg_type = msg.get("type", "")
        channel = msg.get("channel", "")

        if msg_type == "orderbook_snapshot":
            self._handle_orderbook_snapshot(msg)
        elif msg_type == "orderbook_delta":
            self._handle_orderbook_delta(msg)
        elif msg_type == "ticker" or channel == "ticker":
            self._handle_ticker_update(msg)
        elif msg_type == "trade":
            self._handle_trade(msg)
        elif msg_type == "subscribed":
            logger.debug("WS subscription confirmed", msg=msg)
        elif msg_type == "error":
            logger.warning("WS error message", msg=msg)

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(msg)
            except Exception:
                pass

    def _handle_orderbook_snapshot(self, msg: dict) -> None:
        """Handle full orderbook snapshot."""
        data = msg.get("msg", msg)
        ticker = data.get("market_ticker", "")
        if not ticker or ticker not in self._snapshots:
            return

        snap = self._snapshots[ticker]
        snap.yes_bid = data.get("yes_bid", snap.yes_bid) or 0
        snap.yes_ask = data.get("yes_ask", snap.yes_ask) or 0
        snap.no_bid = data.get("no_bid", snap.no_bid) or 0
        snap.no_ask = data.get("no_ask", snap.no_ask) or 0
        snap.last_update = time.time()
        snap.update_count += 1

    def _handle_orderbook_delta(self, msg: dict) -> None:
        """Handle incremental orderbook update."""
        data = msg.get("msg", msg)
        ticker = data.get("market_ticker", "")
        if not ticker or ticker not in self._snapshots:
            return

        snap = self._snapshots[ticker]
        old_mid = snap.mid_price

        if "yes_bid" in data and data["yes_bid"] is not None:
            snap.yes_bid = data["yes_bid"]
        if "yes_ask" in data and data["yes_ask"] is not None:
            snap.yes_ask = data["yes_ask"]
        if "no_bid" in data and data["no_bid"] is not None:
            snap.no_bid = data["no_bid"]
        if "no_ask" in data and data["no_ask"] is not None:
            snap.no_ask = data["no_ask"]

        snap.last_update = time.time()
        snap.update_count += 1

        # Track price changes
        new_mid = snap.mid_price
        if old_mid > 0 and new_mid > 0 and abs(new_mid - old_mid) >= 0.5:
            snap.price_history.append({
                "time": snap.last_update,
                "old_mid": old_mid,
                "new_mid": new_mid,
                "change": new_mid - old_mid,
            })
            # Keep last 50 changes
            if len(snap.price_history) > 50:
                snap.price_history = snap.price_history[-50:]

    def _handle_ticker_update(self, msg: dict) -> None:
        """Handle ticker-level update (price, volume, etc.)."""
        data = msg.get("msg", msg)
        ticker = data.get("market_ticker", "")
        if not ticker or ticker not in self._snapshots:
            return

        snap = self._snapshots[ticker]
        if "yes_bid" in data and data["yes_bid"] is not None:
            snap.yes_bid = data["yes_bid"]
        if "yes_ask" in data and data["yes_ask"] is not None:
            snap.yes_ask = data["yes_ask"]
        if "no_bid" in data and data["no_bid"] is not None:
            snap.no_bid = data["no_bid"]
        if "no_ask" in data and data["no_ask"] is not None:
            snap.no_ask = data["no_ask"]
        if "last_price" in data and data["last_price"] is not None:
            snap.last_price = data["last_price"]
        if "volume" in data and data["volume"] is not None:
            snap.volume = data["volume"]
        if "open_interest" in data and data["open_interest"] is not None:
            snap.open_interest = data["open_interest"]

        snap.last_update = time.time()
        snap.update_count += 1

    def _handle_trade(self, msg: dict) -> None:
        """Handle trade execution message."""
        data = msg.get("msg", msg)
        ticker = data.get("market_ticker", "")
        if not ticker or ticker not in self._snapshots:
            return

        snap = self._snapshots[ticker]
        snap.last_trade_side = data.get("taker_side", "")
        snap.last_trade_count = data.get("count", 0)
        if "yes_price" in data:
            snap.last_price = data["yes_price"]
        snap.last_update = time.time()
        snap.update_count += 1

    # ── Heartbeat ────────────────────────────────────────────────────

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to keep connection alive."""
        try:
            while self._should_run and self._connected:
                await asyncio.sleep(25)
                if self._ws and self._connected:
                    try:
                        await self._ws.ping()
                    except Exception:
                        break
        except asyncio.CancelledError:
            return

    # ── Public API ───────────────────────────────────────────────────

    def get_snapshot(self, ticker: str) -> OrderbookSnapshot | None:
        """Get current orderbook snapshot for a ticker."""
        return self._snapshots.get(ticker)

    def get_all_snapshots(self) -> dict[str, OrderbookSnapshot]:
        """Get all current snapshots."""
        return dict(self._snapshots)

    def get_stale_tickers(self) -> list[str]:
        """Get list of tickers with stale prices."""
        return [
            ticker for ticker, snap in self._snapshots.items()
            if snap.is_stale
        ]

    def get_price_changes(self, ticker: str, last_n: int = 10) -> list[dict]:
        """Get recent price changes for a ticker."""
        snap = self._snapshots.get(ticker)
        if not snap:
            return []
        return snap.price_history[-last_n:]

    def add_callback(self, callback: Callable) -> None:
        """Register a callback for all WS messages."""
        self._callbacks.append(callback)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def subscribed_count(self) -> int:
        return len(self._subscribed_tickers)

    def get_status(self) -> dict[str, Any]:
        """Get WebSocket connection status."""
        return {
            "connected": self._connected,
            "subscribed_tickers": len(self._subscribed_tickers),
            "total_updates": sum(s.update_count for s in self._snapshots.values()),
            "stale_count": len(self.get_stale_tickers()),
        }


# ── Singleton ────────────────────────────────────────────────────────

_ws_client: KalshiWebSocket | None = None


def get_kalshi_ws() -> KalshiWebSocket:
    """Get or create the singleton WebSocket client."""
    global _ws_client
    if _ws_client is None:
        _ws_client = KalshiWebSocket()
    return _ws_client
