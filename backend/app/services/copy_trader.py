"""
Polymarket Copy Trading Engine.

Monitors top traders' wallets on-chain and mirrors their trades.
Uses Polymarket's public APIs and on-chain data to detect new positions.

Strategy:
1. Track a curated list of profitable wallets (from leaderboard / manual)
2. Poll their positions for changes every 30-60 seconds
3. When a tracked wallet opens/increases a position, mirror it
4. Apply position sizing rules relative to our bankroll
5. Log everything for performance tracking
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx

from app.logging_config import get_logger
from app.services.polymarket_trader import PolymarketTrader, get_polymarket_trader

logger = get_logger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"

# Default wallets to track — will be loaded from env/config
DEFAULT_TRACKED_WALLETS: list[str] = []

# Position sizing
MAX_POSITION_USDC = 15.0  # Max $15 per copy trade (conservative for $250 bankroll)
MIN_POSITION_USDC = 2.0   # Min $2 per trade
MAX_PORTFOLIO_PCT = 0.06   # Max 6% of bankroll per position
MAX_TOTAL_DEPLOYED = 0.50  # Max 50% of bankroll deployed at once


class CopyTrader:
    """Monitors whale wallets and mirrors their Polymarket trades."""

    def __init__(self, trader: PolymarketTrader | None = None) -> None:
        self._trader = trader or get_polymarket_trader()
        self._http: httpx.AsyncClient | None = None
        self._tracked_wallets: list[str] = self._load_tracked_wallets()
        self._wallet_positions: dict[str, dict[str, Any]] = {}  # wallet -> {market -> position}
        self._our_copies: dict[str, dict[str, Any]] = {}  # market -> our copy details
        self._last_scan_ts: float = 0.0
        self._scan_interval: float = 45.0  # seconds between scans
        self._bankroll: float = float(os.getenv("POLYMARKET_BANKROLL", "250"))
        self._total_deployed: float = 0.0
        self._paper_mode: bool = os.getenv("PAPER_MODE", "true").lower() == "true"

    def _load_tracked_wallets(self) -> list[str]:
        """Load tracked wallets from env or defaults."""
        env_wallets = os.getenv("POLYMARKET_TRACKED_WALLETS", "")
        if env_wallets:
            return [w.strip().lower() for w in env_wallets.split(",") if w.strip()]
        return [w.lower() for w in DEFAULT_TRACKED_WALLETS]

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=15)
        return self._http

    # ── Wallet Discovery ─────────────────────────────────────────

    async def discover_top_traders(self, limit: int = 20) -> list[dict[str, Any]]:
        """Discover top traders from Polymarket leaderboard.

        Returns list of {address, pnl, volume, num_trades, win_rate} dicts.
        """
        http = await self._get_http()
        traders: list[dict[str, Any]] = []

        try:
            # Try the Polymarket profiles/leaderboard API
            resp = await http.get(
                f"{DATA_BASE}/leaderboard",
                params={"limit": limit, "period": "monthly", "sort": "profit"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for entry in data if isinstance(data, list) else data.get("results", []):
                    traders.append({
                        "address": entry.get("address", entry.get("wallet", "")),
                        "pnl": float(entry.get("pnl", entry.get("profit", 0))),
                        "volume": float(entry.get("volume", 0)),
                        "num_trades": int(entry.get("numTrades", entry.get("trades", 0))),
                        "display_name": entry.get("displayName", entry.get("username", "")),
                    })
                logger.info(f"Discovered {len(traders)} top traders from leaderboard")
                return traders
        except Exception as e:
            logger.debug(f"Leaderboard fetch failed: {e}")

        # Fallback: try gamma API profiles
        try:
            resp = await http.get(
                f"{GAMMA_BASE}/profiles",
                params={"limit": limit, "sortBy": "pnl", "order": "desc"},
            )
            if resp.status_code == 200:
                data = resp.json()
                for entry in data if isinstance(data, list) else []:
                    traders.append({
                        "address": entry.get("proxyWallet", entry.get("address", "")),
                        "pnl": float(entry.get("pnl", 0)),
                        "volume": float(entry.get("volume", 0)),
                        "display_name": entry.get("name", ""),
                    })
        except Exception as e:
            logger.debug(f"Gamma profiles fetch failed: {e}")

        return traders

    async def get_wallet_positions(self, wallet: str) -> list[dict[str, Any]]:
        """Get current positions for a wallet address."""
        http = await self._get_http()
        positions: list[dict[str, Any]] = []

        try:
            # Try data API first
            resp = await http.get(
                f"{DATA_BASE}/positions",
                params={"user": wallet, "sizeThreshold": "0.1"},
            )
            if resp.status_code == 200:
                data = resp.json()
                raw_positions = data if isinstance(data, list) else data.get("positions", [])
                for p in raw_positions:
                    positions.append({
                        "market_id": p.get("market", p.get("conditionId", "")),
                        "token_id": p.get("tokenId", p.get("token_id", "")),
                        "title": p.get("title", p.get("question", "")),
                        "side": p.get("side", ""),
                        "size": float(p.get("size", p.get("amount", 0))),
                        "avg_price": float(p.get("avgPrice", p.get("averagePrice", 0))),
                        "current_value": float(p.get("currentValue", p.get("value", 0))),
                        "pnl": float(p.get("pnl", p.get("realizedPnl", 0))),
                    })
                return positions
        except Exception as e:
            logger.debug(f"Data API positions failed for {wallet[:10]}: {e}")

        # Fallback: try gamma API
        try:
            resp = await http.get(
                f"{GAMMA_BASE}/positions",
                params={"user": wallet},
            )
            if resp.status_code == 200:
                data = resp.json()
                for p in (data if isinstance(data, list) else []):
                    positions.append({
                        "market_id": p.get("market", ""),
                        "token_id": p.get("tokenId", ""),
                        "title": p.get("title", ""),
                        "side": "YES" if p.get("outcome", "") == "Yes" else "NO",
                        "size": float(p.get("size", 0)),
                        "avg_price": float(p.get("avgPrice", 0)),
                    })
        except Exception as e:
            logger.debug(f"Gamma positions failed for {wallet[:10]}: {e}")

        return positions

    # ── Copy Trading Logic ───────────────────────────────────────

    async def scan_for_new_trades(self) -> list[dict[str, Any]]:
        """Scan tracked wallets for new/changed positions.

        Returns list of copy trade signals to execute.
        """
        now = time.time()
        if now - self._last_scan_ts < self._scan_interval:
            return []

        self._last_scan_ts = now
        signals: list[dict[str, Any]] = []

        for wallet in self._tracked_wallets:
            try:
                await asyncio.sleep(0.5)  # Rate limit
                current_positions = await self.get_wallet_positions(wallet)
                previous = self._wallet_positions.get(wallet, {})

                for pos in current_positions:
                    market_id = pos.get("market_id", "")
                    if not market_id:
                        continue

                    prev_pos = previous.get(market_id)

                    # New position detected
                    if prev_pos is None and pos.get("size", 0) > 0:
                        signals.append({
                            "type": "new_position",
                            "wallet": wallet,
                            "market_id": market_id,
                            "token_id": pos.get("token_id", ""),
                            "title": pos.get("title", ""),
                            "side": pos.get("side", ""),
                            "size": pos.get("size", 0),
                            "avg_price": pos.get("avg_price", 0),
                            "timestamp": now,
                        })

                    # Position increased
                    elif prev_pos and pos.get("size", 0) > prev_pos.get("size", 0) * 1.1:
                        signals.append({
                            "type": "position_increase",
                            "wallet": wallet,
                            "market_id": market_id,
                            "token_id": pos.get("token_id", ""),
                            "title": pos.get("title", ""),
                            "side": pos.get("side", ""),
                            "old_size": prev_pos.get("size", 0),
                            "new_size": pos.get("size", 0),
                            "avg_price": pos.get("avg_price", 0),
                            "timestamp": now,
                        })

                # Update stored positions
                self._wallet_positions[wallet] = {
                    p["market_id"]: p for p in current_positions if p.get("market_id")
                }

            except Exception as e:
                logger.debug(f"Error scanning wallet {wallet[:10]}: {e}")
                continue

        if signals:
            logger.info(f"Copy trader: {len(signals)} new signals from {len(self._tracked_wallets)} wallets")

        return signals

    def _calculate_position_size(self, signal: dict[str, Any]) -> float:
        """Calculate how much USDC to deploy for a copy trade."""
        # Never exceed max portfolio percentage
        max_by_pct = self._bankroll * MAX_PORTFOLIO_PCT

        # Never exceed max position
        size = min(MAX_POSITION_USDC, max_by_pct)

        # Check total deployment limit
        remaining_capacity = (self._bankroll * MAX_TOTAL_DEPLOYED) - self._total_deployed
        if remaining_capacity <= MIN_POSITION_USDC:
            return 0.0

        size = min(size, remaining_capacity)

        # Minimum viable trade
        if size < MIN_POSITION_USDC:
            return 0.0

        return round(size, 2)

    async def execute_copy_trade(self, signal: dict[str, Any]) -> dict[str, Any] | None:
        """Execute a copy trade based on a detected signal.

        Returns trade result dict or None.
        """
        token_id = signal.get("token_id", "")
        if not token_id:
            logger.debug("No token_id in signal, skipping")
            return None

        # Check if we already have a copy of this market
        market_id = signal.get("market_id", "")
        if market_id in self._our_copies:
            logger.debug(f"Already have copy position in {market_id[:20]}")
            return None

        # Calculate position size
        usdc_amount = self._calculate_position_size(signal)
        if usdc_amount <= 0:
            logger.debug("Position size too small or max deployment reached")
            return None

        side = signal.get("side", "BUY")
        if side in ("YES", "yes"):
            side = "BUY"
        elif side in ("NO", "no"):
            side = "SELL"

        result: dict[str, Any] = {
            "signal": signal,
            "usdc_amount": usdc_amount,
            "side": side,
            "paper_mode": self._paper_mode,
        }

        if self._paper_mode:
            # Paper trade — log but don't execute
            result["status"] = "paper_filled"
            result["order_id"] = f"PAPER-POLY-{int(time.time())}"
            logger.info(
                "Copy trade (PAPER)",
                wallet=signal.get("wallet", "")[:10],
                title=signal.get("title", "")[:40],
                side=side,
                amount=usdc_amount,
            )
        else:
            # Live execution
            order_result = await self._trader.place_market_order(
                token_id=token_id,
                side=side,
                amount=usdc_amount,
            )
            if order_result:
                result["status"] = "filled"
                result["order_result"] = order_result
            else:
                result["status"] = "failed"
                return result

        # Track our copy
        self._our_copies[market_id] = {
            "token_id": token_id,
            "side": side,
            "amount": usdc_amount,
            "entry_time": time.time(),
            "signal": signal,
        }
        self._total_deployed += usdc_amount

        return result

    # ── Run Cycle ────────────────────────────────────────────────

    async def run_cycle(self) -> list[dict[str, Any]]:
        """Run one copy trading scan cycle.

        Returns list of executed trades (paper or live).
        """
        if not self._tracked_wallets:
            logger.debug("No tracked wallets configured")
            return []

        signals = await self.scan_for_new_trades()
        executed: list[dict[str, Any]] = []

        for signal in signals:
            result = await self.execute_copy_trade(signal)
            if result and result.get("status") in ("filled", "paper_filled"):
                executed.append(result)

        if executed:
            logger.info(f"Copy trader executed {len(executed)} trades")

        return executed

    def get_status(self) -> dict[str, Any]:
        """Get copy trader status for dashboard."""
        return {
            "tracked_wallets": len(self._tracked_wallets),
            "wallet_addresses": [w[:10] + "..." for w in self._tracked_wallets],
            "active_copies": len(self._our_copies),
            "total_deployed": round(self._total_deployed, 2),
            "bankroll": self._bankroll,
            "deployment_pct": round(self._total_deployed / self._bankroll * 100, 1) if self._bankroll > 0 else 0,
            "paper_mode": self._paper_mode,
        }


# ── Singleton ────────────────────────────────────────────────────

_copy_trader: CopyTrader | None = None


def get_copy_trader() -> CopyTrader:
    global _copy_trader
    if _copy_trader is None:
        _copy_trader = CopyTrader()
    return _copy_trader
