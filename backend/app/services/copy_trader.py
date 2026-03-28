"""
Polymarket Copy Trading Engine.

Monitors top traders' wallets on-chain and mirrors their trades.
Uses Polymarket's public APIs and on-chain data to detect new positions.

Strategy:
1. Track a curated list of profitable wallets (from leaderboard / manual)
2. Poll their positions for changes every 45 seconds
3. First scan is snapshot-only (no trades) to establish baseline
4. On subsequent scans, when a whale opens/increases a position, mirror it
5. Proportional sizing: match the whale's portfolio weight (capped at 25%)
6. Mirror exits: if a whale reduces a position by 50%+, sell our copy
7. Log everything for performance tracking
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

# Default wallets to track — env var overrides if set
DEFAULT_TRACKED_WALLETS: list[str] = [
    "0x96489abcb9f583d6835c8ef95ffc923d05a86825",  # anoin123 — $5.6M positions
    "0xda9ed03eb10640d411b35b3773c7af9aade1ac09",  # zhupercyclist — Three Arrows Capital
    "0x6765c1c000e3b99c7fffea752c561cb868f1fa0d",  # JanAMEX — $1.2M positions
]

# Position sizing — proportional with caps
MAX_POSITION_PCT = 0.25     # Max 25% of bankroll per trade ($62.50 on $250)
MIN_POSITION_USDC = 1.0     # Min $1 per trade (roughly 1 contract)
MAX_TOTAL_DEPLOYED = 0.80   # Max 80% of bankroll deployed at once


class CopyTrader:
    """Monitors whale wallets and mirrors their Polymarket trades."""

    def __init__(self, trader: PolymarketTrader | None = None) -> None:
        self._trader = trader or get_polymarket_trader()
        self._http: httpx.AsyncClient | None = None
        self._tracked_wallets: list[str] = self._load_tracked_wallets()
        self._wallet_positions: dict[str, dict[str, Any]] = {}  # wallet -> {market -> position}
        self._wallet_portfolio_value: dict[str, float] = {}  # wallet -> total portfolio USDC
        self._our_copies: dict[str, dict[str, Any]] = {}  # market -> our copy details
        self._last_scan_ts: float = 0.0
        self._scan_interval: float = 45.0  # seconds between scans
        self._bankroll: float = float(os.getenv("POLYMARKET_BANKROLL", "250"))
        self._total_deployed: float = 0.0
        self._paper_mode: bool = os.getenv("POLYMARKET_PAPER_MODE", "true").lower() == "true"
        self._first_scan_done: bool = False  # First scan is snapshot-only

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

    # ── Portfolio Estimation ──────────────────────────────────────

    def _estimate_wallet_portfolio(self, positions: list[dict[str, Any]]) -> float:
        """Estimate a whale's total portfolio value from visible positions.

        Uses current_value if available, otherwise size * avg_price.
        """
        total = 0.0
        for p in positions:
            value = p.get("current_value", 0)
            if value > 0:
                total += value
            else:
                # Fallback: size * avg_price as rough value
                size = p.get("size", 0)
                price = p.get("avg_price", 0)
                if size > 0 and price > 0:
                    total += size * price
        return total

    def _calculate_whale_position_weight(
        self, signal: dict[str, Any], wallet: str
    ) -> float:
        """Calculate what % of their portfolio a whale put into this position.

        Returns a float between 0.0 and 1.0.
        """
        whale_portfolio = self._wallet_portfolio_value.get(wallet, 0)
        if whale_portfolio <= 0:
            # Can't estimate — use a conservative default weight
            return 0.02  # Assume 2% if we can't determine

        position_size = signal.get("size", 0)
        position_price = signal.get("avg_price", 0)
        position_value = position_size * position_price if position_price > 0 else position_size

        if position_value <= 0:
            return 0.01

        weight = position_value / whale_portfolio
        return min(weight, 1.0)  # Cap at 100% (shouldn't happen but safety)

    # ── Copy Trading Logic ───────────────────────────────────────

    async def scan_for_new_trades(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Scan tracked wallets for new/changed/exited positions.

        Returns (entry_signals, exit_signals).
        """
        now = time.time()
        if now - self._last_scan_ts < self._scan_interval:
            return [], []

        self._last_scan_ts = now
        entry_signals: list[dict[str, Any]] = []
        exit_signals: list[dict[str, Any]] = []
        is_first_scan = not self._first_scan_done

        for wallet in self._tracked_wallets:
            try:
                await asyncio.sleep(0.5)  # Rate limit
                current_positions = await self.get_wallet_positions(wallet)
                previous = self._wallet_positions.get(wallet, {})

                # Estimate whale's total portfolio value
                portfolio_value = self._estimate_wallet_portfolio(current_positions)
                if portfolio_value > 0:
                    self._wallet_portfolio_value[wallet] = portfolio_value

                if is_first_scan:
                    # First scan: snapshot only, no trade signals
                    logger.info(
                        "Copy trader: initial snapshot",
                        wallet=wallet[:10],
                        positions=len(current_positions),
                        portfolio_est=round(portfolio_value, 2),
                    )
                else:
                    # Check for new or increased positions (entry signals)
                    for pos in current_positions:
                        market_id = pos.get("market_id", "")
                        if not market_id:
                            continue

                        prev_pos = previous.get(market_id)

                        # New position detected
                        if prev_pos is None and pos.get("size", 0) > 0:
                            entry_signals.append({
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

                        # Position increased by 10%+
                        elif prev_pos and pos.get("size", 0) > prev_pos.get("size", 0) * 1.1:
                            entry_signals.append({
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

                    # Check for exited/reduced positions (exit signals)
                    # Mirror proportionally: if whale cuts 30%, we cut 30%
                    current_market_ids = {
                        p["market_id"] for p in current_positions if p.get("market_id")
                    }
                    for market_id, prev_pos in previous.items():
                        if market_id not in current_market_ids:
                            # Position fully closed — exit 100%
                            exit_signals.append({
                                "type": "position_closed",
                                "wallet": wallet,
                                "market_id": market_id,
                                "token_id": prev_pos.get("token_id", ""),
                                "title": prev_pos.get("title", ""),
                                "old_size": prev_pos.get("size", 0),
                                "new_size": 0,
                                "reduction_pct": 100.0,
                                "timestamp": now,
                            })
                        else:
                            # Check if position was reduced at all (10%+ threshold)
                            current_pos = next(
                                (p for p in current_positions if p.get("market_id") == market_id),
                                None,
                            )
                            if current_pos and prev_pos.get("size", 0) > 0:
                                reduction = 1 - (
                                    current_pos.get("size", 0) / prev_pos.get("size", 0)
                                )
                                if reduction >= 0.10:  # 10% reduction threshold
                                    exit_signals.append({
                                        "type": "position_reduced",
                                        "wallet": wallet,
                                        "market_id": market_id,
                                        "token_id": current_pos.get("token_id", ""),
                                        "title": current_pos.get("title", ""),
                                        "old_size": prev_pos.get("size", 0),
                                        "new_size": current_pos.get("size", 0),
                                        "reduction_pct": round(reduction * 100, 1),
                                        "timestamp": now,
                                    })

                # Update stored positions
                self._wallet_positions[wallet] = {
                    p["market_id"]: p for p in current_positions if p.get("market_id")
                }

            except Exception as e:
                logger.debug(f"Error scanning wallet {wallet[:10]}: {e}")
                continue

        if is_first_scan:
            self._first_scan_done = True
            logger.info(
                f"Copy trader: first scan complete — snapshotted "
                f"{len(self._tracked_wallets)} wallets, trading starts next cycle"
            )

        if entry_signals:
            logger.info(
                f"Copy trader: {len(entry_signals)} entry signals from "
                f"{len(self._tracked_wallets)} wallets"
            )
        if exit_signals:
            logger.info(
                f"Copy trader: {len(exit_signals)} exit signals detected"
            )

        return entry_signals, exit_signals

    def _calculate_position_size(self, signal: dict[str, Any]) -> float:
        """Calculate how much USDC to deploy for a copy trade.

        Uses proportional sizing: match the whale's portfolio weight,
        capped at MAX_POSITION_PCT (25%) of our bankroll.
        """
        wallet = signal.get("wallet", "")

        # Calculate whale's position weight
        whale_weight = self._calculate_whale_position_weight(signal, wallet)

        # Apply whale's weight to our bankroll
        proportional_size = self._bankroll * whale_weight

        # Cap at max percentage of bankroll
        max_size = self._bankroll * MAX_POSITION_PCT
        size = min(proportional_size, max_size)

        # Check total deployment limit
        remaining_capacity = (self._bankroll * MAX_TOTAL_DEPLOYED) - self._total_deployed
        if remaining_capacity <= MIN_POSITION_USDC:
            logger.debug("Max total deployment reached")
            return 0.0

        size = min(size, remaining_capacity)

        # Minimum viable trade
        if size < MIN_POSITION_USDC:
            return 0.0

        logger.info(
            "Proportional sizing",
            whale_weight=f"{whale_weight:.1%}",
            proportional=f"${proportional_size:.2f}",
            capped=f"${size:.2f}",
            whale=wallet[:10],
        )

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

        # Calculate position size (proportional to whale's weight)
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
                logger.info(
                    "Copy trade (LIVE)",
                    wallet=signal.get("wallet", "")[:10],
                    title=signal.get("title", "")[:40],
                    side=side,
                    amount=usdc_amount,
                    order_id=order_result.get("id", ""),
                )
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

    async def execute_exit(self, signal: dict[str, Any]) -> dict[str, Any] | None:
        """Execute a proportional exit when a whale reduces/closes a position.

        If whale cuts 50%, we cut 50%. If whale exits fully, we exit fully.
        Returns exit result dict or None.
        """
        market_id = signal.get("market_id", "")
        if market_id not in self._our_copies:
            # We don't have a copy of this market
            return None

        our_copy = self._our_copies[market_id]
        token_id = our_copy.get("token_id", "")
        if not token_id:
            return None

        # Calculate proportional exit amount
        reduction_pct = signal.get("reduction_pct", 100.0) / 100.0  # e.g. 0.50 for 50%
        full_amount = our_copy.get("amount", 0)
        exit_amount = round(full_amount * reduction_pct, 2)

        if exit_amount < MIN_POSITION_USDC:
            # Too small to exit, skip
            return None

        # Flip side: if we bought, we sell to exit
        exit_side = "SELL" if our_copy.get("side") == "BUY" else "BUY"
        is_full_exit = reduction_pct >= 0.95  # Treat 95%+ as full exit

        result: dict[str, Any] = {
            "signal": signal,
            "market_id": market_id,
            "exit_side": exit_side,
            "exit_amount": exit_amount,
            "full_amount": full_amount,
            "reduction_pct": signal.get("reduction_pct", 100.0),
            "is_full_exit": is_full_exit,
            "paper_mode": self._paper_mode,
        }

        if self._paper_mode:
            result["status"] = "paper_exited"
            result["order_id"] = f"PAPER-EXIT-{int(time.time())}"
            logger.info(
                "Copy exit (PAPER)",
                title=signal.get("title", "")[:40],
                exit_side=exit_side,
                exit_amount=exit_amount,
                reduction=f"{signal.get('reduction_pct', 100)}%",
                reason=signal.get("type", ""),
            )
        else:
            # Live exit
            order_result = await self._trader.place_market_order(
                token_id=token_id,
                side=exit_side,
                amount=exit_amount,
            )
            if order_result:
                result["status"] = "exited"
                result["order_result"] = order_result
                logger.info(
                    "Copy exit (LIVE)",
                    title=signal.get("title", "")[:40],
                    exit_side=exit_side,
                    exit_amount=exit_amount,
                    reduction=f"{signal.get('reduction_pct', 100)}%",
                    order_id=order_result.get("id", ""),
                )
            else:
                result["status"] = "exit_failed"
                logger.warning(
                    "Copy exit FAILED",
                    title=signal.get("title", "")[:40],
                )
                return result

        # Update tracking
        self._total_deployed = max(0, self._total_deployed - exit_amount)
        if is_full_exit:
            del self._our_copies[market_id]
        else:
            # Reduce our tracked position amount
            self._our_copies[market_id]["amount"] = round(full_amount - exit_amount, 2)

        return result

    # ── Run Cycle ────────────────────────────────────────────────

    async def run_cycle(self) -> list[dict[str, Any]]:
        """Run one copy trading scan cycle.

        Returns list of executed trades (paper or live).
        """
        if not self._tracked_wallets:
            logger.debug("No tracked wallets configured")
            return []

        entry_signals, exit_signals = await self.scan_for_new_trades()
        executed: list[dict[str, Any]] = []

        # Process exits first (free up capital before new entries)
        for signal in exit_signals:
            result = await self.execute_exit(signal)
            if result and result.get("status") in ("exited", "paper_exited"):
                executed.append(result)

        # Process entries
        for signal in entry_signals:
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
            "deployment_pct": round(
                self._total_deployed / self._bankroll * 100, 1
            )
            if self._bankroll > 0
            else 0,
            "paper_mode": self._paper_mode,
            "first_scan_done": self._first_scan_done,
            "wallet_portfolios": {
                w[:10]: f"${v:,.0f}"
                for w, v in self._wallet_portfolio_value.items()
            },
        }


# ── Singleton ────────────────────────────────────────────────────

_copy_trader: CopyTrader | None = None


def get_copy_trader() -> CopyTrader:
    global _copy_trader
    if _copy_trader is None:
        _copy_trader = CopyTrader()
    return _copy_trader
