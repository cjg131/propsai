"""
Market Making Strategy for Kalshi Weather Markets.

Posts 2-sided quotes (bid YES + bid NO) on weather markets to capture the spread.
Kalshi maker fee is 0, so the full spread is profit when both sides fill.

Strategy:
1. Identify weather markets with reasonable liquidity
2. Calculate fair value using weather consensus
3. Post bid below fair value and ask above fair value
4. Manage inventory to stay delta-neutral
5. Cancel and reprice when fair value shifts

Risk Management:
- Max inventory per market (default: 50 contracts per side)
- Max total inventory across all markets
- Reprice if fair value moves > 2 cents
- Cancel all on circuit breaker trigger
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)
UTC = timezone.utc


class MarketMaker:
    """
    Two-sided market maker for Kalshi weather markets.
    Posts limit orders on both sides to capture the bid-ask spread.
    """

    def __init__(
        self,
        max_inventory_per_market: int = 50,
        max_total_inventory: int = 200,
        min_spread_to_quote: int = 3,     # Don't quote if spread < 3c (no profit margin)
        target_spread: int = 4,            # Our target spread in cents
        max_spread_to_quote: int = 20,     # Don't quote if spread > 20c (too risky)
        reprice_threshold: int = 2,        # Reprice if fair value moves > 2c
        max_position_delta: int = 30,      # Max net directional exposure per market
    ):
        self.max_inventory_per_market = max_inventory_per_market
        self.max_total_inventory = max_total_inventory
        self.min_spread_to_quote = min_spread_to_quote
        self.target_spread = target_spread
        self.max_spread_to_quote = max_spread_to_quote
        self.reprice_threshold = reprice_threshold
        self.max_position_delta = max_position_delta

        # Track our inventory: ticker -> {"yes": count, "no": count}
        self._inventory: dict[str, dict[str, int]] = {}
        # Track our resting orders: order_id -> order details
        self._resting_orders: dict[str, dict[str, Any]] = {}
        # Track fair values: ticker -> fair_value_cents
        self._fair_values: dict[str, int] = {}
        # Track total inventory
        self._total_yes_inventory: int = 0
        self._total_no_inventory: int = 0

    def calculate_fair_value(self, consensus_prob: float) -> int:
        """
        Convert our probability estimate to a fair value in cents.
        Clamp to 5-95 range (never quote at extremes).
        """
        fair_cents = int(round(consensus_prob * 100))
        return max(5, min(95, fair_cents))

    def calculate_quotes(
        self,
        ticker: str,
        fair_value_cents: int,
        current_yes_bid: int = 0,
        current_yes_ask: int = 0,
    ) -> dict[str, Any] | None:
        """
        Calculate bid/ask quotes for a market.

        Returns:
            {
                "ticker": str,
                "fair_value": int,
                "yes_bid_price": int,  # Our bid to buy YES (below fair value)
                "no_bid_price": int,   # Our bid to buy NO (equivalent to selling YES above fair value)
                "spread": int,
                "action": "new" | "reprice" | "cancel" | None
            }
        """
        # Calculate our spread around fair value
        half_spread = max(self.target_spread // 2, 1)

        # Our YES bid: we want to buy YES below fair value
        yes_bid_price = fair_value_cents - half_spread
        # Our NO bid: we want to buy NO, which is equivalent to selling YES above fair value
        # NO price = 100 - YES price, so if we want to sell YES at fair+half_spread:
        # no_bid_price = 100 - (fair_value + half_spread)
        no_bid_price = 100 - (fair_value_cents + half_spread)

        # Clamp to valid range (1-99)
        yes_bid_price = max(1, min(99, yes_bid_price))
        no_bid_price = max(1, min(99, no_bid_price))

        # Verify the spread is reasonable
        implied_spread = (100 - no_bid_price) - yes_bid_price  # YES ask equivalent - YES bid
        if implied_spread < self.min_spread_to_quote:
            return None  # Not enough spread to be profitable

        # Check inventory limits
        inv = self._inventory.get(ticker, {"yes": 0, "no": 0})
        net_delta = inv["yes"] - inv["no"]

        # Skew quotes based on inventory (market maker risk management)
        # If we're long YES (positive delta), make YES bid less aggressive (lower)
        # and NO bid more aggressive (lower = buying more NO to hedge)
        if abs(net_delta) > self.max_position_delta // 2:
            skew = 1 if net_delta > 0 else -1
            yes_bid_price -= skew  # If long YES, lower YES bid (less aggressive buying)
            no_bid_price += skew   # If long YES, raise NO bid (more aggressive buying NO to hedge)
            yes_bid_price = max(1, min(99, yes_bid_price))
            no_bid_price = max(1, min(99, no_bid_price))

        # Don't quote if we're at max inventory
        can_buy_yes = inv["yes"] < self.max_inventory_per_market
        can_buy_no = inv["no"] < self.max_inventory_per_market

        if not can_buy_yes and not can_buy_no:
            return None

        # Check if we need to reprice existing orders
        old_fair = self._fair_values.get(ticker)
        action = "new"
        if old_fair is not None:
            if abs(fair_value_cents - old_fair) >= self.reprice_threshold:
                action = "reprice"
            elif abs(fair_value_cents - old_fair) < 1:
                action = "hold"  # No change needed

        self._fair_values[ticker] = fair_value_cents

        return {
            "ticker": ticker,
            "fair_value": fair_value_cents,
            "yes_bid_price": yes_bid_price if can_buy_yes else None,
            "no_bid_price": no_bid_price if can_buy_no else None,
            "spread": implied_spread,
            "net_delta": net_delta,
            "action": action,
        }

    def record_fill(self, ticker: str, side: str, count: int, price_cents: int) -> None:
        """Record that one of our market making orders was filled."""
        if ticker not in self._inventory:
            self._inventory[ticker] = {"yes": 0, "no": 0}

        self._inventory[ticker][side] += count
        if side == "yes":
            self._total_yes_inventory += count
        else:
            self._total_no_inventory += count

        logger.info(
            "MM fill recorded",
            ticker=ticker,
            side=side,
            count=count,
            price=price_cents,
            inventory=self._inventory[ticker],
        )

    def record_settlement(self, ticker: str, result: str) -> dict[str, float]:
        """
        Record market settlement and calculate P&L.
        result: "yes" or "no"
        """
        inv = self._inventory.pop(ticker, {"yes": 0, "no": 0})
        self._fair_values.pop(ticker, None)

        # If result is "yes": YES contracts pay $1, NO contracts pay $0
        # If result is "no": YES contracts pay $0, NO contracts pay $1
        if result == "yes":
            pnl_yes = inv["yes"] * 1.0  # Each YES contract pays $1
            pnl_no = 0.0  # NO contracts expire worthless
        else:
            pnl_yes = 0.0  # YES contracts expire worthless
            pnl_no = inv["no"] * 1.0  # Each NO contract pays $1

        self._total_yes_inventory -= inv["yes"]
        self._total_no_inventory -= inv["no"]

        return {
            "ticker": ticker,
            "yes_inventory": inv["yes"],
            "no_inventory": inv["no"],
            "pnl_yes": pnl_yes,
            "pnl_no": pnl_no,
            "total_pnl": pnl_yes + pnl_no,
        }

    def get_markets_to_cancel(self) -> list[str]:
        """Get list of order IDs to cancel (e.g., on circuit breaker)."""
        return list(self._resting_orders.keys())

    def get_status(self) -> dict[str, Any]:
        """Get current market making status."""
        total_inv = sum(
            inv["yes"] + inv["no"]
            for inv in self._inventory.values()
        )

        return {
            "active_markets": len(self._inventory),
            "total_inventory": total_inv,
            "total_yes": self._total_yes_inventory,
            "total_no": self._total_no_inventory,
            "net_delta": self._total_yes_inventory - self._total_no_inventory,
            "resting_orders": len(self._resting_orders),
            "inventory_by_market": {
                ticker: dict(inv)
                for ticker, inv in self._inventory.items()
            },
        }


# Singleton
_mm: MarketMaker | None = None


def get_market_maker() -> MarketMaker:
    """Get or create the singleton market maker."""
    global _mm
    if _mm is None:
        _mm = MarketMaker()
    return _mm
