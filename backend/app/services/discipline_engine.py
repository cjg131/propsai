"""
Discipline Engine — Hard guardrails that prevent account destruction.

This module sits between the trading agent and the execution layer.
Every trade must pass through the discipline engine before being placed.
These limits CANNOT be overridden by strategy logic.
"""
from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

UTC = timezone.utc
DB_PATH = Path(__file__).parent.parent / "data" / "trading_engine.db"


class DisciplineEngine:
    """
    Enforces hard trading discipline rules:
    1. Rate limiting (max orders per hour/day)
    2. Circuit breakers (daily loss, weekly loss, max drawdown)
    3. Concentration limits (per market, per strategy)
    4. Trading hours enforcement
    5. Liquidity requirements
    6. Cooldown after losses
    """

    def __init__(
        self,
        bankroll: float = 500.0,
        # Rate limits
        max_orders_per_hour: int = 10,
        max_orders_per_day: int = 30,
        # Circuit breakers
        daily_loss_halt_pct: float = 0.05,      # Stop at 5% daily loss
        weekly_loss_halt_pct: float = 0.10,      # Stop at 10% weekly loss
        max_drawdown_halt_pct: float = 0.20,     # Stop at 20% drawdown from peak
        # Concentration
        max_single_market_pct: float = 0.15,     # Max 15% in one market
        max_single_strategy_pct: float = 0.40,   # Max 40% in one strategy
        max_total_deployed_pct: float = 0.60,    # Max 60% deployed at once
        cash_reserve_pct: float = 0.20,          # Always keep 20% cash
        # Sizing
        use_maker_orders: bool = True,           # Default to limit orders (maker = no fee)
        min_orderbook_depth: int = 50,           # Min contracts on other side
        min_spread_cents: int = 2,               # Reject if spread < 2c (no room to exit)
        max_spread_cents: int = 15,              # Reject if spread > 15c (illiquid)
    ):
        self.bankroll = bankroll
        self.peak_bankroll = bankroll

        # Rate limits
        self.max_orders_per_hour = max_orders_per_hour
        self.max_orders_per_day = max_orders_per_day
        self._order_timestamps: list[float] = []

        # Circuit breakers
        self.daily_loss_halt_pct = daily_loss_halt_pct
        self.weekly_loss_halt_pct = weekly_loss_halt_pct
        self.max_drawdown_halt_pct = max_drawdown_halt_pct
        self._halted_until: float = 0.0
        self._halt_reason: str = ""
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._current_day: str = ""
        self._current_week: str = ""

        # Concentration
        self.max_single_market_pct = max_single_market_pct
        self.max_single_strategy_pct = max_single_strategy_pct
        self.max_total_deployed_pct = max_total_deployed_pct
        self.cash_reserve_pct = cash_reserve_pct

        # Execution
        self.use_maker_orders = use_maker_orders
        self.min_orderbook_depth = min_orderbook_depth
        self.min_spread_cents = min_spread_cents
        self.max_spread_cents = max_spread_cents

        # Trading hours (ET) — maps strategy to allowed hours
        self._trading_hours: dict[str, tuple[int, int]] = {
            "weather": (5, 22),    # 5 AM - 10 PM ET
            "finance": (9, 16),    # Market hours
            "econ": (8, 17),       # Business hours
            "crypto": (0, 24),     # 24/7 but with caution
            "sports": (10, 23),    # Game day hours
            "nba_props": (10, 23),
        }

        # Tracking
        self._positions: dict[str, float] = {}  # ticker -> exposure
        self._strategy_exposure: dict[str, float] = {}  # strategy -> exposure

        self._init_db()

    def _init_db(self) -> None:
        """Ensure discipline tracking tables exist."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS discipline_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL,
                strategy TEXT DEFAULT '',
                ticker TEXT DEFAULT ''
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS calibration_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                predicted_prob REAL NOT NULL,
                actual_outcome INTEGER NOT NULL,
                edge_claimed REAL NOT NULL,
                pnl REAL NOT NULL,
                ticker TEXT DEFAULT '',
                signal_source TEXT DEFAULT ''
            )
        """)
        conn.commit()
        conn.close()

    def _log_event(self, event_type: str, details: str, strategy: str = "", ticker: str = "") -> None:
        """Log a discipline event."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            c = conn.cursor()
            c.execute(
                "INSERT INTO discipline_log (timestamp, event_type, details, strategy, ticker) VALUES (?, ?, ?, ?, ?)",
                (datetime.now(UTC).isoformat(), event_type, details, strategy, ticker),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
        logger.info(f"DISCIPLINE [{event_type}]: {details}", strategy=strategy)

    # ── Rate Limiting ────────────────────────────────────────────

    def check_rate_limit(self) -> tuple[bool, str]:
        """Check if we're within rate limits."""
        now = time.time()
        # Clean old timestamps
        self._order_timestamps = [ts for ts in self._order_timestamps if ts > now - 86400]

        # Check hourly limit
        hour_ago = now - 3600
        hourly_count = sum(1 for ts in self._order_timestamps if ts > hour_ago)
        if hourly_count >= self.max_orders_per_hour:
            reason = f"Rate limit: {hourly_count}/{self.max_orders_per_hour} orders in last hour"
            self._log_event("rate_limit", reason)
            return False, reason

        # Check daily limit
        day_start = now - 86400
        daily_count = sum(1 for ts in self._order_timestamps if ts > day_start)
        if daily_count >= self.max_orders_per_day:
            reason = f"Rate limit: {daily_count}/{self.max_orders_per_day} orders in last 24h"
            self._log_event("rate_limit", reason)
            return False, reason

        return True, "OK"

    def record_order(self) -> None:
        """Record that an order was placed."""
        self._order_timestamps.append(time.time())

    # ── Circuit Breakers ─────────────────────────────────────────

    def update_pnl(self, pnl_change: float) -> None:
        """Update running P&L trackers."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        week = datetime.now(UTC).strftime("%Y-W%W")

        if today != self._current_day:
            self._daily_pnl = 0.0
            self._current_day = today
        if week != self._current_week:
            self._weekly_pnl = 0.0
            self._current_week = week

        self._daily_pnl += pnl_change
        self._weekly_pnl += pnl_change

    def update_bankroll(self, new_bankroll: float) -> None:
        """Update bankroll and track peak for drawdown calculation."""
        self.bankroll = new_bankroll
        if new_bankroll > self.peak_bankroll:
            self.peak_bankroll = new_bankroll

    def check_circuit_breakers(self) -> tuple[bool, str]:
        """Check all circuit breakers. Returns (can_trade, reason)."""
        now = time.time()

        # Check if we're in a halt period
        if now < self._halted_until:
            mins_left = int((self._halted_until - now) / 60)
            return False, f"Trading halted: {self._halt_reason} ({mins_left}m remaining)"

        # Daily loss check
        daily_limit = -(self.bankroll * self.daily_loss_halt_pct)
        if self._daily_pnl <= daily_limit:
            self._halted_until = now + 86400  # Halt rest of day (24h)
            self._halt_reason = f"Daily loss limit hit (${self._daily_pnl:.2f})"
            self._log_event("circuit_breaker", self._halt_reason)
            return False, self._halt_reason

        # Weekly loss check
        weekly_limit = -(self.bankroll * self.weekly_loss_halt_pct)
        if self._weekly_pnl <= weekly_limit:
            self._halted_until = now + (7 * 86400)  # Halt rest of week
            self._halt_reason = f"Weekly loss limit hit (${self._weekly_pnl:.2f})"
            self._log_event("circuit_breaker", self._halt_reason)
            return False, self._halt_reason

        # Max drawdown check
        if self.peak_bankroll > 0:
            drawdown = (self.peak_bankroll - self.bankroll) / self.peak_bankroll
            if drawdown >= self.max_drawdown_halt_pct:
                self._halted_until = now + (30 * 86400)  # Halt for 30 days — manual review needed
                self._halt_reason = f"Max drawdown {drawdown:.1%} hit (peak ${self.peak_bankroll:.2f} → ${self.bankroll:.2f})"
                self._log_event("circuit_breaker", self._halt_reason)
                return False, self._halt_reason

        return True, "OK"

    # ── Trading Hours ────────────────────────────────────────────

    def check_trading_hours(self, strategy: str) -> tuple[bool, str]:
        """Check if current time is within allowed trading hours for this strategy."""
        try:
            from zoneinfo import ZoneInfo
            et_now = datetime.now(ZoneInfo("America/New_York"))
            current_hour = et_now.hour
        except Exception:
            # Fallback — assume ET is UTC-5
            utc_now = datetime.now(UTC)
            current_hour = (utc_now.hour - 5) % 24

        allowed = self._trading_hours.get(strategy, (0, 24))
        start_hour, end_hour = allowed

        if not (start_hour <= current_hour < end_hour):
            reason = f"Outside trading hours for {strategy}: {current_hour}h ET (allowed {start_hour}-{end_hour})"
            return False, reason

        return True, "OK"

    # ── Concentration Limits ─────────────────────────────────────

    def check_concentration(
        self,
        ticker: str,
        strategy: str,
        proposed_cost: float,
    ) -> tuple[bool, str]:
        """Check concentration limits before a trade."""
        if self.bankroll <= 0:
            return False, "No bankroll available"

        # Cash reserve check
        total_deployed = sum(self._positions.values())
        available = self.bankroll - total_deployed
        min_cash = self.bankroll * self.cash_reserve_pct
        if available - proposed_cost < min_cash:
            return False, f"Would breach cash reserve: ${available - proposed_cost:.2f} < ${min_cash:.2f} minimum"

        # Total deployment check
        if (total_deployed + proposed_cost) / self.bankroll > self.max_total_deployed_pct:
            return False, f"Total deployment would reach {(total_deployed + proposed_cost) / self.bankroll:.1%} (max {self.max_total_deployed_pct:.0%})"

        # Single market check
        current_market = self._positions.get(ticker, 0.0)
        if (current_market + proposed_cost) / self.bankroll > self.max_single_market_pct:
            return False, f"Market {ticker} concentration would reach {(current_market + proposed_cost) / self.bankroll:.1%} (max {self.max_single_market_pct:.0%})"

        # Strategy check
        current_strategy = self._strategy_exposure.get(strategy, 0.0)
        if (current_strategy + proposed_cost) / self.bankroll > self.max_single_strategy_pct:
            return False, f"Strategy {strategy} concentration would reach {(current_strategy + proposed_cost) / self.bankroll:.1%} (max {self.max_single_strategy_pct:.0%})"

        return True, "OK"

    def record_position(self, ticker: str, strategy: str, cost: float) -> None:
        """Record a new position for tracking."""
        self._positions[ticker] = self._positions.get(ticker, 0.0) + cost
        self._strategy_exposure[strategy] = self._strategy_exposure.get(strategy, 0.0) + cost

    def remove_position(self, ticker: str, strategy: str, cost: float) -> None:
        """Remove a position (settlement or exit)."""
        self._positions[ticker] = max(0, self._positions.get(ticker, 0.0) - cost)
        self._strategy_exposure[strategy] = max(0, self._strategy_exposure.get(strategy, 0.0) - cost)
        if self._positions.get(ticker, 0) <= 0:
            self._positions.pop(ticker, None)
        if self._strategy_exposure.get(strategy, 0) <= 0:
            self._strategy_exposure.pop(strategy, None)

    # ── Liquidity Check ──────────────────────────────────────────

    def check_liquidity(
        self,
        orderbook: dict[str, Any],
        side: str,
    ) -> tuple[bool, str]:
        """Check if the market has sufficient liquidity for our trade."""
        yes_bid = orderbook.get("yes_bid", 0) or 0
        yes_ask = orderbook.get("yes_ask", 0) or 0
        no_bid = orderbook.get("no_bid", 0) or 0
        no_ask = orderbook.get("no_ask", 0) or 0

        # Calculate spread
        if side == "yes":
            spread = yes_ask - yes_bid if yes_ask > 0 and yes_bid > 0 else 99
        else:
            spread = no_ask - no_bid if no_ask > 0 and no_bid > 0 else 99

        if spread < self.min_spread_cents:
            return False, f"Spread {spread}c too tight (min {self.min_spread_cents}c) — likely stale quotes"

        if spread > self.max_spread_cents:
            return False, f"Spread {spread}c too wide (max {self.max_spread_cents}c) — illiquid market"

        # Check depth (if available)
        depth = orderbook.get("depth", 0) or 0
        if depth > 0 and depth < self.min_orderbook_depth:
            return False, f"Order book depth {depth} below minimum {self.min_orderbook_depth}"

        return True, "OK"

    # ── Master Gate ──────────────────────────────────────────────

    def approve_trade(
        self,
        strategy: str,
        ticker: str,
        side: str,
        cost: float,
        orderbook: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """
        Master approval gate. Every trade must pass ALL checks.
        Returns (approved, reason).
        """
        # 1. Circuit breakers
        ok, reason = self.check_circuit_breakers()
        if not ok:
            return False, reason

        # 2. Rate limits
        ok, reason = self.check_rate_limit()
        if not ok:
            return False, reason

        # 3. Trading hours
        ok, reason = self.check_trading_hours(strategy)
        if not ok:
            return False, reason

        # 4. Concentration
        ok, reason = self.check_concentration(ticker, strategy, cost)
        if not ok:
            return False, reason

        # 5. Liquidity (if orderbook data provided)
        if orderbook:
            ok, reason = self.check_liquidity(orderbook, side)
            if not ok:
                return False, reason

        return True, "OK"

    # ── Calibration Tracking ─────────────────────────────────────

    def record_prediction(
        self,
        strategy: str,
        predicted_prob: float,
        actual_outcome: bool,
        edge_claimed: float,
        pnl: float,
        ticker: str = "",
        signal_source: str = "",
    ) -> None:
        """Record a prediction outcome for calibration tracking."""
        try:
            conn = sqlite3.connect(str(DB_PATH))
            c = conn.cursor()
            c.execute(
                """INSERT INTO calibration_log
                (timestamp, strategy, predicted_prob, actual_outcome, edge_claimed, pnl, ticker, signal_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(UTC).isoformat(),
                    strategy,
                    predicted_prob,
                    1 if actual_outcome else 0,
                    edge_claimed,
                    pnl,
                    ticker,
                    signal_source,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to record calibration", error=str(e))

    def get_calibration_report(self, strategy: str = "", min_trades: int = 20) -> dict[str, Any]:
        """
        Generate calibration report: how well do our predicted probabilities
        match actual outcomes?

        Groups predictions into buckets (0-10%, 10-20%, ..., 90-100%)
        and compares predicted probability to actual win rate.
        """
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        query = "SELECT predicted_prob, actual_outcome, edge_claimed, pnl FROM calibration_log"
        params: list[str] = []
        if strategy:
            query += " WHERE strategy = ?"
            params.append(strategy)

        c.execute(query, params)
        rows = c.fetchall()
        conn.close()

        if len(rows) < min_trades:
            return {"error": f"Need {min_trades} trades for calibration, only have {len(rows)}"}

        # Bucket into deciles
        buckets: dict[str, list[tuple[float, int]]] = defaultdict(list)
        for row in rows:
            prob = row["predicted_prob"]
            bucket = f"{int(prob * 10) * 10}-{int(prob * 10) * 10 + 10}%"
            buckets[bucket].append((prob, row["actual_outcome"]))

        report: dict[str, Any] = {
            "total_predictions": len(rows),
            "total_pnl": sum(row["pnl"] for row in rows),
            "avg_edge_claimed": sum(row["edge_claimed"] for row in rows) / len(rows),
            "overall_win_rate": sum(row["actual_outcome"] for row in rows) / len(rows),
            "buckets": {},
        }

        for bucket_name, entries in sorted(buckets.items()):
            avg_predicted = sum(e[0] for e in entries) / len(entries)
            actual_rate = sum(e[1] for e in entries) / len(entries)
            report["buckets"][bucket_name] = {
                "count": len(entries),
                "avg_predicted": round(avg_predicted, 3),
                "actual_rate": round(actual_rate, 3),
                "calibration_error": round(abs(avg_predicted - actual_rate), 3),
            }

        # Overall calibration score (lower = better, 0 = perfect)
        total_error = sum(
            b["calibration_error"] * b["count"]
            for b in report["buckets"].values()
        )
        report["calibration_score"] = round(total_error / len(rows), 4)

        return report

    def get_status(self) -> dict[str, Any]:
        """Return current discipline engine status."""
        now = time.time()
        total_deployed = sum(self._positions.values())

        return {
            "bankroll": self.bankroll,
            "peak_bankroll": self.peak_bankroll,
            "drawdown_pct": round((self.peak_bankroll - self.bankroll) / self.peak_bankroll, 4) if self.peak_bankroll > 0 else 0,
            "daily_pnl": round(self._daily_pnl, 2),
            "weekly_pnl": round(self._weekly_pnl, 2),
            "total_deployed": round(total_deployed, 2),
            "deployment_pct": round(total_deployed / self.bankroll, 4) if self.bankroll > 0 else 0,
            "cash_available": round(self.bankroll - total_deployed, 2),
            "is_halted": now < self._halted_until,
            "halt_reason": self._halt_reason if now < self._halted_until else "",
            "orders_last_hour": sum(1 for ts in self._order_timestamps if ts > now - 3600),
            "orders_last_day": sum(1 for ts in self._order_timestamps if ts > now - 86400),
            "positions_count": len(self._positions),
            "strategy_exposure": dict(self._strategy_exposure),
        }


# Singleton
_discipline: DisciplineEngine | None = None


def get_discipline_engine() -> DisciplineEngine:
    """Get or create the singleton discipline engine."""
    global _discipline
    if _discipline is None:
        import os
        bankroll = float(os.environ.get("INITIAL_BANKROLL", "500"))
        _discipline = DisciplineEngine(bankroll=bankroll)
    return _discipline
