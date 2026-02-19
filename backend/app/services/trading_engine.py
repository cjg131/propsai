"""
Trading Engine for Kalshi autonomous agent.
Handles order placement, bankroll management, risk limits, and paper trading.
"""
from __future__ import annotations

import math
import os
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "trading_engine.db"


def _kalshi_maker_fee(count: int, price_cents: int) -> float:
    """Calculate Kalshi maker fee for limit orders.
    Fee = ceil(0.0175 × count × P × (1-P)) where P = price/100
    """
    p = price_cents / 100.0
    raw = 0.0175 * count * p * (1 - p)
    return math.ceil(raw * 100) / 100  # round up to nearest cent


def _kalshi_taker_fee(count: int, price_cents: int) -> float:
    """Calculate Kalshi taker fee for market orders.
    Fee = ceil(0.07 × count × P × (1-P)) where P = price/100
    """
    p = price_cents / 100.0
    raw = 0.07 * count * p * (1 - p)
    return math.ceil(raw * 100) / 100


class TradingEngine:
    """
    Core trading engine with paper trading mode, bankroll management,
    risk limits, and trade logging.
    """

    def __init__(
        self,
        bankroll: float = 2000.0,
        paper_mode: bool = True,
        daily_loss_limit: float | None = None,
        max_bet_size: float | None = None,
        max_bet_pct: float = 0.04,  # 4% of bankroll per bet
        max_position_per_ticker: float | None = None,
        max_total_exposure_pct: float = 1.00,  # Allow full bankroll deployment
        max_strategy_exposure_pct: float = 0.40,  # Soft cap: 40% of bankroll per strategy
        max_strategy_cycle_pct: float = 0.25,  # Max 25% of bankroll deployed per strategy per cycle
    ):
        self.bankroll = bankroll
        self.paper_mode = paper_mode
        self.max_bet_pct = max_bet_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_strategy_exposure_pct = max_strategy_exposure_pct
        self.max_strategy_cycle_pct = max_strategy_cycle_pct
        # Scale risk limits from bankroll (override with explicit values if provided)
        self.daily_loss_limit = daily_loss_limit if daily_loss_limit is not None else bankroll * 0.20
        self.max_bet_size = max_bet_size if max_bet_size is not None else bankroll * 0.04
        self.max_position_per_ticker = max_position_per_ticker if max_position_per_ticker is not None else bankroll * 0.10
        self.kill_switch = False
        # Track per-strategy deployment within current cycle
        self._cycle_deployed: dict[str, float] = {}

        # Strategy-level toggles
        self.strategy_enabled = {
            "weather": True,
            "sports": True,
            "crypto": True,
            "nba_props": True,
            "finance": True,
            "econ": True,
        }

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database for trade logging."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                ticker TEXT NOT NULL,
                market_title TEXT DEFAULT '',
                side TEXT NOT NULL,
                action TEXT NOT NULL,
                count INTEGER NOT NULL,
                price_cents INTEGER NOT NULL,
                cost REAL NOT NULL,
                fee REAL NOT NULL,
                order_type TEXT DEFAULT 'limit',
                paper_mode INTEGER NOT NULL DEFAULT 1,
                order_id TEXT DEFAULT '',
                status TEXT DEFAULT 'pending',
                our_prob REAL DEFAULT 0,
                kalshi_prob REAL DEFAULT 0,
                edge REAL DEFAULT 0,
                signal_source TEXT DEFAULT '',
                result TEXT DEFAULT '',
                pnl REAL DEFAULT 0,
                settled_at TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_pnl (
                date TEXT NOT NULL,
                strategy TEXT NOT NULL,
                gross_pnl REAL DEFAULT 0,
                fees REAL DEFAULT 0,
                net_pnl REAL DEFAULT 0,
                trades_count INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                PRIMARY KEY (date, strategy)
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                ticker TEXT NOT NULL,
                market_title TEXT DEFAULT '',
                side TEXT NOT NULL,
                our_prob REAL NOT NULL,
                kalshi_prob REAL NOT NULL,
                edge REAL NOT NULL,
                confidence REAL DEFAULT 0,
                recommended_size INTEGER DEFAULT 0,
                recommended_price INTEGER DEFAULT 0,
                acted_on INTEGER DEFAULT 0,
                trade_id TEXT DEFAULT '',
                signal_source TEXT DEFAULT '',
                details TEXT DEFAULT ''
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS agent_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                strategy TEXT DEFAULT '',
                message TEXT NOT NULL,
                details TEXT DEFAULT ''
            )
        """)

        conn.commit()
        conn.close()

    def log_event(self, level: str, message: str, strategy: str = "", details: str = "") -> None:
        """Log an agent event to the database."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "INSERT INTO agent_log (timestamp, level, strategy, message, details) VALUES (?, ?, ?, ?, ?)",
            (datetime.now(UTC).isoformat(), level, strategy, message, details),
        )
        conn.commit()
        conn.close()

    def _get_all_realized_pnl(self) -> float:
        """Sum of all realized P&L: settled trades + exited (sell) trades."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("""
            SELECT COALESCE(SUM(pnl), 0) FROM trades
            WHERE status = 'settled' OR action = 'sell'
        """)
        result = c.fetchone()[0]
        conn.close()
        return result

    def get_effective_bankroll(self) -> float:
        """Real available capital = starting bankroll + all realized P&L.
        
        This is the true bankroll after wins and losses are accounted for.
        All risk calculations should use this, not self.bankroll.
        """
        return max(0.0, self.bankroll + self._get_all_realized_pnl())

    def get_today_pnl(self, strategy: str | None = None) -> float:
        """Get today's P&L, optionally filtered by strategy."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        if strategy:
            c.execute(
                "SELECT COALESCE(SUM(net_pnl), 0) FROM daily_pnl WHERE date = ? AND strategy = ?",
                (today, strategy),
            )
        else:
            c.execute(
                "SELECT COALESCE(SUM(net_pnl), 0) FROM daily_pnl WHERE date = ?",
                (today,),
            )
        result = c.fetchone()[0]
        conn.close()
        return result

    def get_today_trade_count(self, strategy: str | None = None) -> int:
        """Get today's trade count."""
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        if strategy:
            c.execute(
                "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ? AND strategy = ?",
                (f"{today}%", strategy),
            )
        else:
            c.execute(
                "SELECT COUNT(*) FROM trades WHERE timestamp LIKE ?",
                (f"{today}%",),
            )
        result = c.fetchone()[0]
        conn.close()
        return result

    def get_total_exposure(self, strategy: str | None = None) -> float:
        """Get total capital currently deployed in open (filled, unsettled) positions.
        
        Nets buy cost against sell proceeds so that exited positions free up capital.
        buy cost is positive, sell cost is negative (proceeds), so SUM(cost) gives net.
        Excludes sell/exit trades from the exposure count since they reduce exposure.
        """
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        if strategy:
            c.execute(
                """SELECT COALESCE(SUM(cost + fee), 0) FROM trades
                   WHERE status = 'filled' AND action = 'buy' AND strategy = ?""",
                (strategy,),
            )
        else:
            c.execute(
                "SELECT COALESCE(SUM(cost + fee), 0) FROM trades WHERE status = 'filled' AND action = 'buy'"
            )
        result = c.fetchone()[0]
        conn.close()
        return max(0, result)

    def has_open_position(self, ticker: str) -> bool:
        """Check if we already have an open (unsettled) position on this ticker."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("""
            SELECT
                SUM(CASE WHEN action = 'buy' THEN count ELSE 0 END)
              - SUM(CASE WHEN action = 'sell' THEN count ELSE 0 END) as net
            FROM trades
            WHERE status = 'filled' AND ticker = ?
        """, (ticker,))
        row = c.fetchone()
        conn.close()
        return row is not None and (row[0] or 0) > 0

    def get_ticker_exposure(self, ticker: str) -> float:
        """Get net capital deployed on a specific ticker (buys minus sell proceeds)."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "SELECT COALESCE(SUM(cost + fee), 0) FROM trades WHERE status = 'filled' AND ticker = ?",
            (ticker,),
        )
        result = c.fetchone()[0]
        conn.close()
        return max(0, result)

    def get_remaining_capital(self, strategy: str = "") -> float:
        """Get remaining deployable capital, respecting per-strategy caps.
        
        Uses effective_bankroll (starting bankroll + realized P&L) so that
        losses reduce available capital in real time.
        
        Limits:
          1. Global: total exposure <= effective_bankroll * max_total_exposure_pct
          2. Per-strategy total: strategy exposure <= effective_bankroll * max_strategy_exposure_pct
          3. Per-strategy cycle: deployment this cycle <= effective_bankroll * max_strategy_cycle_pct
        Returns the minimum of all applicable limits.
        """
        effective = self.get_effective_bankroll()
        total_exposure = self.get_total_exposure()
        max_deployable = effective * self.max_total_exposure_pct
        remaining_global = max(0, max_deployable - total_exposure)
        
        if not strategy:
            return remaining_global
        
        # Per-strategy total exposure cap (soft)
        strategy_exposure = self.get_total_exposure(strategy=strategy)
        max_strategy = effective * self.max_strategy_exposure_pct
        remaining_strategy = max(0, max_strategy - strategy_exposure)
        
        # Per-strategy cycle cap (prevents one cycle from deploying too much)
        cycle_deployed = self._cycle_deployed.get(strategy, 0.0)
        max_cycle = effective * self.max_strategy_cycle_pct
        remaining_cycle = max(0, max_cycle - cycle_deployed)
        
        return min(remaining_global, remaining_strategy, remaining_cycle)
    
    def start_cycle(self, strategy: str) -> None:
        """Call at the start of each strategy cycle to reset cycle deployment tracking."""
        self._cycle_deployed[strategy] = 0.0
    
    def _record_cycle_deployment(self, strategy: str, amount: float) -> None:
        """Track how much capital was deployed in the current cycle for a strategy."""
        self._cycle_deployed[strategy] = self._cycle_deployed.get(strategy, 0.0) + amount
    
    def _get_trade_count(self) -> int:
        """Get total number of buy trades placed."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM trades WHERE action = 'buy'")
        result = c.fetchone()[0]
        conn.close()
        return result

    def check_risk_limits(self, strategy: str, cost: float, ticker: str = "") -> tuple[bool, str]:
        """
        Check if a trade passes all risk limits.
        Uses effective_bankroll (starting bankroll + realized P&L) for all calculations.
        Returns (allowed, reason).
        """
        if self.kill_switch:
            return False, "Kill switch is active"

        if not self.strategy_enabled.get(strategy, False):
            return False, f"Strategy '{strategy}' is disabled"

        effective = self.get_effective_bankroll()
        if effective <= 0:
            return False, f"Effective bankroll is ${effective:.2f} — no capital available"

        # Check percentage of effective bankroll per bet
        if cost > effective * self.max_bet_pct:
            return False, f"Bet size ${cost:.2f} exceeds {self.max_bet_pct*100:.0f}% of effective bankroll (${effective:.2f})"

        # Check total exposure vs effective bankroll
        total_exposure = self.get_total_exposure()
        if total_exposure >= effective:
            return False, f"Over-deployed: ${total_exposure:.2f} deployed vs ${effective:.2f} effective bankroll"
        max_deployable = effective * self.max_total_exposure_pct
        if total_exposure + cost > max_deployable:
            return False, f"Total exposure ${total_exposure + cost:.2f} would exceed {self.max_total_exposure_pct*100:.0f}% of effective bankroll (${max_deployable:.2f})"

        # Check per-ticker position limit (scaled to effective bankroll)
        max_per_ticker = effective * 0.10
        if ticker:
            ticker_exposure = self.get_ticker_exposure(ticker)
            if ticker_exposure + cost > max_per_ticker:
                return False, f"Ticker {ticker} exposure ${ticker_exposure + cost:.2f} would exceed max ${max_per_ticker:.2f}"

        # Check per-strategy total exposure cap
        strategy_exposure = self.get_total_exposure(strategy=strategy)
        max_strategy = effective * self.max_strategy_exposure_pct
        if strategy_exposure + cost > max_strategy:
            return False, f"Strategy '{strategy}' exposure ${strategy_exposure + cost:.2f} would exceed {self.max_strategy_exposure_pct*100:.0f}% of effective bankroll (${max_strategy:.2f})"

        # Check per-strategy cycle deployment cap
        cycle_deployed = self._cycle_deployed.get(strategy, 0.0)
        max_cycle = effective * self.max_strategy_cycle_pct
        if cycle_deployed + cost > max_cycle:
            return False, f"Strategy '{strategy}' cycle cap: ${cycle_deployed + cost:.2f} would exceed ${max_cycle:.2f}/cycle"

        # Check daily loss limit (dynamic: 20% of effective bankroll)
        today_pnl = self.get_today_pnl()
        dynamic_loss_limit = effective * 0.20
        if today_pnl < -dynamic_loss_limit:
            return False, f"Daily loss limit reached: ${today_pnl:.2f} (limit: -${dynamic_loss_limit:.2f})"

        return True, "OK"

    def calculate_position_size(
        self,
        strategy: str,
        edge: float,
        price_cents: int,
        confidence: float = 1.0,
        ticker: str = "",
    ) -> int:
        """
        Calculate position size using quarter-Kelly criterion,
        constrained by remaining capital and per-ticker limits.
        Returns number of contracts.
        """
        if edge <= 0 or price_cents <= 0 or price_cents >= 100:
            return 0

        p = price_cents / 100.0
        kelly_fraction = edge / (1 - p)
        quarter_kelly = kelly_fraction * 0.25 * confidence

        remaining = self.get_remaining_capital(strategy=strategy)
        effective = self.get_effective_bankroll()

        # Cap by remaining capital, Kelly, and effective bankroll percentage
        max_dollars = min(
            remaining,
            effective * quarter_kelly,
            effective * self.max_bet_pct,
        )

        # Also cap by per-ticker limit (scaled to effective bankroll)
        if ticker:
            ticker_exposure = self.get_ticker_exposure(ticker)
            max_per_ticker = effective * 0.10
            ticker_remaining = max(0, max_per_ticker - ticker_exposure)
            max_dollars = min(max_dollars, ticker_remaining)

        if max_dollars <= 0:
            return 0

        cost_per_contract = price_cents / 100.0
        if cost_per_contract <= 0:
            return 0

        count = int(max_dollars / cost_per_contract)
        return max(count, 0)

    def record_signal(
        self,
        strategy: str,
        ticker: str,
        side: str,
        our_prob: float,
        kalshi_prob: float,
        market_title: str = "",
        confidence: float = 0.0,
        recommended_size: int = 0,
        recommended_price: int = 0,
        signal_source: str = "",
        details: str = "",
    ) -> str:
        """Record a trading signal. Returns signal ID."""
        signal_id = str(uuid.uuid4())[:12]
        edge = our_prob - kalshi_prob if side == "no" else kalshi_prob - our_prob
        # For "no" side: edge = (1-our_prob) - (1-kalshi_prob) = kalshi_prob - our_prob...
        # Actually: if we buy NO, edge = our_prob_no - kalshi_prob_no
        # Let's keep it simple: edge is always positive when we have an advantage
        edge = abs(our_prob - kalshi_prob)

        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            """INSERT INTO signals
            (id, timestamp, strategy, ticker, market_title, side, our_prob, kalshi_prob,
             edge, confidence, recommended_size, recommended_price, signal_source, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                signal_id,
                datetime.now(UTC).isoformat(),
                strategy,
                ticker,
                market_title,
                side,
                our_prob,
                kalshi_prob,
                edge,
                confidence,
                recommended_size,
                recommended_price,
                signal_source,
                details,
            ),
        )
        conn.commit()
        conn.close()

        self.log_event(
            "signal",
            f"Signal: {side.upper()} {ticker} | edge={edge:.1%} conf={confidence:.1%}",
            strategy=strategy,
            details=details,
        )
        return signal_id

    async def execute_trade(
        self,
        strategy: str,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        our_prob: float = 0.0,
        kalshi_prob: float = 0.0,
        market_title: str = "",
        signal_source: str = "",
        signal_id: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Execute a trade (paper or live).
        Returns trade record dict.
        """
        cost = count * price_cents / 100.0
        fee = _kalshi_maker_fee(count, price_cents)
        edge = abs(our_prob - kalshi_prob)

        # Risk check
        allowed, reason = self.check_risk_limits(strategy, cost, ticker=ticker)
        if not allowed:
            self.log_event("blocked", f"Trade blocked: {reason}", strategy=strategy)
            return {"status": "blocked", "reason": reason}

        trade_id = str(uuid.uuid4())[:12]
        order_id = ""

        if self.paper_mode:
            # Paper trade — just log it
            order_id = f"PAPER-{trade_id}"
            status = "filled"
            self.log_event(
                "paper_trade",
                f"PAPER: {side.upper()} {count}x {ticker} @ {price_cents}c = ${cost:.2f} (fee ${fee:.2f})",
                strategy=strategy,
            )
        else:
            # Live trade — place on Kalshi
            try:
                from app.services.kalshi_api import get_kalshi_client

                client = get_kalshi_client()
                if side == "yes":
                    result = await client.place_order(
                        ticker=ticker,
                        side="yes",
                        action="buy",
                        count=count,
                        type="limit",
                        yes_price=price_cents,
                    )
                else:
                    result = await client.place_order(
                        ticker=ticker,
                        side="no",
                        action="buy",
                        count=count,
                        type="limit",
                        no_price=price_cents,
                    )
                order_id = result.get("order", {}).get("order_id", "")
                status = result.get("order", {}).get("status", "pending")
                self.log_event(
                    "live_trade",
                    f"LIVE: {side.upper()} {count}x {ticker} @ {price_cents}c order_id={order_id} status={status}",
                    strategy=strategy,
                )

                # Wait for fill confirmation (up to 30 seconds)
                if status not in ("filled", "canceled", "error") and order_id:
                    import asyncio as _asyncio
                    for _attempt in range(6):
                        await _asyncio.sleep(5)
                        try:
                            order_data = await client.get_order(order_id)
                            order_info = order_data.get("order", order_data)
                            status = order_info.get("status", status)
                            fill_count = order_info.get("fill_count", 0)
                            if status == "filled":
                                count = fill_count or count
                                cost = count * price_cents / 100.0
                                fee = _kalshi_maker_fee(count, price_cents)
                                self.log_event("live_trade", f"Order filled: {count}x @ {price_cents}c", strategy=strategy)
                                break
                            elif status in ("canceled", "error"):
                                self.log_event("warning", f"Order {status}: {order_id}", strategy=strategy)
                                return {"status": status, "reason": f"Order {status}"}
                        except Exception:
                            pass
                    else:
                        # Not filled after 30s — cancel and report timeout
                        try:
                            await client.cancel_order(order_id)
                            self.log_event("warning", f"Order timeout, canceled: {order_id}", strategy=strategy)
                        except Exception:
                            pass
                        return {"status": "timeout", "reason": "Order not filled within 30s"}

            except Exception as e:
                self.log_event("error", f"Order failed: {e}", strategy=strategy)
                return {"status": "error", "reason": str(e)}

        # Track cycle deployment for per-strategy caps
        self._record_cycle_deployment(strategy, cost + fee)

        # Record trade
        trade = {
            "id": trade_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "strategy": strategy,
            "ticker": ticker,
            "market_title": market_title,
            "side": side,
            "action": "buy",
            "count": count,
            "price_cents": price_cents,
            "cost": cost,
            "fee": fee,
            "order_type": "limit",
            "paper_mode": self.paper_mode,
            "order_id": order_id,
            "status": status,
            "our_prob": our_prob,
            "kalshi_prob": kalshi_prob,
            "edge": edge,
            "signal_source": signal_source,
            "notes": notes,
        }

        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            """INSERT INTO trades
            (id, timestamp, strategy, ticker, market_title, side, action, count,
             price_cents, cost, fee, order_type, paper_mode, order_id, status,
             our_prob, kalshi_prob, edge, signal_source, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id,
                trade["timestamp"],
                strategy,
                ticker,
                market_title,
                side,
                "buy",
                count,
                price_cents,
                cost,
                fee,
                "limit",
                1 if self.paper_mode else 0,
                order_id,
                status,
                our_prob,
                kalshi_prob,
                edge,
                signal_source,
                notes,
            ),
        )

        # Update signal if linked
        if signal_id:
            c.execute(
                "UPDATE signals SET acted_on = 1, trade_id = ? WHERE id = ?",
                (trade_id, signal_id),
            )

        conn.commit()
        conn.close()

        return trade

    def settle_trade(self, trade_id: str, result: str, settlement_value: float = 0.0) -> dict[str, Any]:
        """
        Settle a trade (mark as won/lost and calculate P&L).
        result: 'yes' or 'no' (the market outcome)
        """
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return {"error": "Trade not found"}

        trade = dict(row)
        side = trade["side"]
        action = trade.get("action", "buy")
        count = trade["count"]
        cost = trade["cost"]
        fee = trade["fee"]

        # Calculate P&L
        if action == "sell":
            # Sell/exit trades: P&L already realized at exit.
            # cost is negative (proceeds), so P&L = -cost - fee = proceeds - fee
            pnl = -cost - fee
        elif result == side:
            # Buy trade won: payout is $1 per contract minus cost minus fee
            pnl = (count * 1.0) - cost - fee
        else:
            # Buy trade lost: lose cost plus fee
            pnl = -cost - fee

        now = datetime.now(UTC).isoformat()
        c.execute(
            "UPDATE trades SET result = ?, pnl = ?, settled_at = ?, status = 'settled' WHERE id = ?",
            (result, pnl, now, trade_id),
        )

        # Update daily P&L
        date = trade["timestamp"][:10]
        strategy = trade["strategy"]
        c.execute(
            """INSERT INTO daily_pnl (date, strategy, gross_pnl, fees, net_pnl, trades_count, wins, losses)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(date, strategy) DO UPDATE SET
                gross_pnl = gross_pnl + ?,
                fees = fees + ?,
                net_pnl = net_pnl + ?,
                trades_count = trades_count + 1,
                wins = wins + ?,
                losses = losses + ?""",
            (
                date, strategy,
                pnl + fee, fee, pnl, 1 if pnl > 0 else 0, 1 if pnl <= 0 else 0,
                pnl + fee, fee, pnl, 1 if pnl > 0 else 0, 1 if pnl <= 0 else 0,
            ),
        )

        conn.commit()
        conn.close()

        self.log_event(
            "settlement",
            f"Settled {trade_id}: {result} | P&L=${pnl:+.2f}",
            strategy=strategy,
        )

        return {"trade_id": trade_id, "result": result, "pnl": pnl}

    def get_unsettled_trades(self) -> list[dict[str, Any]]:
        """Get all unsettled (filled) trades grouped by ticker for settlement checking."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT id, ticker, side, action, count, cost, fee, strategy
            FROM trades
            WHERE status = 'filled'
            ORDER BY ticker
        """)
        trades = [dict(r) for r in c.fetchall()]
        conn.close()
        return trades

    # ── Position Management ────────────────────────────────────────

    def get_open_positions(self) -> list[dict[str, Any]]:
        """
        Get all open (unsettled) positions, aggregated by ticker.
        Properly nets buy and sell trades: net_contracts = buys - sells.
        Only returns positions with net_contracts > 0.
        """
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT
                ticker,
                MAX(CASE WHEN action = 'buy' THEN market_title ELSE '' END) as market_title,
                side,
                strategy,
                MAX(signal_source) as signal_source,
                SUM(CASE WHEN action = 'buy' THEN count ELSE 0 END) as buy_contracts,
                SUM(CASE WHEN action = 'sell' THEN count ELSE 0 END) as sell_contracts,
                SUM(CASE WHEN action = 'buy' THEN cost ELSE 0 END) as buy_cost,
                SUM(CASE WHEN action = 'sell' THEN -cost ELSE 0 END) as sell_proceeds,
                SUM(fee) as total_fees,
                AVG(CASE WHEN action = 'buy' THEN our_prob ELSE NULL END) as avg_our_prob,
                AVG(CASE WHEN action = 'buy' THEN kalshi_prob ELSE NULL END) as avg_entry_kalshi_prob,
                AVG(CASE WHEN action = 'buy' THEN edge ELSE NULL END) as avg_entry_edge,
                MIN(timestamp) as first_entry,
                MAX(timestamp) as last_entry,
                COUNT(*) as num_fills,
                paper_mode
            FROM trades
            WHERE status = 'filled'
            GROUP BY ticker, side
            HAVING SUM(CASE WHEN action = 'buy' THEN count ELSE 0 END)
                 - SUM(CASE WHEN action = 'sell' THEN count ELSE 0 END) > 0
            ORDER BY MAX(timestamp) DESC
        """)

        positions = []
        for row in c.fetchall():
            r = dict(row)
            contracts = r["buy_contracts"] - r["sell_contracts"]
            buy_cost = r["buy_cost"]
            total_fees = r["total_fees"]

            # Net cost = buy cost - sell proceeds
            net_cost = buy_cost - r["sell_proceeds"]

            # Avg entry based on buy trades only
            avg_entry = round(buy_cost * 100.0 / r["buy_contracts"], 1) if r["buy_contracts"] > 0 else 0

            # Max risk = net cost + fees
            max_risk = net_cost + total_fees

            # Max profit = (net contracts * $1) - net_cost - fees
            max_profit = (contracts * 1.0) - net_cost - total_fees

            positions.append({
                "ticker": r["ticker"],
                "title": r["market_title"],
                "side": r["side"],
                "strategy": r["strategy"],
                "signal_source": r["signal_source"],
                "contracts": contracts,
                "avg_entry_cents": round(avg_entry),
                "total_cost": round(net_cost, 2),
                "total_fees": round(total_fees, 2),
                "max_risk": round(max_risk, 2),
                "max_profit": round(max_profit, 2),
                "avg_our_prob": round(r["avg_our_prob"] or 0, 4),
                "avg_entry_kalshi_prob": round(r["avg_entry_kalshi_prob"] or 0, 4),
                "avg_entry_edge": round(r["avg_entry_edge"] or 0, 4),
                "first_entry": r["first_entry"],
                "last_entry": r["last_entry"],
                "num_fills": r["num_fills"],
                "paper_mode": bool(r["paper_mode"]),
                # These will be filled in by the agent with live data
                "current_yes_ask": None,
                "current_no_ask": None,
                "current_yes_bid": None,
                "current_no_bid": None,
                "mark_price_cents": None,
                "unrealized_pnl": None,
                "current_edge": None,
                "status": "open",
            })

        conn.close()
        return positions

    async def exit_trade(
        self,
        strategy: str,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        reason: str = "",
    ) -> dict[str, Any]:
        """
        Exit (sell) an existing position.
        Records as a sell trade with negative cost (proceeds).
        """
        cost = count * price_cents / 100.0
        fee = _kalshi_maker_fee(count, price_cents)

        trade_id = str(uuid.uuid4())[:12]
        order_id = ""

        # Determine the sell side: if we bought YES, we sell YES
        sell_action = "sell"

        if self.paper_mode:
            order_id = f"PAPER-EXIT-{trade_id}"
            status = "filled"
            self.log_event(
                "paper_trade",
                f"PAPER EXIT: {sell_action} {side.upper()} {count}x {ticker} @ {price_cents}c = ${cost:.2f} (fee ${fee:.2f}) reason={reason}",
                strategy=strategy,
            )
        else:
            try:
                from app.services.kalshi_api import get_kalshi_client
                client = get_kalshi_client()
                if side == "yes":
                    result = await client.place_order(
                        ticker=ticker, side="yes", action="sell",
                        count=count, type="limit", yes_price=price_cents,
                    )
                else:
                    result = await client.place_order(
                        ticker=ticker, side="no", action="sell",
                        count=count, type="limit", no_price=price_cents,
                    )
                order_id = result.get("order", {}).get("order_id", "")
                status = result.get("order", {}).get("status", "pending")

                # Wait for fill confirmation (up to 30 seconds)
                if status not in ("filled", "canceled", "error") and order_id:
                    import asyncio as _asyncio
                    for _attempt in range(6):
                        await _asyncio.sleep(5)
                        try:
                            order_data = await client.get_order(order_id)
                            order_info = order_data.get("order", order_data)
                            status = order_info.get("status", status)
                            if status == "filled":
                                fill_count = order_info.get("fill_count", 0)
                                if fill_count:
                                    count = fill_count
                                    cost = count * price_cents / 100.0
                                    fee = _kalshi_maker_fee(count, price_cents)
                                break
                            elif status in ("canceled", "error"):
                                self.log_event("warning", f"Exit order {status}: {order_id}", strategy=strategy)
                                return {"status": status, "reason": f"Exit order {status}"}
                        except Exception:
                            pass
                    else:
                        try:
                            await client.cancel_order(order_id)
                            self.log_event("warning", f"Exit order timeout, canceled: {order_id}", strategy=strategy)
                        except Exception:
                            pass
                        return {"status": "timeout", "reason": "Exit order not filled within 30s"}

            except Exception as e:
                self.log_event("error", f"Exit order failed: {e}", strategy=strategy)
                return {"status": "error", "reason": str(e)}

        # ── Compute realized P&L from entry price ──
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Look up average entry cost+fee from buy trades for this ticker+side
        c.execute(
            """SELECT SUM(count) as total_count, SUM(cost) as total_cost, SUM(fee) as total_fees,
                      MAX(market_title) as market_title
            FROM trades WHERE ticker = ? AND side = ? AND action = 'buy'
            AND status = 'filled' AND paper_mode = ?""",
            (ticker, side, 1 if self.paper_mode else 0),
        )
        entry_row = c.fetchone()
        entry_cost_per_contract = 0.0
        entry_fee_per_contract = 0.0
        original_market_title = ""
        if entry_row and entry_row["total_count"] and entry_row["total_count"] > 0:
            entry_cost_per_contract = entry_row["total_cost"] / entry_row["total_count"]
            entry_fee_per_contract = entry_row["total_fees"] / entry_row["total_count"]
            original_market_title = entry_row["market_title"] or ""

        # P&L = exit_proceeds - entry_cost - entry_fees(proportional) - exit_fee
        exit_price_per = price_cents / 100.0
        pnl = (exit_price_per - entry_cost_per_contract) * count - (entry_fee_per_contract * count) - fee

        now = datetime.now(UTC).isoformat()

        trade = {
            "id": trade_id,
            "timestamp": now,
            "strategy": strategy,
            "ticker": ticker,
            "market_title": original_market_title,
            "side": side,
            "action": sell_action,
            "count": count,
            "price_cents": price_cents,
            "cost": -cost,  # Negative cost = proceeds
            "fee": fee,
            "order_type": "limit",
            "paper_mode": self.paper_mode,
            "order_id": order_id,
            "status": status,
            "our_prob": 0,
            "kalshi_prob": 0,
            "edge": 0,
            "signal_source": "position_monitor",
            "notes": f"EXIT: {reason}",
            "pnl": round(pnl, 4),
        }

        c.execute(
            """INSERT INTO trades
            (id, timestamp, strategy, ticker, market_title, side, action, count,
             price_cents, cost, fee, order_type, paper_mode, order_id, status,
             our_prob, kalshi_prob, edge, signal_source, notes,
             result, pnl, settled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade_id, now, strategy, ticker, original_market_title,
                side, sell_action, count, price_cents, -cost, fee,
                "limit", 1 if self.paper_mode else 0, order_id, status,
                0, 0, 0, "position_monitor", f"EXIT: {reason}",
                "exit", round(pnl, 4), now,
            ),
        )

        # Update daily P&L
        date = now[:10]
        c.execute(
            """INSERT INTO daily_pnl (date, strategy, gross_pnl, fees, net_pnl, trades_count, wins, losses)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(date, strategy) DO UPDATE SET
                gross_pnl = gross_pnl + ?,
                fees = fees + ?,
                net_pnl = net_pnl + ?,
                trades_count = trades_count + 1,
                wins = wins + ?,
                losses = losses + ?""",
            (
                date, strategy,
                round(pnl + fee, 4), fee, round(pnl, 4),
                1 if pnl > 0 else 0, 1 if pnl <= 0 else 0,
                round(pnl + fee, 4), fee, round(pnl, 4),
                1 if pnl > 0 else 0, 1 if pnl <= 0 else 0,
            ),
        )

        conn.commit()
        conn.close()

        self.log_event(
            "exit_trade",
            f"EXIT {side.upper()} {count}x {ticker} @ {price_cents}c | entry_avg={entry_cost_per_contract:.2f} P&L=${pnl:+.2f} | {reason}",
            strategy=strategy,
        )

        return trade

    # ── Query methods ──────────────────────────────────────────────

    def get_trades(
        self,
        strategy: str | None = None,
        status: str | None = None,
        limit: int = 100,
        paper_only: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Get trades from the database."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        query = "SELECT * FROM trades WHERE 1=1"
        params: list[Any] = []

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if status:
            query += " AND status = ?"
            params.append(status)
        if paper_only is not None:
            query += " AND paper_mode = ?"
            params.append(1 if paper_only else 0)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        c.execute(query, params)
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_signals(
        self,
        strategy: str | None = None,
        acted_on: bool | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get signals from the database."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        query = "SELECT * FROM signals WHERE 1=1"
        params: list[Any] = []

        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if acted_on is not None:
            query += " AND acted_on = ?"
            params.append(1 if acted_on else 0)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        c.execute(query, params)
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_agent_log(self, limit: int = 200, strategy: str | None = None) -> list[dict[str, Any]]:
        """Get agent log entries."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        if strategy:
            c.execute(
                "SELECT * FROM agent_log WHERE strategy = ? ORDER BY id DESC LIMIT ?",
                (strategy, limit),
            )
        else:
            c.execute("SELECT * FROM agent_log ORDER BY id DESC LIMIT ?", (limit,))

        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_performance_summary(self) -> dict[str, Any]:
        """Get overall performance summary."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Overall stats — include both market-settled trades AND exit/sell trades
        # Exit trades have pnl computed at exit time but keep status='filled'
        c.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(SUM(fee), 0) as total_fees,
                COALESCE(SUM(CASE WHEN action = 'buy' THEN cost ELSE 0 END), 0) as total_wagered
            FROM trades WHERE status = 'settled' OR action = 'sell'
        """)
        overall = dict(c.fetchone())

        # Per-strategy stats
        c.execute("""
            SELECT
                strategy,
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(SUM(fee), 0) as total_fees
            FROM trades WHERE status = 'settled' OR action = 'sell'
            GROUP BY strategy
        """)
        by_strategy = {row["strategy"]: dict(row) for row in c.fetchall()}

        # Daily P&L
        c.execute("""
            SELECT date, SUM(net_pnl) as net_pnl, SUM(trades_count) as trades
            FROM daily_pnl
            GROUP BY date
            ORDER BY date DESC
            LIMIT 30
        """)
        daily = [dict(r) for r in c.fetchall()]

        conn.close()

        total_trades = overall["total_trades"] or 0
        wins = overall["wins"] or 0

        return {
            "bankroll": self.bankroll,
            "paper_mode": self.paper_mode,
            "kill_switch": self.kill_switch,
            "strategy_enabled": self.strategy_enabled,
            "overall": {
                **overall,
                "win_rate": wins / total_trades if total_trades > 0 else 0,
                "roi": overall["total_pnl"] / overall["total_wagered"] if overall["total_wagered"] else 0,
            },
            "by_strategy": by_strategy,
            "daily_pnl": daily,
            "today_pnl": self.get_today_pnl(),
            "today_trades": self.get_today_trade_count(),
        }

    def get_status(self) -> dict[str, Any]:
        """Get current agent status."""
        effective = self.get_effective_bankroll()
        total_exposure = self.get_total_exposure()
        max_deployable = effective * self.max_total_exposure_pct
        remaining = max_deployable - total_exposure
        dynamic_loss_limit = effective * 0.20
        return {
            "paper_mode": self.paper_mode,
            "kill_switch": self.kill_switch,
            "bankroll": self.bankroll,
            "effective_bankroll": round(effective, 2),
            "strategy_enabled": self.strategy_enabled,
            "daily_loss_limit": round(dynamic_loss_limit, 2),
            "max_bet_size": round(effective * self.max_bet_pct, 2),
            "today_pnl": self.get_today_pnl(),
            "today_trades": self.get_today_trade_count(),
            "total_exposure": round(total_exposure, 2),
            "max_deployable": round(max_deployable, 2),
            "remaining_capital": round(remaining, 2),
            "over_deployed": total_exposure > effective,
        }


# Singleton
_engine: TradingEngine | None = None


def get_trading_engine() -> TradingEngine:
    """Get or create the singleton trading engine.
    Reads from environment variables:
      PAPER_MODE  - 'false' for live trading (default: 'true')
      BANKROLL    - starting bankroll in dollars (default: '2000')
    """
    global _engine
    if _engine is None:
        paper_mode = os.environ.get("PAPER_MODE", "true").lower() != "false"
        bankroll = float(os.environ.get("BANKROLL", "2000"))
        _engine = TradingEngine(paper_mode=paper_mode, bankroll=bankroll)
    return _engine
