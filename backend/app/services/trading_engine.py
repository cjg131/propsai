"""
Trading Engine for Kalshi autonomous agent.
Handles order placement, bankroll management, risk limits, and paper trading.
"""
from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from datetime import datetime, timezone
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
        daily_loss_limit: float = 200.0,
        max_bet_size: float = 40.0,
        max_bet_pct: float = 0.04,  # 4% of strategy bankroll per bet
        max_position_per_ticker: float = 60.0,  # Max $60 exposure per ticker
        max_total_exposure_pct: float = 0.80,  # Never deploy more than 80% of bankroll
    ):
        self.bankroll = bankroll
        self.paper_mode = paper_mode
        self.daily_loss_limit = daily_loss_limit
        self.max_bet_size = max_bet_size
        self.max_bet_pct = max_bet_pct
        self.max_position_per_ticker = max_position_per_ticker
        self.max_total_exposure_pct = max_total_exposure_pct
        self.kill_switch = False

        # Strategy allocations
        self.allocations = {
            "weather": 0.45,
            "sports": 0.45,
            "reserve": 0.10,
        }

        # Strategy-level toggles
        self.strategy_enabled = {
            "weather": True,
            "sports": True,
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
            (datetime.now(timezone.utc).isoformat(), level, strategy, message, details),
        )
        conn.commit()
        conn.close()

    def get_strategy_bankroll(self, strategy: str) -> float:
        """Get the allocated bankroll for a strategy."""
        alloc = self.allocations.get(strategy, 0)
        return self.bankroll * alloc

    def get_today_pnl(self, strategy: str | None = None) -> float:
        """Get today's P&L, optionally filtered by strategy."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
        """Get total capital currently deployed in open (filled, unsettled) trades."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        if strategy:
            c.execute(
                "SELECT COALESCE(SUM(cost + fee), 0) FROM trades WHERE status = 'filled' AND action = 'buy' AND strategy = ?",
                (strategy,),
            )
        else:
            c.execute(
                "SELECT COALESCE(SUM(cost + fee), 0) FROM trades WHERE status = 'filled' AND action = 'buy'"
            )
        result = c.fetchone()[0]
        conn.close()
        return result

    def get_ticker_exposure(self, ticker: str) -> float:
        """Get total capital deployed on a specific ticker."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "SELECT COALESCE(SUM(cost + fee), 0) FROM trades WHERE status = 'filled' AND action = 'buy' AND ticker = ?",
            (ticker,),
        )
        result = c.fetchone()[0]
        conn.close()
        return result

    def get_remaining_capital(self, strategy: str) -> float:
        """Get remaining deployable capital for a strategy."""
        strat_bankroll = self.get_strategy_bankroll(strategy)
        strat_exposure = self.get_total_exposure(strategy)
        max_deployable = self.bankroll * self.max_total_exposure_pct
        total_exposure = self.get_total_exposure()
        global_remaining = max_deployable - total_exposure
        strat_remaining = strat_bankroll - strat_exposure
        return max(0, min(global_remaining, strat_remaining))

    def check_risk_limits(self, strategy: str, cost: float, ticker: str = "") -> tuple[bool, str]:
        """
        Check if a trade passes all risk limits.
        Returns (allowed, reason).
        """
        if self.kill_switch:
            return False, "Kill switch is active"

        if not self.strategy_enabled.get(strategy, False):
            return False, f"Strategy '{strategy}' is disabled"

        # Check per-bet size limit
        if cost > self.max_bet_size:
            return False, f"Bet size ${cost:.2f} exceeds max ${self.max_bet_size:.2f}"

        # Check percentage of strategy bankroll
        strat_bankroll = self.get_strategy_bankroll(strategy)
        if strat_bankroll > 0 and cost > strat_bankroll * self.max_bet_pct:
            return False, f"Bet size ${cost:.2f} exceeds {self.max_bet_pct*100:.0f}% of {strategy} bankroll (${strat_bankroll:.2f})"

        # Check total bankroll exposure — CRITICAL GUARDRAIL
        total_exposure = self.get_total_exposure()
        max_deployable = self.bankroll * self.max_total_exposure_pct
        if total_exposure + cost > max_deployable:
            return False, f"Total exposure ${total_exposure + cost:.2f} would exceed {self.max_total_exposure_pct*100:.0f}% of bankroll (${max_deployable:.2f})"

        # Check strategy-level exposure
        strat_exposure = self.get_total_exposure(strategy)
        if strat_exposure + cost > strat_bankroll:
            return False, f"Strategy '{strategy}' exposure ${strat_exposure + cost:.2f} would exceed allocation ${strat_bankroll:.2f}"

        # Check per-ticker position limit
        if ticker:
            ticker_exposure = self.get_ticker_exposure(ticker)
            if ticker_exposure + cost > self.max_position_per_ticker:
                return False, f"Ticker {ticker} exposure ${ticker_exposure + cost:.2f} would exceed max ${self.max_position_per_ticker:.2f}"

        # Check daily loss limit
        today_pnl = self.get_today_pnl()
        if today_pnl < -self.daily_loss_limit:
            return False, f"Daily loss limit reached: ${today_pnl:.2f} (limit: -${self.daily_loss_limit:.2f})"

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

        strat_bankroll = self.get_strategy_bankroll(strategy)
        remaining = self.get_remaining_capital(strategy)

        # Cap by remaining capital, per-bet max, and Kelly
        max_dollars = min(
            remaining,
            strat_bankroll * quarter_kelly,
            self.max_bet_size,
            strat_bankroll * self.max_bet_pct,
        )

        # Also cap by per-ticker limit
        if ticker:
            ticker_exposure = self.get_ticker_exposure(ticker)
            ticker_remaining = max(0, self.max_position_per_ticker - ticker_exposure)
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
                datetime.now(timezone.utc).isoformat(),
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
                        no_price=100 - price_cents,
                    )
                order_id = result.get("order", {}).get("order_id", "")
                status = result.get("order", {}).get("status", "pending")
                self.log_event(
                    "live_trade",
                    f"LIVE: {side.upper()} {count}x {ticker} @ {price_cents}c order_id={order_id}",
                    strategy=strategy,
                )
            except Exception as e:
                self.log_event("error", f"Order failed: {e}", strategy=strategy)
                return {"status": "error", "reason": str(e)}

        # Record trade
        trade = {
            "id": trade_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
        count = trade["count"]
        cost = trade["cost"]
        fee = trade["fee"]

        # Calculate P&L
        if result == side:
            # Won: payout is $1 per contract minus cost minus fee
            pnl = (count * 1.0) - cost - fee
        else:
            # Lost: lose cost plus fee
            pnl = -cost - fee

        now = datetime.now(timezone.utc).isoformat()
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

    # ── Position Management ────────────────────────────────────────

    def get_open_positions(self) -> list[dict[str, Any]]:
        """
        Get all open (unsettled) positions, aggregated by ticker.
        Each position includes entry price, contracts, cost basis, and max risk.
        """
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("""
            SELECT 
                ticker,
                market_title,
                side,
                strategy,
                signal_source,
                SUM(count) as total_contracts,
                SUM(cost) as total_cost,
                SUM(fee) as total_fees,
                ROUND(SUM(cost * 100.0) / SUM(count), 1) as avg_entry_cents,
                AVG(our_prob) as avg_our_prob,
                AVG(kalshi_prob) as avg_entry_kalshi_prob,
                AVG(edge) as avg_entry_edge,
                MIN(timestamp) as first_entry,
                MAX(timestamp) as last_entry,
                COUNT(*) as num_fills,
                paper_mode
            FROM trades
            WHERE status = 'filled'
            GROUP BY ticker, side
            ORDER BY MAX(timestamp) DESC
        """)

        positions = []
        for row in c.fetchall():
            r = dict(row)
            total_cost = r["total_cost"]
            total_fees = r["total_fees"]
            contracts = r["total_contracts"]

            # Max risk = total cost + fees (if the position goes to zero)
            max_risk = total_cost + total_fees

            # Max profit = (contracts * $1) - total_cost - fees (if position settles YES)
            if r["side"] == "yes":
                max_profit = (contracts * 1.0) - total_cost - total_fees
            else:
                max_profit = (contracts * 1.0) - total_cost - total_fees

            positions.append({
                "ticker": r["ticker"],
                "title": r["market_title"],
                "side": r["side"],
                "strategy": r["strategy"],
                "signal_source": r["signal_source"],
                "contracts": contracts,
                "avg_entry_cents": round(r["avg_entry_cents"]),
                "total_cost": round(total_cost, 2),
                "total_fees": round(total_fees, 2),
                "max_risk": round(max_risk, 2),
                "max_profit": round(max_profit, 2),
                "avg_our_prob": round(r["avg_our_prob"], 4),
                "avg_entry_kalshi_prob": round(r["avg_entry_kalshi_prob"], 4),
                "avg_entry_edge": round(r["avg_entry_edge"], 4),
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
                        count=count, type="limit", no_price=100 - price_cents,
                    )
                order_id = result.get("order", {}).get("order_id", "")
                status = result.get("order", {}).get("status", "pending")
            except Exception as e:
                self.log_event("error", f"Exit order failed: {e}", strategy=strategy)
                return {"status": "error", "reason": str(e)}

        trade = {
            "id": trade_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "ticker": ticker,
            "market_title": "",
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
                trade_id, trade["timestamp"], strategy, ticker, "",
                side, sell_action, count, price_cents, -cost, fee,
                "limit", 1 if self.paper_mode else 0, order_id, status,
                0, 0, 0, "position_monitor", f"EXIT: {reason}",
            ),
        )
        conn.commit()
        conn.close()

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

        # Overall stats
        c.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 AND status = 'settled' THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(SUM(fee), 0) as total_fees,
                COALESCE(SUM(cost), 0) as total_wagered
            FROM trades WHERE status = 'settled'
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
            FROM trades WHERE status = 'settled'
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
        total_exposure = self.get_total_exposure()
        max_deployable = self.bankroll * self.max_total_exposure_pct
        return {
            "paper_mode": self.paper_mode,
            "kill_switch": self.kill_switch,
            "bankroll": self.bankroll,
            "strategy_enabled": self.strategy_enabled,
            "allocations": self.allocations,
            "daily_loss_limit": self.daily_loss_limit,
            "max_bet_size": self.max_bet_size,
            "today_pnl": self.get_today_pnl(),
            "today_trades": self.get_today_trade_count(),
            "total_exposure": round(total_exposure, 2),
            "max_deployable": round(max_deployable, 2),
            "remaining_capital": round(max_deployable - total_exposure, 2),
        }


# Singleton
_engine: TradingEngine | None = None


def get_trading_engine() -> TradingEngine:
    """Get or create the singleton trading engine."""
    global _engine
    if _engine is None:
        _engine = TradingEngine(paper_mode=True)
    return _engine
