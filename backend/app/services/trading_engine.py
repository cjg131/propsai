"""
Trading Engine for Kalshi autonomous agent.
Handles order placement, bankroll management, risk limits, and paper trading.
"""
from __future__ import annotations

import math
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.logging_config import get_logger
from app.services.performance_model import load_performance_model

logger = get_logger(__name__)

UTC = timezone.utc

DB_PATH = Path(__file__).parent.parent / "data" / "trading_engine.db"
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
REALIZED_TRADE_SQL = "(status = 'settled' OR (action = 'sell' AND status IN ('filled', 'executed', 'settled')))"


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


def _load_env_defaults() -> None:
    """Populate os.environ from backend/.env when the process did not export them.

    This keeps the trading engine in sync with the same config source used by the
    pydantic settings loader, even when invoked from scripts or dev servers.
    """
    if not ENV_PATH.exists():
        return

    try:
        for raw_line in ENV_PATH.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # If env loading fails, fall back to the process environment.
        return


def _schedule_background_coro(coro: Any) -> None:
    """Schedule a coroutine when a loop exists; otherwise close it quietly."""
    if coro is None:
        return
    try:
        import asyncio
        import inspect

        if not inspect.iscoroutine(coro):
            return
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        close = getattr(coro, "close", None)
        if callable(close):
            close()


def _price_bucket(price_cents: int) -> str:
    if price_cents < 10:
        return "01-09c"
    if price_cents < 20:
        return "10-19c"
    if price_cents < 30:
        return "20-29c"
    if price_cents < 40:
        return "30-39c"
    if price_cents < 50:
        return "40-49c"
    if price_cents < 60:
        return "50-59c"
    if price_cents < 70:
        return "60-69c"
    if price_cents < 80:
        return "70-79c"
    if price_cents < 90:
        return "80-89c"
    return "90-99c"


FAMILY_QUALITY_PRIORS: dict[str, float] = {
    "weather_observed": 1.25,
    "weather_forecast": 0.45,
    "sports_single": 0.75,
    "sports_single_soccer": 0.45,
    "sports_parlay": 0.0,
    "crypto_momentum": 0.70,
    "finance_bracket": 0.80,
    "finance_threshold": 0.85,
    "econ": 0.80,
    "nba_props": 0.70,
    "other": 0.75,
}

FAMILY_ENTRY_RULES: dict[str, dict[str, float | bool]] = {
    "weather_observed": {"min_edge": 0.10, "min_confidence": 0.75, "min_win_prob": 0.75},
    "weather_forecast": {"min_edge": 0.08, "min_confidence": 0.78, "min_win_prob": 0.60},
    "sports_single": {"min_edge": 0.06, "min_confidence": 0.68, "min_win_prob": 0.57},
    "sports_single_soccer": {"min_edge": 0.08, "min_confidence": 0.72, "min_win_prob": 0.60},
    "sports_parlay": {"blocked": True},
    "crypto_momentum": {"min_edge": 0.05, "min_confidence": 0.70, "min_win_prob": 0.56},
    "finance_bracket": {"min_edge": 0.05, "min_confidence": 0.68, "min_win_prob": 0.56},
    "finance_threshold": {"min_edge": 0.05, "min_confidence": 0.65, "min_win_prob": 0.55},
    "econ": {"min_edge": 0.05, "min_confidence": 0.65, "min_win_prob": 0.55},
    "nba_props": {"min_edge": 0.06, "min_confidence": 0.72, "min_win_prob": 0.58},
    "other": {"min_edge": 0.05, "min_confidence": 0.65, "min_win_prob": 0.55},
}

FAMILY_LIFETIME_LIMITS: dict[str, dict[str, int | bool]] = {
    "weather_observed": {
        "ticker_trades": 2,
        "ticker_contracts": 160,
        "event_trades": 3,
        "event_contracts": 220,
    },
    "weather_forecast": {
        "ticker_trades": 1,
        "ticker_contracts": 40,
        "event_trades": 1,
        "event_contracts": 40,
    },
    "sports_single": {
        "ticker_trades": 1,
        "ticker_contracts": 80,
        "event_trades": 1,
        "event_contracts": 80,
    },
    "sports_single_soccer": {
        "ticker_trades": 1,
        "ticker_contracts": 50,
        "event_trades": 1,
        "event_contracts": 50,
    },
    "sports_parlay": {"blocked": True},
    "crypto_momentum": {
        "ticker_trades": 1,
        "ticker_contracts": 60,
        "event_trades": 1,
        "event_contracts": 60,
    },
    "finance_bracket": {
        "ticker_trades": 1,
        "ticker_contracts": 60,
        "event_trades": 1,
        "event_contracts": 60,
    },
    "finance_threshold": {
        "ticker_trades": 1,
        "ticker_contracts": 70,
        "event_trades": 1,
        "event_contracts": 70,
    },
    "econ": {
        "ticker_trades": 1,
        "ticker_contracts": 60,
        "event_trades": 1,
        "event_contracts": 60,
    },
    "nba_props": {
        "ticker_trades": 1,
        "ticker_contracts": 35,
        "event_trades": 1,
        "event_contracts": 35,
    },
    "other": {
        "ticker_trades": 1,
        "ticker_contracts": 60,
        "event_trades": 1,
        "event_contracts": 60,
    },
}

ALL_STRATEGIES = ("arbitrage", "weather", "sports", "crypto", "nba_props", "finance", "econ")


def _parse_enabled_strategies(value: str | None, *, default: set[str] | None = None) -> set[str]:
    strategies = default or set()
    if value is None:
        return set(strategies)

    parsed = {
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    }
    valid = {strategy for strategy in parsed if strategy in ALL_STRATEGIES}
    return valid or set(strategies)


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
        max_bet_pct: float = 0.03,  # HARD CAP: 3% of bankroll per single bet
        max_position_per_ticker: float | None = None,
        max_total_exposure_pct: float = 0.60,  # Max 60% of bankroll deployed at once
        max_strategy_exposure_pct: float = 0.35,  # Max 35% of bankroll per strategy
        max_strategy_cycle_pct: float = 0.15,  # Max 15% of bankroll deployed per strategy per cycle
        max_single_trade_dollars: float = 50.0,  # Absolute hard dollar cap per trade
        min_bankroll_to_trade: float = 50.0,  # Refuse all trades if bankroll below this
    ):
        self.bankroll = bankroll
        self.paper_mode = paper_mode
        self._base_max_bet_pct = max_bet_pct
        self._base_max_total_exposure_pct = max_total_exposure_pct
        self._base_max_strategy_exposure_pct = max_strategy_exposure_pct
        self._base_max_strategy_cycle_pct = max_strategy_cycle_pct
        self._base_max_single_trade_dollars = max_single_trade_dollars
        self._base_min_bankroll_to_trade = min_bankroll_to_trade
        self._base_max_bet_size = max_bet_size if max_bet_size is not None else bankroll * 0.03
        self._base_max_position_per_ticker = (
            max_position_per_ticker if max_position_per_ticker is not None else bankroll * 0.08
        )
        self.max_bet_pct = max_bet_pct
        self.max_total_exposure_pct = max_total_exposure_pct
        self.max_strategy_exposure_pct = max_strategy_exposure_pct
        self.max_strategy_cycle_pct = max_strategy_cycle_pct
        self.max_single_trade_dollars = max_single_trade_dollars
        self.min_bankroll_to_trade = min_bankroll_to_trade
        # Scale risk limits from bankroll (override with explicit values if provided)
        self.daily_loss_limit = daily_loss_limit if daily_loss_limit is not None else float('inf')
        self.max_bet_size = max_bet_size if max_bet_size is not None else bankroll * 0.03
        self.max_position_per_ticker = max_position_per_ticker if max_position_per_ticker is not None else bankroll * 0.08
        self.kill_switch = False
        # Track per-strategy deployment within current cycle
        self._cycle_deployed: dict[str, float] = {}

        # Runtime health snapshots (updated by agent loop)
        self._runtime_api_healthy = True
        self._runtime_db_healthy = True
        self._runtime_ws_healthy = True
        self._last_monitor_heartbeat = ""
        self._performance_model_cache: dict[str, Any] | None = None
        self._performance_model_loaded_at: float = 0.0
        self._broker_total_exposure: float = 0.0
        self._broker_exposure_by_ticker: dict[str, float] = {}
        self._broker_exposure_by_strategy: dict[str, float] = {}

        # Live guardrail configuration
        self.max_total_resting_orders = int(os.environ.get("MAX_TOTAL_RESTING_ORDERS", "50"))
        self.max_resting_orders_per_strategy = int(os.environ.get("MAX_RESTING_ORDERS_PER_STRATEGY", "15"))
        self.max_order_failures_window = int(os.environ.get("MAX_ORDER_FAILURES_WINDOW", "5"))
        self.order_failure_window_mins = int(os.environ.get("ORDER_FAILURE_WINDOW_MINS", "15"))
        self.require_ws_for_live = os.environ.get("REQUIRE_WS_FOR_LIVE", "false").lower() == "true"
        self.enable_auto_kill_on_failures = os.environ.get("AUTO_KILL_ON_ORDER_FAILURES", "true").lower() == "true"

        # Order failure tracking for circuit breaking
        self._recent_order_failures: list[float] = []
        self._last_order_success_ts: float = 0.0

        # Circuit breaker state
        self._cooldown_until: float = 0.0
        self._last_cooldown_date: str = ""
        self._cooldown_pnl_threshold: float = 0.0

        self.allowed_live_strategies = _parse_enabled_strategies(
            os.environ.get("LIVE_ENABLED_STRATEGIES"),
            default={"weather"},
        )
        self.allowed_paper_strategies = _parse_enabled_strategies(
            os.environ.get("PAPER_ENABLED_STRATEGIES"),
            default=set(self.allowed_live_strategies),
        )
        self.strategy_enabled = {strategy: False for strategy in ALL_STRATEGIES}
        self._enforce_strategy_policy()

        self._apply_bankroll_profile()

        self._init_db()

    def _enforce_strategy_policy(self) -> None:
        allowed = self.allowed_paper_strategies if self.paper_mode else self.allowed_live_strategies
        for strategy in self.strategy_enabled:
            self.strategy_enabled[strategy] = strategy in allowed

    def can_enable_strategy(self, strategy: str) -> tuple[bool, str | None]:
        if strategy not in self.strategy_enabled:
            return False, f"Unknown strategy: {strategy}"

        allowed = self.allowed_paper_strategies if self.paper_mode else self.allowed_live_strategies
        if strategy not in allowed:
            mode = "paper" if self.paper_mode else "live"
            allowed_label = ", ".join(sorted(allowed)) if allowed else "none"
            return False, f"Strategy '{strategy}' is not allowed in {mode} mode (allowed: {allowed_label})"
        return True, None

    def is_strategy_active(self, strategy: str) -> tuple[bool, str | None]:
        allowed, reason = self.can_enable_strategy(strategy)
        if not allowed:
            return False, reason
        if not self.strategy_enabled.get(strategy, False):
            return False, f"Strategy '{strategy}' is disabled"
        return True, None

    def _apply_bankroll_profile(self) -> None:
        """Adapt risk profile to bankroll size.

        Small accounts need survival-mode rules so fees and variance do not
        overwhelm the account before we learn anything from live trading.
        """
        # Reset to the base profile before applying bankroll-tier overrides.
        self.max_bet_pct = self._base_max_bet_pct
        self.max_total_exposure_pct = self._base_max_total_exposure_pct
        self.max_strategy_exposure_pct = self._base_max_strategy_exposure_pct
        self.max_strategy_cycle_pct = self._base_max_strategy_cycle_pct
        self.max_single_trade_dollars = self._base_max_single_trade_dollars
        self.min_bankroll_to_trade = self._base_min_bankroll_to_trade
        self.max_bet_size = self._base_max_bet_size
        self.max_position_per_ticker = self._base_max_position_per_ticker

        if self.bankroll <= 250:
            # Micro / low-bankroll live profile for roughly $50-$250.
            self.max_bet_pct = min(self.max_bet_pct, 0.025)
            self.max_total_exposure_pct = min(self.max_total_exposure_pct, 0.15)
            self.max_strategy_exposure_pct = min(self.max_strategy_exposure_pct, 0.12)
            self.max_strategy_cycle_pct = min(self.max_strategy_cycle_pct, 0.05)
            self.max_single_trade_dollars = min(self.max_single_trade_dollars, 5.0)
            self.max_bet_size = min(self.max_bet_size, 5.0)
            self.max_position_per_ticker = min(self.max_position_per_ticker, 10.0)
            self.min_bankroll_to_trade = min(self.min_bankroll_to_trade, 40.0)

        elif self.bankroll <= 500:
            # Small-but-usable profile for roughly $250-$500.
            self.max_bet_pct = min(self.max_bet_pct, 0.03)
            self.max_total_exposure_pct = min(self.max_total_exposure_pct, 0.20)
            self.max_strategy_exposure_pct = min(self.max_strategy_exposure_pct, 0.15)
            self.max_strategy_cycle_pct = min(self.max_strategy_cycle_pct, 0.08)
            self.max_single_trade_dollars = min(self.max_single_trade_dollars, 8.0)
            self.max_bet_size = min(self.max_bet_size, 8.0)
            self.max_position_per_ticker = min(self.max_position_per_ticker, 15.0)

    def sync_bankroll(self, bankroll: float) -> None:
        """Update bankroll and recompute derived risk limits."""
        self.bankroll = bankroll
        self._apply_bankroll_profile()

    def set_paper_mode(self, enabled: bool) -> None:
        self.paper_mode = enabled
        self._enforce_strategy_policy()

    def _infer_strategy_from_ticker(self, ticker: str) -> str:
        ticker_upper = (ticker or "").upper()
        if ticker_upper.startswith("KXHIGH") or ticker_upper.startswith("KXLOW") or ticker_upper.startswith("KXRAIN"):
            return "weather"
        if "NBA" in ticker_upper:
            return "nba_props"
        if any(tag in ticker_upper for tag in ("BTC", "ETH", "CRYPTO")):
            return "crypto"
        if any(tag in ticker_upper for tag in ("INX", "NASDAQ", "DJIA", "RATE", "CPI", "FED")):
            return "finance"
        if "GAME" in ticker_upper or "MATCH" in ticker_upper or "TOTAL" in ticker_upper:
            return "sports"
        return "unknown"

    def set_broker_positions_snapshot(self, positions: list[dict[str, Any]]) -> None:
        """Store broker-side exposure so risk checks can't rely on stale DB rows alone."""
        self._broker_total_exposure = 0.0
        self._broker_exposure_by_ticker = {}
        self._broker_exposure_by_strategy = {}
        for pos in positions:
            ticker = str(pos.get("ticker") or "")
            exposure = float(pos.get("exposure_dollars") or 0.0)
            strategy = str(pos.get("strategy") or self._infer_strategy_from_ticker(ticker))
            if not ticker or exposure <= 0:
                continue
            self._broker_total_exposure += exposure
            self._broker_exposure_by_ticker[ticker] = self._broker_exposure_by_ticker.get(ticker, 0.0) + exposure
            self._broker_exposure_by_strategy[strategy] = self._broker_exposure_by_strategy.get(strategy, 0.0) + exposure

    def _extract_actual_fill_execution(
        self,
        order_info: dict[str, Any],
        *,
        fallback_count: int,
        fallback_price_cents: int,
        action: str,
    ) -> tuple[int, int, float, float]:
        """Extract broker-reported fill count, avg price, cost/proceeds, and fees."""
        count = int(order_info.get("fill_count") or fallback_count or 0)
        maker_cost = order_info.get("maker_fill_cost_dollars")
        taker_cost = order_info.get("taker_fill_cost_dollars")
        gross_dollars = float(taker_cost or maker_cost or 0.0) if (taker_cost or maker_cost) not in ("", None) else 0.0
        fee_dollars = float(
            order_info.get("taker_fees_dollars")
            or order_info.get("maker_fees_dollars")
            or 0.0
        )
        if not gross_dollars and count > 0 and fallback_price_cents > 0:
            gross_dollars = count * fallback_price_cents / 100.0
        avg_price_cents = fallback_price_cents
        if gross_dollars > 0 and count > 0:
            avg_price_cents = max(1, min(99, round((gross_dollars / count) * 100)))
        signed_cost = -gross_dollars if action == "sell" else gross_dollars
        return count, avg_price_cents, signed_cost, fee_dollars

    def _compute_exit_pnl(self, ticker: str, side: str, count: int, exit_price_cents: int, exit_fee: float) -> tuple[float, str]:
        """Compute realized P&L for an exit using average filled entry cost/fees."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """SELECT SUM(count) as total_count, SUM(cost) as total_cost, SUM(fee) as total_fees,
                      MAX(market_title) as market_title
            FROM trades WHERE ticker = ? AND side = ? AND action = 'buy'
            AND status = 'filled' AND paper_mode = ?""",
            (ticker, side, 1 if self.paper_mode else 0),
        )
        entry_row = c.fetchone()
        conn.close()

        entry_cost_per_contract = 0.0
        entry_fee_per_contract = 0.0
        original_market_title = ""
        if entry_row and entry_row["total_count"] and entry_row["total_count"] > 0:
            entry_cost_per_contract = entry_row["total_cost"] / entry_row["total_count"]
            entry_fee_per_contract = entry_row["total_fees"] / entry_row["total_count"]
            original_market_title = entry_row["market_title"] or ""

        exit_price_per = exit_price_cents / 100.0
        pnl = (exit_price_per - entry_cost_per_contract) * count - (entry_fee_per_contract * count) - exit_fee
        return round(pnl, 4), original_market_title

    def _record_realized_daily_pnl(self, *, date: str, strategy: str, pnl: float, fee: float) -> None:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
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

    def classify_market_family(
        self,
        strategy: str,
        ticker: str = "",
        signal_source: str = "",
    ) -> str:
        """Map a candidate into a family with tailored quality rules."""
        source = (signal_source or "").lower()
        ticker_upper = (ticker or "").upper()

        if strategy == "weather":
            if source == "weather_observed_arbitrage":
                return "weather_observed"
            return "weather_forecast"

        if strategy == "sports":
            if source == "parlay_pricer":
                return "sports_parlay"
            soccer_markers = ("KXSL", "KXLIGA", "KXMLS", "KXEPL", "KXSERIE", "KXBUND", "KXLIGUE")
            if source == "sharp_single_game" and any(marker in ticker_upper for marker in soccer_markers):
                return "sports_single_soccer"
            return "sports_single"

        if strategy == "finance":
            if "bracket" in source:
                return "finance_bracket"
            return "finance_threshold"

        if strategy == "crypto":
            return "crypto_momentum"

        if strategy == "econ":
            return "econ"

        if strategy == "nba_props":
            return "nba_props"

        return "other"

    def get_event_key(self, ticker: str) -> str:
        """Collapse market tickers to an event key for concentration checks."""
        if not ticker:
            return ""
        return ticker.rsplit("-", 1)[0] if "-" in ticker else ticker

    def _get_family_rules(self, family: str) -> dict[str, float | int | bool]:
        return FAMILY_ENTRY_RULES.get(family, FAMILY_ENTRY_RULES["other"])

    def _get_family_lifetime_limits(self, family: str) -> dict[str, int | bool]:
        return FAMILY_LIFETIME_LIMITS.get(family, FAMILY_LIFETIME_LIMITS["other"])

    def load_performance_model(self, force_refresh: bool = False) -> dict[str, Any]:
        if self._performance_model_cache is None or force_refresh:
            try:
                self._performance_model_cache = load_performance_model()
                self._performance_model_loaded_at = time.time()
            except Exception:
                self._performance_model_cache = {}
        return self._performance_model_cache or {}

    def _get_buy_trade_lifetime_stats(self, ticker: str, event_key: str) -> dict[str, int]:
        """Count non-canceled buy attempts and contracts for ticker and event."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        c.execute(
            """SELECT COUNT(*), COALESCE(SUM(count), 0)
            FROM trades
            WHERE action = 'buy'
              AND status NOT IN ('blocked', 'error', 'timeout', 'canceled')
              AND ticker = ?""",
            (ticker,),
        )
        ticker_row = c.fetchone() or (0, 0)

        c.execute(
            """SELECT COUNT(*), COALESCE(SUM(count), 0)
            FROM trades
            WHERE action = 'buy'
              AND status NOT IN ('blocked', 'error', 'timeout', 'canceled')
              AND (ticker = ? OR ticker LIKE ?)""",
            (event_key, f"{event_key}-%"),
        )
        event_row = c.fetchone() or (0, 0)
        conn.close()

        return {
            "ticker_trades": int(ticker_row[0] or 0),
            "ticker_contracts": int(ticker_row[1] or 0),
            "event_trades": int(event_row[0] or 0),
            "event_contracts": int(event_row[1] or 0),
        }

    def check_lifetime_concentration_limits(
        self,
        strategy: str,
        ticker: str,
        signal_source: str,
        count: int,
    ) -> tuple[bool, str]:
        """Block repeated accumulation on the same ticker or event across cycles."""
        family = self.classify_market_family(strategy, ticker=ticker, signal_source=signal_source)
        limits = self._get_family_lifetime_limits(family)
        if limits.get("blocked"):
            return False, f"Family '{family}' is disabled until it proves positive edge"

        event_key = self.get_event_key(ticker)
        stats = self._get_buy_trade_lifetime_stats(ticker=ticker, event_key=event_key)

        for key in ("ticker_trades", "ticker_contracts", "event_trades", "event_contracts"):
            limit = int(limits.get(key, 0) or 0)
            projected = stats[key] + (count if "contracts" in key else 1)
            if limit and projected > limit:
                label = key.replace("_", " ")
                return False, f"{family} lifetime {label} cap exceeded for {ticker}"

        return True, "OK"

    def _get_recent_strategy_quality(self, strategy: str, lookback_days: int = 30) -> dict[str, float]:
        """Estimate whether a strategy has been earning the right to more weight."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            f"""SELECT COUNT(*) as trades,
                      COALESCE(SUM(pnl), 0) as total_pnl,
                      COALESCE(AVG(pnl), 0) as avg_pnl,
                      COALESCE(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), 0.0) as win_rate
               FROM trades
               WHERE {REALIZED_TRADE_SQL}
                 AND strategy = ?
                 AND settled_at >= datetime('now', ?)""",
            (strategy, f"-{lookback_days} days"),
        )
        row = c.fetchone()
        conn.close()
        return {
            "trades": float(row["trades"] or 0),
            "total_pnl": float(row["total_pnl"] or 0),
            "avg_pnl": float(row["avg_pnl"] or 0),
            "win_rate": float(row["win_rate"] or 0),
        }

    def _get_recent_source_quality(self, signal_source: str, lookback_days: int = 45) -> dict[str, float]:
        """Use settled entry trades to judge whether a source is calibrated."""
        if not signal_source:
            return {"trades": 0.0, "total_pnl": 0.0, "avg_pnl": 0.0, "win_rate": 0.0}

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """SELECT COUNT(*) as trades,
                      COALESCE(SUM(pnl), 0) as total_pnl,
                      COALESCE(AVG(pnl), 0) as avg_pnl,
                      COALESCE(AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END), 0.0) as win_rate
               FROM trades
               WHERE status = 'settled'
                 AND action = 'buy'
                 AND signal_source = ?
                 AND settled_at >= datetime('now', ?)""",
            (signal_source, f"-{lookback_days} days"),
        )
        row = c.fetchone()
        conn.close()
        return {
            "trades": float(row["trades"] or 0),
            "total_pnl": float(row["total_pnl"] or 0),
            "avg_pnl": float(row["avg_pnl"] or 0),
            "win_rate": float(row["win_rate"] or 0),
        }

    def evaluate_candidate_quality(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Score a candidate using family priors plus realized historical performance."""
        strategy = candidate.get("strategy", "")
        ticker = candidate.get("ticker", "")
        signal_source = candidate.get("signal_source", "")
        family = self.classify_market_family(strategy, ticker=ticker, signal_source=signal_source)
        rules = self._get_family_rules(family)

        edge = float(candidate.get("edge", 0) or 0)
        confidence = float(candidate.get("confidence", 0) or 0)
        our_prob = float(candidate.get("our_prob", 0) or 0)
        price_cents = int(candidate.get("price_cents", 0) or 0)
        reasons: list[str] = []

        if rules.get("blocked"):
            return {
                "allowed": False,
                "family": family,
                "quality_score": 0.0,
                "reasons": [f"{family} is disabled pending better evidence"],
            }

        min_edge = float(rules.get("min_edge", 0.0) or 0.0)
        min_confidence = float(rules.get("min_confidence", 0.0) or 0.0)
        min_win_prob = float(rules.get("min_win_prob", 0.0) or 0.0)

        if edge < min_edge:
            reasons.append(f"edge {edge:.1%} below {family} floor {min_edge:.1%}")
        if confidence < min_confidence:
            reasons.append(f"confidence {confidence:.1%} below {family} floor {min_confidence:.1%}")
        if our_prob < min_win_prob:
            reasons.append(f"win_prob {our_prob:.1%} below {family} floor {min_win_prob:.1%}")

        speculative_family = family not in {"weather_observed"}
        if speculative_family and (price_cents < 15 or price_cents > 85):
            reasons.append(f"price {price_cents}c too extreme for speculative family")

        strategy_quality = self._get_recent_strategy_quality(strategy)
        source_quality = self._get_recent_source_quality(signal_source)
        model = self.load_performance_model()

        strategy_multiplier = 1.0
        if strategy_quality["trades"] >= 12:
            if strategy_quality["avg_pnl"] < -1.0 or strategy_quality["win_rate"] < 0.42:
                strategy_multiplier = 0.70
                reasons.append(f"{strategy} recent realized performance is weak")
            elif strategy_quality["avg_pnl"] > 0.75 and strategy_quality["win_rate"] > 0.55:
                strategy_multiplier = 1.10

        source_multiplier = 1.0
        if source_quality["trades"] >= 10:
            if source_quality["avg_pnl"] < -1.0 or source_quality["win_rate"] < 0.42:
                source_multiplier = 0.65
                reasons.append(f"{signal_source} source performance is weak")
            elif source_quality["avg_pnl"] > 0.75 and source_quality["win_rate"] > 0.57:
                source_multiplier = 1.08

        model_family = model.get("family_multipliers", {}).get(family, {})
        model_source = model.get("signal_source_multipliers", {}).get(signal_source, {})
        model_bucket = model.get("price_bucket_multipliers", {}).get(_price_bucket(price_cents), {})
        blocked_tickers = set(model.get("blocked_tickers", []))
        blocked_events = set(model.get("blocked_events", []))
        blocked_families = set(model.get("blocked_families", []))
        blocked_sources = set(model.get("blocked_sources", []))
        current_event = self.get_event_key(ticker)
        if ticker in blocked_tickers:
            reasons.append(f"{ticker} is on the recent realized-loss blacklist")
        if current_event in blocked_events:
            reasons.append(f"{current_event} is overconcentrated in the export history")
        if family in blocked_families:
            reasons.append(f"{family} is auto-quarantined from recent realized losses")
        if signal_source and signal_source in blocked_sources:
            reasons.append(f"{signal_source} is auto-quarantined from recent realized losses")

        prior = FAMILY_QUALITY_PRIORS.get(family, FAMILY_QUALITY_PRIORS["other"])
        model_family_multiplier = float(model_family.get("multiplier", 1.0) or 1.0)
        model_source_multiplier = float(model_source.get("multiplier", 1.0) or 1.0)
        model_bucket_multiplier = float(model_bucket.get("multiplier", 1.0) or 1.0)
        quality_score = round(
            edge * confidence * prior * strategy_multiplier * source_multiplier
            * model_family_multiplier * model_source_multiplier * model_bucket_multiplier,
            6,
        )
        hard_failures = [r for r in reasons if "below" in r or "too extreme" in r]
        if (
            ticker in blocked_tickers
            or current_event in blocked_events
            or family in blocked_families
            or signal_source in blocked_sources
        ):
            hard_failures.append("blocked by performance model")

        return {
            "allowed": len(hard_failures) == 0,
            "family": family,
            "quality_score": quality_score,
            "reasons": reasons,
            "strategy_quality": strategy_quality,
            "source_quality": source_quality,
            "model_family": model_family,
            "model_source": model_source,
            "model_bucket": model_bucket,
        }

    def _get_strategy_exposure_cap(self, strategy: str, effective_bankroll: float) -> float:
        """Get the active per-strategy exposure cap used by risk checks."""
        try:
            dyn_alloc = self.get_dynamic_strategy_allocations()
            alloc_pct = dyn_alloc.get(strategy, self.max_strategy_exposure_pct)
        except Exception:
            alloc_pct = self.max_strategy_exposure_pct
        return effective_bankroll * alloc_pct

    def _init_db(self) -> None:
        """Initialize SQLite database for trade logging."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))

        # Enable Write-Ahead Logging to prevent database locking errors
        # when multiple concurrent tasks (weather, crypto, main) write.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

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
                notes TEXT DEFAULT '',
                thesis TEXT DEFAULT ''
            )
        """)

        # Migration: add thesis column to existing DBs
        try:
            c.execute("ALTER TABLE trades ADD COLUMN thesis TEXT DEFAULT ''")
            conn.commit()
        except Exception:
            pass  # Column already exists

        # Migration: add closing_price_cents for CLV tracking
        try:
            c.execute("ALTER TABLE trades ADD COLUMN closing_price_cents INTEGER DEFAULT 0")
            conn.commit()
        except Exception:
            pass  # Column already exists

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

        c.execute("""
            CREATE TABLE IF NOT EXISTS weather_api_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                city TEXT NOT NULL,
                source_name TEXT NOT NULL,
                market_type TEXT NOT NULL,
                forecast_temp REAL,
                actual_temp REAL,
                error REAL,
                target_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_weather_api_perf_source
            ON weather_api_performance(source_name, created_at)
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS candidate_rejections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                ticker TEXT DEFAULT '',
                signal_source TEXT DEFAULT '',
                stage TEXT DEFAULT '',
                reason TEXT NOT NULL,
                near_miss INTEGER NOT NULL DEFAULT 0,
                details TEXT DEFAULT ''
            )
        """)

        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidate_rejections_strategy
            ON candidate_rejections(strategy, timestamp)
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

    def record_candidate_rejection(
        self,
        strategy: str,
        reason: str,
        *,
        ticker: str = "",
        signal_source: str = "",
        stage: str = "",
        near_miss: bool = False,
        details: str = "",
    ) -> None:
        """Persist structured rejection reasons for later diagnosis."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            """INSERT INTO candidate_rejections
            (timestamp, strategy, ticker, signal_source, stage, reason, near_miss, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                strategy,
                ticker,
                signal_source,
                stage,
                reason,
                1 if near_miss else 0,
                details,
            ),
        )
        conn.commit()
        conn.close()

    def get_candidate_rejections(
        self,
        *,
        strategy: str | None = None,
        stage: str | None = None,
        near_miss_only: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        query = "SELECT * FROM candidate_rejections WHERE 1=1"
        params: list[Any] = []
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if stage:
            query += " AND stage = ?"
            params.append(stage)
        if near_miss_only:
            query += " AND near_miss = 1"
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        c.execute(query, params)
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_near_miss_summary(self, strategy: str, limit: int = 20) -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """SELECT stage, reason, COUNT(*) as count
               FROM candidate_rejections
               WHERE strategy = ?
                 AND near_miss = 1
                 AND timestamp >= datetime('now', '-3 days')
               GROUP BY stage, reason
               ORDER BY count DESC, stage ASC
               LIMIT ?""",
            (strategy, limit),
        )
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_rejection_summary_since(self, strategy: str, since_iso: str, limit: int = 10) -> list[dict[str, Any]]:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute(
            """SELECT stage, reason, COUNT(*) as count
               FROM candidate_rejections
               WHERE strategy = ?
                 AND timestamp >= ?
               GROUP BY stage, reason
               ORDER BY count DESC, stage ASC
               LIMIT ?""",
            (strategy, since_iso, limit),
        )
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows

    def get_weather_volume_diagnostics(self, days: int = 7) -> dict[str, Any]:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        def _fetch_value(query: str, params: tuple[Any, ...]) -> int:
            c.execute(query, params)
            row = c.fetchone()
            return int((row[0] if row else 0) or 0)

        window = f"-{max(days, 1)} days"
        params = (window,)

        scanned_cycles = _fetch_value(
            """SELECT COUNT(*)
               FROM agent_log
               WHERE strategy = 'weather'
                 AND message = 'Weather cycle starting'
                 AND timestamp >= datetime('now', ?)""",
            params,
        )
        candidate_cycles = _fetch_value(
            """SELECT COUNT(*)
               FROM agent_log
               WHERE strategy = 'weather'
                 AND message LIKE 'Weather cycle complete:%'
                 AND timestamp >= datetime('now', ?)""",
            params,
        )
        signals = _fetch_value(
            """SELECT COUNT(*)
               FROM signals
               WHERE strategy = 'weather'
                 AND timestamp >= datetime('now', ?)""",
            params,
        )
        attempted_trades = _fetch_value(
            """SELECT COUNT(*)
               FROM trades
               WHERE strategy = 'weather'
                 AND timestamp >= datetime('now', ?)""",
            params,
        )
        filled_trades = _fetch_value(
            """SELECT COUNT(*)
               FROM trades
               WHERE strategy = 'weather'
                 AND status IN ('filled', 'executed', 'settled')
                 AND timestamp >= datetime('now', ?)""",
            params,
        )
        open_positions = _fetch_value(
            """SELECT COUNT(DISTINCT ticker || '|' || side)
               FROM trades
               WHERE strategy = 'weather'
                 AND status = 'filled'
                 AND action = 'buy'
                 AND timestamp >= datetime('now', ?)""",
            params,
        )

        c.execute(
            """SELECT reason, stage, signal_source, COUNT(*) as count
               FROM candidate_rejections
               WHERE strategy = 'weather'
                 AND timestamp >= datetime('now', ?)
               GROUP BY reason, stage, signal_source
               ORDER BY count DESC, reason ASC
               LIMIT 20""",
            params,
        )
        blocker_breakdown = [dict(r) for r in c.fetchall()]

        c.execute(
            """SELECT ticker, COUNT(*) as count
               FROM candidate_rejections
               WHERE strategy = 'weather'
                 AND reason = 'no_two_sided_market'
                 AND timestamp >= datetime('now', ?)
               GROUP BY ticker
               ORDER BY count DESC, ticker ASC
               LIMIT 15""",
            params,
        )
        top_one_sided_tickers = [dict(r) for r in c.fetchall()]

        c.execute(
            """SELECT signal_source, COUNT(*) as count
               FROM candidate_rejections
               WHERE strategy = 'weather'
                 AND timestamp >= datetime('now', ?)
               GROUP BY signal_source
               ORDER BY count DESC, signal_source ASC""",
            params,
        )
        rejection_sources = [dict(r) for r in c.fetchall()]

        c.execute(
            """SELECT date(timestamp) as day,
                      COUNT(*) as rejections,
                      SUM(CASE WHEN near_miss = 1 THEN 1 ELSE 0 END) as near_misses
               FROM candidate_rejections
               WHERE strategy = 'weather'
                 AND timestamp >= datetime('now', ?)
               GROUP BY date(timestamp)
               ORDER BY day DESC""",
            params,
        )
        daily_rejections = [dict(r) for r in c.fetchall()]

        conn.close()

        return {
            "window_days": days,
            "funnel": {
                "weather_cycles_started": scanned_cycles,
                "weather_cycles_completed": candidate_cycles,
                "signals_recorded": signals,
                "trade_attempts": attempted_trades,
                "filled_or_settled_trades": filled_trades,
                "open_positions_seen": open_positions,
            },
            "top_blockers": blocker_breakdown,
            "rejection_sources": rejection_sources,
            "top_one_sided_tickers": top_one_sided_tickers,
            "daily_rejections": daily_rejections,
        }

    def get_recent_source_quality(self, signal_source: str, lookback_days: int = 45) -> dict[str, float]:
        """Public wrapper for source calibration stats used by adaptive runtime rules."""
        return self._get_recent_source_quality(signal_source, lookback_days=lookback_days)

    def _get_all_realized_pnl(self) -> float:
        """Sum of all realized P&L: settled trades + exited (sell) trades."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(f"""
            SELECT COALESCE(SUM(pnl), 0) FROM trades
            WHERE {REALIZED_TRADE_SQL}
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

        Uses net open-risk accounting:
        buy cost + all realized/open fees - filled sell proceeds.
        Pending/resting sell exits do not reduce exposure until they actually fill.
        """
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        if strategy:
            c.execute(
                """
                SELECT COALESCE(SUM(
                    CASE
                        WHEN action = 'buy' AND status IN ('filled', 'executed', 'resting', 'pending')
                            THEN cost + fee
                        WHEN action = 'sell' AND status IN ('filled', 'executed')
                            THEN -(cost - fee)
                        ELSE 0
                    END
                ), 0)
                FROM trades
                WHERE strategy = ?
                """,
                (strategy,),
            )
        else:
            c.execute(
                """
                SELECT COALESCE(SUM(
                    CASE
                        WHEN action = 'buy' AND status IN ('filled', 'executed', 'resting', 'pending')
                            THEN cost + fee
                        WHEN action = 'sell' AND status IN ('filled', 'executed')
                            THEN -(cost - fee)
                        ELSE 0
                    END
                ), 0)
                FROM trades
                """
            )
        result = float(c.fetchone()[0] or 0.0)
        conn.close()
        broker_exposure = self._broker_exposure_by_strategy.get(strategy, 0.0) if strategy else self._broker_total_exposure
        return max(0.0, result, broker_exposure)

    def has_open_position(self, ticker: str) -> bool:
        """Check if we already have an open (unsettled) position on this ticker."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("""
            SELECT
                SUM(CASE WHEN action = 'buy' THEN count ELSE 0 END)
              - SUM(CASE WHEN action = 'sell' THEN count ELSE 0 END) as net
            FROM trades
            WHERE status IN ('filled', 'resting', 'pending') AND ticker = ?
        """, (ticker,))
        row = c.fetchone()
        conn.close()
        db_has_position = row is not None and (row[0] or 0) > 0
        broker_has_position = self._broker_exposure_by_ticker.get(ticker, 0.0) > 0
        return db_has_position or broker_has_position

    def get_ticker_exposure(self, ticker: str) -> float:
        """Get net capital deployed on a specific ticker (buys minus sell proceeds)."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            """
            SELECT COALESCE(SUM(
                CASE
                    WHEN action = 'buy' AND status IN ('filled', 'executed', 'resting', 'pending')
                        THEN cost + fee
                    WHEN action = 'sell' AND status IN ('filled', 'executed')
                        THEN -(cost - fee)
                    ELSE 0
                END
            ), 0)
            FROM trades
            WHERE ticker = ?
            """,
            (ticker,),
        )
        result = float(c.fetchone()[0] or 0.0)
        conn.close()
        return max(0.0, result, self._broker_exposure_by_ticker.get(ticker, 0.0))

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

        # Per-strategy total exposure cap — use dynamic allocation if available
        strategy_exposure = self.get_total_exposure(strategy=strategy)
        max_strategy = self._get_strategy_exposure_cap(strategy, effective)
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

    def set_runtime_health(
        self,
        *,
        api_healthy: bool | None = None,
        db_healthy: bool | None = None,
        ws_healthy: bool | None = None,
    ) -> None:
        """Update runtime health flags sourced from the agent monitor loop."""
        if api_healthy is not None:
            self._runtime_api_healthy = bool(api_healthy)
        if db_healthy is not None:
            self._runtime_db_healthy = bool(db_healthy)
        if ws_healthy is not None:
            self._runtime_ws_healthy = bool(ws_healthy)

    def record_monitor_heartbeat(self) -> None:
        """Record monitor loop heartbeat for health checks."""
        self._last_monitor_heartbeat = datetime.now(UTC).isoformat()

    def _prune_order_failures(self) -> None:
        """Keep only recent order failures inside the configured time window."""
        cutoff = time.time() - (self.order_failure_window_mins * 60)
        self._recent_order_failures = [ts for ts in self._recent_order_failures if ts >= cutoff]

    def _record_order_failure(self) -> None:
        """Track live order failure and optionally auto-activate kill switch."""
        now_ts = time.time()
        self._recent_order_failures.append(now_ts)
        self._prune_order_failures()

        if self.enable_auto_kill_on_failures and len(self._recent_order_failures) >= self.max_order_failures_window:
            if not self.kill_switch:
                self.kill_switch = True
                self.log_event(
                    "critical",
                    f"Kill switch auto-activated: {len(self._recent_order_failures)} order failures "
                    f"in {self.order_failure_window_mins}m",
                    strategy="risk",
                )

    def _record_order_success(self) -> None:
        """Track successful order placements."""
        self._last_order_success_ts = time.time()
        self._prune_order_failures()

    def get_resting_order_count(self, strategy: str | None = None) -> int:
        """Get number of currently resting/pending orders."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        if strategy:
            c.execute(
                "SELECT COUNT(*) FROM trades WHERE status IN ('resting', 'pending') AND strategy = ?",
                (strategy,),
            )
        else:
            c.execute("SELECT COUNT(*) FROM trades WHERE status IN ('resting', 'pending')")
        result = int(c.fetchone()[0] or 0)
        conn.close()
        return result

    def get_guardrail_status(self) -> dict[str, Any]:
        """Summarize live-trading guardrail state for health/status endpoints."""
        self._prune_order_failures()
        return {
            "runtime_health": {
                "api_healthy": self._runtime_api_healthy,
                "db_healthy": self._runtime_db_healthy,
                "ws_healthy": self._runtime_ws_healthy,
                "last_monitor_heartbeat": self._last_monitor_heartbeat,
            },
            "resting_orders": {
                "total": self.get_resting_order_count(),
                "max_total": self.max_total_resting_orders,
                "max_per_strategy": self.max_resting_orders_per_strategy,
            },
            "order_failures": {
                "recent_failures": len(self._recent_order_failures),
                "window_mins": self.order_failure_window_mins,
                "max_before_kill": self.max_order_failures_window,
                "last_success_ts": self._last_order_success_ts,
            },
        }

    def _get_trade_count(self) -> int:
        """Get total number of buy trades placed."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM trades WHERE action = 'buy'")
        result = c.fetchone()[0]
        conn.close()
        return result

    def check_risk_limits(
        self,
        strategy: str,
        cost: float,
        ticker: str = "",
        count: int = 0,
        price_cents: int = 0,
    ) -> tuple[bool, str]:
        """
        Check if a trade passes all risk limits.
        Uses effective_bankroll (starting bankroll + realized P&L) for all calculations.
        Returns (allowed, reason).
        """
        estimated_fee = _kalshi_maker_fee(count, price_cents) if count > 0 and price_cents > 0 else 0.0
        total_risk = cost + estimated_fee

        if self.kill_switch:
            return False, "Kill switch is active"

        if not self.paper_mode:
            if not self._runtime_api_healthy:
                return False, "API health degraded — blocking new live trades"
            if not self._runtime_db_healthy:
                return False, "DB health degraded — blocking new live trades"
            if self.require_ws_for_live and not self._runtime_ws_healthy:
                return False, "WebSocket unhealthy (REQUIRE_WS_FOR_LIVE enabled)"

        # ── Circuit Breaker (2-Hour Cooldown) ──
        import time
        now_ts = time.time()
        if now_ts < self._cooldown_until:
            mins_left = int((self._cooldown_until - now_ts) / 60)
            return False, f"Circuit breaker cooldown ({mins_left}m remaining)"

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        effective = self.get_effective_bankroll()

        if today != self._last_cooldown_date:
            self._last_cooldown_date = today
            # Set initial drawdown threshold to -3% of bankroll for the day
            self._cooldown_pnl_threshold = -(effective * 0.03)

        today_pnl = self.get_today_pnl()
        if today_pnl <= self._cooldown_pnl_threshold:
            self._cooldown_until = now_ts + (2 * 3600)  # 2 hours
            # Lower the threshold by another 3% so it can trigger again if things get worse later
            self._cooldown_pnl_threshold -= (effective * 0.03)
            self.log_event("warning", f"CIRCUIT BREAKER TRIGGERED: Daily P&L hit ${today_pnl:.2f}. Pausing new trades for 2 hours.", strategy="risk")
            return False, f"Circuit breaker triggered at ${today_pnl:.2f}"
        # ───────────────────────────────────────

        strategy_active, strategy_reason = self.is_strategy_active(strategy)
        if not strategy_active:
            return False, strategy_reason or f"Strategy '{strategy}' is disabled"

        effective = self.get_effective_bankroll()
        if effective <= 0:
            return False, f"Effective bankroll is ${effective:.2f} — no capital available"

        # HARD GUARD: refuse all trades if bankroll too low
        if effective < self.min_bankroll_to_trade:
            return False, f"Bankroll ${effective:.2f} below minimum ${self.min_bankroll_to_trade:.2f} — all trading halted"

        # ABSOLUTE DOLLAR CAP: never risk more than max_single_trade_dollars on one trade
        if total_risk > self.max_single_trade_dollars:
            return False, f"Trade ${total_risk:.2f} exceeds absolute cap of ${self.max_single_trade_dollars:.2f}"

        # Check percentage of effective bankroll per bet
        if total_risk > effective * self.max_bet_pct:
            return False, f"Bet size ${total_risk:.2f} exceeds {self.max_bet_pct*100:.0f}% of effective bankroll (${effective:.2f})"

        # Check total exposure vs effective bankroll
        total_exposure = self.get_total_exposure()
        if total_exposure >= effective:
            return False, f"Over-deployed: ${total_exposure:.2f} deployed vs ${effective:.2f} effective bankroll"
        max_deployable = effective * self.max_total_exposure_pct
        if total_exposure + total_risk > max_deployable:
            return False, f"Total exposure ${total_exposure + total_risk:.2f} would exceed {self.max_total_exposure_pct*100:.0f}% of effective bankroll (${max_deployable:.2f})"

        # Check per-ticker position limit (scaled to effective bankroll)
        max_per_ticker = effective * 0.08
        if ticker:
            ticker_exposure = self.get_ticker_exposure(ticker)
            if ticker_exposure + total_risk > max_per_ticker:
                return False, f"Ticker {ticker} exposure ${ticker_exposure + total_risk:.2f} would exceed max ${max_per_ticker:.2f}"

        # Check per-strategy total exposure cap
        strategy_exposure = self.get_total_exposure(strategy=strategy)
        max_strategy = self._get_strategy_exposure_cap(strategy, effective)
        if strategy_exposure + total_risk > max_strategy:
            return False, f"Strategy '{strategy}' exposure ${strategy_exposure + total_risk:.2f} would exceed active cap ${max_strategy:.2f}"

        # Check per-strategy cycle deployment cap
        cycle_deployed = self._cycle_deployed.get(strategy, 0.0)
        max_cycle = effective * self.max_strategy_cycle_pct
        if cycle_deployed + total_risk > max_cycle:
            return False, f"Strategy '{strategy}' cycle cap: ${cycle_deployed + total_risk:.2f} would exceed ${max_cycle:.2f}/cycle"

        return True, "OK"

    def calculate_position_size(
        self,
        strategy: str,
        edge: float,
        price_cents: int,
        confidence: float = 0.5,
        ticker: str = "",
        signal_source: str = "",
    ) -> int:
        """Calculate position size using Kelly criterion with risk limits.

        CRITICAL: signal_source determines sizing tier.
        - "weather_observed_arbitrage": confirmed NOAA data → full Kelly caps
        - "weather_consensus" (forecast): speculative → 1/4 Kelly caps
        - Other strategies: normal caps
        """
        if edge <= 0 or price_cents <= 0:
            return 0

        bankroll = self.get_effective_bankroll()
        if bankroll <= 0:
            return 0

        # Half-Kelly: f = 0.5 * edge / (1 - implied_prob)
        # This scales position size with edge magnitude rather than using fixed fractions.
        implied_prob = price_cents / 100.0
        if implied_prob >= 0.99:
            implied_prob = 0.99  # prevent division by zero
        half_kelly = 0.5 * edge / (1.0 - implied_prob)

        # Safety caps by strategy, signal source, and confidence tier
        is_observed_arb = signal_source == "weather_observed_arbitrage"

        if strategy == "weather":
            if is_observed_arb:
                # OBSERVED ARBITRAGE: confirmed NOAA data — higher caps justified
                if confidence >= 0.85:
                    max_kelly = 0.08
                elif confidence >= 0.75:
                    max_kelly = 0.06
                else:
                    max_kelly = 0.04
            else:
                # FORECAST: speculative — drastically lower caps (1/4 of observed)
                if confidence >= 0.85:
                    max_kelly = 0.02
                elif confidence >= 0.75:
                    max_kelly = 0.015
                else:
                    max_kelly = 0.01
        else:
            if confidence >= 0.85:
                max_kelly = 0.10
            elif confidence >= 0.75:
                max_kelly = 0.08
            else:
                max_kelly = 0.05

        kelly_fraction = min(half_kelly, max_kelly)
        kelly_size = bankroll * kelly_fraction

        remaining = self.get_remaining_capital(strategy=strategy)
        effective = self.get_effective_bankroll()

        # Determine dollar cap: observed arbitrage gets full cap, forecast gets 1/4
        if strategy == "weather" and not is_observed_arb:
            trade_dollar_cap = min(self.max_single_trade_dollars, 15.0)  # forecast: hard $15 cap
        else:
            trade_dollar_cap = self.max_single_trade_dollars  # observed arb / other: $50 cap

        # Cap by remaining capital, Kelly, effective bankroll percentage, AND absolute dollar cap
        max_dollars = min(
            remaining,
            kelly_size,
            effective * self.max_bet_pct,
            trade_dollar_cap,
        )

        # Also cap by per-ticker limit (scaled to effective bankroll)
        if ticker:
            ticker_exposure = self.get_ticker_exposure(ticker)
            max_per_ticker = effective * 0.08
            ticker_remaining = max(0, max_per_ticker - ticker_exposure)
            max_dollars = min(max_dollars, ticker_remaining)

        # Low-bankroll guard: refuse if below minimum
        if effective < self.min_bankroll_to_trade:
            return 0

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

        if not self.paper_mode:
            resting_total = self.get_resting_order_count()
            if resting_total >= self.max_total_resting_orders:
                reason = f"Resting-order cap reached ({resting_total}/{self.max_total_resting_orders})"
                self.log_event("blocked", reason, strategy=strategy)
                return ""

            strategy_resting = self.get_resting_order_count(strategy=strategy)
            if strategy_resting >= self.max_resting_orders_per_strategy:
                reason = (
                    f"Strategy resting-order cap reached for {strategy} "
                    f"({strategy_resting}/{self.max_resting_orders_per_strategy})"
                )
                self.log_event("blocked", reason, strategy=strategy)
                return ""

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

    def get_signal_details(self, ticker: str, strategy: str = "") -> str:
        """Look up the most recent signal details for a given ticker."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        query = "SELECT details FROM signals WHERE ticker = ?"
        params: list[Any] = [ticker]
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        query += " ORDER BY timestamp DESC LIMIT 1"
        c.execute(query, params)
        row = c.fetchone()
        conn.close()
        return row[0] if row else ""

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
        thesis: str = "",
    ) -> dict[str, Any]:
        """
        Execute a trade (paper or live).
        Returns trade record dict.
        """
        cost = count * price_cents / 100.0
        fee = _kalshi_maker_fee(count, price_cents)
        edge = abs(our_prob - kalshi_prob)

        if count <= 0:
            return {"status": "blocked", "reason": "Invalid contract count"}

        if price_cents <= 0 or price_cents >= 100:
            return {"status": "blocked", "reason": "Invalid price"}

        if signal_source not in {"monitor_add", "monitor_reprice"} and self.has_open_position(ticker):
            reason = f"Open position already exists for {ticker}"
            self.log_event("blocked", reason, strategy=strategy)
            return {"status": "blocked", "reason": reason}

        if signal_source not in {"monitor_add", "monitor_reprice"}:
            allowed, reason = self.check_lifetime_concentration_limits(
                strategy=strategy,
                ticker=ticker,
                signal_source=signal_source,
                count=count,
            )
            if not allowed:
                self.log_event("blocked", reason, strategy=strategy)
                return {"status": "blocked", "reason": reason}

        if not self.paper_mode:
            resting_total = self.get_resting_order_count()
            if resting_total >= self.max_total_resting_orders:
                reason = f"Resting-order cap reached ({resting_total}/{self.max_total_resting_orders})"
                self.log_event("blocked", reason, strategy=strategy)
                return {"status": "blocked", "reason": reason}

            strategy_resting = self.get_resting_order_count(strategy=strategy)
            if strategy_resting >= self.max_resting_orders_per_strategy:
                reason = (
                    f"Strategy resting-order cap reached for {strategy} "
                    f"({strategy_resting}/{self.max_resting_orders_per_strategy})"
                )
                self.log_event("blocked", reason, strategy=strategy)
                return {"status": "blocked", "reason": reason}

        # Risk check
        allowed, reason = self.check_risk_limits(
            strategy,
            cost,
            ticker=ticker,
            count=count,
            price_cents=price_cents,
        )
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
            # Live trade — place on Kalshi as a MAKER order
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
                order_info = result.get("order", {})
                status = order_info.get("status", "pending")

                # If it's resting on the book, mark it as 'resting' so the monitor loop can track it
                if status in ("resting", "pending"):
                    status = "resting"
                    self._record_order_success()
                    self.log_event(
                        "live_trade",
                        f"MAKER ORDER PLACED: {side.upper()} {count}x {ticker} @ {price_cents}c order_id={order_id}",
                        strategy=strategy,
                    )
                elif status == "filled":
                    count, price_cents, cost, fee = self._extract_actual_fill_execution(
                        order_info,
                        fallback_count=count,
                        fallback_price_cents=price_cents,
                        action="buy",
                    )
                    self._record_order_success()
                    self.log_event("live_trade", f"Order filled immediately: {count}x @ {price_cents}c", strategy=strategy)
                elif status in ("canceled", "error"):
                    self._record_order_failure()
                    self.log_event("warning", f"Order {status}: {order_id}", strategy=strategy)
                    return {"status": status, "reason": f"Order {status}"}

            except Exception as e:
                self._record_order_failure()
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
            "thesis": thesis,
        }

        # Send Discord notification (fire and forget)
        from app.services.discord_webhook import send_discord_notification
        mode = "PAPER" if self.paper_mode else "LIVE"
        title = f"[{mode}] Trade Executed: {strategy.upper()}"
        message = f"**{side.upper()}** {count}x `{ticker}` @ {price_cents}¢\nCost: ${cost:.2f} (Fee: ${fee:.2f})\nEdge: {edge:.1%}\n\n**Notes:** {notes}"
        color = 0x2ecc71 if side == "yes" else 0xe74c3c
        _schedule_background_coro(send_discord_notification(title, message, color))

        try:
            conn = sqlite3.connect(str(DB_PATH))
            c = conn.cursor()
            c.execute(
                """INSERT INTO trades
                (id, timestamp, strategy, ticker, market_title, side, action, count,
                 price_cents, cost, fee, order_type, paper_mode, order_id, status,
                 our_prob, kalshi_prob, edge, signal_source, notes, thesis)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    thesis,
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
        except Exception as e:
            self.log_event("error", f"DB INSERT failed for trade {trade_id}: {e}", strategy=strategy)
            logger.error(f"Trade DB insert failed: {e}", trade_id=trade_id, ticker=ticker)

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

        # Update daily P&L on the actual settlement date so daily guards see real losses.
        date = now[:10]
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

        # Send Discord notification
        from app.services.discord_webhook import send_discord_notification
        mode = "PAPER" if trade.get("paper_mode") else "LIVE"
        title = f"[{mode}] Trade Settled: {strategy.upper()}"
        message = f"**{trade['ticker']}**\nResult: **{result.upper()}**\n**P&L:** ${pnl:+.2f}"
        color = 0x2ecc71 if pnl > 0 else 0xe74c3c
        _schedule_background_coro(send_discord_notification(title, message, color))

        return {"trade_id": trade_id, "result": result, "pnl": pnl}

    def record_closing_price(self, trade_id: str, closing_price_cents: int) -> None:
        """Record the market closing price for a trade (for CLV computation)."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("UPDATE trades SET closing_price_cents = ? WHERE id = ?", (closing_price_cents, trade_id))
        conn.commit()
        conn.close()

    def get_clv_stats(self, strategy: str = "", limit: int = 100) -> dict[str, Any]:
        """Compute CLV statistics across settled trades.
        CLV = (closing_price - entry_price) for YES bets, inverted for NO.
        Positive CLV means we consistently bought before the market moved our way.
        """
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        query = """
            SELECT side, price_cents, closing_price_cents, strategy, pnl
            FROM trades
            WHERE status = 'settled' AND action = 'buy' AND closing_price_cents > 0
        """
        params: list[Any] = []
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        query += " ORDER BY settled_at DESC LIMIT ?"
        params.append(limit)
        c.execute(query, params)
        rows = c.fetchall()
        conn.close()

        if not rows:
            return {"total_trades": 0, "avg_clv_cents": 0, "positive_clv_pct": 0, "by_strategy": {}}

        clv_values: list[float] = []
        by_strategy: dict[str, list[float]] = {}
        for r in rows:
            entry = r["price_cents"]
            close = r["closing_price_cents"]
            if r["side"] == "yes":
                clv = close - entry  # Positive = market moved toward YES after we bought YES
            else:
                clv = entry - close  # Positive = market moved toward NO after we bought NO (price dropped)
            clv_values.append(clv)
            strat = r["strategy"]
            by_strategy.setdefault(strat, []).append(clv)

        avg_clv = sum(clv_values) / len(clv_values) if clv_values else 0
        positive_count = sum(1 for v in clv_values if v > 0)

        strat_stats = {}
        for strat, vals in by_strategy.items():
            strat_stats[strat] = {
                "count": len(vals),
                "avg_clv_cents": round(sum(vals) / len(vals), 1) if vals else 0,
                "positive_clv_pct": round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1) if vals else 0,
            }

        return {
            "total_trades": len(clv_values),
            "avg_clv_cents": round(avg_clv, 1),
            "positive_clv_pct": round(positive_count / len(clv_values) * 100, 1) if clv_values else 0,
            "by_strategy": strat_stats,
        }

    def get_unsettled_trades(self) -> list[dict[str, Any]]:
        """Get all unsettled (filled) trades grouped by ticker for settlement checking."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT id, ticker, market_title, side, action, count, price_cents,
                   cost, fee, strategy, our_prob, kalshi_prob, edge,
                   signal_source, notes, thesis
            FROM trades
            WHERE status = 'filled' AND action = 'buy'
            ORDER BY ticker
        """)
        trades = [dict(r) for r in c.fetchall()]
        conn.close()
        return trades

    def get_resting_trades(self) -> list[dict[str, Any]]:
        """Get all resting (unfilled limit) orders."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT id, timestamp, strategy, ticker, side, action, count, price_cents,
                   order_id, status
            FROM trades
            WHERE status IN ('resting', 'pending')
        """)
        trades = [dict(r) for r in c.fetchall()]
        conn.close()
        return trades

    def update_trade_status(
        self,
        trade_id: str,
        status: str,
        filled_count: int | None = None,
        cost: float | None = None,
        fee: float | None = None,
        price_cents: int | None = None,
    ) -> None:
        """Update the status of a resting trade (e.g. to 'filled' or 'canceled')."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
        row = c.fetchone()
        if not row:
            conn.close()
            return
        trade = dict(row)
        if filled_count is not None and cost is not None and fee is not None:
            actual_price_cents = price_cents if price_cents is not None else int(trade.get("price_cents", 0) or 0)
            query = "UPDATE trades SET status = ?, count = ?, cost = ?, fee = ?, price_cents = ?"
            params: list[Any] = [status, filled_count, cost, fee, actual_price_cents]
            if status == "filled" and trade.get("action") == "sell":
                pnl, market_title = self._compute_exit_pnl(
                    trade["ticker"],
                    trade["side"],
                    filled_count,
                    actual_price_cents,
                    fee,
                )
                query += ", pnl = ?, settled_at = ?, result = ?, market_title = ?"
                params.extend([pnl, datetime.now(UTC).isoformat(), "exit", market_title])
            query += " WHERE id = ?"
            params.append(trade_id)
            c.execute(query, params)
        else:
            c.execute("UPDATE trades SET status = ? WHERE id = ?", (status, trade_id))
        conn.commit()
        conn.close()

        if filled_count is not None and cost is not None and fee is not None and status == "filled" and trade.get("action") == "sell":
            actual_price_cents = price_cents if price_cents is not None else int(trade.get("price_cents", 0) or 0)
            pnl, _ = self._compute_exit_pnl(
                trade["ticker"],
                trade["side"],
                filled_count,
                actual_price_cents,
                fee,
            )
            self._record_realized_daily_pnl(
                date=datetime.now(UTC).isoformat()[:10],
                strategy=trade["strategy"],
                pnl=pnl,
                fee=fee,
            )

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
            WHERE status IN ('filled', 'executed')
            GROUP BY ticker, side, paper_mode
            HAVING ABS(SUM(CASE WHEN action = 'buy' THEN count ELSE 0 END)
                 - SUM(CASE WHEN action = 'sell' THEN count ELSE 0 END)) > 0
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
        import traceback
        logger.info(
            "exit_trade called",
            ticker=ticker, side=side, count=count, price_cents=price_cents,
            reason=reason, strategy=strategy,
            caller="".join(traceback.format_stack()[-3:-1])[:300],
        )
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
                    # Kalshi API quirk: when selling NO, we must specify the YES price
                    # e.g., if we want to sell NO for 1c, we set yes_price = 99
                    yes_price_equivalent = 100 - price_cents
                    result = await client.place_order(
                        ticker=ticker, side="no", action="sell",
                        count=count, type="limit", yes_price=yes_price_equivalent,
                    )
                order_id = result.get("order", {}).get("order_id", "")
                order_info = result.get("order", {})
                status = order_info.get("status", "pending")

                if status in ("resting", "pending"):
                    status = "resting"
                    self.log_event(
                        "live_trade",
                        f"MAKER EXIT PLACED: {side.upper()} {count}x {ticker} @ {price_cents}c order_id={order_id}",
                        strategy=strategy,
                    )
                elif status == "filled":
                    count, price_cents, signed_cost, fee = self._extract_actual_fill_execution(
                        order_info,
                        fallback_count=count,
                        fallback_price_cents=price_cents,
                        action="sell",
                    )
                    cost = abs(signed_cost)
                elif status in ("canceled", "error"):
                    self.log_event("warning", f"Exit order {status}: {order_id}", strategy=strategy)
                    return {"status": status, "reason": f"Exit order {status}"}

            except Exception as e:
                self.log_event("error", f"Exit order failed: {e}", strategy=strategy)
                return {"status": "error", "reason": str(e)}

        # ── Compute realized P&L from entry price ──
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            c = conn.cursor()

            # ── DUPLICATE SELL PREVENTION ──
            # Check if we already have a filled sell for this ticker+side+count
            # This prevents the reconciliation bug that created 297 phantom duplicate sells
            c.execute(
                """SELECT COUNT(*) FROM trades
                WHERE ticker = ? AND side = ? AND action = 'sell'
                AND status = 'filled' AND paper_mode = ?""",
                (ticker, side, 1 if self.paper_mode else 0),
            )
            c.fetchone()[0]

            # Check total buy count to see if we've already fully exited
            c.execute(
                """SELECT SUM(count) as buy_count FROM trades
                WHERE ticker = ? AND side = ? AND action = 'buy'
                AND status = 'filled' AND paper_mode = ?""",
                (ticker, side, 1 if self.paper_mode else 0),
            )
            buy_row = c.fetchone()
            total_buy_count = buy_row[0] if buy_row and buy_row[0] else 0

            c.execute(
                """SELECT
                       SUM(CASE WHEN status = 'filled' THEN count ELSE 0 END) as filled_sell_count,
                       SUM(CASE WHEN status IN ('resting', 'pending') THEN count ELSE 0 END) as open_exit_count
                   FROM trades
                   WHERE ticker = ? AND side = ? AND action = 'sell'
                     AND paper_mode = ?""",
                (ticker, side, 1 if self.paper_mode else 0),
            )
            sell_row = c.fetchone()
            filled_sell_count = sell_row["filled_sell_count"] if sell_row and sell_row["filled_sell_count"] else 0
            open_exit_count = sell_row["open_exit_count"] if sell_row and sell_row["open_exit_count"] else 0
            total_committed_exit_count = filled_sell_count + open_exit_count

            # If we've already sold or placed exits for as much as we bought, skip this duplicate exit.
            if total_committed_exit_count >= total_buy_count:
                conn.close()
                self.log_event(
                    "warning",
                    f"Skipping duplicate exit: already committed exits for {total_committed_exit_count}/{total_buy_count} contracts for {ticker} {side}",
                    strategy=strategy,
                )
                return {"status": "skipped", "reason": "Position already fully exited"}

            pnl = 0.0
            original_market_title = ""
            if status == "filled":
                pnl, original_market_title = self._compute_exit_pnl(
                    ticker,
                    side,
                    count,
                    price_cents,
                    fee,
                )
            else:
                _, original_market_title = self._compute_exit_pnl(
                    ticker,
                    side,
                    count,
                    price_cents,
                    fee,
                )

            now = datetime.now(UTC).isoformat()

            # Send Discord notification
            from app.services.discord_webhook import send_discord_notification
            mode = "PAPER" if self.paper_mode else "LIVE"
            title = f"[{mode}] Trade Exited: {strategy.upper()}"
            message = f"**{sell_action.upper()} {side.upper()}** {count}x `{ticker}` @ {price_cents}¢\nProceeds: ${cost:.2f} (Fee: ${fee:.2f})\n**P&L:** ${pnl:+.2f}\n\n**Reason:** {reason}"
            color = 0x3498db
            _schedule_background_coro(send_discord_notification(title, message, color))

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
                "pnl": round(pnl, 4) if status == "filled" else 0.0,
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
                    "exit" if status == "filled" else "", round(pnl, 4) if status == "filled" else 0.0, now if status == "filled" else "",
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            self.log_event("error", f"DB INSERT failed for exit trade {trade_id}: {e}", strategy=strategy)
            logger.error(f"Exit trade DB insert failed: {e}", trade_id=trade_id, ticker=ticker)
            return {"status": "error", "reason": f"DB insert failed: {e}"}

        if status == "filled":
            self._record_realized_daily_pnl(
                date=now[:10],
                strategy=strategy,
                pnl=round(pnl, 4),
                fee=fee,
            )

        self.log_event(
            "exit_trade",
            f"EXIT {side.upper()} {count}x {ticker} @ {price_cents}c | P&L=${pnl:+.2f} | {reason}",
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
        c.execute(f"""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(SUM(fee), 0) as total_fees,
                COALESCE(SUM(CASE WHEN action = 'buy' THEN cost ELSE 0 END), 0) as total_wagered
            FROM trades WHERE {REALIZED_TRADE_SQL}
        """)
        overall = dict(c.fetchone())

        # Per-strategy stats
        c.execute(f"""
            SELECT
                strategy,
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                COALESCE(SUM(pnl), 0) as total_pnl,
                COALESCE(SUM(fee), 0) as total_fees
            FROM trades WHERE {REALIZED_TRADE_SQL}
            GROUP BY strategy
        """)
        by_strategy = {row["strategy"]: dict(row) for row in c.fetchall()}

        # Win rate by price bucket (10c buckets) — key diagnostic for model calibration
        c.execute(f"""
            SELECT
                CASE
                    WHEN price_cents < 10 THEN '01-09c'
                    WHEN price_cents < 20 THEN '10-19c'
                    WHEN price_cents < 30 THEN '20-29c'
                    WHEN price_cents < 40 THEN '30-39c'
                    WHEN price_cents < 50 THEN '40-49c'
                    WHEN price_cents < 60 THEN '50-59c'
                    WHEN price_cents < 70 THEN '60-69c'
                    WHEN price_cents < 80 THEN '70-79c'
                    WHEN price_cents < 90 THEN '80-89c'
                    ELSE '90-99c'
                END as bucket,
                strategy,
                side,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                COALESCE(SUM(pnl), 0) as total_pnl
            FROM trades WHERE {REALIZED_TRADE_SQL}
            GROUP BY bucket, strategy, side
            ORDER BY strategy, bucket
        """)
        by_price_bucket = [dict(r) for r in c.fetchall()]

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
            "allowed_live_strategies": sorted(self.allowed_live_strategies),
            "allowed_paper_strategies": sorted(self.allowed_paper_strategies),
            "overall": {
                **overall,
                "win_rate": wins / total_trades if total_trades > 0 else 0,
                "roi": overall["total_pnl"] / overall["total_wagered"] if overall["total_wagered"] else 0,
            },
            "by_strategy": by_strategy,
            "by_price_bucket": by_price_bucket,
            "daily_pnl": daily,
            "today_pnl": self.get_today_pnl(),
            "today_trades": self.get_today_trade_count(),
        }

    def get_dynamic_strategy_allocations(self) -> dict[str, float]:
        """Compute dynamic capital allocation per strategy based on 7-day rolling Sharpe-like ratio.
        Higher risk-adjusted return strategies get more capital. Minimum 10% floor per enabled strategy.
        Returns: {strategy: allocation_pct} where values sum to ~1.0.
        """
        import math as _math
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Get 7-day rolling P&L by strategy
        c.execute(f"""
            SELECT strategy,
                   COUNT(*) as cnt,
                   COALESCE(SUM(pnl), 0) as total_pnl,
                   COALESCE(AVG(pnl), 0) as avg_pnl
            FROM trades
            WHERE {REALIZED_TRADE_SQL}
              AND settled_at >= datetime('now', '-7 days')
            GROUP BY strategy
        """)
        rows = c.fetchall()

        # Also get variance for Sharpe-like calculation
        strat_data: dict[str, dict] = {}
        for r in rows:
            strat = r["strategy"]
            strat_data[strat] = {
                "count": r["cnt"],
                "total_pnl": r["total_pnl"],
                "avg_pnl": r["avg_pnl"],
            }

        # Compute per-trade pnl std dev for each strategy
        for strat in strat_data:
            c.execute(f"""
                SELECT pnl FROM trades
                WHERE {REALIZED_TRADE_SQL} AND strategy = ?
                  AND settled_at >= datetime('now', '-7 days')
            """, (strat,))
            pnls = [r["pnl"] for r in c.fetchall()]
            if len(pnls) > 1:
                mean = sum(pnls) / len(pnls)
                variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
                std = _math.sqrt(variance) if variance > 0 else 0.01
            else:
                std = 0.01
            strat_data[strat]["std"] = std
            # Sharpe-like: avg_pnl / std (higher = better risk-adjusted return)
            strat_data[strat]["sharpe"] = strat_data[strat]["avg_pnl"] / std if std > 0 else 0

        conn.close()

        # Get enabled strategies
        enabled = [s for s, on in self.strategy_enabled.items() if on]
        if not enabled:
            return {}

        # Assign allocations based on Sharpe ratio
        min_floor = 0.10  # 10% minimum per strategy
        remaining = 1.0 - (min_floor * len(enabled))
        if remaining < 0:
            # Too many strategies — equal allocation
            return {s: 1.0 / len(enabled) for s in enabled}

        # Score each enabled strategy
        scores: dict[str, float] = {}
        for s in enabled:
            data = strat_data.get(s, {})
            sharpe = data.get("sharpe", 0)
            # Boost positive Sharpe, dampen negative
            scores[s] = max(sharpe, 0) + 0.1  # 0.1 baseline so new strategies get some allocation

        total_score = sum(scores.values())
        allocations: dict[str, float] = {}
        for s in enabled:
            bonus = (scores[s] / total_score * remaining) if total_score > 0 else (remaining / len(enabled))
            # HARD CAP: no single strategy can exceed max_strategy_exposure_pct regardless of dynamic allocation
            allocations[s] = round(min(min_floor + bonus, self.max_strategy_exposure_pct), 3)

        return allocations

    def get_status(self) -> dict[str, Any]:
        """Get current agent status."""
        effective = self.get_effective_bankroll()
        total_exposure = self.get_total_exposure()
        max_deployable = effective * self.max_total_exposure_pct
        remaining = max_deployable - total_exposure
        return {
            "paper_mode": self.paper_mode,
            "kill_switch": self.kill_switch,
            "bankroll": self.bankroll,
            "effective_bankroll": round(effective, 2),
            "strategy_enabled": self.strategy_enabled,
            "allowed_live_strategies": sorted(self.allowed_live_strategies),
            "allowed_paper_strategies": sorted(self.allowed_paper_strategies),
            "max_bet_size": round(effective * self.max_bet_pct, 2),
            "today_pnl": self.get_today_pnl(),
            "today_trades": self.get_today_trade_count(),
            "total_exposure": round(total_exposure, 2),
            "max_deployable": round(max_deployable, 2),
            "remaining_capital": round(remaining, 2),
            "over_deployed": total_exposure > effective,
            "guardrails": self.get_guardrail_status(),
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
        _load_env_defaults()
        paper_mode = os.environ.get("PAPER_MODE", "true").lower() != "false"
        bankroll = float(os.environ.get("BANKROLL", "2000"))
        _engine = TradingEngine(paper_mode=paper_mode, bankroll=bankroll)
    return _engine
