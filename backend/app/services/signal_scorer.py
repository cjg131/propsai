"""
Signal Quality Scorer.

Tracks which signal components (momentum, funding, mean reversion, etc.)
correlate with wins vs losses. Dynamically adjusts signal weights based on
historical performance.

Usage:
    scorer = get_signal_scorer()
    scorer.record_signal_outcome("crypto", {"momentum_5m": 0.6, "mean_reversion": -0.3}, won=True)
    weights = scorer.get_dynamic_weights("crypto")
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "trading_engine.db"

# Base weights per strategy (used when no data exists)
BASE_WEIGHTS: dict[str, dict[str, float]] = {
    "crypto": {
        "momentum_5m": 0.40,
        "momentum_1m": 0.20,
        "funding_signal": 0.10,
        "mean_reversion": 0.15,
        "volatility": 0.15,
    },
    "finance": {
        "intraday_momentum": 0.30,
        "futures_signal": 0.25,
        "vix_signal": 0.20,
        "ma_signal": 0.15,
        "volume_signal": 0.10,
    },
}

# Minimum trades needed before adjusting weights
MIN_TRADES_FOR_ADJUSTMENT = 20
# How much to adjust (damping factor)
ADJUSTMENT_RATE = 0.3


class SignalScorer:
    """Tracks signal component performance and computes dynamic weights."""

    def __init__(self) -> None:
        self._init_db()
        self._weight_cache: dict[str, dict[str, float]] = {}

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                signal_name TEXT NOT NULL,
                signal_value REAL NOT NULL,
                value_bucket TEXT NOT NULL,
                won INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_signal_perf_strategy
            ON signal_performance(strategy, signal_name)
        """)
        conn.commit()
        conn.close()

    def _bucket(self, value: float) -> str:
        """Bucket a signal value for aggregation."""
        if value > 0.5:
            return "strong_positive"
        elif value > 0.1:
            return "weak_positive"
        elif value > -0.1:
            return "neutral"
        elif value > -0.5:
            return "weak_negative"
        else:
            return "strong_negative"

    def record_signal_outcome(
        self,
        strategy: str,
        signal_components: dict[str, float],
        won: bool,
    ) -> None:
        """Record the outcome of a trade with its signal component values."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        now = datetime.now(UTC).isoformat()

        for name, value in signal_components.items():
            if value is None:
                continue
            bucket = self._bucket(value)
            c.execute(
                """INSERT INTO signal_performance
                (strategy, signal_name, signal_value, value_bucket, won, created_at)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (strategy, name, value, bucket, 1 if won else 0, now),
            )

        conn.commit()
        conn.close()

        # Invalidate cache
        self._weight_cache.pop(strategy, None)

    def get_component_stats(self, strategy: str) -> dict[str, dict[str, Any]]:
        """Get win rate stats per signal component."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute(
            """SELECT signal_name,
                      COUNT(*) as total,
                      SUM(won) as wins,
                      AVG(signal_value) as avg_value
            FROM signal_performance
            WHERE strategy = ?
            GROUP BY signal_name""",
            (strategy,),
        )

        stats = {}
        for row in c.fetchall():
            total = row["total"]
            wins = row["wins"]
            stats[row["signal_name"]] = {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": round(wins / total, 3) if total > 0 else 0.5,
                "avg_value": round(row["avg_value"], 4),
            }

        conn.close()
        return stats

    def get_dynamic_weights(self, strategy: str) -> dict[str, float]:
        """
        Compute dynamic weights based on component win rates.
        Components with higher win rates get more weight.
        """
        if strategy in self._weight_cache:
            return self._weight_cache[strategy]

        base = BASE_WEIGHTS.get(strategy, {})
        if not base:
            return {}

        stats = self.get_component_stats(strategy)
        total_trades = sum(s["total"] for s in stats.values()) // max(len(stats), 1)

        if total_trades < MIN_TRADES_FOR_ADJUSTMENT:
            self._weight_cache[strategy] = dict(base)
            return dict(base)

        # Adjust weights: multiply base weight by (component_WR / 0.50)
        adjusted: dict[str, float] = {}
        for name, base_weight in base.items():
            component = stats.get(name)
            if component and component["total"] >= 5:
                wr = component["win_rate"]
                # Scale: WR=0.5 → 1.0x, WR=0.7 → 1.4x, WR=0.3 → 0.6x
                multiplier = wr / 0.50
                # Dampen the adjustment
                dampened = 1.0 + (multiplier - 1.0) * ADJUSTMENT_RATE
                adjusted[name] = base_weight * dampened
            else:
                adjusted[name] = base_weight

        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: round(v / total, 4) for k, v in adjusted.items()}

        self._weight_cache[strategy] = adjusted

        logger.info(
            "Dynamic weights computed",
            strategy=strategy,
            weights=adjusted,
            trades=total_trades,
        )
        return adjusted

    def get_all_stats(self) -> dict[str, Any]:
        """Get all signal performance stats."""
        result = {}
        for strategy in BASE_WEIGHTS:
            result[strategy] = {
                "components": self.get_component_stats(strategy),
                "dynamic_weights": self.get_dynamic_weights(strategy),
                "base_weights": BASE_WEIGHTS[strategy],
            }
        return result


# Singleton
_scorer: SignalScorer | None = None


def get_signal_scorer() -> SignalScorer:
    global _scorer
    if _scorer is None:
        _scorer = SignalScorer()
    return _scorer
