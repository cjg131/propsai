"""
Adaptive Threshold Tuning.

Automatically adjusts min_edge and min_confidence per strategy based on
rolling win rate. Tightens thresholds when losing, loosens when winning.

Rules:
  - WR < 40% over last 50 trades → tighten min_edge +2%, min_confidence +5%
  - WR 40-55% → no change
  - WR > 60% → loosen min_edge -1%, min_confidence -2%
  - Floor: min_edge >= 3%, min_confidence >= 15%
  - Ceiling: min_edge <= 15%, min_confidence <= 50%
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "trading_engine.db"

# Default thresholds per strategy
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "weather": {"min_edge": 0.08, "min_confidence": 0.30},
    "sports": {"min_edge": 0.03, "min_confidence": 0.85},
    "crypto": {"min_edge": 0.08, "min_confidence": 0.20},
    "finance": {"min_edge": 0.05, "min_confidence": 0.25},
    "econ": {"min_edge": 0.05, "min_confidence": 0.25},
    "nba_props": {"min_edge": 0.05, "min_confidence": 0.30},
}

# Hard limits
MIN_EDGE_FLOOR = 0.03
MIN_EDGE_CEILING = 0.15
MIN_CONF_FLOOR = 0.15
MIN_CONF_CEILING = 0.50

ROLLING_WINDOW = 50  # Number of trades to look back


class AdaptiveThresholds:
    """Manages per-strategy adaptive thresholds."""

    def __init__(self) -> None:
        self._init_db()
        self._cache: dict[str, dict[str, float]] = {}

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS strategy_thresholds (
                strategy TEXT PRIMARY KEY,
                min_edge REAL NOT NULL,
                min_confidence REAL NOT NULL,
                rolling_wr REAL DEFAULT 0.5,
                trades_analyzed INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL,
                reason TEXT DEFAULT ''
            )
        """)
        conn.commit()
        conn.close()

    def get_thresholds(self, strategy: str) -> dict[str, float]:
        """Get current thresholds for a strategy. Returns from DB or defaults."""
        if strategy in self._cache:
            return self._cache[strategy]

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM strategy_thresholds WHERE strategy = ?", (strategy,))
        row = c.fetchone()
        conn.close()

        if row:
            result = {"min_edge": row["min_edge"], "min_confidence": row["min_confidence"]}
        else:
            defaults = DEFAULT_THRESHOLDS.get(strategy, {"min_edge": 0.05, "min_confidence": 0.25})
            result = dict(defaults)

        self._cache[strategy] = result
        return result

    def update_thresholds(self) -> dict[str, Any]:
        """
        Recalculate thresholds for all strategies based on rolling win rate.
        Should be called periodically (e.g., after each settlement cycle).
        """
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        changes: dict[str, Any] = {}

        for strategy, defaults in DEFAULT_THRESHOLDS.items():
            # Get last N settled trades for this strategy
            c.execute(
                """SELECT id, pnl FROM trades
                WHERE strategy = ? AND status = 'settled' AND action = 'buy'
                ORDER BY settled_at DESC LIMIT ?""",
                (strategy, ROLLING_WINDOW),
            )
            trades = c.fetchall()

            if len(trades) < 10:
                continue  # Not enough data to adjust

            wins = sum(1 for t in trades if t["pnl"] and t["pnl"] > 0)
            wr = wins / len(trades)

            current = self.get_thresholds(strategy)
            new_edge = current["min_edge"]
            new_conf = current["min_confidence"]
            reason = ""

            if wr < 0.40:
                new_edge = min(current["min_edge"] + 0.02, MIN_EDGE_CEILING)
                new_conf = min(current["min_confidence"] + 0.05, MIN_CONF_CEILING)
                reason = f"WR={wr:.0%} < 40% → tightened"
            elif wr > 0.60:
                new_edge = max(current["min_edge"] - 0.01, MIN_EDGE_FLOOR)
                new_conf = max(current["min_confidence"] - 0.02, MIN_CONF_FLOOR)
                reason = f"WR={wr:.0%} > 60% → loosened"
            else:
                reason = f"WR={wr:.0%} in range → no change"

            # Only update if changed
            if abs(new_edge - current["min_edge"]) > 0.001 or abs(new_conf - current["min_confidence"]) > 0.001:
                now = datetime.now(UTC).isoformat()
                c.execute(
                    """INSERT INTO strategy_thresholds (strategy, min_edge, min_confidence, rolling_wr, trades_analyzed, updated_at, reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(strategy) DO UPDATE SET
                        min_edge = ?, min_confidence = ?, rolling_wr = ?,
                        trades_analyzed = ?, updated_at = ?, reason = ?""",
                    (
                        strategy, new_edge, new_conf, wr, len(trades), now, reason,
                        new_edge, new_conf, wr, len(trades), now, reason,
                    ),
                )

                self._cache[strategy] = {"min_edge": new_edge, "min_confidence": new_conf}
                changes[strategy] = {
                    "old_edge": current["min_edge"],
                    "new_edge": new_edge,
                    "old_conf": current["min_confidence"],
                    "new_conf": new_conf,
                    "wr": wr,
                    "trades": len(trades),
                    "reason": reason,
                }
                logger.info(
                    "Threshold adjusted",
                    strategy=strategy,
                    new_edge=f"{new_edge:.1%}",
                    new_conf=f"{new_conf:.1%}",
                    wr=f"{wr:.0%}",
                    reason=reason,
                )

        conn.commit()
        conn.close()
        return changes

    def get_all_thresholds(self) -> dict[str, dict[str, Any]]:
        """Get all current thresholds with metadata."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM strategy_thresholds ORDER BY strategy")
        rows = c.fetchall()
        conn.close()

        result = {}
        for row in rows:
            result[row["strategy"]] = {
                "min_edge": row["min_edge"],
                "min_confidence": row["min_confidence"],
                "rolling_wr": row["rolling_wr"],
                "trades_analyzed": row["trades_analyzed"],
                "updated_at": row["updated_at"],
                "reason": row["reason"],
            }

        # Fill in defaults for strategies not yet in DB
        for strategy, defaults in DEFAULT_THRESHOLDS.items():
            if strategy not in result:
                result[strategy] = {
                    "min_edge": defaults["min_edge"],
                    "min_confidence": defaults["min_confidence"],
                    "rolling_wr": None,
                    "trades_analyzed": 0,
                    "updated_at": None,
                    "reason": "default",
                }

        return result


# Singleton
_thresholds: AdaptiveThresholds | None = None


def get_adaptive_thresholds() -> AdaptiveThresholds:
    global _thresholds
    if _thresholds is None:
        _thresholds = AdaptiveThresholds()
    return _thresholds
