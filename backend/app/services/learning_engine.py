"""
Continuous Learning Engine for PropsAI Kalshi Trading Bot.

Runs after each trading cycle to analyze outcomes and auto-adjust parameters:
1. BIAS DRIFT: Updates per-city forecast bias corrections as new data comes in
2. EDGE THRESHOLD: Adjusts minimum edge per strategy based on win rate by bucket
3. KELLY MULTIPLIER: Tightens or loosens sizing based on observed variance
4. SIGNAL WEIGHTS: Feeds outcome data to the SignalScorer for dynamic weight updates
5. CONFIDENCE CALIBRATION: Tracks predicted probability vs actual outcome for Brier score
6. MARKET TIMING: Learns which hours/days produce the best risk-adjusted returns

All adjustments are bounded and conservative — the engine nudges parameters slowly
toward better values, never making sudden large changes. A "learning rate" controls
how fast each parameter moves (default: 10% per cycle).

Data comes from the trading_engine.db SQLite database (trades, signals, weather_api_performance).
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

UTC = timezone.utc
DB_PATH = Path(__file__).parent.parent / "data" / "trading_engine.db"
LEARNING_STATE_PATH = Path(__file__).parent.parent / "data" / "learning_state.json"

# How fast parameters move toward new values (0.0 = never, 1.0 = instant)
LEARNING_RATE = 0.10
# Minimum trades needed before learning kicks in for a parameter
MIN_TRADES = 15
# Maximum adjustment per cycle (prevents runaway)
MAX_ADJUSTMENT_PER_CYCLE = 0.20


class LearningEngine:
    """
    Continuous learning system that improves trading parameters over time.

    Call `run_learning_cycle()` after each trading cycle. It reads outcome data
    from the database, computes adjustments, and updates the persistent state.
    The agent reads the state on each cycle to use the latest tuned parameters.
    """

    def __init__(self) -> None:
        self._state = self._load_state()

    # ── State Persistence ─────────────────────────────────────────

    def _load_state(self) -> dict[str, Any]:
        """Load the learning state from disk, or create defaults."""
        if LEARNING_STATE_PATH.exists():
            try:
                with open(LEARNING_STATE_PATH) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load learning state, using defaults", error=str(e))

        return {
            "version": 1,
            "last_updated": None,
            "cycles_run": 0,

            # Per-city forecast bias (forecast - actual, so negative = model runs warm)
            "city_bias": {},

            # Per-strategy minimum edge thresholds
            "edge_thresholds": {
                "weather": 0.05,
                "weather_observed": 0.02,
                "crypto": 0.08,
                "finance": 0.08,
                "sports": 0.06,
                "econ": 0.10,
            },

            # Kelly multiplier per strategy (applied on top of quarter-Kelly)
            # Values 0.5-1.5: below 1.0 = more conservative, above 1.0 = more aggressive
            "kelly_multipliers": {
                "weather": 1.0,
                "crypto": 1.0,
                "finance": 1.0,
                "sports": 1.0,
                "econ": 1.0,
            },

            # Calibration data: probability bucket → {correct, total}
            "calibration_buckets": {},

            # Win rate by hour of day (0-23) for timing optimization
            "hourly_performance": {},

            # Per-strategy Brier scores (lower is better, 0 = perfect)
            "brier_scores": {},

            # Trade count per strategy (for gating learning)
            "trade_counts": {},
        }

    def _save_state(self) -> None:
        """Persist learning state to disk."""
        self._state["last_updated"] = datetime.now(UTC).isoformat()
        LEARNING_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEARNING_STATE_PATH, "w") as f:
            json.dump(self._state, f, indent=2)
        logger.info("Learning state saved", cycles=self._state["cycles_run"])

    # ── Main Learning Cycle ───────────────────────────────────────

    def run_learning_cycle(self) -> dict[str, Any]:
        """
        Run one learning cycle. Call this after each trading cycle.

        Returns a summary of what was learned and adjusted.
        """
        summary: dict[str, Any] = {"adjustments": [], "observations": []}

        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row

            # 1. Learn from settled trades
            self._learn_from_trades(conn, summary)

            # 2. Learn from weather forecast errors
            self._learn_bias_drift(conn, summary)

            # 3. Update edge thresholds based on win rates
            self._update_edge_thresholds(conn, summary)

            # 4. Update Kelly multipliers based on variance
            self._update_kelly_multipliers(conn, summary)

            # 5. Update calibration / Brier scores
            self._update_calibration(conn, summary)

            # 6. Learn optimal trading hours
            self._learn_timing(conn, summary)

            conn.close()

        except Exception as e:
            logger.error("Learning cycle failed", error=str(e))
            summary["error"] = str(e)

        self._state["cycles_run"] += 1
        self._save_state()

        return summary

    # ── 1. Trade Outcome Analysis ─────────────────────────────────

    def _learn_from_trades(self, conn: sqlite3.Connection, summary: dict) -> None:
        """Analyze settled trades to update strategy-level metrics."""
        rows = conn.execute("""
            SELECT strategy, result, edge, our_prob, kalshi_prob, price_cents, pnl,
                   timestamp, signal_source
            FROM trades
            WHERE result != '' AND result IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 500
        """).fetchall()

        if not rows:
            return

        by_strategy: dict[str, list] = {}
        for row in rows:
            strat = row["strategy"] or "unknown"
            by_strategy.setdefault(strat, []).append(dict(row))

        for strategy, trades in by_strategy.items():
            wins = sum(1 for t in trades if t["pnl"] and t["pnl"] > 0)
            total = len(trades)
            win_rate = wins / total if total > 0 else 0

            self._state["trade_counts"][strategy] = total

            summary["observations"].append({
                "type": "trade_stats",
                "strategy": strategy,
                "total": total,
                "wins": wins,
                "win_rate": round(win_rate, 3),
            })

    # ── 2. Weather Bias Drift Detection ───────────────────────────

    def _learn_bias_drift(self, conn: sqlite3.Connection, summary: dict) -> None:
        """
        Update per-city forecast bias using actual weather_api_performance data.
        If forecasts are consistently X°F too warm for a city, increase the cool bias.
        """
        rows = conn.execute("""
            SELECT city, market_type, forecast_temp, actual_temp, error
            FROM weather_api_performance
            WHERE actual_temp IS NOT NULL AND forecast_temp IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1000
        """).fetchall()

        if not rows:
            return

        # Group by city + metric type
        by_city_metric: dict[str, list[float]] = {}
        for row in rows:
            city = row["city"]
            metric = "high" if "high" in (row["market_type"] or "").lower() else "low"
            key = f"{city}_{metric}"
            error = row["error"] if row["error"] is not None else (row["forecast_temp"] - row["actual_temp"])
            by_city_metric.setdefault(key, []).append(error)

        for key, errors in by_city_metric.items():
            if len(errors) < 5:  # Need at least 5 data points
                continue

            city, metric = key.rsplit("_", 1)
            mean_error = sum(errors) / len(errors)

            # Current bias (negative = we cool the forecast)
            old_bias = self._state["city_bias"].get(city, {}).get(metric, -1.5)

            # The bias correction should equal the negative of the mean error
            # If forecast is 2°F too warm (error=+2), bias should be -2
            target_bias = -mean_error

            # Clamp target to reasonable range
            target_bias = max(-5.0, min(2.0, target_bias))

            # Smooth update
            new_bias = old_bias + LEARNING_RATE * (target_bias - old_bias)
            new_bias = round(new_bias, 2)

            if city not in self._state["city_bias"]:
                self._state["city_bias"][city] = {}
            self._state["city_bias"][city][metric] = new_bias

            if abs(new_bias - old_bias) > 0.05:
                summary["adjustments"].append({
                    "type": "bias_drift",
                    "city": city,
                    "metric": metric,
                    "old": old_bias,
                    "new": new_bias,
                    "samples": len(errors),
                    "mean_error": round(mean_error, 2),
                })

    # ── 3. Edge Threshold Optimization ────────────────────────────

    def _update_edge_thresholds(self, conn: sqlite3.Connection, summary: dict) -> None:
        """
        Adjust minimum edge thresholds per strategy.

        If we're winning at high rates with our current threshold, we could
        cautiously lower it to take more trades. If we're losing, raise it
        to be more selective.

        Target: 55-65% win rate on trades that pass the edge threshold.
        """
        rows = conn.execute("""
            SELECT strategy, edge, result, pnl
            FROM trades
            WHERE result != '' AND result IS NOT NULL
                  AND edge IS NOT NULL AND edge != 0
            ORDER BY timestamp DESC
            LIMIT 500
        """).fetchall()

        by_strategy: dict[str, list] = {}
        for row in rows:
            strat = row["strategy"] or "unknown"
            by_strategy.setdefault(strat, []).append(dict(row))

        for strategy, trades in by_strategy.items():
            if len(trades) < MIN_TRADES:
                continue

            wins = sum(1 for t in trades if t["pnl"] and t["pnl"] > 0)
            win_rate = wins / len(trades)
            current_threshold = self._state["edge_thresholds"].get(strategy, 0.05)

            # Target win rate band: 55-65%
            if win_rate > 0.65 and len(trades) >= 30:
                # We're too conservative — lower threshold to take more trades
                adjustment = -0.005  # Lower by 0.5%
                reason = f"win rate {win_rate:.0%} > 65%, loosening"
            elif win_rate < 0.45:
                # We're losing — raise threshold to be more selective
                adjustment = +0.01  # Raise by 1%
                reason = f"win rate {win_rate:.0%} < 45%, tightening"
            elif win_rate < 0.50:
                # Barely losing — small tighten
                adjustment = +0.005
                reason = f"win rate {win_rate:.0%} < 50%, slight tighten"
            else:
                continue  # Win rate is in the sweet spot, no change

            new_threshold = current_threshold + adjustment
            # Clamp: never go below 2% or above 20%
            new_threshold = max(0.02, min(0.20, round(new_threshold, 4)))

            if new_threshold != current_threshold:
                self._state["edge_thresholds"][strategy] = new_threshold
                summary["adjustments"].append({
                    "type": "edge_threshold",
                    "strategy": strategy,
                    "old": current_threshold,
                    "new": new_threshold,
                    "reason": reason,
                    "trades": len(trades),
                    "win_rate": round(win_rate, 3),
                })

    # ── 4. Kelly Multiplier Adjustment ────────────────────────────

    def _update_kelly_multipliers(self, conn: sqlite3.Connection, summary: dict) -> None:
        """
        Adjust Kelly multiplier based on observed P&L variance.

        High variance → reduce multiplier (more conservative)
        Low variance + profitable → increase multiplier (capture more)

        The multiplier scales the quarter-Kelly fraction: actual_fraction = quarter_kelly * multiplier
        """
        rows = conn.execute("""
            SELECT strategy, pnl, cost
            FROM trades
            WHERE result != '' AND pnl IS NOT NULL AND cost > 0
            ORDER BY timestamp DESC
            LIMIT 300
        """).fetchall()

        by_strategy: dict[str, list[float]] = {}
        for row in rows:
            strat = row["strategy"] or "unknown"
            # Normalize P&L as percentage of cost
            pnl_pct = row["pnl"] / row["cost"] if row["cost"] else 0
            by_strategy.setdefault(strat, []).append(pnl_pct)

        for strategy, returns in by_strategy.items():
            if len(returns) < MIN_TRADES:
                continue

            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = math.sqrt(variance) if variance > 0 else 0

            current_mult = self._state["kelly_multipliers"].get(strategy, 1.0)

            # If profitable with low variance: cautiously increase
            # If unprofitable or high variance: decrease
            if mean_return > 0.05 and std_dev < 0.5:
                target = min(1.5, current_mult + 0.05)
                reason = f"profitable (μ={mean_return:.2f}) with low variance (σ={std_dev:.2f})"
            elif mean_return < -0.05 or std_dev > 1.0:
                target = max(0.5, current_mult - 0.1)
                reason = f"{'unprofitable' if mean_return < 0 else 'high variance'} (μ={mean_return:.2f}, σ={std_dev:.2f})"
            else:
                continue

            new_mult = current_mult + LEARNING_RATE * (target - current_mult)
            new_mult = round(max(0.5, min(1.5, new_mult)), 3)

            if abs(new_mult - current_mult) > 0.01:
                self._state["kelly_multipliers"][strategy] = new_mult
                summary["adjustments"].append({
                    "type": "kelly_multiplier",
                    "strategy": strategy,
                    "old": current_mult,
                    "new": new_mult,
                    "reason": reason,
                    "trades": len(returns),
                })

    # ── 5. Calibration / Brier Score ──────────────────────────────

    def _update_calibration(self, conn: sqlite3.Connection, summary: dict) -> None:
        """
        Track prediction calibration using Brier score.

        For each trade, we predicted a probability. Did the actual outcome
        match that probability? Perfect calibration means: when we say 70%,
        the event happens 70% of the time.

        Brier score = mean((predicted_prob - outcome)²)
        Range: 0 (perfect) to 1 (worst). Below 0.25 is decent, below 0.15 is good.
        """
        rows = conn.execute("""
            SELECT strategy, our_prob, side, result
            FROM trades
            WHERE result != '' AND our_prob > 0 AND our_prob < 1
            ORDER BY timestamp DESC
            LIMIT 500
        """).fetchall()

        by_strategy: dict[str, list[tuple[float, int]]] = {}
        for row in rows:
            strat = row["strategy"] or "unknown"
            predicted = row["our_prob"]
            # outcome: 1 if the side we bet on won, 0 if it lost
            if row["side"] == "yes":
                actual = 1 if row["result"] == "yes" else 0
            else:
                actual = 1 if row["result"] == "no" else 0
            by_strategy.setdefault(strat, []).append((predicted, actual))

            # Update calibration bucket (rounded to nearest 0.10)
            bucket = str(round(predicted, 1))
            if bucket not in self._state["calibration_buckets"]:
                self._state["calibration_buckets"][bucket] = {"correct": 0, "total": 0}
            self._state["calibration_buckets"][bucket]["total"] += 1
            if actual == 1:
                self._state["calibration_buckets"][bucket]["correct"] += 1

        for strategy, predictions in by_strategy.items():
            if len(predictions) < 10:
                continue

            brier = sum((p - a) ** 2 for p, a in predictions) / len(predictions)
            self._state["brier_scores"][strategy] = round(brier, 4)

            summary["observations"].append({
                "type": "brier_score",
                "strategy": strategy,
                "brier_score": round(brier, 4),
                "quality": "excellent" if brier < 0.10 else "good" if brier < 0.20 else "fair" if brier < 0.30 else "poor",
                "trades": len(predictions),
            })

    # ── 6. Timing Optimization ────────────────────────────────────

    def _learn_timing(self, conn: sqlite3.Connection, summary: dict) -> None:
        """
        Track win rate by hour of day to identify optimal trading windows.
        """
        rows = conn.execute("""
            SELECT timestamp, strategy, pnl
            FROM trades
            WHERE result != '' AND pnl IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 500
        """).fetchall()

        by_hour: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
                hour = str(ts.hour)
            except Exception:
                continue

            if hour not in by_hour:
                by_hour[hour] = {"wins": 0, "losses": 0, "total_pnl": 0.0}

            if row["pnl"] > 0:
                by_hour[hour]["wins"] += 1
            else:
                by_hour[hour]["losses"] += 1
            by_hour[hour]["total_pnl"] += row["pnl"]

        self._state["hourly_performance"] = by_hour

        # Find best and worst hours
        if by_hour:
            best_hour = max(by_hour.items(), key=lambda x: x[1]["total_pnl"])
            worst_hour = min(by_hour.items(), key=lambda x: x[1]["total_pnl"])

            total_in_best = best_hour[1]["wins"] + best_hour[1]["losses"]
            total_in_worst = worst_hour[1]["wins"] + worst_hour[1]["losses"]

            if total_in_best >= 5 and total_in_worst >= 5:
                summary["observations"].append({
                    "type": "timing",
                    "best_hour": int(best_hour[0]),
                    "best_pnl": round(best_hour[1]["total_pnl"], 2),
                    "worst_hour": int(worst_hour[0]),
                    "worst_pnl": round(worst_hour[1]["total_pnl"], 2),
                })

    # ── Public Getters for the Agent ──────────────────────────────

    def get_edge_threshold(self, strategy: str) -> float:
        """Get the current learned edge threshold for a strategy."""
        return self._state["edge_thresholds"].get(strategy, 0.05)

    def get_kelly_multiplier(self, strategy: str) -> float:
        """Get the current learned Kelly multiplier for a strategy."""
        return self._state["kelly_multipliers"].get(strategy, 1.0)

    def get_city_bias(self, city: str, metric: str) -> float | None:
        """
        Get learned bias for a city/metric pair.
        Returns None if no learned value exists (caller should use hardcoded default).
        """
        return self._state.get("city_bias", {}).get(city, {}).get(metric)

    def get_brier_score(self, strategy: str) -> float | None:
        """Get the Brier score for a strategy (lower is better)."""
        return self._state.get("brier_scores", {}).get(strategy)

    def get_calibration_report(self) -> dict[str, Any]:
        """Get full calibration report for the dashboard."""
        buckets = self._state.get("calibration_buckets", {})
        calibration = {}
        for bucket, data in sorted(buckets.items()):
            total = data["total"]
            if total > 0:
                calibration[bucket] = {
                    "predicted": float(bucket),
                    "actual": round(data["correct"] / total, 3),
                    "total": total,
                    "gap": round(float(bucket) - data["correct"] / total, 3),
                }

        return {
            "brier_scores": self._state.get("brier_scores", {}),
            "calibration_by_bucket": calibration,
            "edge_thresholds": self._state.get("edge_thresholds", {}),
            "kelly_multipliers": self._state.get("kelly_multipliers", {}),
            "hourly_performance": self._state.get("hourly_performance", {}),
            "trade_counts": self._state.get("trade_counts", {}),
            "cycles_run": self._state.get("cycles_run", 0),
            "last_updated": self._state.get("last_updated"),
        }

    def get_state(self) -> dict[str, Any]:
        """Get the full learning state (for the API/dashboard)."""
        return dict(self._state)


# Singleton
_engine: LearningEngine | None = None


def get_learning_engine() -> LearningEngine:
    """Get or create the singleton learning engine."""
    global _engine
    if _engine is None:
        _engine = LearningEngine()
    return _engine
