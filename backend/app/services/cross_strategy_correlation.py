"""
Cross-Strategy Correlation Engine.

Detects when multiple strategies produce signals on the same or related
Kalshi markets. When strategies agree, confidence is boosted. When they
conflict, the engine flags the disagreement for review.

Correlation types:
  1. Same-ticker agreement: Two strategies both signal YES or NO on the
     same ticker → boost confidence by up to 15%.
  2. Related-market agreement: Crypto momentum + finance VIX both point
     the same direction on an S&P market → boost by up to 10%.
  3. Conflict detection: One strategy says YES, another says NO on the
     same ticker → reduce confidence or skip.
  4. Regime detection: Multiple econ signals pointing the same way
     suggests a macro regime (risk-on / risk-off) that affects all markets.

Usage:
    engine = CrossStrategyCorrelation()
    engine.record_signal("crypto", "KXBTC-UP", "yes", 0.65, 0.55)
    engine.record_signal("finance", "KXBTC-UP", "yes", 0.70, 0.55)
    result = engine.get_correlation("KXBTC-UP")
    # → {"agreement": True, "boost": 0.12, "strategies": ["crypto", "finance"]}
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

# How long a signal stays "active" before expiring (seconds)
SIGNAL_TTL = 1800  # 30 minutes

# Related market groups — tickers in the same group are considered correlated
MARKET_GROUPS = {
    "crypto_btc": ["KXBTC", "KXBITCOIN", "BTC"],
    "crypto_eth": ["KXETH", "KXETHEREUM", "ETH"],
    "crypto_sol": ["KXSOL", "KXSOLANA", "SOL"],
    "sp500": ["KXSP500", "KXSPX", "KXSPY", "S&P"],
    "nasdaq": ["KXNASDAQ", "KXNDX", "KXQQQ", "NASDAQ"],
    "fed_rate": ["KXFED", "KXFOMC", "KXRATE"],
    "cpi": ["KXCPI", "KXINFLATION"],
    "gas": ["KXGAS", "KXGASOLINE", "KXOIL"],
    "jobs": ["KXJOBS", "KXUNEMPLOYMENT", "KXNFP"],
}

# Strategy pairs that reinforce each other when they agree
REINFORCING_PAIRS = {
    ("crypto", "finance"): 0.10,   # Crypto + finance agree on direction
    ("crypto", "orderbook"): 0.08, # Crypto signal + orderbook imbalance
    ("finance", "econ"): 0.10,     # Finance + econ macro alignment
    ("sports", "nba_props"): 0.05, # Sports game-level + player props
    ("weather", "weather"): 0.0,   # Weather is independent
}


class SignalRecord:
    """A single strategy signal on a ticker."""

    def __init__(
        self,
        strategy: str,
        ticker: str,
        side: str,
        our_prob: float,
        kalshi_prob: float,
        confidence: float,
        timestamp: float | None = None,
    ) -> None:
        self.strategy = strategy
        self.ticker = ticker
        self.side = side  # "yes" or "no"
        self.our_prob = our_prob
        self.kalshi_prob = kalshi_prob
        self.confidence = confidence
        self.edge = our_prob - kalshi_prob
        self.timestamp = timestamp or time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > SIGNAL_TTL

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "ticker": self.ticker,
            "side": self.side,
            "our_prob": self.our_prob,
            "kalshi_prob": self.kalshi_prob,
            "confidence": self.confidence,
            "edge": round(self.edge, 4),
            "age_seconds": round(time.time() - self.timestamp, 1),
        }


class CrossStrategyCorrelation:
    """Cross-strategy correlation and signal aggregation engine."""

    def __init__(self) -> None:
        # ticker -> list of active signals
        self._signals: dict[str, list[SignalRecord]] = defaultdict(list)
        # Track macro regime
        self._regime: str = "neutral"  # "risk_on", "risk_off", "neutral"
        self._regime_confidence: float = 0.0
        self._regime_updated: float = 0.0

    # ── Signal Recording ─────────────────────────────────────────────

    def record_signal(
        self,
        strategy: str,
        ticker: str,
        side: str,
        our_prob: float,
        kalshi_prob: float,
        confidence: float = 0.5,
    ) -> None:
        """Record a strategy signal for cross-correlation analysis."""
        # Clean expired signals first
        self._cleanup_expired(ticker)

        # Don't duplicate — update if same strategy+ticker exists
        existing = [
            s for s in self._signals[ticker]
            if s.strategy == strategy
        ]
        if existing:
            self._signals[ticker] = [
                s for s in self._signals[ticker]
                if s.strategy != strategy
            ]

        signal = SignalRecord(
            strategy=strategy,
            ticker=ticker,
            side=side,
            our_prob=our_prob,
            kalshi_prob=kalshi_prob,
            confidence=confidence,
        )
        self._signals[ticker].append(signal)

        # Update macro regime if relevant
        self._update_regime(strategy, ticker, side, confidence)

    # ── Correlation Queries ──────────────────────────────────────────

    def get_correlation(self, ticker: str) -> dict[str, Any]:
        """
        Get cross-strategy correlation analysis for a ticker.

        Returns:
            agreement: bool — do all active signals agree on direction?
            boost: float — confidence boost (0.0 to 0.15)
            penalty: float — confidence penalty for conflicts (0.0 to -0.10)
            strategies: list — which strategies have active signals
            regime_alignment: bool — does this ticker align with macro regime?
        """
        self._cleanup_expired(ticker)
        signals = self._signals.get(ticker, [])

        if len(signals) < 2:
            return {
                "agreement": True,
                "boost": 0.0,
                "penalty": 0.0,
                "strategies": [s.strategy for s in signals],
                "signal_count": len(signals),
                "regime": self._regime,
                "regime_alignment": True,
            }

        # Check agreement
        sides = [s.side for s in signals]
        all_agree = len(set(sides)) == 1

        if all_agree:
            boost = self._compute_agreement_boost(signals)
            return {
                "agreement": True,
                "boost": boost,
                "penalty": 0.0,
                "strategies": [s.strategy for s in signals],
                "signal_count": len(signals),
                "avg_edge": round(
                    sum(s.edge for s in signals) / len(signals), 4
                ),
                "avg_confidence": round(
                    sum(s.confidence for s in signals) / len(signals), 3
                ),
                "regime": self._regime,
                "regime_alignment": self._check_regime_alignment(ticker, sides[0]),
            }
        else:
            penalty = self._compute_conflict_penalty(signals)
            return {
                "agreement": False,
                "boost": 0.0,
                "penalty": penalty,
                "strategies": [s.strategy for s in signals],
                "signal_count": len(signals),
                "conflict_details": [s.to_dict() for s in signals],
                "regime": self._regime,
                "regime_alignment": False,
            }

    def get_related_signals(self, ticker: str) -> list[dict[str, Any]]:
        """Find signals on related markets (same group)."""
        group = self._find_market_group(ticker)
        if not group:
            return []

        related = []
        for other_ticker, signals in self._signals.items():
            if other_ticker == ticker:
                continue
            if self._find_market_group(other_ticker) == group:
                for s in signals:
                    if not s.is_expired:
                        related.append(s.to_dict())

        return related

    def get_confidence_adjustment(self, ticker: str, strategy: str) -> float:
        """
        Get the net confidence adjustment for a ticker from a specific strategy.
        Positive = boost (agreement), negative = penalty (conflict).
        """
        corr = self.get_correlation(ticker)
        base_adjustment = corr["boost"] + corr["penalty"]

        # Check related market signals
        related = self.get_related_signals(ticker)
        if related:
            # Get the side our strategy is signaling
            our_signals = [
                s for s in self._signals.get(ticker, [])
                if s.strategy == strategy
            ]
            if our_signals:
                our_side = our_signals[0].side
                agreeing = sum(1 for r in related if r["side"] == our_side)
                disagreeing = len(related) - agreeing
                related_boost = min(agreeing * 0.03, 0.08)
                related_penalty = min(disagreeing * 0.02, 0.06)
                base_adjustment += related_boost - related_penalty

        # Regime alignment bonus
        if corr.get("regime_alignment") and self._regime_confidence > 0.3:
            base_adjustment += 0.03

        return round(min(max(base_adjustment, -0.15), 0.15), 4)

    # ── Macro Regime Detection ───────────────────────────────────────

    def get_regime(self) -> dict[str, Any]:
        """Get current macro regime assessment."""
        return {
            "regime": self._regime,
            "confidence": round(self._regime_confidence, 3),
            "age_seconds": round(time.time() - self._regime_updated, 1)
            if self._regime_updated > 0
            else None,
        }

    def _update_regime(
        self, strategy: str, ticker: str, side: str, confidence: float
    ) -> None:
        """Update macro regime based on incoming signals."""
        # Only finance and econ strategies affect regime
        if strategy not in ("finance", "econ", "crypto"):
            return

        ticker_upper = ticker.upper()

        # Determine if this signal is risk-on or risk-off
        risk_direction = None

        # S&P/Nasdaq up = risk-on
        if any(kw in ticker_upper for kw in ["SP500", "SPX", "NASDAQ", "NDX"]):
            risk_direction = "risk_on" if side == "yes" else "risk_off"

        # VIX up = risk-off (inverse)
        elif "VIX" in ticker_upper:
            risk_direction = "risk_off" if side == "yes" else "risk_on"

        # Crypto up = risk-on
        elif any(kw in ticker_upper for kw in ["BTC", "ETH", "SOL", "CRYPTO"]):
            risk_direction = "risk_on" if side == "yes" else "risk_off"

        # Fed rate hike = risk-off
        elif any(kw in ticker_upper for kw in ["FED", "FOMC", "RATE"]):
            risk_direction = "risk_off" if side == "yes" else "risk_on"

        if risk_direction:
            # Simple exponential moving average of regime signals
            alpha = 0.3
            current_score = 1.0 if risk_direction == "risk_on" else -1.0
            current_score *= confidence

            if self._regime == "risk_on":
                old_score = self._regime_confidence
            elif self._regime == "risk_off":
                old_score = -self._regime_confidence
            else:
                old_score = 0.0

            new_score = alpha * current_score + (1 - alpha) * old_score

            if new_score > 0.2:
                self._regime = "risk_on"
                self._regime_confidence = new_score
            elif new_score < -0.2:
                self._regime = "risk_off"
                self._regime_confidence = abs(new_score)
            else:
                self._regime = "neutral"
                self._regime_confidence = abs(new_score)

            self._regime_updated = time.time()

    def _check_regime_alignment(self, ticker: str, side: str) -> bool:
        """Check if a signal aligns with the current macro regime."""
        if self._regime == "neutral":
            return True

        ticker_upper = ticker.upper()

        # Risk-on assets should be YES in risk-on regime
        risk_on_assets = ["SP500", "SPX", "NASDAQ", "NDX", "BTC", "ETH", "SOL"]
        if any(kw in ticker_upper for kw in risk_on_assets):
            if self._regime == "risk_on":
                return side == "yes"
            else:
                return side == "no"

        return True  # Unknown asset, no alignment check

    # ── Internal Helpers ─────────────────────────────────────────────

    def _compute_agreement_boost(self, signals: list[SignalRecord]) -> float:
        """Compute confidence boost when strategies agree."""
        if len(signals) < 2:
            return 0.0

        boost = 0.0
        strategies = [s.strategy for s in signals]

        # Check for reinforcing pairs
        for i in range(len(strategies)):
            for j in range(i + 1, len(strategies)):
                pair = (strategies[i], strategies[j])
                reverse_pair = (strategies[j], strategies[i])
                pair_boost = REINFORCING_PAIRS.get(
                    pair, REINFORCING_PAIRS.get(reverse_pair, 0.05)
                )
                boost += pair_boost

        # Scale by average confidence of agreeing signals
        avg_conf = sum(s.confidence for s in signals) / len(signals)
        boost *= min(avg_conf * 1.5, 1.0)

        # More strategies agreeing = stronger signal
        if len(signals) >= 3:
            boost *= 1.2

        return round(min(boost, 0.15), 4)

    def _compute_conflict_penalty(self, signals: list[SignalRecord]) -> float:
        """Compute confidence penalty when strategies disagree."""
        yes_signals = [s for s in signals if s.side == "yes"]
        no_signals = [s for s in signals if s.side == "no"]

        if not yes_signals or not no_signals:
            return 0.0

        # Penalty scales with the confidence of the opposing signal
        max_opposing_conf = max(
            max((s.confidence for s in yes_signals), default=0),
            max((s.confidence for s in no_signals), default=0),
        )

        penalty = -0.05 * max_opposing_conf

        # Stronger penalty if high-confidence signals conflict
        if max_opposing_conf > 0.6:
            penalty *= 1.5

        return round(max(penalty, -0.15), 4)

    def _find_market_group(self, ticker: str) -> str | None:
        """Find which market group a ticker belongs to."""
        ticker_upper = ticker.upper()
        for group_name, prefixes in MARKET_GROUPS.items():
            if any(prefix in ticker_upper for prefix in prefixes):
                return group_name
        return None

    def _cleanup_expired(self, ticker: str) -> None:
        """Remove expired signals for a ticker."""
        if ticker in self._signals:
            self._signals[ticker] = [
                s for s in self._signals[ticker] if not s.is_expired
            ]
            if not self._signals[ticker]:
                del self._signals[ticker]

    # ── Status / Debug ───────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        """Get engine status summary."""
        total_signals = sum(len(sigs) for sigs in self._signals.values())
        active_tickers = len(self._signals)

        # Count agreements vs conflicts
        agreements = 0
        conflicts = 0
        for ticker in self._signals:
            corr = self.get_correlation(ticker)
            if corr["signal_count"] >= 2:
                if corr["agreement"]:
                    agreements += 1
                else:
                    conflicts += 1

        return {
            "total_signals": total_signals,
            "active_tickers": active_tickers,
            "agreements": agreements,
            "conflicts": conflicts,
            "regime": self._regime,
            "regime_confidence": round(self._regime_confidence, 3),
        }

    def get_all_signals(self) -> dict[str, list[dict]]:
        """Get all active signals grouped by ticker."""
        result = {}
        for ticker, signals in self._signals.items():
            active = [s.to_dict() for s in signals if not s.is_expired]
            if active:
                result[ticker] = active
        return result


# ── Singleton ────────────────────────────────────────────────────────

_engine: CrossStrategyCorrelation | None = None


def get_cross_strategy_engine() -> CrossStrategyCorrelation:
    """Get or create the singleton cross-strategy correlation engine."""
    global _engine
    if _engine is None:
        _engine = CrossStrategyCorrelation()
    return _engine
