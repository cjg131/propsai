"""
Crypto Data Service for 15-minute Kalshi markets.

Fetches real-time crypto data from Binance public API (free, no key needed)
and generates directional probability estimates for BTC, ETH, SOL, XRP.

Signals:
  1. Momentum — 5-min price trend continuation bias
  2. Volatility — realized vol vs Kalshi implied vol
  3. Funding rate — perpetual futures directional bias
  4. Mean reversion — fade sharp 1-min moves
"""
from __future__ import annotations

import asyncio
import math
import statistics
from datetime import UTC, datetime
from typing import Any

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

# ── API endpoints ─────────────────────────────────────────────────
# Coinbase Exchange public REST API (free, no key needed, globally accessible)
COINBASE_BASE = "https://api.exchange.coinbase.com"
# OKX public API for funding rates (free, no key needed, globally accessible)
OKX_PUBLIC_BASE = "https://www.okx.com/api/v5/public"

# Coin → Coinbase product ID
CRYPTO_SYMBOLS = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
}

# Coinbase granularity in seconds for each interval
COINBASE_GRANULARITY = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}

# Coin → OKX perpetual swap instrument ID
OKX_INSTRUMENTS = {
    "BTC": "BTC-USDT-SWAP",
    "ETH": "ETH-USDT-SWAP",
    "SOL": "SOL-USDT-SWAP",
    "XRP": "XRP-USDT-SWAP",
}

# Supported coins (keyed same as before for compatibility)
SUPPORTED_COINS = set(CRYPTO_SYMBOLS.keys())

# ── Signal weights (tuned for 15-min horizon) ───────────────────────
# Funding rate now available via OKX public API
WEIGHT_MOMENTUM_5M = 0.35
WEIGHT_MOMENTUM_1M = 0.15
WEIGHT_FUNDING = 0.20
WEIGHT_MEAN_REVERSION = 0.15
WEIGHT_VOLATILITY = 0.15

# Momentum thresholds
STRONG_MOMENTUM_PCT = 0.003  # 0.3% move in 5 min = strong signal
SHARP_MOVE_PCT = 0.005  # 0.5% in 1 min = mean reversion trigger


class CryptoDataService:
    """Fetches real-time crypto data and generates directional signals."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ts: dict[str, float] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Raw data fetchers (Coinbase public API, no key needed) ─────

    async def get_klines(
        self, coin: str, interval: str = "1m", limit: int = 30
    ) -> list[dict[str, float]]:
        """Fetch OHLCV klines from Coinbase Exchange API.
        Supports real 1m, 5m, 15m, 1h intervals.
        """
        product = CRYPTO_SYMBOLS.get(coin)
        if not product:
            return []

        granularity = COINBASE_GRANULARITY.get(interval, 60)
        client = await self._get_client()

        try:
            url = f"{COINBASE_BASE}/products/{product}/candles"
            params = {"granularity": granularity}
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            logger.warning("Coinbase klines failed", coin=coin, interval=interval, error=str(e))
            return []

        # Coinbase candle: [time, low, high, open, close, volume] — newest first
        candles = []
        for k in reversed(raw[:limit]):
            candles.append({
                "open": float(k[3]),
                "high": float(k[2]),
                "low": float(k[1]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": float(k[0]),
            })

        return candles

    async def get_funding_rate(self, symbol: str) -> float | None:
        """Fetch current funding rate from OKX public API.
        Positive rate = longs pay shorts (bearish pressure).
        Negative rate = shorts pay longs (bullish pressure).
        """
        # Map Coinbase product ID to coin key
        coin = None
        for c, s in CRYPTO_SYMBOLS.items():
            if s == symbol:
                coin = c
                break
        inst_id = OKX_INSTRUMENTS.get(coin, "") if coin else ""
        if not inst_id:
            return None

        client = await self._get_client()
        try:
            url = f"{OKX_PUBLIC_BASE}/funding-rate"
            resp = await client.get(url, params={"instId": inst_id})
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "0" and data.get("data"):
                rate = float(data["data"][0].get("fundingRate", 0))
                return rate
        except Exception as e:
            logger.debug("OKX funding rate failed", symbol=symbol, error=str(e))
        return None

    async def get_spot_price(self, coin: str) -> float | None:
        """Fetch current spot price from Coinbase."""
        product = CRYPTO_SYMBOLS.get(coin)
        if not product:
            return None

        client = await self._get_client()
        try:
            url = f"{COINBASE_BASE}/products/{product}/ticker"
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            return float(data["price"])
        except Exception as e:
            logger.debug("Coinbase spot price failed", coin=coin, error=str(e))
        return None

    # ── Signal computation ───────────────────────────────────────────

    def _compute_momentum(self, candles: list[dict[str, float]], lookback: int) -> float:
        """Compute momentum signal from candles.
        Returns value in [-1, 1] where positive = bullish.
        """
        if len(candles) < lookback + 1:
            return 0.0

        recent = candles[-lookback:]
        start_price = recent[0]["open"]
        end_price = recent[-1]["close"]

        if start_price <= 0:
            return 0.0

        pct_change = (end_price - start_price) / start_price

        # Normalize: 0.3% move → ±1.0 signal
        signal = pct_change / STRONG_MOMENTUM_PCT
        return max(-1.0, min(1.0, signal))

    def _compute_mean_reversion(self, candles_1m: list[dict[str, float]]) -> float:
        """Detect sharp 1-min moves and generate mean reversion signal.
        Returns value in [-1, 1] where positive = expect bounce UP.
        """
        if len(candles_1m) < 3:
            return 0.0

        # Look at the last completed 1-min candle
        last = candles_1m[-2]  # -1 is current (incomplete), -2 is last completed
        pct_move = (last["close"] - last["open"]) / last["open"] if last["open"] > 0 else 0

        if abs(pct_move) < SHARP_MOVE_PCT:
            return 0.0  # Not sharp enough to trigger

        # Fade the move: if it dropped sharply, expect bounce up
        signal = -pct_move / SHARP_MOVE_PCT
        return max(-1.0, min(1.0, signal))

    def _compute_funding_signal(self, funding_rate: float | None) -> float:
        """Convert funding rate to directional signal.
        Positive funding → longs pay → slight bearish → negative signal.
        """
        if funding_rate is None:
            return 0.0

        # Typical funding is ±0.01% to ±0.1%
        # Normalize: 0.05% → ±1.0 signal (inverted: positive funding = bearish)
        signal = -funding_rate / 0.0005
        return max(-1.0, min(1.0, signal))

    def _compute_volatility_signal(
        self, candles_5m: list[dict[str, float]], kalshi_price: int
    ) -> float:
        """Compare realized volatility to Kalshi implied volatility.
        If realized vol is high and Kalshi price is near 50c, there may be edge.
        Returns confidence multiplier (0.5 to 1.5).
        """
        if len(candles_5m) < 6 or kalshi_price <= 0:
            return 1.0

        # Realized 15-min vol from 5-min candles
        returns = []
        for i in range(1, len(candles_5m)):
            prev_close = candles_5m[i - 1]["close"]
            curr_close = candles_5m[i]["close"]
            if prev_close > 0:
                returns.append((curr_close - prev_close) / prev_close)

        if len(returns) < 3:
            return 1.0

        realized_vol = statistics.stdev(returns)

        # Kalshi implied vol: price distance from 50c
        # At 50c, implied vol is maximum (most uncertain)
        # At 90c or 10c, implied vol is low (market is confident)
        kalshi_p = kalshi_price / 100.0
        implied_uncertainty = 2 * kalshi_p * (1 - kalshi_p)  # peaks at 0.5

        # If realized vol is high but Kalshi thinks it's certain (price far from 50),
        # there may be mispricing
        if implied_uncertainty < 0.1:
            return 1.0  # Market is very confident, don't fight it

        vol_ratio = realized_vol / max(implied_uncertainty * 0.01, 0.0001)

        # Higher vol ratio = more opportunity
        return min(1.5, max(0.5, 0.8 + vol_ratio * 0.2))

    async def get_crypto_signal(
        self, coin: str, kalshi_price: int = 50
    ) -> dict[str, Any] | None:
        """Generate a directional signal for a coin.

        Args:
            coin: "BTC", "ETH", "SOL", or "XRP"
            kalshi_price: Current Kalshi yes_ask price in cents (for vol comparison)

        Returns:
            {
                "coin": str,
                "p_up": float,  # probability of going up in next 15 min (0-1)
                "confidence": float,  # signal confidence (0-1)
                "momentum_5m": float,
                "momentum_1m": float,
                "funding_signal": float,
                "mean_reversion": float,
                "vol_multiplier": float,
                "spot_price": float,
                "timestamp": str,
            }
        """
        if coin not in SUPPORTED_COINS:
            return None

        symbol = CRYPTO_SYMBOLS.get(coin, "")

        # Fetch real 1-min and 5-min candles + funding rate from Coinbase/OKX
        candles_1m, candles_5m, spot_price, funding_rate = await asyncio.gather(
            self.get_klines(coin, "1m", 30),
            self.get_klines(coin, "5m", 30),
            self.get_spot_price(coin),
            self.get_funding_rate(symbol),
        )

        if not candles_1m or not candles_5m or spot_price is None:
            logger.warning("Insufficient crypto data", coin=coin)
            return None

        # Compute individual signals
        momentum_5m = self._compute_momentum(candles_5m, lookback=1)  # last 5-min candle
        momentum_1m = self._compute_momentum(candles_1m, lookback=5)  # last 5 × 1-min candles
        funding_signal = self._compute_funding_signal(funding_rate)
        mean_reversion = self._compute_mean_reversion(candles_1m)
        vol_multiplier = self._compute_volatility_signal(candles_5m, kalshi_price)

        # Use dynamic weights from signal scorer if available, else hardcoded
        try:
            from app.services.signal_scorer import get_signal_scorer
            dw = get_signal_scorer().get_dynamic_weights("crypto")
        except Exception:
            dw = {}
        w_mom5 = dw.get("momentum_5m", WEIGHT_MOMENTUM_5M)
        w_mom1 = dw.get("momentum_1m", WEIGHT_MOMENTUM_1M)
        w_fund = dw.get("funding_signal", WEIGHT_FUNDING)
        w_mr = dw.get("mean_reversion", WEIGHT_MEAN_REVERSION)

        # Weighted composite signal: [-1, 1] where positive = bullish
        raw_signal = (
            w_mom5 * momentum_5m
            + w_mom1 * momentum_1m
            + w_fund * funding_signal
            + w_mr * mean_reversion
        )

        # Convert to probability: signal of 0 → 50%, signal of 1 → ~70%, -1 → ~30%
        # Using logistic-style mapping
        p_up = 0.5 + 0.2 * raw_signal

        # Confidence: how aligned are the signals?
        signals = [momentum_5m, momentum_1m, funding_signal, mean_reversion]
        nonzero = [s for s in signals if abs(s) > 0.05]
        if len(nonzero) >= 2:
            # Check if signals agree on direction
            positive = sum(1 for s in nonzero if s > 0)
            negative = sum(1 for s in nonzero if s < 0)
            agreement = max(positive, negative) / len(nonzero)
            avg_magnitude = statistics.mean(abs(s) for s in nonzero)
            confidence = agreement * avg_magnitude * vol_multiplier
        elif len(nonzero) == 1:
            confidence = 0.15 * abs(nonzero[0]) * vol_multiplier
        else:
            confidence = 0.1  # Very low confidence when signals are weak

        confidence = min(1.0, max(0.0, confidence))

        return {
            "coin": coin,
            "p_up": round(p_up, 4),
            "confidence": round(confidence, 4),
            "momentum_5m": round(momentum_5m, 4),
            "momentum_1m": round(momentum_1m, 4),
            "funding_signal": round(funding_signal, 4),
            "mean_reversion": round(mean_reversion, 4),
            "vol_multiplier": round(vol_multiplier, 4),
            "spot_price": spot_price,
            "funding_rate": funding_rate,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def get_all_signals(self, kalshi_prices: dict[str, int] | None = None) -> list[dict[str, Any]]:
        """Get signals for all tracked coins.

        Args:
            kalshi_prices: Optional dict of coin → Kalshi yes_ask price in cents.
        """
        if kalshi_prices is None:
            kalshi_prices = {}

        results: list[dict[str, Any]] = []
        coins = list(SUPPORTED_COINS)
        for i, coin in enumerate(coins):
            if i > 0:
                await asyncio.sleep(1)  # Coinbase rate limit is generous, light delay is fine
            price = kalshi_prices.get(coin, 50)
            signal = await self.get_crypto_signal(coin, kalshi_price=price)
            if signal is not None:
                results.append(signal)
        return results
