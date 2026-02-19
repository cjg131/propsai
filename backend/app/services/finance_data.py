"""
Finance Data Service for S&P 500 and Nasdaq daily Kalshi markets.

Fetches real-time market data from Yahoo Finance (free, no key needed)
and generates directional probability estimates.

Signals:
  1. Intraday momentum — current session trend
  2. VIX level — fear gauge for expected volatility
  3. Pre-market / futures — overnight direction
  4. Moving average position — trend context
  5. Volume profile — conviction behind moves
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

# ── Yahoo Finance endpoints (free, no auth) ─────────────────────────
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"

# Symbols we track
FINANCE_SYMBOLS = {
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "VIX": "^VIX",
    "SP500_FUTURES": "ES=F",
    "NASDAQ_FUTURES": "NQ=F",
    "DXY": "DX-Y.NYB",  # Dollar index — inverse correlation
}

# ── Signal weights ──────────────────────────────────────────────────
WEIGHT_INTRADAY_MOMENTUM = 0.30
WEIGHT_FUTURES_DIRECTION = 0.25
WEIGHT_VIX_SIGNAL = 0.20
WEIGHT_MA_POSITION = 0.15
WEIGHT_VOLUME = 0.10

# Thresholds
STRONG_INTRADAY_PCT = 0.005  # 0.5% intraday move = strong
VIX_HIGH = 25.0  # VIX above 25 = elevated fear
VIX_LOW = 15.0   # VIX below 15 = complacency


class FinanceDataService:
    """Fetches real-time finance data and generates directional signals."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── Raw data fetchers ────────────────────────────────────────────

    async def get_quote(self, symbol: str, range_: str = "1d", interval: str = "5m") -> dict[str, Any] | None:
        """Fetch chart data from Yahoo Finance.
        Returns {price, change_pct, high, low, open, prev_close, volume, candles}.
        """
        client = await self._get_client()
        url = YAHOO_QUOTE_URL.format(symbol=symbol)
        params = {"range": range_, "interval": interval, "includePrePost": "true"}

        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Yahoo Finance fetch failed", symbol=symbol, error=str(e))
            return None

        try:
            result = data["chart"]["result"][0]
            meta = result["meta"]
            indicators = result.get("indicators", {})
            quotes = indicators.get("quote", [{}])[0]
            timestamps = result.get("timestamp", [])

            current_price = meta.get("regularMarketPrice", 0)
            prev_close = meta.get("chartPreviousClose", meta.get("previousClose", 0))

            change_pct = 0.0
            if prev_close and prev_close > 0:
                change_pct = (current_price - prev_close) / prev_close

            # Build candle list
            candles = []
            closes = quotes.get("close", [])
            opens = quotes.get("open", [])
            highs = quotes.get("high", [])
            lows = quotes.get("low", [])
            volumes = quotes.get("volume", [])

            for i in range(len(timestamps)):
                c = closes[i] if i < len(closes) else None
                o = opens[i] if i < len(opens) else None
                h = highs[i] if i < len(highs) else None
                lo = lows[i] if i < len(lows) else None
                v = volumes[i] if i < len(volumes) else None
                if c is not None and o is not None:
                    candles.append({
                        "timestamp": timestamps[i],
                        "open": o,
                        "high": h,
                        "low": lo,
                        "close": c,
                        "volume": v or 0,
                    })

            return {
                "symbol": symbol,
                "price": current_price,
                "prev_close": prev_close,
                "change_pct": change_pct,
                "high": meta.get("regularMarketDayHigh", 0),
                "low": meta.get("regularMarketDayLow", 0),
                "open": meta.get("regularMarketOpen", 0),
                "volume": meta.get("regularMarketVolume", 0),
                "candles": candles,
            }

        except (KeyError, IndexError) as e:
            logger.warning("Yahoo Finance parse failed", symbol=symbol, error=str(e))
            return None

    # ── Signal computation ───────────────────────────────────────────

    def _compute_intraday_momentum(self, quote: dict[str, Any]) -> float:
        """Intraday momentum signal from current session.
        Returns [-1, 1] where positive = bullish.
        """
        change_pct = quote.get("change_pct", 0)
        signal = change_pct / STRONG_INTRADAY_PCT
        return max(-1.0, min(1.0, signal))

    def _compute_futures_signal(self, futures_quote: dict[str, Any] | None) -> float:
        """Futures direction signal (pre-market / overnight).
        Returns [-1, 1] where positive = bullish.
        """
        if not futures_quote:
            return 0.0
        change_pct = futures_quote.get("change_pct", 0)
        signal = change_pct / STRONG_INTRADAY_PCT
        return max(-1.0, min(1.0, signal))

    def _compute_vix_signal(self, vix_quote: dict[str, Any] | None) -> float:
        """VIX-based signal.
        High VIX + rising = bearish for equities.
        Low VIX + falling = bullish for equities.
        Returns [-1, 1] where positive = bullish for equities.
        """
        if not vix_quote:
            return 0.0

        vix_price = vix_quote.get("price", 20)
        vix_change = vix_quote.get("change_pct", 0)

        # VIX level component
        if vix_price > VIX_HIGH:
            level_signal = -0.5  # High fear = bearish
        elif vix_price < VIX_LOW:
            level_signal = 0.3   # Low fear = mildly bullish
        else:
            level_signal = 0.0

        # VIX direction component (inverted: rising VIX = bearish for stocks)
        direction_signal = -vix_change / 0.03  # 3% VIX move → ±1.0

        combined = level_signal + direction_signal * 0.5
        return max(-1.0, min(1.0, combined))

    def _compute_ma_signal(self, candles: list[dict]) -> float:
        """Moving average position signal.
        Price above short MA and short MA above long MA = bullish.
        Returns [-1, 1].
        """
        if len(candles) < 20:
            return 0.0

        closes = [c["close"] for c in candles if c["close"] is not None]
        if len(closes) < 20:
            return 0.0

        current = closes[-1]
        ma_10 = statistics.mean(closes[-10:])
        ma_20 = statistics.mean(closes[-20:])

        signal = 0.0
        if current > ma_10:
            signal += 0.3
        else:
            signal -= 0.3

        if ma_10 > ma_20:
            signal += 0.3
        else:
            signal -= 0.3

        # Distance from MA as strength
        ma_dist = (current - ma_10) / ma_10 if ma_10 > 0 else 0
        signal += ma_dist * 10  # 0.1% distance = 1.0 signal contribution

        return max(-1.0, min(1.0, signal))

    def _compute_volume_signal(self, candles: list[dict]) -> float:
        """Volume confirmation signal.
        High volume on up candles = bullish confirmation.
        Returns confidence multiplier (0.5 to 1.5).
        """
        if len(candles) < 5:
            return 1.0

        recent = candles[-5:]
        up_volume = sum(c["volume"] for c in recent if c["close"] > c["open"])
        down_volume = sum(c["volume"] for c in recent if c["close"] <= c["open"])

        total = up_volume + down_volume
        if total == 0:
            return 1.0

        ratio = up_volume / total  # 0.5 = neutral, >0.5 = bullish volume
        return 0.5 + ratio  # Range: 0.5 to 1.5

    async def get_index_signal(self, index: str = "SP500") -> dict[str, Any] | None:
        """Generate a directional signal for S&P 500 or Nasdaq.

        Args:
            index: "SP500" or "NASDAQ"

        Returns:
            {
                "index": str,
                "p_up": float,  # probability of closing up today (0-1)
                "confidence": float,
                "intraday_momentum": float,
                "futures_signal": float,
                "vix_signal": float,
                "ma_signal": float,
                "vol_multiplier": float,
                "current_price": float,
                "change_pct": float,
                "timestamp": str,
            }
        """
        symbol = FINANCE_SYMBOLS.get(index)
        futures_symbol = FINANCE_SYMBOLS.get(f"{index}_FUTURES")
        vix_symbol = FINANCE_SYMBOLS.get("VIX")

        if not symbol:
            return None

        # Fetch all data in parallel
        async def _noop() -> None:
            return None

        tasks = [self.get_quote(symbol, range_="1d", interval="5m")]
        tasks.append(self.get_quote(futures_symbol, range_="1d", interval="5m") if futures_symbol else _noop())
        tasks.append(self.get_quote(vix_symbol, range_="1d", interval="5m") if vix_symbol else _noop())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        index_quote = results[0] if not isinstance(results[0], Exception) else None
        futures_quote = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else None
        vix_quote = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else None

        if not index_quote:
            logger.warning("No index data", index=index)
            return None

        candles = index_quote.get("candles", [])

        # Compute signals
        intraday = self._compute_intraday_momentum(index_quote)
        futures = self._compute_futures_signal(futures_quote)
        vix = self._compute_vix_signal(vix_quote)
        ma = self._compute_ma_signal(candles)
        vol_mult = self._compute_volume_signal(candles)

        # Weighted composite
        raw_signal = (
            WEIGHT_INTRADAY_MOMENTUM * intraday
            + WEIGHT_FUTURES_DIRECTION * futures
            + WEIGHT_VIX_SIGNAL * vix
            + WEIGHT_MA_POSITION * ma
        )

        # Convert to probability
        p_up = 0.5 + 0.2 * raw_signal

        # Confidence from signal agreement
        signals = [intraday, futures, vix, ma]
        nonzero = [s for s in signals if abs(s) > 0.05]
        if len(nonzero) >= 2:
            positive = sum(1 for s in nonzero if s > 0)
            negative = sum(1 for s in nonzero if s < 0)
            agreement = max(positive, negative) / len(nonzero)
            avg_mag = statistics.mean(abs(s) for s in nonzero)
            confidence = agreement * avg_mag * vol_mult
        else:
            confidence = 0.1

        confidence = min(1.0, max(0.0, confidence))

        return {
            "index": index,
            "p_up": round(p_up, 4),
            "confidence": round(confidence, 4),
            "intraday_momentum": round(intraday, 4),
            "futures_signal": round(futures, 4),
            "vix_signal": round(vix, 4),
            "ma_signal": round(ma, 4),
            "vol_multiplier": round(vol_mult, 4),
            "current_price": index_quote["price"],
            "change_pct": round(index_quote["change_pct"], 6),
            "vix_level": vix_quote["price"] if vix_quote else None,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def get_all_signals(self) -> list[dict[str, Any]]:
        """Get signals for both S&P 500 and Nasdaq."""
        tasks = [
            self.get_index_signal("SP500"),
            self.get_index_signal("NASDAQ"),
        ]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    @staticmethod
    def _normal_cdf(x: float, mean: float, std: float) -> float:
        """P(X <= x) for normal distribution using math.erf."""
        if std <= 0:
            return 1.0 if x >= mean else 0.0
        z = (x - mean) / std
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    def get_bracket_probability(
        self,
        current_price: float,
        bracket_low: float,
        bracket_high: float,
        days_to_expiry: float,
        vix_level: float | None = None,
    ) -> float:
        """
        Compute the probability that an index closes within [bracket_low, bracket_high]
        using a Gaussian price distribution.

        Uses log-normal returns: sigma = current_price * daily_vol * sqrt(days_to_expiry)
        Daily vol is estimated from VIX (VIX/16 ≈ 1-day sigma as a fraction) or defaults to 0.8%.

        Returns probability in [0, 1].
        """
        if current_price <= 0 or days_to_expiry <= 0:
            return 0.0

        if vix_level and vix_level > 0:
            daily_vol_frac = vix_level / (100.0 * 16.0)
        else:
            daily_vol_frac = 0.008

        sigma = current_price * daily_vol_frac * math.sqrt(days_to_expiry)
        sigma = max(sigma, current_price * 0.003)

        prob = (
            self._normal_cdf(bracket_high, current_price, sigma)
            - self._normal_cdf(bracket_low, current_price, sigma)
        )
        return max(0.01, min(0.99, prob))
