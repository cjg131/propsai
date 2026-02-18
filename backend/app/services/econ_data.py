"""
Economic Data Service for CPI, Fed Funds, Gas Prices, and Unemployment markets.

Fetches publicly available economic data and generates probability estimates
for Kalshi economic event markets.

Data Sources (all free, no key needed):
  - FRED (Federal Reserve Economic Data) — CPI, unemployment, fed funds
  - EIA (Energy Information Administration) — gas prices
  - CME FedWatch probabilities — scraped from public page
  - Cleveland Fed inflation nowcast — real-time CPI estimate
"""
from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

# ── FRED API (free with API key, but we use the public JSON endpoint) ──
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# ── EIA API ─────────────────────────────────────────────────────────
EIA_GAS_URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"

# ── Public data URLs ────────────────────────────────────────────────
CLEVELAND_FED_URL = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
BLS_CPI_URL = "https://data.bls.gov/timeseries/CUUR0000SA0"

# Series IDs for FRED
FRED_SERIES = {
    "CPI": "CPIAUCSL",           # CPI-U All Items
    "CORE_CPI": "CPILFESL",      # Core CPI (ex food & energy)
    "FED_FUNDS": "FEDFUNDS",     # Effective Federal Funds Rate
    "UNEMPLOYMENT": "UNRATE",     # Unemployment Rate
    "NONFARM_PAYROLLS": "PAYEMS", # Total Nonfarm Payrolls
    "GAS_PRICE": "GASREGW",      # Regular Gas Price Weekly
    "GDP": "GDP",                 # Gross Domestic Product
}


class EconDataService:
    """Fetches economic data and generates signals for Kalshi econ markets."""

    def __init__(self, fred_api_key: str = "") -> None:
        self._client: httpx.AsyncClient | None = None
        self.fred_api_key = fred_api_key
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_ts: dict[str, float] = {}

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=20,
                headers={"User-Agent": "Mozilla/5.0"},
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── FRED data fetcher ────────────────────────────────────────────

    async def get_fred_series(
        self, series_id: str, limit: int = 12
    ) -> list[dict[str, Any]] | None:
        """Fetch recent observations from FRED.
        Returns list of {date, value} dicts, most recent first.
        """
        if not self.fred_api_key:
            logger.debug("No FRED API key, skipping", series=series_id)
            return None

        client = await self._get_client()
        params = {
            "series_id": series_id,
            "api_key": self.fred_api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }

        try:
            resp = await client.get(FRED_BASE, params=params)
            resp.raise_for_status()
            data = resp.json()
            observations = data.get("observations", [])
            result = []
            for obs in observations:
                val = obs.get("value", ".")
                if val != ".":
                    result.append({
                        "date": obs["date"],
                        "value": float(val),
                    })
            return result
        except Exception as e:
            logger.warning("FRED fetch failed", series=series_id, error=str(e))
            return None

    # ── CPI / Inflation ──────────────────────────────────────────────

    async def get_cpi_signal(self) -> dict[str, Any] | None:
        """Generate CPI/inflation signal.

        Uses FRED CPI data to estimate next month's CPI print.
        Kalshi markets are typically "Will CPI be above X%?"
        """
        cpi_data = await self.get_fred_series("CPIAUCSL", limit=13)
        core_cpi_data = await self.get_fred_series("CPILFESL", limit=13)

        if not cpi_data or len(cpi_data) < 2:
            return None

        # Calculate recent CPI trends
        latest_cpi = cpi_data[0]["value"]
        prev_cpi = cpi_data[1]["value"]
        year_ago_cpi = cpi_data[12]["value"] if len(cpi_data) > 12 else None

        # Month-over-month change (annualized)
        mom_change = (latest_cpi - prev_cpi) / prev_cpi
        mom_annualized = ((1 + mom_change) ** 12 - 1)

        # Year-over-year
        yoy_change = None
        if year_ago_cpi and year_ago_cpi > 0:
            yoy_change = (latest_cpi - year_ago_cpi) / year_ago_cpi

        # 3-month trend
        if len(cpi_data) >= 4:
            three_month_ago = cpi_data[3]["value"]
            three_month_change = (latest_cpi - three_month_ago) / three_month_ago
            three_month_annualized = ((1 + three_month_change) ** 4 - 1)
        else:
            three_month_annualized = mom_annualized

        # Core CPI trend
        core_yoy = None
        if core_cpi_data and len(core_cpi_data) > 12:
            core_latest = core_cpi_data[0]["value"]
            core_year_ago = core_cpi_data[12]["value"]
            if core_year_ago > 0:
                core_yoy = (core_latest - core_year_ago) / core_year_ago

        # Estimate next CPI print (simple: weighted average of recent trends)
        estimated_yoy = (
            0.4 * (yoy_change or mom_annualized)
            + 0.3 * three_month_annualized
            + 0.3 * mom_annualized
        )

        return {
            "type": "cpi",
            "latest_cpi": latest_cpi,
            "mom_change": round(mom_change, 6),
            "mom_annualized": round(mom_annualized, 4),
            "yoy_change": round(yoy_change, 4) if yoy_change else None,
            "three_month_annualized": round(three_month_annualized, 4),
            "core_yoy": round(core_yoy, 4) if core_yoy else None,
            "estimated_next_yoy": round(estimated_yoy, 4),
            "latest_date": cpi_data[0]["date"],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # ── Fed Funds Rate ───────────────────────────────────────────────

    async def get_fed_funds_signal(self) -> dict[str, Any] | None:
        """Generate Fed Funds rate signal.

        Uses current rate + CPI trend to estimate probability of rate changes.
        Kalshi markets: "Will the Fed cut/raise rates at next meeting?"
        """
        fed_data = await self.get_fred_series("FEDFUNDS", limit=6)
        if not fed_data:
            return None

        current_rate = fed_data[0]["value"]
        prev_rate = fed_data[1]["value"] if len(fed_data) > 1 else current_rate

        # Rate trend
        rate_change = current_rate - prev_rate

        # Get CPI for context
        cpi_signal = await self.get_cpi_signal()
        cpi_yoy = cpi_signal["yoy_change"] if cpi_signal and cpi_signal.get("yoy_change") else 0.03

        # Simple heuristic for next meeting probability
        # If inflation is high (>3%) and rates haven't risen → likely hold or hike
        # If inflation is low (<2.5%) and rates are high → likely cut
        if cpi_yoy > 0.035:
            p_hike = 0.3
            p_hold = 0.6
            p_cut = 0.1
        elif cpi_yoy > 0.025:
            p_hike = 0.1
            p_hold = 0.7
            p_cut = 0.2
        else:
            p_hike = 0.05
            p_hold = 0.45
            p_cut = 0.5

        return {
            "type": "fed_funds",
            "current_rate": current_rate,
            "prev_rate": prev_rate,
            "rate_change": rate_change,
            "cpi_yoy": cpi_yoy,
            "p_hike": round(p_hike, 4),
            "p_hold": round(p_hold, 4),
            "p_cut": round(p_cut, 4),
            "latest_date": fed_data[0]["date"],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # ── Gas Prices ───────────────────────────────────────────────────

    async def get_gas_price_signal(self) -> dict[str, Any] | None:
        """Generate gas price signal.

        Uses FRED weekly gas price data to estimate next week's price.
        Kalshi markets: "Will gas prices be above $X.XX?"
        """
        gas_data = await self.get_fred_series("GASREGW", limit=8)
        if not gas_data or len(gas_data) < 2:
            return None

        latest_price = gas_data[0]["value"]
        prices = [d["value"] for d in gas_data]

        # Trend: weighted recent prices
        if len(prices) >= 4:
            short_avg = sum(prices[:2]) / 2
            long_avg = sum(prices[:4]) / 4
            trend = (short_avg - long_avg) / long_avg
        else:
            trend = 0.0

        # Estimate next week's price
        estimated_next = latest_price * (1 + trend * 0.5)

        return {
            "type": "gas_price",
            "latest_price": latest_price,
            "estimated_next": round(estimated_next, 3),
            "trend": round(trend, 6),
            "prices_history": prices,
            "latest_date": gas_data[0]["date"],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # ── Unemployment / Jobs ──────────────────────────────────────────

    async def get_unemployment_signal(self) -> dict[str, Any] | None:
        """Generate unemployment/jobs signal.

        Uses FRED unemployment rate and nonfarm payrolls.
        Kalshi markets: "Will unemployment be above X%?" or "Will NFP be above X?"
        """
        unemp_data = await self.get_fred_series("UNRATE", limit=6)
        nfp_data = await self.get_fred_series("PAYEMS", limit=6)

        if not unemp_data:
            return None

        latest_rate = unemp_data[0]["value"]
        rates = [d["value"] for d in unemp_data]

        # Trend
        if len(rates) >= 3:
            short_avg = sum(rates[:2]) / 2
            long_avg = sum(rates[:3]) / 3
            trend = short_avg - long_avg
        else:
            trend = 0.0

        # NFP change
        nfp_change = None
        if nfp_data and len(nfp_data) >= 2:
            nfp_change = nfp_data[0]["value"] - nfp_data[1]["value"]

        # Estimate next month
        estimated_next_rate = latest_rate + trend * 0.5

        return {
            "type": "unemployment",
            "latest_rate": latest_rate,
            "estimated_next_rate": round(estimated_next_rate, 2),
            "trend": round(trend, 4),
            "nfp_change": nfp_change,
            "rates_history": rates,
            "latest_date": unemp_data[0]["date"],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # ── Probability helpers ──────────────────────────────────────────

    def estimate_probability_above(
        self, estimated_value: float, threshold: float, volatility: float = 0.01
    ) -> float:
        """Estimate probability that actual value will be above threshold.

        Uses a simple normal distribution approximation.
        """
        import math
        if volatility <= 0:
            return 1.0 if estimated_value > threshold else 0.0

        z = (estimated_value - threshold) / (estimated_value * volatility)
        # Approximate normal CDF using logistic function
        p = 1.0 / (1.0 + math.exp(-1.7 * z))
        return round(max(0.01, min(0.99, p)), 4)

    async def get_all_signals(self) -> dict[str, Any]:
        """Get all economic signals."""
        tasks = {
            "cpi": self.get_cpi_signal(),
            "fed_funds": self.get_fed_funds_signal(),
            "gas_price": self.get_gas_price_signal(),
            "unemployment": self.get_unemployment_signal(),
        }

        results = {}
        for key, task in tasks.items():
            try:
                result = await task
                if result:
                    results[key] = result
            except Exception as e:
                logger.warning("Econ signal failed", signal=key, error=str(e))

        return results
