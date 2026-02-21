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

# CME FedWatch — public JSON endpoint for Fed meeting probabilities
# Returns market-implied probabilities for each rate outcome at next FOMC meeting
CME_FEDWATCH_URL = "https://www.cmegroup.com/CmeWS/mvc/ProductCalendar/Future/FedFundsFutures.json"
CME_FEDWATCH_PROBS_URL = "https://www.cmegroup.com/CmeWS/mvc/MeetingCalendar/FedFundsFutures.json"

# US Treasury yield curve (free, no key)
TREASURY_YIELD_URL = "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/2026/all?type=daily_treasury_yield_curve&field_tdr_date_value=2026&download=true"

# Series IDs for FRED
FRED_SERIES = {
    "CPI": "CPIAUCSL",           # CPI-U All Items
    "CORE_CPI": "CPILFESL",      # Core CPI (ex food & energy)
    "PCE": "PCEPI",              # PCE Price Index (Fed's preferred inflation measure)
    "CORE_PCE": "PCEPILFE",      # Core PCE (ex food & energy)
    "FED_FUNDS": "FEDFUNDS",     # Effective Federal Funds Rate
    "UNEMPLOYMENT": "UNRATE",     # Unemployment Rate
    "NONFARM_PAYROLLS": "PAYEMS", # Total Nonfarm Payrolls
    "GAS_PRICE": "GASREGW",      # Regular Gas Price Weekly
    "GDP": "GDP",                 # Gross Domestic Product
    "RETAIL_SALES": "RSXFS",     # Advance Retail Sales (ex food services)
    "CONSUMER_SENTIMENT": "UMCSENT", # U of Michigan Consumer Sentiment
    "T10Y": "DGS10",             # 10-Year Treasury Constant Maturity Rate
    "T2Y": "DGS2",               # 2-Year Treasury Constant Maturity Rate
    "JOLTS": "JTSJOL",           # Job Openings (JOLTS)
    "INITIAL_CLAIMS": "ICSA",    # Initial Jobless Claims (weekly)
    "HOUSING_STARTS": "HOUST",   # Housing Starts
    "ISM_MFG": "MANEMP",         # Manufacturing Employment (proxy for ISM)
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

    # ── CME FedWatch (market-implied Fed probabilities) ─────────────

    async def get_fedwatch_probabilities(self) -> dict[str, float] | None:
        """Fetch market-implied Fed rate probabilities from CME FedWatch.

        Returns {"p_cut": float, "p_hold": float, "p_hike": float} for next meeting.
        These are derived from Fed Funds futures pricing — far more accurate than heuristics.
        """
        cache_key = "fedwatch"
        import time
        now = time.time()
        if cache_key in self._cache and now - self._cache_ts.get(cache_key, 0) < 3600:
            return self._cache[cache_key]

        client = await self._get_client()
        try:
            # CME FedWatch public API — returns probabilities for each FOMC meeting
            url = "https://www.cmegroup.com/CmeWS/mvc/MeetingCalendar/FedFundsFutures.json"
            resp = await client.get(url, headers={"Referer": "https://www.cmegroup.com/"})
            resp.raise_for_status()
            data = resp.json()

            # Find the next upcoming meeting
            meetings = data if isinstance(data, list) else data.get("meetings", [])
            next_meeting = None
            for m in meetings:
                date_str = m.get("meetingDate", m.get("date", ""))
                if not date_str:
                    continue
                try:
                    mdate = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                    from datetime import date as _date
                    if mdate >= _date.today():
                        next_meeting = m
                        break
                except ValueError:
                    continue

            if not next_meeting:
                return None

            # Extract probabilities — CME returns them as percentages
            probs_raw = next_meeting.get("probabilities", next_meeting.get("prob", {}))
            if not probs_raw:
                return None

            # CME format: list of {"change": "-25", "probability": 85.2} or dict
            p_cut = 0.0
            p_hold = 0.0
            p_hike = 0.0

            if isinstance(probs_raw, list):
                for entry in probs_raw:
                    change = int(entry.get("change", 0) or 0)
                    prob = float(entry.get("probability", 0) or 0) / 100.0
                    if change < 0:
                        p_cut += prob
                    elif change > 0:
                        p_hike += prob
                    else:
                        p_hold += prob
            elif isinstance(probs_raw, dict):
                p_cut = float(probs_raw.get("cut", probs_raw.get("decrease", 0)) or 0) / 100.0
                p_hold = float(probs_raw.get("hold", probs_raw.get("unchanged", 0)) or 0) / 100.0
                p_hike = float(probs_raw.get("hike", probs_raw.get("increase", 0)) or 0) / 100.0

            # Normalize
            total = p_cut + p_hold + p_hike
            if total > 0:
                p_cut /= total
                p_hold /= total
                p_hike /= total
            else:
                return None

            result = {
                "p_cut": round(p_cut, 4),
                "p_hold": round(p_hold, 4),
                "p_hike": round(p_hike, 4),
                "meeting_date": next_meeting.get("meetingDate", ""),
                "source": "cme_fedwatch",
            }
            self._cache[cache_key] = result
            self._cache_ts[cache_key] = now
            return result

        except Exception as e:
            logger.warning("CME FedWatch fetch failed", error=str(e))
            return None

    # ── Fed Funds Rate ───────────────────────────────────────────────

    async def get_fed_funds_signal(self) -> dict[str, Any] | None:
        """Generate Fed Funds rate signal.

        Uses CME FedWatch market-implied probabilities as primary source.
        Falls back to FRED data + heuristic if FedWatch unavailable.
        Kalshi markets: "Will the Fed cut/raise rates at next meeting?"
        """
        fed_data = await self.get_fred_series("FEDFUNDS", limit=6)
        if not fed_data:
            return None

        current_rate = fed_data[0]["value"]
        prev_rate = fed_data[1]["value"] if len(fed_data) > 1 else current_rate
        rate_change = current_rate - prev_rate

        # PRIMARY: CME FedWatch market-implied probabilities
        fedwatch = await self.get_fedwatch_probabilities()
        source = "cme_fedwatch"

        if fedwatch:
            p_cut = fedwatch["p_cut"]
            p_hold = fedwatch["p_hold"]
            p_hike = fedwatch["p_hike"]
            # FedWatch is highly accurate — boost confidence
            confidence_boost = 0.25
        else:
            # FALLBACK: heuristic based on CPI trend
            cpi_signal = await self.get_cpi_signal()
            cpi_yoy = cpi_signal["yoy_change"] if cpi_signal and cpi_signal.get("yoy_change") else 0.03
            source = "heuristic"
            confidence_boost = 0.0

            if cpi_yoy > 0.035:
                p_hike, p_hold, p_cut = 0.3, 0.6, 0.1
            elif cpi_yoy > 0.025:
                p_hike, p_hold, p_cut = 0.1, 0.7, 0.2
            else:
                p_hike, p_hold, p_cut = 0.05, 0.45, 0.5

        return {
            "type": "fed_funds",
            "current_rate": current_rate,
            "prev_rate": prev_rate,
            "rate_change": rate_change,
            "p_hike": round(p_hike, 4),
            "p_hold": round(p_hold, 4),
            "p_cut": round(p_cut, 4),
            "confidence_boost": confidence_boost,
            "source": source,
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

    # ── PCE Inflation (Fed's preferred measure) ─────────────────────

    async def get_pce_signal(self) -> dict[str, Any] | None:
        """Fetch PCE inflation data — the Fed's preferred inflation gauge.

        PCE is more comprehensive than CPI and is what the Fed actually targets (2%).
        Used to supplement CPI signals for Kalshi inflation markets.
        """
        pce_data = await self.get_fred_series("PCEPI", limit=13)
        core_pce_data = await self.get_fred_series("PCEPILFE", limit=13)

        if not pce_data or len(pce_data) < 2:
            return None

        latest = pce_data[0]["value"]
        prev = pce_data[1]["value"]
        year_ago = pce_data[12]["value"] if len(pce_data) > 12 else None

        mom_change = (latest - prev) / prev if prev > 0 else 0
        yoy_change = (latest - year_ago) / year_ago if year_ago and year_ago > 0 else None

        core_yoy = None
        if core_pce_data and len(core_pce_data) > 12:
            core_latest = core_pce_data[0]["value"]
            core_year_ago = core_pce_data[12]["value"]
            if core_year_ago > 0:
                core_yoy = (core_latest - core_year_ago) / core_year_ago

        return {
            "type": "pce",
            "mom_change": round(mom_change, 6),
            "yoy_change": round(yoy_change, 4) if yoy_change else None,
            "core_yoy": round(core_yoy, 4) if core_yoy else None,
            "latest_date": pce_data[0]["date"],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # ── Treasury Yield Curve ──────────────────────────────────────────

    async def get_treasury_signal(self) -> dict[str, Any] | None:
        """Fetch 2yr and 10yr Treasury yields from FRED.

        Yield curve shape predicts recession probability and Fed policy direction.
        Inverted curve (2yr > 10yr) = recession signal = more likely Fed cuts.
        """
        t10_data, t2_data = await asyncio.gather(
            self.get_fred_series("DGS10", limit=5),
            self.get_fred_series("DGS2", limit=5),
        )

        if not t10_data:
            return None

        t10 = t10_data[0]["value"]
        t2 = t2_data[0]["value"] if t2_data else None
        spread = (t10 - t2) if t2 is not None else None

        # Yield curve inversion = recession signal
        is_inverted = spread is not None and spread < 0

        # 10yr trend (rising = hawkish expectations, falling = dovish)
        t10_trend = 0.0
        if len(t10_data) >= 3:
            t10_trend = t10_data[0]["value"] - t10_data[2]["value"]

        return {
            "type": "treasury",
            "t10y": round(t10, 4),
            "t2y": round(t2, 4) if t2 else None,
            "spread_2s10s": round(spread, 4) if spread is not None else None,
            "is_inverted": is_inverted,
            "t10_trend_3d": round(t10_trend, 4),
            "latest_date": t10_data[0]["date"],
            "timestamp": datetime.now(UTC).isoformat(),
        }

    # ── Consumer Sentiment & Retail Sales ────────────────────────────

    async def get_consumer_signal(self) -> dict[str, Any] | None:
        """Fetch U of Michigan Consumer Sentiment and Retail Sales from FRED.

        Consumer sentiment leads spending by 1-2 months.
        Retail sales directly impacts inflation and GDP readings.
        """
        sentiment_data, retail_data, claims_data = await asyncio.gather(
            self.get_fred_series("UMCSENT", limit=3),
            self.get_fred_series("RSXFS", limit=3),
            self.get_fred_series("ICSA", limit=4),
        )

        result: dict[str, Any] = {"type": "consumer", "timestamp": datetime.now(UTC).isoformat()}

        if sentiment_data:
            sentiment = sentiment_data[0]["value"]
            prev_sentiment = sentiment_data[1]["value"] if len(sentiment_data) > 1 else sentiment
            result["consumer_sentiment"] = sentiment
            result["sentiment_change"] = round(sentiment - prev_sentiment, 1)
            result["sentiment_direction"] = "improving" if sentiment > prev_sentiment else "declining"

        if retail_data and len(retail_data) >= 2:
            retail_latest = retail_data[0]["value"]
            retail_prev = retail_data[1]["value"]
            retail_mom = (retail_latest - retail_prev) / retail_prev if retail_prev > 0 else 0
            result["retail_sales_mom"] = round(retail_mom, 4)
            result["retail_direction"] = "strong" if retail_mom > 0.003 else ("weak" if retail_mom < -0.003 else "neutral")

        if claims_data and len(claims_data) >= 2:
            claims_latest = claims_data[0]["value"]
            claims_prev = claims_data[1]["value"]
            result["initial_claims"] = claims_latest
            result["claims_change"] = round(claims_latest - claims_prev, 0)
            result["claims_trend"] = "rising" if claims_latest > claims_prev else "falling"

        return result if len(result) > 2 else None

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
        volatility is the absolute std dev of the estimate (not a fraction).
        """
        import math
        if volatility <= 0:
            return 1.0 if estimated_value > threshold else 0.0

        z = (estimated_value - threshold) / volatility
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
            "pce": self.get_pce_signal(),
            "treasury": self.get_treasury_signal(),
            "consumer": self.get_consumer_signal(),
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
