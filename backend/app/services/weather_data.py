"""
Weather data pipeline for Kalshi weather market trading.
Fetches forecasts from 4 sources: NWS, Open-Meteo Ensemble, Tomorrow.io, Visual Crossing.
Builds consensus probability distributions for temperature, snowfall, rainfall.
"""
from __future__ import annotations

import asyncio
import math
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

# Kalshi weather cities → NWS grid points and coordinates
CITY_CONFIGS = {
    "NYC": {"lat": 40.7128, "lon": -74.0060, "nws_office": "OKX", "nws_grid": "33,37", "name": "New York", "tz": "America/New_York"},
    "MIA": {"lat": 25.7617, "lon": -80.1918, "nws_office": "MFL", "nws_grid": "110,50", "name": "Miami", "tz": "America/New_York"},
    "LAX": {"lat": 34.0522, "lon": -118.2437, "nws_office": "LOX", "nws_grid": "154,44", "name": "Los Angeles", "tz": "America/Los_Angeles"},
    "CHI": {"lat": 41.8781, "lon": -87.6298, "nws_office": "LOT", "nws_grid": "76,73", "name": "Chicago", "tz": "America/Chicago"},
    "AUS": {"lat": 30.2672, "lon": -97.7431, "nws_office": "EWX", "nws_grid": "156,91", "name": "Austin", "tz": "America/Chicago"},
    "DFW": {"lat": 32.7767, "lon": -96.7970, "nws_office": "FWD", "nws_grid": "80,103", "name": "Dallas", "tz": "America/Chicago"},
    "PHL": {"lat": 39.9526, "lon": -75.1652, "nws_office": "PHI", "nws_grid": "49,75", "name": "Philadelphia", "tz": "America/New_York"},
    "DEN": {"lat": 39.7392, "lon": -104.9903, "nws_office": "BOU", "nws_grid": "62,60", "name": "Denver", "tz": "America/Denver"},
    "SEA": {"lat": 47.6062, "lon": -122.3321, "nws_office": "SEW", "nws_grid": "124,67", "name": "Seattle", "tz": "America/Los_Angeles"},
    "SFO": {"lat": 37.7749, "lon": -122.4194, "nws_office": "MTR", "nws_grid": "85,105", "name": "San Francisco", "tz": "America/Los_Angeles"},
    "DCA": {"lat": 38.9072, "lon": -77.0369, "nws_office": "LWX", "nws_grid": "97,71", "name": "Washington DC", "tz": "America/New_York"},
    "SLC": {"lat": 40.7608, "lon": -111.8910, "nws_office": "SLC", "nws_grid": "97,175", "name": "Salt Lake City", "tz": "America/Denver"},
    "ATL": {"lat": 33.7490, "lon": -84.3880, "nws_office": "FFC", "nws_grid": "50,86", "name": "Atlanta", "tz": "America/New_York"},
    "HOU": {"lat": 29.7604, "lon": -95.3698, "nws_office": "HGX", "nws_grid": "65,97", "name": "Houston", "tz": "America/Chicago"},
    "BOS": {"lat": 42.3601, "lon": -71.0589, "nws_office": "BOX", "nws_grid": "71,90", "name": "Boston", "tz": "America/New_York"},
    "LAS": {"lat": 36.1699, "lon": -115.1398, "nws_office": "VEF", "nws_grid": "126,97", "name": "Las Vegas", "tz": "America/Los_Angeles"},
    "PHX": {"lat": 33.4484, "lon": -112.0740, "nws_office": "PSR", "nws_grid": "159,56", "name": "Phoenix", "tz": "America/Phoenix"},
    "MSP": {"lat": 44.9778, "lon": -93.2650, "nws_office": "MPX", "nws_grid": "107,71", "name": "Minneapolis", "tz": "America/Chicago"},
    "NOL": {"lat": 29.9511, "lon": -90.0715, "nws_office": "LIX", "nws_grid": "76,76", "name": "New Orleans", "tz": "America/Chicago"},
    "DET": {"lat": 42.3314, "lon": -83.0458, "nws_office": "DTX", "nws_grid": "65,33", "name": "Detroit", "tz": "America/Detroit"},
}


class NWSClient:
    """National Weather Service API client (free, no key needed)."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "PropsAI-Weather-Agent/1.0 (contact@propsai.com)"},
        )

    async def get_forecast(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Get NWS point forecast for a city on a specific date. Returns high/low temps."""
        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        if target_date is None:
            target_date = datetime.now(UTC).date()

        try:
            url = f"https://api.weather.gov/gridpoints/{config['nws_office']}/{config['nws_grid']}/forecast"
            resp = await self._http.get(url)
            resp.raise_for_status()
            data = resp.json()

            periods = data.get("properties", {}).get("periods", [])
            if not periods:
                return None

            # Find daytime (high) and nighttime (low) periods matching target_date
            result: dict[str, Any] = {"source": "nws", "city": city_key}
            target_str = target_date.isoformat()
            for period in periods[:14]:
                start = period.get("startTime", "")
                temp = period.get("temperature")
                if temp is None:
                    continue
                if start[:10] == target_str:
                    if period.get("isDaytime", False):
                        result["high_temp_f"] = temp
                        result["period_name"] = period.get("name", "")
                        result["short_forecast"] = period.get("shortForecast", "")
                        result["date"] = target_str
                    else:
                        result["low_temp_f"] = temp
                # Overnight low for target date may start the night before
                # NWS overnight periods start at 6pm and run to 6am next day
                # so a period starting on target_date-1 night covers target_date morning low
                elif not period.get("isDaytime", False) and "low_temp_f" not in result:
                    from datetime import timedelta as _td
                    prev_str = (target_date - _td(days=1)).isoformat()
                    if start[:10] == prev_str:
                        result["low_temp_f"] = temp

            # Fallback: first daytime period if no date match
            if "high_temp_f" not in result:
                for period in periods[:6]:
                    if period.get("isDaytime", False) and period.get("temperature") is not None:
                        result["high_temp_f"] = period.get("temperature")
                        result["period_name"] = period.get("name", "")
                        result["short_forecast"] = period.get("shortForecast", "")
                        result["date"] = period.get("startTime", "")[:10]
                        break

            return result if "high_temp_f" in result else None

        except Exception as e:
            logger.warning("NWS forecast failed", city=city_key, error=str(e))
            return None

    async def get_hourly_forecast(self, city_key: str) -> list[dict[str, Any]]:
        """Get NWS hourly forecast for a city."""
        config = CITY_CONFIGS.get(city_key)
        if not config:
            return []

        try:
            url = f"https://api.weather.gov/gridpoints/{config['nws_office']}/{config['nws_grid']}/forecast/hourly"
            resp = await self._http.get(url)
            resp.raise_for_status()
            data = resp.json()

            periods = data.get("properties", {}).get("periods", [])
            return [
                {
                    "time": p.get("startTime"),
                    "temp_f": p.get("temperature"),
                    "precip_pct": p.get("probabilityOfPrecipitation", {}).get("value", 0),
                    "wind_speed": p.get("windSpeed", ""),
                    "short_forecast": p.get("shortForecast", ""),
                }
                for p in periods[:48]  # Next 48 hours
            ]
        except Exception as e:
            logger.warning("NWS hourly forecast failed", city=city_key, error=str(e))
            return []

    async def close(self) -> None:
        await self._http.aclose()


class OpenMeteoClient:
    """Open-Meteo Ensemble API client (free, no key needed)."""

    _cache: dict[str, dict] = {}
    _cache_ts: dict[str, float] = {}
    _CACHE_TTL: float = 7200.0  # 2 hours — avoids 429 rate limiting

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=15.0)

    async def get_ensemble_forecast(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """
        Get ensemble forecast with probability distributions for a specific date.
        Uses multiple weather models for uncertainty estimation.
        Results are cached for 2 hours to avoid 429 rate limiting.
        """
        import time as _time
        now = _time.time()
        if target_date is None:
            target_date = datetime.now(UTC).date()
        cache_key = f"{city_key}_{target_date.isoformat()}"
        if cache_key in OpenMeteoClient._cache and (now - OpenMeteoClient._cache_ts.get(cache_key, 0)) < OpenMeteoClient._CACHE_TTL:
            return OpenMeteoClient._cache[cache_key]

        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        try:
            # Use ensemble API for probability distributions
            resp = await self._http.get(
                "https://ensemble-api.open-meteo.com/v1/ensemble",
                params={
                    "latitude": config["lat"],
                    "longitude": config["lon"],
                    "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum",
                    "temperature_unit": "fahrenheit",
                    "precipitation_unit": "inch",
                    "timezone": config.get("tz", "America/New_York"),
                    "forecast_days": 7,
                    "models": "icon_seamless,gfs_seamless,ecmwf_ifs025,gem_global,bom_access_global_ensemble",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            times = daily.get("time", [])
            if not times:
                return None

            # Find the index for the target date
            target_str = target_date.isoformat()
            day_idx = 0
            for i, t in enumerate(times):
                if t == target_str:
                    day_idx = i
                    break

            # Collect all model predictions for target date (both high and low)
            high_temps = []
            low_temps = []
            for key, values in daily.items():
                if key.startswith("temperature_2m_max") and values:
                    if day_idx < len(values) and values[day_idx] is not None:
                        high_temps.append(values[day_idx])
                elif key.startswith("temperature_2m_min") and values:
                    if day_idx < len(values) and values[day_idx] is not None:
                        low_temps.append(values[day_idx])

            if not high_temps:
                return None

            # Calculate statistics from ensemble
            high_temps.sort()
            n = len(high_temps)
            mean_high = sum(high_temps) / n
            median_high = high_temps[n // 2]
            min_high = high_temps[0]
            max_high = high_temps[-1]

            # Percentiles
            p10 = high_temps[max(0, int(n * 0.1))]
            p25 = high_temps[max(0, int(n * 0.25))]
            p75 = high_temps[min(n - 1, int(n * 0.75))]
            p90 = high_temps[min(n - 1, int(n * 0.9))]

            low_temps.sort()
            mean_low = sum(low_temps) / len(low_temps) if low_temps else None

            result = {
                "source": "open_meteo",
                "city": city_key,
                "date": target_str,
                "high_temp_f": round(mean_high, 1),
                "low_temp_f": round(mean_low, 1) if mean_low is not None else None,
                "high_temp_median": round(median_high, 1),
                "high_temp_min": round(min_high, 1),
                "high_temp_max": round(max_high, 1),
                "high_temp_p10": round(p10, 1),
                "high_temp_p25": round(p25, 1),
                "high_temp_p75": round(p75, 1),
                "high_temp_p90": round(p90, 1),
                "ensemble_count": n,
                "all_predictions": [round(t, 1) for t in high_temps],
                "all_low_predictions": [round(t, 1) for t in low_temps],
            }
            OpenMeteoClient._cache[cache_key] = result
            OpenMeteoClient._cache_ts[cache_key] = now
            return result

        except Exception as e:
            logger.warning("Open-Meteo ensemble failed", city=city_key, error=str(e))
            return None

    async def close(self) -> None:
        await self._http.aclose()


class TomorrowIOClient:
    """Tomorrow.io API client (free tier — 5-day forecast with percentiles)."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=15.0)

    async def get_forecast(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Get Tomorrow.io forecast for a specific date."""
        if not self.api_key:
            return None

        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        if target_date is None:
            target_date = datetime.now(UTC).date()
        target_str = target_date.isoformat()

        try:
            resp = await self._http.get(
                "https://api.tomorrow.io/v4/weather/forecast",
                params={
                    "location": f"{config['lat']},{config['lon']}",
                    "apikey": self.api_key,
                    "units": "imperial",
                    "timesteps": "1d",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("timelines", {}).get("daily", [])
            if not daily:
                return None

            # Find the entry matching target_date
            target_day = None
            for day in daily:
                if day.get("time", "")[:10] == target_str:
                    target_day = day
                    break
            if target_day is None:
                target_day = daily[0]  # fallback to first

            values = target_day.get("values", {})

            return {
                "source": "tomorrow_io",
                "city": city_key,
                "date": target_day.get("time", "")[:10],
                "high_temp_f": values.get("temperatureMax"),
                "low_temp_f": values.get("temperatureMin"),
                "precip_prob": values.get("precipitationProbabilityMax"),
                "precip_inches": values.get("precipitationIntensityMax"),
                "humidity": values.get("humidityMax"),
            }

        except Exception as e:
            logger.warning("Tomorrow.io forecast failed", city=city_key, error=str(e))
            return None

    async def close(self) -> None:
        await self._http.aclose()


class VisualCrossingClient:
    """Visual Crossing Weather API client ($35/month)."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=15.0)

    async def get_forecast(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Get Visual Crossing forecast for a specific date."""
        if not self.api_key:
            return None

        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        if target_date is None:
            target_date = datetime.now(UTC).date()
        target_str = target_date.isoformat()

        try:
            location = f"{config['lat']},{config['lon']}"
            resp = await self._http.get(
                f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{target_str}",
                params={
                    "unitGroup": "us",
                    "key": self.api_key,
                    "include": "days",
                    "contentType": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            days = data.get("days", [])
            if not days:
                return None

            day = days[0]
            return {
                "source": "visual_crossing",
                "city": city_key,
                "date": day.get("datetime", ""),
                "high_temp_f": day.get("tempmax"),
                "low_temp_f": day.get("tempmin"),
                "precip_inches": day.get("precip"),
                "precip_prob": day.get("precipprob"),
                "snow_inches": day.get("snow"),
                "humidity": day.get("humidity"),
                "conditions": day.get("conditions", ""),
            }

        except Exception as e:
            logger.warning("Visual Crossing forecast failed", city=city_key, error=str(e))
            return None

    async def close(self) -> None:
        await self._http.aclose()


class WeatherConsensus:
    """
    Combines forecasts from multiple sources into a consensus probability distribution.
    Used to generate trading signals for Kalshi weather markets.
    """

    def __init__(
        self,
        tomorrow_io_key: str = "",
        visual_crossing_key: str = "",
    ) -> None:
        self.nws = NWSClient()
        self.open_meteo = OpenMeteoClient()
        self.tomorrow_io = TomorrowIOClient(api_key=tomorrow_io_key)
        self.visual_crossing = VisualCrossingClient(api_key=visual_crossing_key)

    async def get_all_forecasts(self, city_key: str, target_date: date | None = None) -> dict[str, Any]:
        """Fetch forecasts from all available sources for a city on a specific date.
        Staggers calls to avoid 429 rate limits on paid APIs."""
        if target_date is None:
            target_date = datetime.now(UTC).date()
        forecasts: dict[str, Any] = {"city": city_key, "sources": {}, "target_date": target_date.isoformat()}

        # NWS is free/unlimited — call first
        calls = [
            ("nws", self.nws.get_forecast(city_key, target_date)),
            ("open_meteo", self.open_meteo.get_ensemble_forecast(city_key, target_date)),
            ("tomorrow_io", self.tomorrow_io.get_forecast(city_key, target_date)),
            ("visual_crossing", self.visual_crossing.get_forecast(city_key, target_date)),
        ]

        for name, coro in calls:
            try:
                result = await coro
                if result is not None:
                    forecasts["sources"][name] = result
            except Exception as e:
                logger.warning("Forecast source failed", source=name, city=city_key, error=str(e))
            # Delay between API calls to respect Tomorrow.io free tier (~25 req/hr)
            await asyncio.sleep(2)

        return forecasts

    def _estimate_std_dev(self, forecasts: dict[str, Any], is_bracket: bool = False, market_type: str = "high_temp") -> float:
        """Estimate temperature standard deviation from available data.

        For bracket markets a minimum of 3°F is enforced because weather forecasts
        are never accurate enough to warrant a tighter distribution on a 1-2° window.
        """
        sources = forecasts.get("sources", {})
        is_low = market_type == "low_temp"
        temp_field = "low_temp_f" if is_low else "high_temp_f"
        temps = [d.get(temp_field) for d in sources.values() if d.get(temp_field) is not None]
        if len(temps) < 2:
            return 3.0 if is_bracket else 2.5
        spread = max(temps) - min(temps)
        # Ensemble spread gives better estimate
        ensemble = sources.get("open_meteo", {})
        pred_key = "all_low_predictions" if is_low else "all_predictions"
        preds = ensemble.get(pred_key, [])
        if len(preds) >= 5:
            preds_sorted = sorted(preds)
            n = len(preds_sorted)
            iqr = preds_sorted[int(n * 0.75)] - preds_sorted[int(n * 0.25)]
            std = max(iqr / 1.35, 1.5)
        else:
            std = max(spread / 2, 2.0)
        # Bracket markets: enforce a minimum 3°F floor — no forecast is ever
        # accurate enough to price a 1-2° bracket with std_dev < 3°F
        if is_bracket:
            std = max(std, 3.0)
        return std

    def _normal_cdf(self, x: float, mean: float, std: float) -> float:
        """P(X <= x) for normal distribution."""
        if std <= 0:
            return 1.0 if x >= mean else 0.0
        z = (x - mean) / std
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def build_consensus(
        self,
        forecasts: dict[str, Any],
        strike_type: str = "greater",
        floor_strike: float | None = None,
        cap_strike: float | None = None,
        market_type: str = "high_temp",
    ) -> dict[str, Any]:
        """
        Build a consensus probability for a Kalshi weather market.

        strike_type: 'greater', 'less', or 'between'
        floor_strike: lower bound (used by greater and between)
        cap_strike: upper bound (used by less and between)
        market_type: 'high_temp', 'low_temp', 'rain', 'snow', etc.
        """
        sources = forecasts.get("sources", {})
        if not sources:
            return {"error": "No forecast data available"}

        is_low = market_type == "low_temp"
        temp_field = "low_temp_f" if is_low else "high_temp_f"

        high_temps: list[float] = []
        source_details: list[dict[str, Any]] = []

        for name, data in sources.items():
            temp = data.get(temp_field)
            if temp is not None:
                high_temps.append(temp)
                source_details.append({"source": name, temp_field: temp})

        if not high_temps:
            return {"error": f"No {temp_field} data from any source"}

        n = len(high_temps)
        mean_temp = sum(high_temps) / n
        spread = max(high_temps) - min(high_temps) if n > 1 else 0
        confidence = max(0.0, min(1.0, 1.0 - (spread - 2) / 8))
        std_dev = self._estimate_std_dev(forecasts, is_bracket=(strike_type == "between"), market_type=market_type)

        # Use ensemble predictions if available for empirical probability
        pred_key = "all_low_predictions" if is_low else "all_predictions"
        ensemble_preds = sources.get("open_meteo", {}).get(pred_key, [])

        # Calculate probability based on strike type
        our_prob_yes = 0.0
        label = ""

        if strike_type == "greater" and floor_strike is not None:
            label = f">{floor_strike}°F"
            if ensemble_preds:
                our_prob_yes = sum(1 for t in ensemble_preds if t > floor_strike) / len(ensemble_preds)
            else:
                our_prob_yes = 1.0 - self._normal_cdf(floor_strike, mean_temp, std_dev)

        elif strike_type == "less" and cap_strike is not None:
            label = f"<{cap_strike}°F"
            if ensemble_preds:
                our_prob_yes = sum(1 for t in ensemble_preds if t < cap_strike) / len(ensemble_preds)
            else:
                our_prob_yes = self._normal_cdf(cap_strike, mean_temp, std_dev)

        elif strike_type == "between" and floor_strike is not None and cap_strike is not None:
            label = f"{floor_strike}-{cap_strike}°F"
            if ensemble_preds:
                our_prob_yes = sum(
                    1 for t in ensemble_preds if floor_strike <= t <= cap_strike
                ) / len(ensemble_preds)
            else:
                our_prob_yes = (
                    self._normal_cdf(cap_strike, mean_temp, std_dev)
                    - self._normal_cdf(floor_strike, mean_temp, std_dev)
                )
        else:
            return {"error": f"Invalid strike_type={strike_type} or missing strikes"}

        # Clamp to [0.02, 0.98] — no forecast is ever 0% or 100% certain
        our_prob_yes = max(0.02, min(0.98, our_prob_yes))

        return {
            "city": forecasts.get("city", ""),
            "source_count": n,
            "sources": source_details,
            "mean_high_f": round(mean_temp, 1) if not is_low else None,
            "mean_low_f": round(mean_temp, 1) if is_low else None,
            "market_type": market_type,
            "std_dev_f": round(std_dev, 2),
            "spread_f": round(spread, 1),
            "confidence": round(confidence, 3),
            "strike_type": strike_type,
            "floor_strike": floor_strike,
            "cap_strike": cap_strike,
            "label": label,
            "our_prob_yes": round(our_prob_yes, 4),
            "our_prob_no": round(1.0 - our_prob_yes, 4),
            "ensemble_count": len(ensemble_preds),
        }

    def generate_signal(
        self,
        consensus: dict[str, Any],
        kalshi_yes_price: int,
        kalshi_no_price: int,
        min_edge: float = 0.08,
        max_edge: float = 0.25,
        min_confidence: float = 0.3,
        min_sources: int = 2,
    ) -> dict[str, Any] | None:
        """
        Compare consensus to Kalshi prices and generate a trading signal.
        Returns signal dict or None if no edge found.
        """
        our_prob_yes = consensus.get("our_prob_yes")
        if our_prob_yes is None:
            return None

        confidence = consensus.get("confidence", 0)
        source_count = consensus.get("source_count", 0)

        if confidence < min_confidence:
            return None
        if source_count < min_sources:
            return None

        kalshi_yes_prob = kalshi_yes_price / 100.0
        kalshi_no_prob = kalshi_no_price / 100.0
        our_prob_no = 1.0 - our_prob_yes

        # Edge on YES side
        yes_edge = our_prob_yes - kalshi_yes_prob
        # Edge on NO side
        no_edge = our_prob_no - kalshi_no_prob

        signal = None

        if yes_edge >= min_edge and yes_edge <= max_edge and kalshi_yes_price > 0:
            _mkt_type = consensus.get("market_type", "high_temp")
            _temp_key = "mean_low_f" if _mkt_type == "low_temp" else "mean_high_f"
            signal = {
                "side": "yes",
                "our_prob": round(our_prob_yes, 4),
                "kalshi_prob": round(kalshi_yes_prob, 4),
                "edge": round(yes_edge, 4),
                "confidence": round(confidence, 3),
                "source_count": source_count,
                "consensus_temp": consensus.get(_temp_key),
                "label": consensus.get("label", ""),
            }
        elif no_edge >= min_edge and no_edge <= max_edge and kalshi_no_price > 0:
            _mkt_type = consensus.get("market_type", "high_temp")
            _temp_key = "mean_low_f" if _mkt_type == "low_temp" else "mean_high_f"
            signal = {
                "side": "no",
                "our_prob": round(our_prob_no, 4),
                "kalshi_prob": round(kalshi_no_prob, 4),
                "edge": round(no_edge, 4),
                "confidence": round(confidence, 3),
                "source_count": source_count,
                "consensus_temp": consensus.get(_temp_key),
                "label": consensus.get("label", ""),
            }

        return signal

    async def close(self) -> None:
        await asyncio.gather(
            self.nws.close(),
            self.open_meteo.close(),
            self.tomorrow_io.close(),
            self.visual_crossing.close(),
        )
