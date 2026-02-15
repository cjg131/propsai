"""
Weather data pipeline for Kalshi weather market trading.
Fetches forecasts from 4 sources: NWS, Open-Meteo Ensemble, Tomorrow.io, Visual Crossing.
Builds consensus probability distributions for temperature, snowfall, rainfall.
"""
from __future__ import annotations

import asyncio
import math
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

# Kalshi weather cities → NWS grid points and coordinates
CITY_CONFIGS = {
    "NYC": {"lat": 40.7128, "lon": -74.0060, "nws_office": "OKX", "nws_grid": "33,37", "name": "New York"},
    "MIA": {"lat": 25.7617, "lon": -80.1918, "nws_office": "MFL", "nws_grid": "110,50", "name": "Miami"},
    "LAX": {"lat": 34.0522, "lon": -118.2437, "nws_office": "LOX", "nws_grid": "154,44", "name": "Los Angeles"},
    "CHI": {"lat": 41.8781, "lon": -87.6298, "nws_office": "LOT", "nws_grid": "76,73", "name": "Chicago"},
    "AUS": {"lat": 30.2672, "lon": -97.7431, "nws_office": "EWX", "nws_grid": "156,91", "name": "Austin"},
    "DFW": {"lat": 32.7767, "lon": -96.7970, "nws_office": "FWD", "nws_grid": "80,103", "name": "Dallas"},
    "PHL": {"lat": 39.9526, "lon": -75.1652, "nws_office": "PHI", "nws_grid": "49,75", "name": "Philadelphia"},
    "DEN": {"lat": 39.7392, "lon": -104.9903, "nws_office": "BOU", "nws_grid": "62,60", "name": "Denver"},
    "SEA": {"lat": 47.6062, "lon": -122.3321, "nws_office": "SEW", "nws_grid": "124,67", "name": "Seattle"},
    "SFO": {"lat": 37.7749, "lon": -122.4194, "nws_office": "MTR", "nws_grid": "85,105", "name": "San Francisco"},
    "DCA": {"lat": 38.9072, "lon": -77.0369, "nws_office": "LWX", "nws_grid": "97,71", "name": "Washington DC"},
    "SLC": {"lat": 40.7608, "lon": -111.8910, "nws_office": "SLC", "nws_grid": "97,175", "name": "Salt Lake City"},
    "ATL": {"lat": 33.7490, "lon": -84.3880, "nws_office": "FFC", "nws_grid": "50,86", "name": "Atlanta"},
    "HOU": {"lat": 29.7604, "lon": -95.3698, "nws_office": "HGX", "nws_grid": "65,97", "name": "Houston"},
    "BOS": {"lat": 42.3601, "lon": -71.0589, "nws_office": "BOX", "nws_grid": "71,90", "name": "Boston"},
    "LAS": {"lat": 36.1699, "lon": -115.1398, "nws_office": "VEF", "nws_grid": "126,97", "name": "Las Vegas"},
    "PHX": {"lat": 33.4484, "lon": -112.0740, "nws_office": "PSR", "nws_grid": "159,56", "name": "Phoenix"},
    "MSP": {"lat": 44.9778, "lon": -93.2650, "nws_office": "MPX", "nws_grid": "107,71", "name": "Minneapolis"},
    "NOL": {"lat": 29.9511, "lon": -90.0715, "nws_office": "LIX", "nws_grid": "76,76", "name": "New Orleans"},
    "DET": {"lat": 42.3314, "lon": -83.0458, "nws_office": "DTX", "nws_grid": "65,33", "name": "Detroit"},
}


class NWSClient:
    """National Weather Service API client (free, no key needed)."""

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(
            timeout=15.0,
            headers={"User-Agent": "PropsAI-Weather-Agent/1.0 (contact@propsai.com)"},
        )

    async def get_forecast(self, city_key: str) -> dict[str, Any] | None:
        """Get NWS point forecast for a city. Returns high/low temps."""
        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        try:
            url = f"https://api.weather.gov/gridpoints/{config['nws_office']}/{config['nws_grid']}/forecast"
            resp = await self._http.get(url)
            resp.raise_for_status()
            data = resp.json()

            periods = data.get("properties", {}).get("periods", [])
            if not periods:
                return None

            # Find today's daytime period for high temp
            result: dict[str, Any] = {"source": "nws", "city": city_key}
            for period in periods[:4]:
                name = period.get("name", "").lower()
                temp = period.get("temperature")
                if temp is None:
                    continue
                if period.get("isDaytime", False):
                    result["high_temp_f"] = temp
                    result["period_name"] = period.get("name", "")
                    result["short_forecast"] = period.get("shortForecast", "")
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

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=15.0)

    async def get_ensemble_forecast(self, city_key: str) -> dict[str, Any] | None:
        """
        Get ensemble forecast with probability distributions.
        Uses multiple weather models for uncertainty estimation.
        """
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
                    "timezone": "America/New_York",
                    "forecast_days": 3,
                    "models": "icon_seamless,gfs_seamless,ecmwf_ifs025,gem_global,bom_access_global_ensemble",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            daily = data.get("daily", {})
            times = daily.get("time", [])
            if not times:
                return None

            # Collect all model predictions for today
            high_temps = []
            for key, values in daily.items():
                if key.startswith("temperature_2m_max") and values:
                    if values[0] is not None:
                        high_temps.append(values[0])

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

            return {
                "source": "open_meteo",
                "city": city_key,
                "date": times[0],
                "high_temp_f": round(mean_high, 1),
                "high_temp_median": round(median_high, 1),
                "high_temp_min": round(min_high, 1),
                "high_temp_max": round(max_high, 1),
                "high_temp_p10": round(p10, 1),
                "high_temp_p25": round(p25, 1),
                "high_temp_p75": round(p75, 1),
                "high_temp_p90": round(p90, 1),
                "ensemble_count": n,
                "all_predictions": [round(t, 1) for t in high_temps],
            }

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

    async def get_forecast(self, city_key: str) -> dict[str, Any] | None:
        """Get Tomorrow.io forecast with percentile data."""
        if not self.api_key:
            return None

        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

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

            today = daily[0]
            values = today.get("values", {})

            return {
                "source": "tomorrow_io",
                "city": city_key,
                "date": today.get("time", "")[:10],
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

    async def get_forecast(self, city_key: str) -> dict[str, Any] | None:
        """Get Visual Crossing forecast."""
        if not self.api_key:
            return None

        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        try:
            location = f"{config['lat']},{config['lon']}"
            resp = await self._http.get(
                f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/today",
                params={
                    "unitGroup": "us",
                    "key": self.api_key,
                    "include": "days,hours",
                    "contentType": "json",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            days = data.get("days", [])
            if not days:
                return None

            today = days[0]
            return {
                "source": "visual_crossing",
                "city": city_key,
                "date": today.get("datetime", ""),
                "high_temp_f": today.get("tempmax"),
                "low_temp_f": today.get("tempmin"),
                "precip_inches": today.get("precip"),
                "precip_prob": today.get("precipprob"),
                "snow_inches": today.get("snow"),
                "humidity": today.get("humidity"),
                "conditions": today.get("conditions", ""),
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

    async def get_all_forecasts(self, city_key: str) -> dict[str, Any]:
        """Fetch forecasts from all available sources for a city."""
        results = await asyncio.gather(
            self.nws.get_forecast(city_key),
            self.open_meteo.get_ensemble_forecast(city_key),
            self.tomorrow_io.get_forecast(city_key),
            self.visual_crossing.get_forecast(city_key),
            return_exceptions=True,
        )

        forecasts: dict[str, Any] = {"city": city_key, "sources": {}}
        source_names = ["nws", "open_meteo", "tomorrow_io", "visual_crossing"]

        for name, result in zip(source_names, results):
            if isinstance(result, Exception):
                logger.warning("Forecast source failed", source=name, city=city_key, error=str(result))
                continue
            if result is not None:
                forecasts["sources"][name] = result

        return forecasts

    def _estimate_std_dev(self, forecasts: dict[str, Any]) -> float:
        """Estimate temperature standard deviation from available data."""
        sources = forecasts.get("sources", {})
        high_temps = [d.get("high_temp_f") for d in sources.values() if d.get("high_temp_f") is not None]
        if len(high_temps) < 2:
            return 2.5  # Default uncertainty
        spread = max(high_temps) - min(high_temps)
        # Ensemble spread gives better estimate
        ensemble = sources.get("open_meteo", {})
        preds = ensemble.get("all_predictions", [])
        if len(preds) >= 5:
            preds_sorted = sorted(preds)
            n = len(preds_sorted)
            iqr = preds_sorted[int(n * 0.75)] - preds_sorted[int(n * 0.25)]
            return max(iqr / 1.35, 1.5)  # IQR → std dev approximation
        return max(spread / 2, 2.0)

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
    ) -> dict[str, Any]:
        """
        Build a consensus probability for a Kalshi weather market.

        strike_type: 'greater', 'less', or 'between'
        floor_strike: lower bound (used by greater and between)
        cap_strike: upper bound (used by less and between)
        """
        sources = forecasts.get("sources", {})
        if not sources:
            return {"error": "No forecast data available"}

        high_temps: list[float] = []
        source_details: list[dict[str, Any]] = []

        for name, data in sources.items():
            temp = data.get("high_temp_f")
            if temp is not None:
                high_temps.append(temp)
                source_details.append({"source": name, "high_temp_f": temp})

        if not high_temps:
            return {"error": "No temperature data from any source"}

        n = len(high_temps)
        mean_temp = sum(high_temps) / n
        spread = max(high_temps) - min(high_temps) if n > 1 else 0
        confidence = max(0.0, min(1.0, 1.0 - (spread - 2) / 8))
        std_dev = self._estimate_std_dev(forecasts)

        # Use ensemble predictions if available for empirical probability
        ensemble_preds = sources.get("open_meteo", {}).get("all_predictions", [])

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

        our_prob_yes = max(0.0, min(1.0, our_prob_yes))

        return {
            "city": forecasts.get("city", ""),
            "source_count": n,
            "sources": source_details,
            "mean_high_f": round(mean_temp, 1),
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

        if yes_edge >= min_edge and kalshi_yes_price > 0:
            signal = {
                "side": "yes",
                "our_prob": round(our_prob_yes, 4),
                "kalshi_prob": round(kalshi_yes_prob, 4),
                "edge": round(yes_edge, 4),
                "confidence": round(confidence, 3),
                "source_count": source_count,
                "consensus_temp": consensus.get("mean_high_f"),
                "label": consensus.get("label", ""),
            }
        elif no_edge >= min_edge and kalshi_no_price > 0:
            signal = {
                "side": "no",
                "our_prob": round(our_prob_no, 4),
                "kalshi_prob": round(kalshi_no_prob, 4),
                "edge": round(no_edge, 4),
                "confidence": round(confidence, 3),
                "source_count": source_count,
                "consensus_temp": consensus.get("mean_high_f"),
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
