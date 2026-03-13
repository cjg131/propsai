"""
Weather data pipeline for Kalshi weather market trading.
Fetches forecasts from 4 sources: NWS, Open-Meteo Ensemble, Tomorrow.io, Visual Crossing.
Builds consensus probability distributions for temperature, snowfall, rainfall.
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timezone
from typing import Any, Callable

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

UTC = timezone.utc

# Kalshi weather cities → NWS grid points, coordinates, and observation stations
# nws_station: ICAO station ID used for real-time hourly observations
# offsets: Known temperature biases between generic city forecasts and the specific NWS settlement station
CITY_CONFIGS = {
    "NYC": {"lat": 40.7128, "lon": -74.0060, "nws_office": "OKX", "nws_grid": "33,37", "name": "New York", "tz": "America/New_York", "nws_station": "KNYC", "high_offset": -1.0, "low_offset": -0.5, "upstream_city": "PHL"}, # Central Park runs cooler
    "MIA": {"lat": 25.7617, "lon": -80.1918, "nws_office": "MFL", "nws_grid": "110,50", "name": "Miami", "tz": "America/New_York", "nws_station": "KMIA", "high_offset": 0.0, "low_offset": 0.0},
    "LAX": {"lat": 34.0522, "lon": -118.2437, "nws_office": "LOX", "nws_grid": "154,44", "name": "Los Angeles", "tz": "America/Los_Angeles", "nws_station": "KLAX", "high_offset": 0.0, "low_offset": 0.0},
    "CHI": {"lat": 41.8781, "lon": -87.6298, "nws_office": "LOT", "nws_grid": "76,73", "name": "Chicago", "tz": "America/Chicago", "nws_station": "KORD", "high_offset": 0.0, "low_offset": 0.0, "upstream_city": "MSP"},
    "AUS": {"lat": 30.2672, "lon": -97.7431, "nws_office": "EWX", "nws_grid": "156,91", "name": "Austin", "tz": "America/Chicago", "nws_station": "KAUS", "high_offset": 0.0, "low_offset": 0.0},
    "DFW": {"lat": 32.7767, "lon": -96.7970, "nws_office": "FWD", "nws_grid": "80,103", "name": "Dallas", "tz": "America/Chicago", "nws_station": "KDFW", "high_offset": 0.0, "low_offset": 0.0, "upstream_city": "AUS"},
    "PHL": {"lat": 39.9526, "lon": -75.1652, "nws_office": "PHI", "nws_grid": "49,75", "name": "Philadelphia", "tz": "America/New_York", "nws_station": "KPHL", "high_offset": 0.0, "low_offset": 0.0, "upstream_city": "DCA"},
    "DEN": {"lat": 39.7392, "lon": -104.9903, "nws_office": "BOU", "nws_grid": "62,60", "name": "Denver", "tz": "America/Denver", "nws_station": "KDEN", "high_offset": 0.0, "low_offset": 0.0, "upstream_city": "SLC"},
    "SEA": {"lat": 47.6062, "lon": -122.3321, "nws_office": "SEW", "nws_grid": "124,67", "name": "Seattle", "tz": "America/Los_Angeles", "nws_station": "KSEA", "high_offset": 1.0, "low_offset": 0.5}, # Sea-Tac tarmac runs hotter
    "SFO": {"lat": 37.7749, "lon": -122.4194, "nws_office": "MTR", "nws_grid": "85,105", "name": "San Francisco", "tz": "America/Los_Angeles", "nws_station": "KSFO", "high_offset": 0.0, "low_offset": 0.0},
    "DCA": {"lat": 38.9072, "lon": -77.0369, "nws_office": "LWX", "nws_grid": "97,71", "name": "Washington DC", "tz": "America/New_York", "nws_station": "KDCA", "high_offset": 0.0, "low_offset": 0.0, "upstream_city": "ATL"},
    "SLC": {"lat": 40.7608, "lon": -111.8910, "nws_office": "SLC", "nws_grid": "97,175", "name": "Salt Lake City", "tz": "America/Denver", "nws_station": "KSLC", "high_offset": 0.0, "low_offset": 0.0},
    "ATL": {"lat": 33.7490, "lon": -84.3880, "nws_office": "FFC", "nws_grid": "50,86", "name": "Atlanta", "tz": "America/New_York", "nws_station": "KATL", "high_offset": 0.0, "low_offset": 0.0, "upstream_city": "HOU"},
    "HOU": {"lat": 29.7604, "lon": -95.3698, "nws_office": "HGX", "nws_grid": "65,97", "name": "Houston", "tz": "America/Chicago", "nws_station": "KHOU", "high_offset": 0.0, "low_offset": 0.0},
    "BOS": {"lat": 42.3601, "lon": -71.0589, "nws_office": "BOX", "nws_grid": "71,90", "name": "Boston", "tz": "America/New_York", "nws_station": "KBOS", "high_offset": 0.0, "low_offset": 0.0, "upstream_city": "NYC"},
    "LAS": {"lat": 36.1699, "lon": -115.1398, "nws_office": "VEF", "nws_grid": "126,97", "name": "Las Vegas", "tz": "America/Los_Angeles", "nws_station": "KLAS", "high_offset": 0.0, "low_offset": 0.0},
    "PHX": {"lat": 33.4484, "lon": -112.0740, "nws_office": "PSR", "nws_grid": "159,56", "name": "Phoenix", "tz": "America/Phoenix", "nws_station": "KPHX", "high_offset": 0.0, "low_offset": 0.0},
    "MSP": {"lat": 44.9778, "lon": -93.2650, "nws_office": "MPX", "nws_grid": "107,71", "name": "Minneapolis", "tz": "America/Chicago", "nws_station": "KMSP", "high_offset": 0.0, "low_offset": 0.0},
    "NOL": {"lat": 29.9511, "lon": -90.0715, "nws_office": "LIX", "nws_grid": "76,76", "name": "New Orleans", "tz": "America/Chicago", "nws_station": "KMSY", "high_offset": 0.0, "low_offset": 0.0, "upstream_city": "HOU"},
    "DET": {"lat": 42.3314, "lon": -83.0458, "nws_office": "DTX", "nws_grid": "65,33", "name": "Detroit", "tz": "America/Detroit", "nws_station": "KDTW", "high_offset": 0.0, "low_offset": 0.0, "upstream_city": "CHI"},
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

    async def get_hourly_daily_forecast(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Extract daily high/low from NWS hourly forecast using Intra-Hour Curve Fitting (Cubic Spline Interpolation)."""
        if target_date is None:
            target_date = datetime.now(UTC).date()
        
        hourly = await self.get_hourly_forecast(city_key)
        if not hourly:
            return None
        
        target_str = target_date.isoformat()
        
        # Fallback raw lists
        temps_for_date: list[float] = []
        for period in hourly:
            time_str = period.get("time", "")
            if time_str[:10] == target_str:
                temp = period.get("temp_f")
                if temp is not None:
                    temps_for_date.append(temp)
        
        if not temps_for_date:
            return None
            
        raw_max = max(temps_for_date)
        raw_min = min(temps_for_date)
        
        spline_max = raw_max
        spline_min = raw_min
        
        # Attempt to use Cubic Spline to find intra-hour peaks/valleys
        try:
            import numpy as np
            from scipy.interpolate import CubicSpline
            from datetime import datetime
            
            times = []
            temps = []
            
            for period in hourly:
                time_str = period.get("time", "")
                temp = period.get("temp_f")
                if time_str and temp is not None:
                    # Parse ISO format. Python 3.11+ handles Z, 3.9 might need fromisoformat tweaks, 
                    # but NWS usually returns standard formats with timezone offsets
                    try:
                        dt = datetime.fromisoformat(time_str)
                        times.append(dt.timestamp())
                        temps.append(temp)
                    except Exception:
                        pass
                        
            if len(times) >= 4:
                # Sort pairs
                sorted_pairs = sorted(zip(times, temps))
                t_sorted = np.array([p[0] for p in sorted_pairs])
                temp_sorted = np.array([p[1] for p in sorted_pairs])
                
                # We need unique x values for CubicSpline
                _, unique_indices = np.unique(t_sorted, return_index=True)
                t_unique = t_sorted[unique_indices]
                temp_unique = temp_sorted[unique_indices]
                
                if len(t_unique) >= 4:
                    cs = CubicSpline(t_unique, temp_unique)
                    
                    # We only care about interpolating points for the target date
                    # Find min/max timestamps for the target date from our parsed times
                    target_timestamps = []
                    for period in hourly:
                        time_str = period.get("time", "")
                        if time_str[:10] == target_str:
                            try:
                                dt = datetime.fromisoformat(time_str)
                                target_timestamps.append(dt.timestamp())
                            except Exception:
                                pass
                                
                    if target_timestamps:
                        t_start = min(target_timestamps)
                        t_end = max(target_timestamps)
                        
                        # Generate dense points (every 5 minutes)
                        t_dense = np.linspace(t_start, t_end, num=int((t_end - t_start) / 300) + 1)
                        temp_dense = cs(t_dense)
                        
                        calc_max = float(np.max(temp_dense))
                        calc_min = float(np.min(temp_dense))
                        
                        # Apply sanity checks: spline shouldn't deviate wildly from raw hourly points
                        # Usually peaks are within 1-2 degrees of hourly readings
                        if calc_max > raw_max and calc_max <= raw_max + 2.5:
                            spline_max = calc_max
                        if calc_min < raw_min and calc_min >= raw_min - 2.5:
                            spline_min = calc_min
                            
        except Exception as e:
            # Catch import errors or parsing errors and just fall back to raw
            pass
        
        return {
            "source": "nws_hourly",
            "city": city_key,
            "high_temp_f": round(max(raw_max, spline_max), 2),
            "low_temp_f": round(min(raw_min, spline_min), 2),
            "date": target_str,
            "hourly_count": len(temps_for_date),
            "spline_applied": spline_max != raw_max or spline_min != raw_min
        }

    async def get_current_observations(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """
        Fetch real-time hourly observations from NWS for a city.
        Returns the observed high/low so far today from actual station data.
        This is the core of the same-day arbitrage strategy — we KNOW what
        the thermometer has read, not what we predict it will read.
        """
        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None
        station = config.get("nws_station")
        if not station:
            return None

        if target_date is None:
            target_date = datetime.now(UTC).date()

        try:
            # NWS observations endpoint — returns last ~72 hours of hourly obs
            url = f"https://api.weather.gov/stations/{station}/observations"
            resp = await self._http.get(url, params={"limit": 48})
            resp.raise_for_status()
            data = resp.json()

            features = data.get("features", [])
            if not features:
                return None

            target_str = target_date.isoformat()
            temps_today: list[float] = []
            latest_temp_f: float | None = None
            latest_time: str = ""

            for feature in features:
                props = feature.get("properties", {})
                obs_time = props.get("timestamp", "")  # ISO8601 UTC
                if not obs_time or obs_time[:10] != target_str:
                    continue

                # Temperature is in Celsius from NWS — convert to Fahrenheit
                temp_c_obj = props.get("temperature", {})
                temp_c = temp_c_obj.get("value") if isinstance(temp_c_obj, dict) else temp_c_obj
                if temp_c is None:
                    continue

                temp_f = round(temp_c * 9 / 5 + 32, 1)
                temps_today.append(temp_f)

                # Track the most recent observation
                if not latest_time or obs_time > latest_time:
                    latest_time = obs_time
                    latest_temp_f = temp_f

            if not temps_today:
                return None

            observed_high = max(temps_today)
            observed_low = min(temps_today)
            obs_count = len(temps_today)

            logger.debug(
                f"NWS observations {city_key}: {obs_count} readings today, "
                f"high={observed_high}°F low={observed_low}°F current={latest_temp_f}°F"
            )

            return {
                "source": "nws_observations",
                "city": city_key,
                "date": target_str,
                "observed_high_f": observed_high,
                "observed_low_f": observed_low,
                "current_temp_f": latest_temp_f,
                "latest_obs_time": latest_time,
                "obs_count": obs_count,
                "all_temps_f": temps_today,
            }

        except Exception as e:
            logger.warning("NWS observations failed", city=city_key, station=station, error=str(e))
            return None

    async def close(self) -> None:
        await self._http.aclose()


class NOAAHRRRClient:
    """NOAA HRRR (High-Resolution Rapid Refresh) model client.
    
    3km resolution, hourly updates. Uses NOMADS OpenDAP server for easier access.
    HRRR is the highest resolution operational weather model for the US.
    """

    def __init__(self) -> None:
        self._http = httpx.AsyncClient(timeout=20.0)
        self._cache: dict[str, tuple[float, dict[str, Any]]] = {}

    async def get_forecast(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Get HRRR forecast for a city. Uses nearest grid point to city coordinates."""
        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        if target_date is None:
            target_date = datetime.now(UTC).date()

        # Cache key
        cache_key = f"{city_key}_{target_date.isoformat()}"
        now = datetime.now(UTC).timestamp()
        if cache_key in self._cache:
            cached_time, cached_data = self._cache[cache_key]
            if now - cached_time < 3600:  # 1 hour cache
                return cached_data

        try:
            # HRRR data via NOMADS - simplified access
            # We'll use the 2m temperature field from the latest HRRR run
            lat, lon = config["lat"], config["lon"]
            
            # For now, use a simpler approach: fetch from NOAA's weather.gov API
            # which includes HRRR data in their blend
            # Full HRRR implementation would require GRIB2 parsing
            
            # Alternative: Use Iowa State's HRRR archive which provides JSON
            url = "https://mesonet.agron.iastate.edu/json/hrrr.py"
            params = {
                "lat": lat,
                "lon": lon,
                "valid": target_date.isoformat(),
            }
            
            resp = await self._http.get(url, params=params)
            if resp.status_code != 200:
                return None
                
            data = resp.json()
            
            # Extract temperature forecast
            if not data or "data" not in data:
                return None
            
            temps = [d.get("tmpc") for d in data.get("data", []) if d.get("tmpc") is not None]
            if not temps:
                return None
            
            # Convert Celsius to Fahrenheit
            temps_f = [(t * 9/5) + 32 for t in temps]
            
            result = {
                "source": "hrrr",
                "city": city_key,
                "high_temp_f": max(temps_f),
                "low_temp_f": min(temps_f),
                "date": target_date.isoformat(),
            }
            
            self._cache[cache_key] = (now, result)
            return result

        except Exception as e:
            logger.debug("HRRR forecast failed", city=city_key, error=str(e))
            return None

    async def close(self) -> None:
        await self._http.aclose()


class OpenMeteoClient:
    """Open-Meteo Ensemble API client (free, no key needed)."""

    _cache: dict[str, dict] = {}
    _cache_ts: dict[str, float] = {}
    _CACHE_TTL: float = 10800.0  # 3 hours — reduces repeat hits when scans run often
    _STALE_CACHE_TTL: float = 21600.0  # 6 hours — stale fallback during provider incidents
    _backoff_until: float = 0.0
    _backoff_reason: str = ""

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
        cached = OpenMeteoClient._cache.get(cache_key)
        cached_ts = OpenMeteoClient._cache_ts.get(cache_key, 0)
        cache_age = now - cached_ts
        if cached and cache_age < OpenMeteoClient._CACHE_TTL:
            return cached

        if now < OpenMeteoClient._backoff_until:
            if cached and cache_age < OpenMeteoClient._STALE_CACHE_TTL:
                logger.debug(
                    "Open-Meteo in cooldown, using stale cache",
                    city=city_key,
                    stale_age_seconds=round(cache_age),
                    reason=OpenMeteoClient._backoff_reason,
                )
                return cached
            logger.debug(
                "Open-Meteo in cooldown, skipping request",
                city=city_key,
                retry_in_seconds=round(OpenMeteoClient._backoff_until - now),
                reason=OpenMeteoClient._backoff_reason,
            )
            return None

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
            OpenMeteoClient._backoff_until = 0.0
            OpenMeteoClient._backoff_reason = ""
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

        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 429:
                OpenMeteoClient._backoff_until = now + 2 * 3600
                OpenMeteoClient._backoff_reason = "rate_limited"
            elif status and status >= 500:
                OpenMeteoClient._backoff_until = now + 30 * 60
                OpenMeteoClient._backoff_reason = f"server_{status}"

            if cached and cache_age < OpenMeteoClient._STALE_CACHE_TTL:
                logger.warning(
                    "Open-Meteo failed, using stale cache",
                    city=city_key,
                    status=status,
                    stale_age_seconds=round(cache_age),
                )
                return cached

            logger.warning("Open-Meteo ensemble failed", city=city_key, error=str(e))
            return None
        except Exception as e:
            if cached and cache_age < OpenMeteoClient._STALE_CACHE_TTL:
                logger.warning(
                    "Open-Meteo failed, using stale cache",
                    city=city_key,
                    stale_age_seconds=round(cache_age),
                    error=str(e),
                )
                return cached
            logger.warning("Open-Meteo ensemble failed", city=city_key, error=str(e))
            return None

    async def close(self) -> None:
        await self._http.aclose()


class TomorrowIOClient:
    """Tomorrow.io API client (free tier — 5-day forecast with percentiles)."""

    _cache: dict[str, dict] = {}
    _cache_ts: dict[str, float] = {}
    _CACHE_TTL: float = 10800.0  # 3 hours — keeps us comfortably under Tomorrow.io free-tier limits
    _STALE_CACHE_TTL: float = 21600.0  # 6 hours — stale fallback during provider incidents
    _backoff_until: float = 0.0
    _backoff_reason: str = ""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=15.0)

    async def get_forecast(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Get Tomorrow.io forecast for a specific date.

        Fetches BOTH daily summary AND hourly data. The hourly data gives us
        per-hour temperatures which we aggregate to get true daily max/min —
        more accurate than the daily summary alone.
        Results are cached for 1 hour per city+date to stay within free-tier limits (25 req/hr).
        """
        if not self.api_key:
            return None

        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        if target_date is None:
            target_date = datetime.now(UTC).date()
        target_str = target_date.isoformat()

        import time as _time
        _now = _time.time()
        cache_key = f"{city_key}_{target_str}"
        cached = TomorrowIOClient._cache.get(cache_key)
        cached_ts = TomorrowIOClient._cache_ts.get(cache_key, 0)
        cache_age = _now - cached_ts
        if cached and cache_age < TomorrowIOClient._CACHE_TTL:
            return cached

        if _now < TomorrowIOClient._backoff_until:
            if cached and cache_age < TomorrowIOClient._STALE_CACHE_TTL:
                logger.debug(
                    "Tomorrow.io in cooldown, using stale cache",
                    city=city_key,
                    stale_age_seconds=round(cache_age),
                    reason=TomorrowIOClient._backoff_reason,
                )
                return cached
            logger.debug(
                "Tomorrow.io in cooldown, skipping request",
                city=city_key,
                retry_in_seconds=round(TomorrowIOClient._backoff_until - _now),
                reason=TomorrowIOClient._backoff_reason,
            )
            return None

        try:
            # Fetch daily + hourly in one request using comma-separated timesteps
            resp = await self._http.get(
                "https://api.tomorrow.io/v4/weather/forecast",
                params={
                    "location": f"{config['lat']},{config['lon']}",
                    "apikey": self.api_key,
                    "units": "imperial",
                    "timesteps": "1d,1h",
                    "fields": "temperatureMax,temperatureMin,temperature,precipitationProbabilityMax,precipitationIntensityMax,humidityMax,windSpeedMax",
                },
            )
            resp.raise_for_status()
            TomorrowIOClient._backoff_until = 0.0
            TomorrowIOClient._backoff_reason = ""
            data = resp.json()

            timelines = data.get("timelines", {})
            daily = timelines.get("daily", [])
            hourly = timelines.get("hourly", [])

            if not daily:
                return None

            # Find the daily entry matching target_date
            target_day = None
            for day in daily:
                if day.get("time", "")[:10] == target_str:
                    target_day = day
                    break
            if target_day is None:
                target_day = daily[0]  # fallback to first

            values = target_day.get("values", {})
            high_from_daily = values.get("temperatureMax")
            low_from_daily = values.get("temperatureMin")

            # Refine high/low from hourly data for the target date
            # Hourly gives us actual per-hour temps — more precise than daily summary
            hourly_temps_for_date = [
                h.get("values", {}).get("temperature")
                for h in hourly
                if h.get("time", "")[:10] == target_str
                and h.get("values", {}).get("temperature") is not None
            ]

            if hourly_temps_for_date:
                high_from_hourly = max(hourly_temps_for_date)
                low_from_hourly = min(hourly_temps_for_date)
                # Use hourly-derived values if available (more precise)
                high_temp_f = high_from_hourly
                low_temp_f = low_from_hourly
            else:
                high_temp_f = high_from_daily
                low_temp_f = low_from_daily

            result = {
                "source": "tomorrow_io",
                "city": city_key,
                "date": target_day.get("time", "")[:10],
                "high_temp_f": high_temp_f,
                "low_temp_f": low_temp_f,
                "precip_prob": values.get("precipitationProbabilityMax"),
                "precip_inches": values.get("precipitationIntensityMax"),
                "humidity": values.get("humidityMax"),
                "wind_speed": values.get("windSpeedMax"),
                "hourly_count": len(hourly_temps_for_date),
            }
            TomorrowIOClient._cache[cache_key] = result
            TomorrowIOClient._cache_ts[cache_key] = _now
            return result

        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 429:
                TomorrowIOClient._backoff_until = _now + 4 * 3600
                TomorrowIOClient._backoff_reason = "rate_limited"
            elif status and status >= 500:
                TomorrowIOClient._backoff_until = _now + 45 * 60
                TomorrowIOClient._backoff_reason = f"server_{status}"

            if cached and cache_age < TomorrowIOClient._STALE_CACHE_TTL:
                logger.warning(
                    "Tomorrow.io failed, using stale cache",
                    city=city_key,
                    status=status,
                    stale_age_seconds=round(cache_age),
                )
                return cached

            logger.warning("Tomorrow.io forecast failed", city=city_key, error=str(e))
            return None
        except Exception as e:
            if cached and cache_age < TomorrowIOClient._STALE_CACHE_TTL:
                logger.warning(
                    "Tomorrow.io failed, using stale cache",
                    city=city_key,
                    stale_age_seconds=round(cache_age),
                    error=str(e),
                )
                return cached
            logger.warning("Tomorrow.io forecast failed", city=city_key, error=str(e))
            return None

    async def close(self) -> None:
        await self._http.aclose()


class VisualCrossingClient:
    """Visual Crossing Weather API client ($35/month)."""

    _cache: dict[str, dict] = {}
    _cache_ts: dict[str, float] = {}
    _CACHE_TTL: float = 3600.0  # 1 hour — refreshes data while staying within API limits

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=15.0)

    async def get_forecast(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Get Visual Crossing forecast for a specific date.
        Results are cached for 1 hour per city+date to minimize redundant API calls.
        """
        if not self.api_key:
            return None

        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        if target_date is None:
            target_date = datetime.now(UTC).date()
        target_str = target_date.isoformat()

        import time as _time
        _now = _time.time()
        cache_key = f"{city_key}_{target_str}"
        if cache_key in VisualCrossingClient._cache and (_now - VisualCrossingClient._cache_ts.get(cache_key, 0)) < VisualCrossingClient._CACHE_TTL:
            return VisualCrossingClient._cache[cache_key]

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
            result = {
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
            VisualCrossingClient._cache[cache_key] = result
            VisualCrossingClient._cache_ts[cache_key] = _now
            return result

        except Exception as e:
            logger.warning("Visual Crossing forecast failed", city=city_key, error=str(e))
            return None

    async def close(self) -> None:
        await self._http.aclose()


class WeatherbitClient:
    """Weatherbit.io API client (free tier: 50 calls/day)."""

    _cache: dict[str, dict] = {}
    _cache_ts: dict[str, float] = {}
    _CACHE_TTL: float = 3600.0  # 1 hour

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=15.0)

    async def get_forecast(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Get Weatherbit forecast for a specific date."""
        if not self.api_key:
            return None

        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        if target_date is None:
            target_date = datetime.now(UTC).date()
        target_str = target_date.isoformat()

        import time as _time
        _now = _time.time()
        cache_key = f"{city_key}_{target_str}"
        if cache_key in WeatherbitClient._cache and (_now - WeatherbitClient._cache_ts.get(cache_key, 0)) < WeatherbitClient._CACHE_TTL:
            return WeatherbitClient._cache[cache_key]

        try:
            resp = await self._http.get(
                "https://api.weatherbit.io/v2.0/forecast/daily",
                params={
                    "lat": config["lat"],
                    "lon": config["lon"],
                    "key": self.api_key,
                    "days": 7,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            forecasts = data.get("data", [])
            if not forecasts:
                return None

            # Find forecast for target date
            target_forecast = None
            for fc in forecasts:
                if fc.get("valid_date") == target_str:
                    target_forecast = fc
                    break
            if not target_forecast:
                target_forecast = forecasts[0]

            # Convert Celsius to Fahrenheit
            max_temp_c = target_forecast.get("max_temp")
            min_temp_c = target_forecast.get("min_temp")
            high_temp_f = round(max_temp_c * 9 / 5 + 32, 1) if max_temp_c is not None else None
            low_temp_f = round(min_temp_c * 9 / 5 + 32, 1) if min_temp_c is not None else None

            result = {
                "source": "weatherbit",
                "city": city_key,
                "date": target_forecast.get("valid_date", ""),
                "high_temp_f": high_temp_f,
                "low_temp_f": low_temp_f,
                "precip_inches": target_forecast.get("precip"),
                "precip_prob": target_forecast.get("pop"),
                "snow_inches": target_forecast.get("snow"),
            }
            WeatherbitClient._cache[cache_key] = result
            WeatherbitClient._cache_ts[cache_key] = _now
            return result

        except Exception as e:
            logger.warning("Weatherbit forecast failed", city=city_key, error=str(e))
            return None

    async def close(self) -> None:
        await self._http.aclose()


class OpenWeatherMapClient:
    """OpenWeatherMap API client (free tier: 60 calls/min)."""

    _cache: dict[str, dict] = {}
    _cache_ts: dict[str, float] = {}
    _CACHE_TTL: float = 3600.0  # 1 hour

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self._http = httpx.AsyncClient(timeout=15.0)

    async def get_forecast(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Get OpenWeatherMap forecast for a specific date."""
        if not self.api_key:
            return None

        config = CITY_CONFIGS.get(city_key)
        if not config:
            return None

        if target_date is None:
            target_date = datetime.now(UTC).date()
        target_str = target_date.isoformat()

        import time as _time
        _now = _time.time()
        cache_key = f"{city_key}_{target_str}"
        if cache_key in OpenWeatherMapClient._cache and (_now - OpenWeatherMapClient._cache_ts.get(cache_key, 0)) < OpenWeatherMapClient._CACHE_TTL:
            return OpenWeatherMapClient._cache[cache_key]

        try:
            resp = await self._http.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={
                    "lat": config["lat"],
                    "lon": config["lon"],
                    "appid": self.api_key,
                    "units": "imperial",
                },
            )
            resp.raise_for_status()
            data = resp.json()

            forecasts = data.get("list", [])
            if not forecasts:
                return None

            # Group 3-hour forecasts by date and find min/max for target date
            temps_for_date = []
            for fc in forecasts:
                fc_time = fc.get("dt_txt", "")
                if fc_time[:10] == target_str:
                    temp = fc.get("main", {}).get("temp")
                    if temp is not None:
                        temps_for_date.append(temp)

            if not temps_for_date:
                # Use first available forecast
                temps_for_date = [fc.get("main", {}).get("temp") for fc in forecasts[:8] if fc.get("main", {}).get("temp")]

            if not temps_for_date:
                return None

            high_temp_f = max(temps_for_date)
            low_temp_f = min(temps_for_date)

            result = {
                "source": "openweathermap",
                "city": city_key,
                "date": target_str,
                "high_temp_f": round(high_temp_f, 1),
                "low_temp_f": round(low_temp_f, 1),
            }
            OpenWeatherMapClient._cache[cache_key] = result
            OpenWeatherMapClient._cache_ts[cache_key] = _now
            return result

        except Exception as e:
            logger.warning("OpenWeatherMap forecast failed", city=city_key, error=str(e))
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
        weatherbit_key: str = "",
        openweathermap_key: str = "",
    ) -> None:
        self.nws = NWSClient()
        self.hrrr = NOAAHRRRClient()
        self.open_meteo = OpenMeteoClient()
        self.tomorrow_io = TomorrowIOClient(api_key=tomorrow_io_key)
        self.visual_crossing = VisualCrossingClient(api_key=visual_crossing_key)
        self.weatherbit = WeatherbitClient(api_key=weatherbit_key)
        self.openweathermap = OpenWeatherMapClient(api_key=openweathermap_key)

        # Circuit breaker: track consecutive failures per source
        self._source_failures: dict[str, int] = {}
        self._source_circuit_open_until: dict[str, float] = {}
        self._circuit_breaker_threshold = 3   # failures before tripping
        self._circuit_breaker_cooldown = 1800  # 30 minutes

    async def get_current_observations(self, city_key: str, target_date: date | None = None) -> dict[str, Any] | None:
        """Fetch real-time NWS observations for same-day arbitrage."""
        return await self.nws.get_current_observations(city_key, target_date)

    def get_dynamic_weights(self, city: str = None) -> dict[str, float]:
        """
        Retrieve dynamic weights for weather APIs based on historical MAE.
        Falls back to static weights if insufficient data.
        """
        base_weights = {
            "nws": 2.0,
            "nws_hourly": 2.0,
            "hrrr": 2.0,
            "open_meteo": 1.0,
            "tomorrow_io": 1.0,
            "visual_crossing": 1.0,
            "weatherbit": 1.0,
            "openweathermap": 1.0
        }
        
        try:
            import os
            import sqlite3
            db_path = os.path.join(os.path.dirname(__file__), "..", "data", "trading_engine.db")
            if not os.path.exists(db_path):
                return base_weights
                
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            
            # Get average error per source over the last 30 days
            query = '''
                SELECT source_name, AVG(ABS(error)) as mae, COUNT(*) as samples
                FROM weather_api_performance
                WHERE error IS NOT NULL AND actual_temp IS NOT NULL
                  AND created_at >= date('now', '-30 days')
            '''
            if city:
                query += f" AND city = '{city}'"
            query += " GROUP BY source_name HAVING samples >= 5"
            
            c.execute(query)
            results = c.fetchall()
            conn.close()
            
            if not results:
                return base_weights
                
            # Calculate new weights: inverse of MAE (lower error = higher weight)
            dynamic_weights = {}
            for row in results:
                source_name, mae, samples = row
                if mae < 0.1: mae = 0.1
                # Scale so an MAE of 1.0 gives a weight of 2.0, MAE of 2.0 gives weight 1.0
                weight = 2.0 / mae
                # Cap weight between 0.5 and 3.0
                weight = max(0.5, min(3.0, weight))
                dynamic_weights[source_name] = round(weight, 2)
                
            # Merge with base weights for any missing sources
            for src, bw in base_weights.items():
                if src not in dynamic_weights:
                    dynamic_weights[src] = bw
                    
            return dynamic_weights
            
        except Exception as e:
            logger.error("Error fetching dynamic weights", error=str(e))
            return base_weights

    async def get_all_forecasts(self, city_key: str, target_date: date | None = None) -> dict[str, Any]:
        """Fetch forecasts from all available sources for a city on a specific date.
        Staggers calls to avoid 429 rate limits on paid APIs."""
        if target_date is None:
            target_date = datetime.now(UTC).date()
        forecasts: dict[str, Any] = {"city": city_key, "sources": {}, "target_date": target_date.isoformat()}

        # Build provider calls lazily so skipped sources do not leave pending coroutines behind.
        calls: list[tuple[str, Callable[[], Any]]] = [
            ("nws", lambda: self.nws.get_forecast(city_key, target_date)),
            ("nws_hourly", lambda: self.nws.get_hourly_daily_forecast(city_key, target_date)),
            ("hrrr", lambda: self.hrrr.get_forecast(city_key, target_date)),
            ("open_meteo", lambda: self.open_meteo.get_ensemble_forecast(city_key, target_date)),
            ("tomorrow_io", lambda: self.tomorrow_io.get_forecast(city_key, target_date)),
            ("visual_crossing", lambda: self.visual_crossing.get_forecast(city_key, target_date)),
            ("weatherbit", lambda: self.weatherbit.get_forecast(city_key, target_date)),
            ("openweathermap", lambda: self.openweathermap.get_forecast(city_key, target_date)),
        ]

        import time as _time
        now_ts = _time.time()

        for name, make_coro in calls:
            # Circuit breaker: skip sources that have failed repeatedly
            open_until = self._source_circuit_open_until.get(name, 0)
            if now_ts < open_until:
                logger.debug("Circuit breaker open, skipping source", source=name, city=city_key)
                continue

            try:
                coro = make_coro()
                result = await coro
                if result is not None:
                    forecasts["sources"][name] = result
                    self._source_failures[name] = 0  # reset on success
                else:
                    self._source_failures[name] = self._source_failures.get(name, 0) + 1
            except Exception as e:
                logger.warning("Forecast source failed", source=name, city=city_key, error=str(e))
                fails = self._source_failures.get(name, 0) + 1
                self._source_failures[name] = fails
                if fails >= self._circuit_breaker_threshold:
                    self._source_circuit_open_until[name] = now_ts + self._circuit_breaker_cooldown
                    logger.warning("Circuit breaker tripped", source=name, failures=fails,
                                   cooldown_sec=self._circuit_breaker_cooldown)
            # Delay between API calls to respect Tomorrow.io free tier (~25 req/hr)
            await asyncio.sleep(2)

        return forecasts

    async def get_all_city_forecasts(self, target_date: date | None = None) -> dict[str, dict[str, Any]]:
        """
        Fetch forecasts for ALL cities and calculate confidence scores.
        
        Returns:
            {
                "NYC": {
                    "high": {"mean": 45, "range": (43, 47), "confidence": 0.82, "sources": 4, "weighted_mean": 45.2},
                    "low": {"mean": 32, "range": (30, 34), "confidence": 0.78, "sources": 4, "weighted_mean": 31.8}
                },
                ...
            }
        """
        if target_date is None:
            target_date = datetime.now(UTC).date()
        
        all_city_forecasts = {}
        
        # Fetch forecasts for all cities
        for city_key in CITY_CONFIGS.keys():
            try:
                forecasts = await self.get_all_forecasts(city_key, target_date)
                sources = forecasts.get("sources", {})
                
                if not sources:
                    continue
                
                # Get dynamic weights based on historical accuracy
                SOURCE_WEIGHTS = self.get_dynamic_weights(city_key)

                high_temps = []
                high_weights = []
                low_temps = []
                low_weights = []

                for source_name, s in sources.items():
                    w = SOURCE_WEIGHTS.get(source_name, 1.0)
                    h = s.get("high_temp_f")
                    l = s.get("low_temp_f")
                    if h is not None:
                        high_temps.append(h)
                        high_weights.append(w)
                    if l is not None:
                        low_temps.append(l)
                        low_weights.append(w)
                
                city_forecast = {}
                
                if high_temps:
                    high_temps_sorted = sorted(high_temps)
                    n = len(high_temps_sorted)
                    median_high = high_temps_sorted[n // 2] if n % 2 == 1 else (high_temps_sorted[n // 2 - 1] + high_temps_sorted[n // 2]) / 2
                    
                    # Calculate weighted mean (raw — no bias correction here;
                    # bias is applied only in build_consensus to avoid double-counting)
                    weighted_mean_high = sum(h * w for h, w in zip(high_temps, high_weights)) / sum(high_weights)
                    mean_high = median_high
                    
                    # Calculate confidence based on source agreement
                    spread = max(high_temps) - min(high_temps)
                    if spread <= 2:
                        confidence = 0.90  # Very high confidence
                    elif spread <= 4:
                        confidence = 0.80  # High confidence
                    elif spread <= 6:
                        confidence = 0.70  # Medium confidence
                    else:
                        confidence = 0.50  # Low confidence
                        
                    city_forecast["high"] = {
                        "mean": round(mean_high, 2),
                        "weighted_mean": round(weighted_mean_high, 2),
                        "median_raw": round(median_high, 2),
                        "range": (min(high_temps), max(high_temps)),
                        "confidence": confidence,
                        "sources": len(high_temps)
                    }
                    
                if low_temps:
                    low_temps_sorted = sorted(low_temps)
                    n = len(low_temps_sorted)
                    median_low = low_temps_sorted[n // 2] if n % 2 == 1 else (low_temps_sorted[n // 2 - 1] + low_temps_sorted[n // 2]) / 2
                    
                    # Raw weighted mean — no bias correction here (applied in build_consensus)
                    weighted_mean_low = sum(l * w for l, w in zip(low_temps, low_weights)) / sum(low_weights)
                    mean_low = median_low
                    
                    # Calculate confidence based on source agreement
                    spread = max(low_temps) - min(low_temps)
                    if spread <= 2:
                        confidence = 0.90
                    elif spread <= 4:
                        confidence = 0.80
                    elif spread <= 6:
                        confidence = 0.70
                    else:
                        confidence = 0.50
                        
                    city_forecast["low"] = {
                        "mean": round(mean_low, 2),
                        "weighted_mean": round(weighted_mean_low, 2),
                        "median_raw": round(median_low, 2),
                        "range": (min(low_temps), max(low_temps)),
                        "confidence": confidence,
                        "sources": len(low_temps)
                    }
                    
                if city_forecast:
                    all_city_forecasts[city_key] = city_forecast
                    
            except Exception as e:
                logger.warning("Failed to calculate consensus forecast", city=city_key, error=str(e))
                
        return all_city_forecasts

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
        all_city_forecasts: dict[str, dict[str, Any]] | None = None,
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
        temp_weights: list[float] = []
        source_details: list[dict[str, Any]] = []

        # Use dynamic weights from historical API accuracy (or base weights if no data)
        city_key = forecasts.get("city", "")
        dyn_weights = self.get_dynamic_weights(city_key)

        # Time-horizon scaling: NWS/HRRR excel at short-range, ensemble at multi-day
        target_date_str = forecasts.get("target_date", "")
        days_ahead = 1  # default
        try:
            if target_date_str:
                from datetime import date as date_cls
                target_dt = date_cls.fromisoformat(target_date_str)
                days_ahead = max(0, (target_dt - datetime.now(UTC).date()).days)
        except (ValueError, TypeError):
            pass

        short_range_sources = {"nws", "nws_hourly", "hrrr"}
        ensemble_sources = {"open_meteo"}

        for name, data in sources.items():
            temp = data.get(temp_field)
            if temp is not None:
                w = dyn_weights.get(name, 1.0)
                # Apply time-horizon multiplier
                if name in short_range_sources:
                    if days_ahead == 0:
                        w *= 1.5  # Same-day: short-range models are most accurate
                    elif days_ahead >= 3:
                        w *= 0.7  # 3+ days: short-range models lose edge
                elif name in ensemble_sources:
                    if days_ahead >= 3:
                        w *= 1.3  # Multi-day: ensemble spread captures uncertainty better
                    elif days_ahead == 0:
                        w *= 0.8  # Same-day: ensemble lags behind rapid-update models
                high_temps.append(temp)
                temp_weights.append(w)
                source_details.append({"source": name, temp_field: temp, "weight": round(w, 2)})

        if not high_temps:
            return {"error": f"No {temp_field} data from any source"}

        n = len(high_temps)

        # Weighted median: sort by temp, walk cumulative weight to find the midpoint
        paired = sorted(zip(high_temps, temp_weights), key=lambda x: x[0])
        total_weight = sum(temp_weights)
        cumulative = 0.0
        median_temp = paired[-1][0]
        for temp_val, w in paired:
            cumulative += w
            if cumulative >= total_weight / 2.0:
                median_temp = temp_val
                break
        
        # Retrieve city configuration to apply station-specific microclimate offsets
        city_code = forecasts.get("city", "")
        city_config = CITY_CONFIGS.get(city_code, {})
        
        # Bias correction: forecast models tend to over-predict temps.
        # HIGH temp: warm bias → subtract 2°F to be conservative (prevents false YES on "above X")
        # LOW temp: warm bias on lows → subtract 2°F to push consensus colder
        #           (models under-predict how cold it gets overnight)
        bias_correction = -2.0
        
        # Apply station offset
        station_offset = city_config.get("high_offset", 0.0) if market_type == "high_temp" else city_config.get("low_offset", 0.0)
        
        mean_temp = median_temp + bias_correction + station_offset
        
        # ── Spatial Correlation Arbitrage ──
        # If upstream city forecasts are higher/lower than expected, apply a fraction of that momentum.
        spatial_offset = 0.0
        try:
            upstream_city = city_config.get("upstream_city")
            if upstream_city and all_city_forecasts:
                upstream_data = all_city_forecasts.get(upstream_city)
                if upstream_data:
                    temp_key = "high" if market_type == "high_temp" else "low"
                    up_forecast = upstream_data.get(temp_key, {})
                    if up_forecast:
                        # Simple momentum heuristic: if upstream's weighted mean is significantly different 
                        # from its raw median, it indicates the high-res short-term models (NWS/HRRR) 
                        # are diverging from the broad consensus. We apply 50% of that divergence here.
                        up_divergence = up_forecast.get("weighted_mean", 0) - up_forecast.get("median_raw", 0)
                        if abs(up_divergence) >= 0.5:
                            spatial_offset = up_divergence * 0.5
                            mean_temp += spatial_offset
                            logger.info("Applied spatial correlation", city=city_code, upstream=upstream_city, offset=round(spatial_offset, 2))
        except Exception as e:
            logger.warning("Failed to apply spatial correlation", city=city_code, error=str(e))
        spread = max(high_temps) - min(high_temps) if n > 1 else 0
        confidence = max(0.0, min(1.0, 1.0 - (spread - 2) / 8))
        std_dev = self._estimate_std_dev(forecasts, is_bracket=(strike_type == "between"), market_type=market_type)

        # Use ensemble predictions if available for empirical probability
        pred_key = "all_low_predictions" if is_low else "all_predictions"
        ensemble_preds_raw = sources.get("open_meteo", {}).get(pred_key, [])
        # Apply all corrections to ensemble predictions (bias + station + spatial)
        # so ensemble-derived probabilities are consistent with the Gaussian fallback
        total_correction = bias_correction + station_offset + spatial_offset
        ensemble_preds = [p + total_correction for p in ensemble_preds_raw] if ensemble_preds_raw else []

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

    def record_forecast_accuracy(
        self,
        city: str,
        target_date: str,
        market_type: str,
        actual_temp: float,
        source_forecasts: list[dict[str, Any]],
    ) -> int:
        """Record per-source forecast accuracy into weather_api_performance.

        Called after a weather market settles to feed back into dynamic API weighting.
        Returns number of records inserted.
        """
        import os
        import sqlite3

        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "trading_engine.db")
        if not os.path.exists(db_path):
            return 0

        now = datetime.now(UTC).isoformat()
        inserted = 0
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            for sf in source_forecasts:
                source_name = sf.get("source", "")
                temp_field = "low_temp_f" if market_type == "low_temp" else "high_temp_f"
                forecast_temp = sf.get(temp_field)
                if source_name and forecast_temp is not None:
                    error = forecast_temp - actual_temp
                    c.execute(
                        """INSERT INTO weather_api_performance
                        (city, source_name, market_type, forecast_temp, actual_temp, error, target_date, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (city, source_name, market_type, forecast_temp, actual_temp, error, target_date, now),
                    )
                    inserted += 1
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to record forecast accuracy", city=city, error=str(e))
        return inserted

    async def close(self) -> None:
        await asyncio.gather(
            self.nws.close(),
            self.hrrr.close(),
            self.open_meteo.close(),
            self.tomorrow_io.close(),
            self.visual_crossing.close(),
            self.weatherbit.close(),
            self.openweathermap.close(),
        )

    def get_source_diagnostics(self) -> dict[str, Any]:
        """Expose weather provider health/cooldown info for diagnostics."""
        import time as _time

        now = _time.time()
        return {
            "nws": {
                "enabled": True,
                "cooldown_remaining_sec": 0,
            },
            "nws_hourly": {
                "enabled": True,
                "cooldown_remaining_sec": 0,
            },
            "hrrr": {
                "enabled": True,
                "cooldown_remaining_sec": 0,
            },
            "open_meteo": {
                "enabled": True,
                "cooldown_remaining_sec": max(0, int(OpenMeteoClient._backoff_until - now)),
                "cooldown_reason": OpenMeteoClient._backoff_reason,
                "cache_ttl_sec": int(OpenMeteoClient._CACHE_TTL),
                "stale_cache_ttl_sec": int(OpenMeteoClient._STALE_CACHE_TTL),
            },
            "tomorrow_io": {
                "enabled": bool(self.tomorrow_io.api_key),
                "cooldown_remaining_sec": max(0, int(TomorrowIOClient._backoff_until - now)),
                "cooldown_reason": TomorrowIOClient._backoff_reason,
                "cache_ttl_sec": int(TomorrowIOClient._CACHE_TTL),
                "stale_cache_ttl_sec": int(TomorrowIOClient._STALE_CACHE_TTL),
            },
            "visual_crossing": {
                "enabled": bool(self.visual_crossing.api_key),
                "cooldown_remaining_sec": 0,
                "cache_ttl_sec": int(VisualCrossingClient._CACHE_TTL),
            },
            "weatherbit": {
                "enabled": bool(self.weatherbit.api_key),
                "cooldown_remaining_sec": 0,
                "cache_ttl_sec": int(WeatherbitClient._CACHE_TTL),
            },
            "openweathermap": {
                "enabled": bool(self.openweathermap.api_key),
                "cooldown_remaining_sec": 0,
                "cache_ttl_sec": int(OpenWeatherMapClient._CACHE_TTL),
            },
        }
