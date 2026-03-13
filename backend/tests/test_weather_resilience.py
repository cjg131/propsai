from __future__ import annotations

import asyncio
import sys
import types
import unittest
from datetime import date
from unittest.mock import AsyncMock, patch

import httpx


class _StructlogStub(types.SimpleNamespace):
    def get_logger(self, *_args, **_kwargs):
        return types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None, debug=lambda *a, **k: None)


sys.modules.setdefault(
    "structlog",
    _StructlogStub(
        configure=lambda **_kwargs: None,
        get_logger=lambda *_args, **_kwargs: types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        ),
        contextvars=types.SimpleNamespace(merge_contextvars=lambda *_a, **_k: None),
        processors=types.SimpleNamespace(
            add_log_level=lambda *_a, **_k: None,
            StackInfoRenderer=lambda *_a, **_k: None,
            TimeStamper=lambda *_a, **_k: None,
            JSONRenderer=lambda *_a, **_k: None,
        ),
        dev=types.SimpleNamespace(
            set_exc_info=lambda *_a, **_k: None,
            ConsoleRenderer=lambda *_a, **_k: None,
        ),
        make_filtering_bound_logger=lambda *_a, **_k: None,
        PrintLoggerFactory=lambda *_a, **_k: None,
        stdlib=types.SimpleNamespace(BoundLogger=object),
    ),
)


class TomorrowIOResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop("app.services.weather_data", None)
        from app.services.weather_data import TomorrowIOClient

        TomorrowIOClient._cache = {}
        TomorrowIOClient._cache_ts = {}
        TomorrowIOClient._backoff_until = 0.0
        TomorrowIOClient._backoff_reason = ""

    def test_tomorrow_io_uses_stale_cache_after_429(self) -> None:
        from app.services.weather_data import TomorrowIOClient

        client = TomorrowIOClient(api_key="test-key")
        cache_key = "NYC_2026-03-12"
        cached = {"source": "tomorrow_io", "city": "NYC", "high_temp_f": 61.0}
        TomorrowIOClient._cache[cache_key] = cached
        TomorrowIOClient._cache_ts[cache_key] = 0.0

        request = httpx.Request("GET", "https://api.tomorrow.io/v4/weather/forecast")
        response = httpx.Response(429, request=request)

        with patch("time.time", return_value=12000.0):
            client._http.get = AsyncMock(side_effect=httpx.HTTPStatusError("rate limited", request=request, response=response))
            result = asyncio.run(client.get_forecast("NYC", date(2026, 3, 12)))

        self.assertEqual(result, cached)
        self.assertGreater(TomorrowIOClient._backoff_until, 12000.0)

    def test_tomorrow_io_skips_request_during_backoff_and_uses_stale_cache(self) -> None:
        from app.services.weather_data import TomorrowIOClient

        client = TomorrowIOClient(api_key="test-key")
        cache_key = "NYC_2026-03-12"
        cached = {"source": "tomorrow_io", "city": "NYC", "high_temp_f": 61.0}
        TomorrowIOClient._cache[cache_key] = cached
        TomorrowIOClient._cache_ts[cache_key] = 0.0
        TomorrowIOClient._backoff_until = 20000.0
        TomorrowIOClient._backoff_reason = "rate_limited"

        client._http.get = AsyncMock(side_effect=AssertionError("request should not be sent during backoff"))
        with patch("time.time", return_value=12000.0):
            result = asyncio.run(client.get_forecast("NYC", date(2026, 3, 12)))

        self.assertEqual(result, cached)


class OpenMeteoResilienceTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.modules.pop("app.services.weather_data", None)
        from app.services.weather_data import OpenMeteoClient

        OpenMeteoClient._cache = {}
        OpenMeteoClient._cache_ts = {}
        OpenMeteoClient._backoff_until = 0.0
        OpenMeteoClient._backoff_reason = ""

    def test_open_meteo_uses_stale_cache_after_429(self) -> None:
        from app.services.weather_data import OpenMeteoClient

        client = OpenMeteoClient()
        cache_key = "NYC_2026-03-12"
        cached = {"source": "open_meteo", "city": "NYC", "high_temp_f": 61.0}
        OpenMeteoClient._cache[cache_key] = cached
        OpenMeteoClient._cache_ts[cache_key] = 0.0

        request = httpx.Request("GET", "https://ensemble-api.open-meteo.com/v1/ensemble")
        response = httpx.Response(429, request=request)

        with patch("time.time", return_value=12000.0):
            client._http.get = AsyncMock(side_effect=httpx.HTTPStatusError("rate limited", request=request, response=response))
            result = asyncio.run(client.get_ensemble_forecast("NYC", date(2026, 3, 12)))

        self.assertEqual(result, cached)
        self.assertGreater(OpenMeteoClient._backoff_until, 12000.0)

    def test_open_meteo_skips_request_during_backoff_and_uses_stale_cache(self) -> None:
        from app.services.weather_data import OpenMeteoClient

        client = OpenMeteoClient()
        cache_key = "NYC_2026-03-12"
        cached = {"source": "open_meteo", "city": "NYC", "high_temp_f": 61.0}
        OpenMeteoClient._cache[cache_key] = cached
        OpenMeteoClient._cache_ts[cache_key] = 0.0
        OpenMeteoClient._backoff_until = 20000.0
        OpenMeteoClient._backoff_reason = "rate_limited"

        client._http.get = AsyncMock(side_effect=AssertionError("request should not be sent during backoff"))
        with patch("time.time", return_value=12000.0):
            result = asyncio.run(client.get_ensemble_forecast("NYC", date(2026, 3, 12)))

        self.assertEqual(result, cached)


class WeatherScannerHydrationTests(unittest.TestCase):
    def test_weather_scanner_hydrates_missing_quotes_from_market_detail(self) -> None:
        sys.modules.pop("app.services.kalshi_scanner", None)
        sys.modules.pop("app.services.kalshi_api", None)
        from app.services.kalshi_scanner import KalshiScanner

        kalshi = types.SimpleNamespace(
            get_market=AsyncMock(return_value={
                "market": {
                    "ticker": "KXHIGHTNOLA-26MAR13-T78",
                    "yes_ask_dollars": "0.0200",
                    "yes_bid_dollars": "0.0100",
                    "no_ask_dollars": "0.9900",
                    "no_bid_dollars": "0.9800",
                    "volume_fp": "472.00",
                    "open_interest_fp": "471.00",
                },
            }),
        )
        scanner = KalshiScanner(kalshi)

        hydrated = asyncio.run(scanner._hydrate_weather_market_quotes({
            "ticker": "KXHIGHTNOLA-26MAR13-T78",
            "yes_ask": None,
            "no_ask": None,
        }))
        parsed = scanner._enrich_market(hydrated, "weather")

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["yes_ask"], 2)
        self.assertEqual(parsed["yes_bid"], 1)
        self.assertEqual(parsed["no_ask"], 99)
        self.assertEqual(parsed["no_bid"], 98)
        self.assertEqual(parsed["volume"], 472)

    def test_weather_scanner_hydrates_when_one_side_is_missing(self) -> None:
        sys.modules.pop("app.services.kalshi_scanner", None)
        sys.modules.pop("app.services.kalshi_api", None)
        from app.services.kalshi_scanner import KalshiScanner

        kalshi = types.SimpleNamespace(
            get_market=AsyncMock(return_value={
                "market": {
                    "ticker": "KXHIGHTNOLA-26MAR13-T78",
                    "yes_ask_dollars": "0.4200",
                    "yes_bid_dollars": "0.4100",
                    "no_ask_dollars": "0.5900",
                    "no_bid_dollars": "0.5800",
                },
            }),
        )
        scanner = KalshiScanner(kalshi)

        hydrated = asyncio.run(scanner._hydrate_weather_market_quotes({
            "ticker": "KXHIGHTNOLA-26MAR13-T78",
            "yes_ask": 42,
            "yes_bid": 41,
            "no_ask": None,
            "no_bid": None,
        }))

        self.assertEqual(hydrated["yes_ask_dollars"], "0.4200")
        self.assertEqual(hydrated["no_ask_dollars"], "0.5900")

    def test_weather_scan_stats_count_rescued_quotes(self) -> None:
        sys.modules.pop("app.services.kalshi_scanner", None)
        sys.modules.pop("app.services.kalshi_api", None)
        from app.services.kalshi_scanner import KalshiScanner

        async def _get_markets(*, series_ticker: str, limit: int = 50, **_kwargs):
            if series_ticker != "KXHIGHTNOLA":
                return {"markets": []}
            return {
                "markets": [
                    {
                        "ticker": "KXHIGHTNOLA-26MAR13-T78",
                        "status": "active",
                        "yes_ask": 42,
                        "yes_bid": 41,
                        "no_ask": None,
                        "no_bid": None,
                    }
                ]
            }

        kalshi = types.SimpleNamespace(
            get_markets=AsyncMock(side_effect=_get_markets),
            get_market=AsyncMock(return_value={
                "market": {
                    "ticker": "KXHIGHTNOLA-26MAR13-T78",
                    "yes_ask_dollars": "0.4200",
                    "yes_bid_dollars": "0.4100",
                    "no_ask_dollars": "0.5900",
                    "no_bid_dollars": "0.5800",
                    "status": "active",
                },
            }),
        )
        scanner = KalshiScanner(kalshi)
        markets = asyncio.run(scanner.scan_weather_markets())
        stats = scanner.get_weather_scan_stats()

        self.assertEqual(len(markets), 1)
        self.assertEqual(stats["markets_seen"], 1)
        self.assertEqual(stats["hydration_attempted"], 1)
        self.assertEqual(stats["detail_updates"], 1)
        self.assertEqual(stats["rescued_two_sided_asks"], 1)
        self.assertEqual(stats["rescued_full_quotes"], 1)


class KalshiClientOrderbookPathTests(unittest.TestCase):
    def test_get_orderbook_uses_market_orderbook_path(self) -> None:
        sys.modules.pop("app.services.kalshi_api", None)
        from app.services.kalshi_api import KalshiClient

        client = KalshiClient.__new__(KalshiClient)
        client._get = AsyncMock(return_value={"orderbook": {}})

        result = asyncio.run(client.get_orderbook("KXTEST"))

        self.assertEqual(result, {"orderbook": {}})
        client._get.assert_awaited_once_with("/markets/KXTEST/orderbook")


if __name__ == "__main__":
    unittest.main()
