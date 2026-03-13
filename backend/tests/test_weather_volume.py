from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _StructlogStub(types.SimpleNamespace):
    def get_logger(self, *_args, **_kwargs):
        return types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        )


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


class WeatherVolumeTests(unittest.TestCase):
    def test_weather_volume_diagnostics_summarize_blockers_and_funnel(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=200.0, paper_mode=True)
                engine.log_event("info", "Weather cycle starting", strategy="weather")
                engine.log_event("info", "Weather cycle complete: 0 candidates", strategy="weather")
                engine.record_candidate_rejection(
                    "weather",
                    "no_two_sided_market",
                    ticker="KXHIGHTNOLA-26MAR13-T78",
                    stage="forecast_precheck",
                    signal_source="weather_consensus",
                )

                diagnostics = engine.get_weather_volume_diagnostics(days=7)

        self.assertEqual(diagnostics["funnel"]["weather_cycles_started"], 1)
        self.assertEqual(diagnostics["funnel"]["signals_recorded"], 0)
        self.assertEqual(diagnostics["top_blockers"][0]["reason"], "no_two_sided_market")
        self.assertEqual(diagnostics["top_one_sided_tickers"][0]["ticker"], "KXHIGHTNOLA-26MAR13-T78")


if __name__ == "__main__":
    unittest.main()
