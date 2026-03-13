from __future__ import annotations

import asyncio
import sys
import types
import unittest
from datetime import datetime


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

sys.modules.setdefault(
    "app.services.kalshi_ws",
    types.SimpleNamespace(
        KalshiWebSocket=object,
        get_kalshi_ws=lambda: types.SimpleNamespace(connected=False),
    ),
)
sys.modules.setdefault("app.config", types.SimpleNamespace(get_settings=lambda: types.SimpleNamespace()))
sys.modules.setdefault("app.services.kalshi_api", types.SimpleNamespace(get_kalshi_client=lambda: None))
sys.modules.setdefault("app.services.kalshi_scanner", types.SimpleNamespace(KalshiScanner=object, parse_parlay_legs=lambda _t: []))
sys.modules.setdefault("app.services.cross_strategy_correlation", types.SimpleNamespace(get_cross_strategy_engine=lambda: None))
sys.modules.setdefault("app.services.crypto_data", types.SimpleNamespace(CryptoDataService=object))
sys.modules.setdefault("app.services.econ_data", types.SimpleNamespace(EconDataService=object))
sys.modules.setdefault("app.services.finance_data", types.SimpleNamespace(FinanceDataService=object))
sys.modules.setdefault("app.services.nba_data", types.SimpleNamespace(NBADataService=object, get_nba_data=lambda: None))
sys.modules.setdefault("app.services.news_sentiment", types.SimpleNamespace(get_market_news_sentiment=lambda: None))
sys.modules.setdefault("app.services.polymarket_data", types.SimpleNamespace(get_polymarket_data=lambda: None))
sys.modules.setdefault("app.services.referee_data", types.SimpleNamespace(RefereeDataService=object, get_referee_data=lambda **_k: None))
sys.modules.setdefault("app.services.signal_scorer", types.SimpleNamespace(get_signal_scorer=lambda: None))
sys.modules.setdefault("app.services.smart_predictor", types.SimpleNamespace(SmartPredictor=object, get_smart_predictor=lambda: None))
sys.modules.setdefault("app.services.trade_analyzer", types.SimpleNamespace(get_trade_analyzer=lambda: None))
sys.modules.setdefault("app.services.discord_webhook", types.SimpleNamespace(send_discord_notification=lambda *_a, **_k: None))
sys.modules.setdefault("app.services.event_bus", types.SimpleNamespace(get_event_bus=lambda: None))
sys.modules.setdefault("app.services.weather_data", types.SimpleNamespace(CITY_CONFIGS={}, WeatherConsensus=object))


class FinanceDirectionParsingTests(unittest.TestCase):
    def test_finance_market_skips_ambiguous_directional_titles(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        agent = KalshiAgent.__new__(KalshiAgent)
        agent.finance = types.SimpleNamespace(get_bracket_probability=lambda **_kwargs: 0.5)
        agent.adaptive = types.SimpleNamespace(get_thresholds=lambda _strategy: {"min_edge": 0.03, "min_confidence": 0.2})
        agent.engine = types.SimpleNamespace(log_event=lambda *a, **k: None, record_signal=lambda *a, **k: None)
        agent.polymarket = types.SimpleNamespace(get_edge_signal=self._async_return_none())
        agent._record_cross_signal = lambda *a, **k: None
        agent._apply_correlation_adjustment = lambda _strategy, _ticker, confidence: confidence

        market = {
            "ticker": "KXFINTEST-26MAR12",
            "title": "Will the S&P 500 finish at 5800?",
            "finance": {"index": "SPX"},
            "yes_bid": 45,
            "yes_ask": 46,
            "no_bid": 53,
            "no_ask": 54,
            "strike_type": "greater",
            "floor_strike": 5800,
        }
        signal_by_index = {
            "SPX": {
                "p_up": 0.63,
                "confidence": 0.8,
                "current_price": 5795,
                "intraday_momentum": 0.2,
                "futures_signal": 0.2,
                "vix_signal": 0.15,
                "ma_signal": 0.18,
                "news_sentiment": 0.11,
                "vix_level": 18.0,
            }
        }

        result = asyncio.run(agent._evaluate_finance_market(market, signal_by_index))
        self.assertIsNone(result)

    @staticmethod
    def _async_return_none():
        async def _inner(*_args, **_kwargs):
            return None

        return _inner


class EconDirectionParsingTests(unittest.TestCase):
    @staticmethod
    def _async_return_none():
        async def _inner(*_args, **_kwargs):
            return None

        return _inner

    def test_econ_below_market_inverts_yes_probability(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        agent = KalshiAgent.__new__(KalshiAgent)
        agent.econ = types.SimpleNamespace(estimate_probability_above=lambda *_a, **_k: 0.7)
        agent.adaptive = types.SimpleNamespace(get_thresholds=lambda _strategy: {"min_edge": 0.03, "min_confidence": 0.2})
        recorded: dict[str, float | str] = {}
        agent.engine = types.SimpleNamespace(
            log_event=lambda *a, **k: None,
            record_signal=lambda **kwargs: recorded.update(kwargs),
        )
        agent.polymarket = types.SimpleNamespace(get_edge_signal=self._async_return_none())
        agent._record_cross_signal = lambda *a, **k: None
        agent._apply_correlation_adjustment = lambda _strategy, _ticker, confidence: confidence

        market = {
            "ticker": "KXECONTEST-26MAR12",
            "title": "Will unemployment be below 4.0%?",
            "econ": {"type": "unemployment"},
            "yes_bid": 45,
            "yes_ask": 46,
            "no_bid": 53,
            "no_ask": 54,
        }
        econ_signals = {
            "unemployment": {"estimated_next_rate": 4.2}
        }

        result = asyncio.run(agent._evaluate_econ_market(market, econ_signals))

        self.assertIsNotNone(result)
        self.assertEqual(result["side"], "no")
        self.assertAlmostEqual(float(result["our_prob"]), 0.7, places=3)

    def test_econ_ambiguous_threshold_title_is_skipped(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        agent = KalshiAgent.__new__(KalshiAgent)
        agent.econ = types.SimpleNamespace(estimate_probability_above=lambda *_a, **_k: 0.7)
        agent.adaptive = types.SimpleNamespace(get_thresholds=lambda _strategy: {"min_edge": 0.03, "min_confidence": 0.2})
        agent.engine = types.SimpleNamespace(log_event=lambda *a, **k: None, record_signal=lambda *a, **k: None)
        agent.polymarket = types.SimpleNamespace(get_edge_signal=self._async_return_none())
        agent._record_cross_signal = lambda *a, **k: None
        agent._apply_correlation_adjustment = lambda _strategy, _ticker, confidence: confidence

        market = {
            "ticker": "KXECONTEST-26MAR12",
            "title": "Will unemployment be 4.0%?",
            "econ": {"type": "unemployment"},
            "yes_bid": 45,
            "yes_ask": 46,
            "no_bid": 53,
            "no_ask": 54,
        }
        econ_signals = {
            "unemployment": {"estimated_next_rate": 4.2}
        }

        result = asyncio.run(agent._evaluate_econ_market(market, econ_signals))
        self.assertIsNone(result)


class SportsSuffixMatchingTests(unittest.TestCase):
    def test_suffix_match_uses_aliases_for_club_abbreviations(self) -> None:
        from app.services.kalshi_agent import _suffix_matches_team_name

        self.assertTrue(_suffix_matches_team_name("PSG", "Paris Saint-Germain"))

    def test_suffix_match_rejects_broad_prefix_collisions(self) -> None:
        from app.services.kalshi_agent import _suffix_matches_team_name

        self.assertFalse(_suffix_matches_team_name("REAL", "Real Sociedad"))

    def test_person_name_match_allows_initial_and_last_name(self) -> None:
        from app.services.kalshi_agent import _person_name_matches

        self.assertTrue(_person_name_matches("J Brunson", "Jalen Brunson"))

    def test_person_name_match_rejects_same_prefix_different_player(self) -> None:
        from app.services.kalshi_agent import _person_name_matches

        self.assertFalse(_person_name_matches("Jalen Williams", "Jaylin Williams"))

    def test_find_unique_prop_match_fails_closed_on_ambiguous_prefix(self) -> None:
        from app.services.kalshi_agent import _find_unique_prop_match

        odds_lookup = {
            "jalen williams|points": {"consensus_over_prob": 0.58},
            "jaylin williams|points": {"consensus_over_prob": 0.41},
        }

        self.assertIsNone(_find_unique_prop_match(odds_lookup, "J Williams", "points"))


class LivePositionRecordTests(unittest.TestCase):
    def test_live_position_record_includes_fees_in_max_profit_and_risk(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        agent = KalshiAgent.__new__(KalshiAgent)
        record = agent._build_live_position_record(
            {
                "ticker": "KXHIGHNYC-26MAR12-T75",
                "position": 2,
                "market_exposure": 80,
                "fees_paid": 4,
                "market_title": "NYC High Temp",
            },
            {"avg_our_prob": 0.62, "strategy": "weather"},
            include_title=True,
        )

        self.assertEqual(record["total_cost"], 0.8)
        self.assertEqual(record["total_fees"], 0.04)
        self.assertEqual(record["max_risk"], 0.84)
        self.assertEqual(record["max_profit"], 1.16)
        self.assertEqual(round(0.8 - record["total_cost"] - record["total_fees"], 2), -0.04)

    def test_compute_side_market_prices_separates_mark_from_exit_price(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        mark_price, exit_price = KalshiAgent._compute_side_market_prices(
            side="yes",
            yes_bid=40,
            yes_ask=90,
            no_bid=10,
            no_ask=60,
            last_price=0,
            avg_entry_cents=50,
        )

        self.assertEqual(mark_price, 90)
        self.assertEqual(exit_price, 40)

    def test_compute_side_market_prices_ignores_stale_last_outside_book(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        mark_price, exit_price = KalshiAgent._compute_side_market_prices(
            side="yes",
            yes_bid=40,
            yes_ask=90,
            no_bid=10,
            no_ask=60,
            last_price=99,
            avg_entry_cents=50,
        )

        self.assertEqual(mark_price, 90)
        self.assertEqual(exit_price, 40)

    def test_compute_hold_ev_uses_thesis_prob_against_exit_price(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        self.assertAlmostEqual(
            KalshiAgent._compute_hold_ev_cents(our_prob=0.62, exit_price=70),
            -8.0,
            places=3,
        )

    def test_compute_position_pnl_distinguishes_mark_from_liquidation(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        marked = KalshiAgent._compute_position_pnl(
            contracts=2,
            price_cents=90,
            total_cost=0.8,
            total_fees=0.04,
        )
        liquidated = KalshiAgent._compute_position_pnl(
            contracts=2,
            price_cents=40,
            total_cost=0.8,
            total_fees=0.04,
        )

        self.assertEqual(marked, 0.96)
        self.assertEqual(liquidated, -0.04)

    def test_nba_prop_time_gate_fails_closed_on_unparseable_ticker(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        self.assertFalse(KalshiAgent._nba_prop_within_time_gate("KXNBAPTS-BADTICKER"))

    def test_nba_prop_time_gate_allows_within_twelve_hours(self) -> None:
        from app.services.kalshi_agent import KalshiAgent
        from zoneinfo import ZoneInfo

        now_dt = datetime(2026, 3, 12, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        self.assertTrue(KalshiAgent._nba_prop_within_time_gate("KXNBAPTS-26MAR12PHXSAS-BOOKER", now_dt=now_dt))


class WeatherObservedTradeTests(unittest.TestCase):
    def test_observed_weather_trade_allows_large_real_edge(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        agent = KalshiAgent.__new__(KalshiAgent)
        agent.engine = types.SimpleNamespace(
            log_event=lambda *a, **k: None,
            record_signal=lambda **_kwargs: "sig-1",
            calculate_position_size=lambda **_kwargs: 1,
        )
        agent._enrich_weather_title = lambda _ticker, title: title

        market = {
            "ticker": "KXHIGHTNYC-26MAR12-T60",
            "title": "NYC high above 60",
            "yes_bid": 39,
            "yes_ask": 40,
            "no_bid": 60,
            "no_ask": 61,
            "volume": 200,
            "weather": {
                "city_code": "NYC",
                "market_type": "high_temp",
                "strike_type": "greater",
                "floor_strike": 60,
            },
        }
        obs = {
            "observed_high_f": 61.0,
            "observed_low_f": 48.0,
            "obs_count": 12,
        }

        result = asyncio.run(agent._evaluate_weather_market_observed(market, obs))

        self.assertIsNotNone(result)
        self.assertEqual(result["side"], "yes")
        self.assertGreater(result["edge"], 0.5)


class WeatherDiagnosticsTests(unittest.TestCase):
    def test_weather_diagnostics_exposes_thresholds_providers_and_near_misses(self) -> None:
        from app.services.kalshi_agent import KalshiAgent

        agent = KalshiAgent.__new__(KalshiAgent)
        agent.weather = types.SimpleNamespace(
            get_source_diagnostics=lambda: {
                "tomorrow_io": {"enabled": True, "cooldown_remaining_sec": 120},
            },
        )
        agent.engine = types.SimpleNamespace(
            get_near_miss_summary=lambda strategy, limit=8: [{"reason": "observed_price_too_low", "count": 3}],
            get_candidate_rejections=lambda **_kwargs: [{"ticker": "KXHIGHNYC-26MAR12-T72", "reason": "observed_price_too_low"}],
            get_trades=lambda **_kwargs: [
                {"ticker": "A", "signal_source": "forecast_weather", "status": "filled"},
                {"ticker": "B", "signal_source": "weather_observed_arbitrage", "status": "filled", "pnl": 1.2},
            ],
            get_recent_source_quality=lambda _source, lookback_days=45: {
                "trades": 9,
                "avg_pnl": 0.6,
                "win_rate": 0.56,
            },
        )

        diagnostics = agent.get_weather_diagnostics()

        self.assertEqual(diagnostics["observed_thresholds"]["min_edge"], 0.07)
        self.assertIn("tomorrow_io", diagnostics["provider_health"])
        self.assertEqual(diagnostics["near_miss_summary"][0]["reason"], "observed_price_too_low")
        self.assertEqual(len(diagnostics["recent_observed_trades"]), 1)
        self.assertEqual(diagnostics["recent_observed_trades"][0]["ticker"], "B")


if __name__ == "__main__":
    unittest.main()
