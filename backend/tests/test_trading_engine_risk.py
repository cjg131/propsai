from __future__ import annotations

import asyncio
import os
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


class TradingEngineRiskTests(unittest.TestCase):
    def test_execute_trade_blocks_duplicate_open_position(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)

                first = asyncio.run(
                    engine.execute_trade(
                        strategy="weather",
                        ticker="KXHIGHNYC-26MAR12-T75",
                        side="yes",
                        count=5,
                        price_cents=45,
                        our_prob=0.62,
                        kalshi_prob=0.45,
                        signal_source="weather_observed_arbitrage",
                    )
                )
                second = asyncio.run(
                    engine.execute_trade(
                        strategy="weather",
                        ticker="KXHIGHNYC-26MAR12-T75",
                        side="yes",
                        count=5,
                        price_cents=46,
                        our_prob=0.63,
                        kalshi_prob=0.46,
                        signal_source="weather_observed_arbitrage",
                    )
                )

                self.assertEqual(first["status"], "filled")
                self.assertEqual(second["status"], "blocked")
                self.assertIn("Open position already exists", second["reason"])

    def test_check_risk_limits_counts_fees_toward_trade_cap(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(
                    bankroll=2000.0,
                    paper_mode=True,
                    max_single_trade_dollars=30.0,
                    max_bet_pct=1.0,
                    max_total_exposure_pct=1.0,
                    max_strategy_exposure_pct=1.0,
                    max_strategy_cycle_pct=1.0,
                )
                allowed, reason = engine.check_risk_limits(
                    "weather",
                    cost=29.99,
                    ticker="KXHIGHMIA-26MAR12-T80",
                    count=100,
                    price_cents=30,
                )

                self.assertFalse(allowed)
                self.assertIn("exceeds absolute cap", reason)

    def test_low_bankroll_profile_applies_for_two_hundred_dollars(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(
                    bankroll=200.0,
                    paper_mode=False,
                )

                self.assertLessEqual(engine.max_single_trade_dollars, 5.0)
                self.assertLessEqual(engine.max_total_exposure_pct, 0.15)
                self.assertLessEqual(engine.max_strategy_cycle_pct, 0.05)
                self.assertTrue(engine.strategy_enabled["weather"])
                self.assertFalse(engine.strategy_enabled["sports"])
                self.assertFalse(engine.strategy_enabled["finance"])

    def test_sync_bankroll_recomputes_profile(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=False)
                engine.sync_bankroll(200.0)

                self.assertFalse(engine.paper_mode)
                self.assertEqual(engine.bankroll, 200.0)
                self.assertLessEqual(engine.max_single_trade_dollars, 5.0)
                self.assertLessEqual(engine.max_total_exposure_pct, 0.15)

    def test_load_env_defaults_populates_missing_process_env(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("PAPER_MODE=false\nBANKROLL=145.5\n")

            with patch.object(trading_engine_module, "ENV_PATH", env_path):
                with patch.dict(os.environ, {}, clear=True):
                    trading_engine_module._load_env_defaults()
                    self.assertEqual(os.environ["PAPER_MODE"], "false")
                    self.assertEqual(os.environ["BANKROLL"], "145.5")

    def test_live_mode_reenforces_weather_only_strategy_policy(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                with patch.dict(
                    os.environ,
                    {"LIVE_ENABLED_STRATEGIES": "weather", "PAPER_ENABLED_STRATEGIES": "weather"},
                    clear=False,
                ):
                    engine = trading_engine_module.TradingEngine(bankroll=200.0, paper_mode=True)
                    engine.strategy_enabled["sports"] = True
                    engine.set_paper_mode(False)

                    self.assertTrue(engine.strategy_enabled["weather"])
                    self.assertFalse(engine.strategy_enabled["sports"])
                    self.assertEqual(engine.allowed_live_strategies, {"weather"})

                    allowed, reason = engine.can_enable_strategy("sports")
                    self.assertFalse(allowed)
                    self.assertIn("not allowed in live mode", reason or "")

    def test_live_strategy_policy_blocks_execution_even_after_manual_toggle(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                with patch.dict(
                    os.environ,
                    {"LIVE_ENABLED_STRATEGIES": "weather", "PAPER_ENABLED_STRATEGIES": "weather"},
                    clear=False,
                ):
                    engine = trading_engine_module.TradingEngine(bankroll=200.0, paper_mode=False)
                    engine.strategy_enabled["sports"] = True

                    allowed, reason = engine.check_risk_limits(
                        "sports",
                        cost=3.0,
                        ticker="KXNBAGAME-TEST",
                        count=10,
                        price_cents=30,
                    )

                    self.assertFalse(allowed)
                    self.assertIn("not allowed in live mode", reason)

    def test_switching_back_to_paper_restores_paper_allowed_strategies(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                with patch.dict(
                    os.environ,
                    {"LIVE_ENABLED_STRATEGIES": "weather", "PAPER_ENABLED_STRATEGIES": "weather,sports"},
                    clear=False,
                ):
                    engine = trading_engine_module.TradingEngine(bankroll=200.0, paper_mode=False)
                    engine.set_paper_mode(True)

                    self.assertTrue(engine.strategy_enabled["weather"])
                    self.assertTrue(engine.strategy_enabled["sports"])
                    self.assertFalse(engine.strategy_enabled["crypto"])

    def test_has_open_position_uses_broker_snapshot_when_db_is_empty(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=200.0, paper_mode=False)
                engine.set_broker_positions_snapshot([
                    {
                        "ticker": "KXHIGHNYC-26MAR12-T75",
                        "exposure_dollars": 4.25,
                        "strategy": "weather",
                    }
                ])

                self.assertTrue(engine.has_open_position("KXHIGHNYC-26MAR12-T75"))

    def test_exposure_nets_filled_sell_proceeds(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status)
                    VALUES ('buy1', datetime('now'), 'weather', 'KXHIGHNYC-26MAR12-T75', 'yes', 'buy', 2, 50,
                            1.00, 0.02, 1, 'o1', 'filled')"""
                )
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status)
                    VALUES ('sell1', datetime('now'), 'weather', 'KXHIGHNYC-26MAR12-T75', 'yes', 'sell', 1, 60,
                            0.60, 0.01, 1, 'o2', 'filled')"""
                )
                conn.commit()
                conn.close()

                self.assertAlmostEqual(engine.get_ticker_exposure("KXHIGHNYC-26MAR12-T75"), 0.43, places=2)
                self.assertAlmostEqual(engine.get_total_exposure("weather"), 0.43, places=2)


class DynamicAllocationTests(unittest.TestCase):
    def test_dynamic_allocations_include_realized_exit_losses(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()

                rows = [
                    ("w1", "weather", -12.0),
                    ("w2", "weather", -9.0),
                    ("f1", "finance", 6.0),
                    ("f2", "finance", 5.0),
                ]
                for trade_id, strategy, pnl in rows:
                    c.execute(
                        """INSERT INTO trades
                        (id, timestamp, strategy, ticker, side, action, count, price_cents,
                         cost, fee, paper_mode, order_id, status, pnl, settled_at)
                        VALUES (?, datetime('now'), ?, ?, 'yes', 'sell', 1, 50, -0.5, 0, 1, ?, 'filled', ?, datetime('now'))""",
                        (trade_id, strategy, f"{strategy}-{trade_id}", trade_id, pnl),
                    )
                conn.commit()
                conn.close()

                engine.strategy_enabled["weather"] = True
                engine.strategy_enabled["finance"] = True

                allocations = engine.get_dynamic_strategy_allocations()

                self.assertLess(allocations["weather"], allocations["finance"])


class CandidateRejectionTests(unittest.TestCase):
    def test_candidate_rejection_summary_tracks_near_misses(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=200.0, paper_mode=False)
                engine.record_candidate_rejection(
                    "weather",
                    "observed_price_too_low",
                    ticker="KXHIGHNYC-26MAR12-T60",
                    signal_source="weather_observed_arbitrage",
                    stage="observed_price",
                    near_miss=True,
                )
                engine.record_candidate_rejection(
                    "weather",
                    "observed_price_too_low",
                    ticker="KXHIGHNYC-26MAR12-T61",
                    signal_source="weather_observed_arbitrage",
                    stage="observed_price",
                    near_miss=True,
                )

                rows = engine.get_candidate_rejections(strategy="weather", near_miss_only=True, limit=10)
                summary = engine.get_near_miss_summary("weather", limit=10)

                self.assertEqual(len(rows), 2)
                self.assertEqual(summary[0]["reason"], "observed_price_too_low")
                self.assertEqual(summary[0]["count"], 2)


class CandidateQualityTests(unittest.TestCase):
    def test_evaluate_candidate_quality_blocks_sports_parlays(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)

                quality = engine.evaluate_candidate_quality({
                    "strategy": "sports",
                    "ticker": "KXPARLAY-26MAR12-NBA",
                    "signal_source": "parlay_pricer",
                    "edge": 0.12,
                    "confidence": 0.92,
                    "our_prob": 0.71,
                    "price_cents": 44,
                })

                self.assertFalse(quality["allowed"])
                self.assertEqual(quality["family"], "sports_parlay")

    def test_observed_weather_scores_above_forecast_weather(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                engine._performance_model_cache = {}

                observed = engine.evaluate_candidate_quality({
                    "strategy": "weather",
                    "ticker": "KXHIGHNYC-26MAR12-T75",
                    "signal_source": "weather_observed_arbitrage",
                    "edge": 0.14,
                    "confidence": 0.90,
                    "our_prob": 0.92,
                    "price_cents": 47,
                })
                forecast = engine.evaluate_candidate_quality({
                    "strategy": "weather",
                    "ticker": "KXHIGHNYC-26MAR13-T75",
                    "signal_source": "weather_consensus",
                    "edge": 0.14,
                    "confidence": 0.90,
                    "our_prob": 0.92,
                    "price_cents": 47,
                })

                self.assertTrue(observed["allowed"])
                self.assertTrue(forecast["allowed"])
                self.assertGreater(observed["quality_score"], forecast["quality_score"])

    def test_performance_model_can_block_bad_ticker(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                engine._performance_model_cache = {
                    "blocked_tickers": ["KXHIGHNYC-26MAR12-T75"],
                    "blocked_events": [],
                    "family_multipliers": {},
                    "signal_source_multipliers": {},
                    "price_bucket_multipliers": {},
                }
                quality = engine.evaluate_candidate_quality({
                    "strategy": "weather",
                    "ticker": "KXHIGHNYC-26MAR12-T75",
                    "signal_source": "weather_observed_arbitrage",
                    "edge": 0.14,
                    "confidence": 0.90,
                    "our_prob": 0.92,
                    "price_cents": 47,
                })

                self.assertFalse(quality["allowed"])
                self.assertIn("blacklist", " ".join(quality["reasons"]))

    def test_performance_model_can_quarantine_family_and_source(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                engine._performance_model_cache = {
                    "blocked_tickers": [],
                    "blocked_events": [],
                    "blocked_families": ["sports_single_soccer"],
                    "blocked_sources": ["sharp_single_game"],
                    "family_multipliers": {},
                    "signal_source_multipliers": {},
                    "price_bucket_multipliers": {},
                }
                quality = engine.evaluate_candidate_quality({
                    "strategy": "sports",
                    "ticker": "KXEPLGAME-26MAR04FULWHU-FUL",
                    "signal_source": "sharp_single_game",
                    "edge": 0.09,
                    "confidence": 0.82,
                    "our_prob": 0.64,
                    "price_cents": 44,
                })

                self.assertFalse(quality["allowed"])
                reason_text = " ".join(quality["reasons"])
                self.assertIn("auto-quarantined", reason_text)


class LifetimeConcentrationTests(unittest.TestCase):
    def test_execute_trade_blocks_after_lifetime_ticker_cap(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)

                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status, signal_source)
                    VALUES (?, datetime('now'), 'weather', ?, 'yes', 'buy', 10, 45,
                            4.5, 0.01, 1, 'seed', 'settled', 'weather_consensus')""",
                    ("seed-trade", "KXHIGHNYC-26MAR12-T75"),
                )
                conn.commit()
                conn.close()

                blocked = asyncio.run(
                    engine.execute_trade(
                        strategy="weather",
                        ticker="KXHIGHNYC-26MAR12-T75",
                        side="yes",
                        count=5,
                        price_cents=46,
                        our_prob=0.70,
                        kalshi_prob=0.46,
                        signal_source="weather_consensus",
                    )
                )

                self.assertEqual(blocked["status"], "blocked")
                self.assertIn("lifetime", blocked["reason"])


class SettlementAndExitTests(unittest.TestCase):
    def test_get_unsettled_trades_excludes_sell_exits(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status)
                    VALUES ('buy1', datetime('now'), 'weather', 'KXHIGHNYC-26MAR12-T75', 'yes', 'buy', 1, 45,
                            0.45, 0.01, 1, 'o1', 'filled')"""
                )
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status, result, pnl, settled_at)
                    VALUES ('sell1', datetime('now'), 'weather', 'KXHIGHNYC-26MAR12-T75', 'yes', 'sell', 1, 60,
                            -0.60, 0.01, 1, 'o2', 'filled', 'exit', 0.14, datetime('now'))"""
                )
                conn.commit()
                conn.close()

                unsettled = engine.get_unsettled_trades()
                self.assertEqual(len(unsettled), 1)
                self.assertEqual(unsettled[0]["action"], "buy")

    def test_resting_exit_does_not_book_pnl_until_filled(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, market_title, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status)
                    VALUES ('entry1', datetime('now'), 'weather', 'KXHIGHNYC-26MAR12-T75', 'Test market',
                            'yes', 'buy', 2, 40, 0.80, 0.02, 1, 'entry-order', 'filled')"""
                )
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, market_title, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status)
                    VALUES ('exit1', datetime('now'), 'weather', 'KXHIGHNYC-26MAR12-T75', 'Test market',
                            'yes', 'sell', 2, 60, -1.20, 0.02, 1, 'exit-order', 'resting')"""
                )
                conn.commit()
                conn.close()

                engine.update_trade_status("exit1", "filled", filled_count=2, cost=-1.20, fee=0.02, price_cents=60)

                conn = trading_engine_module.sqlite3.connect(str(db_path))
                conn.row_factory = trading_engine_module.sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT status, result, pnl, settled_at FROM trades WHERE id = 'exit1'")
                row = dict(c.fetchone())
                c.execute("SELECT SUM(net_pnl) FROM daily_pnl")
                daily_pnl = c.fetchone()[0]
                conn.close()

                self.assertEqual(row["status"], "filled")
                self.assertEqual(row["result"], "exit")
                self.assertIsNotNone(row["settled_at"])
                self.assertAlmostEqual(row["pnl"], 0.36, places=2)
                self.assertAlmostEqual(daily_pnl, 0.36, places=2)

    def test_exit_trade_skips_when_resting_exit_already_covers_position(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, market_title, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status)
                    VALUES ('entry1', datetime('now'), 'weather', 'KXHIGHNYC-26MAR12-T75', 'Test market',
                            'yes', 'buy', 2, 40, 0.80, 0.02, 1, 'entry-order', 'filled')"""
                )
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, market_title, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status)
                    VALUES ('exit_resting', datetime('now'), 'weather', 'KXHIGHNYC-26MAR12-T75', 'Test market',
                            'yes', 'sell', 2, 60, -1.20, 0.02, 1, 'exit-order', 'resting')"""
                )
                conn.commit()
                conn.close()

                result = asyncio.run(
                    engine.exit_trade(
                        strategy="weather",
                        ticker="KXHIGHNYC-26MAR12-T75",
                        side="yes",
                        count=2,
                        price_cents=61,
                        reason="duplicate check",
                    )
                )

                self.assertEqual(result["status"], "skipped")
                self.assertIn("fully exited", result["reason"])

    def test_settlement_books_daily_pnl_on_settlement_date(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status)
                    VALUES ('buy_old', '2026-03-01T12:00:00+00:00', 'weather', 'KXHIGHNYC-26MAR12-T75',
                            'yes', 'buy', 1, 40, 0.40, 0.01, 1, 'o1', 'filled')"""
                )
                conn.commit()
                conn.close()

                engine.settle_trade("buy_old", "yes")

                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute("SELECT date, net_pnl FROM daily_pnl")
                rows = c.fetchall()
                conn.close()

                self.assertEqual(len(rows), 1)
                self.assertNotEqual(rows[0][0], "2026-03-01")
                self.assertAlmostEqual(rows[0][1], 0.59, places=2)

    def test_resting_exit_fill_books_daily_pnl_on_fill_date(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, market_title, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status)
                    VALUES ('entry_old', '2026-03-01T12:00:00+00:00', 'weather', 'KXHIGHNYC-26MAR12-T75', 'Test market',
                            'yes', 'buy', 2, 40, 0.80, 0.02, 1, 'entry-order', 'filled')"""
                )
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, market_title, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status)
                    VALUES ('exit_old', '2026-03-01T12:05:00+00:00', 'weather', 'KXHIGHNYC-26MAR12-T75', 'Test market',
                            'yes', 'sell', 2, 60, -1.20, 0.02, 1, 'exit-order', 'resting')"""
                )
                conn.commit()
                conn.close()

                engine.update_trade_status("exit_old", "filled", filled_count=2, cost=-1.20, fee=0.02, price_cents=60)

                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute("SELECT date, net_pnl FROM daily_pnl")
                rows = c.fetchall()
                conn.close()

                self.assertEqual(len(rows), 1)
                self.assertNotEqual(rows[0][0], "2026-03-01")
                self.assertAlmostEqual(rows[0][1], 0.36, places=2)

    def test_performance_summary_ignores_unfilled_sell_exits(self) -> None:
        from app.services import trading_engine as trading_engine_module

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            with patch.object(trading_engine_module, "DB_PATH", db_path):
                engine = trading_engine_module.TradingEngine(bankroll=2000.0, paper_mode=True)
                conn = trading_engine_module.sqlite3.connect(str(db_path))
                c = conn.cursor()
                c.execute(
                    """INSERT INTO trades
                    (id, timestamp, strategy, ticker, side, action, count, price_cents,
                     cost, fee, paper_mode, order_id, status, pnl)
                    VALUES ('sell_resting', datetime('now'), 'weather', 'KXHIGHNYC-26MAR12-T75',
                            'yes', 'sell', 1, 60, 0.60, 0.01, 1, 'o2', 'resting', 0.25)"""
                )
                conn.commit()
                conn.close()

                summary = engine.get_performance_summary()

                self.assertEqual(summary["overall"]["total_trades"], 0)
                self.assertAlmostEqual(summary["overall"]["total_pnl"], 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
