from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class LiveReadinessTests(unittest.TestCase):
    def test_live_readiness_blocks_non_weather_policy_and_ws_requirement(self) -> None:
        from app.services.live_readiness import evaluate_readiness

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            db_path.write_text("")
            engine = type(
                "EngineStub",
                (),
                {
                    "paper_mode": False,
                    "kill_switch": False,
                    "bankroll": 250.0,
                    "min_bankroll_to_trade": 40.0,
                    "allowed_live_strategies": {"weather", "sports"},
                    "allowed_paper_strategies": {"weather"},
                    "require_ws_for_live": True,
                    "get_guardrail_status": lambda self: {
                        "runtime_health": {
                            "api_healthy": True,
                            "db_healthy": True,
                            "ws_healthy": False,
                        }
                    },
                    "get_resting_trades": lambda self: [],
                },
            )()
            agent = type(
                "AgentStub",
                (),
                {"ws": type("WsStub", (), {"get_status": lambda self: {"connected": False}})()},
            )()
            readiness = evaluate_readiness(
                engine=engine,
                agent=agent,
                environ={
                    "SUPABASE_URL": "x",
                    "SUPABASE_KEY": "y",
                    "SPORTSDATAIO_API_KEY": "z",
                    "KALSHI_API_KEY_ID": "kid",
                    "KALSHI_PRIVATE_KEY_PATH": str(Path(tmpdir) / "missing.key"),
                },
                db_path=db_path,
            )

        self.assertFalse(readiness["live"]["ready"])
        blockers = " | ".join(readiness["live"]["blockers"])
        self.assertIn("weather-only", blockers)
        self.assertIn("WebSocket required for live but disconnected", blockers)

    def test_paper_readiness_passes_when_core_checks_are_green(self) -> None:
        from app.services.live_readiness import evaluate_readiness

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "trading_engine.db"
            import sqlite3

            conn = sqlite3.connect(str(db_path))
            c = conn.cursor()
            c.execute("CREATE TABLE trades (id TEXT)")
            c.execute("CREATE TABLE agent_log (id INTEGER)")
            conn.commit()
            conn.close()

            engine = type(
                "EngineStub",
                (),
                {
                    "paper_mode": True,
                    "kill_switch": False,
                    "bankroll": 250.0,
                    "min_bankroll_to_trade": 40.0,
                    "allowed_live_strategies": {"weather"},
                    "allowed_paper_strategies": {"weather"},
                    "require_ws_for_live": False,
                    "get_guardrail_status": lambda self: {
                        "runtime_health": {
                            "api_healthy": True,
                            "db_healthy": True,
                            "ws_healthy": True,
                        }
                    },
                    "get_resting_trades": lambda self: [],
                },
            )()
            readiness = evaluate_readiness(
                engine=engine,
                agent=None,
                environ={
                    "SUPABASE_URL": "x",
                    "SUPABASE_KEY": "y",
                    "SPORTSDATAIO_API_KEY": "z",
                },
                db_path=db_path,
            )

        self.assertTrue(readiness["paper"]["ready"])
        self.assertEqual(readiness["paper"]["recommended_start_endpoint"], "/api/kalshi/agent/start")


if __name__ == "__main__":
    unittest.main()
