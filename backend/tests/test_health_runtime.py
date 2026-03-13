from __future__ import annotations

import asyncio
import sqlite3
import unittest
from unittest.mock import patch


class HealthRuntimeTests(unittest.TestCase):
    def test_health_check_reports_strategy_policy_and_degraded_live_ws(self) -> None:
        from app.api import health as health_module

        engine = type(
            "EngineStub",
            (),
            {
                "paper_mode": False,
                "require_ws_for_live": True,
                "allowed_live_strategies": {"weather"},
                "allowed_paper_strategies": {"weather"},
                "get_guardrail_status": lambda self: {
                    "runtime_health": {
                        "api_healthy": True,
                        "db_healthy": True,
                        "ws_healthy": False,
                        "last_monitor_heartbeat": "",
                    }
                },
            },
        )()
        agent = type(
            "AgentStub",
            (),
            {
                "_running": True,
                "ws": type("WsStub", (), {"get_status": lambda self: {"connected": False}})(),
            },
        )()

        with patch.object(health_module, "get_trading_engine", return_value=engine):
            with patch.object(health_module, "get_kalshi_agent", return_value=agent):
                with patch.object(health_module.sqlite3, "connect", side_effect=sqlite3.OperationalError("db down")):
                    payload = asyncio.run(health_module.health_check())

        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["mode"], "live")
        self.assertTrue(payload["strategy_policy"]["live_weather_only"])
        self.assertEqual(payload["strategy_policy"]["allowed_live_strategies"], ["weather"])


if __name__ == "__main__":
    unittest.main()
