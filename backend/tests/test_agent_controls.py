from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest.mock import AsyncMock, patch


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
sys.modules.setdefault(
    "eval_type_backport",
    types.SimpleNamespace(
        eval_type_backport=lambda value, globalns=None, localns=None, **_kwargs: value
    ),
)


class AgentControlRouteTests(unittest.TestCase):
    def test_generic_start_rejects_live_mode(self) -> None:
        from fastapi import HTTPException
        from app.api import agent as agent_api

        engine = types.SimpleNamespace(paper_mode=False)

        with patch.object(agent_api, "get_trading_engine", return_value=engine):
            with self.assertRaises(HTTPException) as exc:
                asyncio.run(agent_api.start_agent())

        self.assertEqual(exc.exception.status_code, 409)
        self.assertIn("paper-only", str(exc.exception.detail))

    def test_live_weather_start_rejects_non_weather_policy(self) -> None:
        from fastapi import HTTPException
        from app.api import agent as agent_api

        engine = types.SimpleNamespace(
            allowed_live_strategies={"weather", "sports"},
            set_paper_mode=lambda *_a, **_k: None,
            sync_bankroll=lambda *_a, **_k: None,
            bankroll=200.0,
        )
        agent = types.SimpleNamespace(start=AsyncMock(), engine=engine)

        with patch.object(agent_api, "get_trading_engine", return_value=engine):
            with patch.object(agent_api, "get_kalshi_agent", return_value=agent):
                with self.assertRaises(HTTPException) as exc:
                    asyncio.run(agent_api.start_live_weather_agent())

        self.assertEqual(exc.exception.status_code, 409)
        self.assertIn("weather-only", str(exc.exception.detail))

    def test_cancel_resting_orders_skips_in_paper_mode(self) -> None:
        from app.api import agent as agent_api

        engine = types.SimpleNamespace(paper_mode=True)
        with patch.object(agent_api, "get_trading_engine", return_value=engine):
            payload = asyncio.run(agent_api.cancel_resting_orders())

        self.assertEqual(payload["status"], "skipped")
        self.assertEqual(payload["reason"], "paper_mode")


if __name__ == "__main__":
    unittest.main()
