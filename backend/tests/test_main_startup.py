from __future__ import annotations

import asyncio
import importlib
import os
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


class MainStartupTests(unittest.TestCase):
    def test_auto_start_refuses_non_weather_live_policy(self) -> None:
        settings = types.SimpleNamespace(
            app_debug=False,
            app_env="test",
            sentry_dsn="",
            cors_origin_list=["http://localhost:3000"],
        )
        sys.modules["app.config"] = types.SimpleNamespace(get_settings=lambda: settings)
        sys.modules.pop("app.main", None)
        main_module = importlib.import_module("app.main")

        engine = types.SimpleNamespace(
            paper_mode=False,
            allowed_live_strategies={"weather", "sports"},
            allowed_paper_strategies={"weather"},
        )
        agent = types.SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), _running=False)

        async def _run() -> None:
            with patch.dict(os.environ, {"AGENT_AUTO_START": "true"}, clear=False):
                with patch.object(main_module, "setup_logging", return_value=None):
                    with patch("app.services.trading_engine.get_trading_engine", return_value=engine):
                        with patch("app.api.agent.get_kalshi_agent", return_value=agent):
                            async with main_module.lifespan(object()):
                                pass

        asyncio.run(_run())
        agent.start.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
