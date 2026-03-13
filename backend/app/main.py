from __future__ import annotations

import os
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import agent, bets, data, health, kalshi, odds, players, predictions
from app.api import settings as settings_router
from app.config import get_settings
from app.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_settings()
    engine = None
    setup_logging(debug=config.app_debug)
    logger.info("Starting PropsAI backend", env=config.app_env)

    if config.sentry_dsn:
        sentry_sdk.init(
            dsn=config.sentry_dsn,
            traces_sample_rate=1.0 if config.app_debug else 0.2,
            environment=config.app_env,
        )
        logger.info("Sentry initialized")

    auto_start = os.environ.get("AGENT_AUTO_START", "false").lower() == "true"
    if auto_start:
        try:
            from app.services.trading_engine import get_trading_engine
            from app.api.agent import get_kalshi_agent

            engine = get_trading_engine()
            live_weather_only = engine.allowed_live_strategies == {"weather"}
            logger.info(
                "Auto-start requested",
                paper_mode=engine.paper_mode,
                allowed_live_strategies=sorted(engine.allowed_live_strategies),
                allowed_paper_strategies=sorted(engine.allowed_paper_strategies),
            )
            if not engine.paper_mode and not live_weather_only:
                raise RuntimeError(
                    "Refusing AGENT_AUTO_START in live mode unless LIVE_ENABLED_STRATEGIES=weather"
                )

            agent = get_kalshi_agent()
            await agent.start()
            logger.info("Kalshi agent auto-started")
        except Exception as e:
            logger.error("Failed to auto-start agent", error=str(e))
    else:
        logger.info("Kalshi agent auto-start DISABLED - must be started manually")

    yield

    # Gracefully stop agent on shutdown
    try:
        from app.api.agent import get_kalshi_agent

        agent = get_kalshi_agent()
        if getattr(agent, "_running", False):
            await agent.stop()
            logger.info("Kalshi agent stopped")
    except Exception:
        pass

    logger.info("Shutting down PropsAI backend")


def create_app() -> FastAPI:
    config = get_settings()

    app = FastAPI(
        title="PropsAI",
        description="NBA player props prediction engine",
        version="0.1.0",
        lifespan=lifespan,
        redirect_slashes=True,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
    app.include_router(players.router, prefix="/api/players", tags=["Players"])
    app.include_router(odds.router, prefix="/api/odds", tags=["Odds"])
    app.include_router(bets.router, prefix="/api/bets", tags=["Bets"])
    app.include_router(data.router, prefix="/api/data", tags=["Data"])
    app.include_router(settings_router.router, prefix="/api/settings", tags=["Settings"])
    app.include_router(kalshi.router, prefix="/api/kalshi", tags=["Kalshi"])
    app.include_router(agent.router, prefix="/api/kalshi/agent", tags=["Agent"])

    return app


app = create_app()
