from __future__ import annotations
from contextlib import asynccontextmanager

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logging_config import get_logger, setup_logging
from app.api import health, predictions, players, odds, bets, data, settings as settings_router, kalshi, agent

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_settings()
    setup_logging(debug=config.app_debug)
    logger.info("Starting PropsAI backend", env=config.app_env)

    if config.sentry_dsn:
        sentry_sdk.init(
            dsn=config.sentry_dsn,
            traces_sample_rate=1.0 if config.app_debug else 0.2,
            environment=config.app_env,
        )
        logger.info("Sentry initialized")

    yield

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
