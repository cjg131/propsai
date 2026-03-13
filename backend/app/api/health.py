from __future__ import annotations

import sqlite3

from fastapi import APIRouter

from app.services.kalshi_agent import get_existing_kalshi_agent, get_kalshi_agent
from app.services.trading_engine import DB_PATH, get_trading_engine

router = APIRouter()


@router.get("/health")
async def health_check():
    engine = get_trading_engine()
    agent = get_existing_kalshi_agent()
    if agent is None:
        try:
            agent = get_kalshi_agent()
        except Exception:
            agent = None

    db_ok = True
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT 1")
        c.fetchone()
        conn.close()
    except Exception:
        db_ok = False

    ws_status = agent.ws.get_status() if agent is not None else {"connected": False, "status": "unavailable"}
    guardrails = engine.get_guardrail_status()
    runtime_health = guardrails.get("runtime_health", {})
    api_ok = bool(runtime_health.get("api_healthy", True))
    runtime_db_ok = bool(runtime_health.get("db_healthy", True))
    ws_required = (not engine.paper_mode) and engine.require_ws_for_live
    ws_ok = bool(runtime_health.get("ws_healthy", True)) if ws_required else True
    live_weather_only = engine.allowed_live_strategies == {"weather"}

    healthy = db_ok and runtime_db_ok and api_ok and ws_ok
    return {
        "status": "healthy" if healthy else "degraded",
        "service": "propsai-backend",
        "version": "0.1.0",
        "agent_running": bool(getattr(agent, "_running", False)),
        "paper_mode": engine.paper_mode,
        "mode": "paper" if engine.paper_mode else "live",
        "strategy_policy": {
            "allowed_live_strategies": sorted(engine.allowed_live_strategies),
            "allowed_paper_strategies": sorted(engine.allowed_paper_strategies),
            "live_weather_only": live_weather_only,
        },
        "db_healthy": db_ok,
        "websocket": ws_status,
        "guardrails": guardrails,
    }
