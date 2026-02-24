from __future__ import annotations

import sqlite3

from fastapi import APIRouter

from app.services.kalshi_agent import get_kalshi_agent
from app.services.trading_engine import DB_PATH, get_trading_engine

router = APIRouter()


@router.get("/health")
async def health_check():
    engine = get_trading_engine()
    agent = get_kalshi_agent()

    db_ok = True
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT 1")
        c.fetchone()
        conn.close()
    except Exception:
        db_ok = False

    ws_status = agent.ws.get_status()
    guardrails = engine.get_guardrail_status()

    healthy = db_ok and guardrails["runtime_health"].get("api_healthy", True)
    return {
        "status": "healthy" if healthy else "degraded",
        "service": "propsai-backend",
        "version": "0.1.0",
        "agent_running": agent.get_status().get("running", False),
        "paper_mode": engine.paper_mode,
        "db_healthy": db_ok,
        "websocket": ws_status,
        "guardrails": guardrails,
    }
