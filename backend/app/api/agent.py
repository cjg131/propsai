"""
API endpoints for the Kalshi autonomous trading agent.
Provides status, control, signals, trades, and performance data.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.logging_config import get_logger
from app.services.kalshi_agent import get_kalshi_agent
from app.services.trading_engine import get_trading_engine

logger = get_logger(__name__)

router = APIRouter()


# ── Request models ─────────────────────────────────────────────


class ToggleRequest(BaseModel):
    enabled: bool


class KillSwitchRequest(BaseModel):
    active: bool


class PaperModeRequest(BaseModel):
    enabled: bool


# ── Status & Control ───────────────────────────────────────────


@router.get("/status")
async def get_agent_status():
    """Get current agent status including paper mode, kill switch, bankroll."""
    try:
        agent = get_kalshi_agent()
        return agent.get_status()
    except Exception as e:
        logger.error("Failed to get agent status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_agent_performance():
    """Get agent performance summary with P&L, win rate, etc."""
    try:
        engine = get_trading_engine()
        return engine.get_performance_summary()
    except Exception as e:
        logger.error("Failed to get performance", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kill-switch")
async def set_kill_switch(req: KillSwitchRequest):
    """Activate or deactivate the kill switch."""
    engine = get_trading_engine()
    engine.kill_switch = req.active
    engine.log_event(
        "control",
        f"Kill switch {'ACTIVATED' if req.active else 'deactivated'}",
    )
    return {"kill_switch": engine.kill_switch}


@router.post("/paper-mode")
async def set_paper_mode(req: PaperModeRequest):
    """Toggle paper trading mode."""
    engine = get_trading_engine()
    engine.paper_mode = req.enabled
    engine.log_event(
        "control",
        f"Paper mode {'enabled' if req.enabled else 'DISABLED — LIVE TRADING'}",
    )
    return {"paper_mode": engine.paper_mode}


@router.post("/strategy/{strategy}/toggle")
async def toggle_strategy(strategy: str, req: ToggleRequest):
    """Enable or disable a specific strategy."""
    engine = get_trading_engine()
    if strategy not in engine.strategy_enabled:
        raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy}")
    engine.strategy_enabled[strategy] = req.enabled
    engine.log_event(
        "control",
        f"Strategy '{strategy}' {'enabled' if req.enabled else 'disabled'}",
        strategy=strategy,
    )
    return {"strategy": strategy, "enabled": req.enabled}


# ── Agent Start/Stop ───────────────────────────────────────────


@router.post("/start")
async def start_agent():
    """Start the autonomous agent loops."""
    try:
        agent = get_kalshi_agent()
        await agent.start()
        return {"status": "started", "paper_mode": agent.engine.paper_mode}
    except Exception as e:
        logger.error("Failed to start agent", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_agent():
    """Stop the autonomous agent loops."""
    try:
        agent = get_kalshi_agent()
        await agent.stop()
        return {"status": "stopped"}
    except Exception as e:
        logger.error("Failed to stop agent", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Manual Triggers ────────────────────────────────────────────


@router.post("/run/weather")
async def run_weather_cycle():
    """Manually trigger one weather strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_weather_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("Weather cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/sports")
async def run_sports_cycle():
    """Manually trigger one sports strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_sports_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("Sports cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/crypto")
async def run_crypto_cycle():
    """Manually trigger one crypto strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_crypto_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("Crypto cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/finance")
async def run_finance_cycle():
    """Manually trigger one finance strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_finance_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("Finance cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/econ")
async def run_econ_cycle():
    """Manually trigger one econ strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_econ_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("Econ cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/nba-props")
async def run_nba_props_cycle():
    """Manually trigger one NBA props strategy cycle."""
    try:
        agent = get_kalshi_agent()
        results = await agent.run_nba_props_cycle()
        return {"signals": len(results), "results": results}
    except Exception as e:
        logger.error("NBA props cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── Data Queries ───────────────────────────────────────────────


@router.get("/positions")
async def get_positions():
    """Get open positions with live market data and unrealized P&L."""
    try:
        agent = get_kalshi_agent()
        positions = await agent.get_positions_with_market_data()
        total_cost = sum(p["total_cost"] for p in positions)
        total_unrealized = sum(p.get("unrealized_pnl") or 0 for p in positions)
        total_max_risk = sum(p["max_risk"] for p in positions)
        return {
            "positions": positions,
            "total": len(positions),
            "total_cost": round(total_cost, 2),
            "total_unrealized_pnl": round(total_unrealized, 2),
            "total_max_risk": round(total_max_risk, 2),
        }
    except Exception as e:
        logger.error("Failed to get positions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/monitor")
async def run_monitor_cycle():
    """Manually trigger one position monitor cycle."""
    try:
        agent = get_kalshi_agent()
        actions = await agent.run_monitor_cycle()
        return {"actions": len(actions), "results": actions}
    except Exception as e:
        logger.error("Monitor cycle failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades")
async def get_trades(
    strategy: str | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get trade history."""
    engine = get_trading_engine()
    trades = engine.get_trades(strategy=strategy, status=status, limit=limit)
    return {"trades": trades, "total": len(trades)}


@router.get("/signals")
async def get_signals(
    strategy: str | None = Query(None),
    acted_on: bool | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """Get trading signals."""
    engine = get_trading_engine()
    signals = engine.get_signals(strategy=strategy, acted_on=acted_on, limit=limit)
    return {"signals": signals, "total": len(signals)}


@router.get("/log")
async def get_agent_log(
    strategy: str | None = Query(None),
    limit: int = Query(200, ge=1, le=1000),
):
    """Get agent activity log."""
    engine = get_trading_engine()
    log = engine.get_agent_log(limit=limit, strategy=strategy)
    return {"log": log, "total": len(log)}


# ── New Endpoints (OpenClaw, Adaptive Thresholds, Signal Stats, Reset) ──


@router.get("/reviews")
async def get_trade_reviews(
    strategy: str | None = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """Get GPT-powered trade reviews from OpenClaw."""
    from app.services.trade_analyzer import get_trade_analyzer
    analyzer = get_trade_analyzer()
    reviews = analyzer.get_recent_reviews(strategy=strategy, limit=limit)
    patterns = analyzer.get_pattern_summary()
    return {"reviews": reviews, "patterns": patterns}


@router.get("/thresholds")
async def get_adaptive_thresholds():
    """Get current adaptive thresholds per strategy."""
    from app.services.adaptive_thresholds import get_adaptive_thresholds as _get
    thresholds = _get()
    return thresholds.get_all_thresholds()


@router.get("/signal-stats")
async def get_signal_stats():
    """Get signal component quality stats and dynamic weights."""
    from app.services.signal_scorer import get_signal_scorer
    scorer = get_signal_scorer()
    return scorer.get_all_stats()


@router.post("/reset")
async def reset_paper_trades():
    """Clear all paper trades, signals, and logs for a fresh start."""
    import sqlite3

    from app.services.trading_engine import DB_PATH
    engine = get_trading_engine()
    engine.log_event("control", "DB reset requested via API")

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("DELETE FROM trades WHERE paper_mode = 1")
    c.execute("DELETE FROM signals")
    c.execute("DELETE FROM daily_pnl")
    c.execute("DELETE FROM agent_log")
    conn.commit()
    conn.close()

    engine._first_cycle_done = False
    return {"status": "reset", "message": "Paper trades, signals, and logs cleared"}
