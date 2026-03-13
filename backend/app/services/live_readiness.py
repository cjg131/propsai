from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

from app.services.trading_engine import DB_PATH


def _bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _check_db(db_path: Path) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not db_path.exists():
        return False, [f"SQLite DB missing at {db_path}"]

    try:
        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()
        for table in ("trades", "agent_log"):
            c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if not c.fetchone():
                failures.append(f"DB table missing: {table}")
        conn.close()
    except Exception as exc:
        failures.append(f"DB check failed: {exc}")

    return not failures, failures


def _base_env_failures(environ: dict[str, str]) -> list[str]:
    failures: list[str] = []
    required = ("SUPABASE_URL", "SUPABASE_KEY", "SPORTSDATAIO_API_KEY")
    for key in required:
        if not environ.get(key):
            failures.append(f"Missing env: {key}")
    return failures


def _live_env_failures(environ: dict[str, str]) -> list[str]:
    failures: list[str] = []
    required = ("KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH")
    for key in required:
        if not environ.get(key):
            failures.append(f"Missing live env: {key}")

    private_key = environ.get("KALSHI_PRIVATE_KEY_PATH")
    if private_key and not Path(private_key).expanduser().exists():
        failures.append(f"Kalshi private key file not found: {private_key}")
    return failures


def _build_target_readiness(
    *,
    target_mode: str,
    engine: Any,
    agent: Any | None,
    environ: dict[str, str],
    db_ok: bool,
    db_failures: list[str],
) -> dict[str, Any]:
    is_live = target_mode == "live"
    blockers: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    def add_check(name: str, ok: bool, detail: str, *, blocker: bool = False) -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})
        if ok:
            return
        if blocker:
            blockers.append(detail)
        else:
            warnings.append(detail)

    for detail in _base_env_failures(environ):
        add_check("env", False, detail, blocker=True)
    if not any(c["name"] == "env" for c in checks):
        add_check("env", True, "Core env vars present")

    add_check("db", db_ok, "Trading DB ready" if db_ok else "; ".join(db_failures), blocker=True)

    if is_live:
        live_failures = _live_env_failures(environ)
        for detail in live_failures:
            add_check("live_env", False, detail, blocker=True)
        if not live_failures:
            add_check("live_env", True, "Kalshi live credentials present")

        live_weather_only = set(getattr(engine, "allowed_live_strategies", set())) == {"weather"}
        add_check(
            "strategy_policy",
            live_weather_only,
            "LIVE_ENABLED_STRATEGIES is weather-only"
            if live_weather_only
            else "Live strategy policy must be exactly weather-only",
            blocker=True,
        )

    if getattr(engine, "kill_switch", False):
        add_check("kill_switch", False, "Kill switch is active", blocker=True)
    else:
        add_check("kill_switch", True, "Kill switch is off")

    bankroll = float(getattr(engine, "bankroll", 0.0) or 0.0)
    min_bankroll = float(getattr(engine, "min_bankroll_to_trade", 0.0) or 0.0)
    bankroll_ok = bankroll >= min_bankroll if is_live else bankroll > 0
    bankroll_detail = (
        f"Bankroll ${bankroll:.2f} meets minimum ${min_bankroll:.2f}"
        if bankroll_ok
        else f"Bankroll ${bankroll:.2f} below minimum ${min_bankroll:.2f}"
    )
    add_check("bankroll", bankroll_ok, bankroll_detail, blocker=is_live)

    guardrails = getattr(engine, "get_guardrail_status", lambda: {})() or {}
    runtime_health = guardrails.get("runtime_health", {})
    add_check(
        "runtime_api",
        bool(runtime_health.get("api_healthy", True)),
        "API health is green" if runtime_health.get("api_healthy", True) else "API health is degraded",
        blocker=is_live,
    )
    add_check(
        "runtime_db",
        bool(runtime_health.get("db_healthy", True)),
        "Runtime DB health is green" if runtime_health.get("db_healthy", True) else "Runtime DB health is degraded",
        blocker=is_live,
    )

    ws_required = bool(getattr(engine, "require_ws_for_live", False)) if is_live else False
    ws_status = {}
    if agent is not None:
        try:
            ws_status = agent.ws.get_status()
        except Exception:
            ws_status = {}
    ws_connected = bool(ws_status.get("connected", runtime_health.get("ws_healthy", True)))
    if ws_required:
        add_check(
            "websocket",
            ws_connected,
            "WebSocket connected" if ws_connected else "WebSocket required for live but disconnected",
            blocker=True,
        )
    else:
        add_check("websocket", True, "WebSocket optional for current target mode")

    resting_total = 0
    get_resting_trades = getattr(engine, "get_resting_trades", None)
    if callable(get_resting_trades):
        try:
            resting_total = len(get_resting_trades())
        except Exception:
            resting_total = 0
    if is_live and resting_total > 0:
        add_check("resting_orders", False, f"{resting_total} resting orders still recorded", blocker=True)
    else:
        add_check("resting_orders", True, "No resting orders blocking start")

    target_ready = not blockers
    return {
        "target_mode": target_mode,
        "ready": target_ready,
        "status": "ready" if target_ready else "blocked",
        "recommended_start_endpoint": (
            "/api/kalshi/agent/start-live-weather" if is_live else "/api/kalshi/agent/start"
        ),
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
    }


def evaluate_readiness(
    *,
    engine: Any,
    agent: Any | None = None,
    environ: dict[str, str] | None = None,
    db_path: Path = DB_PATH,
) -> dict[str, Any]:
    current_environ = dict(environ or os.environ)
    db_ok, db_failures = _check_db(db_path)
    current_mode = "paper" if getattr(engine, "paper_mode", True) else "live"

    paper = _build_target_readiness(
        target_mode="paper",
        engine=engine,
        agent=agent,
        environ=current_environ,
        db_ok=db_ok,
        db_failures=db_failures,
    )
    live = _build_target_readiness(
        target_mode="live",
        engine=engine,
        agent=agent,
        environ=current_environ,
        db_ok=db_ok,
        db_failures=db_failures,
    )
    current = paper if current_mode == "paper" else live

    return {
        "current_mode": current_mode,
        "current": current,
        "paper": paper,
        "live": live,
        "strategy_policy": {
            "allowed_live_strategies": sorted(getattr(engine, "allowed_live_strategies", set())),
            "allowed_paper_strategies": sorted(getattr(engine, "allowed_paper_strategies", set())),
            "live_weather_only": set(getattr(engine, "allowed_live_strategies", set())) == {"weather"},
        },
        "summary": {
            "db_ok": db_ok,
            "db_failures": db_failures,
            "live_ready": live["ready"],
            "paper_ready": paper["ready"],
        },
    }
