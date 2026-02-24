from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "app" / "data" / "trading_engine.db"


def _check_env() -> list[str]:
    errors: list[str] = []
    required = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SPORTSDATAIO_API_KEY",
    ]
    for key in required:
        if not os.environ.get(key):
            errors.append(f"missing env: {key}")

    if os.environ.get("PAPER_MODE", "true").lower() == "false":
        live_required = ["KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH"]
        for key in live_required:
            if not os.environ.get(key):
                errors.append(f"missing live env: {key}")

    return errors


def _check_db() -> list[str]:
    errors: list[str] = []
    if not DB_PATH.exists():
        errors.append(f"db missing: {DB_PATH}")
        return errors

    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
        if not c.fetchone():
            errors.append("db table missing: trades")
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_log'")
        if not c.fetchone():
            errors.append("db table missing: agent_log")
        conn.close()
    except Exception as e:
        errors.append(f"db check failed: {e}")

    return errors


def _print_config_summary() -> None:
    keys = [
        "PAPER_MODE",
        "BANKROLL",
        "ENABLE_DYNAMIC_REPRICING",
        "MAX_TOTAL_RESTING_ORDERS",
        "MAX_RESTING_ORDERS_PER_STRATEGY",
        "MAX_ORDER_FAILURES_WINDOW",
        "ORDER_FAILURE_WINDOW_MINS",
        "REQUIRE_WS_FOR_LIVE",
        "AUTO_KILL_ON_ORDER_FAILURES",
        "CANCEL_ALERT_WINDOW_MINS",
        "CANCEL_ALERT_THRESHOLD",
        "AUTO_KILL_ON_CANCEL_STORM",
    ]
    print("\nConfig summary:")
    for key in keys:
        print(f"  {key}={os.environ.get(key, '<unset>')}")


def main() -> int:
    print("Live readiness check")
    env_errors = _check_env()
    db_errors = _check_db()

    _print_config_summary()

    all_errors = env_errors + db_errors
    if all_errors:
        print("\nFAILED:")
        for err in all_errors:
            print(f"  - {err}")
        return 1

    print("\nPASS: core readiness checks succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
