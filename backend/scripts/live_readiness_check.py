from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.live_readiness import evaluate_readiness
from app.services.trading_engine import get_trading_engine


ENV_PATH = ROOT / ".env"


def _load_env_file() -> None:
    if not ENV_PATH.exists():
        return

    for raw_line in ENV_PATH.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _print_target(title: str, payload: dict) -> None:
    icon = "PASS" if payload.get("ready") else "BLOCKED"
    print(f"\n{title}: {icon}")
    print(f"  Start endpoint: {payload.get('recommended_start_endpoint')}")
    for check in payload.get("checks", []):
        marker = "ok" if check.get("ok") else "x"
        print(f"  [{marker}] {check.get('name')}: {check.get('detail')}")
    if payload.get("warnings"):
        print("  Warnings:")
        for warning in payload["warnings"]:
            print(f"    - {warning}")
    if payload.get("blockers"):
        print("  Blockers:")
        for blocker in payload["blockers"]:
            print(f"    - {blocker}")


def main() -> int:
    _load_env_file()
    engine = get_trading_engine()
    readiness = evaluate_readiness(engine=engine, agent=None)

    print("PropsAI readiness check")
    print(f"Current mode: {readiness['current_mode']}")
    print(
        "Strategy policy: "
        f"live={','.join(readiness['strategy_policy']['allowed_live_strategies']) or 'none'} "
        f"paper={','.join(readiness['strategy_policy']['allowed_paper_strategies']) or 'none'}"
    )

    _print_target("Paper readiness", readiness["paper"])
    _print_target("Live readiness", readiness["live"])

    if readiness["live"]["ready"]:
        print("\nPASS: live weather-only readiness checks succeeded")
        return 0

    print("\nFAILED: live readiness is blocked")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
