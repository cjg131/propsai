"""
Export authenticated Kalshi account history into clean CSV/JSON files.

This pulls from the account endpoints we already use in the app:
  - /portfolio/orders
  - /portfolio/fills
  - /portfolio/positions

It writes:
  1. a merged fill-level export with matched order metadata
  2. raw JSON snapshots for orders, fills, and positions

Usage:
    cd backend
    python3 scripts/export_kalshi_account_history.py

Optional:
    python3 scripts/export_kalshi_account_history.py --limit 500 --prefix my_export
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.services.kalshi_api import KalshiClient


EXPORT_DIR = ROOT / "app" / "data" / "kalshi_exports"


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    num = _safe_float(value)
    if num is None:
        return None
    return int(num)


def _extract_order_id(payload: dict[str, Any]) -> str:
    return str(
        payload.get("order_id")
        or payload.get("id")
        or payload.get("client_order_id")
        or ""
    )


def _extract_fill_order_id(payload: dict[str, Any]) -> str:
    return str(
        payload.get("order_id")
        or payload.get("trade_id")
        or payload.get("id")
        or ""
    )


async def _paginate(
    fetch_page,
    *,
    item_key: str,
    limit: int,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()

    while True:
        page = await fetch_page(limit=limit, cursor=cursor)
        page_items = page.get(item_key, []) if isinstance(page, dict) else []
        if not isinstance(page_items, list):
            break
        items.extend(page_items)

        next_cursor = page.get("cursor") if isinstance(page, dict) else None
        if not next_cursor or next_cursor in seen_cursors or not page_items:
            break
        seen_cursors.add(next_cursor)
        cursor = str(next_cursor)

    return items


def _build_fill_export_rows(
    fills: list[dict[str, Any]],
    orders_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for fill in fills:
        order_id = _extract_fill_order_id(fill)
        order = orders_by_id.get(order_id, {})

        fill_count = (
            _safe_int(fill.get("count"))
            or _safe_int(fill.get("fill_count"))
            or _safe_int(fill.get("filled_count"))
            or _safe_int(fill.get("quantity"))
            or 0
        )
        price_cents = (
            _safe_int(fill.get("price"))
            or _safe_int(fill.get("price_cents"))
            or _safe_int(fill.get("yes_price"))
            or _safe_int(order.get("price"))
            or _safe_int(order.get("price_cents"))
            or _safe_int(order.get("yes_price"))
        )
        side = str(fill.get("side") or order.get("side") or "")
        action = str(fill.get("action") or order.get("action") or "")
        ticker = str(fill.get("ticker") or order.get("ticker") or "")
        created_at = str(
            fill.get("created_time")
            or fill.get("created_at")
            or fill.get("ts")
            or fill.get("timestamp")
            or order.get("created_time")
            or order.get("created_at")
            or ""
        )
        fee = (
            _safe_float(fill.get("fee"))
            or _safe_float(fill.get("fee_dollars"))
            or _safe_float(fill.get("fee_cost"))
            or _safe_float(order.get("maker_fees_dollars"))
            or _safe_float(order.get("taker_fees_dollars"))
            or _safe_float(order.get("fee"))
            or 0.0
        )
        proceeds = (
            _safe_float(fill.get("cost_proceeds"))
            or _safe_float(fill.get("cost_proceeds_dollars"))
            or _safe_float(order.get("maker_fill_cost_dollars"))
            or _safe_float(order.get("taker_fill_cost_dollars"))
        )
        amount_dollars = _safe_float(fill.get("amount"))
        if amount_dollars is None and price_cents is not None and fill_count:
            amount_dollars = round((price_cents / 100.0) * fill_count, 4)

        rows.append(
            {
                "fill_id": str(fill.get("fill_id") or fill.get("id") or ""),
                "order_id": order_id,
                "ticker": ticker,
                "side": side,
                "action": action,
                "status": str(order.get("status") or ""),
                "fill_count": fill_count,
                "price_cents": price_cents if price_cents is not None else "",
                "amount_dollars": amount_dollars if amount_dollars is not None else "",
                "fee_dollars": fee,
                "cost_proceeds_dollars": proceeds if proceeds is not None else "",
                "created_at": created_at,
                "order_created_at": str(order.get("created_time") or order.get("created_at") or ""),
                "raw_fill_side": str(fill.get("side") or ""),
                "raw_fill_action": str(fill.get("action") or ""),
                "raw_order_type": str(order.get("type") or ""),
                "raw_order": json.dumps(order, separators=(",", ":")),
                "raw_fill": json.dumps(fill, separators=(",", ":")),
            }
        )

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


async def export_history(limit: int, prefix: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.kalshi_api_key_id or not settings.kalshi_private_key_path:
        raise RuntimeError(
            "Kalshi credentials are not configured. Set KALSHI_API_KEY_ID and "
            "KALSHI_PRIVATE_KEY_PATH in backend/.env first."
        )

    client = KalshiClient()
    if not client.private_key:
        raise RuntimeError(
            "Kalshi private key could not be loaded. Check KALSHI_PRIVATE_KEY_PATH "
            "and confirm the key file exists."
        )

    try:
        orders = await _paginate(
            lambda **kwargs: client.get_orders(**kwargs),
            item_key="orders",
            limit=limit,
        )
        fills = await _paginate(
            lambda **kwargs: client.get_fills(**kwargs),
            item_key="fills",
            limit=limit,
        )
        positions_resp = await client.get_positions(limit=limit)
        positions = positions_resp.get("market_positions", []) if isinstance(positions_resp, dict) else []
    finally:
        await client.close()

    orders_by_id = {
        order_id: order
        for order in orders
        if (order_id := _extract_order_id(order))
    }
    fill_rows = _build_fill_export_rows(fills, orders_by_id)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    export_base = EXPORT_DIR / f"{prefix}_{ts}"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw_json_path = export_base.with_name(f"{export_base.name}_raw.json")
    fill_csv_path = export_base.with_name(f"{export_base.name}_fills.csv")
    summary_json_path = export_base.with_name(f"{export_base.name}_summary.json")

    raw_json_path.write_text(
        json.dumps(
            {
                "exported_at": datetime.now(UTC).isoformat(),
                "orders": orders,
                "fills": fills,
                "positions": positions,
            },
            indent=2,
        )
    )
    _write_csv(fill_csv_path, fill_rows)

    summary = {
        "exported_at": datetime.now(UTC).isoformat(),
        "orders": len(orders),
        "fills": len(fills),
        "positions": len(positions),
        "csv_path": str(fill_csv_path),
        "raw_json_path": str(raw_json_path),
        "sample_order_fields": sorted({k for order in orders[:5] for k in order.keys()}),
        "sample_fill_fields": sorted({k for fill in fills[:5] for k in fill.keys()}),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export authenticated Kalshi account history.")
    parser.add_argument("--limit", type=int, default=500, help="Page size for Kalshi pagination.")
    parser.add_argument(
        "--prefix",
        default="kalshi_account_history",
        help="Filename prefix for generated export files.",
    )
    args = parser.parse_args()

    summary = asyncio.run(export_history(limit=args.limit, prefix=args.prefix))
    print("Kalshi account export complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
