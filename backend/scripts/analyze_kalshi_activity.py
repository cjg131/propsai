"""
Analyze Kalshi account activity exports for concentration, churn, and realized P&L.

The script is designed to work with the "Recent Activity" CSV export you already
have, while also taking advantage of richer exports if they include populated
price, amount, fee, direction, or profit columns.

Usage:
    python3 backend/scripts/analyze_kalshi_activity.py \
        --csv "Trade History/Kalshi-Recent-Activity-All.csv"

Optional:
    --fetch
    --top 15
    --json /tmp/kalshi_activity_report.json
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.performance_model import build_performance_model


ISO_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
)


def parse_datetime(raw: str) -> datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    for fmt in ISO_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def parse_float(raw: str) -> float | None:
    raw = (raw or "").strip().replace("$", "").replace(",", "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_int(raw: str) -> int:
    value = parse_float(raw)
    return int(value or 0)


def market_family(ticker: str) -> str:
    ticker = ticker.upper()
    if ticker.startswith("KXHIGH") or ticker.startswith("KXLOW"):
        return "weather"
    if "SPORTSMULTIGAME" in ticker or "MULTIGAMEEXTENDED" in ticker:
        return "sports_multigame"
    if any(tag in ticker for tag in ("GAME", "MATCH", "TOTAL")):
        return "sports_single"
    if any(tag in ticker for tag in ("BTC", "ETH", "CRYPTO")):
        return "crypto"
    if any(tag in ticker for tag in ("NASDAQ", "INX", "SPX", "DJIA", "RATE", "CPI", "FED")):
        return "macro"
    return "other"


def event_key(ticker: str) -> str:
    return ticker.rsplit("-", 1)[0] if "-" in ticker else ticker


@dataclass
class TickerSummary:
    ticker: str
    family: str
    fills: int = 0
    canceled: int = 0
    partial_cancel: int = 0
    contracts: int = 0
    amount_dollars: float = 0.0
    fees_dollars: float = 0.0
    profit_dollars: float = 0.0
    realized_pnl_dollars: float = 0.0
    first_seen: str = ""
    last_seen: str = ""
    best_15m_fill_burst: int = 0


@dataclass
class ClosedTradeSummary:
    ticker: str
    side: str
    contracts_closed: int
    realized_pnl_dollars: float
    round_trips: int


def detect_csv_shape(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "unknown"
    keys = set(rows[0].keys())
    if "Market_Ticker" in keys and "Status" in keys:
        return "recent_activity"
    if "ticker" in keys and "fill_count" in keys:
        return "fill_export"
    return "unknown"


def _row_ticker(row: dict[str, str], shape: str) -> str:
    return (row.get("Market_Ticker") if shape == "recent_activity" else row.get("ticker") or "").strip()


def _row_status(row: dict[str, str], shape: str) -> str:
    if shape == "recent_activity":
        return (row.get("Status") or "").strip()
    return (row.get("status") or "filled").strip() or "filled"


def _row_datetime(row: dict[str, str], shape: str) -> datetime | None:
    if shape == "recent_activity":
        return parse_datetime(row.get("Original_Date", "")) or parse_datetime(row.get("Last_Updated", ""))
    return parse_datetime(row.get("created_at", "")) or parse_datetime(row.get("order_created_at", ""))


def _row_filled_contracts(row: dict[str, str], shape: str) -> int:
    return parse_int(row.get("Filled", "")) if shape == "recent_activity" else parse_int(row.get("fill_count", ""))


def _row_amount(row: dict[str, str], shape: str) -> float:
    key = "Amount_In_Dollars" if shape == "recent_activity" else "amount_dollars"
    return parse_float(row.get(key, "")) or 0.0


def _row_fees(row: dict[str, str], shape: str) -> float:
    key = "Fee_In_Dollars" if shape == "recent_activity" else "fee_dollars"
    return parse_float(row.get(key, "")) or 0.0


def _row_profit(row: dict[str, str], shape: str) -> float:
    if shape == "recent_activity":
        return parse_float(row.get("Profit_In_Dollars", "")) or 0.0
    proceeds = parse_float(row.get("cost_proceeds_dollars", ""))
    amount = parse_float(row.get("amount_dollars", ""))
    fees = parse_float(row.get("fee_dollars", "")) or 0.0
    if proceeds is not None and amount is not None:
        return proceeds - amount - fees
    return 0.0


def _row_side(row: dict[str, str], shape: str) -> str:
    if shape == "recent_activity":
        return (row.get("Direction") or "").strip().lower()
    return (row.get("side") or "").strip().lower()


def _row_action(row: dict[str, str], shape: str) -> str:
    if shape == "recent_activity":
        return ""
    return (row.get("action") or "").strip().lower()


def _row_price_cents(row: dict[str, str], shape: str) -> int:
    key = "Price_In_Cents" if shape == "recent_activity" else "price_cents"
    return parse_int(row.get(key, ""))


def compute_fifo_realized_pnl(
    rows: list[dict[str, str]],
    shape: str,
) -> tuple[dict[tuple[str, str], ClosedTradeSummary], list[dict[str, Any]]]:
    if shape != "fill_export":
        return {}, []

    fills = []
    for row in rows:
        ticker = _row_ticker(row, shape)
        side = _row_side(row, shape)
        action = _row_action(row, shape)
        count = _row_filled_contracts(row, shape)
        price_cents = _row_price_cents(row, shape)
        dt = _row_datetime(row, shape)
        fees = _row_fees(row, shape)
        if not ticker or not side or action not in {"buy", "sell"} or count <= 0 or price_cents <= 0 or dt is None:
            continue
        fills.append(
            {
                "ticker": ticker,
                "side": side,
                "action": action,
                "count": count,
                "price_cents": price_cents,
                "fees": fees,
                "dt": dt,
            }
        )

    fills.sort(key=lambda item: item["dt"])
    inventories: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    closed: dict[tuple[str, str], ClosedTradeSummary] = {}
    unmatched_sells: list[dict[str, Any]] = []

    for fill in fills:
        key = (fill["ticker"], fill["side"])
        unit_price = fill["price_cents"] / 100.0
        unit_fee = (fill["fees"] / fill["count"]) if fill["count"] else 0.0

        if fill["action"] == "buy":
            inventories[key].append(
                {
                    "remaining": fill["count"],
                    "unit_cost": unit_price + unit_fee,
                }
            )
            continue

        sell_remaining = fill["count"]
        realized = 0.0
        round_trips = 0
        net_sell_unit = unit_price - unit_fee

        while sell_remaining > 0 and inventories[key]:
            lot = inventories[key][0]
            matched = min(sell_remaining, lot["remaining"])
            realized += matched * (net_sell_unit - lot["unit_cost"])
            lot["remaining"] -= matched
            sell_remaining -= matched
            round_trips += 1
            if lot["remaining"] <= 0:
                inventories[key].pop(0)

        if sell_remaining > 0:
            unmatched_sells.append(
                {
                    "ticker": fill["ticker"],
                    "side": fill["side"],
                    "unmatched_contracts": sell_remaining,
                    "dt": fill["dt"].isoformat(),
                }
            )

        summary = closed.setdefault(
            key,
            ClosedTradeSummary(
                ticker=fill["ticker"],
                side=fill["side"],
                contracts_closed=0,
                realized_pnl_dollars=0.0,
                round_trips=0,
            ),
        )
        matched_contracts = fill["count"] - sell_remaining
        summary.contracts_closed += matched_contracts
        summary.realized_pnl_dollars += round(realized, 6)
        summary.round_trips += round_trips

    return closed, unmatched_sells


def load_positions_realized_pnl(rows: list[dict[str, str]]) -> dict[str, float]:
    if not rows:
        return {}
    sample = rows[0]
    raw_fill = sample.get("raw_fill", "")
    raw_order = sample.get("raw_order", "")
    if not raw_fill and not raw_order:
        return {}

    raw_order_blob = sample.get("raw_order", "")
    raw_json_path = ""
    try:
        # We only know the export family, so discover the latest raw JSON locally.
        export_dir = ROOT / "app" / "data" / "kalshi_exports"
        summaries = sorted(export_dir.glob("*_summary.json"))
        if summaries:
            latest_summary = json.loads(summaries[-1].read_text())
            raw_json_path = latest_summary.get("raw_json_path", "")
    except Exception:
        raw_json_path = ""

    if not raw_json_path:
        return {}

    try:
        raw = json.loads(Path(raw_json_path).read_text())
    except Exception:
        return {}

    pnl_by_ticker: dict[str, float] = {}
    for pos in raw.get("positions", []):
        ticker = str(pos.get("ticker") or "")
        if not ticker:
            continue
        pnl = parse_float(pos.get("realized_pnl_dollars"))
        if pnl is None:
            pnl = (_safe_number_from_cents(pos.get("realized_pnl")) if "realized_pnl" in pos else None)
        if pnl is None:
            continue
        pnl_by_ticker[ticker] = pnl
    return pnl_by_ticker


def load_positions_snapshot() -> list[dict[str, Any]]:
    raw_json_path = ""
    try:
        export_dir = ROOT / "app" / "data" / "kalshi_exports"
        summaries = sorted(export_dir.glob("*_summary.json"))
        if summaries:
            latest_summary = json.loads(summaries[-1].read_text())
            raw_json_path = latest_summary.get("raw_json_path", "")
    except Exception:
        raw_json_path = ""

    if not raw_json_path:
        return []

    try:
        raw = json.loads(Path(raw_json_path).read_text())
    except Exception:
        return []

    return raw.get("positions", [])


def _safe_number_from_cents(value: Any) -> float | None:
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return None


def summarize_activity(rows: list[dict[str, str]], top_n: int) -> dict[str, Any]:
    shape = detect_csv_shape(rows)
    if shape == "recent_activity":
        orders = [row for row in rows if (row.get("type") or "").strip() == "Order"]
        deposits = [row for row in rows if (row.get("type") or "").strip() == "Deposit"]
    else:
        orders = rows
        deposits = []

    fill_statuses = {"Filled", "Partially Filled / Partially Canceled"}
    cancel_statuses = {"Canceled", "Partially Filled / Partially Canceled"}

    ticker_stats: dict[str, TickerSummary] = {}
    fill_times: dict[str, list[datetime]] = defaultdict(list)
    fills_by_date = Counter()
    contracts_by_date = Counter()
    family_fills = Counter()
    family_contracts = Counter()
    family_amount = Counter()
    family_profit = Counter()
    unique_tickers_by_family: dict[str, set[str]] = defaultdict(set)
    event_fill_counts = Counter()
    event_contract_counts = Counter()
    five_minute_buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"fills": 0, "cancels": 0, "contracts": 0, "tickers": set()}
    )

    known_profit_rows = 0
    known_amount_rows = 0
    known_fee_rows = 0
    known_direction_rows = 0
    known_price_rows = 0
    fifo_realized_by_key, unmatched_sells = compute_fifo_realized_pnl(rows, shape)
    positions_realized_pnl = load_positions_realized_pnl(rows)
    positions_snapshot = load_positions_snapshot()

    for row in orders:
        ticker = _row_ticker(row, shape)
        if not ticker:
            continue
        family = market_family(ticker)
        stats = ticker_stats.setdefault(ticker, TickerSummary(ticker=ticker, family=family))

        status = _row_status(row, shape)
        filled_contracts = _row_filled_contracts(row, shape)
        amount = _row_amount(row, shape)
        fees = _row_fees(row, shape)
        profit = _row_profit(row, shape)
        dt = _row_datetime(row, shape)

        if (parse_float(row.get("Profit_In_Dollars", "")) is not None) or (
            shape == "fill_export"
            and parse_float(row.get("cost_proceeds_dollars", "")) is not None
            and parse_float(row.get("amount_dollars", "")) is not None
        ):
            known_profit_rows += 1
        amount_key = "Amount_In_Dollars" if shape == "recent_activity" else "amount_dollars"
        fee_key = "Fee_In_Dollars" if shape == "recent_activity" else "fee_dollars"
        if parse_float(row.get(amount_key, "")) is not None:
            known_amount_rows += 1
        if parse_float(row.get(fee_key, "")) is not None:
            known_fee_rows += 1
        if (row.get("Direction") or row.get("side") or "").strip():
            known_direction_rows += 1
        price_key = "Price_In_Cents" if shape == "recent_activity" else "price_cents"
        if parse_float(row.get(price_key, "")) is not None:
            known_price_rows += 1

        stats.amount_dollars += amount
        stats.fees_dollars += fees
        stats.profit_dollars += profit
        key = (ticker, _row_side(row, shape))
        if key in fifo_realized_by_key:
            stats.realized_pnl_dollars = round(
                stats.realized_pnl_dollars + 0.0,
                6,
            )

        is_fill = filled_contracts > 0 and (
            shape == "fill_export" or status in fill_statuses
        )
        if is_fill:
            stats.fills += 1
            stats.contracts += filled_contracts
            family_fills[family] += 1
            family_contracts[family] += filled_contracts
            family_amount[family] += amount
            family_profit[family] += profit
            unique_tickers_by_family[family].add(ticker)

            if dt is not None:
                fill_times[ticker].append(dt)
                date_key = dt.date().isoformat()
                fills_by_date[date_key] += 1
                contracts_by_date[date_key] += filled_contracts

                bucket_dt = dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
                bucket = five_minute_buckets[bucket_dt.isoformat()]
                bucket["fills"] += 1
                bucket["contracts"] += filled_contracts
                bucket["tickers"].add(ticker)

            ek = event_key(ticker)
            event_fill_counts[ek] += 1
            event_contract_counts[ek] += filled_contracts

        if shape == "recent_activity" and status in cancel_statuses:
            stats.canceled += 1
            if status == "Partially Filled / Partially Canceled":
                stats.partial_cancel += 1
            if dt is not None:
                bucket_dt = dt.replace(minute=(dt.minute // 5) * 5, second=0, microsecond=0)
                bucket = five_minute_buckets[bucket_dt.isoformat()]
                bucket["cancels"] += 1
                bucket["tickers"].add(ticker)

        if dt is not None:
            iso_dt = dt.isoformat()
            if not stats.first_seen or iso_dt < stats.first_seen:
                stats.first_seen = iso_dt
            if not stats.last_seen or iso_dt > stats.last_seen:
                stats.last_seen = iso_dt

    for ticker, times in fill_times.items():
        times.sort()
        left = 0
        best = 0
        for right, current in enumerate(times):
            while current - times[left] > timedelta(minutes=15):
                left += 1
            best = max(best, right - left + 1)
        ticker_stats[ticker].best_15m_fill_burst = best

    contract_totals = Counter({ticker: stat.contracts for ticker, stat in ticker_stats.items()})
    fill_totals = Counter({ticker: stat.fills for ticker, stat in ticker_stats.items()})
    realized_pnl_by_ticker = Counter()
    realized_pnl_by_family = Counter()
    for (ticker, _side), summary in fifo_realized_by_key.items():
        realized_pnl_by_ticker[ticker] += summary.realized_pnl_dollars
        realized_pnl_by_family[market_family(ticker)] += summary.realized_pnl_dollars
    for ticker, stat in ticker_stats.items():
        stat.realized_pnl_dollars = round(realized_pnl_by_ticker[ticker], 4)

    positions_realized_by_family = Counter()
    for ticker, pnl in positions_realized_pnl.items():
        positions_realized_by_family[market_family(ticker)] += pnl
    open_positions_by_family = Counter()
    open_exposure_by_family = Counter()
    for pos in positions_snapshot:
        ticker = str(pos.get("ticker") or "")
        if not ticker:
            continue
        family = market_family(ticker)
        position = parse_int(str(pos.get("position_fp") or pos.get("position") or "0"))
        exposure = parse_float(str(pos.get("market_exposure_dollars") or "0")) or 0.0
        if position != 0:
            open_positions_by_family[family] += 1
            open_exposure_by_family[family] += exposure

    top_by_contracts = []
    total_contracts = sum(contract_totals.values())
    for ticker, contracts in contract_totals.most_common(top_n):
        stat = ticker_stats[ticker]
        share = (contracts / total_contracts) if total_contracts else 0.0
        top_by_contracts.append(
            {
                "ticker": ticker,
                "family": stat.family,
                "contracts": contracts,
                "fills": stat.fills,
                "share_of_contracts": round(share, 4),
                "cancelish_events": stat.canceled,
                "best_15m_fill_burst": stat.best_15m_fill_burst,
            }
        )

    top_by_fill_count = []
    for ticker, fills in fill_totals.most_common(top_n):
        stat = ticker_stats[ticker]
        top_by_fill_count.append(
            {
                "ticker": ticker,
                "family": stat.family,
                "fills": fills,
                "contracts": stat.contracts,
                "cancelish_events": stat.canceled,
                "best_15m_fill_burst": stat.best_15m_fill_burst,
            }
        )

    top_events = []
    for ek, fills in event_fill_counts.most_common(top_n):
        top_events.append(
            {
                "event": ek,
                "fills": fills,
                "contracts": event_contract_counts[ek],
            }
        )

    cancel_heavy = []
    for ticker, stat in sorted(
        ticker_stats.items(),
        key=lambda item: (item[1].canceled, item[1].fills, item[1].contracts),
        reverse=True,
    )[:top_n]:
        cancel_heavy.append(
            {
                "ticker": ticker,
                "family": stat.family,
                "fills": stat.fills,
                "cancelish_events": stat.canceled,
                "contracts": stat.contracts,
            }
        )

    top_time_bursts = []
    sorted_buckets = sorted(
        five_minute_buckets.items(),
        key=lambda item: (item[1]["fills"], item[1]["contracts"], item[1]["cancels"]),
        reverse=True,
    )
    for bucket_key, bucket in sorted_buckets[:top_n]:
        top_time_bursts.append(
            {
                "bucket_start": bucket_key,
                "fills": bucket["fills"],
                "contracts": bucket["contracts"],
                "cancels": bucket["cancels"],
                "unique_tickers": len(bucket["tickers"]),
            }
        )

    family_summary = {}
    for family, fills in family_fills.items():
        family_summary[family] = {
            "fills": fills,
            "contracts": family_contracts[family],
            "unique_tickers": len(unique_tickers_by_family[family]),
            "avg_contracts_per_fill": round(family_contracts[family] / fills, 2) if fills else 0.0,
                "known_amount_dollars": round(family_amount[family], 2),
                "known_profit_dollars": round(family_profit[family], 2),
                "fifo_realized_pnl_dollars": round(realized_pnl_by_family[family], 2),
                "positions_realized_pnl_dollars": round(positions_realized_by_family[family], 2),
        }

    total_fills = sum(family_fills.values())
    top10_contract_share = 0.0
    for ticker, contracts in contract_totals.most_common(10):
        if total_contracts:
            top10_contract_share += contracts / total_contracts

    coverage = {
        "csv_shape": shape,
        "rows": len(rows),
        "order_rows": len(orders),
        "deposit_rows": len(deposits),
        "filled_order_rows": total_fills,
        "csv_has_direction_rows": known_direction_rows,
        "csv_has_price_rows": known_price_rows,
        "csv_has_amount_rows": known_amount_rows,
        "csv_has_fee_rows": known_fee_rows,
        "csv_has_profit_rows": known_profit_rows,
    }
    performance_model = build_performance_model()

    return {
        "coverage": coverage,
        "summary": {
            "total_fills": total_fills,
            "total_contracts": total_contracts,
            "total_cancelish_events": sum(stat.canceled for stat in ticker_stats.values()),
            "top_10_contract_share": round(top10_contract_share, 4),
            "fifo_realized_pnl_dollars": round(sum(realized_pnl_by_ticker.values()), 2),
            "unmatched_sell_rows": len(unmatched_sells),
            "positions_realized_pnl_dollars": round(sum(positions_realized_pnl.values()), 2),
        },
        "fills_by_date": [
            {
                "date": date_key,
                "fills": fills_by_date[date_key],
                "contracts": contracts_by_date[date_key],
            }
            for date_key in sorted(fills_by_date)
        ],
        "family_summary": family_summary,
        "open_positions_by_family": [
            {
                "family": family,
                "open_positions": open_positions_by_family[family],
                "open_exposure_dollars": round(open_exposure_by_family[family], 2),
            }
            for family in sorted(open_positions_by_family, key=lambda item: open_exposure_by_family[item], reverse=True)
        ],
        "top_tickers_by_contracts": top_by_contracts,
        "top_tickers_by_fill_count": top_by_fill_count,
        "top_events": top_events,
        "cancel_heavy_tickers": cancel_heavy,
        "top_time_bursts": top_time_bursts,
        "top_winners_by_fifo_pnl": [
            {
                "ticker": ticker,
                "family": market_family(ticker),
                "realized_pnl_dollars": round(pnl, 2),
            }
            for ticker, pnl in sorted(realized_pnl_by_ticker.items(), key=lambda item: item[1], reverse=True)[:top_n]
        ],
        "top_losers_by_fifo_pnl": [
            {
                "ticker": ticker,
                "family": market_family(ticker),
                "realized_pnl_dollars": round(pnl, 2),
            }
            for ticker, pnl in sorted(realized_pnl_by_ticker.items(), key=lambda item: item[1])[:top_n]
        ],
        "top_realized_by_positions": [
            {
                "ticker": ticker,
                "family": market_family(ticker),
                "realized_pnl_dollars": round(pnl, 2),
            }
            for ticker, pnl in sorted(positions_realized_pnl.items(), key=lambda item: item[1], reverse=True)[:top_n]
        ],
        "worst_realized_by_positions": [
            {
                "ticker": ticker,
                "family": market_family(ticker),
                "realized_pnl_dollars": round(pnl, 2),
            }
            for ticker, pnl in sorted(positions_realized_pnl.items(), key=lambda item: item[1])[:top_n]
        ],
        "unmatched_sells": unmatched_sells[:top_n],
        "all_ticker_stats": [asdict(stat) for stat in sorted(ticker_stats.values(), key=lambda s: (s.contracts, s.fills), reverse=True)],
        "performance_model": performance_model,
    }


def print_report(report: dict[str, Any], top_n: int) -> None:
    coverage = report["coverage"]
    summary = report["summary"]

    print("Kalshi Activity Analysis")
    print("=" * 80)
    print(f"Rows: {coverage['rows']} total | {coverage['order_rows']} orders | {coverage['deposit_rows']} deposits")
    print(
        "CSV coverage:"
        f" direction={coverage['csv_has_direction_rows']},"
        f" price={coverage['csv_has_price_rows']},"
        f" amount={coverage['csv_has_amount_rows']},"
        f" fee={coverage['csv_has_fee_rows']},"
        f" profit={coverage['csv_has_profit_rows']}"
    )
    print(
        f"Filled rows: {summary['total_fills']} | Contracts: {summary['total_contracts']} | "
        f"Cancelish events: {summary['total_cancelish_events']} | "
        f"Top-10 contract share: {summary['top_10_contract_share']:.1%}"
    )
    if "fifo_realized_pnl_dollars" in summary:
        print(
            f"FIFO realized P&L: ${summary['fifo_realized_pnl_dollars']:+.2f} | "
            f"Unmatched sell rows: {summary['unmatched_sell_rows']}"
        )
    if "positions_realized_pnl_dollars" in summary:
        print(f"Kalshi positions realized P&L: ${summary['positions_realized_pnl_dollars']:+.2f}")

    print("\nFills By Date")
    print("-" * 80)
    for item in report["fills_by_date"]:
        print(f"{item['date']}: fills={item['fills']}, contracts={item['contracts']}")

    print("\nBy Family")
    print("-" * 80)
    for family, stats in sorted(report["family_summary"].items(), key=lambda item: item[1]["contracts"], reverse=True):
        print(
            f"{family}: fills={stats['fills']}, contracts={stats['contracts']}, "
            f"unique_tickers={stats['unique_tickers']}, avg_contracts_per_fill={stats['avg_contracts_per_fill']}, "
            f"fifo_pnl=${stats.get('fifo_realized_pnl_dollars', 0):+.2f}, "
            f"kalshi_realized=${stats.get('positions_realized_pnl_dollars', 0):+.2f}"
        )

    if report.get("open_positions_by_family"):
        print("\nOpen Positions By Family")
        print("-" * 80)
        for item in report["open_positions_by_family"][:top_n]:
            print(
                f"{item['family']}: open_positions={item['open_positions']}, "
                f"open_exposure=${item['open_exposure_dollars']:.2f}"
            )

    print(f"\nTop {top_n} Tickers By Contracts")
    print("-" * 80)
    for item in report["top_tickers_by_contracts"][:top_n]:
        print(
            f"{item['contracts']:>5} contracts | {item['fills']:>3} fills | "
            f"{item['share_of_contracts']:.1%} share | burst15={item['best_15m_fill_burst']:>2} | "
            f"{item['ticker']}"
        )

    print(f"\nTop {top_n} Events")
    print("-" * 80)
    for item in report["top_events"][:top_n]:
        print(f"{item['contracts']:>5} contracts | {item['fills']:>3} fills | {item['event']}")

    print(f"\nTop {top_n} Cancel-Heavy Tickers")
    print("-" * 80)
    for item in report["cancel_heavy_tickers"][:top_n]:
        print(
            f"{item['cancelish_events']:>3} cancelish | {item['fills']:>3} fills | "
            f"{item['contracts']:>5} contracts | {item['ticker']}"
        )

    print(f"\nTop {top_n} Five-Minute Bursts")
    print("-" * 80)
    for item in report["top_time_bursts"][:top_n]:
        print(
            f"{item['bucket_start']} | fills={item['fills']:>2} | contracts={item['contracts']:>4} | "
            f"cancels={item['cancels']:>2} | tickers={item['unique_tickers']:>2}"
        )

    if report.get("top_winners_by_fifo_pnl"):
        print(f"\nTop {top_n} Winners By FIFO P&L")
        print("-" * 80)
        for item in report["top_winners_by_fifo_pnl"][:top_n]:
            print(f"${item['realized_pnl_dollars']:+8.2f} | {item['ticker']}")

        print(f"\nTop {top_n} Losers By FIFO P&L")
        print("-" * 80)
        for item in report["top_losers_by_fifo_pnl"][:top_n]:
            print(f"${item['realized_pnl_dollars']:+8.2f} | {item['ticker']}")

    if report.get("top_realized_by_positions"):
        print(f"\nTop {top_n} Winners By Kalshi Realized P&L")
        print("-" * 80)
        for item in report["top_realized_by_positions"][:top_n]:
            print(f"${item['realized_pnl_dollars']:+8.2f} | {item['ticker']}")

        print(f"\nTop {top_n} Losers By Kalshi Realized P&L")
        print("-" * 80)
        for item in report["worst_realized_by_positions"][:top_n]:
            print(f"${item['realized_pnl_dollars']:+8.2f} | {item['ticker']}")

    if coverage["csv_has_profit_rows"] == 0 and coverage["csv_shape"] == "recent_activity":
        print("\nNote")
        print("-" * 80)
        print(
            "This CSV does not contain realized P&L per order, so the report focuses on "
            "concentration, churn, and execution behavior. If you export a richer history "
            "with populated price/amount/profit columns, the same script will include that coverage."
        )

    perf_model = report.get("performance_model", {})
    if perf_model:
        print("\nPerformance Model")
        print("-" * 80)
        print(
            f"blocked_tickers={len(perf_model.get('blocked_tickers', []))}, "
            f"blocked_events={len(perf_model.get('blocked_events', []))}, "
            f"blocked_families={len(perf_model.get('blocked_families', []))}, "
            f"blocked_sources={len(perf_model.get('blocked_sources', []))}, "
            f"family_multipliers={len(perf_model.get('family_multipliers', {}))}, "
            f"source_multipliers={len(perf_model.get('signal_source_multipliers', {}))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Kalshi account activity exports.")
    parser.add_argument("--csv", help="Path to an existing Kalshi CSV export.")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch authenticated Kalshi account history first, then analyze the generated fills CSV.",
    )
    parser.add_argument(
        "--fetch-limit",
        type=int,
        default=500,
        help="Page size to use when fetching account history with --fetch.",
    )
    parser.add_argument(
        "--fetch-prefix",
        default="kalshi_account_history",
        help="Filename prefix to use for the fetched account-history export.",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of top rows to print in each section.")
    parser.add_argument("--json", help="Optional path to write the full report as JSON.")
    args = parser.parse_args()

    csv_path: Path | None = Path(args.csv) if args.csv else None
    if args.fetch:
        from scripts.export_kalshi_account_history import export_history

        summary = asyncio.run(export_history(limit=args.fetch_limit, prefix=args.fetch_prefix))
        csv_path = Path(summary["csv_path"])
        print(f"Fetched fresh Kalshi account history to {csv_path}")

    if csv_path is None:
        raise SystemExit("Pass --csv <path> or use --fetch.")

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    report = summarize_activity(rows, top_n=args.top)
    print_report(report, top_n=args.top)

    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2))
        print(f"\nJSON report written to {json_path}")


if __name__ == "__main__":
    main()
