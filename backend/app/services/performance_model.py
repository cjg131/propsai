"""
Build and load a lightweight performance model from recent local trades and
the latest Kalshi account export.
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "trading_engine.db"
EXPORT_DIR = ROOT / "data" / "kalshi_exports"
MODEL_PATH = ROOT / "data" / "performance_model.json"


def market_family(strategy: str, ticker: str, signal_source: str = "") -> str:
    strategy = (strategy or "").lower()
    source = (signal_source or "").lower()
    ticker_upper = (ticker or "").upper()

    if strategy == "weather":
        if source == "weather_observed_arbitrage":
            return "weather_observed"
        return "weather_forecast"
    if strategy == "sports":
        if source == "parlay_pricer":
            return "sports_parlay"
        if any(tag in ticker_upper for tag in ("KXSL", "KXLIGA", "KXMLS", "KXEPL", "KXSERIE", "KXBUND", "KXLIGUE")):
            return "sports_single_soccer"
        return "sports_single"
    if strategy == "finance":
        if "bracket" in source:
            return "finance_bracket"
        return "finance_threshold"
    if strategy == "crypto":
        return "crypto_momentum"
    if strategy == "econ":
        return "econ"
    if strategy == "nba_props":
        return "nba_props"
    return "other"


def _price_bucket(price_cents: int) -> str:
    if price_cents < 10:
        return "01-09c"
    if price_cents < 20:
        return "10-19c"
    if price_cents < 30:
        return "20-29c"
    if price_cents < 40:
        return "30-39c"
    if price_cents < 50:
        return "40-49c"
    if price_cents < 60:
        return "50-59c"
    if price_cents < 70:
        return "60-69c"
    if price_cents < 80:
        return "70-79c"
    if price_cents < 90:
        return "80-89c"
    return "90-99c"


def _latest_export_paths() -> tuple[Path | None, Path | None]:
    summaries = sorted(EXPORT_DIR.glob("*_summary.json"))
    if not summaries:
        return None, None
    try:
        summary = json.loads(summaries[-1].read_text())
    except Exception:
        return None, None
    raw_path = Path(summary.get("raw_json_path", "")) if summary.get("raw_json_path") else None
    csv_path = Path(summary.get("csv_path", "")) if summary.get("csv_path") else None
    return raw_path, csv_path


def build_performance_model() -> dict[str, Any]:
    model: dict[str, Any] = {
        "generated_from": {"db_path": str(DB_PATH), "export_dir": str(EXPORT_DIR)},
        "family_multipliers": {},
        "signal_source_multipliers": {},
        "price_bucket_multipliers": {},
        "blocked_families": [],
        "blocked_sources": [],
        "blocked_tickers": [],
        "blocked_events": [],
        "notes": [],
    }

    if DB_PATH.exists():
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute(
            """SELECT strategy, ticker, signal_source, price_cents, pnl
               FROM trades
               WHERE (status = 'settled' OR action = 'sell')
                 AND settled_at >= datetime('now', '-45 days')"""
        )
        rows = c.fetchall()
        conn.close()

        family_pnls: dict[str, list[float]] = defaultdict(list)
        source_pnls: dict[str, list[float]] = defaultdict(list)
        bucket_pnls: dict[str, list[float]] = defaultdict(list)

        for row in rows:
            family = market_family(row["strategy"], row["ticker"], row["signal_source"])
            family_pnls[family].append(float(row["pnl"] or 0.0))
            if row["signal_source"]:
                source_pnls[str(row["signal_source"])].append(float(row["pnl"] or 0.0))
            if row["price_cents"] is not None:
                bucket_pnls[_price_bucket(int(row["price_cents"]))].append(float(row["pnl"] or 0.0))

        def compute_multiplier(values: list[float], min_trades: int) -> float | None:
            if len(values) < min_trades:
                return None
            avg_pnl = sum(values) / len(values)
            win_rate = sum(1 for v in values if v > 0) / len(values)
            if avg_pnl < -1.0 or win_rate < 0.42:
                return 0.70
            if avg_pnl > 0.75 and win_rate > 0.56:
                return 1.10
            if avg_pnl > 0:
                return 1.03
            return 0.95

        for family, vals in family_pnls.items():
            multiplier = compute_multiplier(vals, min_trades=8)
            if multiplier is not None:
                model["family_multipliers"][family] = {
                    "multiplier": multiplier,
                    "trades": len(vals),
                    "avg_pnl": round(sum(vals) / len(vals), 4),
                    "win_rate": round(sum(1 for v in vals if v > 0) / len(vals), 4),
                }
                if multiplier <= 0.70 and len(vals) >= 10:
                    model["blocked_families"].append(family)

        for source, vals in source_pnls.items():
            multiplier = compute_multiplier(vals, min_trades=6)
            if multiplier is not None:
                model["signal_source_multipliers"][source] = {
                    "multiplier": multiplier,
                    "trades": len(vals),
                    "avg_pnl": round(sum(vals) / len(vals), 4),
                    "win_rate": round(sum(1 for v in vals if v > 0) / len(vals), 4),
                }
                if multiplier <= 0.70 and len(vals) >= 8:
                    model["blocked_sources"].append(source)

        for bucket, vals in bucket_pnls.items():
            multiplier = compute_multiplier(vals, min_trades=6)
            if multiplier is not None:
                model["price_bucket_multipliers"][bucket] = {
                    "multiplier": multiplier,
                    "trades": len(vals),
                    "avg_pnl": round(sum(vals) / len(vals), 4),
                    "win_rate": round(sum(1 for v in vals if v > 0) / len(vals), 4),
                }

    raw_path, csv_path = _latest_export_paths()
    if raw_path and raw_path.exists():
        try:
            raw = json.loads(raw_path.read_text())
            positions = raw.get("positions", [])
            bad_tickers: list[tuple[str, float]] = []
            for pos in positions:
                pnl_raw = pos.get("realized_pnl_dollars")
                try:
                    realized_pnl = float(pnl_raw)
                except (TypeError, ValueError):
                    realized_pnl = 0.0
                total_traded = float(pos.get("total_traded_dollars") or 0.0)
                if realized_pnl <= -1.0 and total_traded >= 5.0:
                    bad_tickers.append((str(pos.get("ticker") or ""), realized_pnl))
            bad_tickers.sort(key=lambda item: item[1])
            model["blocked_tickers"] = [ticker for ticker, _ in bad_tickers[:10] if ticker]
        except Exception:
            model["notes"].append("Latest raw export could not be parsed")

    if csv_path and csv_path.exists():
        try:
            import csv

            event_contracts = Counter()
            with csv_path.open() as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    ticker = str(row.get("ticker") or "")
                    if not ticker:
                        continue
                    event = ticker.rsplit("-", 1)[0] if "-" in ticker else ticker
                    try:
                        count = int(float(row.get("fill_count") or 0))
                    except ValueError:
                        count = 0
                    event_contracts[event] += count
            model["blocked_events"] = [event for event, contracts in event_contracts.most_common(10) if contracts >= 500]
        except Exception:
            model["notes"].append("Latest fill export could not be parsed")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2))
    return model


def load_performance_model() -> dict[str, Any]:
    if MODEL_PATH.exists():
        try:
            return json.loads(MODEL_PATH.read_text())
        except Exception:
            pass
    return build_performance_model()
