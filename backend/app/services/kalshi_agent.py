"""
Kalshi Autonomous Trading Agent.
Orchestrates weather, sports, crypto, finance, and econ strategies.
Runs on a schedule: weather 4x/day, sports 5min, crypto 2min, finance 10min, econ 30min.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings
from app.logging_config import get_logger
from app.services.adaptive_thresholds import get_adaptive_thresholds
from app.services.cross_market_sports import CrossMarketScanner
from app.services.cross_strategy_correlation import get_cross_strategy_engine
from app.services.crypto_data import CryptoDataService
from app.services.econ_data import EconDataService
from app.services.finance_data import FinanceDataService
from app.services.kalshi_api import get_kalshi_client
from app.services.kalshi_scanner import KalshiScanner, parse_parlay_legs
from app.services.kalshi_ws import KalshiWebSocket, get_kalshi_ws
from app.services.nba_data import NBADataService, get_nba_data
from app.services.news_sentiment import get_market_news_sentiment
from app.services.parlay_pricer import TEAM_ALIASES, normalize_team, price_parlay_legs, teams_match
from app.services.polymarket_data import get_polymarket_data
from app.services.referee_data import RefereeDataService, get_referee_data
from app.services.signal_scorer import get_signal_scorer
from app.services.smart_predictor import SmartPredictor, get_smart_predictor
from app.services.trade_analyzer import get_trade_analyzer
from app.services.trading_engine import get_trading_engine
from app.services.discord_webhook import send_discord_notification
from app.services.event_bus import get_event_bus
from app.services.weather_data import CITY_CONFIGS, WeatherConsensus

logger = get_logger(__name__)

UTC = timezone.utc


def _normalize_person_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _person_name_matches(left: str, right: str) -> bool:
    """Require a high-confidence name match instead of loose substring guesses."""
    left_norm = _normalize_person_name(left)
    right_norm = _normalize_person_name(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_parts = left_norm.split()
    right_parts = right_norm.split()
    if not left_parts or not right_parts:
        return False

    left_first, left_last = left_parts[0], left_parts[-1]
    right_first, right_last = right_parts[0], right_parts[-1]
    if left_last != right_last:
        return False
    if left_first == right_first:
        return True

    # Allow "J Brunson" / "Jalen Brunson" style matches, but not loose prefixes.
    return (
        len(left_first) == 1 and left_first == right_first[:1]
    ) or (
        len(right_first) == 1 and right_first == left_first[:1]
    )


def _find_unique_person_match(records: dict[str, Any], player_name: str) -> Any | None:
    normalized_target = _normalize_person_name(player_name)
    if not normalized_target:
        return None

    if normalized_target in records:
        return records[normalized_target]

    matches = [
        value
        for candidate_name, value in records.items()
        if _person_name_matches(player_name, candidate_name)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _find_unique_prop_match(
    odds_props_lookup: dict[str, dict[str, Any]],
    player_name: str,
    prop_type: str,
) -> dict[str, Any] | None:
    exact_key = f"{_normalize_person_name(player_name)}|{prop_type}"
    if exact_key in odds_props_lookup:
        return odds_props_lookup[exact_key]

    matches = []
    for odds_key, odds_value in odds_props_lookup.items():
        try:
            odds_player, odds_prop_type = odds_key.rsplit("|", 1)
        except ValueError:
            continue
        if odds_prop_type != prop_type:
            continue
        if _person_name_matches(player_name, odds_player):
            matches.append(odds_value)
    if len(matches) == 1:
        return matches[0]
    return None


def _suffix_matches_team_name(team_suffix: str, team_name: str) -> bool:
    """Match Kalshi ticker suffixes to team names without broad substring guesses."""
    suffix = normalize_team(team_suffix)
    target = normalize_team(team_name)
    if not suffix or not target:
        return False

    if suffix == target:
        return True

    # Exact alias-group match covers club abbreviations like PSG, BVB, etc.
    for canonical, aliases in TEAM_ALIASES.items():
        normalized_names = {normalize_team(canonical), *(normalize_team(name) for name in aliases)}
        if suffix in normalized_names and target in normalized_names:
            return True

    target_words = target.split()

    # Acronym match: "psg" -> "paris saint germain"
    acronym = "".join(word[0] for word in target_words if word)
    if suffix == acronym:
        return True

    # Fall back to token-prefix matching for compact abbreviations only.
    if len(suffix) < 2 or len(suffix) > 4:
        return False

    for word in target_words:
        if len(word) > len(suffix) and word.startswith(suffix):
            return True

    return False


class KalshiAgent:
    """
    Autonomous trading agent for Kalshi markets.
    Coordinates weather, sports, and crypto strategies with the trading engine.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.engine = get_trading_engine()
        self.kalshi = get_kalshi_client()

        # Weather consensus engine
        self.weather = WeatherConsensus(
            tomorrow_io_key=getattr(settings, "tomorrow_io_api_key", ""),
            visual_crossing_key=getattr(settings, "visual_crossing_api_key", ""),
            weatherbit_key=getattr(settings, "weatherbit_api_key", ""),
            openweathermap_key=getattr(settings, "openweathermap_api_key", ""),
        )

        # Cross-market sports scanner
        self.sports = CrossMarketScanner(
            odds_api_key=settings.the_odds_api_key,
        )

        # Dynamic market scanner
        self.scanner = KalshiScanner(self.kalshi)

        # Crypto data service
        self.crypto = CryptoDataService()

        # Finance data service (S&P/Nasdaq)
        self.finance = FinanceDataService()

        # Economic data service (CPI/Fed/Gas/Jobs)
        self.econ = EconDataService(
            fred_api_key=getattr(settings, "fred_api_key", ""),
        )

        # NBA props prediction engine
        self.nba_data: NBADataService | None = None
        self.predictor: SmartPredictor | None = None

        # Market news sentiment (crypto/finance)
        self.market_news = get_market_news_sentiment()

        # Post-trade analysis agent (OpenClaw)
        self.trade_analyzer = get_trade_analyzer()

        # Adaptive threshold tuning
        self.adaptive = get_adaptive_thresholds()

        # Signal quality scorer
        self.signal_scorer = get_signal_scorer()

        # Polymarket cross-reference
        self.polymarket = get_polymarket_data()

        # Kalshi WebSocket for real-time price updates
        self.ws: KalshiWebSocket = get_kalshi_ws()

        # Cross-strategy correlation engine
        self.correlation = get_cross_strategy_engine()

        # Referee data service (NBA totals edge)
        self.referee_data: RefereeDataService = get_referee_data(
            api_key=getattr(settings, "sportsdataio_api_key", "")
        )

        self._running = False
        self._weather_task: asyncio.Task | None = None
        self._main_task: asyncio.Task | None = None
        self._crypto_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None
        self._health_task: asyncio.Task | None = None
        self._last_api_check_ok: bool = True
        self._consecutive_api_failures: int = 0

        # Cross-cycle exit tracker: prevents monitor from re-exiting a ticker
        # that was already exited in a previous cycle (belt-and-suspenders for
        # the _from_fallback guard). Maps ticker → timestamp of exit attempt.
        self._exited_tickers: dict[str, float] = {}

        # Resting-order cancel alerts (to detect persistent non-fills/chasing)
        self._cancel_alert_window_mins = int(os.environ.get("CANCEL_ALERT_WINDOW_MINS", "30"))
        self._cancel_alert_threshold = int(os.environ.get("CANCEL_ALERT_THRESHOLD", "6"))
        self._auto_kill_on_cancel_storm = os.environ.get("AUTO_KILL_ON_CANCEL_STORM", "false").lower() == "true"
        self._resting_cancel_timestamps: dict[str, list[float]] = {}
        self._performance_model_refresh_secs = int(os.environ.get("PERFORMANCE_MODEL_REFRESH_SECS", "900"))
        self._last_performance_model_refresh: float = 0.0

    def _build_live_position_record(
        self,
        market_position: dict[str, Any],
        db_rec: dict[str, Any] | None = None,
        *,
        include_title: bool = False,
    ) -> dict[str, Any]:
        """Merge Kalshi position truth with local metadata using fee-aware economics."""
        db_rec = db_rec or {}
        ticker = market_position.get("ticker", "")
        pos_qty = market_position.get("position", 0) or 0
        qty = abs(pos_qty)
        side = "yes" if pos_qty > 0 else "no"
        exposure_cents = float(market_position.get("market_exposure", 0) or 0)
        fees_cents = float(market_position.get("fees_paid", 0) or 0)
        total_cost_cents = exposure_cents + fees_cents
        avg_entry = round(total_cost_cents / qty, 1) if qty > 0 else 50

        record: dict[str, Any] = {
            "ticker": ticker,
            "side": side,
            "contracts": qty,
            "total_cost": round(exposure_cents / 100.0, 2),
            "total_fees": round(fees_cents / 100.0, 2),
            "avg_entry_cents": avg_entry,
            "avg_our_prob": db_rec.get("avg_our_prob", 0.5),
            "strategy": db_rec.get("strategy") or (
                "weather" if "HIGH" in ticker or "LOW" in ticker else "unknown"
            ),
            "max_risk": round(total_cost_cents / 100.0, 2),
            "max_profit": round(max(0.0, qty * 1.0 - total_cost_cents / 100.0), 2),
        }

        if include_title:
            record.update({
                "title": db_rec.get("title") or market_position.get("market_title") or ticker,
                "signal_source": db_rec.get("signal_source", ""),
                "avg_entry_kalshi_prob": db_rec.get("avg_entry_kalshi_prob", 0),
                "avg_entry_edge": db_rec.get("avg_entry_edge", 0),
                "first_entry": db_rec.get("first_entry", ""),
                "last_entry": db_rec.get("last_entry", ""),
                "num_fills": db_rec.get("num_fills", 0),
                "paper_mode": db_rec.get("paper_mode", False),
                "current_yes_bid": None,
                "current_yes_ask": None,
                "current_no_bid": None,
                "current_no_ask": None,
                "mark_price_cents": None,
                "unrealized_pnl": None,
                "current_edge": None,
                "status": "open",
            })

        return record

    @staticmethod
    def _compute_side_market_prices(
        *,
        side: str,
        yes_bid: int,
        yes_ask: int,
        no_bid: int,
        no_ask: int,
        last_price: int,
        avg_entry_cents: int,
    ) -> tuple[int, int]:
        """Return (valuation_mark, executable_exit_price) for a side contract."""
        if side == "yes":
            bid, ask = yes_bid, yes_ask
            last_side_price = last_price if last_price > 0 else 0
        else:
            bid, ask = no_bid, no_ask
            last_side_price = (100 - last_price) if last_price > 0 else 0

        if bid > 0 and ask > 0 and bid >= ask * 0.5:
            mark_price = bid
        elif bid > 0 and ask > 0 and bid <= last_side_price <= ask:
            mark_price = last_side_price
        elif ask > 0:
            mark_price = ask
        elif last_side_price > 0:
            mark_price = last_side_price
        elif bid > 0:
            mark_price = bid
        else:
            mark_price = avg_entry_cents

        if bid > 0:
            exit_price = bid
        elif last_side_price > 0:
            exit_price = last_side_price
        elif ask > 0:
            exit_price = ask
        else:
            exit_price = avg_entry_cents

        return int(mark_price), int(exit_price)

    @staticmethod
    def _compute_hold_ev_cents(*, our_prob: float, exit_price: int) -> float:
        """Expected liquidation edge of holding vs selling now, in cents per contract."""
        prob = max(0.0, min(1.0, float(our_prob)))
        return (prob * 100.0) - float(exit_price)

    @staticmethod
    def _compute_position_pnl(*, contracts: int, price_cents: int, total_cost: float, total_fees: float) -> float:
        """P&L of a position marked or liquidated at a given side price."""
        value = contracts * price_cents / 100.0
        return round(value - total_cost - total_fees, 2)

    @staticmethod
    def _nba_prop_within_time_gate(ticker: str, *, now_dt: datetime | None = None) -> bool:
        """Return True only for NBA prop tickers within the 12-hour trading window."""
        ticker_date_match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker or "")
        if not ticker_date_match:
            return False

        months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                  "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
        try:
            yr = int("20" + ticker_date_match.group(1))
            mon = months[ticker_date_match.group(2)]
            day = int(ticker_date_match.group(3))
            from zoneinfo import ZoneInfo

            game_dt = datetime(yr, mon, day, 19, 0, tzinfo=ZoneInfo("America/New_York"))
            now = now_dt or datetime.now(ZoneInfo("America/New_York"))
            hours_until_game = (game_dt - now).total_seconds() / 3600
            return hours_until_game <= 12
        except Exception:
            return False

    # ── Cross-Strategy Correlation Helpers ────────────────────────────

    def _record_cross_signal(
        self,
        strategy: str,
        ticker: str,
        side: str,
        our_prob: float,
        kalshi_prob: float,
        confidence: float,
    ) -> None:
        """Record a signal to the cross-strategy correlation engine."""
        try:
            self.correlation.record_signal(
                strategy=strategy,
                ticker=ticker,
                side=side,
                our_prob=our_prob,
                kalshi_prob=kalshi_prob,
                confidence=confidence,
            )
        except Exception as e:
            logger.debug("Cross-strategy record failed", error=str(e))

    def _apply_correlation_adjustment(
        self, strategy: str, ticker: str, confidence: float
    ) -> float:
        """Apply cross-strategy correlation adjustment to confidence.

        Returns adjusted confidence (may be boosted or penalized).
        """
        try:
            adjustment = self.correlation.get_confidence_adjustment(ticker, strategy)
            if adjustment != 0:
                adjusted = confidence + adjustment
                adjusted = max(0.1, min(adjusted, 1.0))
                if abs(adjustment) >= 0.02:
                    logger.info(
                        "Cross-strategy adjustment",
                        strategy=strategy,
                        ticker=ticker,
                        original=f"{confidence:.3f}",
                        adjustment=f"{adjustment:+.3f}",
                        adjusted=f"{adjusted:.3f}",
                    )
                return adjusted
        except Exception as e:
            logger.debug("Cross-strategy adjustment failed", error=str(e))
        return confidence

    def _refresh_performance_model_if_due(self, *, force: bool = False) -> None:
        """Refresh runtime performance model so filters learn from new realized results."""
        import time

        now_ts = time.time()
        if not force and (now_ts - self._last_performance_model_refresh) < self._performance_model_refresh_secs:
            return

        try:
            model = self.engine.load_performance_model(force_refresh=True)
            self._last_performance_model_refresh = now_ts
            self.engine.log_event(
                "info",
                "Performance model refreshed "
                f"(families={len(model.get('family_multipliers', {}))}, "
                f"sources={len(model.get('signal_source_multipliers', {}))}, "
                f"blocked_families={len(model.get('blocked_families', []))}, "
                f"blocked_sources={len(model.get('blocked_sources', []))})",
                strategy="risk",
            )
        except Exception as e:
            self.engine.log_event("warning", f"Performance model refresh failed: {e}", strategy="risk")

    def _extract_signal_components(self, trade: dict[str, Any]) -> dict[str, float]:
        """Extract signal component values from a trade's notes/details for quality scoring."""
        components: dict[str, float] = {}
        # Try to parse from signal details stored in the DB
        try:
            import sqlite3

            from app.services.trading_engine import DB_PATH
            conn = sqlite3.connect(str(DB_PATH))
            c = conn.cursor()
            c.execute("SELECT details FROM signals WHERE trade_id = ?", (trade.get("id", ""),))
            row = c.fetchone()
            conn.close()
            if row and row[0]:
                details = row[0]
                import re
                for match in re.finditer(r"(\w+)=([-\d.]+)", details):
                    key, val = match.group(1), match.group(2)
                    try:
                        components[key] = float(val)
                    except ValueError:
                        pass
        except Exception:
            pass
        return components

    # ── Order Book Imbalance Signal ──────────────────────────────────

    async def get_orderbook_imbalance(self, ticker: str) -> dict[str, Any]:
        """Analyze order book for a Kalshi market to detect directional imbalance.

        Fetches the orderbook and computes:
          - bid/ask spread tightness
          - volume-weighted imbalance (bid volume vs ask volume)
          - mid-price vs last trade direction

        Returns:
            {
                "ticker": str,
                "yes_bid": int, "yes_ask": int,
                "no_bid": int, "no_ask": int,
                "spread_cents": int,
                "imbalance": float,      # [-1, 1] positive = bullish (yes-heavy)
                "confidence_adj": float,  # multiplier 0.7-1.3 for confidence
                "tight_spread": bool,
                "signal": str,           # "bullish", "bearish", "neutral"
            }
        """
        result = {
            "ticker": ticker,
            "yes_bid": 0, "yes_ask": 0,
            "no_bid": 0, "no_ask": 0,
            "spread_cents": 99,
            "imbalance": 0.0,
            "confidence_adj": 1.0,
            "tight_spread": False,
            "signal": "neutral",
        }

        try:
            data = await self.kalshi._get(f"/markets/{ticker}/orderbook")
            orderbook = data if isinstance(data, dict) else {}

            yes_bids = orderbook.get("yes", [])
            no_bids = orderbook.get("no", [])

            # Also get market snapshot for top-of-book
            mkt_data = await self.kalshi._get(f"/markets/{ticker}")
            market = mkt_data.get("market", mkt_data)

            yes_bid = market.get("yes_bid", 0) or 0
            yes_ask = market.get("yes_ask", 0) or 0
            no_bid = market.get("no_bid", 0) or 0
            no_ask = market.get("no_ask", 0) or 0

            result["yes_bid"] = yes_bid
            result["yes_ask"] = yes_ask
            result["no_bid"] = no_bid
            result["no_ask"] = no_ask

            # Spread analysis
            spread = yes_ask - yes_bid if yes_ask > 0 and yes_bid > 0 else 99
            result["spread_cents"] = spread
            result["tight_spread"] = spread <= 5

            # Volume imbalance from orderbook depth
            yes_bid_volume = sum(
                level.get("quantity", 0) or level.get("count", 0)
                for level in (yes_bids if isinstance(yes_bids, list) else [])
            )
            no_bid_volume = sum(
                level.get("quantity", 0) or level.get("count", 0)
                for level in (no_bids if isinstance(no_bids, list) else [])
            )

            total_volume = yes_bid_volume + no_bid_volume
            if total_volume > 0:
                # Positive = more yes bids (bullish), negative = more no bids (bearish)
                imbalance = (yes_bid_volume - no_bid_volume) / total_volume
                result["imbalance"] = round(imbalance, 4)

                if imbalance > 0.2:
                    result["signal"] = "bullish"
                elif imbalance < -0.2:
                    result["signal"] = "bearish"

            # Confidence adjustment based on spread + imbalance
            if result["tight_spread"]:
                result["confidence_adj"] = 1.1  # Tight spread = more reliable price
            elif spread > 15:
                result["confidence_adj"] = 0.7  # Wide spread = less reliable

            # Boost if imbalance aligns with our trade direction (caller checks this)
            abs_imbalance = abs(result["imbalance"])
            if abs_imbalance > 0.3:
                result["confidence_adj"] *= 1.15  # Strong imbalance = conviction

        except Exception as e:
            logger.debug("Orderbook imbalance fetch failed", ticker=ticker, error=str(e))

        return result

    # ── Weather City Code Map ─────────────────────────────────────
    WEATHER_CITY_CODES: dict[str, str] = {
        "NYC": "New York", "MIA": "Miami", "LAX": "Los Angeles", "CHI": "Chicago",
        "AUS": "Austin", "DFW": "Dallas", "PHL": "Philadelphia", "DEN": "Denver",
        "SEA": "Seattle", "SFO": "San Francisco", "DCA": "Washington DC",
        "SLC": "Salt Lake City", "ATL": "Atlanta", "HOU": "Houston", "BOS": "Boston",
    }

    def _enrich_weather_title(self, ticker: str, title: str) -> str:
        """Prefix weather market title with city name from ticker code, if missing."""
        title = title.replace("**", "")
        m = re.match(r'KX(?:HIGH|LOW)T?([A-Z]{2,4})-', ticker)
        if not m:
            return title
        city_code = m.group(1)
        city_name = KalshiAgent.WEATHER_CITY_CODES.get(city_code, "")
        if not city_name:
            return title
        if city_name.lower() in title.lower():
            return title
        return f"[{city_name}] {title}"

    def _record_weather_rejection(
        self,
        market: dict[str, Any],
        reason: str,
        *,
        stage: str,
        signal_source: str,
        near_miss: bool = False,
        details: str = "",
    ) -> None:
        ticker = market.get("ticker", "")
        self.engine.record_candidate_rejection(
            "weather",
            reason,
            ticker=ticker,
            signal_source=signal_source,
            stage=stage,
            near_miss=near_miss,
            details=details,
        )

    def _observed_weather_thresholds(self) -> dict[str, float]:
        min_edge = 0.08
        max_edge = 0.60
        min_price_cents = 15.0

        get_quality = getattr(self.engine, "get_recent_source_quality", None)
        if callable(get_quality):
            stats = get_quality("weather_observed_arbitrage", lookback_days=45)
        else:
            stats = {"trades": 0.0, "avg_pnl": 0.0, "win_rate": 0.0}
        if stats["trades"] >= 8:
            if stats["avg_pnl"] > 0.5 and stats["win_rate"] >= 0.55:
                min_edge = 0.07
                min_price_cents = 12.0
            elif stats["avg_pnl"] < -0.5 or stats["win_rate"] < 0.45:
                min_edge = 0.10
                min_price_cents = 18.0

        return {
            "min_edge": min_edge,
            "max_edge": max_edge,
            "min_price_cents": min_price_cents,
            "trades": stats["trades"],
            "avg_pnl": stats["avg_pnl"],
            "win_rate": stats["win_rate"],
        }

    def _record_resting_cancel(self, ticker: str) -> None:
        """Track canceled resting orders per ticker and alert on cancel storms."""
        now_ts = datetime.now(UTC).timestamp()
        window_seconds = self._cancel_alert_window_mins * 60
        series = self._resting_cancel_timestamps.setdefault(ticker, [])
        series.append(now_ts)
        series[:] = [ts for ts in series if ts >= now_ts - window_seconds]

        if len(series) >= self._cancel_alert_threshold:
            self.engine.log_event(
                "warning",
                f"Cancel storm detected for {ticker}: {len(series)} cancels in {self._cancel_alert_window_mins}m",
                strategy="monitor",
            )
            if self._auto_kill_on_cancel_storm and not self.engine.kill_switch:
                self.engine.kill_switch = True
                self.engine.log_event(
                    "critical",
                    f"Kill switch auto-activated due to cancel storm on {ticker}",
                    strategy="risk",
                )

    async def _reconcile_resting_orders(self) -> list[dict[str, Any]]:
        """Reconcile DB resting orders against broker open resting orders."""
        actions: list[dict[str, Any]] = []
        if self.engine.paper_mode:
            return actions

        db_resting = self.engine.get_resting_trades()
        if not db_resting:
            return actions

        try:
            client = get_kalshi_client()
            broker_resp = await client.get_orders(status="resting", limit=200)
            broker_orders = broker_resp.get("orders", []) if isinstance(broker_resp, dict) else []
            broker_open_ids = {
                o.get("order_id") or o.get("id")
                for o in broker_orders
                if (o.get("order_id") or o.get("id"))
            }

            now_dt = datetime.now(UTC)
            for rt in db_resting:
                order_id = rt.get("order_id", "")
                if not order_id or order_id.startswith("PAPER-"):
                    continue
                if order_id in broker_open_ids:
                    continue

                # If broker no longer reports it as open and it has aged enough,
                # treat as canceled to avoid ghost resting exposure.
                placed_dt = datetime.fromisoformat(rt["timestamp"].replace("Z", "+00:00"))
                age_mins = (now_dt - placed_dt).total_seconds() / 60.0
                if age_mins >= 2.0:
                    self.engine.update_trade_status(rt["id"], "canceled")
                    self.engine.log_event(
                        "warning",
                        f"Reconciliation canceled stale DB-resting order {order_id} (not open at broker)",
                        strategy=rt.get("strategy", "monitor"),
                    )
                    actions.append({"action": "reconcile_cancel", "ticker": rt.get("ticker", ""), "order_id": order_id})

            db_order_ids = {rt.get("order_id") for rt in db_resting if rt.get("order_id")}
            orphan_broker = [oid for oid in broker_open_ids if oid not in db_order_ids]
            if orphan_broker:
                self.engine.log_event(
                    "warning",
                    f"Reconciliation found {len(orphan_broker)} broker resting orders missing in DB",
                    strategy="monitor",
                    details=",".join(orphan_broker[:10]),
                )
        except Exception as e:
            logger.warning("Resting order reconciliation failed", error=str(e))

        return actions

    # ── Global Signal Ranking & Execution ──────────────────────────

    async def execute_ranked_signals(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Rank all candidates globally by historical quality, then execute
        the top ones.  Each cycle deploys up to 85% of AVAILABLE capital
        (bankroll − current net exposure).  The remaining 15% is held back
        for the next cycle to DCA into existing positions or catch new
        opportunities.  Over multiple cycles the system can deploy up to
        100% of bankroll.
        """
        if not candidates:
            return []
        self._refresh_performance_model_if_due()

        if (
            not self.engine.paper_mode
            and self.engine.require_ws_for_live
            and not self.ws.get_status().get("connected", False)
        ):
            self.engine.set_runtime_health(ws_healthy=False)
            self.engine.log_event(
                "blocked",
                "Global execution blocked: WebSocket disconnected and REQUIRE_WS_FOR_LIVE=true",
                strategy="risk",
            )
            return []

        # ── Deduplicate mutually exclusive outcomes ──────────────────────
        # Finance bracket markets: SP500 can only close in ONE bracket per expiry.
        # Sports game markets: a game has only ONE winner.
        # Keep only the highest edge×confidence candidate per exclusive group.
        deduped: list[dict[str, Any]] = []
        seen_exclusive: dict[str, dict[str, Any]] = {}

        for c in candidates:
            strategy = c.get("strategy", "")
            exclusive_key: str | None = None

            if strategy == "finance":
                fin_index = c.get("finance_index", "")
                fin_expiry = c.get("finance_expiry", "")
                if fin_index and fin_expiry:
                    # Both bracket AND threshold markets on the same index+expiry are
                    # mutually correlated — only trade the single highest-edge one.
                    market_type = "bracket" if c.get("is_bracket") else "threshold"
                    exclusive_key = f"finance_{market_type}_{fin_index}_{fin_expiry}"

            elif strategy == "sports":
                ticker = c.get("ticker", "")
                # Game prefix: everything before the last hyphen-separated suffix
                # e.g. KXMLSGAME-26FEB22SEACOL-SEA → KXMLSGAME-26FEB22SEACOL
                parts = ticker.rsplit("-", 1)
                if len(parts) == 2:
                    exclusive_key = f"sports_game_{parts[0]}"

            if exclusive_key:
                score = c.get("edge", 0) * c.get("confidence", 0)
                existing = seen_exclusive.get(exclusive_key)
                if existing is None:
                    seen_exclusive[exclusive_key] = c
                    deduped.append(c)
                else:
                    existing_score = existing.get("edge", 0) * existing.get("confidence", 0)
                    if score > existing_score:
                        deduped.remove(existing)
                        seen_exclusive[exclusive_key] = c
                        deduped.append(c)
            else:
                deduped.append(c)

        candidates = deduped

        quality_filtered: list[dict[str, Any]] = []
        blocked_for_quality = 0
        for candidate in candidates:
            quality = self.engine.evaluate_candidate_quality(candidate)
            candidate["quality_family"] = quality.get("family", "other")
            candidate["quality_score"] = quality.get("quality_score", 0.0)
            if quality.get("reasons"):
                candidate["quality_reasons"] = ", ".join(quality["reasons"])
            if not quality.get("allowed", False):
                blocked_for_quality += 1
                self.engine.record_candidate_rejection(
                    candidate.get("strategy", ""),
                    "quality_filter_blocked",
                    ticker=candidate.get("ticker", ""),
                    signal_source=candidate.get("signal_source", ""),
                    stage="ranking_quality",
                    near_miss=True,
                    details=", ".join(quality.get("reasons", [])),
                )
                continue
            quality_filtered.append(candidate)

        if blocked_for_quality:
            self.engine.log_event(
                "info",
                f"Quality filter: blocked {blocked_for_quality} low-quality candidates",
                strategy="risk",
            )

        candidates = quality_filtered
        candidates.sort(key=lambda c: c.get("quality_score", 0.0), reverse=True)

        # ── Skip tickers where we already hold an open position on the same side ──
        # Prevents re-entering the same market every cycle (econ CPI stacking, etc.)
        db_open_positions: list[dict[str, Any]] = []
        try:
            db_open_positions = self.engine.get_open_positions()
        except Exception:
            db_open_positions = []

        open_positions = db_open_positions
        try:
            kalshi_positions = await self.kalshi.get_positions()
            db_by_ticker = {p.get("ticker", ""): p for p in db_open_positions if p.get("ticker")}
            kalshi_open_positions: list[dict[str, Any]] = []
            mismatched_sides = 0
            for mp in kalshi_positions.get("market_positions", []):
                pos_qty = mp.get("position", 0)
                if pos_qty == 0:
                    continue
                ticker = mp.get("ticker", "")
                if not ticker:
                    continue
                kalshi_side = "yes" if pos_qty > 0 else "no"
                merged = dict(db_by_ticker.get(ticker, {}))
                if merged.get("side") and merged.get("side") != kalshi_side:
                    mismatched_sides += 1
                merged["ticker"] = ticker
                merged["side"] = kalshi_side
                kalshi_open_positions.append(merged)
            open_positions = kalshi_open_positions
            if mismatched_sides > 0:
                self.engine.log_event(
                    "warning",
                    f"Execution dedup: corrected {mismatched_sides} DB/Kalshi side mismatches using Kalshi positions",
                    strategy="risk",
                )
        except Exception:
            open_positions = db_open_positions

        open_keys: set[str] = {
            f"{p['ticker']}_{p['side']}" for p in open_positions
        }
        existing_tickers: set[str] = {
            p["ticker"] for p in open_positions if p.get("ticker")
        }

        pre_filter = len(candidates)
        candidates = [
            c for c in candidates
            if f"{c.get('ticker', '')}_{c.get('side', '')}" not in open_keys
        ]
        if pre_filter != len(candidates):
            self.engine.log_event(
                "info",
                f"Position dedup: skipped {pre_filter - len(candidates)} candidates already held",
            )

        # ── Bracket Correlation Guard ──
        # For bracket markets (weather/crypto/finance), prevent contradictory positions
        # on the same event. E.g., don't buy YES on ">80°F" and YES on "<75°F" for the same city/date.
        # Build a map of event_prefix → existing position direction
        open_event_directions: dict[str, str] = {}
        for p in open_positions:
            pticker = p.get("ticker", "")
            pside = p.get("side", "")
            # Extract event prefix: everything before the bracket specifier (-T45, -B36.5, etc.)
            bracket_m = re.match(r'(.+)-[TB]\d', pticker)
            if bracket_m:
                event_prefix = bracket_m.group(1)
                # Direction: YES on ">" means bullish, NO on ">" means bearish
                if "-T" in pticker:  # "T" = threshold/greater
                    direction = "bullish" if pside == "yes" else "bearish"
                else:  # "B" = below/between
                    direction = "bearish" if pside == "yes" else "bullish"
                open_event_directions[event_prefix] = direction

        bracket_filtered = []
        bracket_blocked = 0
        for c in candidates:
            cticker = c.get("ticker", "")
            cside = c.get("side", "")
            bracket_m = re.match(r'(.+)-[TB]\d', cticker)
            if bracket_m:
                event_prefix = bracket_m.group(1)
                existing_dir = open_event_directions.get(event_prefix)
                if existing_dir:
                    if "-T" in cticker:
                        new_dir = "bullish" if cside == "yes" else "bearish"
                    else:
                        new_dir = "bearish" if cside == "yes" else "bullish"
                    if new_dir != existing_dir:
                        bracket_blocked += 1
                        continue  # Skip contradictory bracket
            bracket_filtered.append(c)
        if bracket_blocked > 0:
            self.engine.log_event(
                "info",
                f"Bracket guard: blocked {bracket_blocked} contradictory bracket candidates",
            )
        candidates = bracket_filtered

        total_exposure = self.engine.get_total_exposure()
        effective_bankroll = self.engine.get_effective_bankroll()
        available_capital = max(0, effective_bankroll - total_exposure)
        cycle_budget = available_capital * 0.60  # deploy 60% of what's free (40% reserve)

        # HARD GUARD: refuse to trade if bankroll is critically low
        if effective_bankroll < self.engine.min_bankroll_to_trade:
            self.engine.log_event(
                "warning",
                f"Bankroll ${effective_bankroll:.2f} below minimum ${self.engine.min_bankroll_to_trade:.2f} — skipping all trades",
            )
            return []

        self.engine.log_event(
            "info",
            f"Global ranking: {len(candidates)} candidates, "
            f"exposure=${total_exposure:.2f}, available=${available_capital:.2f}, "
            f"cycle budget=${cycle_budget:.2f}",
        )

        # ── Skip tickers we already hold ──
        # Uses the Kalshi-first open_positions view built above.
        if existing_tickers:
            self.engine.log_event(
                "info",
                f"Existing positions ({len(existing_tickers)} tickers) — will skip duplicates",
            )

        # ── Per-city/event concentration limit ──
        # Track how many trades we execute per event group this cycle
        # Prevents dumping multiple bets on the same city/date
        MAX_TRADES_PER_EVENT = 2
        event_trade_count: dict[str, int] = {}
        family_trade_count: dict[str, int] = {}
        family_cycle_caps = {
            "weather_observed": 2,
            "weather_forecast": 0 if not self.engine.paper_mode else 1,
            "sports_single": 1,
            "sports_single_soccer": 0 if not self.engine.paper_mode else 1,
            "sports_parlay": 0,
            "crypto_momentum": 1,
            "finance_bracket": 1,
            "finance_threshold": 1,
            "econ": 1,
            "nba_props": 1,
            "other": 1,
        }
        max_trades_per_cycle = int(os.environ.get("MAX_TRADES_PER_CYCLE", "3"))
        if self.engine.get_effective_bankroll() <= 250:
            max_trades_per_cycle = min(max_trades_per_cycle, 1)

        traded: list[dict[str, Any]] = []
        for candidate in candidates:
            if cycle_budget <= 0:
                break
            if len(traded) >= max_trades_per_cycle:
                self.engine.log_event(
                    "info",
                    f"Global cycle cap reached ({max_trades_per_cycle} trades)",
                    strategy="risk",
                )
                break

            strategy = candidate.get("strategy", "")
            ticker = candidate.get("ticker", "")
            edge = candidate.get("edge", 0)
            confidence = candidate.get("confidence", 0)
            price_cents = candidate.get("price_cents", 0)
            sig_source = candidate.get("signal_source", "")
            quality_family = candidate.get("quality_family", "other")

            if not ticker or price_cents <= 0:
                continue

            if (
                not self.engine.paper_mode
                and strategy == "weather"
                and sig_source != "weather_observed_arbitrage"
            ):
                continue

            # ── Skip if we already hold this ticker ──
            if ticker in existing_tickers:
                continue

            family_cap = family_cycle_caps.get(quality_family, 1)
            if family_cap <= 0:
                continue
            if family_trade_count.get(quality_family, 0) >= family_cap:
                self.engine.log_event(
                    "info",
                    f"Family cycle cap: skipping {ticker} in {quality_family}",
                    strategy="risk",
                )
                continue

            # ── Per-event concentration guard ──
            # Group weather tickers by city+date (e.g. KXHIGHNYC-25MAR03)
            # Group other tickers by base event (everything before last hyphen segment)
            event_key = ticker.rsplit("-", 1)[0] if "-" in ticker else ticker
            if event_trade_count.get(event_key, 0) >= MAX_TRADES_PER_EVENT:
                self.engine.log_event(
                    "info",
                    f"Event concentration limit: skipping {ticker} (already {MAX_TRADES_PER_EVENT} trades on {event_key})",
                )
                continue

            # ── Block longshot bets ──
            # If our model says probability of winning is <40%, don't trade it
            # even if we think the market is mispriced. Longshots lose too often.
            our_prob = candidate.get("our_prob", 0)
            win_prob = our_prob
            if win_prob < 0.40:
                continue

            # Position sizing (respects per-strategy and per-ticker caps)
            # CRITICAL: signal_source differentiates forecast (small) vs observed arb (full)
            count = self.engine.calculate_position_size(
                strategy=strategy,
                edge=edge,
                price_cents=price_cents,
                confidence=confidence,
                ticker=ticker,
                signal_source=sig_source,
            )
            if count <= 0:
                continue

            cost_estimate = count * price_cents / 100.0

            # PRE-EXECUTION SANITY CHECK: reject if cost > 3% of bankroll
            if cost_estimate > effective_bankroll * 0.03:
                self.engine.log_event(
                    "warning",
                    f"Sanity check: {ticker} cost ${cost_estimate:.2f} exceeds 3% of bankroll ${effective_bankroll:.2f} — reducing",
                )
                max_cost = effective_bankroll * 0.03
                count = int(max_cost / (price_cents / 100.0))
                if count <= 0:
                    continue
                cost_estimate = count * price_cents / 100.0

            if cost_estimate > cycle_budget:
                # Reduce count to fit budget
                count = int(cycle_budget / (price_cents / 100.0))
                if count <= 0:
                    continue

            # Generate pre-trade thesis via GPT-5.2 (non-blocking best-effort)
            thesis = ""
            try:
                thesis = await self.trade_analyzer.generate_thesis(candidate)
            except Exception:
                pass

            trade = await self.engine.execute_trade(
                strategy=strategy,
                ticker=ticker,
                side=candidate.get("side", "yes"),
                count=count,
                price_cents=price_cents,
                our_prob=candidate.get("our_prob", 0),
                kalshi_prob=candidate.get("kalshi_prob", 0),
                market_title=candidate.get("title", ""),
                signal_source=candidate.get("signal_source", ""),
                signal_id=candidate.get("signal_id", ""),
                notes=candidate.get("notes", ""),
                thesis=thesis,
            )

            if trade and trade.get("status") not in ("blocked", "error", "timeout"):
                actual_cost = trade.get("cost", 0) + trade.get("fee", 0)
                cycle_budget -= actual_cost
                candidate["trade"] = trade
                traded.append(candidate)
                # Track event concentration
                event_trade_count[event_key] = event_trade_count.get(event_key, 0) + 1
                family_trade_count[quality_family] = family_trade_count.get(quality_family, 0) + 1

                # Discord alert + SSE event for trade execution
                try:
                    side = candidate.get("side", "yes").upper()
                    asyncio.create_task(send_discord_notification(
                        title=f"🔔 Trade: {side} {count}x {ticker} @ {price_cents}c",
                        message=(
                            f"**Strategy:** {strategy}\n"
                            f"**Edge:** {edge:.1%} | **Confidence:** {confidence:.1%}\n"
                            f"**Cost:** ${actual_cost:.2f}\n"
                            f"**Title:** {candidate.get('title', '')[:80]}\n"
                            f"{thesis[:120] if thesis else ''}"
                        ),
                        color=0x2ecc71,  # green
                    ))
                except Exception:
                    pass
                try:
                    get_event_bus().publish("trade", {
                        "ticker": ticker, "side": candidate.get("side", "yes"),
                        "count": count, "price_cents": price_cents,
                        "strategy": strategy, "edge": round(edge, 4),
                        "confidence": round(confidence, 4), "cost": actual_cost,
                        "title": candidate.get("title", "")[:80],
                    })
                except Exception:
                    pass

        self.engine.log_event(
            "info",
            f"Global execution: {len(traded)}/{len(candidates)} traded, "
            f"cycle budget remaining=${cycle_budget:.2f}",
        )
        return traded

    # ── Weather Strategy ───────────────────────────────────────────

    async def run_weather_cycle(self) -> list[dict[str, Any]]:
        """
        Run one weather strategy cycle:
        1. Fetch Kalshi weather markets
        2. Fetch forecasts from all sources
        3. Build consensus
        4. Generate signals
        5. Execute trades (paper or live)
        """
        self.engine.log_event("info", "Weather cycle starting", strategy="weather")
        self.engine.start_cycle("weather")
        results = []
        cycle_started_at = datetime.now(UTC).isoformat()

        if not self.engine.strategy_enabled.get("weather", True):
            self.engine.log_event("info", "Weather strategy disabled", strategy="weather")
            return results

        try:
            # Dynamically scan ALL open Kalshi weather markets across all cities
            weather_markets = await self.scanner.scan_weather_markets()
            self.engine.log_event(
                "info",
                f"Found {len(weather_markets)} weather markets",
                strategy="weather",
            )
            scan_stats = self.scanner.get_weather_scan_stats()
            self.engine.log_event(
                "info",
                "Weather quote hydration: "
                f"attempted={scan_stats.get('hydration_attempted', 0)}, "
                f"detail_updates={scan_stats.get('detail_updates', 0)}, "
                f"rescued_two_sided_asks={scan_stats.get('rescued_two_sided_asks', 0)}, "
                f"rescued_full_quotes={scan_stats.get('rescued_full_quotes', 0)}",
                strategy="weather",
            )

            # Group by (city_code, target_date) so each group gets the right forecast
            import re as _re
            from datetime import date as _date
            from datetime import datetime as _datetime
            today_str = _datetime.now(UTC).date().isoformat()
            by_city_date: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for m in weather_markets:
                city_code = m.get("weather", {}).get("city_code", "")
                if not city_code:
                    continue
                # Parse target date from the ticker (e.g. KXHIGHDEN-26FEB19-B36.5 → 2026-02-19)
                # close_time is UTC and can be off by one day; ticker date is authoritative
                ticker = m.get("ticker", "")
                try:
                    date_match = _re.search(r'-(\d{2})([A-Z]{3})(\d{2})-', ticker)
                    if date_match:
                        target_date_str = _datetime.strptime(
                            date_match.group(1) + date_match.group(2) + date_match.group(3), "%y%b%d"
                        ).date().isoformat()
                    else:
                        target_date_str = today_str
                except Exception:
                    target_date_str = today_str
                by_city_date.setdefault((city_code, target_date_str), []).append(m)

            # Pre-fetch NWS observations for same-day cities (for observed arbitrage)
            obs_by_city: dict[str, dict[str, Any]] = {}
            same_day_cities = {ck for ck, ds in by_city_date.keys() if ds == today_str}
            for city_key in same_day_cities:
                try:
                    obs = await self.weather.get_current_observations(city_key)
                    if obs:
                        obs_by_city[city_key] = obs
                except Exception as e:
                    logger.debug("NWS observation fetch failed", city=city_key, error=str(e))

            # Pre-fetch forecasts for all city/date combos
            forecasts_by_city_date: dict[tuple[str, str], dict[str, Any]] = {}
            for city_key, target_date_str in by_city_date.keys():
                try:
                    target_dt = _datetime.fromisoformat(target_date_str).date()
                    # Hit the 8-API consensus model
                    fcasts = await self.weather.get_all_forecasts(city_key, target_dt)
                    if fcasts and "sources" in fcasts and fcasts["sources"]:
                        forecasts_by_city_date[(city_key, target_date_str)] = fcasts
                        self.engine.log_event(
                            "info",
                            f"Fetched {len(fcasts['sources'])} forecast sources for {city_key} on {target_date_str}",
                            strategy="weather",
                        )
                except Exception as e:
                    logger.debug("Forecast fetch failed", city=city_key, date=target_date_str, error=str(e))

            for i, ((city_key, target_date_str), city_markets) in enumerate(by_city_date.items()):
                if city_key not in CITY_CONFIGS:
                    self.engine.log_event(
                        "warning",
                        f"Unknown city code {city_key}, skipping",
                        strategy="weather",
                    )
                    continue

                is_same_day = (target_date_str == today_str)
                obs = obs_by_city.get(city_key) if is_same_day else None
                forecasts = forecasts_by_city_date.get((city_key, target_date_str))

                # ── Path A: Near-certainty observation path (Same Day) ──
                # If we have real observed data, evaluate markets from actuals first
                if is_same_day and obs:
                    for market in city_markets:
                        # Will return None if the outcome is not yet deterministic
                        candidate = await self._evaluate_weather_market_observed(market, obs)
                        if candidate:
                            results.append(candidate)
                            continue  # If we got an observed signal, skip forecast evaluation
                        
                        # If observation didn't yield a near-certain trade, fall back to forecast
                        if forecasts:
                            # Build all_city_forecasts for spatial correlation
                            all_city_fcasts = {k[0]: v for k, v in forecasts_by_city_date.items() if k[1] == target_date_str}
                            candidate_forecast = await self._evaluate_weather_market(market, forecasts, all_city_forecasts=all_city_fcasts)
                            if candidate_forecast:
                                results.append(candidate_forecast)
                
                # ── Path B: Forecast consensus path (Future or Same Day early) ──
                elif forecasts:
                    # Build all_city_forecasts for spatial correlation
                    all_city_fcasts = {k[0]: v for k, v in forecasts_by_city_date.items() if k[1] == target_date_str}
                    for market in city_markets:
                        candidate = await self._evaluate_weather_market(market, forecasts, all_city_forecasts=all_city_fcasts)
                        if candidate:
                            results.append(candidate)
                
                # ── Path C: No data ──
                else:
                    self.engine.log_event(
                        "info",
                        f"{city_key}: no observations or forecasts available for {target_date_str} — skipping",
                        strategy="weather",
                    )
                    continue

        except Exception as e:
            self.engine.log_event("error", f"Weather cycle failed: {e}", strategy="weather")
            logger.error("Weather cycle error", error=str(e))

        # Enrich titles with city names and tag strategy
        for r in results:
            if r.get("action") != "skip":
                r["title"] = self._enrich_weather_title(r.get("ticker", ""), r.get("title", ""))
                r["strategy"] = "weather"

        self.engine.log_event(
            "info",
            f"Weather cycle complete: {len(results)} candidates",
            strategy="weather",
        )
        rejection_summary = self.engine.get_rejection_summary_since("weather", cycle_started_at, limit=6)
        if rejection_summary:
            summary_text = ", ".join(f"{row['reason']}={row['count']}" for row in rejection_summary)
            self.engine.log_event(
                "info",
                f"Weather rejection summary: {summary_text}",
                strategy="weather",
            )
        return results

    async def _evaluate_weather_market(
        self,
        market: dict[str, Any],
        forecasts: dict[str, Any],
        all_city_forecasts: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Evaluate a single weather market against consensus forecast."""
        yes_ask = market.get("yes_ask", 0)
        no_ask = market.get("no_ask", 0)
        yes_bid = market.get("yes_bid", 0)
        no_bid = market.get("no_bid", 0)

        # Safety filter: Max Spread Limit = 5¢
        if yes_ask > 0 and yes_bid > 0 and (yes_ask - yes_bid) > 5:
            self._record_weather_rejection(market, "yes_spread_too_wide", stage="forecast_precheck", signal_source="weather_consensus")
            return None
        if no_ask > 0 and no_bid > 0 and (no_ask - no_bid) > 5:
            self._record_weather_rejection(market, "no_spread_too_wide", stage="forecast_precheck", signal_source="weather_consensus")
            return None

        strike_type = market.get("strike_type", "")
        floor_strike = market.get("floor_strike")
        cap_strike = market.get("cap_strike")

        # Scanner stores weather-specific info in a sub-dict
        weather_info = market.get("weather", {})
        if not strike_type and weather_info:
            strike_type = weather_info.get("strike_type", "")
        if floor_strike is None and weather_info:
            floor_strike = weather_info.get("floor_strike")
        if cap_strike is None and weather_info:
            cap_strike = weather_info.get("cap_strike")

        # Parse strikes to float
        try:
            floor_strike = float(floor_strike) if floor_strike is not None else None
        except (ValueError, TypeError):
            floor_strike = None
        try:
            cap_strike = float(cap_strike) if cap_strike is not None else None
        except (ValueError, TypeError):
            cap_strike = None

        if not yes_ask and not no_ask:
            self._record_weather_rejection(market, "no_two_sided_market", stage="forecast_precheck", signal_source="weather_consensus")
            return None

        # Calculate Maker Prices (Pennying the bid)
        yes_maker_price = min(yes_bid + 1, yes_ask) if yes_bid > 0 else yes_ask
        no_maker_price = min(no_bid + 1, no_ask) if no_bid > 0 else no_ask

        # Safety filter: skip effectively-settled markets where either side is <=3c
        # (means the outcome is near-certain, no real edge to capture)
        if yes_maker_price <= 3 or no_maker_price <= 3:
            self._record_weather_rejection(market, "near_settled_market", stage="forecast_precheck", signal_source="weather_consensus")
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        # These are long-shot bets with high loss rates even with edge
        if min(yes_maker_price, no_maker_price) < 10:
            self._record_weather_rejection(market, "cheap_longshot_filtered", stage="forecast_precheck", signal_source="weather_consensus")
            return None

        # Time-aware trading rules
        close_time = market.get("close_time", "")
        city_code = market.get("weather", {}).get("city_code", "")
        if close_time and city_code:
            try:
                from datetime import datetime as _dt
                import pytz
                
                close_dt = _dt.fromisoformat(close_time.replace("Z", "+00:00"))
                now = _dt.now(UTC)
                hours_until_close = (close_dt - now).total_seconds() / 3600
                
                # Skip markets closing within 2 hours
                if hours_until_close < 2:
                    self._record_weather_rejection(market, "close_too_soon", stage="forecast_time_gate", signal_source="weather_consensus")
                    return None
                
                # Get city timezone for local time
                city_config = CITY_CONFIGS.get(city_code, {})
                tz_name = city_config.get("timezone", "America/New_York")
                city_tz = pytz.timezone(tz_name)
                local_time = now.astimezone(city_tz)
                local_hour = local_time.hour
                
                # Early day (<12pm local): Skip expensive contracts (>90¢)
                # Money is better spent elsewhere when outcome is uncertain
                if local_hour < 12:
                    if yes_maker_price > 90 or no_maker_price > 90:
                        self._record_weather_rejection(market, "late_day_expensive_market", stage="forecast_time_gate", signal_source="weather_consensus")
                        return None
                
                # Late day (>3pm local): Only take locked outcomes
                # If temp already hit/locked, take 95¢+ bets (guaranteed 5% return)
                # Otherwise skip - too late for probabilistic bets
                elif local_hour >= 15:
                    # This will be handled by observed data path
                    # For forecast-based evaluation, skip late-day unless very cheap
                    if min(yes_maker_price, no_maker_price) > 50:
                        self._record_weather_rejection(market, "late_day_forecast_disabled", stage="forecast_time_gate", signal_source="weather_consensus")
                        return None

                # LOW TEMP GUARD: During daytime (10 AM - 6 PM), the overnight
                # minimum hasn't happened yet. Forecast-based low temp bets are
                # unreliable — tonight could drop much further than forecasted.
                # Skip low temp forecast bets entirely during the day.
                market_type_check = weather_info.get("market_type", "high_temp")
                if market_type_check == "low_temp" and 10 <= local_hour <= 18:
                    self._record_weather_rejection(market, "daytime_low_temp_forecast_disabled", stage="forecast_time_gate", signal_source="weather_consensus")
                    return None  # Let observed-data path handle same-day low temp
                        
            except (ValueError, TypeError, Exception):
                # If time parsing fails, apply conservative 2hr close filter
                if hours_until_close < 2:
                    self._record_weather_rejection(market, "close_too_soon", stage="forecast_time_gate", signal_source="weather_consensus")
                    return None

        # Safety filter: skip very low volume markets (< 50 contracts traded)
        if market.get("volume", 0) < 50:
            self._record_weather_rejection(market, "low_volume", stage="forecast_precheck", signal_source="weather_consensus")
            return None

        # Build consensus for this specific market's strike structure
        market_type = weather_info.get("market_type", "high_temp")
        
        consensus = self.weather.build_consensus(
            forecasts,
            strike_type=strike_type,
            floor_strike=floor_strike,
            cap_strike=cap_strike,
            market_type=market_type,
            all_city_forecasts=all_city_forecasts,
        )
        if "error" in consensus:
            self._record_weather_rejection(market, "consensus_unavailable", stage="forecast_consensus", signal_source="weather_consensus")
            return None

        # Generate signal
        wx_thresh = self.adaptive.get_thresholds("weather")
        signal = self.weather.generate_signal(
            consensus,
            kalshi_yes_price=yes_maker_price,
            kalshi_no_price=no_maker_price,
            min_edge=wx_thresh["min_edge"],
            min_confidence=wx_thresh["min_confidence"],
            min_sources=2,
        )

        if not signal:
            self._record_weather_rejection(
                market,
                "signal_below_threshold",
                stage="forecast_signal",
                signal_source="weather_consensus",
                near_miss=(consensus.get("confidence", 0) >= max(0.2, wx_thresh["min_confidence"] - 0.1)),
                details=json.dumps({
                    "confidence": consensus.get("confidence", 0),
                    "source_count": consensus.get("source_count", 0),
                    "mean_temp": consensus.get("mean_low_f") if market_type == "low_temp" else consensus.get("mean_high_f"),
                }),
            )
            return None

        # ── Order Book Imbalance / Smart Money Flow ──
        # Fetch the orderbook to check if "smart money" is against us.
        # If we are buying YES, we want to see if there's massive resistance (huge asks) 
        # or if the flow is with us (huge bids pushing price up).
        try:
            ob = await self.kalshi.get_orderbook(market["ticker"])
            if ob and "orderbook" in ob:
                book = ob["orderbook"]
                side_key = "yes" if signal["side"] == "yes" else "no"
                
                # Sum resting liquidity within 5 cents of the current price
                bids = book.get(side_key, {}).get("bids", [])
                asks = book.get(side_key, {}).get("asks", [])
                
                maker_price = yes_maker_price if signal["side"] == "yes" else no_maker_price
                
                bid_vol = sum(b[1] for b in bids if maker_price - b[0] <= 5)
                ask_vol = sum(a[1] for a in asks if a[0] - maker_price <= 5)
                
                # If there is huge resistance (asks > 3x bids) and we are buying,
                # we might be stepping in front of a freight train. 
                # Require a higher edge.
                if ask_vol > bid_vol * 3 and ask_vol > 500:
                    self.engine.log_event("info", f"High resistance for {market['ticker']} {signal['side'].upper()}: {ask_vol} asks vs {bid_vol} bids", strategy="weather")
                    # Reduce confidence to potentially skip
                    signal["confidence"] -= 0.2
                    if signal["edge"] < wx_thresh["min_edge"] + 0.05:
                        self._record_weather_rejection(market, "orderbook_resistance", stage="forecast_orderbook", signal_source="weather_consensus", near_miss=True)
                        return None # Need +5% extra edge to fight the flow
                        
                # Conversely, if flow is heavily with us, we can boost confidence
                elif bid_vol > ask_vol * 3 and bid_vol > 500:
                    signal["confidence"] = min(1.0, signal["confidence"] + 0.1)
        except Exception as e:
            # Ignore orderbook errors, proceed with original signal
            pass

        if signal["confidence"] < wx_thresh["min_confidence"]:
            self._record_weather_rejection(market, "confidence_after_orderbook_below_floor", stage="forecast_signal", signal_source="weather_consensus", near_miss=True)
            return None

        # Record signal (with enriched city title)
        label = consensus.get("label", "")
        enriched_title = self._enrich_weather_title(market["ticker"], market["title"])
        signal_id = self.engine.record_signal(
            strategy="weather",
            ticker=market["ticker"],
            side=signal["side"],
            our_prob=signal["our_prob"],
            kalshi_prob=signal["kalshi_prob"],
            market_title=enriched_title,
            confidence=signal["confidence"],
            signal_source="weather_consensus",
            details=json.dumps({
                "consensus_temp": consensus.get('mean_low_f') if market_type == 'low_temp' else consensus.get('mean_high_f'),
                "label": label,
                "source_count": signal['source_count'],
                "market_type": market_type,
                "source_forecasts": consensus.get("sources", []),
            }),
        )

        # Calculate position size — FORECAST gets reduced sizing vs observed arbitrage
        price_cents = yes_maker_price if signal["side"] == "yes" else no_maker_price
        count = self.engine.calculate_position_size(
            strategy="weather",
            edge=signal["edge"],
            price_cents=price_cents,
            confidence=signal["confidence"],
            ticker=market["ticker"],
            signal_source="weather_consensus",
        )

        if count <= 0:
            self._record_weather_rejection(market, "position_size_zero", stage="forecast_sizing", signal_source="weather_consensus", near_miss=True)
            return {**signal, "ticker": market["ticker"], "action": "skip", "reason": "position_size_zero"}

        # Return candidate for ranking (execution happens in the cycle)
        return {
            **signal,
            "ticker": market["ticker"],
            "title": market["title"],
            "count": count,
            "price_cents": price_cents,
            "signal_id": signal_id,
            "signal_source": "weather_consensus",
        }

    async def _evaluate_weather_market_observed(
        self,
        market: dict[str, Any],
        obs: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Evaluate a same-day weather market using ACTUAL NWS observations.

        This is the real-time arbitrage path. Instead of forecasting what the
        temperature will be, we read what it HAS been from the NWS station.

        Logic:
        - If observed_high > threshold: "above threshold" is near-certain YES
        - If observed_high < threshold AND day is late (>3 PM local): near-certain NO
        - If outcome is not yet deterministic (early in day, close to threshold): skip
        - Minimum 3 observations required (data quality gate)
        """
        import zoneinfo as _zi
        from datetime import datetime as _dt

        ticker = market.get("ticker", "")
        title = market.get("title", "")
        yes_ask = market.get("yes_ask", 0)
        no_ask = market.get("no_ask", 0)
        weather_info = market.get("weather", {})
        market_type = weather_info.get("market_type", "high_temp")
        strike_type = weather_info.get("strike_type", "") or market.get("strike_type", "")

        # Parse strikes
        floor_strike = weather_info.get("floor_strike") or market.get("floor_strike")
        cap_strike = weather_info.get("cap_strike") or market.get("cap_strike")
        try:
            floor_strike = float(floor_strike) if floor_strike is not None else None
        except (ValueError, TypeError):
            floor_strike = None
        try:
            cap_strike = float(cap_strike) if cap_strike is not None else None
        except (ValueError, TypeError):
            cap_strike = None

        if not yes_ask and not no_ask:
            self._record_weather_rejection(market, "no_two_sided_market", stage="observed_precheck", signal_source="weather_observed_arbitrage")
            return None

        # Skip markets where one side is at near-zero — already resolved or no real liquidity.
        # yes_ask=1, no_ask=100 means Kalshi already knows the answer; there's nothing to buy.
        if yes_ask <= 5 or no_ask <= 5:
            self._record_weather_rejection(market, "near_settled_market", stage="observed_precheck", signal_source="weather_observed_arbitrage")
            return None
        # Both sides must have real two-sided liquidity (not 100¢ = no sellers)
        if yes_ask >= 97 or no_ask >= 97:
            self._record_weather_rejection(market, "one_sided_liquidity", stage="observed_precheck", signal_source="weather_observed_arbitrage")
            return None
            
        # Safety filter: Max Spread Limit = 5¢
        yes_bid = market.get("yes_bid", 0)
        no_bid = market.get("no_bid", 0)
        if yes_ask > 0 and yes_bid > 0 and (yes_ask - yes_bid) > 5:
            self._record_weather_rejection(market, "yes_spread_too_wide", stage="observed_precheck", signal_source="weather_observed_arbitrage")
            return None
        if no_ask > 0 and no_bid > 0 and (no_ask - no_bid) > 5:
            self._record_weather_rejection(market, "no_spread_too_wide", stage="observed_precheck", signal_source="weather_observed_arbitrage")
            return None

        # Calculate Maker Prices (Pennying the bid)
        yes_maker_price = min(yes_bid + 1, yes_ask) if yes_bid > 0 else yes_ask
        no_maker_price = min(no_bid + 1, no_ask) if no_bid > 0 else no_ask

        if market.get("volume", 0) < 50:
            self._record_weather_rejection(market, "low_volume", stage="observed_precheck", signal_source="weather_observed_arbitrage")
            return None

        # Skip markets closing within 30 min (too late to act)
        close_time = market.get("close_time", "")
        if close_time:
            try:
                close_dt = _dt.fromisoformat(close_time.replace("Z", "+00:00"))
                mins_left = (close_dt - _dt.now(UTC)).total_seconds() / 60
                if mins_left < 30:
                    self._record_weather_rejection(market, "close_too_soon", stage="observed_time_gate", signal_source="weather_observed_arbitrage")
                    return None
            except (ValueError, TypeError):
                pass

        # Need at least 3 hourly readings for data quality
        if obs.get("obs_count", 0) < 3:
            self._record_weather_rejection(market, "insufficient_observations", stage="observed_data", signal_source="weather_observed_arbitrage")
            return None

        observed_high = obs.get("observed_high_f")
        observed_low = obs.get("observed_low_f")
        obs_count = obs.get("obs_count", 0)

        # Determine local time of day to assess if the day's high is likely set
        city_key = weather_info.get("city_code", "")
        city_cfg = CITY_CONFIGS.get(city_key, {})
        city_tz_str = city_cfg.get("tz", "America/New_York")
        try:
            city_tz = _zi.ZoneInfo(city_tz_str)
            local_hour = _dt.now(city_tz).hour
        except Exception:
            local_hour = _dt.now(UTC).hour

        # Daily high is typically reached between noon and 4 PM local time.
        # After 4 PM we can be confident the high is set for the day.
        # Before noon, the high could still rise significantly.
        HIGH_LIKELY_SET_HOUR = 16   # 4 PM local
        HIGH_POSSIBLY_SET_HOUR = 13  # 1 PM local — use with larger margin

        our_prob_yes: float | None = None
        certainty_label = ""

        if market_type in ("high_temp",) and observed_high is not None:
            if strike_type == "greater" and floor_strike is not None:
                # "Will high be > X°F?"
                if observed_high > floor_strike:
                    # Already exceeded — YES is near-certain
                    our_prob_yes = 0.97
                    certainty_label = f"obs_high={observed_high}°F > threshold={floor_strike}°F ✓CONFIRMED"
                elif local_hour >= HIGH_LIKELY_SET_HOUR and observed_high < floor_strike - 3.0:
                    # Day is done, high never reached threshold — NO is near-certain
                    our_prob_yes = 0.03
                    certainty_label = f"obs_high={observed_high}°F < threshold={floor_strike}°F day_done ✓CONFIRMED"
                elif local_hour >= HIGH_POSSIBLY_SET_HOUR and observed_high < floor_strike - 6.0:
                    # Afternoon, high is 6°F below threshold — very unlikely to reach it
                    our_prob_yes = 0.08
                    certainty_label = f"obs_high={observed_high}°F << threshold={floor_strike}°F afternoon"

            elif strike_type == "less" and cap_strike is not None:
                # "Will high be < X°F?"
                if observed_high >= cap_strike:
                    # Already exceeded cap — NO is near-certain (YES = false)
                    our_prob_yes = 0.03
                    certainty_label = f"obs_high={observed_high}°F >= cap={cap_strike}°F ✓CONFIRMED NO"
                elif local_hour >= HIGH_LIKELY_SET_HOUR and observed_high < cap_strike - 3.0:
                    # Day done, high stayed below cap — YES is near-certain
                    our_prob_yes = 0.97
                    certainty_label = f"obs_high={observed_high}°F < cap={cap_strike}°F day_done ✓CONFIRMED YES"
                elif local_hour >= HIGH_POSSIBLY_SET_HOUR and observed_high < cap_strike - 6.0:
                    # Afternoon, high is well below cap — likely YES
                    our_prob_yes = 0.90
                    certainty_label = f"obs_high={observed_high}°F << cap={cap_strike}°F afternoon"

        elif market_type in ("low_temp",) and observed_low is not None:
            # CRITICAL: Low temperature = overnight minimum (typically 3-6 AM).
            # During the daytime (10 AM+), temps are RISING, so the current observed_low
            # is just the minimum SO FAR. But the low hasn't been finalized — tonight
            # the temp could drop much further. We can NEVER confirm "low > X" during
            # the daytime because the overnight drop hasn't happened yet.
            #
            # What we CAN confirm during the day:
            # - "low < X" is YES if observed_low already dropped below X (it happened)
            # - "low > X" is NO if observed_low already dropped below X (it already went lower)
            #
            # What we can ONLY confirm after overnight:
            # - "low > X" is YES (the low never went below X) — only after ~8 AM next day
            # - "low < X" is NO (the low stayed above X all night) — only after ~8 AM next day

            LOW_CONFIRMED_HOUR = 8  # After 8 AM, overnight low is set
            LOW_MARGIN = 8.0  # Need 8°F buffer for daytime low temp estimates

            if strike_type == "greater" and floor_strike is not None:
                # "Will the low be > X°F?"
                if observed_low < floor_strike:
                    # Low already dropped below threshold — NO is confirmed
                    our_prob_yes = 0.03
                    certainty_label = f"obs_low={observed_low}°F < threshold={floor_strike}°F ✓CONFIRMED NO (already breached)"
                elif local_hour >= LOW_CONFIRMED_HOUR and local_hour <= 10:
                    # Early morning: overnight low is likely set (warmup has begun)
                    if observed_low > floor_strike + LOW_MARGIN:
                        our_prob_yes = 0.90
                        certainty_label = f"obs_low={observed_low}°F > threshold={floor_strike}°F + {LOW_MARGIN}°F margin, post-overnight"
                    elif observed_low > floor_strike + 3.0:
                        our_prob_yes = 0.75
                        certainty_label = f"obs_low={observed_low}°F > threshold={floor_strike}°F + 3°F margin, post-overnight"
                # During daytime (after 10 AM): DO NOT confirm YES — tonight could still drop

            elif strike_type == "less" and cap_strike is not None:
                # "Will the low be < X°F?"
                if observed_low < cap_strike:
                    # Low already dropped below cap — YES is confirmed
                    our_prob_yes = 0.93
                    certainty_label = f"obs_low={observed_low}°F < cap={cap_strike}°F ✓CONFIRMED YES (already breached)"
                elif local_hour >= LOW_CONFIRMED_HOUR and local_hour <= 10:
                    # Early morning: overnight low is set, and it stayed above cap
                    if observed_low >= cap_strike + LOW_MARGIN:
                        our_prob_yes = 0.05
                        certainty_label = f"obs_low={observed_low}°F >= cap={cap_strike}°F + {LOW_MARGIN}°F margin, post-overnight ✓CONFIRMED NO"
                # During daytime: DO NOT confirm NO — tonight could still drop below cap

            elif strike_type == "between" and floor_strike is not None and cap_strike is not None:
                # "Will low be between X and Y°F?"
                if observed_low < floor_strike:
                    # Already dropped below the floor — NO (below range)
                    our_prob_yes = 0.03
                    certainty_label = f"obs_low={observed_low}°F < floor={floor_strike}°F ✓BELOW RANGE"
                elif observed_low > cap_strike + LOW_MARGIN and local_hour >= LOW_CONFIRMED_HOUR and local_hour <= 10:
                    # Post-overnight, low stayed well above cap — NO (above range)
                    our_prob_yes = 0.05
                    certainty_label = f"obs_low={observed_low}°F > cap={cap_strike}°F + margin, post-overnight ✓ABOVE RANGE"

        if our_prob_yes is None:
            # Outcome not yet deterministic — skip, don't guess
            self._record_weather_rejection(
                market,
                "outcome_not_deterministic_yet",
                stage="observed_logic",
                signal_source="weather_observed_arbitrage",
                near_miss=True,
                details=f"obs_count={obs_count} local_hour={local_hour}",
            )
            return None

        # Calculate edge
        kalshi_yes_prob = yes_maker_price / 100.0 if yes_maker_price > 0 else 0
        kalshi_no_prob = no_maker_price / 100.0 if no_maker_price > 0 else 0
        our_prob_no = 1.0 - our_prob_yes

        yes_edge = our_prob_yes - kalshi_yes_prob if kalshi_yes_prob > 0 else 0
        no_edge = our_prob_no - kalshi_no_prob if kalshi_no_prob > 0 else 0

        if yes_edge >= no_edge and yes_edge > 0:
            side, edge, our_prob, kalshi_prob, price_cents = "yes", yes_edge, our_prob_yes, kalshi_yes_prob, yes_maker_price
        elif no_edge > yes_edge and no_edge > 0:
            side, edge, our_prob, kalshi_prob, price_cents = "no", no_edge, our_prob_no, kalshi_no_prob, no_maker_price
        else:
            return None

        # Require meaningful edge — but observed/weather-confirmed trades can justify
        # larger mispricings than the forecast path, especially on a slow retail book.
        observed_rules = self._observed_weather_thresholds()
        min_observed_edge = observed_rules["min_edge"]
        max_observed_edge = observed_rules["max_edge"]
        if edge < min_observed_edge or edge > max_observed_edge:
            self._record_weather_rejection(
                market,
                "observed_edge_outside_band",
                stage="observed_edge",
                signal_source="weather_observed_arbitrage",
                near_miss=edge >= max(0.05, min_observed_edge - 0.03),
                details=f"edge={edge:.4f} min_edge={min_observed_edge:.4f} max_edge={max_observed_edge:.4f}",
            )
            return None

        # Require Kalshi price >= 15¢ on our side — still avoids ultra-cheap near-settled
        # contracts while allowing slightly earlier entry on confirmed observation moves.
        if price_cents < observed_rules["min_price_cents"]:
            self._record_weather_rejection(
                market,
                "observed_price_too_low",
                stage="observed_price",
                signal_source="weather_observed_arbitrage",
                near_miss=price_cents >= max(8, observed_rules["min_price_cents"] - 3),
                details=f"price_cents={price_cents} min_price_cents={observed_rules['min_price_cents']}",
            )
            return None

        # High confidence since this is based on actuals, not forecast
        confidence = 0.95 if our_prob_yes >= 0.90 or our_prob_yes <= 0.10 else 0.80

        enriched_title = self._enrich_weather_title(ticker, title)

        self.engine.log_event(
            "info",
            f"OBSERVED ARBITRAGE: {ticker} {side} edge={edge:.1%} (our={our_prob:.0%} vs kalshi={kalshi_prob:.0%}) "
            f"obs_count={obs_count} {certainty_label} | {enriched_title[:60]}",
            strategy="weather",
        )

        signal_id = self.engine.record_signal(
            strategy="weather",
            ticker=ticker,
            side=side,
            our_prob=our_prob,
            kalshi_prob=kalshi_prob,
            market_title=enriched_title,
            confidence=confidence,
            signal_source="weather_observed_arbitrage",
            details=f"{certainty_label} obs_count={obs_count} local_hour={local_hour}",
        )

        count = self.engine.calculate_position_size(
            strategy="weather",
            edge=edge,
            price_cents=price_cents,
            confidence=confidence,
            ticker=ticker,
            signal_source="weather_observed_arbitrage",
        )

        if count <= 0:
            self._record_weather_rejection(market, "position_size_zero", stage="observed_sizing", signal_source="weather_observed_arbitrage", near_miss=True)
            return {
                "ticker": ticker, "title": title, "side": side, "edge": edge,
                "our_prob": our_prob, "kalshi_prob": kalshi_prob, "confidence": confidence,
                "action": "skip", "reason": "position_size_zero",
            }

        return {
            "strategy": "weather",
            "ticker": ticker,
            "title": title,
            "side": side,
            "edge": edge,
            "our_prob": our_prob,
            "kalshi_prob": kalshi_prob,
            "confidence": confidence,
            "count": count,
            "price_cents": price_cents,
            "signal_id": signal_id,
            "signal_source": "weather_observed_arbitrage",
        }

    # ── Cross-Market Sports Strategy ───────────────────────────────

    async def run_sports_cycle(self) -> list[dict[str, Any]]:
        """
        Run one sports strategy cycle:
        1. Scan Kalshi for single-game sports markets (moneylines, spreads, totals)
        2. Fetch sharp lines from The Odds API
        3. Match each Kalshi market to its sharp line and calculate edge
        4. Execute trades where edge exceeds threshold
        5. Also scan parlays for additional opportunities
        """
        self.engine.log_event("info", "Sports cycle starting", strategy="sports")
        self.engine.start_cycle("sports")
        results: list[dict[str, Any]] = []

        if not self.engine.strategy_enabled.get("sports", True):
            self.engine.log_event("info", "Sports strategy disabled", strategy="sports")
            return results

        try:
            # ── Step 0: Fetch referee signals for today (NBA totals edge) ──
            ref_signals: dict[str, Any] = {}
            try:
                ref_signals = await self.referee_data.get_game_ref_signals()
                if ref_signals:
                    self.engine.log_event(
                        "info",
                        f"Referee data: {len(ref_signals)} games, "
                        + ", ".join(
                            f"{k}: {v['direction']}({v['foul_adjustment']:+.1f})"
                            for k, v in list(ref_signals.items())[:3]
                        ),
                        strategy="sports",
                    )
            except Exception as e:
                logger.debug("Referee data fetch failed", error=str(e))

            # ── Step 1: Scan single-game markets ──────────────────
            single_markets = await self.scanner.scan_single_game_markets(min_volume=0)
            self.engine.log_event(
                "info",
                f"Found {len(single_markets)} single-game markets",
                strategy="sports",
            )

            # ── Step 2: Fetch sharp odds only for sports with open Kalshi markets ──
            sports_needed: set[str] = set()
            for m in single_markets:
                sport = m.get("odds_sport", "")
                if sport:
                    sports_needed.add(sport)

            odds_by_sport: dict[str, list[dict[str, Any]]] = {}
            all_odds_events: list[dict[str, Any]] = []
            for sport in sports_needed:
                try:
                    events = await self.sports.get_odds(sport, markets="h2h,spreads,totals")
                    if events:
                        odds_by_sport[sport] = events
                        all_odds_events.extend(events)
                        self.engine.log_event(
                            "info",
                            f"{sport}: {len(events)} events with odds",
                            strategy="sports",
                        )
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.warning("Odds fetch failed", sport=sport, error=str(e))

            self.engine.log_event(
                "info",
                f"Fetched odds for {len(all_odds_events)} events across {len(sports_needed)} sports",
                strategy="sports",
            )

            if not all_odds_events:
                self.engine.log_event("info", "No odds data available", strategy="sports")
                return results

            # ── Step 2b: Fetch breaking injury news (Twitter/RSS) ────────
            # Catches injuries reported AFTER the sharp line was set but before
            # Kalshi reprices. Reduces confidence on affected game markets.
            breaking_injury_texts: list[str] = []
            try:
                from app.services.injury_scraper import get_injury_scraper
                _inj = get_injury_scraper()
                inj_news = await _inj.get_all_injury_news()
                breaking_injury_texts = [
                    item.get("text", "") or item.get("title", "") or item.get("summary", "")
                    for item in inj_news.get("twitter", []) + inj_news.get("rss", [])
                ]
                if breaking_injury_texts:
                    self.engine.log_event(
                        "info",
                        f"Sports: {len(breaking_injury_texts)} breaking injury items from Twitter/RSS",
                        strategy="sports",
                    )
            except Exception as e:
                logger.debug("Breaking injury fetch failed", error=str(e))

            # ── Step 3: Evaluate single-game markets against sharp lines ──
            single_stats = {"matched": 0, "no_odds": 0, "cheap_skip": 0, "no_team_match": 0, "no_sharp": 0, "no_edge": 0}
            for market in single_markets:
                try:
                    signal = await self._evaluate_single_game(
                        market, odds_by_sport, single_stats, ref_signals,
                        breaking_injury_texts=breaking_injury_texts,
                    )
                    if signal:
                        results.append(signal)
                except Exception as e:
                    logger.debug("Single game eval failed", ticker=market.get("ticker"), error=str(e))

            self.engine.log_event(
                "info",
                f"Single-game: {len(single_markets)} scanned | {single_stats['matched']} matched, "
                f"{single_stats['no_odds']} no_odds, {single_stats['cheap_skip']} cheap, "
                f"{single_stats['no_team_match']} no_team, {single_stats['no_sharp']} no_sharp, "
                f"{single_stats['no_edge']} no_edge",
                strategy="sports",
            )

            # ── Step 4: Also evaluate parlays ──
            parlays = await self.scanner.scan_sports_parlays(min_volume=10)
            parlay_stats = {"no_legs": 0, "no_fair_prob": 0, "low_confidence": 0, "no_edge": 0, "illiquid": 0, "traded": 0}
            for parlay_market in parlays:
                try:
                    signal = await self._evaluate_parlay(parlay_market, all_odds_events, parlay_stats)
                    if signal:
                        results.append(signal)
                        parlay_stats["traded"] += 1
                except Exception as e:
                    logger.debug("Parlay eval failed", ticker=parlay_market.get("ticker"), error=str(e))

            self.engine.log_event(
                "info",
                f"Parlays: {len(parlays)} eval | traded={parlay_stats['traded']} no_edge={parlay_stats['no_edge']}",
                strategy="sports",
            )

        except Exception as e:
            self.engine.log_event("error", f"Sports cycle failed: {e}", strategy="sports")
            logger.error("Sports cycle error", error=str(e))

        self.engine.log_event(
            "info",
            f"Sports cycle complete: {len(results)} signals",
            strategy="sports",
        )
        return results

    async def _evaluate_single_game(
        self,
        market: dict[str, Any],
        odds_by_sport: dict[str, list[dict[str, Any]]],
        stats: dict[str, int],
        ref_signals: dict[str, Any] | None = None,
        breaking_injury_texts: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """
        Evaluate a single-game Kalshi market against sharp Odds API lines.

        Kalshi market structure:
        - Game Winner: ticker ends with team abbrev (-PSG, -ASM) or -TIE
        - Totals: ticker ends with line number (-3, -4), floor_strike has actual line (3.5, 4.5)
        - Spread: ticker ends with team+line (-BVB2), floor_strike has the spread line
        """
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        odds_sport = market.get("odds_sport", "")
        kalshi_type = market.get("kalshi_market_type", "")
        yes_ask = market.get("yes_ask", 0)
        no_ask = market.get("no_ask", 0)
        floor_strike = market.get("floor_strike")

        # Safety filter: Max Spread Limit = 5¢
        yes_bid = market.get("yes_bid", 0)
        no_bid = market.get("no_bid", 0)
        if yes_ask > 0 and yes_bid > 0 and (yes_ask - yes_bid) > 5:
            return None
        if no_ask > 0 and no_bid > 0 and (no_ask - no_bid) > 5:
            return None

        # Calculate Maker Prices (Pennying the bid)
        yes_maker_price = min(yes_bid + 1, yes_ask) if yes_bid > 0 else yes_ask
        no_maker_price = min(no_bid + 1, no_ask) if no_bid > 0 else no_ask

        # Skip multi-game / extended series tickers — these aggregate multiple
        # games and cannot be matched to a single Odds API event
        ticker_upper = ticker.upper()
        if "MULTIGAME" in ticker_upper or "EXTENDED" in ticker_upper or "SERIES" in ticker_upper:
            stats["no_odds"] += 1
            return None

        if not odds_sport or not title or not ticker:
            stats["no_odds"] += 1
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        # These are long-shot bets with high loss rates even with edge
        if min(yes_ask, no_ask) < 10:
            stats["cheap_skip"] += 1
            return None

        events = odds_by_sport.get(odds_sport, [])
        if not events:
            stats["no_odds"] += 1
            return None

        # Parse the ticker suffix to understand what this contract represents
        # Format: KXSERIES-DATECODETEAMS-OUTCOME
        ticker_parts = ticker.rsplit("-", 1)
        if len(ticker_parts) < 2:
            stats["no_odds"] += 1
            return None
        outcome_suffix = ticker_parts[1].upper()

        # Extract team names from title (format: "TeamA vs TeamB Winner?" or "TeamB at TeamA: Totals")
        title.lower()

        # Find the matching Odds API event by team names in title
        # Extract the two team names from the Kalshi title
        # Formats: "TeamA vs TeamB Winner?" or "TeamB at TeamA: Totals" or "Will X win the Y vs Z match?"
        matched_event = None
        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            if not home or not away:
                continue

            # Use teams_match which handles aliases (PSG, Inter, Atletico, etc.)
            # Only match full team names — not individual words to avoid false positives
            home_in_title = teams_match(home, title)
            away_in_title = teams_match(away, title)

            if home_in_title and away_in_title:
                matched_event = event
                break

        if not matched_event:
            stats["no_team_match"] += 1
            return None

        # Extract sharp consensus for this event
        consensus = self.sports.extract_sharp_consensus(matched_event)
        if consensus.get("sharp_books_found", 0) == 0:
            stats["no_sharp"] += 1
            return None

        sharp = consensus.get("sharp_consensus", {})
        if not sharp:
            stats["no_sharp"] += 1
            return None

        # Now match based on market type
        sharp_prob = None
        match_desc = ""

        if kalshi_type == "h2h":
            # Game Winner — outcome_suffix is team abbrev or TIE
            if outcome_suffix == "TIE":
                # Draw — look for "Draw" in h2h consensus
                for key, prob in sharp.items():
                    if key.startswith("h2h|") and "draw" in key.lower():
                        sharp_prob = prob
                        match_desc = "h2h|Draw"
                        break
            else:
                # Team win — match suffix to team name
                for key, prob in sharp.items():
                    if not key.startswith("h2h|"):
                        continue
                    team_name = key.split("|", 1)[1]
                    if _suffix_matches_team_name(outcome_suffix, team_name):
                        sharp_prob = prob
                        match_desc = key
                        break

        elif kalshi_type == "totals":
            # Totals — floor_strike has the actual line (e.g., 2.5, 3.5)
            if floor_strike is not None:
                target_line = float(floor_strike)
                for key, prob in sharp.items():
                    if not key.startswith("totals|"):
                        continue
                    parts = key.split("|")
                    if len(parts) < 3:
                        continue
                    direction = parts[1]  # "Over" or "Under"
                    try:
                        line = float(parts[2])
                    except ValueError:
                        continue
                    # Must match the exact line
                    if abs(line - target_line) < 0.1 and direction.lower() == "over":
                        sharp_prob = prob
                        match_desc = key
                        break

        elif kalshi_type == "spreads":
            # Spread — floor_strike has the spread line
            if floor_strike is not None:
                target_line = float(floor_strike)
                for key, prob in sharp.items():
                    if not key.startswith("spreads|"):
                        continue
                    parts = key.split("|")
                    if len(parts) < 3:
                        continue
                    team_name = parts[1]
                    try:
                        line = float(parts[2])
                    except ValueError:
                        continue
                    # Match team from suffix and line from floor_strike.
                    # Kalshi "Team wins by over X.5" corresponds to that team at -X.5,
                    # not +X.5. Matching on absolute value would wrongly turn
                    # an underdog +X.5 cover price into a margin-of-victory price.
                    if line >= 0:
                        continue
                    # Remove trailing digits from suffix to get team part
                    team_part = outcome_suffix.rstrip("0123456789")
                    if _suffix_matches_team_name(team_part, team_name):
                        if abs(abs(line) - target_line) < 0.5:
                            sharp_prob = prob
                            match_desc = key
                            break

        if sharp_prob is None:
            stats["no_sharp"] += 1
            return None

        # Calculate edge: Kalshi YES price vs sharp probability
        kalshi_implied = yes_maker_price / 100.0 if yes_maker_price else 0
        if kalshi_implied <= 0:
            stats["no_edge"] += 1
            return None

        edge = sharp_prob - kalshi_implied
        side = "yes"

        # Also check NO side
        no_implied = no_maker_price / 100.0 if no_maker_price else 0
        no_sharp = 1.0 - sharp_prob
        no_edge = no_sharp - no_implied
        if no_edge > edge and no_implied > 0:
            edge = no_edge
            side = "no"
            sharp_prob = no_sharp
            kalshi_implied = no_implied

        sports_thresh = self.adaptive.get_thresholds("sports")
        min_edge = sports_thresh["min_edge"]
        max_edge = 0.25  # 25% cap — anything higher is likely a bad match
        if edge < min_edge:
            stats["no_edge"] += 1
            return None
        if edge > max_edge:
            logger.info("Edge too high (likely bad match)", ticker=ticker, edge=f"{edge:.1%}", title=title[:60])
            stats["no_edge"] += 1
            return None

        stats["matched"] += 1

        # ── Referee adjustment for NBA totals ──────────────────────────
        confidence = 0.85
        ref_detail = ""
        if kalshi_type == "totals" and ref_signals and matched_event:
            home = matched_event.get("home_team", "")
            away = matched_event.get("away_team", "")
            # Try multiple key formats to find the ref signal
            ref_sig = (
                ref_signals.get(f"{home}_{away}")
                or ref_signals.get(f"{away}_{home}")
            )
            if not ref_sig:
                # Partial match on team abbreviations
                for rk, rv in (ref_signals or {}).items():
                    rk_teams = rk.lower().replace("_", " ")
                    if (home.lower()[:3] in rk_teams or away.lower()[:3] in rk_teams):
                        ref_sig = rv
                        break

            if ref_sig and ref_sig.get("strength", 0) > 0.1:
                ref_direction = ref_sig["direction"]  # "over", "under", "neutral"
                ref_strength = ref_sig["strength"]    # 0-1
                ref_point_adj = ref_sig["point_adjustment"]

                # Does the ref signal agree with our trade direction?
                trade_is_over = (side == "yes")  # YES on totals = over
                ref_agrees = (
                    (ref_direction == "over" and trade_is_over) or
                    (ref_direction == "under" and not trade_is_over)
                )

                if ref_agrees:
                    confidence = min(0.95, confidence + ref_strength * 0.10)
                    ref_detail = f" ref={ref_direction}({ref_point_adj:+.1f}pts,str={ref_strength:.2f}) ✓"
                elif ref_direction != "neutral":
                    confidence = max(0.60, confidence - ref_strength * 0.08)
                    ref_detail = f" ref={ref_direction}({ref_point_adj:+.1f}pts,str={ref_strength:.2f}) ✗"

        # ── Breaking injury check (Twitter/RSS) ────────────────────────
        # If a team name from this market appears in a breaking injury report,
        # reduce confidence — the sharp line may not yet reflect the news.
        injury_detail = ""
        if breaking_injury_texts and matched_event:
            home = matched_event.get("home_team", "").lower()
            away = matched_event.get("away_team", "").lower()
            injury_keywords = ["out", "ruled out", "injured", "sidelined", "doubtful",
                               "questionable", "day-to-day", "miss", "surgery", "won't play"]
            for text in breaking_injury_texts:
                text_lower = text.lower()
                has_injury = any(kw in text_lower for kw in injury_keywords)
                team_mentioned = (
                    any(w in text_lower for w in home.split() if len(w) >= 4) or
                    any(w in text_lower for w in away.split() if len(w) >= 4)
                )
                if has_injury and team_mentioned:
                    confidence = max(0.50, confidence - 0.20)
                    injury_detail = " breaking_injury=⚠️"
                    logger.debug(f"Breaking injury penalty applied for {ticker}: {text[:80]}")
                    break

        self.engine.log_event(
            "info",
            f"Single-game edge: {ticker} {side} edge={edge:.1%} (sharp={sharp_prob:.1%} vs kalshi={kalshi_implied:.1%}) {match_desc}{ref_detail}{injury_detail} | {title[:60]}",
            strategy="sports",
        )

        # Record signal
        self.engine.record_signal(
            strategy="sports",
            ticker=ticker,
            side=side,
            our_prob=sharp_prob,
            kalshi_prob=kalshi_implied,
            market_title=title,
            confidence=confidence,
            signal_source="sharp_single_game",
            details=f"sport={odds_sport} type={kalshi_type} match={match_desc} edge={edge:.4f}{ref_detail}{injury_detail}",
        )

        # Return candidate for global ranking (execution happens later)
        price_cents = yes_maker_price if side == "yes" else no_maker_price
        return {
            "strategy": "sports",
            "ticker": ticker,
            "title": title,
            "side": side,
            "edge": edge,
            "confidence": confidence,
            "our_prob": sharp_prob,
            "kalshi_prob": kalshi_implied,
            "price_cents": price_cents,
            "signal_source": "sharp_single_game",
        }

    async def _evaluate_parlay(
        self,
        parlay_market: dict[str, Any],
        odds_events: list[dict[str, Any]],
        stats: dict[str, int] | None = None,
    ) -> dict[str, Any] | None:
        """Evaluate a single Kalshi parlay market against Odds API data."""
        ticker = parlay_market.get("ticker", "")
        title = parlay_market.get("title", "")
        yes_ask = parlay_market.get("yes_ask", 0)
        no_ask = parlay_market.get("no_ask", 0)

        # Safety filter: Max Spread Limit = 5¢
        yes_bid = parlay_market.get("yes_bid", 0)
        no_bid = parlay_market.get("no_bid", 0)
        if yes_ask > 0 and yes_bid > 0 and (yes_ask - yes_bid) > 5:
            if stats:
                stats["illiquid"] += 1
            return None
        if no_ask > 0 and no_bid > 0 and (no_ask - no_bid) > 5:
            if stats:
                stats["illiquid"] += 1
            return None

        # Calculate Maker Prices (Pennying the bid)
        yes_maker_price = min(yes_bid + 1, yes_ask) if yes_bid > 0 else yes_ask
        no_maker_price = min(no_bid + 1, no_ask) if no_bid > 0 else no_ask

        # Safety filters
        if yes_maker_price <= 3 or no_maker_price <= 3:
            if stats:
                stats["illiquid"] += 1
            return None
        # Skip very cheap contracts (< 10c on either side)
        if min(yes_maker_price, no_maker_price) < 10:
            if stats:
                stats["illiquid"] += 1
            return None
        if parlay_market.get("volume", 0) < 10:
            if stats:
                stats["illiquid"] += 1
            return None

        # Parse legs
        parlay_info = parlay_market.get("parlay", {})
        legs = parlay_info.get("legs", [])
        if not legs:
            legs = parse_parlay_legs(title)
        if not legs:
            if stats:
                stats["no_legs"] += 1
            return None

        # Price the parlay
        pricing = price_parlay_legs(legs, odds_events)
        fair_prob = pricing.get("fair_prob")
        if fair_prob is None:
            if stats:
                stats["no_fair_prob"] += 1
            return None

        legs_priced = pricing.get("legs_priced", 0)
        legs_total = pricing.get("legs_total", 0)
        confidence = pricing.get("confidence", 0)

        # Require at least 50% of legs priced for a signal
        if confidence < 0.5:
            if stats:
                stats["low_confidence"] += 1
            return None

        # Compare fair value to Kalshi Maker price
        kalshi_yes_prob = yes_maker_price / 100.0
        kalshi_no_prob = no_maker_price / 100.0

        yes_edge = fair_prob - kalshi_yes_prob
        no_edge = (1.0 - fair_prob) - kalshi_no_prob

        parlay_thresh = self.adaptive.get_thresholds("sports")
        min_edge = parlay_thresh["min_edge"]
        best_edge = max(yes_edge, no_edge)
        best_side = "yes" if yes_edge >= no_edge else "no"

        # Track edge distribution in stats
        if stats is not None:
            if best_edge >= 0.05:
                stats.setdefault("edge_5pct+", 0)
                stats["edge_5pct+"] += 1
            elif best_edge >= 0.03:
                stats.setdefault("edge_3-5pct", 0)
                stats["edge_3-5pct"] += 1
            elif best_edge >= 0.01:
                stats.setdefault("edge_1-3pct", 0)
                stats["edge_1-3pct"] += 1
            else:
                stats.setdefault("edge_<1pct", 0)
                stats["edge_<1pct"] += 1

        # Log top edges for debugging
        if best_edge >= 0.03:
            self.engine.log_event(
                "info",
                f"Edge found: {ticker} {best_side} edge={best_edge:.1%} (fair={fair_prob:.1%} vs kalshi={kalshi_yes_prob:.1%}) legs={legs_priced}/{legs_total}",
                strategy="sports",
            )

        signal = None

        if yes_edge >= min_edge:
            signal = {
                "side": "yes",
                "our_prob": round(fair_prob, 4),
                "kalshi_prob": round(kalshi_yes_prob, 4),
                "edge": round(yes_edge, 4),
            }
        elif no_edge >= min_edge:
            signal = {
                "side": "no",
                "our_prob": round(1.0 - fair_prob, 4),
                "kalshi_prob": round(kalshi_no_prob, 4),
                "edge": round(no_edge, 4),
            }

        if not signal:
            if stats:
                stats["no_edge"] += 1
            return None

        # Record signal
        signal_id = self.engine.record_signal(
            strategy="sports",
            ticker=ticker,
            side=signal["side"],
            our_prob=signal["our_prob"],
            kalshi_prob=signal["kalshi_prob"],
            market_title=title,
            confidence=confidence,
            signal_source="parlay_pricer",
            details=f"legs={legs_priced}/{legs_total} fair={fair_prob:.4f}",
        )

        # Return candidate for global ranking (execution happens later)
        price_cents = yes_maker_price if signal["side"] == "yes" else no_maker_price
        return {
            "strategy": "sports",
            "ticker": ticker,
            "title": title,
            "side": signal["side"],
            "edge": signal["edge"],
            "confidence": confidence,
            "our_prob": signal["our_prob"],
            "kalshi_prob": signal["kalshi_prob"],
            "price_cents": price_cents,
            "signal_source": "parlay_pricer",
            "signal_id": signal_id,
        }

    # ── Crypto Strategy ──────────────────────────────────────────────

    async def run_crypto_cycle(self) -> list[dict[str, Any]]:
        """
        Run one crypto strategy cycle:
        1. Scan Kalshi for active 15-min crypto markets
        2. Fetch real-time signals from Binance (momentum, funding, vol)
        3. Compare signal probability to Kalshi price
        4. Execute trades where edge > 3%
        """
        self.engine.log_event("info", "Crypto cycle starting", strategy="crypto")
        self.engine.start_cycle("crypto")
        results: list[dict[str, Any]] = []

        if not self.engine.strategy_enabled.get("crypto", True):
            self.engine.log_event("info", "Crypto strategy disabled", strategy="crypto")
            return results

        try:
            # Step 1: Scan for crypto markets
            crypto_markets = await self.scanner.scan_crypto_markets()
            self.engine.log_event(
                "info",
                f"Found {len(crypto_markets)} crypto markets",
                strategy="crypto",
            )

            if not crypto_markets:
                return results

            # Build a map of coin → Kalshi yes_ask for vol comparison
            kalshi_prices: dict[str, int] = {}
            for m in crypto_markets:
                coin = m.get("crypto", {}).get("coin")
                if coin and coin not in kalshi_prices:
                    kalshi_prices[coin] = m.get("yes_ask", 50)

            # Step 2: Fetch signals for all coins
            signals = await self.crypto.get_all_signals(kalshi_prices=kalshi_prices)
            signal_by_coin: dict[str, dict] = {s["coin"]: s for s in signals}

            # Step 2.5: Enrich signals with news sentiment
            try:
                loop = asyncio.get_event_loop()
                news = await loop.run_in_executor(
                    None, self.market_news.get_market_sentiment, list(signal_by_coin.keys())
                )
                for coin, sig in signal_by_coin.items():
                    ns = news.get(coin, {})
                    sentiment = ns.get("sentiment", 0.0)
                    article_count = ns.get("article_count", 0)
                    if article_count > 0:
                        # Sentiment aligns with signal direction → boost confidence
                        # Sentiment opposes signal direction → reduce confidence
                        signal_direction = 1.0 if sig["p_up"] > 0.5 else -1.0
                        alignment = sentiment * signal_direction  # positive = aligned
                        conf_adj = alignment * min(article_count / 10.0, 1.0) * 0.15
                        sig["confidence"] = min(1.0, max(0.0, sig["confidence"] + conf_adj))
                        sig["news_sentiment"] = sentiment
                        sig["news_articles"] = article_count
            except Exception as e:
                logger.debug("Crypto news sentiment failed", error=str(e))

            self.engine.log_event(
                "info",
                f"Got signals for {len(signals)} coins: "
                + ", ".join(f"{s['coin']} p_up={s['p_up']:.2f} conf={s['confidence']:.2f}" for s in signals),
                strategy="crypto",
            )

            # Step 3: Evaluate each market (limit 1 bracket per coin to avoid
            # buying multiple contradictory range brackets)
            traded_coins: set[str] = set()
            for market in crypto_markets:
                try:
                    coin = market.get("crypto", {}).get("coin")
                    if coin and coin in traded_coins:
                        continue  # Already traded a bracket for this coin this cycle
                    signal = await self._evaluate_crypto_market(market, signal_by_coin)
                    if signal:
                        results.append(signal)
                        if coin:
                            traded_coins.add(coin)
                except Exception as e:
                    logger.debug("Crypto eval failed", ticker=market.get("ticker"), error=str(e))

        except Exception as e:
            self.engine.log_event("error", f"Crypto cycle failed: {e}", strategy="crypto")
            logger.error("Crypto cycle error", error=str(e))

        self.engine.log_event(
            "info",
            f"Crypto cycle complete: {len(results)} trades",
            strategy="crypto",
        )
        return results

    async def _evaluate_crypto_market(
        self,
        market: dict[str, Any],
        signal_by_coin: dict[str, dict],
    ) -> dict[str, Any] | None:
        """Evaluate a single crypto market against our signal."""
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        coin = market.get("crypto", {}).get("coin")
        yes_ask = market.get("yes_ask", 0)
        no_ask = market.get("no_ask", 0)

        # Safety filter: Max Spread Limit = 5¢
        yes_bid = market.get("yes_bid", 0)
        no_bid = market.get("no_bid", 0)
        if yes_ask > 0 and yes_bid > 0 and (yes_ask - yes_bid) > 5:
            return None
        if no_ask > 0 and no_bid > 0 and (no_ask - no_bid) > 5:
            return None

        # Calculate Maker Prices (Pennying the bid)
        yes_maker_price = min(yes_bid + 1, yes_ask) if yes_bid > 0 else yes_ask
        no_maker_price = min(no_bid + 1, no_ask) if no_bid > 0 else no_ask

        if not coin or coin not in signal_by_coin:
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        if min(yes_maker_price, no_maker_price) < 10:
            return None

        signal = signal_by_coin[coin]
        p_up = signal["p_up"]
        confidence = signal["confidence"]

        # Determine if this is an "up" or "above" market
        title_lower = title.lower()
        is_up_market = any(kw in title_lower for kw in ["up", "above", "higher", "over"])
        is_down_market = any(kw in title_lower for kw in ["down", "below", "lower", "under"])

        # CRITICAL: Skip range/bracket markets (e.g., "Bitcoin price range on Feb 17")
        # Our directional signal (p_up) does NOT map to the probability of landing
        # in a specific $250 price bracket. Only trade genuinely directional markets.
        is_range_market = any(kw in title_lower for kw in ["price range", "range", "between", "bracket"])
        if is_range_market:
            return None

        if not is_up_market and not is_down_market:
            # No directional keywords found — skip (don't guess)
            return None

        # Our probability for YES on this market
        if is_up_market:
            our_prob_yes = p_up
        else:
            our_prob_yes = 1.0 - p_up

        # Calculate edge for both sides
        kalshi_yes_implied = yes_maker_price / 100.0 if yes_maker_price > 0 else 0
        kalshi_no_implied = no_maker_price / 100.0 if no_maker_price > 0 else 0

        yes_edge = our_prob_yes - kalshi_yes_implied if kalshi_yes_implied > 0 else 0
        no_edge = (1.0 - our_prob_yes) - kalshi_no_implied if kalshi_no_implied > 0 else 0

        # Pick the better side
        if yes_edge >= no_edge:
            edge = yes_edge
            side = "yes"
            our_prob = our_prob_yes
            kalshi_prob = kalshi_yes_implied
            price_cents = yes_maker_price
        else:
            edge = no_edge
            side = "no"
            our_prob = 1.0 - our_prob_yes
            kalshi_prob = kalshi_no_implied
            price_cents = no_maker_price

        thresholds = self.adaptive.get_thresholds("crypto")
        min_edge = thresholds["min_edge"]
        if edge < min_edge:
            return None

        # Confidence gate: require meaningful signal agreement
        if confidence < thresholds["min_confidence"]:
            return None

        # Price filter: skip very cheap contracts (< 10c) — too speculative
        if price_cents < 10:
            return None

        # ── MVP Thesis Gate: funding rate must not contradict momentum ──
        # Funding rate is the institutional directional bias signal.
        # When it directly opposes momentum, the trade has no thesis.
        funding_s = signal.get("funding_signal", 0)
        momentum_5m = signal.get("momentum_5m", 0)
        FUNDING_CONTRADICTION_THRESHOLD = 0.20  # both signals must be meaningful
        MOMENTUM_CONTRADICTION_THRESHOLD = 0.20
        if (abs(funding_s) > FUNDING_CONTRADICTION_THRESHOLD
                and abs(momentum_5m) > MOMENTUM_CONTRADICTION_THRESHOLD
                and funding_s * momentum_5m < 0):
            # Funding and momentum point opposite directions — no thesis
            logger.debug(
                f"Crypto thesis rejected: funding vs momentum contradiction for {coin} "
                f"— funding={funding_s:.3f} momentum_5m={momentum_5m:.3f} side={side}"
            )
            return None

        # Polymarket cross-reference: boost confidence if prices diverge
        poly_details = ""
        try:
            poly_signal = await self.polymarket.get_edge_signal(title, price_cents, category="crypto")
            if poly_signal:
                confidence = min(1.0, confidence + poly_signal["confidence_boost"])
                poly_details = f" poly={poly_signal['poly_price_cents']:.0f}c div={poly_signal['divergence']:+.1%}"
        except Exception:
            pass

        self.engine.log_event(
            "info",
            f"Crypto edge: {ticker} {side} edge={edge:.1%} (our={our_prob:.1%} vs kalshi={kalshi_prob:.1%}) "
            f"coin={coin} p_up={p_up:.2f} conf={confidence:.2f}{poly_details} | {title[:60]}",
            strategy="crypto",
        )

        # Record signal
        self.engine.record_signal(
            strategy="crypto",
            ticker=ticker,
            side=side,
            our_prob=our_prob,
            kalshi_prob=kalshi_prob,
            market_title=title,
            confidence=confidence,
            signal_source="crypto_momentum",
            details=f"coin={coin} p_up={p_up:.4f} mom5={signal['momentum_5m']:.4f} "
                    f"mom1={signal['momentum_1m']:.4f} fund={signal['funding_signal']:.4f} "
                    f"mr={signal['mean_reversion']:.4f} vol={signal['vol_multiplier']:.2f}{poly_details}",
        )
        self._record_cross_signal("crypto", ticker, side, our_prob, kalshi_prob, confidence)

        # Apply cross-strategy correlation adjustment
        confidence = self._apply_correlation_adjustment("crypto", ticker, confidence)

        # Return candidate for global ranking (execution happens later)
        return {
            "strategy": "crypto",
            "ticker": ticker,
            "title": title,
            "side": side,
            "edge": edge,
            "confidence": confidence,
            "our_prob": our_prob,
            "kalshi_prob": kalshi_prob,
            "price_cents": price_cents,
            "signal_source": "crypto_momentum",
        }

    # ── Finance Strategy (S&P / Nasdaq) ─────────────────────────────

    async def run_finance_cycle(self) -> list[dict[str, Any]]:
        """
        Run one finance strategy cycle:
        1. Scan Kalshi for S&P/Nasdaq daily close markets
        2. Fetch real-time signals from Yahoo Finance (momentum, VIX, futures)
        3. Compare signal probability to Kalshi price
        4. Execute trades where edge > 3%
        """
        self.engine.log_event("info", "Finance cycle starting", strategy="finance")
        self.engine.start_cycle("finance")
        results: list[dict[str, Any]] = []

        if not self.engine.strategy_enabled.get("finance", True):
            self.engine.log_event("info", "Finance strategy disabled", strategy="finance")
            return results

        try:
            # Step 1: Scan for finance markets
            finance_markets = await self.scanner.scan_finance_markets()
            self.engine.log_event(
                "info",
                f"Found {len(finance_markets)} finance markets",
                strategy="finance",
            )

            if not finance_markets:
                return results

            # Step 2: Fetch signals
            signals = await self.finance.get_all_signals()
            signal_by_index: dict[str, dict] = {s["index"]: s for s in signals}

            # Step 2.5: Enrich signals with news sentiment
            try:
                loop = asyncio.get_event_loop()
                news = await loop.run_in_executor(
                    None, self.market_news.get_market_sentiment, list(signal_by_index.keys())
                )
                for idx, sig in signal_by_index.items():
                    ns = news.get(idx, {})
                    sentiment = ns.get("sentiment", 0.0)
                    article_count = ns.get("article_count", 0)
                    if article_count > 0:
                        signal_direction = 1.0 if sig["p_up"] > 0.5 else -1.0
                        alignment = sentiment * signal_direction
                        conf_adj = alignment * min(article_count / 10.0, 1.0) * 0.15
                        sig["confidence"] = min(1.0, max(0.0, sig["confidence"] + conf_adj))
                        sig["news_sentiment"] = sentiment
                        sig["news_articles"] = article_count
            except Exception as e:
                logger.debug("Finance news sentiment failed", error=str(e))

            self.engine.log_event(
                "info",
                f"Got signals for {len(signals)} indices: "
                + ", ".join(f"{s['index']} p_up={s['p_up']:.2f} conf={s['confidence']:.2f}" for s in signals),
                strategy="finance",
            )

            # Step 3: Evaluate each market
            for market in finance_markets:
                try:
                    signal = await self._evaluate_finance_market(market, signal_by_index)
                    if signal:
                        results.append(signal)
                except Exception as e:
                    logger.debug("Finance eval failed", ticker=market.get("ticker"), error=str(e))

        except Exception as e:
            self.engine.log_event("error", f"Finance cycle failed: {e}", strategy="finance")
            logger.error("Finance cycle error", error=str(e))

        self.engine.log_event(
            "info",
            f"Finance cycle complete: {len(results)} trades",
            strategy="finance",
        )
        return results

    async def _evaluate_finance_market(
        self,
        market: dict[str, Any],
        signal_by_index: dict[str, dict],
    ) -> dict[str, Any] | None:
        """Evaluate a single finance market against our signal."""
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        index = market.get("finance", {}).get("index")
        yes_ask = market.get("yes_ask", 0)
        no_ask = market.get("no_ask", 0)
        strike_type = market.get("strike_type", "")
        floor_strike = market.get("floor_strike")
        cap_strike = market.get("cap_strike")

        # Safety filter: Max Spread Limit = 5¢
        yes_bid = market.get("yes_bid", 0)
        no_bid = market.get("no_bid", 0)
        if yes_ask > 0 and yes_bid > 0 and (yes_ask - yes_bid) > 5:
            return None
        if no_ask > 0 and no_bid > 0 and (no_ask - no_bid) > 5:
            return None

        # Calculate Maker Prices (Pennying the bid)
        yes_maker_price = min(yes_bid + 1, yes_ask) if yes_bid > 0 else yes_ask
        no_maker_price = min(no_bid + 1, no_ask) if no_bid > 0 else no_ask

        if not index or index not in signal_by_index:
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        if min(yes_maker_price, no_maker_price) < 10:
            return None

        signal = signal_by_index[index]
        p_up = signal["p_up"]
        confidence = signal["confidence"]
        current_price = signal.get("current_price", 0)
        vix_level = signal.get("vix_level")

        title_lower = title.lower()
        is_bracket = strike_type == "between" or (
            floor_strike is not None and cap_strike is not None
        )

        if is_bracket:
            # ── Bracket market: use Gaussian price distribution ──────────
            # p_up is a directional signal and is NOT valid for bracket markets.
            # We need the probability of landing in a specific price window.
            if not current_price or floor_strike is None or cap_strike is None:
                return None

            # Bracket viability check: skip if current price is more than
            # 3× bracket width away from the bracket midpoint (dead bracket)
            bracket_width = cap_strike - floor_strike
            bracket_mid = (floor_strike + cap_strike) / 2.0
            if bracket_width > 0 and abs(current_price - bracket_mid) > 3 * bracket_width:
                return None

            # Days to expiry from close_time
            close_time_str = market.get("finance", {}).get("close_time", "") or market.get("close_time", "")
            days_to_expiry = 1.0
            if close_time_str:
                try:
                    close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                    days_to_expiry = max(0.1, (close_dt - datetime.now(UTC)).total_seconds() / 86400)
                except (ValueError, TypeError):
                    pass

            our_prob_yes = self.finance.get_bracket_probability(
                current_price=current_price,
                bracket_low=float(floor_strike),
                bracket_high=float(cap_strike),
                days_to_expiry=days_to_expiry,
                vix_level=vix_level,
            )
            signal_source = "finance_bracket_gaussian"
            prob_details = f"bracket=[{floor_strike},{cap_strike}] price={current_price:.0f} days={days_to_expiry:.2f}"
        else:
            # ── Threshold / directional market: use p_up ────────────────
            is_up_market = any(kw in title_lower for kw in ["up", "above", "higher", "close above", "gain"])
            is_down_market = any(kw in title_lower for kw in ["down", "below", "lower", "close below", "lose", "drop"])

            if not is_up_market and not is_down_market:
                return None

            our_prob_yes = p_up if is_up_market else (1.0 - p_up)
            signal_source = "finance_momentum"
            prob_details = f"index={index} p_up={p_up:.4f} intraday={signal['intraday_momentum']:.4f} " \
                           f"futures={signal['futures_signal']:.4f} vix={signal['vix_signal']:.4f} " \
                           f"ma={signal['ma_signal']:.4f}"

            # ── MVP Thesis Gate: require ≥3 of 5 signals aligned ────────
            # A single momentum reading is noise. Three aligned signals is a thesis.
            trade_direction = 1 if is_up_market else -1  # +1 = bullish bet, -1 = bearish bet
            intraday_s = signal.get("intraday_momentum", 0)
            futures_s = signal.get("futures_signal", 0)
            vix_s = signal.get("vix_signal", 0)
            ma_s = signal.get("ma_signal", 0)
            news_s = signal.get("news_sentiment", 0)
            vix_level = signal.get("vix_level") or 20.0

            # Hard reject: extreme volatility (VIX > 35) — daily close unpredictable
            if vix_level > 35:
                logger.debug("Finance thesis rejected: VIX > 35", ticker=ticker, vix=vix_level)
                return None

            # Hard reject: futures and intraday momentum directly contradict each other
            # (both must be non-trivial and pointing opposite directions)
            if abs(intraday_s) > 0.15 and abs(futures_s) > 0.15 and (intraday_s * futures_s < 0):
                logger.debug("Finance thesis rejected: futures vs intraday contradiction", ticker=ticker)
                return None

            # Count signals aligned with trade direction (threshold: signal > 0.1 in direction)
            ALIGN_THRESHOLD = 0.10
            aligned = 0
            if (intraday_s * trade_direction) > ALIGN_THRESHOLD: aligned += 1
            if (futures_s * trade_direction) > ALIGN_THRESHOLD: aligned += 1
            if (vix_s * trade_direction) > ALIGN_THRESHOLD: aligned += 1
            if (ma_s * trade_direction) > ALIGN_THRESHOLD: aligned += 1
            if (news_s * trade_direction) > ALIGN_THRESHOLD: aligned += 1

            if aligned < 3:
                logger.debug(f"Finance thesis rejected: only {aligned}/5 signals aligned", ticker=ticker)
                return None

        # Calculate edge (common for both bracket and threshold markets)
        kalshi_yes_implied = yes_maker_price / 100.0 if yes_maker_price > 0 else 0
        kalshi_no_implied = no_maker_price / 100.0 if no_maker_price > 0 else 0

        yes_edge = our_prob_yes - kalshi_yes_implied if kalshi_yes_implied > 0 else 0
        no_edge = (1.0 - our_prob_yes) - kalshi_no_implied if kalshi_no_implied > 0 else 0

        if yes_edge >= no_edge:
            edge, side = yes_edge, "yes"
            our_prob, kalshi_prob, price_cents = our_prob_yes, kalshi_yes_implied, yes_maker_price
        else:
            edge, side = no_edge, "no"
            our_prob, kalshi_prob, price_cents = 1.0 - our_prob_yes, kalshi_no_implied, no_maker_price

        fin_thresh = self.adaptive.get_thresholds("finance")
        if edge < fin_thresh["min_edge"] or confidence < fin_thresh["min_confidence"]:
            return None

        # Polymarket cross-reference: boost confidence if prices diverge
        poly_details = ""
        try:
            poly_signal = await self.polymarket.get_edge_signal(title, price_cents, category="finance")
            if poly_signal:
                confidence = min(1.0, confidence + poly_signal["confidence_boost"])
                poly_details = f" poly={poly_signal['poly_price_cents']:.0f}c div={poly_signal['divergence']:+.1%}"
        except Exception:
            pass

        market_type_tag = "bracket" if is_bracket else "threshold"
        self.engine.log_event(
            "info",
            f"Finance edge: {ticker} {side} edge={edge:.1%} (our={our_prob:.1%} vs kalshi={kalshi_prob:.1%}) "
            f"[{market_type_tag}] index={index} conf={confidence:.2f}{poly_details} | {title[:60]}",
            strategy="finance",
        )

        self.engine.record_signal(
            strategy="finance",
            ticker=ticker,
            side=side,
            our_prob=our_prob,
            kalshi_prob=kalshi_prob,
            market_title=title,
            confidence=confidence,
            signal_source=signal_source,
            details=prob_details,
        )
        self._record_cross_signal("finance", ticker, side, our_prob, kalshi_prob, confidence)

        confidence = self._apply_correlation_adjustment("finance", ticker, confidence)

        # Return candidate for global ranking (execution happens later)
        return {
            "strategy": "finance",
            "ticker": ticker,
            "title": title,
            "side": side,
            "edge": edge,
            "confidence": confidence,
            "our_prob": our_prob,
            "kalshi_prob": kalshi_prob,
            "price_cents": price_cents,
            "signal_source": signal_source,
            "is_bracket": is_bracket,
            "finance_index": index,
            "finance_expiry": market.get("finance", {}).get("close_time", "")[:10],
        }

    # ── Econ Strategy (CPI / Fed / Gas / Jobs) ───────────────────────

    async def run_econ_cycle(self) -> list[dict[str, Any]]:
        """
        Run one economic data strategy cycle:
        1. Scan Kalshi for CPI, Fed, Gas, Unemployment markets
        2. Fetch economic data from FRED
        3. Generate probability estimates
        4. Execute trades where edge > 3%
        """
        self.engine.log_event("info", "Econ cycle starting", strategy="econ")
        self.engine.start_cycle("econ")
        results: list[dict[str, Any]] = []

        if not self.engine.strategy_enabled.get("econ", True):
            self.engine.log_event("info", "Econ strategy disabled", strategy="econ")
            return results

        try:
            # Step 1: Scan for econ markets
            econ_markets = await self.scanner.scan_econ_markets()
            self.engine.log_event(
                "info",
                f"Found {len(econ_markets)} econ markets",
                strategy="econ",
            )

            if not econ_markets:
                return results

            # Step 2: Fetch all econ signals
            econ_signals = await self.econ.get_all_signals()
            self.engine.log_event(
                "info",
                f"Got econ signals: {list(econ_signals.keys())}",
                strategy="econ",
            )

            # Step 3: Evaluate each market
            for market in econ_markets:
                try:
                    signal = await self._evaluate_econ_market(market, econ_signals)
                    if signal:
                        results.append(signal)
                except Exception as e:
                    logger.debug("Econ eval failed", ticker=market.get("ticker"), error=str(e))

        except Exception as e:
            self.engine.log_event("error", f"Econ cycle failed: {e}", strategy="econ")
            logger.error("Econ cycle error", error=str(e))

        self.engine.log_event(
            "info",
            f"Econ cycle complete: {len(results)} trades",
            strategy="econ",
        )
        return results

    async def _evaluate_econ_market(
        self,
        market: dict[str, Any],
        econ_signals: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Evaluate a single econ market against our data."""
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        econ_type = market.get("econ", {}).get("type")
        yes_ask = market.get("yes_ask", 0)
        no_ask = market.get("no_ask", 0)

        # Safety filter: Max Spread Limit = 5¢
        yes_bid = market.get("yes_bid", 0)
        no_bid = market.get("no_bid", 0)
        if yes_ask > 0 and yes_bid > 0 and (yes_ask - yes_bid) > 5:
            return None
        if no_ask > 0 and no_bid > 0 and (no_ask - no_bid) > 5:
            return None

        if not econ_type or econ_type not in econ_signals:
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        if min(yes_ask, no_ask) < 10:
            return None

        signal = econ_signals[econ_type]
        title_lower = title.lower()
        is_above_market = any(kw in title_lower for kw in ["above", "over", "higher"])
        is_below_market = any(kw in title_lower for kw in ["below", "under", "lower"])

        # Extract threshold from title (e.g., "Will CPI be above 3.0%?")
        threshold_match = re.search(r'(\d+\.?\d*)\s*%', title)
        threshold_price_match = re.search(r'\$(\d+\.?\d*)', title)

        our_prob_yes = 0.5  # Default
        confidence = 0.3

        if econ_type == "cpi" and signal.get("mom_change") is not None:
            if threshold_match:
                threshold = float(threshold_match.group(1)) / 100.0
                # Kalshi CPI markets are month-over-month (e.g. "above 0.5%")
                # Use mom_change for MoM markets, yoy_change for YoY markets
                title_lower_cpi = title_lower
                if "year" in title_lower_cpi or "annual" in title_lower_cpi or "yoy" in title_lower_cpi:
                    estimated = signal.get("estimated_next_yoy") or signal.get("yoy_change") or signal["mom_change"]
                else:
                    estimated = signal["mom_change"]  # MoM decimal (e.g. 0.003 = 0.3%)
                # volatility = absolute std dev of the CPI estimate
                # MoM CPI std dev ~0.002 (0.2 percentage points), YoY ~0.005
                is_yoy = "year" in title_lower_cpi or "annual" in title_lower_cpi or "yoy" in title_lower_cpi
                cpi_vol = 0.005 if is_yoy else 0.002
                our_prob_yes = self.econ.estimate_probability_above(
                    estimated, threshold, volatility=cpi_vol
                )
                if is_below_market and not is_above_market:
                    our_prob_yes = 1.0 - our_prob_yes
                elif not is_above_market and not is_below_market:
                    return None
                confidence = 0.4
            else:
                return None

        elif econ_type == "fed_funds":
            if "cut" in title_lower:
                our_prob_yes = signal.get("p_cut", 0.3)
            elif "hike" in title_lower or "raise" in title_lower:
                our_prob_yes = signal.get("p_hike", 0.1)
            elif "hold" in title_lower or "unchanged" in title_lower:
                our_prob_yes = signal.get("p_hold", 0.6)
            else:
                return None
            # FedWatch gives real market-implied probs — boost confidence accordingly
            confidence = 0.4 + signal.get("confidence_boost", 0.0)

        elif econ_type == "gas_price" and signal.get("estimated_next") is not None:
            if threshold_price_match:
                threshold = float(threshold_price_match.group(1))
                # Gas price std dev ~$0.10/week
                our_prob_yes = self.econ.estimate_probability_above(
                    signal["estimated_next"], threshold, volatility=0.10
                )
                if is_below_market and not is_above_market:
                    our_prob_yes = 1.0 - our_prob_yes
                elif not is_above_market and not is_below_market:
                    return None
                confidence = 0.4
            else:
                return None

        elif econ_type == "unemployment" and signal.get("estimated_next_rate") is not None:
            if threshold_match:
                threshold = float(threshold_match.group(1))
                # Unemployment std dev ~0.2 percentage points
                our_prob_yes = self.econ.estimate_probability_above(
                    signal["estimated_next_rate"], threshold, volatility=0.2
                )
                if is_below_market and not is_above_market:
                    our_prob_yes = 1.0 - our_prob_yes
                elif not is_above_market and not is_below_market:
                    return None
                confidence = 0.4
            else:
                return None
        else:
            return None

        # Calculate Maker Prices (Pennying the bid)
        yes_maker_price = min(yes_bid + 1, yes_ask) if yes_bid > 0 else yes_ask
        no_maker_price = min(no_bid + 1, no_ask) if no_bid > 0 else no_ask

        # Safety filter: skip very cheap contracts (< 10c on either side)
        if min(yes_maker_price, no_maker_price) < 10:
            return None

        kalshi_yes_implied = yes_maker_price / 100.0 if yes_maker_price > 0 else 0
        kalshi_prob_no = no_maker_price / 100.0 if no_maker_price > 0 else 0

        our_prob_no = 1.0 - our_prob_yes

        yes_edge = our_prob_yes - kalshi_yes_implied if kalshi_yes_implied > 0 else 0
        no_edge = our_prob_no - kalshi_prob_no if kalshi_prob_no > 0 else 0

        if yes_edge >= no_edge:
            edge, side = yes_edge, "yes"
            our_prob, kalshi_prob, price_cents = our_prob_yes, kalshi_yes_implied, yes_maker_price
        else:
            edge, side = no_edge, "no"
            our_prob, kalshi_prob, price_cents = 1.0 - our_prob_yes, kalshi_prob_no, no_maker_price

        econ_thresh = self.adaptive.get_thresholds("econ")
        if edge < econ_thresh["min_edge"] or confidence < econ_thresh["min_confidence"]:
            return None

        # Polymarket cross-reference
        poly_details = ""
        try:
            poly_signal = await self.polymarket.get_edge_signal(title, price_cents, category="econ")
            if poly_signal:
                confidence = min(1.0, confidence + poly_signal["confidence_boost"])
                poly_details = f" poly={poly_signal['poly_price_cents']:.0f}c div={poly_signal['divergence']:+.1%}"
        except Exception:
            pass

        self.engine.log_event(
            "info",
            f"Econ edge: {ticker} {side} edge={edge:.1%} (our={our_prob:.1%} vs kalshi={kalshi_prob:.1%}) "
            f"type={econ_type} conf={confidence:.2f}{poly_details} | {title[:60]}",
            strategy="econ",
        )

        self.engine.record_signal(
            strategy="econ", ticker=ticker, side=side,
            our_prob=our_prob, kalshi_prob=kalshi_prob,
            market_title=title, confidence=confidence,
            signal_source=f"econ_{econ_type}",
            details=f"type={econ_type}",
        )
        self._record_cross_signal("econ", ticker, side, our_prob, kalshi_prob, confidence)

        confidence = self._apply_correlation_adjustment("econ", ticker, confidence)

        # Return candidate for global ranking (execution happens later)
        return {
            "strategy": "econ",
            "ticker": ticker,
            "title": title,
            "side": side,
            "edge": edge,
            "confidence": confidence,
            "our_prob": our_prob,
            "kalshi_prob": kalshi_prob,
            "price_cents": price_cents,
            "signal_source": f"econ_{econ_type}",
        }

    # ── NBA Props Strategy ─────────────────────────────────────────

    def _ensure_nba_services(self) -> bool:
        """Lazy-initialize NBA data service and predictor (they use sync HTTP).
        Also triggers a background retrain if model artifacts are >7 days old.
        """
        try:
            if self.nba_data is None:
                self.nba_data = get_nba_data()
            if self.predictor is None:
                self.predictor = get_smart_predictor()

            # ── Weekly retrain: kick off if .joblib files are stale ──────
            self._maybe_trigger_retrain()

            return True
        except Exception as e:
            logger.warning("NBA services init failed", error=str(e))
            return False

    _retrain_triggered_today: str = ""  # date string — only trigger once per day

    def _maybe_trigger_retrain(self) -> None:
        """Trigger a background SmartPredictor retrain if models are >7 days old."""
        import time
        from datetime import date as _date

        from app.services.smart_predictor import ARTIFACTS_DIR

        today = str(_date.today())
        if self._retrain_triggered_today == today:
            return  # Already checked today

        smart_dir = ARTIFACTS_DIR / "smart"
        if not smart_dir.exists():
            return

        # Find the oldest .joblib file
        joblib_files = list(smart_dir.glob("*.joblib"))
        if not joblib_files:
            return

        oldest_mtime = min(f.stat().st_mtime for f in joblib_files)
        age_days = (time.time() - oldest_mtime) / 86400

        if age_days < 7:
            return  # Models are fresh enough

        self._retrain_triggered_today = today
        logger.info(f"NBA models are {age_days:.1f} days old — triggering background retrain")
        self.engine.log_event(
            "info",
            f"NBA models stale ({age_days:.1f}d old) — background retrain started",
            strategy="nba_props",
        )

        import threading

        def _retrain_worker():
            try:
                import asyncio

                from app.services.nba_analysis_cache import build_nba_analysis
                loop = asyncio.new_event_loop()
                cache_data = loop.run_until_complete(build_nba_analysis(force=True))
                loop.close()
                enriched = cache_data.get("name_to_player", {})
                if enriched:
                    predictor = get_smart_predictor()
                    predictor.train_all_props(enriched)
                    logger.info("Background NBA retrain complete")
                else:
                    logger.warning("Background NBA retrain skipped: no enriched players")
            except Exception as e:
                logger.error(f"Background NBA retrain failed: {e}")

        t = threading.Thread(target=_retrain_worker, daemon=True, name="nba-retrain")
        t.start()

    async def run_arbitrage_cycle(self) -> list[dict[str, Any]]:
        """
        Scan for simple arbitrage opportunities where YES + NO prices != 100¢.
        These are risk-free profits:
        - If YES_ASK + NO_ASK < 100: Buy both sides, guaranteed profit
        - If YES_BID + NO_BID > 100: Sell both sides, collect premium
        """
        self.engine.log_event("info", "Arbitrage cycle starting", strategy="arbitrage")
        self.engine.start_cycle("arbitrage")
        results: list[dict[str, Any]] = []

        if not self.engine.strategy_enabled.get("arbitrage", True):
            self.engine.log_event("info", "Arbitrage strategy disabled", strategy="arbitrage")
            return results

        try:
            # Scan all open markets for arbitrage opportunities
            markets_data = await self.kalshi._get("/markets", params={"limit": 200, "status": "open"})
            markets = markets_data.get('markets', [])

            arb_count = 0
            for market in markets:
                ticker = market.get('ticker', '')
                yes_bid = market.get('yes_bid', 0) or 0
                no_bid = market.get('no_bid', 0) or 0
                yes_ask = market.get('yes_ask', 0) or 0
                no_ask = market.get('no_ask', 0) or 0

                # Skip if no prices
                if not yes_bid or not no_bid or not yes_ask or not no_ask:
                    continue

                # Skip very low liquidity (< 10 volume)
                if market.get('volume', 0) < 10:
                    continue

                # Type 1: Buy both sides for < 100¢ (guaranteed profit)
                buy_both_cost = yes_ask + no_ask
                if buy_both_cost < 99:  # Leave 1¢ buffer for fees
                    profit_cents = 100 - buy_both_cost
                    roi = (profit_cents / buy_both_cost) * 100

                    # Only trade if ROI > 1% (worth the execution risk)
                    if roi > 1.0:
                        self.engine.log_event(
                            "info",
                            f"Arbitrage found: {ticker} - Buy both @ {buy_both_cost}¢, profit {profit_cents}¢ ({roi:.1f}% ROI)",
                            strategy="arbitrage",
                        )

                        # Buy YES side
                        results.append({
                            "ticker": ticker,
                            "side": "yes",
                            "action": "buy",
                            "price_cents": yes_ask,
                            "count": 1,  # Start with 1 contract
                            "strategy": "arbitrage",
                            "edge": roi / 100,
                            "confidence": 1.0,  # Risk-free = 100% confidence
                            "signal_source": "arbitrage_buy_both",
                            "notes": f"Arbitrage: Buy both @ {buy_both_cost}¢, profit {profit_cents}¢",
                        })

                        # Buy NO side
                        results.append({
                            "ticker": ticker,
                            "side": "no",
                            "action": "buy",
                            "price_cents": no_ask,
                            "count": 1,
                            "strategy": "arbitrage",
                            "edge": roi / 100,
                            "confidence": 1.0,
                            "signal_source": "arbitrage_buy_both",
                            "notes": f"Arbitrage: Buy both @ {buy_both_cost}¢, profit {profit_cents}¢",
                        })

                        arb_count += 1

            if arb_count > 0:
                self.engine.log_event(
                    "info",
                    f"Found {arb_count} arbitrage opportunities",
                    strategy="arbitrage",
                )
            else:
                logger.debug("No arbitrage opportunities found")

        except Exception as e:
            self.engine.log_event("error", f"Arbitrage cycle error: {e}", strategy="arbitrage")
            logger.error("Arbitrage cycle error", error=str(e))

        return results

    async def run_nba_props_cycle(self) -> list[dict[str, Any]]:
        """
        Run one NBA player props strategy cycle:
        1. Scan Kalshi for NBA player prop markets
        2. Build enriched player features from SportsDataIO + BallDontLie
        3. Run SmartPredictor to get over/under probabilities
        4. Compare to Kalshi prices and trade on edge > 3%
        """
        self.engine.log_event("info", "NBA props cycle starting", strategy="nba_props")
        self.engine.start_cycle("nba_props")
        results: list[dict[str, Any]] = []

        if not self.engine.strategy_enabled.get("nba_props", True):
            self.engine.log_event("info", "NBA props strategy disabled", strategy="nba_props")
            return results

        try:
            # Step 1: Scan for NBA prop markets
            props_markets = await self.scanner.scan_nba_props_markets()
            self.engine.log_event(
                "info",
                f"Found {len(props_markets)} NBA prop markets",
                strategy="nba_props",
            )

            if not props_markets:
                return results

            # Step 2: Initialize NBA predictor
            loop = asyncio.get_event_loop()
            initialized = await loop.run_in_executor(None, self._ensure_nba_services)
            if not initialized or not self.predictor:
                self.engine.log_event("warning", "NBA predictor not available", strategy="nba_props")
                return results

            # Step 3: Get enriched player features from shared cache
            # Built once per hour, shared with the sportsbook props tab — no duplicate BDL fetches.
            from app.services.nba_analysis_cache import build_nba_analysis
            cache_data = await build_nba_analysis()
            name_to_player: dict[str, dict] = cache_data.get("name_to_player", {})
            self.engine.log_event(
                "info",
                f"NBA analysis cache: {len(name_to_player)} players "
                f"(age={cache_data.get('built_at', 'unknown')})",
                strategy="nba_props",
            )

            # Step 4: Fetch Odds API consensus props for cross-reference
            # {"player_name|prop_type": {consensus_over_prob, line, books_count}}
            odds_props_lookup: dict[str, dict] = {}
            try:
                from app.services.odds_api import get_odds_api
                odds_client = get_odds_api()
                all_odds_props = await odds_client.get_all_todays_props()
                for op in all_odds_props:
                    key = f"{op['player'].lower()}|{op['prop_type']}"
                    if key not in odds_props_lookup or op.get("books_count", 0) > odds_props_lookup[key].get("books_count", 0):
                        odds_props_lookup[key] = op
                self.engine.log_event(
                    "info",
                    f"Odds API props: {len(all_odds_props)} lines across {len(odds_props_lookup)} player/prop combos",
                    strategy="nba_props",
                )
            except Exception as e:
                logger.debug("Odds API props fetch failed for NBA cross-ref", error=str(e))

            # Step 5: Fetch SportsDataIO player news for injury/status context
            sdio_news: dict[str, list[dict]] = {}
            try:
                from app.services.sportsdataio import get_sportsdataio
                sdio = get_sportsdataio()
                news_items = await sdio.get_news()
                for item in news_items:
                    pname = (item.get("Name") or item.get("PlayerName") or "").lower().strip()
                    if pname:
                        sdio_news.setdefault(pname, []).append(item)
                logger.info(f"SportsDataIO news: {len(news_items)} items for {len(sdio_news)} players")
            except Exception as e:
                logger.debug("SportsDataIO news fetch failed", error=str(e))

            # Step 5b: SportsDataIO injury report
            sdio_injuries: dict[str, str] = {}
            try:
                from app.services.sportsdataio import get_sportsdataio as _get_sdio
                _sdio = _get_sdio()
                injury_items = await _sdio.get_injuries()
                for inj in injury_items:
                    pname = (inj.get("Name") or inj.get("PlayerName") or "").lower().strip()
                    status = (inj.get("Status") or "").strip()
                    if pname and status:
                        sdio_injuries[pname] = status
                logger.info(f"SportsDataIO injuries: {len(sdio_injuries)} players with injury status")
            except Exception as e:
                logger.debug("SportsDataIO injuries fetch failed", error=str(e))

            # Step 6: Evaluate each market
            # We need our_prob_yes (P(>=line)) for every candidate to enforce monotonicity,
            # so store it temporarily on the candidate dict.
            raw_results: list[dict[str, Any]] = []
            for market in props_markets:
                try:
                    result = await self._evaluate_nba_prop(
                        market, name_to_player, loop,
                        odds_props_lookup=odds_props_lookup,
                        sdio_news=sdio_news,
                        sdio_injuries=sdio_injuries,
                    )
                    if result:
                        raw_results.append(result)
                except Exception as e:
                    logger.debug("NBA prop eval failed", ticker=market.get("ticker"), error=str(e))

            # Step 7: Enforce monotonicity across lines for the same player+prop.
            # P(>=higher_line) must be <= P(>=lower_line). If not, the model gave
            # contradictory signals (e.g. YES 16+ rebounds but NO 10+ rebounds).
            # Drop the violating candidate — keep only the internally consistent ones.
            from collections import defaultdict as _dd
            # Group by (player_name, prop_type) → list of (line, over_prob_yes, candidate)
            # over_prob_yes = our_prob if side==yes, else 1-our_prob
            _groups: dict[str, list[tuple[float, float, dict]]] = _dd(list)
            for c in raw_results:
                notes = c.get("notes", "")
                # notes format: "{player_name} {prop_type} line={line} pred={pred}"
                try:
                    import re as _re
                    _m = _re.search(r'line=([\d.]+)', notes)
                    _line = float(_m.group(1)) if _m else None
                    # Reconstruct over_prob_yes from side + our_prob
                    _over_p = c["our_prob"] if c["side"] == "yes" else 1.0 - c["our_prob"]
                    # Group key: extract player+prop from notes (everything before " line=")
                    _key = _re.sub(r'\s+line=.*', '', notes).strip()
                    if _line is not None:
                        _groups[_key].append((_line, _over_p, c))
                except Exception:
                    results.append(c)  # can't parse — keep as-is
                    continue

            _dropped = 0
            for _key, _entries in _groups.items():
                if len(_entries) == 1:
                    results.append(_entries[0][2])
                    continue
                # Sort ascending by line
                _entries.sort(key=lambda x: x[0])
                # Enforce: over_prob must be non-increasing as line increases
                # Walk from lowest to highest line; cap each over_prob at the previous one
                _prev_over_p = 1.0
                _keep: list[dict] = []
                for _line, _over_p, _c in _entries:
                    if _over_p > _prev_over_p + 0.02:  # 2% tolerance for model noise
                        # Monotonicity violated — this candidate is incoherent, drop it
                        _dropped += 1
                        logger.info(
                            f"NBA monotonicity drop: {_key} line={_line} "
                            f"over_p={_over_p:.3f} > prev={_prev_over_p:.3f}",
                        )
                    else:
                        _prev_over_p = min(_prev_over_p, _over_p)
                        _keep.append(_c)
                results.extend(_keep)

            if _dropped:
                self.engine.log_event(
                    "info",
                    f"NBA monotonicity filter: dropped {_dropped} incoherent candidates",
                    strategy="nba_props",
                )

        except Exception as e:
            self.engine.log_event("error", f"NBA props cycle failed: {e}", strategy="nba_props")
            logger.error("NBA props cycle error", error=str(e))

        self.engine.log_event(
            "info",
            f"NBA props cycle complete: {len(results)} trades",
            strategy="nba_props",
        )
        return results

    async def _evaluate_nba_prop(
        self,
        market: dict[str, Any],
        name_to_player: dict[str, dict],
        loop: asyncio.AbstractEventLoop,
        odds_props_lookup: dict[str, dict] | None = None,
        sdio_news: dict[str, list[dict]] | None = None,
        sdio_injuries: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        """Evaluate a single NBA prop market using SmartPredictor."""
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        props_info = market.get("nba_props", {})
        prop_type = props_info.get("prop_type")
        player_name = props_info.get("player_name")
        line = props_info.get("line")
        yes_ask = market.get("yes_ask", 0)
        no_ask = market.get("no_ask", 0)
        if not yes_ask and not no_ask:
            return None

        # Safety filter: Max Spread Limit = 5¢
        yes_bid = market.get("yes_bid", 0)
        no_bid = market.get("no_bid", 0)
        if yes_ask > 0 and yes_bid > 0 and (yes_ask - yes_bid) > 5:
            return None
        if no_ask > 0 and no_bid > 0 and (no_ask - no_bid) > 5:
            return None

        # 12-hour time gate: fail closed if we can't confidently parse the game date.
        if not self._nba_prop_within_time_gate(ticker):
            return None

        # Match player name to our data
        player = _find_unique_person_match(name_to_player, player_name)

        if not player:
            logger.debug("Player not found in data", player=player_name)
            return None

        # Skip players with no game log data — predictions would be garbage
        has_stats = any(player.get(k) for k in ("pts_pg", "reb_pg", "ast_pg", "last3_pts", "last3_reb"))
        if not has_stats:
            logger.debug("Skipping player with no game log data", player=player_name)
            return None

        # ── SportsDataIO injury status: skip Out/Doubtful, penalize Questionable ──
        injury_status_detail = ""
        if sdio_injuries and player_name:
            inj_key = _normalize_person_name(player_name)
            inj_status = sdio_injuries.get(inj_key, "")
            if not inj_status:
                inj_status = _find_unique_person_match(sdio_injuries, player_name) or ""
            if inj_status:
                status_upper = inj_status.upper()
                if status_upper in ("OUT", "DOUBTFUL", "IR", "SUSPENDED"):
                    logger.debug(f"Skipping {player_name}: injury status={inj_status}")
                    return None
                elif status_upper in ("QUESTIONABLE", "DAY-TO-DAY", "DTD"):
                    injury_status_detail = f" inj={inj_status}"
                    # Will apply confidence penalty after prediction

        if not self.predictor:
            return None

        # Run prediction (sync, in executor)
        try:
            prediction = await loop.run_in_executor(
                None, self.predictor.predict_prop, player, prop_type, line,
            )
        except Exception as e:
            logger.debug("Prediction failed", player=player_name, prop=prop_type, error=str(e))
            return None

        if not prediction:
            return None

        over_prob = prediction.get("over_probability", 0.5)
        confidence_score = prediction.get("confidence_score", 30)
        predicted_value = prediction.get("predicted_value", 0)

        # Kalshi "yes" typically means "over the line"
        our_prob_yes = over_prob
        confidence = confidence_score / 100.0  # Normalize to 0-1

        # Apply injury status confidence penalty for Questionable/DTD players
        if injury_status_detail:
            confidence = max(0.10, confidence - 0.15)

        # ── Cross-reference: Odds API consensus probability ──────────────────
        # If sportsbooks' consensus agrees with our prediction, boost confidence.
        # If they disagree significantly, reduce confidence.
        odds_detail = ""
        if odds_props_lookup and prop_type and player_name:
            odds_prop = _find_unique_prop_match(odds_props_lookup, player_name, prop_type)

            if odds_prop:
                books_consensus_over = odds_prop.get("consensus_over_prob", 0.5)
                books_count = odds_prop.get("books_count", 0)
                # Agreement = both predict same direction (over vs under)
                our_direction = over_prob > 0.5
                books_direction = books_consensus_over > 0.5
                agreement = our_direction == books_direction
                divergence = abs(over_prob - books_consensus_over)

                if agreement and books_count >= 2:
                    # Both agree: boost confidence proportional to books coverage
                    boost = min(0.12, divergence * 0.3 + (books_count / 10) * 0.05)
                    confidence = min(0.95, confidence + boost)
                    odds_detail = f" books={books_consensus_over:.2f}({books_count}bks) ✓"
                elif not agreement and divergence > 0.08:
                    # Significant disagreement: reduce confidence
                    penalty = min(0.15, divergence * 0.5)
                    confidence = max(0.10, confidence - penalty)
                    odds_detail = f" books={books_consensus_over:.2f}({books_count}bks) ✗"

        # ── SportsDataIO news: injury/status context ───────────────────────
        news_detail = ""
        if sdio_news and player_name:
            player_news = sdio_news.get(_normalize_person_name(player_name), [])
            if not player_news:
                player_news = _find_unique_person_match(sdio_news, player_name) or []

            if player_news:
                # Check for injury/questionable keywords in most recent news
                latest = player_news[0]
                headline = (latest.get("Title") or latest.get("Content") or "").lower()
                injury_keywords = ["out", "questionable", "doubtful", "injured", "sidelined",
                                   "day-to-day", "miss", "surgery", "sprain", "rest", "load"]
                positive_keywords = ["return", "active", "available", "cleared", "full", "healthy"]

                if any(kw in headline for kw in injury_keywords):
                    # Player may be limited/out — reduce confidence significantly
                    confidence = max(0.10, confidence - 0.20)
                    news_detail = " news=INJURY_RISK"
                elif any(kw in headline for kw in positive_keywords):
                    # Player confirmed healthy/active
                    confidence = min(0.95, confidence + 0.05)
                    news_detail = " news=HEALTHY"

        # Calculate Maker Prices
        yes_maker_price = min(yes_bid + 1, yes_ask) if yes_bid > 0 else yes_ask
        no_maker_price = min(no_bid + 1, no_ask) if no_bid > 0 else no_ask

        # Calculate edge
        kalshi_yes_implied = yes_maker_price / 100.0 if yes_maker_price > 0 else 0
        kalshi_no_implied = no_maker_price / 100.0 if no_maker_price > 0 else 0

        yes_edge = our_prob_yes - kalshi_yes_implied if kalshi_yes_implied > 0 else 0
        no_edge = (1.0 - our_prob_yes) - kalshi_no_implied if kalshi_no_implied > 0 else 0

        if yes_edge >= no_edge:
            edge, side = yes_edge, "yes"
            our_prob, kalshi_prob, price_cents = our_prob_yes, kalshi_yes_implied, yes_maker_price
        else:
            edge, side = no_edge, "no"
            our_prob, kalshi_prob, price_cents = 1.0 - our_prob_yes, kalshi_no_implied, no_maker_price

        # Price range filter: skip extreme ends of the market where pricing is reliable
        # YES <20c = longshot the market correctly prices; NO >80c = near-certainty, model over-confident
        if side == "yes" and price_cents < 20:
            return None
        if side == "no" and price_cents > 80:
            return None

        nba_thresh = self.adaptive.get_thresholds("nba_props")

        # Log prediction details for every evaluated prop (debug NO bias)
        self.engine.log_event(
            "info",
            f"NBA pred: {player_name} {prop_type} line={line} pred={predicted_value:.1f} "
            f"over_p={over_prob:.2f} → {side} edge={edge:.1%} conf={confidence:.2f}"
            f"{injury_status_detail}{odds_detail}{news_detail} | {title[:50]}",
            strategy="nba_props",
        )

        if edge < nba_thresh["min_edge"] or confidence < nba_thresh["min_confidence"]:
            return None

        self.engine.record_signal(
            strategy="nba_props", ticker=ticker, side=side,
            our_prob=our_prob, kalshi_prob=kalshi_prob,
            market_title=title, confidence=confidence,
            signal_source="smart_predictor",
            details=f"player={player_name} prop={prop_type} line={line} "
                    f"pred={predicted_value} over_p={over_prob:.3f} "
                    f"agreement={prediction.get('ensemble_agreement', 0):.3f}"
                    f"{odds_detail}{news_detail}",
        )
        self._record_cross_signal("nba_props", ticker, side, our_prob, kalshi_prob, confidence)

        confidence = self._apply_correlation_adjustment("nba_props", ticker, confidence)

        # Return candidate for global ranking (execution happens later)
        return {
            "strategy": "nba_props",
            "ticker": ticker,
            "title": title,
            "side": side,
            "edge": edge,
            "confidence": confidence,
            "our_prob": our_prob,
            "kalshi_prob": kalshi_prob,
            "price_cents": price_cents,
            "signal_source": "smart_predictor",
            "notes": f"{player_name} {prop_type} line={line} pred={predicted_value}",
        }

    async def run_monitor_cycle(self) -> list[dict[str, Any]]:
        """
        Monitor open positions and actively manage them:
        1. Fetch live prices for each position
        2. Calculate unrealized P&L
        3. EXIT if edge has flipped (>5% against us) — cut losers
        4. TAKE PROFIT if unrealized gain > 50% of max profit — lock in wins
        5. ADD TO POSITION if edge has increased and we have capital — press winners
        """
        self.engine.log_event("info", "Position monitor cycle starting", strategy="monitor")
        actions: list[dict[str, Any]] = []

        try:
            # Monitor heartbeat + coarse runtime health snapshot
            self.engine.record_monitor_heartbeat()
            api_ok = True

            db_ok = True
            try:
                _ = self.engine.get_resting_order_count()
            except Exception:
                db_ok = False

            ws_ok = bool(self.ws.get_status().get("connected", False))
            self.engine.set_runtime_health(db_healthy=db_ok, ws_healthy=ws_ok)

            # Reconcile resting orders DB vs broker before per-order checks.
            actions.extend(await self._reconcile_resting_orders())

            # ── 1. Check and Manage Resting Maker Orders ──
            resting_trades = self.engine.get_resting_trades()
            if resting_trades:
                self.engine.log_event("info", f"Checking {len(resting_trades)} resting orders", strategy="monitor")
                now_dt = datetime.now(UTC)
                try:
                    from app.services.kalshi_api import get_kalshi_client
                    client = get_kalshi_client()
                    for rt in resting_trades:
                        if not rt.get("order_id") or rt["order_id"].startswith("PAPER-"):
                            continue
                        
                        # Fetch latest status
                        order_data = await client.get_order(rt["order_id"])
                        order_info = order_data.get("order", order_data)
                        status = order_info.get("status", "pending")
                        
                        if status == "filled":
                            fill_count, actual_price_cents, cost, fee = self.engine._extract_actual_fill_execution(
                                order_info,
                                fallback_count=rt["count"],
                                fallback_price_cents=rt["price_cents"],
                                action=rt.get("action", "buy"),
                            )
                            self.engine.update_trade_status(
                                rt["id"],
                                "filled",
                                fill_count,
                                cost,
                                fee,
                                price_cents=actual_price_cents,
                            )
                            self.engine.log_event("live_trade", f"Resting order filled: {fill_count}x @ {rt['price_cents']}c", strategy=rt["strategy"])
                            actions.append({"action": "filled_resting", "ticker": rt["ticker"]})
                        elif status in ("canceled", "error"):
                            self.engine.update_trade_status(rt["id"], status)
                            self.engine.log_event("warning", f"Resting order {status}", strategy=rt["strategy"])
                        elif status in ("resting", "pending"):
                            # Check age. If > 5 minutes, cancel it.
                            placed_dt = datetime.fromisoformat(rt["timestamp"].replace("Z", "+00:00"))
                            age_mins = (now_dt - placed_dt).total_seconds() / 60.0
                            if age_mins > 5.0:
                                await client.cancel_order(rt["order_id"])
                                self.engine.update_trade_status(rt["id"], "canceled")
                                self.engine.log_event("info", f"Canceled resting order (age {age_mins:.1f}m > 5m)", strategy=rt["strategy"])
                                self._record_resting_cancel(rt["ticker"])
                                actions.append({"action": "cancel_resting", "ticker": rt["ticker"]})

                                # Optional: dynamic repricing after cancel (buy orders only).
                                # Re-submit as a fresh maker order at new bid+1 when enabled.
                                if (
                                    os.environ.get("ENABLE_DYNAMIC_REPRICING", "false").lower() == "true"
                                    and rt.get("action") == "buy"
                                ):
                                    try:
                                        data = await self.kalshi._get(f"/markets/{rt['ticker']}")
                                        market = data.get("market", data)
                                        yes_bid = market.get("yes_bid", 0) or 0
                                        yes_ask = market.get("yes_ask", 0) or 0
                                        no_bid = market.get("no_bid", 0) or 0
                                        no_ask = market.get("no_ask", 0) or 0

                                        if rt.get("side") == "yes":
                                            new_price = min(yes_bid + 1, yes_ask) if yes_bid > 0 and yes_ask > 0 else (yes_ask or yes_bid)
                                        else:
                                            new_price = min(no_bid + 1, no_ask) if no_bid > 0 and no_ask > 0 else (no_ask or no_bid)

                                        if new_price and int(new_price) > 0:
                                            repriced = await self.engine.execute_trade(
                                                strategy=rt["strategy"],
                                                ticker=rt["ticker"],
                                                side=rt["side"],
                                                count=rt["count"],
                                                price_cents=int(new_price),
                                                signal_source="monitor_reprice",
                                                notes=f"Repriced after 5m timeout (prev_order={rt['order_id']})",
                                            )
                                            if repriced.get("status") in ("resting", "filled"):
                                                actions.append({"action": "reprice_resting", "ticker": rt["ticker"], "price_cents": int(new_price)})
                                    except Exception as repr_err:
                                        logger.warning("Resting order repricing failed", ticker=rt.get("ticker"), error=str(repr_err))
                except Exception as e:
                    logger.warning("Resting order check failed", error=str(e))
                    api_ok = False

            # ── 2. Monitor Open Positions ──
            # ALWAYS use Kalshi API as source of truth for side/contracts.
            # The local DB can be wrong (boot cycle places YES buys on tickers
            # we already hold NO, container rebuilds lose data, etc.).
            # DB records are only used for metadata (avg_our_prob, strategy).
            positions: list[dict[str, Any]] = []
            db_positions = self.engine.get_open_positions()
            db_by_ticker: dict[str, dict] = {p["ticker"]: p for p in db_positions}

            try:
                kalshi_pos = await self.kalshi.get_positions()
                for mp in kalshi_pos.get("market_positions", []):
                    pos_qty = mp.get("position", 0)
                    if pos_qty == 0:
                        continue
                    ticker = mp.get("ticker", "")

                    # Merge with DB metadata if available
                    db_rec = db_by_ticker.get(ticker, {})
                    pos_record = self._build_live_position_record(mp, db_rec)
                    side = pos_record["side"]
                    qty = pos_record["contracts"]
                    if db_rec and db_rec.get("side") != side:
                        logger.warning(
                            "DB/Kalshi side mismatch — using Kalshi truth",
                            ticker=ticker, db_side=db_rec.get("side"),
                            kalshi_side=side, kalshi_qty=qty,
                        )

                    positions.append(pos_record)
                self.engine.set_broker_positions_snapshot([
                    {
                        "ticker": p["ticker"],
                        "exposure_dollars": p["total_cost"],
                        "strategy": p["strategy"],
                    }
                    for p in positions
                ])
                if positions:
                    logger.info("Monitor positions from Kalshi API", count=len(positions))
            except Exception as e:
                logger.warning("Kalshi API positions failed, falling back to DB", error=str(e))
                positions = db_positions  # Last resort — use DB as-is

            if not positions:
                return actions

            # Build mapping from event ticker to specific market ticker
            # (reuses positions already loaded from Kalshi API above)
            event_to_market_ticker: dict[str, str] = {}
            for p in positions:
                market_ticker = p["ticker"]
                parts = market_ticker.rsplit("-", 1)
                if len(parts) == 2 and (parts[1].startswith("B") or parts[1].startswith("T")):
                    event_ticker = parts[0]
                    event_to_market_ticker[event_ticker] = market_ticker

            total_exposure = self.engine.get_total_exposure()
            self.engine.log_event(
                "info",
                f"Monitoring {len(positions)} positions, total exposure: ${total_exposure:.2f}",
                strategy="monitor",
            )

            # Subscribe open position tickers to WebSocket for real-time updates
            pos_tickers = [p["ticker"] for p in positions]
            if self.ws.connected:
                try:
                    await self.ws.subscribe(pos_tickers)
                except Exception:
                    pass

            # Track tickers we've already tried to exit to prevent repeated sells
            exited_this_cycle: set[str] = set()

            for pos in positions:
                ticker = pos["ticker"]

                # Skip if we already tried to exit this ticker this cycle
                if ticker in exited_this_cycle:
                    continue

                # Cross-cycle guard: skip if exited in a previous cycle within 30 min
                import time as _time  # noqa: already at module level if available
                _exit_ts = self._exited_tickers.get(ticker, 0)
                if _exit_ts and (_time.time() - _exit_ts) < 1800:
                    logger.info(
                        "Skipping ticker (exited in recent cycle)",
                        ticker=ticker, mins_ago=round((_time.time() - _exit_ts) / 60, 1),
                    )
                    continue

                # Use specific market ticker for API calls (event tickers 404)
                market_ticker = event_to_market_ticker.get(ticker, ticker)
                try:
                    # Always use REST API for monitor — WS snapshots can be
                    # stale/empty right after subscribe, causing wrong marks.
                    await asyncio.sleep(0.3)
                    data = await self.kalshi._get(f"/markets/{market_ticker}")
                    market = data.get("market", data)
                    if not market:
                        logger.warning("Empty market data from REST", ticker=market_ticker)
                        continue
                    yes_bid = market.get("yes_bid", 0) or 0
                    yes_ask = market.get("yes_ask", 0) or 0
                    no_bid = market.get("no_bid", 0) or 0
                    no_ask = market.get("no_ask", 0) or 0
                    last_price = market.get("last_price", 0) or 0
                    mkt_status = market.get("status", "")

                    mark_price, exit_price = self._compute_side_market_prices(
                        side=pos["side"],
                        yes_bid=yes_bid,
                        yes_ask=yes_ask,
                        no_bid=no_bid,
                        no_ask=no_ask,
                        last_price=last_price,
                        avg_entry_cents=pos["avg_entry_cents"],
                    )

                    unrealized_pnl = self._compute_position_pnl(
                        contracts=pos["contracts"],
                        price_cents=mark_price,
                        total_cost=pos["total_cost"],
                        total_fees=pos["total_fees"],
                    )
                    liquidation_pnl = self._compute_position_pnl(
                        contracts=pos["contracts"],
                        price_cents=exit_price,
                        total_cost=pos["total_cost"],
                        total_fees=pos["total_fees"],
                    )

                    logger.info(
                        "Position MTM",
                        ticker=ticker, side=pos["side"], qty=pos["contracts"],
                        mark=mark_price, cost=pos["total_cost"],
                        pnl=unrealized_pnl, avg_entry=pos["avg_entry_cents"],
                    )

                    # Current edge vs our original probability
                    current_edge = 0.0
                    if pos["side"] == "yes" and yes_ask > 0:
                        current_edge = round(pos["avg_our_prob"] - (yes_ask / 100.0), 4)
                    elif pos["side"] == "no" and no_ask > 0:
                        current_edge = round(pos["avg_our_prob"] - (no_ask / 100.0), 4)

                    # Check if market is settled/finalized
                    if mkt_status in ("finalized", "settled", "closed"):
                        continue

                    # ── DECISION 0a: STOP-LOSS — position down >50% of cost ──
                    # Catches cases where original signal was wrong but edge
                    # never technically "flips" (e.g., broken crypto bracket trades)
                    if pos["total_cost"] > 0 and liquidation_pnl < -(pos["total_cost"] * 0.50):
                        if exit_price > 0:
                            await self.engine.exit_trade(
                                strategy=pos["strategy"],
                                ticker=ticker,
                                side=pos["side"],
                                count=pos["contracts"],
                                price_cents=exit_price,
                                reason=f"stop_loss pnl=${liquidation_pnl:+.2f} ({liquidation_pnl/pos['total_cost']:.0%} of cost)",
                            )
                            self.engine.log_event(
                                "paper_trade",
                                f"STOP-LOSS {pos['side'].upper()} {pos['contracts']}x {ticker} @ {exit_price}c — down ${liquidation_pnl:+.2f}",
                                strategy="monitor",
                            )
                            actions.append({"action": "exit", "reason": "stop_loss", "ticker": ticker,
                                            "pnl": liquidation_pnl})
                            exited_this_cycle.add(ticker)
                            self._exited_tickers[ticker] = _time.time()
                            continue

                    # ── DECISION 0b: DEAD CONTRACT — mark price near zero ──
                    # If our side is trading at <=3c, the position is effectively dead
                    if 0 < exit_price <= 3:
                        await self.engine.exit_trade(
                            strategy=pos["strategy"],
                            ticker=ticker,
                            side=pos["side"],
                            count=pos["contracts"],
                            price_cents=exit_price,
                            reason=f"dead_contract mark={exit_price}c",
                        )
                        self.engine.log_event(
                            "paper_trade",
                            f"DEAD CONTRACT {pos['side'].upper()} {pos['contracts']}x {ticker} @ {exit_price}c — P&L: ${liquidation_pnl:+.2f}",
                            strategy="monitor",
                        )
                        actions.append({"action": "exit", "reason": "dead_contract", "ticker": ticker,
                                        "pnl": liquidation_pnl})
                        exited_this_cycle.add(ticker)
                        self._exited_tickers[ticker] = _time.time()
                        continue
                        
                    # ── DECISION 0c: DYNAMIC TAKE PROFIT (95c rule) ──
                    # If YES reaches 95c, probability of winning is very high. 
                    # Instead of waiting days for settlement to free up $1.00, sell now for 0.95 
                    # to free up the capital immediately for other trades.
                    if exit_price >= 95 and liquidation_pnl > 0:
                        await self.engine.exit_trade(
                            strategy=pos["strategy"],
                            ticker=ticker,
                            side=pos["side"],
                            count=pos["contracts"],
                            price_cents=exit_price,
                            reason=f"dynamic_take_profit mark={exit_price}c (freeing capital)",
                        )
                        self.engine.log_event(
                            "paper_trade",
                            f"DYNAMIC TP {pos['side'].upper()} {pos['contracts']}x {ticker} @ {exit_price}c — P&L: ${liquidation_pnl:+.2f}",
                            strategy="monitor",
                        )
                        actions.append({"action": "take_profit", "ticker": ticker,
                                        "pnl": liquidation_pnl, "profit_pct": 1.0})
                        exited_this_cycle.add(ticker)
                        self._exited_tickers[ticker] = _time.time()
                        continue

                    # ── DECISION 1: EXIT — edge has flipped significantly ──
                    if current_edge < -0.05 and exit_price > 0:
                        await self.engine.exit_trade(
                            strategy=pos["strategy"],
                            ticker=ticker,
                            side=pos["side"],
                            count=pos["contracts"],
                            price_cents=exit_price,
                            reason=f"edge_flipped edge={current_edge:.3f}",
                        )
                        self.engine.log_event(
                            "paper_trade",
                            f"EXIT {pos['side'].upper()} {pos['contracts']}x {ticker} @ {exit_price}c — edge flipped to {current_edge:.1%}, P&L: ${liquidation_pnl:+.2f}",
                            strategy="monitor",
                        )
                        actions.append({"action": "exit", "reason": "edge_flipped", "ticker": ticker,
                                        "current_edge": current_edge, "pnl": liquidation_pnl})
                        exited_this_cycle.add(ticker)
                        self._exited_tickers[ticker] = _time.time()
                        continue

                    # ── DECISION 1b: WEATHER OBS EARLY EXIT — DISABLED ──
                    # Previously this fired every 2-min cycle and kept selling NO,
                    # which on Kalshi creates new YES exposure. This caused the SFO
                    # 30-contract blowup. Weather positions should be held to settlement
                    # (they settle within 24h) or exited via stop-loss/edge-flip above.

                    # ── DECISION 2 & 3: OPPORTUNITY COST & DYNAMIC TAKE-PROFIT ──
                    # Instead of an automatic take-profit when hitting 90c or 50% max profit,
                    # we do an A/B opportunity cost analysis. If the market is moving in our favor,
                    # we should only sell if the capital is better deployed elsewhere.
                    max_profit = pos.get("max_profit", 0) or 0
                    
                    if max_profit > 0 and liquidation_pnl > 0 and exit_price > 0:
                        profit_pct = liquidation_pnl / max_profit
                        
                        # Compare our stored thesis value to the executable exit price.
                        # Using the same current price as both probability and liquidation
                        # value cancels to ~0 and makes this branch meaningless.
                        thesis_prob = pos.get("avg_our_prob", 0.0) or 0.0
                        ev_of_holding = self._compute_hold_ev_cents(
                            our_prob=thesis_prob,
                            exit_price=exit_price,
                        ) / 100.0
                        
                        # If the EV of holding the remaining position is negative or very low, AND we have 
                        # captured a significant chunk of profit, it makes sense to exit.
                        # Also check if we are constrained on capital.
                        try:
                            available_capital = self.engine.get_effective_bankroll()
                        except Exception:
                            available_capital = 100.0 # fallback
                            
                        should_take_profit = False
                        reason = ""
                        
                        if exit_price >= 95:
                            should_take_profit = True
                            reason = "locked_in_guaranteed_profit"
                        elif ev_of_holding < 0 and profit_pct > 0.50:
                            should_take_profit = True
                            reason = "ev_holding_negative"
                        elif available_capital < 50.0 and profit_pct > 0.80:
                            should_take_profit = True
                            reason = "rotate_capital_opportunity_cost"
                            
                        if should_take_profit:
                            await self.engine.exit_trade(
                                strategy=pos["strategy"],
                                ticker=ticker,
                                side=pos["side"],
                                count=pos["contracts"],
                                price_cents=exit_price,
                                reason=f"dynamic_take_profit ({reason}) pnl=${liquidation_pnl:+.2f}",
                            )
                            self.engine.log_event(
                                "paper_trade",
                                f"DYNAMIC TAKE-PROFIT {pos['side'].upper()} {pos['contracts']}x {ticker} @ {exit_price}c — {reason}, P&L: ${liquidation_pnl:+.2f}",
                                strategy="monitor",
                            )
                            actions.append({"action": "dynamic_take_profit", "ticker": ticker,
                                            "reason": reason, "pnl": liquidation_pnl})
                            exited_this_cycle.add(ticker)
                            self._exited_tickers[ticker] = _time.time()
                            continue

                    # ── DECISION 4: ADD TO POSITION — DISABLED for weather ──
                    # Weather markets settle in 24h. No need to DCA into them.
                    # The monitor_add was the main cause of position accumulation
                    # (e.g., Denver Low got 34 contracts from repeated monitor_add).
                    if pos.get("strategy") == "weather":
                        continue

                    # HARD BLOCK: Never add to soccer positions (losing strategy)
                    soccer_keywords = ["EPL", "LALIGA", "UCL", "SERIE", "BUNDESLIGA", "LIGUE1", 
                                       "MLS", "LIGAMX", "BRASILEIRO", "FACUP", "EWSL", "SLGREECE",
                                       "EREDIVISIE", "LIGAPORTUGAL", "SCOTTISHPREM", "SUPERLIG"]
                    if any(kw in ticker.upper() for kw in soccer_keywords):
                        continue
                    
                    original_edge = pos.get("avg_entry_edge", 0) or 0
                    edge_increase = current_edge - original_edge
                    # Add if edge increased by >3% AND we still have positive edge >5%
                    if edge_increase > 0.03 and current_edge > 0.05:
                        add_price = yes_ask if pos["side"] == "yes" else no_ask
                        if add_price > 0:
                            add_count = self.engine.calculate_position_size(
                                strategy=pos["strategy"],
                                edge=current_edge,
                                price_cents=add_price,
                                confidence=0.8,  # Slightly conservative on adds
                                ticker=ticker,
                                signal_source="weather_consensus" if pos["strategy"] == "weather" else "",
                            )
                            if add_count > 0:
                                trade = await self.engine.execute_trade(
                                    strategy=pos["strategy"],
                                    ticker=ticker,
                                    side=pos["side"],
                                    count=add_count,
                                    price_cents=add_price,
                                    our_prob=pos["avg_our_prob"],
                                    kalshi_prob=add_price / 100.0,
                                    market_title=pos.get("title", ""),
                                    signal_source="monitor_add",
                                    notes=f"Adding to position: edge {original_edge:.1%} → {current_edge:.1%}",
                                )
                                if trade.get("status") == "filled":
                                    self.engine.log_event(
                                        "paper_trade",
                                        f"ADD {pos['side'].upper()} +{add_count}x {ticker} @ {add_price}c — edge increased {original_edge:.1%} → {current_edge:.1%}",
                                        strategy="monitor",
                                    )
                                    actions.append({"action": "add", "ticker": ticker,
                                                    "added": add_count, "current_edge": current_edge})

                except Exception as e:
                    self.engine.log_event("warning", f"Monitor failed for {ticker}: {e}", strategy="monitor")
                    logger.warning("Position monitor failed for ticker", ticker=ticker, error=str(e))
                    api_ok = False

            self.engine.set_runtime_health(api_healthy=api_ok)

        except Exception as e:
            self.engine.set_runtime_health(api_healthy=False)
            import traceback
            tb = traceback.format_exc()
            self.engine.log_event("error", f"Monitor cycle failed: {e}\n{tb}", strategy="monitor")
            logger.error("Monitor cycle error", error=str(e), traceback=tb)

        summary_parts = []
        exits = [a for a in actions if a["action"] == "exit"]
        profits = [a for a in actions if a["action"] == "take_profit"]
        adds = [a for a in actions if a["action"] == "add"]
        if exits:
            summary_parts.append(f"{len(exits)} exits")
        if profits:
            summary_parts.append(f"{len(profits)} profit-takes")
        if adds:
            summary_parts.append(f"{len(adds)} adds")
        if summary_parts:
            self.engine.log_event("info", f"Monitor: {', '.join(summary_parts)}", strategy="monitor")
        else:
            self.engine.log_event("info", f"Monitor: {len(positions)} positions checked, no actions needed", strategy="monitor")

        return actions

    async def run_settlement_cycle(self) -> int:
        """
        Check all unsettled trades against Kalshi market status.
        When a market is finalized/settled, record the result and P&L.
        Returns the number of trades settled.
        """
        unsettled = self.engine.get_unsettled_trades()
        if not unsettled:
            return 0

        # Group trades by ticker to minimize API calls
        ticker_trades: dict[str, list[dict[str, Any]]] = {}
        for trade in unsettled:
            ticker_trades.setdefault(trade["ticker"], []).append(trade)

        settled_count = 0
        total_pnl = 0.0

        for ticker, trades in ticker_trades.items():
            try:
                await asyncio.sleep(0.5)
                data = await self.kalshi._get(f"/markets/{ticker}")
                market = data.get("market", data)
                mkt_status = market.get("status", "")

                if mkt_status not in ("finalized", "settled", "closed"):
                    continue

                # Kalshi result field: "yes" or "no"
                result = market.get("result", "")
                if not result:
                    continue

                # Record closing price for CLV before settlement
                last_price = market.get("last_price", 0) or 0
                yes_bid = market.get("yes_bid", 0) or 0
                yes_ask = market.get("yes_ask", 0) or 0
                closing_yes = last_price or yes_bid or yes_ask

                for trade in trades:
                    # CLV: record the market's closing price for this trade
                    try:
                        if trade.get("side") == "yes":
                            closing_price = closing_yes
                        else:
                            closing_price = 100 - closing_yes if closing_yes > 0 else 0
                        if closing_price > 0:
                            self.engine.record_closing_price(trade["id"], int(closing_price))
                    except Exception:
                        pass

                    settle_result = self.engine.settle_trade(trade["id"], result)
                    if "error" not in settle_result:
                        settled_count += 1
                        pnl = settle_result.get("pnl", 0)
                        total_pnl += pnl
                        # Send to OpenClaw for post-trade analysis
                        try:
                            trade_for_review = {**trade, **settle_result, "market_title": market.get("title", "")}
                            asyncio.create_task(self.trade_analyzer.analyze_trade(trade_for_review, market_result=result))
                        except Exception:
                            pass
                        # Discord alert + SSE event for settlement
                        won = pnl > 0
                        try:
                            asyncio.create_task(send_discord_notification(
                                title=f"{'✅ WIN' if won else '❌ LOSS'}: {trade.get('ticker', '')}",
                                message=(
                                    f"**Strategy:** {trade.get('strategy', '')}\n"
                                    f"**Side:** {trade.get('side', '').upper()} | **Result:** {result.upper()}\n"
                                    f"**P&L:** ${pnl:+.2f}\n"
                                    f"**Title:** {market.get('title', '')[:80]}"
                                ),
                                color=0x2ecc71 if won else 0xe74c3c,
                            ))
                        except Exception:
                            pass
                        try:
                            get_event_bus().publish("settlement", {
                                "ticker": trade.get("ticker", ""),
                                "strategy": trade.get("strategy", ""),
                                "side": trade.get("side", ""),
                                "result": result, "won": won,
                                "pnl": round(pnl, 2),
                                "title": market.get("title", "")[:80],
                            })
                        except Exception:
                            pass

                        # Record signal component outcomes for quality scoring
                        try:
                            self.signal_scorer.record_signal_outcome(
                                strategy=trade.get("strategy", ""),
                                signal_components=self._extract_signal_components(trade),
                                won=pnl > 0,
                            )
                        except Exception:
                            pass

                        # Weather API accuracy feedback — populate dynamic weights
                        if trade.get("strategy") == "weather":
                            try:
                                self._record_weather_api_accuracy(trade, market)
                            except Exception:
                                pass

            except Exception as e:
                logger.warning("Settlement check failed for ticker", ticker=ticker, error=str(e))

        if settled_count > 0:
            self.engine.log_event(
                "settlement",
                f"Settlement cycle: {settled_count} trades settled, total P&L: ${total_pnl:+.2f}",
                strategy="monitor",
            )
            # Update adaptive thresholds based on new settlement data
            try:
                changes = self.adaptive.update_thresholds()
                if changes:
                    for strat, chg in changes.items():
                        self.engine.log_event(
                            "info",
                            f"Threshold adjusted: {strat} edge {chg['old_edge']:.1%}→{chg['new_edge']:.1%} "
                            f"conf {chg['old_conf']:.1%}→{chg['new_conf']:.1%} ({chg['reason']})",
                            strategy=strat,
                        )
            except Exception as e:
                logger.debug("Adaptive threshold update failed", error=str(e))

        return settled_count

    def _record_weather_api_accuracy(self, trade: dict[str, Any], market: dict[str, Any]) -> None:
        """After a weather trade settles, record per-source forecast accuracy."""
        ticker = trade.get("ticker", "")
        # Parse city code and market type from ticker (e.g., KXHIGHTNYC-26FEB19-T45)
        m = re.match(r'KX(HIGH|LOW)T?([A-Z]{2,4})-', ticker)
        if not m:
            return
        temp_type = m.group(1)  # HIGH or LOW
        city_code = m.group(2)
        market_type = "low_temp" if temp_type == "LOW" else "high_temp"

        # Try to get the actual temperature from Kalshi's settlement data
        # Kalshi weather markets have a "floor_strike" for the settled bracket
        # For greater/less markets, we can infer actual from result + strike
        # Best effort: use the market's strike value as a reference point
        # The floor_strike of the winning YES bracket ~ actual temp
        floor_strike = market.get("floor_strike")
        cap_strike = market.get("cap_strike")
        actual_temp = None
        if floor_strike is not None and cap_strike is not None:
            actual_temp = (floor_strike + cap_strike) / 2.0
        elif floor_strike is not None:
            actual_temp = floor_strike
        elif cap_strike is not None:
            actual_temp = cap_strike

        if actual_temp is None:
            return

        # Look up original per-source forecasts from signal details
        details_str = self.engine.get_signal_details(ticker, strategy="weather")
        if not details_str:
            return
        try:
            details = json.loads(details_str)
        except (json.JSONDecodeError, TypeError):
            return

        source_forecasts = details.get("source_forecasts", [])
        if not source_forecasts:
            return

        # Extract target date from ticker (e.g., KXHIGHTNYC-26FEB19 → 2026-02-19)
        date_match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})', ticker)
        target_date = ""
        if date_match:
            try:
                day = date_match.group(1)
                mon = date_match.group(2)
                yr = date_match.group(3)
                from datetime import datetime as dt
                parsed = dt.strptime(f"{day}{mon}{yr}", "%d%b%y")
                target_date = parsed.strftime("%Y-%m-%d")
            except ValueError:
                target_date = ""

        self.weather.record_forecast_accuracy(
            city=city_code,
            target_date=target_date,
            market_type=market_type,
            actual_temp=actual_temp,
            source_forecasts=source_forecasts,
        )
        logger.info(
            "Recorded weather API accuracy",
            city=city_code,
            market_type=market_type,
            actual_temp=actual_temp,
            sources=len(source_forecasts),
        )

    async def get_positions_with_market_data(self) -> list[dict[str, Any]]:
        """
        Get open positions enriched with live market data.
        Called by the API for the frontend display.

        IMPORTANT: Uses Kalshi API as the sole source of truth for
        side, contracts, and ticker. The local DB is only used for
        metadata (strategy, avg_our_prob, signal_source, etc.).
        This prevents the dashboard from showing phantom/stale DB
        positions that don't exist on Kalshi.
        """
        # ── 1. Kalshi API = source of truth for positions ──
        db_positions = self.engine.get_open_positions()
        db_by_ticker: dict[str, dict] = {p["ticker"]: p for p in db_positions}

        positions: list[dict[str, Any]] = []
        try:
            kalshi_pos = await self.kalshi.get_positions()
            for mp in kalshi_pos.get("market_positions", []):
                pos_qty = mp.get("position", 0)
                if pos_qty == 0:
                    continue
                ticker = mp.get("ticker", "")

                # Merge DB metadata if available
                db_rec = db_by_ticker.get(ticker, {})
                positions.append(self._build_live_position_record(mp, db_rec, include_title=True))
        except Exception as e:
            logger.warning("Kalshi API positions failed for dashboard, falling back to DB", error=str(e))
            positions = db_positions  # Last resort

        # ── 2. Enrich each position with live market data ──
        for pos in positions:
            market_ticker = pos["ticker"]
            try:
                await asyncio.sleep(0.15)
                data = await self.kalshi._get(f"/markets/{market_ticker}")
                market = data.get("market", data)

                yes_bid = market.get("yes_bid", 0) or 0
                yes_ask = market.get("yes_ask", 0) or 0
                no_bid = market.get("no_bid", 0) or 0
                no_ask = market.get("no_ask", 0) or 0

                pos["current_yes_bid"] = yes_bid
                pos["current_yes_ask"] = yes_ask
                pos["current_no_bid"] = no_bid
                pos["current_no_ask"] = no_ask

                # Use title from market data if we don't have one
                if pos.get("title") == pos["ticker"] and market.get("title"):
                    pos["title"] = market["title"]

                last_price = market.get("last_price", 0) or 0
                mark_price, _ = self._compute_side_market_prices(
                    side=pos["side"],
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=no_bid,
                    no_ask=no_ask,
                    last_price=last_price,
                    avg_entry_cents=pos["avg_entry_cents"],
                )

                pos["mark_price_cents"] = mark_price
                mark_value = pos["contracts"] * mark_price / 100.0
                pos["unrealized_pnl"] = round(mark_value - pos["total_cost"] - pos["total_fees"], 2)

                if pos["side"] == "yes" and yes_ask > 0:
                    pos["current_edge"] = round(pos["avg_our_prob"] - (yes_ask / 100.0), 4)
                elif pos["side"] == "no" and no_ask > 0:
                    pos["current_edge"] = round(pos["avg_our_prob"] - (no_ask / 100.0), 4)

                mkt_status = market.get("status", "")
                if mkt_status in ("finalized", "settled", "closed"):
                    pos["status"] = "settled"

            except Exception as e:
                logger.debug("Position live price failed", ticker=market_ticker, error=str(e))

        return positions

    # ── Agent Lifecycle ────────────────────────────────────────────

    async def _staggered_loop(self, loop_fn, delay_seconds: int) -> None:
        """Wait `delay_seconds` then run the loop. Prevents all strategies from hitting the API at once."""
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        await loop_fn()

    async def start(self) -> None:
        """Start the autonomous agent loops."""
        if self._running:
            return

        self._running = True
        self.engine.log_event("info", "Agent starting")
        logger.info("Kalshi agent starting", paper_mode=self.engine.paper_mode)

        # Live mode: sync bankroll to actual Kalshi portfolio value (source of truth)
        # Kalshi "balance" = uninvested cash only. True bankroll = cash + position exposure.
        if not self.engine.paper_mode:
            balance_verified = False
            for attempt in range(3):
                try:
                    balance_data = await self.kalshi.get_balance()
                    balance_cents = balance_data.get("balance", 0)
                    cash_dollars = balance_cents / 100.0

                    # Sum up exposure from all open positions
                    position_exposure = 0.0
                    try:
                        kalshi_pos = await self.kalshi.get_positions()
                        broker_positions_snapshot: list[dict[str, Any]] = []
                        for mp in kalshi_pos.get("market_positions", []):
                            if mp.get("position", 0) != 0:
                                exposure = float(mp.get("market_exposure_dollars", 0) or 0)
                                position_exposure += exposure
                                broker_positions_snapshot.append({
                                    "ticker": mp.get("ticker", ""),
                                    "exposure_dollars": exposure,
                                })
                        self.engine.set_broker_positions_snapshot(broker_positions_snapshot)
                    except Exception:
                        pass

                    total_portfolio = cash_dollars + position_exposure
                    self.engine.log_event(
                        "info",
                        f"Kalshi portfolio: ${total_portfolio:.2f} "
                        f"(cash=${cash_dollars:.2f} + positions=${position_exposure:.2f}) "
                        f"env BANKROLL=${self.engine.bankroll:.2f}",
                    )
                    logger.info(
                        "Kalshi portfolio synced",
                        cash=cash_dollars,
                        positions=position_exposure,
                        total=total_portfolio,
                        env_bankroll=self.engine.bankroll,
                    )
                    if abs(total_portfolio - self.engine.bankroll) > 1.0:
                        self.engine.log_event(
                            "warning",
                            f"Syncing bankroll: ${self.engine.bankroll:.2f} → ${total_portfolio:.2f} (actual portfolio value)",
                        )
                    self.engine.sync_bankroll(total_portfolio)
                    balance_verified = True
                    break
                except Exception as e:
                    self.engine.log_event("warning", f"Balance check attempt {attempt+1}/3 failed: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2)
            if not balance_verified:
                self.engine.log_event(
                    "error",
                    "CRITICAL: Could not verify Kalshi balance after 3 attempts. "
                    "Activating kill switch to prevent trading with wrong bankroll.",
                )
                self.engine.kill_switch = True

        # Start WebSocket connection for real-time price updates
        try:
            ws_ok = await self.ws.connect()
            if ws_ok:
                self.engine.log_event("info", "Kalshi WebSocket connected")
            else:
                self.engine.log_event("warning", "Kalshi WebSocket failed to connect, using REST fallback")
        except Exception as e:
            self.engine.log_event("warning", f"Kalshi WebSocket error: {e}, using REST fallback")

        # Start monitor loop IMMEDIATELY — position management (sell 99c winners,
        # stop-loss, etc.) should never wait for the boot cycle to finish evaluating signals.
        self._monitor_task = asyncio.create_task(self._monitor_loop(sleep_first=False))

        # Boot cycle: run ALL strategies once immediately, rank globally, deploy best signals.
        # After this completes the independent loops take over on their normal schedules.
        asyncio.create_task(self._boot_then_start_loops())

    async def stop(self) -> None:
        """Stop the autonomous agent loops."""
        self._running = False
        self.engine.log_event("info", "Agent stopping")

        for task in [self._weather_task, self._main_task, self._crypto_task, self._monitor_task, self._health_task]:
            if task:
                task.cancel()

        try:
            await self.ws.close()
        except Exception:
            pass

        logger.info("Kalshi agent stopped")

    async def _boot_then_start_loops(self) -> None:
        """
        On agent startup: run ALL strategies concurrently, merge every candidate
        into one pool, rank globally by edge×confidence, and deploy the best signals.
        Then start the independent recurring loops.
        """
        self.engine.log_event("info", "Boot cycle: running all strategies sequentially")
        try:
            self._refresh_performance_model_if_due(force=True)
            all_candidates: list[dict[str, Any]] = []

            for name, coro in [
                ("arbitrage", self.run_arbitrage_cycle()),
                ("weather", self.run_weather_cycle()),
                ("crypto", self.run_crypto_cycle()),
                ("sports", self.run_sports_cycle()),
                ("nba_props", self.run_nba_props_cycle()),
            ] + ([
                ("finance", self.run_finance_cycle()),
                ("econ", self.run_econ_cycle()),
            ] if self._is_us_market_hours() else []):
                try:
                    result = await coro
                    tradeable = [c for c in result if c.get("action") != "skip"]
                    all_candidates.extend(tradeable)
                    self.engine.log_event("info", f"Boot cycle: {name} → {len(tradeable)} candidates")
                except Exception as e:
                    self.engine.log_event("warning", f"Boot cycle: {name} error: {e}")

            self.engine.log_event(
                "info",
                f"Boot cycle: {len(all_candidates)} total candidates across all strategies",
            )

            if all_candidates:
                await self.execute_ranked_signals(all_candidates)

        except Exception as e:
            self.engine.log_event("error", f"Boot cycle failed: {e}")
            logger.error("Boot cycle error", error=str(e))

        # Start recurring loops in sleep-first mode so they don't immediately
        # re-run what the boot cycle just executed.
        # NOTE: monitor loop is already started before boot cycle (see start())
        self._arbitrage_task = asyncio.create_task(self._arbitrage_loop(sleep_first=True))
        self._weather_task = asyncio.create_task(self._weather_loop(sleep_first=True))
        self._crypto_task = asyncio.create_task(self._crypto_loop(sleep_first=True))
        self._main_task = asyncio.create_task(self._main_strategy_loop(sleep_first=True))

        # Start health watchdog
        self._health_task = asyncio.create_task(self._health_watchdog())

        # Notify Discord that agent has started
        try:
            asyncio.create_task(send_discord_notification(
                title="🟢 Agent Started",
                message=f"Paper mode: {self.engine.paper_mode}\nBankroll: ${self.engine.bankroll:.2f}",
                color=0x2ecc71,
            ))
        except Exception:
            pass

    async def _health_watchdog(self) -> None:
        """Health monitoring loop — runs every 5 minutes.
        - Checks if strategy loops are alive, restarts crashed ones
        - Pings Kalshi API to verify connectivity
        - Alerts Discord on connectivity loss or task recovery
        - Updates engine health snapshot
        """
        await asyncio.sleep(3 * 60)  # initial delay — let loops stabilize
        while self._running:
            try:
                recovered = []

                # ── Task Health: restart crashed loops ──
                task_map = {
                    "weather": (self._weather_task, self._weather_loop),
                    "crypto": (self._crypto_task, self._crypto_loop),
                    "main": (self._main_task, self._main_strategy_loop),
                    "monitor": (self._monitor_task, self._monitor_loop),
                    "arbitrage": (getattr(self, "_arbitrage_task", None), self._arbitrage_loop),
                }
                for name, (task, loop_fn) in task_map.items():
                    if task is not None and task.done():
                        exc = task.exception() if not task.cancelled() else None
                        self.engine.log_event(
                            "warning",
                            f"Health watchdog: {name} loop died"
                            + (f" ({exc})" if exc else " (cancelled)"),
                            strategy="health",
                        )
                        # Restart the loop
                        new_task = asyncio.create_task(loop_fn(sleep_first=False))
                        if name == "weather":
                            self._weather_task = new_task
                        elif name == "crypto":
                            self._crypto_task = new_task
                        elif name == "main":
                            self._main_task = new_task
                        elif name == "monitor":
                            self._monitor_task = new_task
                        elif name == "arbitrage":
                            self._arbitrage_task = new_task
                        recovered.append(name)

                if recovered:
                    msg = f"Auto-recovered loops: {', '.join(recovered)}"
                    self.engine.log_event("warning", msg, strategy="health")
                    try:
                        asyncio.create_task(send_discord_notification(
                            title="🔄 Agent Auto-Recovery",
                            message=msg,
                            color=0xf39c12,
                        ))
                    except Exception:
                        pass

                # ── API Health Ping ──
                api_ok = False
                try:
                    resp = await self.kalshi._get("/exchange/status")
                    if resp:
                        api_ok = True
                except Exception:
                    pass

                if not api_ok:
                    self._consecutive_api_failures += 1
                    self.engine.log_event(
                        "warning",
                        f"Kalshi API unreachable (consecutive failures: {self._consecutive_api_failures})",
                        strategy="health",
                    )
                    if self._last_api_check_ok:
                        # Just went down — alert
                        try:
                            asyncio.create_task(send_discord_notification(
                                title="⚠️ Kalshi API Unreachable",
                                message=f"The Kalshi API is not responding. Trading paused until connectivity restored.",
                                color=0xe74c3c,
                            ))
                        except Exception:
                            pass
                    self._last_api_check_ok = False
                else:
                    if not self._last_api_check_ok:
                        # Just recovered — notify
                        self.engine.log_event("info", "Kalshi API connectivity restored", strategy="health")
                        try:
                            asyncio.create_task(send_discord_notification(
                                title="✅ Kalshi API Restored",
                                message=f"Connectivity restored after {self._consecutive_api_failures} failed checks.",
                                color=0x2ecc71,
                            ))
                        except Exception:
                            pass
                    self._consecutive_api_failures = 0
                    self._last_api_check_ok = True

                # ── Update engine health snapshot ──
                self.engine._runtime_api_healthy = api_ok
                try:
                    ws_status = self.ws.get_status()
                    self.engine._runtime_ws_healthy = ws_status.get("connected", False)
                except Exception:
                    self.engine._runtime_ws_healthy = False
                self.engine._last_monitor_heartbeat = datetime.now(UTC).isoformat()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health watchdog error", error=str(e))

            await asyncio.sleep(5 * 60)  # check every 5 minutes

    async def _arbitrage_loop(self, sleep_first: bool = False) -> None:
        """Arbitrage strategy loop — runs every 30 seconds to catch fleeting opportunities."""
        if sleep_first:
            await asyncio.sleep(30)
        while self._running:
            try:
                if not self.engine.kill_switch:
                    candidates = await self.run_arbitrage_cycle()
                    if candidates:
                        # Execute immediately - arbitrage is time-sensitive
                        await self.execute_ranked_signals(candidates)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Arbitrage loop error: {e}", strategy="arbitrage")
                logger.error("Arbitrage loop error", error=str(e))

            await asyncio.sleep(30)

    def _get_weather_loop_interval(self) -> int:
        """Adaptive weather loop interval based on nearest same-day market settlement.
        - Same-day markets exist with <3 hours to close → 15 min
        - Same-day markets exist with 3-6 hours → 30 min
        - Next-day markets only → 60 min
        """
        try:
            from zoneinfo import ZoneInfo
            now_utc = datetime.now(UTC)
            # Check open weather positions for nearest settlement
            positions = self.engine.get_open_positions()
            weather_positions = [p for p in positions if p.get("strategy") == "weather"]
            if weather_positions:
                return 5 * 60

            # Check if there are same-day weather markets
            today_str = now_utc.strftime("%Y-%m-%d")
            # Approximate: if it's past noon ET, same-day markets close soon
            now_et = datetime.now(ZoneInfo("America/New_York"))
            if now_et.hour >= 18:
                return 5 * 60
            elif now_et.hour >= 12:
                return 10 * 60
            elif now_et.hour >= 6:
                return 15 * 60
            else:
                return 60 * 60
        except Exception:
            return 60 * 60  # fallback to 1 hour

    async def _weather_loop(self, sleep_first: bool = False) -> None:
        """Weather strategy loop — adaptive frequency based on market urgency."""
        if sleep_first:
            interval = self._get_weather_loop_interval()
            await asyncio.sleep(interval)
        while self._running:
            try:
                if not self.engine.kill_switch:
                    candidates = await self.run_weather_cycle()
                    # Filter out skips
                    tradeable = [c for c in candidates if c.get("action") != "skip"]
                    if tradeable:
                        await self.execute_ranked_signals(tradeable)

                    # Intraday obs re-eval: for open weather positions on same-day markets,
                    # fetch fresh NWS observations and consider adding if obs strongly confirm
                    await self._weather_intraday_reeval()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Weather loop error: {e}", strategy="weather")
                logger.error("Weather loop error", error=str(e))

            interval = self._get_weather_loop_interval()
            self.engine.log_event("info", f"Weather loop sleeping {interval // 60}m (adaptive)", strategy="weather")
            await asyncio.sleep(interval)

    async def _weather_intraday_reeval(self) -> None:
        """For open weather positions, fetch NWS obs and boost/add if observations confirm our position."""
        positions = self.engine.get_open_positions()
        weather_positions = [p for p in positions if p.get("strategy") == "weather"]
        if not weather_positions:
            return

        for pos in weather_positions:
            try:
                ticker = pos["ticker"]
                wx_m = re.match(r'KX(HIGH|LOW)T?([A-Z]{2,4})-', ticker)
                if not wx_m:
                    continue
                wx_type = wx_m.group(1)
                wx_city = wx_m.group(2)

                obs = await self.weather.get_current_observations(wx_city)
                if not obs or obs.get("obs_count", 0) < 4:
                    continue

                strike_match = re.search(r'-[TB](\d+\.?\d*)', ticker)
                if not strike_match:
                    continue
                strike = float(strike_match.group(1))
                obs_key = "observed_high_f" if wx_type == "HIGH" else "observed_low_f"
                obs_temp = obs.get(obs_key)
                if obs_temp is None:
                    continue

                gap = obs_temp - strike
                # If observations strongly CONFIRM our position, log it (monitor exit handles contradictions)
                confirms = False
                if pos["side"] == "yes" and gap > 2.0:
                    confirms = True
                elif pos["side"] == "no" and gap < -2.0:
                    confirms = True

                if confirms:
                    self.engine.log_event(
                        "info",
                        f"Weather obs confirms {pos['side'].upper()} {ticker}: obs={obs_temp}°F vs strike={strike}°F (gap={gap:+.1f}°F, {obs.get('obs_count', 0)} readings)",
                        strategy="weather",
                    )
            except Exception as e:
                logger.debug("Weather intraday reeval failed", ticker=pos.get("ticker"), error=str(e))

    async def _crypto_loop(self, sleep_first: bool = False) -> None:
        """Crypto strategy loop — runs every 2 minutes (15-min markets need fast reaction).
        Crypto runs on its own fast loop and executes via global ranking."""
        if sleep_first:
            await asyncio.sleep(2 * 60)
        while self._running:
            try:
                if not self.engine.kill_switch:
                    candidates = await self.run_crypto_cycle()
                    if candidates:
                        await self.execute_ranked_signals(candidates)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Crypto loop error: {e}", strategy="crypto")
                logger.error("Crypto loop error", error=str(e))

            await asyncio.sleep(2 * 60)

    @staticmethod
    def _is_us_market_hours() -> bool:
        """Check if US stock markets are open (9:00am-5:00pm ET, weekdays)."""
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
        if now_et.weekday() >= 5:  # Saturday/Sunday
            return False
        return 9 <= now_et.hour < 17

    async def _main_strategy_loop(self, sleep_first: bool = False) -> None:
        """
        Unified strategy loop for sports, finance, econ, NBA props.
        Runs every 5 minutes. Collects candidates from all strategies,
        ranks them globally, and executes the best ones.
        """
        if sleep_first:
            await asyncio.sleep(5 * 60)
        while self._running:
            try:
                if not self.engine.kill_switch:
                    all_candidates: list[dict[str, Any]] = []

                    # Sports — always run
                    try:
                        sports_candidates = await self.run_sports_cycle()
                        all_candidates.extend(sports_candidates)
                    except Exception as e:
                        self.engine.log_event("error", f"Sports cycle error: {e}", strategy="sports")

                    # Finance — only during market hours
                    if self._is_us_market_hours():
                        try:
                            finance_candidates = await self.run_finance_cycle()
                            all_candidates.extend(finance_candidates)
                        except Exception as e:
                            self.engine.log_event("error", f"Finance cycle error: {e}", strategy="finance")

                        # Econ — also during market hours
                        try:
                            econ_candidates = await self.run_econ_cycle()
                            all_candidates.extend(econ_candidates)
                        except Exception as e:
                            self.engine.log_event("error", f"Econ cycle error: {e}", strategy="econ")

                    # NBA props — always run (has its own 12hr time gate)
                    try:
                        nba_candidates = await self.run_nba_props_cycle()
                        all_candidates.extend(nba_candidates)
                    except Exception as e:
                        self.engine.log_event("error", f"NBA props cycle error: {e}", strategy="nba_props")

                    # Global ranking and execution
                    if all_candidates:
                        await self.execute_ranked_signals(all_candidates)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Main strategy loop error: {e}")
                logger.error("Main strategy loop error", error=str(e))

            await asyncio.sleep(5 * 60)

    async def _monitor_loop(self, sleep_first: bool = False) -> None:
        """Position monitor + settlement loop — runs every 2 minutes."""
        if sleep_first:
            await asyncio.sleep(2 * 60)
        last_summary_date = ""
        _kill_switch_notified = False
        while self._running:
            try:
                # Discord alert if kill switch just activated
                if self.engine.kill_switch and not _kill_switch_notified:
                    _kill_switch_notified = True
                    try:
                        asyncio.create_task(send_discord_notification(
                            title="🚨 KILL SWITCH ACTIVATED",
                            message="The trading agent kill switch has been triggered. All trading is paused.",
                            color=0xe74c3c,
                        ))
                    except Exception:
                        pass
                elif not self.engine.kill_switch:
                    _kill_switch_notified = False

                if not self.engine.kill_switch:
                    # Reconnect WebSocket if it dropped
                    try:
                        ws_status = self.ws.get_status()
                        if not ws_status.get("connected", False):
                            logger.info("WebSocket disconnected — attempting reconnect")
                            await self.ws.connect()
                    except Exception as ws_err:
                        logger.warning("WebSocket reconnect failed", error=str(ws_err))

                    await self.run_monitor_cycle()
                    await self.run_settlement_cycle()

                    # Daily summary at midnight UTC
                    today = datetime.now(UTC).strftime("%Y-%m-%d")
                    if today != last_summary_date:
                        last_summary_date = today
                        self._log_daily_summary()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Monitor loop error: {e}", strategy="monitor")
                logger.error("Monitor loop error", error=str(e))

            await asyncio.sleep(2 * 60)

    def _log_daily_summary(self) -> None:
        """Log a daily performance summary."""
        import sqlite3 as _sqlite3

        from app.services.trading_engine import DB_PATH
        try:
            yesterday = (datetime.now(UTC) - __import__("datetime").timedelta(days=1)).strftime("%Y-%m-%d")
            conn = _sqlite3.connect(str(DB_PATH))
            c = conn.cursor()
            c.execute(
                "SELECT strategy, COUNT(*), COALESCE(SUM(cost+fee),0) FROM trades WHERE action='buy' AND timestamp LIKE ? GROUP BY strategy",
                (f"{yesterday}%",),
            )
            trades_by_strat = {r[0]: (r[1], r[2]) for r in c.fetchall()}
            c.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE status='settled' AND settled_at LIKE ?",
                (f"{yesterday}%",),
            )
            settled_pnl = c.fetchone()[0]
            conn.close()

            total_trades = sum(v[0] for v in trades_by_strat.values())
            total_deployed = sum(v[1] for v in trades_by_strat.values())
            strat_summary = ", ".join(f"{k}={v[0]}/${v[1]:.0f}" for k, v in trades_by_strat.items())

            summary_msg = (
                f"Daily summary {yesterday}: {total_trades} trades, ${total_deployed:.2f} deployed, "
                f"settled P&L=${settled_pnl:+.2f} | {strat_summary}"
            )
            self.engine.log_event("daily_summary", summary_msg, strategy="monitor")

            # Send daily summary to Discord
            try:
                import asyncio as _aio
                _aio.create_task(send_discord_notification(
                    title=f"📊 Daily Summary — {yesterday}",
                    message=(
                        f"**Trades:** {total_trades} | **Deployed:** ${total_deployed:.2f}\n"
                        f"**Settled P&L:** ${settled_pnl:+.2f}\n"
                        f"**By Strategy:** {strat_summary}\n"
                        f"**Bankroll:** ${self.engine.bankroll:.2f}"
                    ),
                    color=0x3498db,
                ))
            except Exception:
                pass
        except Exception as e:
            logger.debug("Daily summary failed", error=str(e))

    async def get_status(self) -> dict[str, Any]:
        """Get agent status.  Uses Kalshi API for balance/exposure (source of truth)."""
        # Loop health check
        loop_health = {}
        for name, task in [
            ("weather", self._weather_task),
            ("crypto", self._crypto_task),
            ("main", self._main_task),
            ("monitor", self._monitor_task),
            ("health", self._health_task),
            ("arbitrage", getattr(self, "_arbitrage_task", None)),
        ]:
            if task is None:
                loop_health[name] = "not_started"
            elif task.done():
                loop_health[name] = "crashed"
            elif task.cancelled():
                loop_health[name] = "cancelled"
            else:
                loop_health[name] = "running"

        # Fetch live balance + portfolio value from Kalshi API
        kalshi_balance_cents = 0
        kalshi_portfolio_cents = 0
        try:
            bal = await self.kalshi.get_balance()
            kalshi_balance_cents = bal.get("balance", 0)
            kalshi_portfolio_cents = bal.get("portfolio_value", 0)
        except Exception:
            pass  # Fall back to DB values below

        base_status = self.engine.get_status()

        # Override bankroll/exposure/remaining with Kalshi API truth
        if kalshi_balance_cents > 0 or kalshi_portfolio_cents > 0:
            total_value = (kalshi_balance_cents + kalshi_portfolio_cents) / 100.0
            deployed = kalshi_portfolio_cents / 100.0
            cash = kalshi_balance_cents / 100.0
            base_status["bankroll"] = round(total_value, 2)
            base_status["effective_bankroll"] = round(total_value, 2)
            base_status["total_exposure"] = round(deployed, 2)
            base_status["remaining_capital"] = round(cash, 2)
            base_status["max_deployable"] = round(total_value, 2)
            base_status["over_deployed"] = False

        return {
            "running": self._running,
            "paper_mode": self.engine.paper_mode,
            "kill_switch": self.engine.kill_switch,
            **base_status,
            "odds_api_credits_remaining": self.sports.remaining_credits,
            "websocket": self.ws.get_status(),
            "health": {
                "api_healthy": self._last_api_check_ok,
                "consecutive_api_failures": self._consecutive_api_failures,
                "ws_healthy": self.engine._runtime_ws_healthy,
                "last_heartbeat": self.engine._last_monitor_heartbeat,
                "loops": loop_health,
            },
        }

    def get_weather_diagnostics(self) -> dict[str, Any]:
        """Summarize observed-weather trade gates, provider health, and recent misses."""
        thresholds = self._observed_weather_thresholds()
        provider_health = self.weather.get_source_diagnostics()
        near_miss_summary = self.engine.get_near_miss_summary(strategy="weather", limit=8)
        recent_near_misses = self.engine.get_candidate_rejections(
            strategy="weather",
            near_miss_only=True,
            limit=12,
        )

        recent_live_weather_trades = self.engine.get_trades(
            strategy="weather",
            limit=40,
            paper_only=False,
        )
        recent_observed_trades = [
            trade for trade in recent_live_weather_trades
            if trade.get("signal_source") == "weather_observed_arbitrage"
        ][:10]

        return {
            "observed_thresholds": thresholds,
            "provider_health": provider_health,
            "near_miss_summary": near_miss_summary,
            "recent_near_misses": recent_near_misses,
            "recent_observed_trades": recent_observed_trades,
            "recent_observed_quality": self.engine.get_recent_source_quality(
                "weather_observed_arbitrage",
                lookback_days=45,
            ),
        }


# Singleton
_agent: KalshiAgent | None = None


def get_kalshi_agent() -> KalshiAgent:
    """Get or create the singleton agent."""
    global _agent
    if _agent is None:
        _agent = KalshiAgent()
    return _agent
