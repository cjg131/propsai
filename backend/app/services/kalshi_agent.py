"""
Kalshi Autonomous Trading Agent.
Orchestrates weather, sports, crypto, finance, and econ strategies.
Runs on a schedule: weather 4x/day, sports 5min, crypto 2min, finance 10min, econ 30min.
"""
from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any

from app.config import get_settings
from app.logging_config import get_logger
from app.services.cross_market_sports import CrossMarketScanner
from app.services.cross_strategy_correlation import CrossStrategyCorrelation, get_cross_strategy_engine
from app.services.crypto_data import CryptoDataService
from app.services.econ_data import EconDataService
from app.services.finance_data import FinanceDataService
from app.services.kalshi_api import get_kalshi_client
from app.services.kalshi_ws import KalshiWebSocket, get_kalshi_ws
from app.services.kalshi_scanner import KalshiScanner, parse_parlay_legs
from app.services.nba_data import NBADataService, get_nba_data
from app.services.parlay_pricer import price_parlay_legs, teams_match
from app.services.smart_predictor import SmartPredictor, get_smart_predictor
from app.services.adaptive_thresholds import get_adaptive_thresholds
from app.services.polymarket_data import get_polymarket_data
from app.services.signal_scorer import get_signal_scorer
from app.services.news_sentiment import get_market_news_sentiment
from app.services.trade_analyzer import get_trade_analyzer
from app.services.trading_engine import get_trading_engine
from app.services.weather_data import CITY_CONFIGS, WeatherConsensus
from app.services.referee_data import get_referee_data, RefereeDataService

logger = get_logger(__name__)


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

    def _extract_signal_components(self, trade: dict[str, Any]) -> dict[str, float]:
        """Extract signal component values from a trade's notes/details for quality scoring."""
        components: dict[str, float] = {}
        # Try to parse from signal details stored in the DB
        try:
            from app.services.trading_engine import DB_PATH
            import sqlite3
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
        "LAS": "Las Vegas", "PHX": "Phoenix", "MSP": "Minneapolis", "NOL": "New Orleans",
        "DET": "Detroit", "NOLA": "New Orleans",
    }

    @staticmethod
    def _enrich_weather_title(ticker: str, title: str) -> str:
        """Prepend city name to weather market title if not already present."""
        # Strip markdown bold markers left by some Kalshi titles
        title = title.replace("**", "")
        # Extract city code from ticker: KXHIGHTBOS-... or KXHIGHMIA-... or KXHIGHDEN-...
        # Pattern: KX(HIGH|LOW)(T?)<CITY>-...
        m = re.match(r'KX(?:HIGH|LOW)T?([A-Z]{2,4})-', ticker)
        if not m:
            return title
        city_code = m.group(1)
        city_name = KalshiAgent.WEATHER_CITY_CODES.get(city_code, "")
        if not city_name:
            return title
        # Only prepend if city name isn't already in the title
        if city_name.lower() in title.lower():
            return title
        return f"[{city_name}] {title}"

    # ── Global Signal Ranking & Execution ──────────────────────────

    async def execute_ranked_signals(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Rank all candidates globally by edge * confidence, then execute
        the top ones.  Each cycle deploys up to 80% of AVAILABLE capital
        (bankroll − current net exposure).  The remaining 20% is held back
        for the next cycle to DCA into existing positions or catch new
        opportunities.  Over multiple cycles the system can deploy up to
        100% of bankroll.
        """
        if not candidates:
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

            if strategy == "finance" and c.get("is_bracket"):
                fin_index = c.get("finance_index", "")
                fin_expiry = c.get("finance_expiry", "")
                if fin_index and fin_expiry:
                    exclusive_key = f"finance_bracket_{fin_index}_{fin_expiry}"

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

        # Sort by edge * confidence descending
        candidates.sort(key=lambda c: c.get("edge", 0) * c.get("confidence", 0), reverse=True)

        total_exposure = self.engine.get_total_exposure()
        effective_bankroll = self.engine.get_effective_bankroll()
        available_capital = max(0, effective_bankroll - total_exposure)
        cycle_budget = available_capital * 0.80  # deploy 80% of what's free this cycle

        self.engine.log_event(
            "info",
            f"Global ranking: {len(candidates)} candidates, "
            f"exposure=${total_exposure:.2f}, available=${available_capital:.2f}, "
            f"cycle budget=${cycle_budget:.2f}",
        )

        traded: list[dict[str, Any]] = []
        for candidate in candidates:
            if cycle_budget <= 0:
                break

            strategy = candidate.get("strategy", "")
            ticker = candidate.get("ticker", "")
            edge = candidate.get("edge", 0)
            confidence = candidate.get("confidence", 0)
            price_cents = candidate.get("price_cents", 0)

            if not ticker or price_cents <= 0:
                continue

            # Position sizing (respects per-strategy and per-ticker caps)
            count = self.engine.calculate_position_size(
                strategy=strategy,
                edge=edge,
                price_cents=price_cents,
                confidence=confidence,
                ticker=ticker,
            )
            if count <= 0:
                continue

            cost_estimate = count * price_cents / 100.0
            if cost_estimate > cycle_budget:
                # Reduce count to fit budget
                count = int(cycle_budget / (price_cents / 100.0))
                if count <= 0:
                    continue

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
            )

            if trade and trade.get("status") not in ("blocked", "error", "timeout"):
                actual_cost = trade.get("cost", 0) + trade.get("fee", 0)
                cycle_budget -= actual_cost
                candidate["trade"] = trade
                traded.append(candidate)

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

            # Group by (city_code, target_date) so each group gets the right forecast
            from datetime import UTC, date as _date, datetime as _datetime
            by_city_date: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for m in weather_markets:
                city_code = m.get("weather", {}).get("city_code", "")
                if not city_code:
                    continue
                # Parse target date from the ticker (e.g. KXHIGHDEN-26FEB19-B36.5 → 2026-02-19)
                # close_time is UTC and can be off by one day; ticker date is authoritative
                ticker = m.get("ticker", "")
                try:
                    import re as _re
                    date_match = _re.search(r'-(\d{2})([A-Z]{3})(\d{2})-', ticker)
                    if date_match:
                        # e.g. KXHIGHDEN-26FEB19-B36.5 → year=26, month=FEB, day=19
                        target_date_str = _datetime.strptime(
                            date_match.group(1) + date_match.group(2) + date_match.group(3), "%y%b%d"
                        ).date().isoformat()
                    else:
                        target_date_str = _datetime.now(UTC).date().isoformat()
                except Exception:
                    target_date_str = _datetime.now(UTC).date().isoformat()
                by_city_date.setdefault((city_code, target_date_str), []).append(m)

            fetched_forecasts: dict[tuple[str, str], dict[str, Any]] = {}
            for i, ((city_key, target_date_str), city_markets) in enumerate(by_city_date.items()):
                if city_key not in CITY_CONFIGS:
                    self.engine.log_event(
                        "warning",
                        f"Unknown city code {city_key}, skipping",
                        strategy="weather",
                    )
                    continue

                # Rate-limit: pause between city/date combos to avoid 429s
                if i > 0:
                    await asyncio.sleep(2.0)

                target_date = _date.fromisoformat(target_date_str)

                # Get forecasts from all sources for the correct date
                forecasts = await self.weather.get_all_forecasts(city_key, target_date=target_date)
                source_count = len(forecasts.get("sources", {}))

                if source_count < 2:
                    self.engine.log_event(
                        "warning",
                        f"Only {source_count} sources for {city_key} {target_date_str}, skipping",
                        strategy="weather",
                    )
                    continue

                self.engine.log_event(
                    "info",
                    f"{city_key}: {len(city_markets)} markets, {source_count} forecast sources",
                    strategy="weather",
                )

                # Evaluate each market (collect candidates, don't trade yet)
                for market in city_markets:
                    candidate = await self._evaluate_weather_market(market, forecasts)
                    if candidate:
                        results.append(candidate)

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
        return results

    async def _evaluate_weather_market(
        self,
        market: dict[str, Any],
        forecasts: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Evaluate a single weather market against consensus forecast."""
        yes_ask = market.get("yes_ask", 0)
        no_ask = market.get("no_ask", 0)
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
            return None

        # Safety filter: skip effectively-settled markets where either side is <=3c
        # (means the outcome is near-certain, no real edge to capture)
        if yes_ask <= 3 or no_ask <= 3:
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        # These are long-shot bets with high loss rates even with edge
        if min(yes_ask, no_ask) < 10:
            return None

        # Safety filter: skip markets closing within 2 hours
        close_time = market.get("close_time", "")
        if close_time:
            try:
                close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                now = datetime.now(UTC)
                hours_until_close = (close_dt - now).total_seconds() / 3600
                if hours_until_close < 2:
                    return None
            except (ValueError, TypeError):
                pass

        # Safety filter: skip very low volume markets (< 50 contracts traded)
        if market.get("volume", 0) < 50:
            return None

        # Build consensus for this specific market's strike structure
        market_type = weather_info.get("market_type", "high_temp")
        consensus = self.weather.build_consensus(
            forecasts,
            strike_type=strike_type,
            floor_strike=floor_strike,
            cap_strike=cap_strike,
            market_type=market_type,
        )
        if "error" in consensus:
            return None

        # Generate signal
        wx_thresh = self.adaptive.get_thresholds("weather")
        signal = self.weather.generate_signal(
            consensus,
            kalshi_yes_price=yes_ask,
            kalshi_no_price=no_ask,
            min_edge=wx_thresh["min_edge"],
            min_confidence=wx_thresh["min_confidence"],
            min_sources=2,
        )

        if not signal:
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
            details=f"consensus={consensus.get('mean_low_f') if market_type == 'low_temp' else consensus.get('mean_high_f')}°F {label} sources={signal['source_count']}",
        )

        # Calculate position size
        price_cents = yes_ask if signal["side"] == "yes" else no_ask
        count = self.engine.calculate_position_size(
            strategy="weather",
            edge=signal["edge"],
            price_cents=price_cents,
            confidence=signal["confidence"],
            ticker=market["ticker"],
        )

        if count <= 0:
            return {**signal, "ticker": market["ticker"], "action": "skip", "reason": "position_size_zero"}

        # Return candidate for ranking (execution happens in the cycle)
        return {
            **signal,
            "ticker": market["ticker"],
            "title": market["title"],
            "count": count,
            "price_cents": price_cents,
            "signal_id": signal_id,
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

            # ── Step 3: Evaluate single-game markets against sharp lines ──
            single_stats = {"matched": 0, "no_odds": 0, "cheap_skip": 0, "no_team_match": 0, "no_sharp": 0, "no_edge": 0}
            for market in single_markets:
                try:
                    signal = await self._evaluate_single_game(market, odds_by_sport, single_stats, ref_signals)
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
                    # Check if the suffix matches the team (e.g., PSG matches "Paris Saint Germain")
                    team_words = team_name.lower().split()
                    suffix_lower = outcome_suffix.lower()
                    if (suffix_lower in team_name.lower() or
                        any(w.startswith(suffix_lower) for w in team_words) or
                        team_name.lower().startswith(suffix_lower)):
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
                    # Match team from suffix and line from floor_strike
                    suffix_lower = outcome_suffix.lower()
                    # Remove trailing digits from suffix to get team part
                    team_part = suffix_lower.rstrip("0123456789")
                    if (team_part in team_name.lower() or
                        team_name.lower().startswith(team_part)):
                        if abs(abs(line) - target_line) < 0.5:
                            sharp_prob = prob
                            match_desc = key
                            break

        if sharp_prob is None:
            stats["no_sharp"] += 1
            return None

        # Calculate edge: Kalshi YES price vs sharp probability
        kalshi_implied = yes_ask / 100.0 if yes_ask else 0
        if kalshi_implied <= 0:
            stats["no_edge"] += 1
            return None

        edge = sharp_prob - kalshi_implied
        side = "yes"

        # Also check NO side
        no_implied = no_ask / 100.0 if no_ask else 0
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

        self.engine.log_event(
            "info",
            f"Single-game edge: {ticker} {side} edge={edge:.1%} (sharp={sharp_prob:.1%} vs kalshi={kalshi_implied:.1%}) {match_desc}{ref_detail} | {title[:60]}",
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
            details=f"sport={odds_sport} type={kalshi_type} match={match_desc} edge={edge:.4f}{ref_detail}",
        )

        # Return candidate for global ranking (execution happens later)
        price_cents = yes_ask if side == "yes" else no_ask
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

        # Safety filters
        if yes_ask <= 3 or no_ask <= 3:
            if stats:
                stats["illiquid"] += 1
            return None
        # Skip very cheap contracts (< 10c on either side)
        if min(yes_ask, no_ask) < 10:
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

        # Compare fair value to Kalshi price
        kalshi_yes_prob = yes_ask / 100.0
        kalshi_no_prob = no_ask / 100.0

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
        price_cents = yes_ask if signal["side"] == "yes" else no_ask
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

        if not coin or coin not in signal_by_coin:
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        if min(yes_ask, no_ask) < 10:
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
        kalshi_yes_implied = yes_ask / 100.0 if yes_ask > 0 else 0
        kalshi_no_implied = no_ask / 100.0 if no_ask > 0 else 0

        yes_edge = our_prob_yes - kalshi_yes_implied if kalshi_yes_implied > 0 else 0
        no_edge = (1.0 - our_prob_yes) - kalshi_no_implied if kalshi_no_implied > 0 else 0

        # Pick the better side
        if yes_edge >= no_edge:
            edge = yes_edge
            side = "yes"
            our_prob = our_prob_yes
            kalshi_prob = kalshi_yes_implied
            price_cents = yes_ask
        else:
            edge = no_edge
            side = "no"
            our_prob = 1.0 - our_prob_yes
            kalshi_prob = kalshi_no_implied
            price_cents = no_ask

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

        if not index or index not in signal_by_index:
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        if min(yes_ask, no_ask) < 10:
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
                is_up_market = True

            our_prob_yes = p_up if is_up_market else (1.0 - p_up)
            signal_source = "finance_momentum"
            prob_details = f"index={index} p_up={p_up:.4f} intraday={signal['intraday_momentum']:.4f} " \
                           f"futures={signal['futures_signal']:.4f} vix={signal['vix_signal']:.4f} " \
                           f"ma={signal['ma_signal']:.4f}"

        # Calculate edge
        kalshi_yes_implied = yes_ask / 100.0 if yes_ask > 0 else 0
        kalshi_no_implied = no_ask / 100.0 if no_ask > 0 else 0

        yes_edge = our_prob_yes - kalshi_yes_implied if kalshi_yes_implied > 0 else 0
        no_edge = (1.0 - our_prob_yes) - kalshi_no_implied if kalshi_no_implied > 0 else 0

        if yes_edge >= no_edge:
            edge, side = yes_edge, "yes"
            our_prob, kalshi_prob, price_cents = our_prob_yes, kalshi_yes_implied, yes_ask
        else:
            edge, side = no_edge, "no"
            our_prob, kalshi_prob, price_cents = 1.0 - our_prob_yes, kalshi_no_implied, no_ask

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

        if not econ_type or econ_type not in econ_signals:
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        if min(yes_ask, no_ask) < 10:
            return None

        signal = econ_signals[econ_type]
        title_lower = title.lower()

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
                confidence = 0.4
            else:
                return None
        else:
            return None

        # Calculate edge
        kalshi_yes_implied = yes_ask / 100.0 if yes_ask > 0 else 0
        kalshi_no_implied = no_ask / 100.0 if no_ask > 0 else 0

        # Check if title says "above" or "below"
        if "below" in title_lower or "under" in title_lower:
            our_prob_yes = 1.0 - our_prob_yes

        yes_edge = our_prob_yes - kalshi_yes_implied if kalshi_yes_implied > 0 else 0
        no_edge = (1.0 - our_prob_yes) - kalshi_no_implied if kalshi_no_implied > 0 else 0

        if yes_edge >= no_edge:
            edge, side = yes_edge, "yes"
            our_prob, kalshi_prob, price_cents = our_prob_yes, kalshi_yes_implied, yes_ask
        else:
            edge, side = no_edge, "no"
            our_prob, kalshi_prob, price_cents = 1.0 - our_prob_yes, kalshi_no_implied, no_ask

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
        """Lazy-initialize NBA data service and predictor (they use sync HTTP)."""
        try:
            if self.nba_data is None:
                self.nba_data = get_nba_data()
            if self.predictor is None:
                self.predictor = get_smart_predictor()
            return True
        except Exception as e:
            logger.warning("NBA services init failed", error=str(e))
            return False

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

            # Step 2: Initialize NBA services (sync, so run in executor)
            loop = asyncio.get_event_loop()
            initialized = await loop.run_in_executor(None, self._ensure_nba_services)
            if not initialized or not self.predictor or not self.nba_data:
                self.engine.log_event("warning", "NBA services not available", strategy="nba_props")
                return results

            # Step 3: Build enriched feature set (sync call in executor)
            try:
                feature_data = await loop.run_in_executor(
                    None, self.nba_data.build_full_feature_set
                )
                players = feature_data.get("players", {})
                schedule = feature_data.get("schedule", {})
                self.engine.log_event(
                    "info",
                    f"Built features for {len(players)} players",
                    strategy="nba_props",
                )
            except Exception as e:
                self.engine.log_event("warning", f"Feature build failed: {e}", strategy="nba_props")
                return results

            if not players:
                self.engine.log_event("info", "No player data available (off-season?)", strategy="nba_props")
                return results

            # Build name -> player lookup for matching
            name_to_player: dict[str, dict] = {}
            for pid, pf in players.items():
                name = (pf.get("name") or "").strip().lower()
                if name:
                    name_to_player[name] = pf

            # Step 3b: Enrich players with BallDontLie rolling averages + rest/matchup features
            # These are the most predictive features in the model but were always 0 without this.
            try:
                from app.services.balldontlie import get_balldontlie
                bdl = get_balldontlie()

                # Get all active player IDs from BDL (name -> BDL player dict)
                active_players = await loop.run_in_executor(None, bdl.get_active_players)
                bdl_name_map: dict[str, dict] = {}
                for ap in active_players:
                    fname = (ap.get("first_name") or "").strip()
                    lname = (ap.get("last_name") or "").strip()
                    full = f"{fname} {lname}".lower().strip()
                    if full:
                        bdl_name_map[full] = ap

                # Fetch game logs for all players we care about (bulk, one request)
                bdl_ids_needed = []
                name_to_bdl_id: dict[str, int] = {}
                for pname in name_to_player:
                    bdl_p = bdl_name_map.get(pname)
                    if not bdl_p:
                        # Try partial match on last name
                        parts = pname.split()
                        if parts:
                            last = parts[-1]
                            for bn, bp in bdl_name_map.items():
                                if bn.endswith(last) and len(last) >= 4:
                                    bdl_p = bp
                                    break
                    if bdl_p:
                        bid = bdl_p.get("id")
                        if bid:
                            bdl_ids_needed.append(bid)
                            name_to_bdl_id[pname] = bid

                if bdl_ids_needed:
                    bulk_logs = await loop.run_in_executor(
                        None, bdl.get_bulk_game_logs, bdl_ids_needed
                    )
                    # Enrich each player with BDL features
                    from datetime import date as _date
                    today = _date.today()
                    for pname, pf in name_to_player.items():
                        bid = name_to_bdl_id.get(pname)
                        if not bid or bid not in bulk_logs:
                            continue
                        logs = bulk_logs[bid]
                        if not logs:
                            continue
                        # Rolling averages
                        gl_feats = bdl.build_player_game_log_features(logs)
                        pf.update(gl_feats)
                        # Rest / schedule load features
                        rest_feats = bdl.build_rest_features_from_logs(logs, today)
                        pf["rest_days"] = rest_feats.get("days_rest", pf.get("rest_days", 2))
                        pf["is_b2b"] = bool(rest_feats.get("is_b2b", 0))
                        pf["games_last_7"] = rest_feats.get("games_last_7", pf.get("games_last_7", 3))
                        pf["is_3_in_4"] = bool(rest_feats.get("is_3_in_4", 0))
                        # Matchup features (vs today's opponent)
                        opp_team = pf.get("opponent", "")
                        if opp_team:
                            # Find BDL team ID for opponent
                            opp_bdl_id = None
                            for log in logs[:5]:
                                game = log.get("game", {})
                                team_id = log.get("team", {}).get("id")
                                home_id = game.get("home_team_id")
                                vis_id = game.get("visitor_team_id")
                                opp_id = vis_id if team_id == home_id else home_id
                                if opp_id:
                                    opp_bdl_id = opp_id
                                    break
                            if opp_bdl_id:
                                mu_feats = bdl.build_matchup_features(logs, opp_bdl_id)
                                pf.update(mu_feats)

                    self.engine.log_event(
                        "info",
                        f"BDL enrichment: {len(bulk_logs)}/{len(bdl_ids_needed)} players enriched with rolling averages",
                        strategy="nba_props",
                    )
            except Exception as e:
                logger.warning("BDL enrichment failed", error=str(e))

            # Step 3c: Fetch Odds API game lines (spread, over_under) and wire into player features
            # These are key SmartPredictor features that were always 0 without this.
            try:
                from app.services.odds_api import get_odds_api as _get_odds_api
                _odds_client = _get_odds_api()
                game_lines_dict = await _odds_client.get_game_lines()
                # get_game_lines() returns {event_id: {home_team, away_team, spread_home, spread_away, total}}
                game_lines_map: dict[str, dict] = {}
                for gl in game_lines_dict.values():
                    home = (gl.get("home_team") or "").upper()
                    away = (gl.get("away_team") or "").upper()
                    spread_home = gl.get("spread_home", 0.0) or 0.0
                    spread_away = gl.get("spread_away", 0.0) or 0.0
                    total = gl.get("total", 220.0) or 220.0
                    if home:
                        game_lines_map[home] = {"spread": spread_home, "over_under": total}
                    if away:
                        game_lines_map[away] = {"spread": spread_away, "over_under": total}
                # Inject into each player's feature dict
                for pf in name_to_player.values():
                    team = (pf.get("team") or "").upper()
                    gl_entry = game_lines_map.get(team, {})
                    if gl_entry:
                        pf["spread"] = gl_entry["spread"]
                        pf["over_under"] = gl_entry["over_under"]
                logger.info(f"Game lines: enriched players from {len(game_lines_map)} team lines")
            except Exception as e:
                logger.debug("Game lines enrichment failed", error=str(e))

            # Step 3d: Fetch NewsSentimentService player sentiment and wire into player features
            # injury_mentioned / rest_mentioned / news_sentiment were always 0 without this.
            try:
                from app.services.news_sentiment import get_news_sentiment as _get_ns
                _ns_svc = _get_ns()
                known_players = set(name_to_player.keys())
                player_sentiment = await loop.run_in_executor(
                    None, _ns_svc.build_player_sentiment, known_players, 48
                )
                for pname, sent in player_sentiment.items():
                    pf = name_to_player.get(pname)
                    if pf:
                        pf["news_sentiment"] = sent.get("news_sentiment", 0.0)
                        pf["injury_mentioned"] = sent.get("injury_mentioned", 0)
                        pf["rest_mentioned"] = sent.get("rest_mentioned", 0)
                        pf["hot_streak_mentioned"] = sent.get("hot_streak_mentioned", 0)
                mentioned = sum(1 for v in player_sentiment.values() if v.get("news_volume", 0) > 0)
                logger.info(f"News sentiment: {mentioned}/{len(known_players)} players with news")
            except Exception as e:
                logger.debug("News sentiment enrichment failed", error=str(e))

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
            sdio_news: dict[str, list[dict]] = {}  # player_name_lower -> news items
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

            # Step 5b: Fetch SportsDataIO injury report and build player status lookup
            # get_injuries() was never called — wire it in now to penalize injured players
            sdio_injuries: dict[str, str] = {}  # player_name_lower -> injury status
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
            for market in props_markets:
                try:
                    result = await self._evaluate_nba_prop(
                        market, name_to_player, loop,
                        odds_props_lookup=odds_props_lookup,
                        sdio_news=sdio_news,
                        sdio_injuries=sdio_injuries,
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.debug("NBA prop eval failed", ticker=market.get("ticker"), error=str(e))

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

        if not prop_type or not player_name:
            return None

        # 12-hour time gate: parse game date from ticker and skip if >12hrs away
        # Ticker format: KXNBAPTS-26FEB19PHXSAS-...  (26FEB19 = Feb 19, 2026)
        try:
            ticker_date_match = re.search(r'(\d{2})([A-Z]{3})(\d{2})', ticker)
            if ticker_date_match:
                yr = int("20" + ticker_date_match.group(1))
                mon_str = ticker_date_match.group(2)
                day = int(ticker_date_match.group(3))
                months = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
                          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
                mon = months.get(mon_str, 0)
                if mon > 0:
                    from zoneinfo import ZoneInfo
                    # NBA games typically start 7pm ET, use that as reference
                    game_dt = datetime(yr, mon, day, 19, 0, tzinfo=ZoneInfo("America/New_York"))
                    now = datetime.now(ZoneInfo("America/New_York"))
                    hours_until_game = (game_dt - now).total_seconds() / 3600
                    if hours_until_game > 12:
                        return None
        except Exception:
            pass  # If parsing fails, proceed anyway

        # Match player name to our data
        player_key = player_name.lower().strip()
        player = name_to_player.get(player_key)

        # Try partial match if exact fails
        if not player:
            for name, pf in name_to_player.items():
                if player_key in name or name in player_key:
                    player = pf
                    break

        if not player:
            logger.debug("Player not found in data", player=player_name)
            return None

        # ── SportsDataIO injury status: skip Out/Doubtful, penalize Questionable ──
        injury_status_detail = ""
        if sdio_injuries and player_name:
            inj_key = player_name.lower().strip()
            inj_status = sdio_injuries.get(inj_key, "")
            if not inj_status:
                # Partial match on last name
                parts = inj_key.split()
                if parts:
                    last = parts[-1]
                    for ik, iv in sdio_injuries.items():
                        if ik.endswith(last) and len(last) >= 4:
                            inj_status = iv
                            break
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
            odds_key = f"{player_name.lower()}|{prop_type}"
            odds_prop = odds_props_lookup.get(odds_key)
            if not odds_prop:
                # Try partial player name match
                for ok, ov in odds_props_lookup.items():
                    if ok.endswith(f"|{prop_type}") and player_name.lower()[:6] in ok:
                        odds_prop = ov
                        break

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
            player_news = sdio_news.get(player_name.lower(), [])
            if not player_news:
                # Try partial match
                for nk, nv in sdio_news.items():
                    if player_name.lower()[:6] in nk:
                        player_news = nv
                        break

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
                    news_detail = f" news=INJURY_RISK"
                elif any(kw in headline for kw in positive_keywords):
                    # Player confirmed healthy/active
                    confidence = min(0.95, confidence + 0.05)
                    news_detail = f" news=HEALTHY"

        # Calculate edge
        kalshi_yes_implied = yes_ask / 100.0 if yes_ask > 0 else 0
        kalshi_no_implied = no_ask / 100.0 if no_ask > 0 else 0

        yes_edge = our_prob_yes - kalshi_yes_implied if kalshi_yes_implied > 0 else 0
        no_edge = (1.0 - our_prob_yes) - kalshi_no_implied if kalshi_no_implied > 0 else 0

        if yes_edge >= no_edge:
            edge, side = yes_edge, "yes"
            our_prob, kalshi_prob, price_cents = our_prob_yes, kalshi_yes_implied, yes_ask
        else:
            edge, side = no_edge, "no"
            our_prob, kalshi_prob, price_cents = 1.0 - our_prob_yes, kalshi_no_implied, no_ask

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
            positions = self.engine.get_open_positions()
            if not positions:
                return actions

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

            for pos in positions:
                ticker = pos["ticker"]
                try:
                    # Try WebSocket snapshot first (instant, no API call)
                    ws_snap = self.ws.get_snapshot(ticker) if self.ws.connected else None
                    if ws_snap and not ws_snap.is_stale:
                        yes_bid = ws_snap.yes_bid
                        yes_ask = ws_snap.yes_ask
                        no_bid = ws_snap.no_bid
                        no_ask = ws_snap.no_ask
                        last_price = ws_snap.last_price
                        mkt_status = ""  # WS doesn't provide status, check via REST if needed
                    else:
                        # Fall back to REST API
                        await asyncio.sleep(0.5)
                        data = await self.kalshi._get(f"/markets/{ticker}")
                        market = data.get("market", data)
                        if not market:
                            continue
                        yes_bid = market.get("yes_bid", 0) or 0
                        yes_ask = market.get("yes_ask", 0) or 0
                        no_bid = market.get("no_bid", 0) or 0
                        no_ask = market.get("no_ask", 0) or 0
                        last_price = market.get("last_price", 0) or 0
                        mkt_status = market.get("status", "")

                    # Mark-to-market: use same-side bid if spread is tight.
                    # Wide spread = bid is garbage lowball, use last_price or ask.
                    if pos["side"] == "yes":
                        bid, ask = yes_bid, yes_ask
                    else:
                        bid, ask = no_bid, no_ask
                    if bid > 0 and ask > 0 and bid >= ask * 0.5:
                        mark_price = bid  # tight spread — bid is real
                    elif last_price > 0:
                        mark_price = last_price if pos["side"] == "yes" else (100 - last_price)
                    elif ask > 0:
                        mark_price = ask  # wide spread, no trades — ask is best proxy
                    elif bid > 0:
                        mark_price = bid
                    else:
                        mark_price = pos["avg_entry_cents"]

                    mark_value = pos["contracts"] * mark_price / 100.0
                    unrealized_pnl = round(mark_value - pos["total_cost"] - pos["total_fees"], 2)

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
                    if pos["total_cost"] > 0 and unrealized_pnl < -(pos["total_cost"] * 0.50):
                        if mark_price > 0:
                            await self.engine.exit_trade(
                                strategy=pos["strategy"],
                                ticker=ticker,
                                side=pos["side"],
                                count=pos["contracts"],
                                price_cents=mark_price,
                                reason=f"stop_loss pnl=${unrealized_pnl:+.2f} ({unrealized_pnl/pos['total_cost']:.0%} of cost)",
                            )
                            self.engine.log_event(
                                "paper_trade",
                                f"STOP-LOSS {pos['side'].upper()} {pos['contracts']}x {ticker} @ {mark_price}c — down ${unrealized_pnl:+.2f}",
                                strategy="monitor",
                            )
                            actions.append({"action": "exit", "reason": "stop_loss", "ticker": ticker,
                                            "pnl": unrealized_pnl})
                            continue

                    # ── DECISION 0b: DEAD CONTRACT — mark price near zero ──
                    # If our side is trading at <=3c, the position is effectively dead
                    if 0 < mark_price <= 3:
                        await self.engine.exit_trade(
                            strategy=pos["strategy"],
                            ticker=ticker,
                            side=pos["side"],
                            count=pos["contracts"],
                            price_cents=mark_price,
                            reason=f"dead_contract mark={mark_price}c",
                        )
                        self.engine.log_event(
                            "paper_trade",
                            f"DEAD CONTRACT {pos['side'].upper()} {pos['contracts']}x {ticker} @ {mark_price}c — P&L: ${unrealized_pnl:+.2f}",
                            strategy="monitor",
                        )
                        actions.append({"action": "exit", "reason": "dead_contract", "ticker": ticker,
                                        "pnl": unrealized_pnl})
                        continue

                    # ── DECISION 1: EXIT — edge has flipped significantly ──
                    if current_edge < -0.05 and mark_price > 0:
                        await self.engine.exit_trade(
                            strategy=pos["strategy"],
                            ticker=ticker,
                            side=pos["side"],
                            count=pos["contracts"],
                            price_cents=mark_price,
                            reason=f"edge_flipped edge={current_edge:.3f}",
                        )
                        self.engine.log_event(
                            "paper_trade",
                            f"EXIT {pos['side'].upper()} {pos['contracts']}x {ticker} @ {mark_price}c — edge flipped to {current_edge:.1%}, P&L: ${unrealized_pnl:+.2f}",
                            strategy="monitor",
                        )
                        actions.append({"action": "exit", "reason": "edge_flipped", "ticker": ticker,
                                        "current_edge": current_edge, "pnl": unrealized_pnl})
                        continue

                    # ── DECISION 2: AUTO-EXIT — captured most of the upside, free capital ──
                    # Exit when mark >= 90c AND we've captured >80% of max profit.
                    # This lets us buy at 90c+ when edge exists, but exits positions
                    # that have already run up (e.g., bought at 50c, now at 95c).
                    max_profit_2 = pos.get("max_profit", 0) or 0
                    if mark_price >= 90 and max_profit_2 > 0 and unrealized_pnl > 0:
                        profit_captured = unrealized_pnl / max_profit_2
                        if profit_captured > 0.80:
                            await self.engine.exit_trade(
                                strategy=pos["strategy"],
                                ticker=ticker,
                                side=pos["side"],
                                count=pos["contracts"],
                                price_cents=mark_price,
                                reason=f"auto_exit_near_max mark={mark_price}c entry={pos['avg_entry_cents']}c captured={profit_captured:.0%}",
                            )
                            self.engine.log_event(
                                "paper_trade",
                                f"AUTO-EXIT {pos['side'].upper()} {pos['contracts']}x {ticker} @ {mark_price}c — {profit_captured:.0%} of max captured, P&L: ${unrealized_pnl:+.2f}",
                                strategy="monitor",
                            )
                            actions.append({"action": "exit", "reason": "near_max_value", "ticker": ticker,
                                            "mark_price": mark_price, "pnl": unrealized_pnl})
                            continue

                    # ── DECISION 3: TAKE PROFIT — up significantly ──
                    max_profit = pos.get("max_profit", 0) or 0
                    if max_profit > 0 and unrealized_pnl > 0:
                        profit_pct = unrealized_pnl / max_profit
                        # Take profit if we've captured >50% of max profit
                        if profit_pct > 0.50 and mark_price > 0:
                            await self.engine.exit_trade(
                                strategy=pos["strategy"],
                                ticker=ticker,
                                side=pos["side"],
                                count=pos["contracts"],
                                price_cents=mark_price,
                                reason=f"take_profit pnl=${unrealized_pnl:+.2f} ({profit_pct:.0%} of max)",
                            )
                            self.engine.log_event(
                                "paper_trade",
                                f"TAKE PROFIT {pos['side'].upper()} {pos['contracts']}x {ticker} @ {mark_price}c — P&L: ${unrealized_pnl:+.2f} ({profit_pct:.0%} of max ${max_profit:.2f})",
                                strategy="monitor",
                            )
                            actions.append({"action": "take_profit", "ticker": ticker,
                                            "pnl": unrealized_pnl, "profit_pct": profit_pct})
                            continue

                    # ── DECISION 4: ADD TO POSITION — edge has increased ──
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

        except Exception as e:
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

                for trade in trades:
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
                        # Record signal component outcomes for quality scoring
                        try:
                            self.signal_scorer.record_signal_outcome(
                                strategy=trade.get("strategy", ""),
                                signal_components=self._extract_signal_components(trade),
                                won=pnl > 0,
                            )
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

    async def get_positions_with_market_data(self) -> list[dict[str, Any]]:
        """
        Get open positions enriched with live market data.
        Called by the API for the frontend display.
        """
        positions = self.engine.get_open_positions()

        for pos in positions:
            ticker = pos["ticker"]
            try:
                await asyncio.sleep(0.15)
                data = await self.kalshi._get(f"/markets/{ticker}")
                market = data.get("market", data)

                yes_bid = market.get("yes_bid", 0) or 0
                yes_ask = market.get("yes_ask", 0) or 0
                no_bid = market.get("no_bid", 0) or 0
                no_ask = market.get("no_ask", 0) or 0

                pos["current_yes_bid"] = yes_bid
                pos["current_yes_ask"] = yes_ask
                pos["current_no_bid"] = no_bid
                pos["current_no_ask"] = no_ask

                # Mark-to-market: use same-side bid if spread is tight.
                # Wide spread = bid is garbage lowball, use last_price or ask.
                last_price = market.get("last_price", 0) or 0
                if pos["side"] == "yes":
                    bid, ask = yes_bid, yes_ask
                else:
                    bid, ask = no_bid, no_ask
                if bid > 0 and ask > 0 and bid >= ask * 0.5:
                    mark_price = bid  # tight spread — bid is real
                elif last_price > 0:
                    mark_price = last_price if pos["side"] == "yes" else (100 - last_price)
                elif ask > 0:
                    mark_price = ask  # wide spread, no trades — ask is best proxy
                elif bid > 0:
                    mark_price = bid
                else:
                    mark_price = pos["avg_entry_cents"]

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
                logger.debug("Position live price failed", ticker=ticker, error=str(e))

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

        # Live mode: verify Kalshi account balance matches BANKROLL
        if not self.engine.paper_mode:
            try:
                balance_data = await self.kalshi._get("/portfolio/balance")
                balance_cents = balance_data.get("balance", 0)
                balance_dollars = balance_cents / 100.0
                self.engine.log_event(
                    "info",
                    f"Kalshi balance: ${balance_dollars:.2f} (BANKROLL=${self.engine.bankroll:.2f})",
                )
                if balance_dollars < self.engine.bankroll * 0.5:
                    self.engine.log_event(
                        "warning",
                        f"Kalshi balance ${balance_dollars:.2f} is less than 50% of BANKROLL ${self.engine.bankroll:.2f}. "
                        f"Adjusting bankroll to match actual balance.",
                    )
                    self.engine.bankroll = balance_dollars
            except Exception as e:
                self.engine.log_event("warning", f"Could not verify Kalshi balance: {e}")

        # Start WebSocket connection for real-time price updates
        try:
            ws_ok = await self.ws.connect()
            if ws_ok:
                self.engine.log_event("info", "Kalshi WebSocket connected")
            else:
                self.engine.log_event("warning", "Kalshi WebSocket failed to connect, using REST fallback")
        except Exception as e:
            self.engine.log_event("warning", f"Kalshi WebSocket error: {e}, using REST fallback")

        # Boot cycle: run ALL strategies once immediately, rank globally, deploy best signals.
        # After this completes the independent loops take over on their normal schedules.
        asyncio.create_task(self._boot_then_start_loops())

    async def stop(self) -> None:
        """Stop the autonomous agent loops."""
        self._running = False
        self.engine.log_event("info", "Agent stopping")

        for task in [self._weather_task, self._main_task, self._crypto_task, self._monitor_task]:
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
            all_candidates: list[dict[str, Any]] = []

            for name, coro in [
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
        self._weather_task = asyncio.create_task(self._weather_loop(sleep_first=True))
        self._monitor_task = asyncio.create_task(self._monitor_loop(sleep_first=True))
        self._crypto_task = asyncio.create_task(self._crypto_loop(sleep_first=True))
        self._main_task = asyncio.create_task(self._main_strategy_loop(sleep_first=True))

    async def _weather_loop(self, sleep_first: bool = False) -> None:
        """Weather strategy loop — runs every 1 hour. Collects candidates and executes via global ranking."""
        if sleep_first:
            await asyncio.sleep(60 * 60)
        while self._running:
            try:
                if not self.engine.kill_switch:
                    candidates = await self.run_weather_cycle()
                    # Filter out skips
                    tradeable = [c for c in candidates if c.get("action") != "skip"]
                    if tradeable:
                        await self.execute_ranked_signals(tradeable)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Weather loop error: {e}", strategy="weather")
                logger.error("Weather loop error", error=str(e))

            await asyncio.sleep(60 * 60)

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
        while self._running:
            try:
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

                    # Proactive kill-switch: fire if daily loss limit breached
                    today_pnl = self.engine.get_today_pnl()
                    effective = self.engine.get_effective_bankroll()
                    dynamic_loss_limit = effective * 0.20
                    if today_pnl < -dynamic_loss_limit and not self.engine.kill_switch:
                        self.engine.kill_switch = True
                        self.engine.log_event(
                            "kill_switch",
                            f"Daily loss limit hit: ${today_pnl:.2f} < -${dynamic_loss_limit:.2f} "
                            f"(20% of effective bankroll ${effective:.2f}) — kill switch activated",
                            strategy="monitor",
                        )
                        logger.warning(
                            "Kill switch activated: daily loss limit breached",
                            today_pnl=today_pnl,
                            limit=-dynamic_loss_limit,
                        )

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

            self.engine.log_event(
                "daily_summary",
                f"Daily summary {yesterday}: {total_trades} trades, ${total_deployed:.2f} deployed, "
                f"settled P&L=${settled_pnl:+.2f} | {strat_summary}",
                strategy="monitor",
            )
        except Exception as e:
            logger.debug("Daily summary failed", error=str(e))

    def get_status(self) -> dict[str, Any]:
        """Get agent status."""
        return {
            "running": self._running,
            "paper_mode": self.engine.paper_mode,
            "kill_switch": self.engine.kill_switch,
            **self.engine.get_status(),
            "odds_api_credits_remaining": self.sports.remaining_credits,
            "websocket": self.ws.get_status(),
        }


# Singleton
_agent: KalshiAgent | None = None


def get_kalshi_agent() -> KalshiAgent:
    """Get or create the singleton agent."""
    global _agent
    if _agent is None:
        _agent = KalshiAgent()
    return _agent
