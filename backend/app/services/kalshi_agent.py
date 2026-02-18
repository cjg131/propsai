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

        self._running = False
        self._weather_task: asyncio.Task | None = None
        self._sports_task: asyncio.Task | None = None
        self._crypto_task: asyncio.Task | None = None
        self._finance_task: asyncio.Task | None = None
        self._econ_task: asyncio.Task | None = None
        self._nba_props_task: asyncio.Task | None = None
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

            # Group by city code (scanner stores CITY_CONFIGS keys directly)
            by_city: dict[str, list[dict[str, Any]]] = {}
            for m in weather_markets:
                city_code = m.get("weather", {}).get("city_code", "")
                if city_code:
                    by_city.setdefault(city_code, []).append(m)

            for i, (city_key, city_markets) in enumerate(by_city.items()):
                if city_key not in CITY_CONFIGS:
                    self.engine.log_event(
                        "warning",
                        f"Unknown city code {city_key}, skipping",
                        strategy="weather",
                    )
                    continue

                # Rate-limit: pause between cities to avoid 429s
                if i > 0:
                    await asyncio.sleep(2.0)

                # Get forecasts from all sources
                forecasts = await self.weather.get_all_forecasts(city_key)
                source_count = len(forecasts.get("sources", {}))

                if source_count < 2:
                    self.engine.log_event(
                        "warning",
                        f"Only {source_count} sources for {city_key}, skipping",
                        strategy="weather",
                    )
                    continue

                self.engine.log_event(
                    "info",
                    f"{city_key}: {len(city_markets)} markets, {source_count} forecast sources",
                    strategy="weather",
                )

                # Evaluate each market
                for market in city_markets:
                    signal = await self._evaluate_weather_market(market, forecasts)
                    if signal:
                        results.append(signal)

        except Exception as e:
            self.engine.log_event("error", f"Weather cycle failed: {e}", strategy="weather")
            logger.error("Weather cycle error", error=str(e))

        self.engine.log_event(
            "info",
            f"Weather cycle complete: {len(results)} signals",
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
        consensus = self.weather.build_consensus(
            forecasts,
            strike_type=strike_type,
            floor_strike=floor_strike,
            cap_strike=cap_strike,
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

        # Record signal
        label = consensus.get("label", "")
        signal_id = self.engine.record_signal(
            strategy="weather",
            ticker=market["ticker"],
            side=signal["side"],
            our_prob=signal["our_prob"],
            kalshi_prob=signal["kalshi_prob"],
            market_title=market["title"],
            confidence=signal["confidence"],
            signal_source="weather_consensus",
            details=f"consensus={consensus.get('mean_high_f')}°F {label} sources={signal['source_count']}",
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

        # Execute trade (DCA allowed — per-ticker limit handles overexposure)
        trade = await self.engine.execute_trade(
            strategy="weather",
            ticker=market["ticker"],
            side=signal["side"],
            count=count,
            price_cents=price_cents,
            our_prob=signal["our_prob"],
            kalshi_prob=signal["kalshi_prob"],
            market_title=market["title"],
            signal_source="weather_consensus",
            signal_id=signal_id,
        )

        return {
            **signal,
            "ticker": market["ticker"],
            "title": market["title"],
            "trade": trade,
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
        results: list[dict[str, Any]] = []

        if not self.engine.strategy_enabled.get("sports", True):
            self.engine.log_event("info", "Sports strategy disabled", strategy="sports")
            return results

        try:
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
            single_stats = {"matched": 0, "no_match": 0, "no_edge": 0, "traded": 0}
            for market in single_markets:
                try:
                    signal = await self._evaluate_single_game(market, odds_by_sport, single_stats)
                    if signal:
                        results.append(signal)
                except Exception as e:
                    logger.debug("Single game eval failed", ticker=market.get("ticker"), error=str(e))

            self.engine.log_event(
                "info",
                f"Single-game: {len(single_markets)} eval | matched={single_stats['matched']} no_match={single_stats['no_match']} no_edge={single_stats['no_edge']} traded={single_stats['traded']}",
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

        if not odds_sport or not title or not ticker:
            stats["no_match"] += 1
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        # These are long-shot bets with high loss rates even with edge
        if min(yes_ask, no_ask) < 10:
            stats["no_match"] += 1
            return None

        events = odds_by_sport.get(odds_sport, [])
        if not events:
            stats["no_match"] += 1
            return None

        # Parse the ticker suffix to understand what this contract represents
        # Format: KXSERIES-DATECODETEAMS-OUTCOME
        ticker_parts = ticker.rsplit("-", 1)
        if len(ticker_parts) < 2:
            stats["no_match"] += 1
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
            stats["no_match"] += 1
            return None

        # Extract sharp consensus for this event
        consensus = self.sports.extract_sharp_consensus(matched_event)
        if consensus.get("sharp_books_found", 0) == 0:
            stats["no_match"] += 1
            return None

        sharp = consensus.get("sharp_consensus", {})
        if not sharp:
            stats["no_match"] += 1
            return None

        matched_event.get("home_team", "")
        matched_event.get("away_team", "")

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
            stats["no_match"] += 1
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
        max_edge = 0.15  # 15% cap — anything higher is likely a bad match
        if edge < min_edge:
            stats["no_edge"] += 1
            return None
        if edge > max_edge:
            logger.info("Edge too high (likely bad match)", ticker=ticker, edge=f"{edge:.1%}", title=title[:60])
            stats["no_edge"] += 1
            return None

        stats["matched"] += 1

        self.engine.log_event(
            "info",
            f"Single-game edge: {ticker} {side} edge={edge:.1%} (sharp={sharp_prob:.1%} vs kalshi={kalshi_implied:.1%}) {match_desc} | {title[:60]}",
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
            confidence=0.85,
            signal_source="sharp_single_game",
            details=f"sport={odds_sport} type={kalshi_type} match={match_desc} edge={edge:.4f}",
        )

        # Position sizing
        price_cents = yes_ask if side == "yes" else no_ask
        count = self.engine.calculate_position_size(
            strategy="sports",
            edge=edge,
            price_cents=price_cents,
            confidence=0.85,
            ticker=ticker,
        )

        if count > 0:
            # DCA allowed — per-ticker limit handles overexposure
            trade = await self.engine.execute_trade(
                strategy="sports",
                ticker=ticker,
                side=side,
                count=count,
                price_cents=price_cents,
                our_prob=sharp_prob,
                kalshi_prob=kalshi_implied,
                market_title=title,
                signal_source="sharp_single_game",
            )
            if trade:
                stats["traded"] += 1
                return {
                    "ticker": ticker,
                    "title": title,
                    "side": side,
                    "edge": edge,
                    "sharp_prob": sharp_prob,
                    "kalshi_implied": kalshi_implied,
                    "trade": trade,
                }

        return None

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

        # Position sizing
        price_cents = yes_ask if signal["side"] == "yes" else no_ask
        count = self.engine.calculate_position_size(
            strategy="sports",
            edge=signal["edge"],
            price_cents=price_cents,
            confidence=confidence,
            ticker=ticker,
        )

        result: dict[str, Any] = {
            **signal,
            "ticker": ticker,
            "title": title,
            "legs_priced": legs_priced,
            "legs_total": legs_total,
            "confidence": confidence,
        }

        if count > 0:
            # DCA allowed — per-ticker limit handles overexposure
            trade = await self.engine.execute_trade(
                strategy="sports",
                ticker=ticker,
                side=signal["side"],
                count=count,
                price_cents=price_cents,
                our_prob=signal["our_prob"],
                kalshi_prob=signal["kalshi_prob"],
                market_title=title,
                signal_source="parlay_pricer",
                signal_id=signal_id,
            )
            result["trade"] = trade
        else:
            result["action"] = "skip"
            result["reason"] = "position_size_zero"

        return result

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

        # Position sizing
        count = self.engine.calculate_position_size(
            strategy="crypto",
            edge=edge,
            price_cents=price_cents,
            confidence=confidence,
            ticker=ticker,
        )

        if count <= 0:
            return None

        # Execute trade
        trade = await self.engine.execute_trade(
            strategy="crypto",
            ticker=ticker,
            side=side,
            count=count,
            price_cents=price_cents,
            our_prob=our_prob,
            kalshi_prob=kalshi_prob,
            market_title=title,
            signal_source="crypto_momentum",
        )

        return {
            "ticker": ticker,
            "title": title,
            "coin": coin,
            "side": side,
            "edge": edge,
            "our_prob": our_prob,
            "confidence": confidence,
            "trade": trade,
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

        if not index or index not in signal_by_index:
            return None

        # Safety filter: skip very cheap contracts (< 10c on either side)
        if min(yes_ask, no_ask) < 10:
            return None

        signal = signal_by_index[index]
        p_up = signal["p_up"]
        confidence = signal["confidence"]

        # Determine market direction
        title_lower = title.lower()
        is_up_market = any(kw in title_lower for kw in ["up", "above", "higher", "close above", "gain"])
        is_down_market = any(kw in title_lower for kw in ["down", "below", "lower", "close below", "lose", "drop"])

        if not is_up_market and not is_down_market:
            is_up_market = True

        our_prob_yes = p_up if is_up_market else (1.0 - p_up)

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

        self.engine.log_event(
            "info",
            f"Finance edge: {ticker} {side} edge={edge:.1%} (our={our_prob:.1%} vs kalshi={kalshi_prob:.1%}) "
            f"index={index} p_up={p_up:.2f} conf={confidence:.2f}{poly_details} | {title[:60]}",
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
            signal_source="finance_momentum",
            details=f"index={index} p_up={p_up:.4f} intraday={signal['intraday_momentum']:.4f} "
                    f"futures={signal['futures_signal']:.4f} vix={signal['vix_signal']:.4f} "
                    f"ma={signal['ma_signal']:.4f}",
        )
        self._record_cross_signal("finance", ticker, side, our_prob, kalshi_prob, confidence)

        confidence = self._apply_correlation_adjustment("finance", ticker, confidence)

        count = self.engine.calculate_position_size(
            strategy="finance", edge=edge, price_cents=price_cents,
            confidence=confidence, ticker=ticker,
        )

        if count <= 0:
            return None

        trade = await self.engine.execute_trade(
            strategy="finance", ticker=ticker, side=side, count=count,
            price_cents=price_cents, our_prob=our_prob, kalshi_prob=kalshi_prob,
            market_title=title, signal_source="finance_momentum",
        )

        return {
            "ticker": ticker, "title": title, "index": index,
            "side": side, "edge": edge, "our_prob": our_prob,
            "confidence": confidence, "trade": trade,
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

        if econ_type == "cpi" and signal.get("estimated_next_yoy") is not None:
            if threshold_match:
                threshold = float(threshold_match.group(1)) / 100.0
                our_prob_yes = self.econ.estimate_probability_above(
                    signal["estimated_next_yoy"], threshold, volatility=0.05
                )
                confidence = 0.5
            else:
                return None

        elif econ_type == "fed_funds":
            if "cut" in title_lower:
                our_prob_yes = signal.get("p_cut", 0.3)
                confidence = 0.4
            elif "hike" in title_lower or "raise" in title_lower:
                our_prob_yes = signal.get("p_hike", 0.1)
                confidence = 0.4
            elif "hold" in title_lower or "unchanged" in title_lower:
                our_prob_yes = signal.get("p_hold", 0.6)
                confidence = 0.4
            else:
                return None

        elif econ_type == "gas_price" and signal.get("estimated_next") is not None:
            if threshold_price_match:
                threshold = float(threshold_price_match.group(1))
                our_prob_yes = self.econ.estimate_probability_above(
                    signal["estimated_next"], threshold, volatility=0.03
                )
                confidence = 0.4
            else:
                return None

        elif econ_type == "unemployment" and signal.get("estimated_next_rate") is not None:
            if threshold_match:
                threshold = float(threshold_match.group(1))
                our_prob_yes = self.econ.estimate_probability_above(
                    signal["estimated_next_rate"], threshold, volatility=0.05
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

        count = self.engine.calculate_position_size(
            strategy="econ", edge=edge, price_cents=price_cents,
            confidence=confidence, ticker=ticker,
        )

        if count <= 0:
            return None

        trade = await self.engine.execute_trade(
            strategy="econ", ticker=ticker, side=side, count=count,
            price_cents=price_cents, our_prob=our_prob, kalshi_prob=kalshi_prob,
            market_title=title, signal_source=f"econ_{econ_type}",
        )

        return {
            "ticker": ticker, "title": title, "econ_type": econ_type,
            "side": side, "edge": edge, "our_prob": our_prob,
            "confidence": confidence, "trade": trade,
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

            # Step 4: Evaluate each market
            for market in props_markets:
                try:
                    result = await self._evaluate_nba_prop(market, name_to_player, loop)
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
        if edge < nba_thresh["min_edge"] or confidence < nba_thresh["min_confidence"]:
            return None

        self.engine.log_event(
            "info",
            f"NBA prop edge: {ticker} {side} edge={edge:.1%} (our={our_prob:.1%} vs kalshi={kalshi_prob:.1%}) "
            f"{player_name} {prop_type} line={line} pred={predicted_value} conf={confidence:.2f} | {title[:60]}",
            strategy="nba_props",
        )

        self.engine.record_signal(
            strategy="nba_props", ticker=ticker, side=side,
            our_prob=our_prob, kalshi_prob=kalshi_prob,
            market_title=title, confidence=confidence,
            signal_source="smart_predictor",
            details=f"player={player_name} prop={prop_type} line={line} "
                    f"pred={predicted_value} over_p={over_prob:.3f} "
                    f"agreement={prediction.get('ensemble_agreement', 0):.3f}",
        )
        self._record_cross_signal("nba_props", ticker, side, our_prob, kalshi_prob, confidence)

        confidence = self._apply_correlation_adjustment("nba_props", ticker, confidence)

        count = self.engine.calculate_position_size(
            strategy="nba_props", edge=edge, price_cents=price_cents,
            confidence=confidence, ticker=ticker,
        )

        if count <= 0:
            return None

        trade = await self.engine.execute_trade(
            strategy="nba_props", ticker=ticker, side=side, count=count,
            price_cents=price_cents, our_prob=our_prob, kalshi_prob=kalshi_prob,
            market_title=title, signal_source="smart_predictor",
            notes=f"{player_name} {prop_type} line={line} pred={predicted_value}",
        )

        return {
            "ticker": ticker, "title": title,
            "player": player_name, "prop_type": prop_type,
            "line": line, "predicted_value": predicted_value,
            "side": side, "edge": edge, "our_prob": our_prob,
            "confidence": confidence, "trade": trade,
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

        # Start WebSocket connection for real-time price updates
        try:
            ws_ok = await self.ws.connect()
            if ws_ok:
                self.engine.log_event("info", "Kalshi WebSocket connected")
            else:
                self.engine.log_event("warning", "Kalshi WebSocket failed to connect, using REST fallback")
        except Exception as e:
            self.engine.log_event("warning", f"Kalshi WebSocket error: {e}, using REST fallback")

        self._weather_task = asyncio.create_task(self._staggered_loop(self._weather_loop, 0))
        self._monitor_task = asyncio.create_task(self._staggered_loop(self._monitor_loop, 20))
        self._crypto_task = asyncio.create_task(self._staggered_loop(self._crypto_loop, 60))
        self._sports_task = asyncio.create_task(self._staggered_loop(self._sports_loop, 180))
        self._finance_task = asyncio.create_task(self._staggered_loop(self._finance_loop, 300))
        self._econ_task = asyncio.create_task(self._staggered_loop(self._econ_loop, 420))
        self._nba_props_task = asyncio.create_task(self._staggered_loop(self._nba_props_loop, 540))

    async def stop(self) -> None:
        """Stop the autonomous agent loops."""
        self._running = False
        self.engine.log_event("info", "Agent stopping")

        for task in [self._weather_task, self._sports_task, self._crypto_task, self._finance_task, self._econ_task, self._nba_props_task, self._monitor_task]:
            if task:
                task.cancel()

        try:
            await self.ws.close()
        except Exception:
            pass

        logger.info("Kalshi agent stopped")

    async def _weather_loop(self) -> None:
        """Weather strategy loop — runs every 4 hours."""
        while self._running:
            try:
                if not self.engine.kill_switch:
                    await self.run_weather_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Weather loop error: {e}", strategy="weather")
                logger.error("Weather loop error", error=str(e))

            await asyncio.sleep(4 * 60 * 60)

    async def _sports_loop(self) -> None:
        """Sports strategy loop — runs every 5 minutes."""
        while self._running:
            try:
                if not self.engine.kill_switch:
                    await self.run_sports_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Sports loop error: {e}", strategy="sports")
                logger.error("Sports loop error", error=str(e))

            await asyncio.sleep(5 * 60)

    async def _crypto_loop(self) -> None:
        """Crypto strategy loop — runs every 2 minutes (15-min markets need fast reaction)."""
        while self._running:
            try:
                if not self.engine.kill_switch:
                    await self.run_crypto_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Crypto loop error: {e}", strategy="crypto")
                logger.error("Crypto loop error", error=str(e))

            await asyncio.sleep(2 * 60)

    async def _nba_props_loop(self) -> None:
        """NBA props strategy loop — runs every 15 minutes."""
        while self._running:
            try:
                if not self.engine.kill_switch:
                    await self.run_nba_props_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"NBA props loop error: {e}", strategy="nba_props")
                logger.error("NBA props loop error", error=str(e))

            await asyncio.sleep(15 * 60)

    async def _finance_loop(self) -> None:
        """Finance strategy loop — runs every 10 minutes (daily markets, no rush)."""
        while self._running:
            try:
                if not self.engine.kill_switch:
                    await self.run_finance_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Finance loop error: {e}", strategy="finance")
                logger.error("Finance loop error", error=str(e))

            await asyncio.sleep(10 * 60)

    async def _econ_loop(self) -> None:
        """Econ strategy loop — runs every 30 minutes (slow-moving data)."""
        while self._running:
            try:
                if not self.engine.kill_switch:
                    await self.run_econ_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Econ loop error: {e}", strategy="econ")
                logger.error("Econ loop error", error=str(e))

            await asyncio.sleep(30 * 60)

    async def _monitor_loop(self) -> None:
        """Position monitor + settlement loop — runs every 2 minutes."""
        while self._running:
            try:
                if not self.engine.kill_switch:
                    await self.run_monitor_cycle()
                    await self.run_settlement_cycle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.engine.log_event("error", f"Monitor loop error: {e}", strategy="monitor")
                logger.error("Monitor loop error", error=str(e))

            await asyncio.sleep(2 * 60)

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
