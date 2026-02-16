"""
Kalshi Autonomous Trading Agent.
Orchestrates weather and cross-market sports strategies.
Runs on a schedule: weather 4x/day, sports every 2-5 minutes.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import Any

from app.config import get_settings
from app.logging_config import get_logger
from app.services.kalshi_api import get_kalshi_client, KalshiClient
from app.services.trading_engine import get_trading_engine, TradingEngine
from app.services.weather_data import WeatherConsensus, CITY_CONFIGS
from app.services.cross_market_sports import CrossMarketScanner, MONITORED_SPORTS
from app.services.kalshi_scanner import KalshiScanner, parse_parlay_legs, categorize_market
from app.services.parlay_pricer import price_parlay_legs, teams_match

logger = get_logger(__name__)


class KalshiAgent:
    """
    Autonomous trading agent for Kalshi markets.
    Coordinates weather and sports strategies with the trading engine.
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

        self._running = False
        self._weather_task: asyncio.Task | None = None
        self._sports_task: asyncio.Task | None = None
        self._monitor_task: asyncio.Task | None = None

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

            for city_key, city_markets in by_city.items():
                if city_key not in CITY_CONFIGS:
                    self.engine.log_event(
                        "warning",
                        f"Unknown city code {city_key}, skipping",
                        strategy="weather",
                    )
                    continue

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

        # Safety filter: skip markets closing within 2 hours
        close_time = market.get("close_time", "")
        if close_time:
            try:
                close_dt = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
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
        signal = self.weather.generate_signal(
            consensus,
            kalshi_yes_price=yes_ask,
            kalshi_no_price=no_ask,
            min_edge=0.08,
            min_confidence=0.3,
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

        # Execute trade
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
        title_lower = title.lower()

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

        home_team = matched_event.get("home_team", "")
        away_team = matched_event.get("away_team", "")

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
                        match_desc = f"h2h|Draw"
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

        min_edge = 0.03  # 3% minimum edge
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

        min_edge = 0.03  # 3% minimum edge for parlays (sharp-priced)
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

            for pos in positions:
                ticker = pos["ticker"]
                try:
                    await asyncio.sleep(0.3)
                    data = await self.kalshi.get_markets(ticker=ticker)
                    market = data.get("market", data.get("markets", [{}])[0] if data.get("markets") else {})
                    if not market:
                        continue

                    yes_bid = market.get("yes_bid", 0) or 0
                    yes_ask = market.get("yes_ask", 0) or 0
                    no_bid = market.get("no_bid", 0) or 0
                    no_ask = market.get("no_ask", 0) or 0
                    mkt_status = market.get("status", "")

                    # Mark-to-market: use same-side bid (what we'd get if we exit).
                    # Fallback: last_price (flipped for NO), then ask, then entry.
                    last_price = market.get("last_price", 0) or 0
                    if pos["side"] == "yes":
                        if yes_bid > 0:
                            mark_price = yes_bid
                        elif last_price > 0:
                            mark_price = last_price
                        else:
                            mark_price = yes_ask if yes_ask > 0 else pos["avg_entry_cents"]
                    else:
                        if no_bid > 0:
                            mark_price = no_bid
                        elif last_price > 0:
                            mark_price = 100 - last_price
                        else:
                            mark_price = no_ask if no_ask > 0 else pos["avg_entry_cents"]
                    entry_price = pos["avg_entry_cents"]

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

                    # ── DECISION 1: EXIT — edge has flipped significantly ──
                    if current_edge < -0.05 and mark_price > 0:
                        exit_result = await self.engine.exit_trade(
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

                    # ── DECISION 2: TAKE PROFIT — up significantly ──
                    max_profit = pos.get("max_profit", 0) or 0
                    if max_profit > 0 and unrealized_pnl > 0:
                        profit_pct = unrealized_pnl / max_profit
                        # Take profit if we've captured >50% of max profit
                        if profit_pct > 0.50 and mark_price > 0:
                            exit_result = await self.engine.exit_trade(
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

                    # ── DECISION 3: ADD TO POSITION — edge has increased ──
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
                    logger.debug("Position monitor failed for ticker", ticker=ticker, error=str(e))

        except Exception as e:
            self.engine.log_event("error", f"Monitor cycle failed: {e}", strategy="monitor")
            logger.error("Monitor cycle error", error=str(e))

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

                # Mark-to-market: use same-side bid (what we'd get if we exit).
                # Fallback: last_price (flipped for NO), then ask, then entry.
                last_price = market.get("last_price", 0) or 0
                if pos["side"] == "yes":
                    if yes_bid > 0:
                        mark_price = yes_bid
                    elif last_price > 0:
                        mark_price = last_price
                    else:
                        mark_price = yes_ask if yes_ask > 0 else pos["avg_entry_cents"]
                else:
                    if no_bid > 0:
                        mark_price = no_bid
                    elif last_price > 0:
                        mark_price = 100 - last_price
                    else:
                        mark_price = no_ask if no_ask > 0 else pos["avg_entry_cents"]

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

    async def start(self) -> None:
        """Start the autonomous agent loops."""
        if self._running:
            return

        self._running = True
        self.engine.log_event("info", "Agent starting")
        logger.info("Kalshi agent starting", paper_mode=self.engine.paper_mode)

        self._weather_task = asyncio.create_task(self._weather_loop())
        self._sports_task = asyncio.create_task(self._sports_loop())
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self) -> None:
        """Stop the autonomous agent loops."""
        self._running = False
        self.engine.log_event("info", "Agent stopping")

        for task in [self._weather_task, self._sports_task, self._monitor_task]:
            if task:
                task.cancel()

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

    async def _monitor_loop(self) -> None:
        """Position monitor loop — runs every 2 minutes."""
        while self._running:
            try:
                if not self.engine.kill_switch:
                    await self.run_monitor_cycle()
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
        }


# Singleton
_agent: KalshiAgent | None = None


def get_kalshi_agent() -> KalshiAgent:
    """Get or create the singleton agent."""
    global _agent
    if _agent is None:
        _agent = KalshiAgent()
    return _agent
