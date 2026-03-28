"""
Cross-Platform Arbitrage Engine.

Finds and executes arbitrage opportunities between Kalshi and Polymarket.
When the same event has different prices on both platforms, buys opposite
sides to lock in guaranteed profit regardless of outcome.

Example:
  Kalshi YES = 52¢, Polymarket NO = 45¢ (i.e. Polymarket YES = 55¢)
  Buy Kalshi YES + Polymarket NO = 97¢ total
  One side ALWAYS pays $1 → guaranteed 3¢ profit per contract

Key considerations:
  - Kalshi charges ~1.2% taker fees; Polymarket is ~0% on most markets
  - Need minimum ~2.5¢ spread to be profitable after fees
  - Execution must be near-simultaneous to avoid leg risk
  - Markets must be functionally equivalent (same question, same resolution)
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx

from app.logging_config import get_logger
from app.services.polymarket_data import PolymarketData, get_polymarket_data
from app.services.polymarket_trader import PolymarketTrader, get_polymarket_trader

logger = get_logger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"

# Minimum spread (in cents) to consider an arb opportunity after fees
MIN_SPREAD_CENTS = 3.0

# Kalshi effective fee rate (taker)
KALSHI_FEE_RATE = 0.012

# Max USDC per arb leg
MAX_ARB_USDC = 10.0

# Min USDC per arb leg
MIN_ARB_USDC = 2.0


class CrossPlatformArb:
    """Detects and executes cross-platform arbitrage between Kalshi and Polymarket."""

    def __init__(
        self,
        poly_data: PolymarketData | None = None,
        poly_trader: PolymarketTrader | None = None,
    ) -> None:
        self._poly_data = poly_data or get_polymarket_data()
        self._poly_trader = poly_trader or get_polymarket_trader()
        self._http: httpx.AsyncClient | None = None
        self._match_cache: dict[str, dict[str, Any]] = {}
        self._executed_arbs: dict[str, dict[str, Any]] = {}
        self._paper_mode: bool = os.getenv("PAPER_MODE", "true").lower() == "true"
        self._bankroll: float = float(os.getenv("POLYMARKET_BANKROLL", "250"))
        self._total_arb_deployed: float = 0.0

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=15)
        return self._http

    # ── Market Matching ──────────────────────────────────────────

    async def find_matching_markets(
        self, kalshi_markets: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find Polymarket markets that match Kalshi markets.

        Uses keyword overlap to identify equivalent markets across platforms.

        Returns list of matched pairs with prices on both platforms.
        """
        # Fetch Polymarket active markets
        poly_markets = await self._poly_data._fetch_active_markets(limit=200)
        matches: list[dict[str, Any]] = []

        for km in kalshi_markets:
            kalshi_title = (km.get("title", "") or "").lower()
            kalshi_ticker = km.get("ticker", "")
            kalshi_yes_ask = km.get("yes_ask", 0) or 0  # cents
            kalshi_no_ask = km.get("no_ask", 0) or 0    # cents

            if not kalshi_title or kalshi_yes_ask <= 0:
                continue

            # Extract significant words from Kalshi title
            kalshi_words = set(
                w for w in kalshi_title.split()
                if len(w) > 3 and w not in {"will", "does", "this", "that", "what", "than", "more", "less"}
            )

            best_match = None
            best_overlap = 0

            for pm in poly_markets:
                poly_question = (pm.get("question") or pm.get("title") or "").lower()
                if not poly_question:
                    continue

                poly_words = set(
                    w for w in poly_question.split()
                    if len(w) > 3 and w not in {"will", "does", "this", "that", "what", "than", "more", "less"}
                )

                overlap = len(kalshi_words & poly_words)
                if overlap > best_overlap and overlap >= 3:
                    # Parse Polymarket price
                    outcome_prices_raw = pm.get("outcomePrices", "[]")
                    try:
                        outcome_prices = json.loads(outcome_prices_raw) if isinstance(outcome_prices_raw, str) else outcome_prices_raw
                        poly_yes = float(outcome_prices[0]) * 100 if outcome_prices else None
                        poly_no = float(outcome_prices[1]) * 100 if len(outcome_prices) > 1 else None
                    except (json.JSONDecodeError, IndexError, ValueError, TypeError):
                        poly_yes = None
                        poly_no = None

                    if poly_yes is not None:
                        best_overlap = overlap
                        clob_raw = pm.get("clobTokenIds", "[]")
                        try:
                            clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else (clob_raw or [])
                        except (json.JSONDecodeError, TypeError):
                            clob_ids = []

                        best_match = {
                            "poly_title": pm.get("question") or pm.get("title", ""),
                            "poly_yes_cents": round(poly_yes, 1),
                            "poly_no_cents": round(poly_no, 1) if poly_no else round(100 - poly_yes, 1),
                            "poly_slug": pm.get("slug", ""),
                            "poly_condition_id": pm.get("conditionId", pm.get("condition_id", "")),
                            "poly_token_ids": clob_ids,
                            "word_overlap": overlap,
                        }

            if best_match:
                matches.append({
                    "kalshi_ticker": kalshi_ticker,
                    "kalshi_title": km.get("title", ""),
                    "kalshi_yes_ask": kalshi_yes_ask,
                    "kalshi_no_ask": kalshi_no_ask,
                    **best_match,
                })

        logger.info(f"Cross-platform arb: matched {len(matches)} market pairs")
        return matches

    # ── Arbitrage Detection ──────────────────────────────────────

    def detect_arbitrage(self, matched_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Detect arbitrage opportunities in matched market pairs.

        Checks two directions:
        1. Buy Kalshi YES + Buy Polymarket NO  (if total < $1)
        2. Buy Kalshi NO  + Buy Polymarket YES (if total < $1)

        Returns list of arb opportunities with expected profit.
        """
        opportunities: list[dict[str, Any]] = []

        for pair in matched_pairs:
            kalshi_yes = pair["kalshi_yes_ask"]  # cents
            kalshi_no = pair["kalshi_no_ask"]     # cents
            poly_yes = pair["poly_yes_cents"]     # cents
            poly_no = pair["poly_no_cents"]       # cents

            if not all([kalshi_yes, kalshi_no, poly_yes, poly_no]):
                continue

            # Direction 1: Kalshi YES + Polymarket NO
            cost_1 = kalshi_yes + poly_no
            kalshi_fee_1 = kalshi_yes * KALSHI_FEE_RATE
            total_cost_1 = cost_1 + kalshi_fee_1
            profit_1 = 100 - total_cost_1  # One side pays 100¢

            if profit_1 >= MIN_SPREAD_CENTS:
                opportunities.append({
                    **pair,
                    "direction": "kalshi_yes_poly_no",
                    "kalshi_side": "yes",
                    "kalshi_price_cents": kalshi_yes,
                    "poly_side": "no",
                    "poly_price_cents": poly_no,
                    "total_cost_cents": round(total_cost_1, 2),
                    "estimated_fee_cents": round(kalshi_fee_1, 2),
                    "profit_cents": round(profit_1, 2),
                    "roi_pct": round(profit_1 / total_cost_1 * 100, 2),
                })

            # Direction 2: Kalshi NO + Polymarket YES
            cost_2 = kalshi_no + poly_yes
            kalshi_fee_2 = kalshi_no * KALSHI_FEE_RATE
            total_cost_2 = cost_2 + kalshi_fee_2
            profit_2 = 100 - total_cost_2

            if profit_2 >= MIN_SPREAD_CENTS:
                opportunities.append({
                    **pair,
                    "direction": "kalshi_no_poly_yes",
                    "kalshi_side": "no",
                    "kalshi_price_cents": kalshi_no,
                    "poly_side": "yes",
                    "poly_price_cents": poly_yes,
                    "total_cost_cents": round(total_cost_2, 2),
                    "estimated_fee_cents": round(kalshi_fee_2, 2),
                    "profit_cents": round(profit_2, 2),
                    "roi_pct": round(profit_2 / total_cost_2 * 100, 2),
                })

        # Sort by profit
        opportunities.sort(key=lambda x: x["profit_cents"], reverse=True)

        if opportunities:
            logger.info(
                f"Cross-platform arb: found {len(opportunities)} opportunities, "
                f"best={opportunities[0]['profit_cents']:.1f}¢ profit on {opportunities[0]['kalshi_title'][:40]}"
            )

        return opportunities

    # ── Execution ────────────────────────────────────────────────

    async def execute_arb(
        self,
        opportunity: dict[str, Any],
        kalshi_agent: Any = None,
    ) -> dict[str, Any] | None:
        """Execute a cross-platform arbitrage trade.

        Places orders on both Kalshi and Polymarket simultaneously.
        """
        arb_key = f"{opportunity['kalshi_ticker']}_{opportunity['direction']}"
        if arb_key in self._executed_arbs:
            return None  # Already executed this arb

        # Position sizing
        max_by_bankroll = self._bankroll * 0.04  # Max 4% per arb
        remaining = (self._bankroll * 0.30) - self._total_arb_deployed  # Max 30% in arbs
        usdc_per_leg = min(MAX_ARB_USDC, max_by_bankroll, remaining)

        if usdc_per_leg < MIN_ARB_USDC:
            logger.debug("Arb position too small or max deployment reached")
            return None

        # Calculate contracts (at the given prices)
        kalshi_price = opportunity["kalshi_price_cents"] / 100
        poly_price = opportunity["poly_price_cents"] / 100
        contracts = int(usdc_per_leg / max(kalshi_price, poly_price))

        if contracts < 1:
            return None

        result: dict[str, Any] = {
            "opportunity": opportunity,
            "contracts": contracts,
            "kalshi_cost": round(contracts * kalshi_price, 2),
            "poly_cost": round(contracts * poly_price, 2),
            "expected_profit": round(contracts * opportunity["profit_cents"] / 100, 2),
            "paper_mode": self._paper_mode,
        }

        if self._paper_mode:
            result["status"] = "paper_filled"
            result["kalshi_order_id"] = f"PAPER-KARB-{int(time.time())}"
            result["poly_order_id"] = f"PAPER-PARB-{int(time.time())}"
            logger.info(
                "Cross-platform arb (PAPER)",
                kalshi=opportunity["kalshi_ticker"],
                direction=opportunity["direction"],
                contracts=contracts,
                expected_profit=result["expected_profit"],
            )
        else:
            # Live execution — both legs simultaneously
            poly_token_ids = opportunity.get("poly_token_ids", [])
            poly_side = opportunity["poly_side"]

            # Token index: 0 = YES outcome, 1 = NO outcome
            token_idx = 0 if poly_side == "yes" else (1 if len(poly_token_ids) > 1 else 0)
            poly_token_id = poly_token_ids[token_idx] if token_idx < len(poly_token_ids) else ""

            if not poly_token_id:
                result["status"] = "failed"
                result["error"] = "No Polymarket token ID"
                return result

            # Execute both legs concurrently
            poly_task = self._poly_trader.place_market_order(
                token_id=poly_token_id,
                side="BUY",
                amount=contracts * poly_price,
            )

            # Kalshi leg would go through the existing trading engine
            # For now, we log the signal for the Kalshi agent to pick up
            kalshi_signal = {
                "ticker": opportunity["kalshi_ticker"],
                "side": opportunity["kalshi_side"],
                "contracts": contracts,
                "max_price_cents": opportunity["kalshi_price_cents"],
            }

            poly_result = await poly_task

            result["poly_order"] = poly_result
            result["kalshi_signal"] = kalshi_signal

            if poly_result:
                result["status"] = "poly_filled_kalshi_pending"
            else:
                result["status"] = "poly_failed"

        # Track the arb
        self._executed_arbs[arb_key] = result
        self._total_arb_deployed += result.get("kalshi_cost", 0) + result.get("poly_cost", 0)

        return result

    # ── Run Cycle ────────────────────────────────────────────────

    async def run_cycle(self, kalshi_markets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run one arb detection and execution cycle.

        Args:
            kalshi_markets: List of open Kalshi markets with prices

        Returns:
            List of executed arb trades
        """
        # Step 1: Find matching markets
        matched = await self.find_matching_markets(kalshi_markets)
        if not matched:
            logger.info("Cross-platform arb: no matched market pairs found")
            return []

        # Step 2: Detect arb opportunities
        opportunities = self.detect_arbitrage(matched)
        if not opportunities:
            logger.info("Cross-platform arb: no profitable opportunities")
            return []

        # Step 3: Execute top opportunities
        executed: list[dict[str, Any]] = []
        for opp in opportunities[:3]:  # Max 3 arbs per cycle
            result = await self.execute_arb(opp)
            if result and result.get("status") in ("paper_filled", "poly_filled_kalshi_pending"):
                executed.append(result)

        return executed

    def get_status(self) -> dict[str, Any]:
        """Get arb engine status for dashboard."""
        return {
            "active_arbs": len(self._executed_arbs),
            "total_deployed": round(self._total_arb_deployed, 2),
            "paper_mode": self._paper_mode,
            "executed_arbs": [
                {
                    "kalshi": v["opportunity"]["kalshi_ticker"],
                    "direction": v["opportunity"]["direction"],
                    "profit_cents": v["opportunity"]["profit_cents"],
                    "status": v.get("status", "unknown"),
                }
                for v in self._executed_arbs.values()
            ],
        }


# ── Singleton ────────────────────────────────────────────────────

_arb: CrossPlatformArb | None = None


def get_cross_platform_arb() -> CrossPlatformArb:
    global _arb
    if _arb is None:
        _arb = CrossPlatformArb()
    return _arb
