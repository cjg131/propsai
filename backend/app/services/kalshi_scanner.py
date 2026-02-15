"""
Dynamic Kalshi market scanner.
Discovers ALL open markets, categorizes them, parses multi-game parlays,
and identifies tradeable opportunities across weather, sports, and other categories.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone
from typing import Any

from app.logging_config import get_logger
from app.services.kalshi_api import KalshiClient

logger = get_logger(__name__)

# ── Weather series — confirmed from Kalshi API ───────────────────
# Kalshi uses two naming conventions:
#   Old: KXHIGH + 2-letter city (KXHIGHNY, KXHIGHCHI, etc.)
#   New: KXHIGHT + 3-letter city (KXHIGHTATL, KXHIGHTDC, etc.)
# We enumerate ALL known series to avoid missing any.

WEATHER_SERIES_ALL: dict[str, dict[str, str]] = {
    # High temperature — old format
    "KXHIGHNY":   {"city_code": "NYC", "type": "high_temp"},
    "KXHIGHCHI":  {"city_code": "CHI", "type": "high_temp"},
    "KXHIGHLAX":  {"city_code": "LAX", "type": "high_temp"},
    "KXHIGHMIA":  {"city_code": "MIA", "type": "high_temp"},
    "KXHIGHDEN":  {"city_code": "DEN", "type": "high_temp"},
    # High temperature — new format
    "KXHIGHTATL": {"city_code": "ATL", "type": "high_temp"},
    "KXHIGHTAUS": {"city_code": "AUS", "type": "high_temp"},
    "KXHIGHTBOS": {"city_code": "BOS", "type": "high_temp"},
    "KXHIGHTCHI": {"city_code": "CHI", "type": "high_temp"},
    "KXHIGHTDC":  {"city_code": "DCA", "type": "high_temp"},
    "KXHIGHTDEN": {"city_code": "DEN", "type": "high_temp"},
    "KXHIGHTDFW": {"city_code": "DFW", "type": "high_temp"},
    "KXHIGHTHOU": {"city_code": "HOU", "type": "high_temp"},
    "KXHIGHTLAX": {"city_code": "LAX", "type": "high_temp"},
    "KXHIGHTLV":  {"city_code": "LAS", "type": "high_temp"},
    "KXHIGHTMIA": {"city_code": "MIA", "type": "high_temp"},
    "KXHIGHTMSP": {"city_code": "MSP", "type": "high_temp"},
    "KXHIGHTNYC": {"city_code": "NYC", "type": "high_temp"},
    "KXHIGHTPHL": {"city_code": "PHL", "type": "high_temp"},
    "KXHIGHTPHX": {"city_code": "PHX", "type": "high_temp"},
    "KXHIGHTSEA": {"city_code": "SEA", "type": "high_temp"},
    "KXHIGHTSFO": {"city_code": "SFO", "type": "high_temp"},
    # Low temperature
    "KXLOWTAUS":  {"city_code": "AUS", "type": "low_temp"},
    "KXLOWTCHI":  {"city_code": "CHI", "type": "low_temp"},
    "KXLOWTDEN":  {"city_code": "DEN", "type": "low_temp"},
    "KXLOWTLAX":  {"city_code": "LAX", "type": "low_temp"},
    "KXLOWTMIA":  {"city_code": "MIA", "type": "low_temp"},
    "KXLOWTNYC":  {"city_code": "NYC", "type": "low_temp"},
    # Rain
    "KXRAINNYC":  {"city_code": "NYC", "type": "rain"},
    "KXRAINCHI":  {"city_code": "CHI", "type": "rain"},
    "KXRAINLAX":  {"city_code": "LAX", "type": "rain"},
    "KXRAINDC":   {"city_code": "DCA", "type": "rain"},
    # Snow
    "KXSNOWNYC":  {"city_code": "NYC", "type": "snow"},
    "KXSNOWCHI":  {"city_code": "CHI", "type": "snow"},
    "KXSNOWBOS":  {"city_code": "BOS", "type": "snow"},
    "KXSNOWLA":   {"city_code": "LAX", "type": "snow"},
    "KXSNOWLAX":  {"city_code": "LAX", "type": "snow"},
    "KXSNOWDEN":  {"city_code": "DEN", "type": "snow"},
    "KXSNOWDC":   {"city_code": "DCA", "type": "snow"},
    # High temperature — additional discovered
    "KXHIGHAUS":  {"city_code": "AUS", "type": "high_temp"},
    "KXHIGHTNOLA": {"city_code": "NOL", "type": "high_temp"},
    # Low temperature — additional
    "KXLOWTLAX":  {"city_code": "LAX", "type": "low_temp"},
    "KXLOWTPHL":  {"city_code": "PHL", "type": "low_temp"},
    "KXLOWTDC":   {"city_code": "DCA", "type": "low_temp"},
    "KXLOWTATL":  {"city_code": "ATL", "type": "low_temp"},
    "KXLOWTBOS":  {"city_code": "BOS", "type": "low_temp"},
    "KXLOWTHOU":  {"city_code": "HOU", "type": "low_temp"},
    "KXLOWTLV":   {"city_code": "LAS", "type": "low_temp"},
    "KXLOWTPHX":  {"city_code": "PHX", "type": "low_temp"},
    "KXLOWTSEA":  {"city_code": "SEA", "type": "low_temp"},
    "KXLOWTSFO":  {"city_code": "SFO", "type": "low_temp"},
    "KXLOWTDFW":  {"city_code": "DFW", "type": "low_temp"},
    "KXLOWTMSP":  {"city_code": "MSP", "type": "low_temp"},
    # Snow — monthly
    "KXSNOWDET":  {"city_code": "DET", "type": "snow"},
    "KXSNOWSLC":  {"city_code": "SLC", "type": "snow"},
    "KXSNOWPHL":  {"city_code": "PHL", "type": "snow"},
    # Rain — monthly
    "KXRAINNYCM": {"city_code": "NYC", "type": "rain_monthly"},
    # Standalone
    "KXTORNADO":  {"city_code": "", "type": "tornado"},
    "KXHURRICANE": {"city_code": "", "type": "hurricane"},
}

# City code mapping for weather forecasts
WEATHER_CITY_CODES = {
    "NYC": "New York", "CHI": "Chicago", "LAX": "Los Angeles", "MIA": "Miami",
    "DCA": "Washington DC", "HOU": "Houston", "ATL": "Atlanta", "DFW": "Dallas",
    "DEN": "Denver", "PHL": "Philadelphia", "SFO": "San Francisco", "SEA": "Seattle",
    "BOS": "Boston", "AUS": "Austin", "LAS": "Las Vegas", "PHX": "Phoenix",
    "MSP": "Minneapolis", "NOL": "New Orleans", "DET": "Detroit", "SLC": "Salt Lake City",
}

# ── Parlay leg patterns ────────────────────────────────────────────

# "yes TeamName" → moneyline win
RE_MONEYLINE = re.compile(r"^(yes|no)\s+(.+?)$")
# "yes TeamName wins by over X.5 Points" → spread
RE_SPREAD = re.compile(r"^(yes|no)\s+(.+?)\s+wins\s+by\s+over\s+([\d.]+)\s+Points$")
# "yes Over X.5 points scored" → total
RE_TOTAL = re.compile(r"^(yes|no)\s+Over\s+([\d.]+)\s+points\s+scored$")
# "yes Both Teams To Score" → BTTS (soccer)
RE_BTTS = re.compile(r"^(yes|no)\s+Both\s+Teams?\s+To\s+Score$", re.IGNORECASE)
# "yes Over X.5 goals scored" → soccer total
RE_GOALS = re.compile(r"^(yes|no)\s+Over\s+([\d.]+)\s+goals\s+scored$")


def parse_parlay_legs(title: str) -> list[dict[str, Any]]:
    """
    Parse a multi-game parlay title into individual legs.
    Title format: "yes TeamA,yes TeamB wins by over 3.5 Points,no Over 150.5 points scored"
    """
    legs = []
    # Split on comma, but be careful with team names containing commas
    raw_legs = [leg.strip() for leg in title.split(",") if leg.strip()]

    for raw in raw_legs:
        leg = _parse_single_leg(raw)
        if leg:
            legs.append(leg)

    return legs


def _parse_single_leg(raw: str) -> dict[str, Any] | None:
    """Parse a single parlay leg string."""
    raw = raw.strip()
    if not raw:
        return None

    # Try spread first (more specific)
    m = RE_SPREAD.match(raw)
    if m:
        return {
            "type": "spread",
            "direction": m.group(1),  # yes/no
            "team": m.group(2).strip(),
            "line": float(m.group(3)),
            "raw": raw,
        }

    # Try total points
    m = RE_TOTAL.match(raw)
    if m:
        return {
            "type": "total",
            "direction": m.group(1),
            "line": float(m.group(2)),
            "raw": raw,
        }

    # Try goals total (soccer)
    m = RE_GOALS.match(raw)
    if m:
        return {
            "type": "goals_total",
            "direction": m.group(1),
            "line": float(m.group(2)),
            "raw": raw,
        }

    # Try BTTS
    m = RE_BTTS.match(raw)
    if m:
        return {
            "type": "btts",
            "direction": m.group(1),
            "raw": raw,
        }

    # Try moneyline (catch-all for "yes TeamName" or "no TeamName")
    m = RE_MONEYLINE.match(raw)
    if m:
        team = m.group(2).strip()
        # Filter out things that look like other patterns we missed
        if team and not team.startswith("Over") and not team.startswith("Under"):
            return {
                "type": "moneyline",
                "direction": m.group(1),
                "team": team,
                "raw": raw,
            }

    return None


def categorize_market(market: dict[str, Any]) -> str:
    """Categorize a Kalshi market into weather, sports, politics, finance, or other."""
    ticker = market.get("ticker", "")
    title = market.get("title", "").lower()
    series = market.get("series_ticker", "")

    # Weather — check if series_ticker or ticker prefix matches any known weather series
    for ws in WEATHER_SERIES_ALL:
        if ticker.startswith(ws) or series.startswith(ws):
            return "weather"
    if any(kw in title for kw in ["high temp", "low temp", "snowfall", "rainfall", "tornado", "hurricane", "maximum temperature", "minimum temperature", "rain in", "snow in"]):
        return "weather"

    # Sports
    if "KXMVESPORTS" in ticker or "KXMVESPORTS" in series:
        return "sports_parlay"
    if any(ticker.startswith(p) for p in ["KXNBA", "KXNFL", "KXNHL", "KXMLB", "KXNCAA", "KXMLS", "KXUFC", "KXPGA"]):
        return "sports_futures"
    if any(ticker.startswith(p) for p in ["KXNBAPTS", "KXNBAAST", "KXNBAREB"]):
        return "sports_props"

    # Politics
    if any(kw in title for kw in ["president", "election", "congress", "senate", "democrat", "republican"]):
        return "politics"

    # Finance
    if any(kw in title for kw in ["s&p", "nasdaq", "bitcoin", "fed rate", "interest rate", "gdp", "inflation"]):
        return "finance"

    return "other"


class KalshiScanner:
    """
    Scans all open Kalshi markets, categorizes them, and identifies
    tradeable opportunities with liquidity.
    """

    def __init__(self, kalshi: KalshiClient) -> None:
        self.kalshi = kalshi

    async def scan_all_open_markets(
        self,
        min_volume: int = 0,
        max_pages: int = 15,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Fetch all open markets and categorize them.
        Returns dict of category → list of parsed markets.
        """
        all_markets: list[dict[str, Any]] = []
        cursor = None

        for i in range(max_pages):
            if i > 0:
                await asyncio.sleep(1.0)  # Rate limit
            try:
                data = await self.kalshi.get_markets(
                    status="open", limit=200, cursor=cursor,
                )
                markets = data.get("markets", [])
                all_markets.extend(markets)
                cursor = data.get("cursor")
                if not cursor or not markets:
                    break
            except Exception as e:
                logger.warning("Market scan page failed", page=i, error=str(e))
                break

        logger.info("Market scan complete", total=len(all_markets))

        # Categorize
        categorized: dict[str, list[dict[str, Any]]] = {}
        for m in all_markets:
            cat = categorize_market(m)
            if cat not in categorized:
                categorized[cat] = []

            parsed = self._enrich_market(m, cat)
            if parsed:
                categorized[cat].append(parsed)

        # Log summary
        for cat, ms in categorized.items():
            liquid = [m for m in ms if (m.get("volume", 0) or 0) >= min_volume]
            logger.info(
                "Market category",
                category=cat,
                total=len(ms),
                liquid=len(liquid),
            )

        return categorized

    async def scan_weather_markets(self) -> list[dict[str, Any]]:
        """Scan all confirmed Kalshi weather series for active markets.
        
        NOTE: Kalshi uses status='active' for live tradeable markets,
        but the API filter accepts 'open' which returns active markets.
        We fetch without status filter and check locally to be safe.
        """
        weather_markets: list[dict[str, Any]] = []

        for series_ticker, config in WEATHER_SERIES_ALL.items():
            try:
                await asyncio.sleep(0.35)  # Rate limit: ~3 req/sec
                data = await self.kalshi.get_markets(
                    series_ticker=series_ticker, limit=50,
                )
                for m in data.get("markets", []):
                    # Only include active (tradeable) markets
                    if m.get("status") not in ("active", "open"):
                        continue
                    parsed = self._enrich_market(m, "weather")
                    if parsed:
                        # Attach city code and market type from our config
                        if "weather" not in parsed:
                            parsed["weather"] = {}
                        parsed["weather"]["city_code"] = config["city_code"]
                        parsed["weather"]["market_type"] = config["type"]
                        parsed["weather"]["series_ticker"] = series_ticker
                        weather_markets.append(parsed)
            except Exception as e:
                logger.debug("Weather series scan failed", series=series_ticker, error=str(e))
                continue

        logger.info("Weather scan complete", series_checked=len(WEATHER_SERIES_ALL), found=len(weather_markets))
        return weather_markets

    async def scan_sports_parlays(self, min_volume: int = 10) -> list[dict[str, Any]]:
        """Scan for sports parlay markets with liquidity."""
        parlays: list[dict[str, Any]] = []
        cursor = None

        for i in range(10):
            if i > 0:
                await asyncio.sleep(1.0)
            try:
                data = await self.kalshi.get_markets(
                    status="open",
                    series_ticker="KXMVESPORTSMULTIGAMEEXTENDED",
                    limit=200,
                    cursor=cursor,
                )
                markets = data.get("markets", [])
                for m in markets:
                    vol = m.get("volume", 0) or 0
                    yes_ask = m.get("yes_ask", 0) or 0
                    no_ask = m.get("no_ask", 0) or 0

                    # Skip no-liquidity markets
                    if vol < min_volume:
                        continue
                    if yes_ask <= 0 or no_ask <= 0:
                        continue
                    if yes_ask <= 2 or no_ask <= 2:
                        continue

                    parsed = self._enrich_market(m, "sports_parlay")
                    if parsed:
                        parlays.append(parsed)

                cursor = data.get("cursor")
                if not cursor or not markets:
                    break
            except Exception as e:
                logger.warning("Parlay scan page failed", page=i, error=str(e))
                break

        logger.info("Parlay scan complete", found=len(parlays))
        return parlays

    def _enrich_market(self, market: dict[str, Any], category: str) -> dict[str, Any] | None:
        """Add parsed metadata to a raw Kalshi market."""
        ticker = market.get("ticker", "")
        title = market.get("title", "")
        yes_ask = market.get("yes_ask", 0) or 0
        no_ask = market.get("no_ask", 0) or 0

        result: dict[str, Any] = {
            "ticker": ticker,
            "title": title,
            "category": category,
            "series_ticker": market.get("series_ticker", ""),
            "event_ticker": market.get("event_ticker", ""),
            "yes_bid": market.get("yes_bid", 0) or 0,
            "yes_ask": yes_ask,
            "no_bid": market.get("no_bid", 0) or 0,
            "no_ask": no_ask,
            "volume": market.get("volume", 0) or 0,
            "open_interest": market.get("open_interest", 0) or 0,
            "close_time": market.get("close_time", ""),
            "status": market.get("status", ""),
            "strike_type": market.get("strike_type", ""),
            "floor_strike": market.get("floor_strike"),
            "cap_strike": market.get("cap_strike"),
        }

        # Weather-specific parsing
        if category == "weather":
            result["weather"] = {
                "strike_type": market.get("strike_type", ""),
                "floor_strike": market.get("floor_strike"),
                "cap_strike": market.get("cap_strike"),
            }
            # Extract city from known series mapping
            for ws, ws_config in WEATHER_SERIES_ALL.items():
                if ticker.startswith(ws) or result["series_ticker"] == ws:
                    result["weather"]["city_code"] = ws_config["city_code"]
                    result["weather"]["market_type"] = ws_config["type"]
                    result["weather"]["city_name"] = WEATHER_CITY_CODES.get(ws_config["city_code"], ws_config["city_code"])
                    break

        # Parlay-specific parsing
        if category == "sports_parlay":
            legs = parse_parlay_legs(title)
            result["parlay"] = {
                "legs": legs,
                "num_legs": len(legs),
                "sports_detected": _detect_sports(legs),
            }

        return result


def _detect_sports(legs: list[dict[str, Any]]) -> list[str]:
    """Detect which sports are in a parlay based on leg content."""
    sports = set()
    for leg in legs:
        team = leg.get("team", "")
        leg_type = leg.get("type", "")

        # Soccer indicators
        if leg_type in ("goals_total", "btts"):
            sports.add("soccer")
            continue

        # Known soccer teams
        soccer_teams = [
            "Liverpool", "Barcelona", "Bayern Munich", "Aston Villa", "Lyon",
            "Roma", "Mallorca", "Villarreal", "Arsenal", "Chelsea", "Man City",
            "Man United", "Tottenham", "Real Madrid", "Atletico", "PSG",
            "Juventus", "Inter Milan", "AC Milan", "Napoli", "Dortmund",
        ]
        if any(st.lower() in team.lower() for st in soccer_teams):
            sports.add("soccer")
            continue

        # Tennis indicators
        tennis_names = [
            "Fritz", "Shelton", "Navarro", "Tauson", "Jovic", "Cerundolo",
            "Djokovic", "Sinner", "Alcaraz", "Swiatek", "Gauff", "Sabalenka",
        ]
        if any(tn.lower() in team.lower() for tn in tennis_names):
            sports.add("tennis")
            continue

        # Points-based → basketball
        if leg_type == "total" or leg_type == "spread":
            line = leg.get("line", 0)
            if line > 100:  # Basketball totals are 100+
                sports.add("basketball")
            elif line > 30:  # Could be basketball spread
                sports.add("basketball")
            continue

        # Default: likely basketball (college)
        if leg_type == "moneyline":
            sports.add("basketball")

    return sorted(sports)
