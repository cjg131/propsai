"""
Cross-market sports arbitrage strategy for Kalshi.
Compares Kalshi prices to sharp sportsbook lines (Pinnacle, Betfair) via The Odds API.
Finds mispricings where Kalshi retail traders are slow to react to sharp line moves.
"""
from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# ALL sports we monitor on The Odds API that also have Kalshi markets
MONITORED_SPORTS = [
    # Basketball
    "basketball_nba",
    "basketball_nba_all_stars",
    "basketball_ncaab",
    "basketball_wncaab",
    "basketball_euroleague",
    # Hockey
    "icehockey_nhl",
    "icehockey_ahl",
    "icehockey_liiga",
    "icehockey_sweden_hockey_league",
    # Soccer — Major leagues
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_germany_bundesliga",
    "soccer_italy_serie_a",
    "soccer_france_ligue_one",
    "soccer_uefa_champs_league",
    "soccer_uefa_champs_league_women",
    "soccer_uefa_europa_league",
    "soccer_uefa_europa_conference_league",
    # Soccer — Secondary leagues
    "soccer_efl_champ",
    "soccer_england_efl_cup",
    "soccer_fa_cup",
    "soccer_spain_segunda_division",
    "soccer_spain_copa_del_rey",
    "soccer_italy_serie_b",
    "soccer_germany_bundesliga2",
    "soccer_france_ligue_two",
    "soccer_netherlands_eredivisie",
    "soccer_portugal_primeira_liga",
    "soccer_spl",
    "soccer_denmark_superliga",
    "soccer_poland_ekstraklasa",
    "soccer_greece_super_league",
    "soccer_turkey_super_league",
    "soccer_switzerland_superleague",
    "soccer_austria_bundesliga",
    "soccer_sweden_allsvenskan",
    # Soccer — Americas & Asia
    "soccer_usa_mls",
    "soccer_mexico_ligamx",
    "soccer_brazil_campeonato",
    "soccer_argentina_primera_division",
    "soccer_saudi_arabia_pro_league",
    "soccer_japan_j_league",
    "soccer_australia_aleague",
    # Soccer — International
    "soccer_fifa_world_cup",
    "soccer_fifa_world_cup_qualifiers_europe",
    # Football
    "americanfootball_nfl",
    "americanfootball_ncaaf",
    # Tennis
    "tennis_atp_qatar_open",
    "tennis_wta_dubai",
    # MMA / Boxing
    "mma_mixed_martial_arts",
    "boxing_boxing",
    # Baseball
    "baseball_mlb",
    "baseball_ncaa",
    # Cricket
    "cricket_ipl",
    "cricket_t20_world_cup",
    # Rugby
    "rugbyunion_six_nations",
    "rugbyleague_nrl",
    # Lacrosse
    "lacrosse_ncaa",
]

# Sharp bookmakers — these set the "true" line
SHARP_BOOKS = {"pinnacle", "betfair_ex_eu", "betfair"}

# All bookmakers we want for comparison
ALL_BOOKS = {"pinnacle", "betfair_ex_eu", "betfair", "draftkings", "fanduel", "bovada"}

# Kalshi sports series prefixes (discovered from their API)
KALSHI_SPORTS_SERIES = {
    "basketball_nba": ["KXNBAGAME", "KXNBATOTAL", "KXNBATEAMTOTAL"],
    "basketball_ncaab": ["KXNCAABGAME"],
    "icehockey_nhl": ["KXNHLGAME"],
    "mma_mixed_martial_arts": ["KXUFC"],
    "soccer_epl": ["KXEPL"],
    "soccer_spain_la_liga": ["KXLALIGA"],
    "soccer_uefa_champs_league": ["KXUCL"],
}


def _american_to_implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def _decimal_to_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1:
        return 1.0
    return 1.0 / decimal_odds


class CrossMarketScanner:
    """
    Scans The Odds API for sharp lines and compares to Kalshi prices.
    Identifies mispricings where Kalshi implied probability diverges from sharp consensus.
    """

    def __init__(self, odds_api_key: str = "") -> None:
        self.api_key = odds_api_key
        self._http = httpx.AsyncClient(timeout=30.0)
        self._remaining_credits: int | None = None

    async def get_sports_list(self) -> list[dict[str, Any]]:
        """Get list of available sports from The Odds API."""
        if not self.api_key:
            return []

        try:
            resp = await self._http.get(
                f"{ODDS_API_BASE}/sports/",
                params={"apiKey": self.api_key},
            )
            self._update_credits(resp)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Failed to get sports list", error=str(e))
            return []

    async def get_events(self, sport: str) -> list[dict[str, Any]]:
        """Get upcoming events for a sport."""
        if not self.api_key:
            return []

        try:
            resp = await self._http.get(
                f"{ODDS_API_BASE}/sports/{sport}/events/",
                params={"apiKey": self.api_key},
            )
            self._update_credits(resp)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Failed to get events", sport=sport, error=str(e))
            return []

    async def get_odds(
        self,
        sport: str,
        markets: str = "h2h",
        bookmakers: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get odds for all events in a sport.
        markets: h2h, spreads, totals
        """
        if not self.api_key:
            return []

        try:
            params: dict[str, Any] = {
                "apiKey": self.api_key,
                "regions": "us,eu,uk",
                "markets": markets,
                "oddsFormat": "american",
            }
            if bookmakers:
                params["bookmakers"] = bookmakers

            resp = await self._http.get(
                f"{ODDS_API_BASE}/sports/{sport}/odds/",
                params=params,
            )
            self._update_credits(resp)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Failed to get odds", sport=sport, error=str(e))
            return []

    async def get_event_odds(
        self,
        sport: str,
        event_id: str,
        markets: str = "h2h",
    ) -> dict[str, Any]:
        """Get odds for a specific event."""
        if not self.api_key:
            return {}

        try:
            resp = await self._http.get(
                f"{ODDS_API_BASE}/sports/{sport}/events/{event_id}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "us,eu,uk",
                    "markets": markets,
                    "oddsFormat": "american",
                },
            )
            self._update_credits(resp)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Failed to get event odds", event_id=event_id, error=str(e))
            return {}

    def _update_credits(self, resp: httpx.Response) -> None:
        """Track remaining API credits from response headers."""
        remaining = resp.headers.get("x-requests-remaining")
        if remaining is not None:
            try:
                self._remaining_credits = int(remaining)
            except ValueError:
                pass
        used = resp.headers.get("x-requests-used")
        if used:
            logger.debug("Odds API credits", remaining=remaining, used=used)

    @property
    def remaining_credits(self) -> int | None:
        return self._remaining_credits

    def extract_sharp_consensus(self, event_data: dict[str, Any]) -> dict[str, Any]:
        """
        Extract sharp bookmaker consensus from event odds data.
        Returns implied probabilities from Pinnacle/Betfair.
        """
        bookmakers = event_data.get("bookmakers", [])
        sharp_probs: dict[str, list[float]] = {}  # outcome_key -> list of probs

        for bm in bookmakers:
            book_key = bm.get("key", "")
            if book_key not in SHARP_BOOKS:
                continue

            for market in bm.get("markets", []):
                market_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)
                    point = outcome.get("point")

                    # Build a unique key for this outcome
                    if point is not None:
                        outcome_key = f"{market_key}|{name}|{point}"
                    else:
                        outcome_key = f"{market_key}|{name}"

                    # Convert to implied probability
                    if isinstance(price, (int, float)):
                        if abs(price) >= 100:  # American odds
                            prob = _american_to_implied_prob(int(price))
                        elif price > 1:  # Decimal odds
                            prob = _decimal_to_implied_prob(price)
                        else:
                            continue

                        if outcome_key not in sharp_probs:
                            sharp_probs[outcome_key] = []
                        sharp_probs[outcome_key].append(prob)

        # Average sharp probabilities
        consensus: dict[str, float] = {}
        for key, probs in sharp_probs.items():
            consensus[key] = sum(probs) / len(probs)

        return {
            "event_id": event_data.get("id", ""),
            "home_team": event_data.get("home_team", ""),
            "away_team": event_data.get("away_team", ""),
            "commence_time": event_data.get("commence_time", ""),
            "sport": event_data.get("sport_key", ""),
            "sharp_consensus": consensus,
            "sharp_books_found": len([b for b in bookmakers if b.get("key") in SHARP_BOOKS]),
        }

    def find_mispricings(
        self,
        sharp_consensus: dict[str, Any],
        kalshi_markets: list[dict[str, Any]],
        min_edge: float = 0.05,
    ) -> list[dict[str, Any]]:
        """
        Compare sharp consensus to Kalshi prices.
        Returns list of mispricings with edge > min_edge.
        """
        mispricings = []
        consensus = sharp_consensus.get("sharp_consensus", {})

        for kalshi_market in kalshi_markets:
            ticker = kalshi_market.get("ticker", "")
            title = kalshi_market.get("title", "")
            yes_bid = kalshi_market.get("yes_bid", 0)
            yes_ask = kalshi_market.get("yes_ask", 0)
            no_bid = kalshi_market.get("no_bid", 0)
            no_ask = kalshi_market.get("no_ask", 0)
            volume = kalshi_market.get("volume", 0)

            if not yes_ask and not no_ask:
                continue

            # Try to match Kalshi market to sharp consensus outcome
            # This requires fuzzy matching between Kalshi titles and Odds API outcomes
            matched_outcome = self._match_kalshi_to_odds(
                kalshi_market, sharp_consensus
            )

            if not matched_outcome:
                continue

            sharp_prob = matched_outcome["sharp_prob"]
            outcome_side = matched_outcome["side"]  # "yes" or "no"

            # Calculate edge
            if outcome_side == "yes":
                kalshi_implied = yes_ask / 100.0 if yes_ask else 0
                if kalshi_implied <= 0:
                    continue
                edge = sharp_prob - kalshi_implied
                buy_price = yes_ask
            else:
                kalshi_implied = no_ask / 100.0 if no_ask else 0
                if kalshi_implied <= 0:
                    continue
                edge = sharp_prob - kalshi_implied
                buy_price = no_ask

            if edge >= min_edge:
                mispricings.append({
                    "ticker": ticker,
                    "title": title,
                    "side": outcome_side,
                    "sharp_prob": round(sharp_prob, 4),
                    "kalshi_implied": round(kalshi_implied, 4),
                    "edge": round(edge, 4),
                    "buy_price_cents": buy_price,
                    "volume": volume,
                    "home_team": sharp_consensus.get("home_team", ""),
                    "away_team": sharp_consensus.get("away_team", ""),
                    "event_id": sharp_consensus.get("event_id", ""),
                    "matched_outcome": matched_outcome.get("outcome_key", ""),
                })

        return mispricings

    def _match_kalshi_to_odds(
        self,
        kalshi_market: dict[str, Any],
        sharp_consensus: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Try to match a Kalshi market to a sharp consensus outcome.
        Uses team names and market type to find the right match.
        """
        title = kalshi_market.get("title", "").lower()
        home = sharp_consensus.get("home_team", "").lower()
        away = sharp_consensus.get("away_team", "").lower()
        consensus = sharp_consensus.get("sharp_consensus", {})

        # Try h2h (moneyline) matching
        for outcome_key, prob in consensus.items():
            if not outcome_key.startswith("h2h|"):
                continue

            parts = outcome_key.split("|")
            if len(parts) < 2:
                continue

            team_name = parts[1].lower()

            # Check if this team appears in the Kalshi title
            # Handle common abbreviations and partial matches
            if self._team_matches(team_name, title):
                return {
                    "outcome_key": outcome_key,
                    "sharp_prob": prob,
                    "side": "yes",  # Kalshi YES = team wins
                    "match_type": "h2h",
                }

        # Try totals matching
        for outcome_key, prob in consensus.items():
            if not outcome_key.startswith("totals|"):
                continue

            parts = outcome_key.split("|")
            if len(parts) < 3:
                continue

            direction = parts[1].lower()  # "Over" or "Under"
            point = parts[2]

            if "over" in title and direction == "over" and point in title:
                return {
                    "outcome_key": outcome_key,
                    "sharp_prob": prob,
                    "side": "yes",
                    "match_type": "totals",
                }
            elif "under" in title and direction == "under" and point in title:
                return {
                    "outcome_key": outcome_key,
                    "sharp_prob": prob,
                    "side": "yes",
                    "match_type": "totals",
                }

        return None

    def _team_matches(self, odds_team: str, kalshi_title: str) -> bool:
        """Check if an Odds API team name matches a Kalshi market title."""
        # Direct substring match
        if odds_team in kalshi_title:
            return True

        # Try last word of team name (e.g., "Lakers" from "Los Angeles Lakers")
        words = odds_team.split()
        if len(words) > 1 and words[-1] in kalshi_title:
            return True

        # Common abbreviation patterns
        abbrevs = {
            "los angeles lakers": ["lakers", "lal"],
            "boston celtics": ["celtics", "bos"],
            "golden state warriors": ["warriors", "gsw"],
            "new york knicks": ["knicks", "nyk"],
            "milwaukee bucks": ["bucks", "mil"],
            "denver nuggets": ["nuggets", "den"],
            "philadelphia 76ers": ["76ers", "sixers", "phi"],
            "phoenix suns": ["suns", "phx"],
            "dallas mavericks": ["mavericks", "mavs", "dal"],
            "miami heat": ["heat", "mia"],
            "cleveland cavaliers": ["cavaliers", "cavs", "cle"],
            "oklahoma city thunder": ["thunder", "okc"],
            "minnesota timberwolves": ["timberwolves", "wolves", "min"],
            "sacramento kings": ["kings", "sac"],
            "indiana pacers": ["pacers", "ind"],
            "new orleans pelicans": ["pelicans", "nop"],
            "orlando magic": ["magic", "orl"],
            "chicago bulls": ["bulls", "chi"],
            "houston rockets": ["rockets", "hou"],
            "atlanta hawks": ["hawks", "atl"],
            "toronto raptors": ["raptors", "tor"],
            "brooklyn nets": ["nets", "bkn"],
            "memphis grizzlies": ["grizzlies", "mem"],
            "portland trail blazers": ["blazers", "por"],
            "san antonio spurs": ["spurs", "sas"],
            "utah jazz": ["jazz", "uta"],
            "detroit pistons": ["pistons", "det"],
            "charlotte hornets": ["hornets", "cha"],
            "washington wizards": ["wizards", "was"],
        }

        for full_name, aliases in abbrevs.items():
            if odds_team == full_name or odds_team in aliases:
                if any(alias in kalshi_title for alias in aliases):
                    return True

        return False

    async def scan_all_sports(self, min_edge: float = 0.05) -> list[dict[str, Any]]:
        """
        Scan all monitored sports for mispricings.
        This is the main entry point for the sports strategy.
        """
        if not self.api_key:
            logger.warning("No Odds API key configured — skipping sports scan")
            return []

        all_signals = []

        for sport in MONITORED_SPORTS:
            try:
                events = await self.get_odds(sport, markets="h2h,totals")
                if not events:
                    continue

                for event in events:
                    consensus = self.extract_sharp_consensus(event)
                    if consensus.get("sharp_books_found", 0) == 0:
                        continue

                    # TODO: Match against actual Kalshi markets
                    # For now, store the sharp consensus for comparison
                    all_signals.append({
                        "sport": sport,
                        "event_id": event.get("id", ""),
                        "home_team": event.get("home_team", ""),
                        "away_team": event.get("away_team", ""),
                        "commence_time": event.get("commence_time", ""),
                        "sharp_consensus": consensus.get("sharp_consensus", {}),
                        "sharp_books_found": consensus.get("sharp_books_found", 0),
                    })

            except Exception as e:
                logger.warning("Sport scan failed", sport=sport, error=str(e))
                continue

        logger.info("Sports scan complete", total_events=len(all_signals))
        return all_signals

    async def close(self) -> None:
        await self._http.aclose()
