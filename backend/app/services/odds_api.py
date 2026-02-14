from __future__ import annotations
import asyncio
import os
import httpx
from app.logging_config import get_logger

logger = get_logger(__name__)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Map The Odds API market keys to our prop type names
MARKET_MAP = {
    "player_points": "points",
    "player_rebounds": "rebounds",
    "player_assists": "assists",
    "player_threes": "threes",
}

ALLOWED_BOOKS = {"draftkings", "fanduel"}

BOOK_DISPLAY = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
}


class OddsAPIClient:
    def __init__(self):
        from app.config import get_settings
        settings = get_settings()
        self.api_key = settings.the_odds_api_key
        self.client = httpx.AsyncClient(timeout=30)
        self._sem = asyncio.Semaphore(5)  # max 5 concurrent requests

    async def get_nba_events(self) -> list[dict]:
        """Get today's NBA events/games."""
        r = await self.client.get(
            f"{ODDS_API_BASE}/sports/{SPORT}/events/",
            params={"apiKey": self.api_key},
        )
        remaining = r.headers.get("x-requests-remaining", "?")
        logger.info(f"Odds API: events fetched, {remaining} requests remaining")
        r.raise_for_status()
        return r.json()

    async def get_player_props(self, event_id: str) -> list[dict]:
        """
        Get player prop odds for a specific event.
        Returns a flat list of prop lines:
        [{"player": "...", "prop_type": "points", "line": 24.5,
          "over_odds": -110, "under_odds": -110, "book": "DraftKings"}, ...]
        """
        markets = ",".join(MARKET_MAP.keys())
        r = await self.client.get(
            f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
            params={
                "apiKey": self.api_key,
                "regions": "us",
                "markets": markets,
                "oddsFormat": "american",
            },
        )
        remaining = r.headers.get("x-requests-remaining", "?")
        logger.info(f"Odds API: props for {event_id}, {remaining} remaining")
        r.raise_for_status()
        data = r.json()

        props = []
        home_team = data.get("home_team", "")
        away_team = data.get("away_team", "")

        for bookmaker in data.get("bookmakers", []):
            book_key = bookmaker.get("key", "")
            if book_key not in ALLOWED_BOOKS:
                continue
            book_name = BOOK_DISPLAY.get(book_key, book_key.title())

            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                prop_type = MARKET_MAP.get(market_key)
                if not prop_type:
                    continue

                # Group outcomes by player (Over/Under pairs)
                player_outcomes: dict[str, dict] = {}
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", "")
                    name = outcome.get("name", "")  # "Over" or "Under"
                    point = outcome.get("point", 0)
                    price = outcome.get("price", 0)

                    if player not in player_outcomes:
                        player_outcomes[player] = {
                            "player": player,
                            "prop_type": prop_type,
                            "line": point,
                            "book": book_name,
                            "book_key": book_key,
                            "home_team": home_team,
                            "away_team": away_team,
                            "event_id": event_id,
                        }

                    if name == "Over":
                        player_outcomes[player]["over_odds"] = price
                        player_outcomes[player]["line"] = point
                    elif name == "Under":
                        player_outcomes[player]["under_odds"] = price

                for po in player_outcomes.values():
                    if "over_odds" in po and "under_odds" in po:
                        props.append(po)

        return props

    async def _fetch_event_props(self, event_id: str) -> list[dict]:
        """Fetch props for one event, respecting the concurrency semaphore."""
        async with self._sem:
            try:
                return await self.get_player_props(event_id)
            except Exception as e:
                logger.warning(f"Error fetching props for {event_id}: {e}")
                return []

    async def get_game_lines(self) -> dict[str, dict]:
        """
        Fetch real spreads and totals (over/under) for today's NBA games.
        Returns {event_id: {"spread_home": -5.5, "spread_away": 5.5,
                            "total": 220.5, "home_team": "...", "away_team": "..."}}.
        Uses 1 API request (counts toward free tier quota).
        """
        try:
            r = await self.client.get(
                f"{ODDS_API_BASE}/sports/{SPORT}/odds/",
                params={
                    "apiKey": self.api_key,
                    "regions": "us",
                    "markets": "spreads,totals",
                    "oddsFormat": "american",
                    "bookmakers": "draftkings,fanduel",
                },
            )
            remaining = r.headers.get("x-requests-remaining", "?")
            logger.info(f"Odds API: game lines fetched, {remaining} requests remaining")
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            logger.warning(f"Odds API game lines failed: {e}")
            return {}

        game_lines: dict[str, dict] = {}
        for event in data:
            eid = event.get("id", "")
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            info: dict = {"home_team": home, "away_team": away}

            for bookmaker in event.get("bookmakers", []):
                if bookmaker.get("key") not in ALLOWED_BOOKS:
                    continue
                for market in bookmaker.get("markets", []):
                    mk = market.get("key", "")
                    outcomes = market.get("outcomes", [])
                    if mk == "spreads":
                        for o in outcomes:
                            if o.get("name") == home:
                                info.setdefault("spread_home", o.get("point", 0))
                            elif o.get("name") == away:
                                info.setdefault("spread_away", o.get("point", 0))
                    elif mk == "totals":
                        for o in outcomes:
                            if o.get("name") == "Over":
                                info.setdefault("total", o.get("point", 0))

            if "spread_home" in info or "total" in info:
                game_lines[eid] = info

        logger.info(f"Game lines: {len(game_lines)} games with spreads/totals")
        return game_lines

    async def get_all_todays_props(self) -> list[dict]:
        """
        Fetch player props for ALL today's NBA games concurrently.
        Returns a flat list of all prop lines across all games and books.
        """
        events = await self.get_nba_events()
        logger.info(f"Found {len(events)} NBA events today")

        # Fetch all events concurrently (max 5 at a time via semaphore)
        tasks = [self._fetch_event_props(e["id"]) for e in events]
        results = await asyncio.gather(*tasks)

        all_props = []
        for props in results:
            all_props.extend(props)

        logger.info(f"Total props fetched: {len(all_props)}")
        return all_props


_client: OddsAPIClient | None = None


def get_odds_api() -> OddsAPIClient:
    global _client
    if _client is None:
        _client = OddsAPIClient()
    return _client
