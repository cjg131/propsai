from __future__ import annotations

import asyncio

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
    "player_steals": "steals",
    "player_blocks": "blocks",
    "player_turnovers": "turnovers",
    "player_points_rebounds_assists": "pra",
    "player_points_rebounds": "points_rebounds",
    "player_points_assists": "points_assists",
    "player_rebounds_assists": "rebounds_assists",
    "player_first_basket": "first_basket",
    "player_double_double": "double_double",
}

# Sharp books for game lines (h2h, spreads, totals)
SHARP_BOOKS = {"pinnacle", "betfair_ex_eu", "betfair"}

# Soft books for player props (Pinnacle rarely has props; DK/FD are the market)
PROP_BOOKS = {"draftkings", "fanduel", "betmgm", "caesars", "pointsbet"}

# All books we care about
ALLOWED_BOOKS = SHARP_BOOKS | PROP_BOOKS

BOOK_DISPLAY = {
    "draftkings": "DraftKings",
    "fanduel": "FanDuel",
    "pinnacle": "Pinnacle",
    "betfair_ex_eu": "Betfair",
    "betmgm": "BetMGM",
    "caesars": "Caesars",
    "pointsbet": "PointsBet",
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
        Returns a flat list of prop lines aggregated across books.
        For each player+prop_type, we compute a consensus line from all available books.
        [{"player": "...", "prop_type": "points", "line": 24.5,
          "over_odds": -110, "under_odds": -110, "book": "DraftKings",
          "consensus_over_prob": 0.52, "books_count": 3}, ...]
        """
        markets = ",".join(MARKET_MAP.keys())
        r = await self.client.get(
            f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
            params={
                "apiKey": self.api_key,
                "regions": "us,eu",  # eu gives Pinnacle access
                "markets": markets,
                "oddsFormat": "american",
                "bookmakers": ",".join(PROP_BOOKS),  # focus on prop-active books
            },
        )
        remaining = r.headers.get("x-requests-remaining", "?")
        logger.info(f"Odds API: props for {event_id}, {remaining} remaining")
        r.raise_for_status()
        data = r.json()

        home_team = data.get("home_team", "")
        away_team = data.get("away_team", "")

        # Aggregate by (player, prop_type) across all books
        # key: (player, prop_type) -> {line, over_probs: [], under_probs: [], books: []}
        aggregated: dict[tuple, dict] = {}

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

                player_outcomes: dict[str, dict] = {}
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", "")
                    name = outcome.get("name", "")  # "Over" or "Under"
                    point = outcome.get("point", 0)
                    price = outcome.get("price", 0)

                    if player not in player_outcomes:
                        player_outcomes[player] = {"line": point}
                    if name == "Over":
                        player_outcomes[player]["over_odds"] = price
                        player_outcomes[player]["line"] = point
                    elif name == "Under":
                        player_outcomes[player]["under_odds"] = price

                for player, po in player_outcomes.items():
                    if "over_odds" not in po or "under_odds" not in po:
                        continue
                    key = (player, prop_type)
                    if key not in aggregated:
                        aggregated[key] = {
                            "player": player,
                            "prop_type": prop_type,
                            "line": po["line"],
                            "home_team": home_team,
                            "away_team": away_team,
                            "event_id": event_id,
                            "over_probs": [],
                            "under_probs": [],
                            "books": [],
                            "best_book": book_name,
                            "over_odds": po["over_odds"],
                            "under_odds": po["under_odds"],
                        }
                    # Convert American odds to implied probability (no vig yet)
                    def _to_prob(odds: float) -> float:
                        if odds >= 0:
                            return 100.0 / (odds + 100.0)
                        return abs(odds) / (abs(odds) + 100.0)

                    over_p = _to_prob(po["over_odds"])
                    under_p = _to_prob(po["under_odds"])
                    # Remove vig by normalizing
                    total = over_p + under_p
                    if total > 0:
                        over_p /= total
                        under_p /= total
                    aggregated[key]["over_probs"].append(over_p)
                    aggregated[key]["under_probs"].append(under_p)
                    aggregated[key]["books"].append(book_name)

        # Build final prop list with consensus probabilities
        props = []
        for (player, prop_type), agg in aggregated.items():
            if not agg["over_probs"]:
                continue
            consensus_over = sum(agg["over_probs"]) / len(agg["over_probs"])
            consensus_under = sum(agg["under_probs"]) / len(agg["under_probs"])
            props.append({
                "player": player,
                "prop_type": prop_type,
                "line": agg["line"],
                "book": agg["best_book"],
                "book_key": agg["best_book"].lower(),
                "home_team": agg["home_team"],
                "away_team": agg["away_team"],
                "event_id": agg["event_id"],
                "over_odds": agg["over_odds"],
                "under_odds": agg["under_odds"],
                "consensus_over_prob": round(consensus_over, 4),
                "consensus_under_prob": round(consensus_under, 4),
                "books_count": len(agg["books"]),
                "books": agg["books"],
            })

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
        Uses sharp books (Pinnacle, Betfair) for accurate consensus lines.
        Returns {event_id: {"spread_home": -5.5, "spread_away": 5.5,
                            "total": 220.5, "home_team": "...", "away_team": "..."}}.
        """
        try:
            r = await self.client.get(
                f"{ODDS_API_BASE}/sports/{SPORT}/odds/",
                params={
                    "apiKey": self.api_key,
                    "regions": "us,eu",  # eu gives Pinnacle
                    "markets": "h2h,spreads,totals",
                    "oddsFormat": "american",
                    "bookmakers": ",".join(SHARP_BOOKS | PROP_BOOKS),
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
