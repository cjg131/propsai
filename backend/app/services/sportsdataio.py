from __future__ import annotations
import httpx
from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.sportsdata.io/v3/nba"


class SportsDataIOClient:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.sportsdataio_api_key
        self.client = httpx.AsyncClient(
            base_url=BASE_URL,
            params={"key": self.api_key},
            timeout=30.0,
        )

    async def close(self):
        await self.client.aclose()

    async def _get(self, endpoint: str, params: dict | None = None) -> dict | list:
        try:
            response = await self.client.get(endpoint, params=params or {})
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "SportsDataIO API error",
                endpoint=endpoint,
                status_code=e.response.status_code,
                detail=e.response.text,
            )
            raise
        except httpx.RequestError as e:
            logger.error("SportsDataIO request error", endpoint=endpoint, error=str(e))
            raise

    # ---- Teams ----
    async def get_teams(self) -> list:
        return await self._get("/scores/json/AllTeams")

    # ---- Players ----
    async def get_players(self) -> list:
        return await self._get("/scores/json/Players")

    async def get_players_by_team(self, team: str) -> list:
        return await self._get(f"/scores/json/Players/{team}")

    # ---- Games / Schedule ----
    async def get_games_by_season(self, season: str) -> list:
        return await self._get(f"/scores/json/Games/{season}")

    async def get_games_by_date(self, date: str) -> list:
        """date format: YYYY-MM-DD"""
        return await self._get(f"/scores/json/GamesByDate/{date}")

    async def get_today_games(self) -> list:
        from datetime import date

        today = date.today().isoformat()
        return await self.get_games_by_date(today)

    # ---- Player Stats ----
    async def get_player_game_stats_by_date(self, date: str) -> list:
        return await self._get(f"/stats/json/PlayerGameStatsByDate/{date}")

    async def get_player_season_stats(self, season: str) -> list:
        return await self._get(f"/stats/json/PlayerSeasonStats/{season}")

    async def get_player_game_log(self, season: str, player_id: int) -> list:
        return await self._get(
            f"/stats/json/PlayerGameStatsBySeason/{season}/{player_id}"
        )

    # ---- Odds / Props ----
    async def get_player_props_by_date(self, date: str) -> list:
        return await self._get(f"/odds/json/PlayerPropsByDate/{date}")

    async def get_betting_player_props_by_game(self, game_id: int) -> list:
        return await self._get(f"/odds/json/BettingPlayerPropsByGameID/{game_id}")

    # ---- Referees ----
    async def get_referees(self) -> list:
        return await self._get("/scores/json/Referees")

    # ---- Injury Reports ----
    async def get_injuries(self) -> list:
        return await self._get("/scores/json/Injuries")

    async def get_injuries_by_date(self, date: str) -> list:
        return await self._get(f"/scores/json/InjuriesByDate/{date}")

    # ---- Standings ----
    async def get_standings(self, season: str) -> list:
        return await self._get(f"/scores/json/Standings/{season}")

    # ---- News ----
    async def get_news(self) -> list:
        return await self._get("/scores/json/News")

    async def get_news_by_player(self, player_id: int) -> list:
        return await self._get(f"/scores/json/NewsByPlayerID/{player_id}")


_client: SportsDataIOClient | None = None


def get_sportsdataio() -> SportsDataIOClient:
    global _client
    if _client is None:
        _client = SportsDataIOClient()
    return _client
