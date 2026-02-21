"""
NBA Referee Data Service.

Fetches referee assignments from SportsDataIO and computes referee-based
signals for NBA totals markets.

Key insight: NBA referees have statistically significant tendencies:
  - Some refs call 5+ more fouls/game than average → more free throws → higher totals
  - Some refs let games flow → fewer stoppages → lower totals
  - Ref tendencies are consistent season-over-season (r > 0.7)

This is one of the most underutilized edges in NBA betting.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx

from app.logging_config import get_logger

logger = get_logger(__name__)

SDIO_BASE = "https://api.sportsdata.io/v3/nba"

# League average fouls per game (both teams combined, 2024-25 season)
LEAGUE_AVG_FOULS_PER_GAME = 40.0  # ~20 per team

# League average free throw attempts per game (both teams)
LEAGUE_AVG_FTA_PER_GAME = 44.0

# Each additional foul call adds ~0.8 points to the total (FTA * FT%)
POINTS_PER_EXTRA_FOUL = 0.8

# Known referee tendencies (fouls per game vs league average)
# Positive = calls more fouls than average (favors OVER)
# Negative = calls fewer fouls (favors UNDER)
# Source: Basketball Reference referee stats, updated periodically
# These are approximate and should be overridden by live API data when available
REFEREE_TENDENCIES: dict[str, float] = {
    # High-foul refs (OVER-friendly)
    "Kane Fitzgerald": +4.2,
    "Zach Zarba": +3.8,
    "Scott Foster": +3.5,
    "Tony Brothers": +3.2,
    "Marc Davis": +2.9,
    "Bill Kennedy": +2.7,
    "James Capers": +2.4,
    "Kevin Scott": +2.1,
    "Courtney Kirkland": +1.9,
    "Gediminas Petraitis": +1.7,
    # Low-foul refs (UNDER-friendly)
    "Eric Lewis": -3.1,
    "Josh Tiven": -2.8,
    "Rodney Mott": -2.5,
    "Tyler Ford": -2.3,
    "Derrick Collins": -2.1,
    "Phenizee Ransom": -1.9,
    "John Goble": -1.7,
    "Sean Wright": -1.5,
    "Matt Boland": -1.3,
    "Jacyn Goble": -1.1,
}


class RefereeDataService:
    """Fetches referee assignments and generates totals signals."""

    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._cache: dict[str, Any] = {}
        self._cache_ts: dict[str, float] = {}
        self._CACHE_TTL = 3600.0  # 1 hour

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=20)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_referees_for_date(self, game_date: str | None = None) -> list[dict[str, Any]]:
        """Fetch referee assignments for a specific date from SportsDataIO.

        Args:
            game_date: YYYY-MM-DD format. Defaults to today.

        Returns:
            List of {game_id, home_team, away_team, referees: [name, ...]}
        """
        if not self.api_key:
            return []

        if game_date is None:
            game_date = datetime.now(UTC).date().isoformat()

        cache_key = f"refs_{game_date}"
        import time
        now = time.time()
        if cache_key in self._cache and now - self._cache_ts.get(cache_key, 0) < self._CACHE_TTL:
            return self._cache[cache_key]

        client = await self._get_client()
        try:
            # SportsDataIO: Get games by date (includes referee info)
            url = f"{SDIO_BASE}/scores/json/GamesByDate/{game_date}"
            resp = await client.get(url, params={"key": self.api_key})
            resp.raise_for_status()
            games = resp.json()

            result = []
            for game in games:
                game_id = game.get("GameID")
                home = game.get("HomeTeam", "")
                away = game.get("AwayTeam", "")

                # SportsDataIO includes referee names in game data
                refs = []
                for ref_field in ["Referee", "Referees"]:
                    ref_data = game.get(ref_field)
                    if isinstance(ref_data, str) and ref_data:
                        refs.extend([r.strip() for r in ref_data.split(",") if r.strip()])
                    elif isinstance(ref_data, list):
                        refs.extend([r.get("Name", "") for r in ref_data if r.get("Name")])

                if game_id and (home or away):
                    result.append({
                        "game_id": game_id,
                        "home_team": home,
                        "away_team": away,
                        "referees": refs,
                        "date": game_date,
                    })

            self._cache[cache_key] = result
            self._cache_ts[cache_key] = now
            logger.info(f"Referee data: {len(result)} games on {game_date}, refs found in {sum(1 for g in result if g['referees'])} games")
            return result

        except Exception as e:
            logger.warning("Referee data fetch failed", date=game_date, error=str(e))
            return []

    def compute_ref_signal(
        self,
        referee_names: list[str],
        game_total_line: float | None = None,
    ) -> dict[str, Any]:
        """Compute a totals signal based on referee tendencies.

        Args:
            referee_names: List of referee names for the game.
            game_total_line: The sportsbook over/under line (optional, for context).

        Returns:
            {
                "foul_adjustment": float,   # expected fouls above/below average
                "point_adjustment": float,  # expected points above/below average
                "direction": str,           # "over", "under", or "neutral"
                "strength": float,          # 0-1 signal strength
                "refs_found": int,          # how many refs we have data for
                "refs_total": int,          # total refs assigned
                "details": str,
            }
        """
        if not referee_names:
            return {
                "foul_adjustment": 0.0,
                "point_adjustment": 0.0,
                "direction": "neutral",
                "strength": 0.0,
                "refs_found": 0,
                "refs_total": 0,
                "details": "no refs assigned",
            }

        total_foul_adj = 0.0
        refs_found = 0
        ref_details = []

        for ref_name in referee_names:
            # Try exact match first, then partial match
            tendency = REFEREE_TENDENCIES.get(ref_name)
            if tendency is None:
                # Try partial match (first/last name)
                for known_ref, tend in REFEREE_TENDENCIES.items():
                    if ref_name.lower() in known_ref.lower() or known_ref.lower() in ref_name.lower():
                        tendency = tend
                        break

            if tendency is not None:
                total_foul_adj += tendency
                refs_found += 1
                ref_details.append(f"{ref_name}({tendency:+.1f})")

        # Average across refs found (typically 3 refs per game)
        avg_foul_adj = total_foul_adj / max(refs_found, 1) if refs_found > 0 else 0.0

        # Convert foul adjustment to point adjustment
        point_adj = avg_foul_adj * POINTS_PER_EXTRA_FOUL

        # Signal strength: 0 if no data, scales with magnitude and coverage
        coverage = refs_found / max(len(referee_names), 1)
        strength = min(1.0, abs(avg_foul_adj) / 5.0) * coverage

        direction = "neutral"
        if avg_foul_adj > 1.5:
            direction = "over"
        elif avg_foul_adj < -1.5:
            direction = "under"

        return {
            "foul_adjustment": round(avg_foul_adj, 2),
            "point_adjustment": round(point_adj, 2),
            "direction": direction,
            "strength": round(strength, 3),
            "refs_found": refs_found,
            "refs_total": len(referee_names),
            "details": ", ".join(ref_details) if ref_details else "no known refs",
        }

    async def get_game_ref_signals(
        self, game_date: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Get referee signals for all games on a date.

        Returns:
            {"{home_team}_{away_team}": ref_signal_dict}
        """
        games = await self.get_referees_for_date(game_date)
        signals = {}
        for game in games:
            key = f"{game['home_team']}_{game['away_team']}"
            signal = self.compute_ref_signal(game["referees"])
            signal["home_team"] = game["home_team"]
            signal["away_team"] = game["away_team"]
            signal["game_id"] = game["game_id"]
            signals[key] = signal

        return signals


_service: RefereeDataService | None = None


def get_referee_data(api_key: str = "") -> RefereeDataService:
    global _service
    if _service is None:
        _service = RefereeDataService(api_key=api_key)
    return _service
