"""
Enriched NBA data service.

Pulls advanced stats from SportsDataIO free tier and computes derived features
for the ML prediction pipeline:
  - Player season averages + advanced metrics (usage, PER, TS%)
  - Team pace (possessions/game) and opponent defensive stats
  - Home/away detection for today's games
  - Back-to-back detection (played yesterday)
  - Rest days since last game
"""
from __future__ import annotations

from datetime import date, timedelta

import httpx

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

SDIO_BASE = "https://api.sportsdata.io/v3/nba"


class NBADataService:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.sportsdataio_api_key
        self.client = httpx.Client(timeout=20)

    def _get(self, path: str) -> list | dict:
        url = f"{SDIO_BASE}/{path}"
        sep = "&" if "?" in url else "?"
        r = self.client.get(f"{url}{sep}key={self.api_key}")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Raw data fetchers
    # ------------------------------------------------------------------

    def get_player_season_stats(self, season: int = 2026) -> list[dict]:
        """All player season stats (per-game averages + advanced metrics)."""
        return self._get(f"stats/json/PlayerSeasonStats/{season}")

    def get_team_season_stats(self, season: int = 2026) -> list[dict]:
        """Team season stats including opponent stats (defense)."""
        return self._get(f"stats/json/TeamSeasonStats/{season}")

    def get_standings(self, season: int = 2026) -> list[dict]:
        return self._get(f"scores/json/Standings/{season}")

    def get_games_by_date(self, dt: str) -> list[dict]:
        """Games for a specific date (YYYY-MM-DD)."""
        return self._get(f"scores/json/GamesByDate/{dt}")

    def get_season_games(self, season: int = 2026) -> list[dict]:
        """All games for the season."""
        return self._get(f"scores/json/Games/{season}")

    # ------------------------------------------------------------------
    # Derived feature builders
    # ------------------------------------------------------------------

    def build_team_features(self, season: int = 2026) -> dict[str, dict]:
        """
        Build per-team feature dict keyed by team abbreviation.
        Includes pace, defensive ratings, opponent stats.
        """
        team_stats = self.get_team_season_stats(season)
        features = {}

        for t in team_stats:
            abbr = t.get("Team", "")
            games = t.get("Games", 1) or 1

            # Pace = possessions per game
            possessions = t.get("Possessions", 0) or 0
            pace = round(possessions / games, 1)

            # Team offensive stats per game
            pts_pg = round((t.get("Points", 0) or 0) / games, 1)
            ast_pg = round((t.get("Assists", 0) or 0) / games, 1)
            reb_pg = round((t.get("Rebounds", 0) or 0) / games, 1)
            tov_pg = round((t.get("Turnovers", 0) or 0) / games, 1)
            stl_pg = round((t.get("Steals", 0) or 0) / games, 1)
            blk_pg = round((t.get("BlockedShots", 0) or 0) / games, 1)

            # Opponent (defensive) stats
            opp = t.get("OpponentStat") or {}
            opp_pts_pg = round((opp.get("Points", 0) or 0) / games, 1)
            opp_reb_pg = round((opp.get("Rebounds", 0) or 0) / games, 1)
            opp_ast_pg = round((opp.get("Assists", 0) or 0) / games, 1)
            opp_3pm_pg = round((opp.get("ThreePointersMade", 0) or 0) / games, 1)
            opp_stl_pg = round((opp.get("Steals", 0) or 0) / games, 1)
            opp_blk_pg = round((opp.get("BlockedShots", 0) or 0) / games, 1)
            opp_tov_pg = round((opp.get("Turnovers", 0) or 0) / games, 1)

            wins = t.get("Wins", 0) or 0
            losses = t.get("Losses", 0) or 0
            win_pct = round(wins / max(wins + losses, 1), 3)

            features[abbr] = {
                "pace": pace,
                "pts_pg": pts_pg,
                "ast_pg": ast_pg,
                "reb_pg": reb_pg,
                "tov_pg": tov_pg,
                "stl_pg": stl_pg,
                "blk_pg": blk_pg,
                "win_pct": win_pct,
                # Opponent defense (what they allow)
                "opp_pts_allowed_pg": opp_pts_pg,
                "opp_reb_allowed_pg": opp_reb_pg,
                "opp_ast_allowed_pg": opp_ast_pg,
                "opp_3pm_allowed_pg": opp_3pm_pg,
                "opp_stl_allowed_pg": opp_stl_pg,
                "opp_blk_allowed_pg": opp_blk_pg,
                "opp_tov_forced_pg": opp_tov_pg,
            }

        logger.info(f"Built team features for {len(features)} teams")
        return features

    def build_schedule_context(self) -> dict:
        """
        Build today's game context:
        - Which teams are playing
        - Home/away mapping
        - Back-to-back detection
        - Rest days
        Returns dict keyed by team abbreviation.
        """
        today = date.today()
        yesterday = today - timedelta(days=1)
        two_days_ago = today - timedelta(days=2)

        today_games = self.get_games_by_date(today.isoformat())
        yesterday_games = self.get_games_by_date(yesterday.isoformat())

        # Teams that played yesterday (B2B candidates)
        played_yesterday = set()
        for g in yesterday_games:
            played_yesterday.add(g.get("HomeTeam", ""))
            played_yesterday.add(g.get("AwayTeam", ""))

        # Teams that played 2 days ago (for rest day calc)
        try:
            two_days_games = self.get_games_by_date(two_days_ago.isoformat())
            played_two_days_ago = set()
            for g in two_days_games:
                played_two_days_ago.add(g.get("HomeTeam", ""))
                played_two_days_ago.add(g.get("AwayTeam", ""))
        except Exception:
            played_two_days_ago = set()

        context = {}
        for game in today_games:
            home = game.get("HomeTeam", "")
            away = game.get("AwayTeam", "")
            game_id = str(game.get("GameID", ""))

            for team, is_home, opponent in [(home, True, away), (away, False, home)]:
                is_b2b = team in played_yesterday
                if is_b2b:
                    rest_days = 0
                elif team in played_two_days_ago:
                    rest_days = 1
                else:
                    rest_days = 2  # 2+ days rest

                context[team] = {
                    "game_id": game_id,
                    "is_home": is_home,
                    "opponent": opponent,
                    "is_b2b": is_b2b,
                    "rest_days": rest_days,
                }

        logger.info(f"Schedule context: {len(today_games)} games, {len(context)} teams playing")
        return context

    def build_player_features(self, season: int = 2026) -> dict[str, dict]:
        """
        Build enriched per-player feature dict keyed by player_id (str).
        Includes per-game stats + advanced metrics.
        """
        stats = self.get_player_season_stats(season)
        features = {}

        for p in stats:
            pid = str(p.get("PlayerID", ""))
            games = p.get("Games", 0) or 0
            if games < 3:
                continue  # Skip players with too few games

            minutes = p.get("Minutes", 0) or 0
            mpg = round(minutes / games, 1)
            if mpg < 5:
                continue

            features[pid] = {
                "name": p.get("Name", ""),
                "team": p.get("Team", ""),
                "position": p.get("Position", ""),
                "games_played": games,
                "mpg": mpg,
                "started": p.get("Started", 0) or 0,
                "starter_pct": round((p.get("Started", 0) or 0) / games, 2),
                # Per-game stats
                "pts_pg": round((p.get("Points", 0) or 0) / games, 1),
                "reb_pg": round((p.get("Rebounds", 0) or 0) / games, 1),
                "ast_pg": round((p.get("Assists", 0) or 0) / games, 1),
                "stl_pg": round((p.get("Steals", 0) or 0) / games, 1),
                "blk_pg": round((p.get("BlockedShots", 0) or 0) / games, 1),
                "tov_pg": round((p.get("Turnovers", 0) or 0) / games, 1),
                "three_pm_pg": round((p.get("ThreePointersMade", 0) or 0) / games, 1),
                "fgm_pg": round((p.get("FieldGoalsMade", 0) or 0) / games, 1),
                "fga_pg": round((p.get("FieldGoalsAttempted", 0) or 0) / games, 1),
                "ftm_pg": round((p.get("FreeThrowsMade", 0) or 0) / games, 1),
                "fta_pg": round((p.get("FreeThrowsAttempted", 0) or 0) / games, 1),
                # Advanced metrics
                "usage_rate": p.get("UsageRatePercentage", 0) or 0,
                "per": p.get("PlayerEfficiencyRating", 0) or 0,
                "ts_pct": p.get("TrueShootingPercentage", 0) or 0,
                "ast_pct": p.get("AssistsPercentage", 0) or 0,
                "reb_pct": p.get("TotalReboundsPercentage", 0) or 0,
                "plus_minus": p.get("PlusMinus", 0) or 0,
                # Fantasy (useful as composite stat)
                "fpts_dk": round((p.get("FantasyPointsDraftKings", 0) or 0) / games, 1),
            }

        logger.info(f"Built player features for {len(features)} players")
        return features

    def build_full_feature_set(self) -> dict:
        """
        Build the complete enriched feature set for prediction.
        Returns {
            "players": {player_id: {...features}},
            "teams": {team_abbr: {...features}},
            "schedule": {team_abbr: {...context}},
        }
        """
        players = self.build_player_features()
        teams = self.build_team_features()
        schedule = self.build_schedule_context()

        # Enrich players with team + opponent context
        for pid, pf in players.items():
            team = pf.get("team", "")
            team_feat = teams.get(team, {})
            sched = schedule.get(team, {})

            # Team context
            pf["team_pace"] = team_feat.get("pace", 100)
            pf["team_win_pct"] = team_feat.get("win_pct", 0.5)

            # Schedule context
            pf["is_home"] = sched.get("is_home", None)
            pf["is_b2b"] = sched.get("is_b2b", False)
            pf["rest_days"] = sched.get("rest_days", 2)
            pf["opponent"] = sched.get("opponent", "")
            pf["game_id"] = sched.get("game_id", "")

            # Opponent defense context
            opp_team = pf["opponent"]
            opp_feat = teams.get(opp_team, {})
            pf["opp_pace"] = opp_feat.get("pace", 100)
            pf["opp_pts_allowed"] = opp_feat.get("opp_pts_allowed_pg", 110)
            pf["opp_reb_allowed"] = opp_feat.get("opp_reb_allowed_pg", 44)
            pf["opp_ast_allowed"] = opp_feat.get("opp_ast_allowed_pg", 25)
            pf["opp_3pm_allowed"] = opp_feat.get("opp_3pm_allowed_pg", 12)
            pf["opp_stl_allowed"] = opp_feat.get("opp_stl_allowed_pg", 7)
            pf["opp_blk_allowed"] = opp_feat.get("opp_blk_allowed_pg", 5)

            # Pace factor: combined pace relative to league average (~100)
            league_avg_pace = 100.0
            combined_pace = (pf["team_pace"] + pf["opp_pace"]) / 2
            pf["pace_factor"] = round(combined_pace / league_avg_pace, 3)

        return {
            "players": players,
            "teams": teams,
            "schedule": schedule,
        }


_service: NBADataService | None = None


def get_nba_data() -> NBADataService:
    global _service
    if _service is None:
        _service = NBADataService()
    return _service
