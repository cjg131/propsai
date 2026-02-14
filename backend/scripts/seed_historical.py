from __future__ import annotations
"""
CLI script to seed historical NBA data into Supabase.
Usage: poetry run python scripts/seed_historical.py --seasons 10
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.logging_config import setup_logging, get_logger
from app.services.sportsdataio import SportsDataIOClient
from app.services.supabase_client import get_supabase

logger = get_logger(__name__)

CURRENT_SEASON_YEAR = 2026
SEASON_FORMAT = lambda year: f"{year}"  # SportsDataIO uses year format


def get_season_list(num_seasons: int) -> list[str]:
    """Generate list of season identifiers going back num_seasons."""
    seasons = []
    for i in range(num_seasons):
        year = CURRENT_SEASON_YEAR - i
        seasons.append(str(year))
    return seasons


async def seed_teams(client: SportsDataIOClient, supabase) -> int:
    """Seed all NBA teams."""
    logger.info("Seeding teams...")
    teams = await client.get_teams()
    count = 0
    for team in teams:
        try:
            supabase.table("teams").upsert({
                "id": str(team.get("TeamID", "")),
                "name": team.get("Name", ""),
                "abbreviation": team.get("Key", ""),
                "city": team.get("City", ""),
                "conference": team.get("Conference", ""),
                "division": team.get("Division", ""),
                "logo_url": team.get("WikipediaLogoUrl", ""),
                "arena_name": team.get("StadiumID", ""),
            }).execute()
            count += 1
        except Exception as e:
            logger.error("Error seeding team", team=team.get("Name"), error=str(e))
    logger.info(f"Seeded {count} teams")
    return count


async def seed_players(client: SportsDataIOClient, supabase) -> int:
    """Seed all NBA players."""
    logger.info("Seeding players...")
    players = await client.get_players()
    count = 0
    for player in players:
        try:
            supabase.table("players").upsert({
                "id": str(player.get("PlayerID", "")),
                "name": f"{player.get('FirstName', '')} {player.get('LastName', '')}".strip(),
                "team_id": str(player.get("TeamID", "")),
                "position": player.get("Position", ""),
                "jersey_number": str(player.get("Jersey", "")),
                "height": f"{player.get('Height', '')}",
                "weight": player.get("Weight"),
                "birth_date": player.get("BirthDate"),
                "college": player.get("College", ""),
                "draft_year": player.get("DraftYear"),
                "draft_round": player.get("DraftRound"),
                "draft_pick": player.get("DraftNumber"),
                "headshot_url": player.get("PhotoUrl", ""),
                "is_active": player.get("Status", "") == "Active",
            }).execute()
            count += 1
        except Exception as e:
            logger.error("Error seeding player", player=player.get("LastName"), error=str(e))
    logger.info(f"Seeded {count} players")
    return count


async def seed_games_for_season(
    client: SportsDataIOClient, supabase, season: str
) -> int:
    """Seed all games for a given season."""
    logger.info(f"Seeding games for season {season}...")
    games = await client.get_games_by_season(season)
    count = 0
    for game in games:
        try:
            supabase.table("games").upsert({
                "id": str(game.get("GameID", "")),
                "season_id": season,
                "game_date": game.get("Day", ""),
                "home_team_id": str(game.get("HomeTeamID", "")),
                "away_team_id": str(game.get("AwayTeamID", "")),
                "home_score": game.get("HomeTeamScore"),
                "away_score": game.get("AwayTeamScore"),
                "status": game.get("Status", "scheduled"),
                "is_playoff": game.get("IsPlayoffs", False),
                "over_under": game.get("OverUnder"),
                "spread": game.get("PointSpread"),
            }).execute()
            count += 1
        except Exception as e:
            logger.error("Error seeding game", game_id=game.get("GameID"), error=str(e))
    logger.info(f"Seeded {count} games for season {season}")
    return count


async def seed_player_stats_for_season(
    client: SportsDataIOClient, supabase, season: str
) -> int:
    """Seed player game stats for a given season."""
    logger.info(f"Seeding player stats for season {season}...")
    stats = await client.get_player_season_stats(season)
    count = 0
    for stat in stats:
        try:
            supabase.table("player_game_stats").upsert({
                "player_id": str(stat.get("PlayerID", "")),
                "game_id": str(stat.get("GameID", "")),
                "team_id": str(stat.get("TeamID", "")),
                "minutes": stat.get("Minutes"),
                "points": stat.get("Points", 0),
                "rebounds": stat.get("Rebounds", 0),
                "assists": stat.get("Assists", 0),
                "steals": stat.get("Steals", 0),
                "blocks": stat.get("BlockedShots", 0),
                "turnovers": stat.get("Turnovers", 0),
                "three_pointers_made": stat.get("ThreePointersMade", 0),
                "three_pointers_attempted": stat.get("ThreePointersAttempted", 0),
                "field_goals_made": stat.get("FieldGoalsMade", 0),
                "field_goals_attempted": stat.get("FieldGoalsAttempted", 0),
                "free_throws_made": stat.get("FreeThrowsMade", 0),
                "free_throws_attempted": stat.get("FreeThrowsAttempted", 0),
                "offensive_rebounds": stat.get("OffensiveRebounds", 0),
                "defensive_rebounds": stat.get("DefensiveRebounds", 0),
                "personal_fouls": stat.get("PersonalFouls", 0),
                "plus_minus": stat.get("PlusMinus", 0),
                "is_starter": stat.get("Started", 0) == 1,
            }).execute()
            count += 1
        except Exception as e:
            logger.error("Error seeding stat", error=str(e))
    logger.info(f"Seeded {count} player stats for season {season}")
    return count


async def seed_season(client: SportsDataIOClient, supabase, season: str) -> None:
    """Seed a season record."""
    try:
        supabase.table("seasons").upsert({
            "id": season,
            "start_date": f"{int(season) - 1}-10-01",
            "end_date": f"{season}-06-30",
            "is_current": season == str(CURRENT_SEASON_YEAR),
            "data_loaded": True,
        }).execute()
    except Exception as e:
        logger.error("Error seeding season", season=season, error=str(e))


async def main(num_seasons: int):
    setup_logging(debug=True)
    logger.info(f"Starting historical data seed for {num_seasons} seasons")

    settings = get_settings()
    supabase = get_supabase()
    client = SportsDataIOClient()

    try:
        # Seed teams and players first
        await seed_teams(client, supabase)
        await seed_players(client, supabase)

        # Seed each season
        seasons = get_season_list(num_seasons)
        for season in seasons:
            logger.info(f"Processing season {season}...")
            await seed_season(client, supabase, season)
            await seed_games_for_season(client, supabase, season)
            await seed_player_stats_for_season(client, supabase, season)
            logger.info(f"Completed season {season}")

        logger.info("Historical data seed completed successfully!")
    except Exception as e:
        logger.error("Seed failed", error=str(e))
        raise
    finally:
        await client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed historical NBA data")
    parser.add_argument(
        "--seasons", type=int, default=10, help="Number of seasons to load (default: 10)"
    )
    args = parser.parse_args()
    asyncio.run(main(args.seasons))
