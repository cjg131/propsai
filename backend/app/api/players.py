from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.logging_config import get_logger
from app.schemas.players import (
    PlayerDetail,
    PlayerListResponse,
    PlayerResponse,
    PlayerStats,
    ScoutingReportResponse,
)
from app.services.openai_client import get_openai_client
from app.services.supabase_client import get_supabase

logger = get_logger(__name__)
router = APIRouter()


@router.get("/", response_model=PlayerListResponse)
async def search_players(
    q: str | None = Query(None, description="Search query for player name"),
    team: str | None = Query(None, description="Filter by team abbreviation"),
):
    """Search for NBA players."""
    sb = get_supabase()
    try:
        query = sb.table("players").select("*, teams(abbreviation, name)")
        if q:
            # Use or_ filter with ilike for first/last name matching
            query = query.or_(f"name.ilike.%{q}%")
        if team:
            query = query.eq("team_id", team)
        query = query.eq("is_active", True).limit(50)
        result = query.execute()

        players = []
        for row in result.data or []:
            team_data = row.get("teams") or {}
            players.append(PlayerDetail(
                id=row["id"],
                name=row["name"],
                team=team_data.get("abbreviation", ""),
                team_id=row.get("team_id", ""),
                position=row.get("position", ""),
                jersey_number=row.get("jersey_number"),
                headshot_url=row.get("headshot_url"),
                is_starter=row.get("is_starter", False),
                is_recently_traded=row.get("is_recently_traded", False),
                is_rookie=row.get("is_rookie", False),
            ))
        return PlayerListResponse(players=players, total=len(players))
    except Exception as e:
        logger.error("Error searching players", error=str(e))
        return PlayerListResponse(players=[], total=0)


@router.get("/{player_id}", response_model=PlayerResponse)
async def get_player(player_id: str):
    """Get player details with stats, trends, and matchup data."""
    sb = get_supabase()
    try:
        result = sb.table("players").select("*, teams(abbreviation, name)").eq("id", player_id).single().execute()
        row = result.data
        if not row:
            raise HTTPException(status_code=404, detail="Player not found")

        team_data = row.get("teams") or {}

        # Fetch season stats
        stats_result = (
            sb.table("player_game_stats")
            .select("*")
            .eq("player_id", player_id)
            .order("game_id", desc=True)
            .limit(82)
            .execute()
        )
        stats_rows = stats_result.data or []

        player_stats = None
        if stats_rows:
            import statistics
            player_stats = PlayerStats(
                season_avg_points=statistics.mean([r.get("points", 0) for r in stats_rows]),
                season_avg_rebounds=statistics.mean([r.get("rebounds", 0) for r in stats_rows]),
                season_avg_assists=statistics.mean([r.get("assists", 0) for r in stats_rows]),
                season_avg_threes=statistics.mean([r.get("three_pointers_made", 0) for r in stats_rows]),
                season_avg_steals=statistics.mean([r.get("steals", 0) for r in stats_rows]),
                season_avg_blocks=statistics.mean([r.get("blocks", 0) for r in stats_rows]),
                season_avg_turnovers=statistics.mean([r.get("turnovers", 0) for r in stats_rows]),
                season_avg_minutes=statistics.mean([r.get("minutes", 0) or 0 for r in stats_rows]),
                last5_avg_points=statistics.mean([r.get("points", 0) for r in stats_rows[:5]]) if len(stats_rows) >= 5 else 0,
                last5_avg_rebounds=statistics.mean([r.get("rebounds", 0) for r in stats_rows[:5]]) if len(stats_rows) >= 5 else 0,
                last5_avg_assists=statistics.mean([r.get("assists", 0) for r in stats_rows[:5]]) if len(stats_rows) >= 5 else 0,
                last10_avg_points=statistics.mean([r.get("points", 0) for r in stats_rows[:10]]) if len(stats_rows) >= 10 else 0,
                last10_avg_rebounds=statistics.mean([r.get("rebounds", 0) for r in stats_rows[:10]]) if len(stats_rows) >= 10 else 0,
                last10_avg_assists=statistics.mean([r.get("assists", 0) for r in stats_rows[:10]]) if len(stats_rows) >= 10 else 0,
                usage_rate=statistics.mean([r.get("usage_rate", 0) or 0 for r in stats_rows]),
                games_played=len(stats_rows),
            )

        player = PlayerDetail(
            id=row["id"],
            name=row["name"],
            team=team_data.get("abbreviation", ""),
            team_id=row.get("team_id", ""),
            position=row.get("position", ""),
            jersey_number=row.get("jersey_number"),
            headshot_url=row.get("headshot_url"),
            stats=player_stats,
            is_starter=row.get("is_starter", False),
            is_recently_traded=row.get("is_recently_traded", False),
            is_rookie=row.get("is_rookie", False),
        )
        return PlayerResponse(id=player_id, player=player)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting player", player_id=player_id, error=str(e))
        return PlayerResponse(id=player_id, message=str(e))


@router.get("/{player_id}/scouting-report", response_model=ScoutingReportResponse)
async def get_scouting_report(player_id: str):
    """Get AI-generated scouting report for a player."""
    sb = get_supabase()
    try:
        player_result = sb.table("players").select("*, teams(abbreviation, name)").eq("id", player_id).single().execute()
        player = player_result.data
        if not player:
            raise HTTPException(status_code=404, detail="Player not found")

        team_data = player.get("teams") or {}
        stats_result = (
            sb.table("player_game_stats")
            .select("*")
            .eq("player_id", player_id)
            .order("game_id", desc=True)
            .limit(20)
            .execute()
        )
        stats_rows = stats_result.data or []

        stats_summary = {}
        if stats_rows:
            import statistics
            stats_summary = {
                "Season Avg Points": round(statistics.mean([r.get("points", 0) for r in stats_rows]), 1),
                "Season Avg Rebounds": round(statistics.mean([r.get("rebounds", 0) for r in stats_rows]), 1),
                "Season Avg Assists": round(statistics.mean([r.get("assists", 0) for r in stats_rows]), 1),
                "Season Avg 3PM": round(statistics.mean([r.get("three_pointers_made", 0) for r in stats_rows]), 1),
                "Season Avg Minutes": round(statistics.mean([r.get("minutes", 0) or 0 for r in stats_rows]), 1),
                "Games Analyzed": len(stats_rows),
            }

        # Look up today's opponent from the schedule
        from datetime import date
        today_str = date.today().isoformat()
        opponent = "Unknown"
        team_id = str(player.get("team_id", ""))
        today_games = sb.table("games").select("*, home_team:teams!games_home_team_id_fkey(abbreviation), away_team:teams!games_away_team_id_fkey(abbreviation)").or_(
            f"home_team_id.eq.{team_id},away_team_id.eq.{team_id}"
        ).eq("game_date", today_str).limit(1).execute()
        if today_games.data:
            game = today_games.data[0]
            if str(game.get("home_team_id")) == team_id:
                opponent = (game.get("away_team") or {}).get("abbreviation", "Unknown")
            else:
                opponent = (game.get("home_team") or {}).get("abbreviation", "Unknown")
        else:
            # No game today — find next upcoming game
            upcoming = sb.table("games").select("*, home_team:teams!games_home_team_id_fkey(abbreviation), away_team:teams!games_away_team_id_fkey(abbreviation)").or_(
                f"home_team_id.eq.{team_id},away_team_id.eq.{team_id}"
            ).gte("game_date", today_str).order("game_date").limit(1).execute()
            if upcoming.data:
                game = upcoming.data[0]
                if str(game.get("home_team_id")) == team_id:
                    opponent = (game.get("away_team") or {}).get("abbreviation", "Unknown")
                else:
                    opponent = (game.get("home_team") or {}).get("abbreviation", "Unknown")

        openai = get_openai_client()
        report = await openai.generate_scouting_report(
            player_name=player["name"],
            team=team_data.get("abbreviation", ""),
            opponent=opponent,
            stats_summary=stats_summary,
        )

        return ScoutingReportResponse(
            player_id=player_id,
            report=report,
            model_used="gpt-4o-mini",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error generating scouting report", player_id=player_id, error=str(e))
        return ScoutingReportResponse(
            player_id=player_id,
            report=f"Error generating report: {str(e)}",
        )
