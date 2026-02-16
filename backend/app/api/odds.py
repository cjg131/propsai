from __future__ import annotations

from fastapi import APIRouter, Query

from app.logging_config import get_logger
from app.schemas.odds import BookOdds, OddsComparison, OddsComparisonResponse
from app.services.supabase_client import get_supabase

logger = get_logger(__name__)
router = APIRouter()


@router.get("/compare", response_model=OddsComparisonResponse)
async def compare_odds(
    player_id: str | None = Query(None, description="Filter by player ID"),
    prop_type: str | None = Query(None, description="Filter by prop type"),
    game_id: str | None = Query(None, description="Filter by game ID"),
):
    """Compare prop odds across all available sportsbooks."""
    sb = get_supabase()
    try:
        query = sb.table("prop_lines").select(
            "*, players(name, team_id)"
        )
        if player_id:
            query = query.eq("player_id", player_id)
        if prop_type:
            query = query.eq("prop_type", prop_type)
        if game_id:
            query = query.eq("game_id", game_id)
        query = query.order("fetched_at", desc=True).limit(500)
        result = query.execute()

        # Group by player_id + prop_type + game_id
        from collections import defaultdict
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in result.data or []:
            key = f"{row.get('player_id')}|{row.get('prop_type')}|{row.get('game_id')}"
            grouped[key].append(row)

        comparisons = []
        for key, rows in grouped.items():
            first = rows[0]
            player_data = first.get("players") or {}

            books = []
            best_over_odds = -99999
            best_under_odds = -99999
            best_over_book = None
            best_under_book = None

            for r in rows:
                book = BookOdds(
                    sportsbook=r.get("sportsbook", ""),
                    line=r.get("line", 0),
                    over_odds=r.get("over_odds", 0),
                    under_odds=r.get("under_odds", 0),
                    last_updated=r.get("fetched_at"),
                )
                books.append(book)
                if r.get("over_odds", 0) > best_over_odds:
                    best_over_odds = r.get("over_odds", 0)
                    best_over_book = r.get("sportsbook")
                if r.get("under_odds", 0) > best_under_odds:
                    best_under_odds = r.get("under_odds", 0)
                    best_under_book = r.get("sportsbook")

            opening = [r for r in rows if r.get("is_opening")]
            comparisons.append(OddsComparison(
                player_id=first.get("player_id", ""),
                player_name=player_data.get("name", "Unknown"),
                team=str(player_data.get("team_id", "")),
                opponent="",
                game_id=first.get("game_id", ""),
                prop_type=first.get("prop_type", ""),
                books=books,
                best_over_book=best_over_book,
                best_over_odds=best_over_odds if best_over_odds > -99999 else None,
                best_under_book=best_under_book,
                best_under_odds=best_under_odds if best_under_odds > -99999 else None,
                opening_line=opening[0].get("line") if opening else None,
                current_consensus_line=rows[0].get("line"),
            ))

        return OddsComparisonResponse(comparisons=comparisons, total=len(comparisons))
    except Exception as e:
        logger.error("Error comparing odds", error=str(e))
        return OddsComparisonResponse(comparisons=[], total=0)


@router.post("/refresh")
async def refresh_odds():
    """Refresh real-time player prop odds from The Odds API."""
    from app.services.odds_api import get_odds_api
    sb = get_supabase()

    try:
        odds_api = get_odds_api()
        all_props = await odds_api.get_all_todays_props()

        if not all_props:
            return {"status": "completed", "props_loaded": 0, "message": "No props available (no games today or markets not open yet)"}

        # Match player names to our player IDs
        players_result = sb.table("players").select("id, name").execute()
        # Build name lookup (lowercase for fuzzy matching)
        name_to_id: dict[str, str] = {}
        for p in (players_result.data or []):
            name_to_id[p["name"].lower()] = p["id"]

        count = 0
        batch = []
        for prop in all_props:
            player_name = prop.get("player", "")
            player_id = name_to_id.get(player_name.lower())
            if not player_id:
                continue

            batch.append({
                "player_id": player_id,
                "game_id": prop.get("event_id", ""),
                "prop_type": prop.get("prop_type", ""),
                "sportsbook": prop.get("book", ""),
                "line": prop.get("line", 0),
                "over_odds": prop.get("over_odds", 0),
                "under_odds": prop.get("under_odds", 0),
            })

        # Batch insert
        for i in range(0, len(batch), 200):
            chunk = batch[i:i + 200]
            try:
                sb.table("prop_lines").insert(chunk).execute()
                count += len(chunk)
            except Exception as e:
                logger.warning(f"Batch insert error: {e}")

        logger.info(f"Loaded {count} real prop lines from The Odds API")
        return {"status": "completed", "props_loaded": count, "total_fetched": len(all_props)}
    except Exception as e:
        logger.error("Error refreshing odds", error=str(e))
        return {"status": "failed", "error": str(e)}
