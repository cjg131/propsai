from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.logging_config import get_logger
from app.schemas.bets import BetCreate, BetDetail, BetListResponse, BetResponse, BetSummaryResponse
from app.services.supabase_client import get_supabase

logger = get_logger(__name__)
router = APIRouter()


@router.get("/summary", response_model=BetSummaryResponse)
async def get_bet_summary():
    """Get betting performance summary (record, ROI, P&L)."""
    sb = get_supabase()
    try:
        result = sb.table("bets").select("*").execute()
        bets = result.data or []

        wins = [b for b in bets if b.get("status") == "won"]
        losses = [b for b in bets if b.get("status") == "lost"]
        pending = [b for b in bets if b.get("status") == "pending"]
        pushes = [b for b in bets if b.get("status") == "push"]

        total = len(bets)
        resolved = len(wins) + len(losses)
        win_rate = len(wins) / resolved if resolved > 0 else 0.0
        total_wagered = sum(b.get("stake", 0) for b in bets if b.get("status") != "pending")
        total_profit = sum(b.get("profit", 0) or 0 for b in bets if b.get("profit") is not None)
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0

        # Streak calculation
        resolved_bets = sorted(
            [b for b in bets if b.get("status") in ("won", "lost")],
            key=lambda b: b.get("created_at", ""),
            reverse=True,
        )
        streak = 0
        streak_type = "none"
        if resolved_bets:
            streak_type = resolved_bets[0].get("status", "none")
            for b in resolved_bets:
                if b.get("status") == streak_type:
                    streak += 1
                else:
                    break

        # Best/worst prop type
        from collections import Counter
        prop_wins = Counter(b.get("prop_type") for b in wins)
        prop_losses = Counter(b.get("prop_type") for b in losses)
        best_prop = prop_wins.most_common(1)[0][0] if prop_wins else None
        worst_prop = prop_losses.most_common(1)[0][0] if prop_losses else None

        return BetSummaryResponse(
            total_bets=total,
            wins=len(wins),
            losses=len(losses),
            pending=len(pending),
            pushes=len(pushes),
            win_rate=round(win_rate, 4),
            roi=round(roi, 2),
            total_wagered=round(total_wagered, 2),
            total_profit=round(total_profit, 2),
            current_streak=streak,
            streak_type=streak_type,
            best_prop_type=best_prop,
            worst_prop_type=worst_prop,
        )
    except Exception as e:
        logger.error("Error getting bet summary", error=str(e))
        return BetSummaryResponse()


@router.get("/", response_model=BetListResponse)
async def get_bets(
    status: str | None = Query(None, description="Filter by status: pending, won, lost"),
    prop_type: str | None = Query(None, description="Filter by prop type"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Get bet history with filters."""
    sb = get_supabase()
    try:
        query = sb.table("bets").select("*", count="exact")
        if status:
            query = query.eq("status", status)
        if prop_type:
            query = query.eq("prop_type", prop_type)
        query = query.order("created_at", desc=True).range(offset, offset + limit - 1)
        result = query.execute()

        bets = [BetDetail(**row) for row in (result.data or [])]
        return BetListResponse(bets=bets, total=result.count or len(bets))
    except Exception as e:
        logger.error("Error getting bets", error=str(e))
        return BetListResponse(bets=[], total=0)


@router.post("/", response_model=BetResponse)
async def create_bet(bet: BetCreate):
    """Log a new bet."""
    sb = get_supabase()
    try:
        row = bet.dict()
        result = sb.table("bets").insert(row).execute()
        if result.data:
            return BetResponse(id=result.data[0]["id"], bet=BetDetail(**result.data[0]))
        raise HTTPException(status_code=500, detail="Failed to create bet")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error creating bet", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/paper-trading/status")
async def paper_trading_status():
    """Get paper trading session status and summary."""
    from app.services.paper_trading import get_paper_trading_status
    return await get_paper_trading_status()


@router.post("/paper-trading/place")
async def paper_trading_place(
    min_confidence: float = Query(60.0),
    target_date: str | None = Query(None, description="Date to place bets for (YYYY-MM-DD). Defaults to today."),
):
    """Auto-place virtual bets on high-confidence predictions for a given date."""
    from datetime import date as d

    from app.services.paper_trading import place_paper_trades
    td = None
    if target_date:
        try:
            td = d.fromisoformat(target_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    return await place_paper_trades(min_confidence=min_confidence, target_date=td)


@router.post("/paper-trading/resolve")
async def paper_trading_resolve(target_date: str | None = Query(None)):
    """Resolve pending paper trades using actual game results."""
    from datetime import date as d

    from app.services.paper_trading import resolve_paper_trades
    td = None
    if target_date:
        try:
            td = d.fromisoformat(target_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    return await resolve_paper_trades(target_date=td)


@router.post("/paper-trading/reset")
async def paper_trading_reset():
    """Reset all paper trades and start fresh."""
    from app.services.paper_trading import reset_paper_trading
    return await reset_paper_trading()


@router.put("/{bet_id}/resolve")
async def resolve_bet(bet_id: str, result: str, actual_value: float | None = None):
    """Mark a bet as won, lost, or push."""
    sb = get_supabase()
    try:
        bet_result = sb.table("bets").select("*").eq("id", bet_id).single().execute()
        bet = bet_result.data
        if not bet:
            raise HTTPException(status_code=404, detail="Bet not found")

        profit = 0.0
        if result == "won":
            from app.utils.kelly import american_to_decimal
            decimal_odds = american_to_decimal(bet["odds"])
            profit = bet["stake"] * (decimal_odds - 1)
        elif result == "lost":
            profit = -bet["stake"]

        from datetime import datetime
        update = {
            "status": result,
            "profit": round(profit, 2),
            "resolved_at": datetime.utcnow().isoformat(),
        }
        if actual_value is not None:
            update["actual_value"] = actual_value

        sb.table("bets").update(update).eq("id", bet_id).execute()
        return {"id": bet_id, "status": result, "profit": round(profit, 2)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error resolving bet", bet_id=bet_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
