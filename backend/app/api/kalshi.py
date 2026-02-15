from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from app.services.kalshi_api import get_kalshi_client
from app.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

HISTORY_DIR = Path(__file__).parent.parent / "cache" / "kalshi_history"


@router.get("/markets")
async def get_kalshi_markets():
    """
    Fetch all open NBA player prop markets from Kalshi.
    Returns parsed market data with player name, prop type, line, prices.
    """
    try:
        client = get_kalshi_client()
        markets = await client.get_nba_player_prop_markets()
        return {
            "markets": markets,
            "total": len(markets),
        }
    except Exception as e:
        logger.error("Failed to fetch Kalshi markets", error=str(e))
        raise HTTPException(status_code=502, detail=f"Kalshi API error: {str(e)}")


@router.get("/markets/raw")
async def get_kalshi_markets_raw(
    status: str = "open",
    series_ticker: str | None = None,
    limit: int = 100,
    cursor: str | None = None,
):
    """
    Fetch raw markets from Kalshi API (for debugging/exploration).
    """
    try:
        client = get_kalshi_client()
        data = await client.get_markets(
            status=status,
            series_ticker=series_ticker,
            limit=limit,
            cursor=cursor,
        )
        return data
    except Exception as e:
        logger.error("Failed to fetch raw Kalshi markets", error=str(e))
        raise HTTPException(status_code=502, detail=f"Kalshi API error: {str(e)}")


@router.get("/edges")
async def get_kalshi_edges():
    """
    Find NBA player prop markets where our model disagrees with Kalshi's implied probability.
    Compares L10 average vs line to find mispriced contracts.
    """
    try:
        client = get_kalshi_client()
        markets = await client.get_nba_player_prop_markets()

        edges = []
        for market in markets:
            line = market.get("line")
            prop_type = market.get("prop_type")
            player_name = market.get("player_name")
            implied_over = market.get("implied_prob_over", 0)
            implied_under = market.get("implied_prob_under", 0)

            if not line or not prop_type or not player_name:
                continue

            # TODO: Look up player's L10 average from BDL cache/DB
            # For now, flag markets with extreme implied probabilities
            # (very cheap YES or NO contracts = potential edge)
            edge_info = {
                **market,
                "edge_type": None,
                "edge_pct": 0,
                "our_prob_over": None,
                "our_prob_under": None,
                "recommendation": None,
            }

            # Flag markets where implied prob is very low (cheap contracts)
            if implied_over > 0 and implied_over <= 0.30:
                edge_info["edge_type"] = "cheap_yes"
                edge_info["recommendation"] = "Potential YES value"
            elif implied_under > 0 and implied_under <= 0.30:
                edge_info["edge_type"] = "cheap_no"
                edge_info["recommendation"] = "Potential NO value"

            if edge_info["edge_type"]:
                edges.append(edge_info)

        # Sort by volume (more liquid = more tradeable)
        edges.sort(key=lambda x: x.get("volume", 0), reverse=True)

        return {
            "edges": edges,
            "total": len(edges),
            "total_markets_scanned": len(markets),
        }
    except Exception as e:
        logger.error("Failed to compute Kalshi edges", error=str(e))
        raise HTTPException(status_code=502, detail=f"Kalshi edge computation error: {str(e)}")


@router.get("/balance")
async def get_kalshi_balance():
    """Get Kalshi account balance (requires API key auth)."""
    try:
        client = get_kalshi_client()
        if not client.private_key:
            raise HTTPException(status_code=401, detail="Kalshi API key not configured")
        balance = await client.get_balance()
        return balance
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch Kalshi balance", error=str(e))
        raise HTTPException(status_code=502, detail=f"Kalshi API error: {str(e)}")


@router.get("/positions")
async def get_kalshi_positions():
    """Get current Kalshi positions (requires API key auth)."""
    try:
        client = get_kalshi_client()
        if not client.private_key:
            raise HTTPException(status_code=401, detail="Kalshi API key not configured")
        positions = await client.get_positions()
        return positions
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch Kalshi positions", error=str(e))
        raise HTTPException(status_code=502, detail=f"Kalshi API error: {str(e)}")


@router.get("/history")
async def get_kalshi_history(
    prop_type: str | None = Query(None, description="Filter by prop type"),
    player: str | None = Query(None, description="Filter by player name (case-insensitive)"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """
    Serve cached settled Kalshi NBA player prop markets.
    Supports filtering by prop type and player name, with pagination.
    """
    try:
        all_markets: list[dict] = []
        for path in sorted(HISTORY_DIR.glob("*_parsed.json")):
            data = json.loads(path.read_text())
            all_markets.extend(data)

        if not all_markets:
            return {"markets": [], "total": 0, "filtered": 0}

        # Apply filters
        filtered = all_markets
        if prop_type:
            filtered = [m for m in filtered if m.get("prop_type") == prop_type]
        if player:
            player_lower = player.lower()
            filtered = [
                m for m in filtered
                if player_lower in (m.get("player_name") or "").lower()
            ]

        # Sort by close_time descending (most recent first)
        filtered.sort(key=lambda m: m.get("close_time", ""), reverse=True)

        total_filtered = len(filtered)
        page = filtered[offset: offset + limit]

        # Compute summary stats
        results = [m.get("result") for m in all_markets if m.get("result") in ("yes", "no")]
        unique_players = len(set(m.get("player_name", "") for m in all_markets if m.get("player_name")))
        dates = sorted(set(m.get("close_time", "")[:10] for m in all_markets if m.get("close_time")))
        prop_counts = {}
        for m in all_markets:
            pt = m.get("prop_type", "unknown")
            prop_counts[pt] = prop_counts.get(pt, 0) + 1

        return {
            "markets": page,
            "total": len(all_markets),
            "filtered": total_filtered,
            "offset": offset,
            "limit": limit,
            "summary": {
                "total_markets": len(all_markets),
                "unique_players": unique_players,
                "date_range": {
                    "start": dates[0] if dates else None,
                    "end": dates[-1] if dates else None,
                },
                "game_days": len(dates),
                "by_prop_type": prop_counts,
                "yes_results": results.count("yes"),
                "no_results": results.count("no"),
            },
        }
    except Exception as e:
        logger.error("Failed to load Kalshi history", error=str(e))
        raise HTTPException(status_code=500, detail=f"History load error: {str(e)}")


@router.get("/history/backtest")
async def get_kalshi_backtest():
    """
    Serve the latest backtest results (generated by backtest_kalshi.py).
    """
    results_path = HISTORY_DIR / "backtest_results.json"
    if not results_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No backtest results found. Run backtest_kalshi.py first.",
        )
    try:
        data = json.loads(results_path.read_text())
        return data
    except Exception as e:
        logger.error("Failed to load backtest results", error=str(e))
        raise HTTPException(status_code=500, detail=f"Backtest load error: {str(e)}")
