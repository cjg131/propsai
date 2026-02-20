"""
NBA Analysis Cache — shared enriched player feature set.

Built once per day (or on-demand), consumed by both:
  1. Sportsbook props tab (/api/predictions/generate)
  2. Kalshi agent (run_nba_props_cycle)

This avoids re-fetching BDL game logs every 5-minute Kalshi cycle
and ensures both pipelines use identical player features.
"""
from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

# ── Module-level cache ────────────────────────────────────────────────────────
_cache_date: str = ""
_cache_data: dict[str, Any] = {}
_build_lock = asyncio.Lock()


def get_cache_age_minutes() -> float:
    """Return how many minutes ago the cache was last built. 999 if never."""
    if not _cache_data:
        return 999.0
    built_at = _cache_data.get("built_at")
    if not built_at:
        return 999.0
    delta = datetime.utcnow() - built_at
    return delta.total_seconds() / 60


def is_cache_fresh(max_age_minutes: float = 60.0) -> bool:
    """True if cache was built today and is less than max_age_minutes old."""
    today = date.today().isoformat()
    return _cache_date == today and get_cache_age_minutes() < max_age_minutes


def get_cached_players() -> dict[str, dict]:
    """Return the cached name→player feature dict. Empty dict if stale."""
    return _cache_data.get("name_to_player", {})


def get_cached_schedule() -> dict[str, dict]:
    """Return the cached team→schedule dict."""
    return _cache_data.get("schedule", {})


async def build_nba_analysis(force: bool = False) -> dict[str, Any]:
    """
    Build (or return cached) enriched NBA player features.

    Returns:
        {
            "name_to_player": {player_name_lower: {features...}},
            "schedule": {team_abbr: {is_home, opponent, ...}},
            "built_at": datetime,
            "player_count": int,
        }

    Both the sportsbook predictions pipeline and the Kalshi agent
    call this function. The result is cached for 60 minutes.
    """
    global _cache_date, _cache_data

    if not force and is_cache_fresh():
        logger.info(
            f"NBA analysis cache hit: {len(_cache_data.get('name_to_player', {}))} players, "
            f"age={get_cache_age_minutes():.0f}min"
        )
        return _cache_data

    async with _build_lock:
        # Double-check after acquiring lock
        if not force and is_cache_fresh():
            return _cache_data

        logger.info("NBA analysis cache: building enriched player features...")
        loop = asyncio.get_event_loop()

        name_to_player: dict[str, dict] = {}
        schedule: dict[str, dict] = {}

        # ── Step 1: SportsDataIO player + team features ───────────────────────
        try:
            from app.services.nba_data import get_nba_data
            nba = get_nba_data()
            feature_data = await loop.run_in_executor(None, nba.build_full_feature_set)
            sdio_players = feature_data.get("players", {})
            schedule = feature_data.get("schedule", {})
            for pid, pf in sdio_players.items():
                name = (pf.get("name") or "").strip().lower()
                if name:
                    name_to_player[name] = pf
            logger.info(f"NBA analysis: SportsDataIO gave {len(name_to_player)} players")
        except Exception as e:
            logger.warning(f"NBA analysis: SportsDataIO failed ({e}), continuing with BDL-only")

        # ── Step 2: BDL active players + game log enrichment ─────────────────
        try:
            from app.services.balldontlie import get_balldontlie
            bdl = get_balldontlie()

            active_players = await loop.run_in_executor(None, bdl.get_active_players)
            bdl_name_map: dict[str, dict] = {}
            for ap in active_players:
                fname = (ap.get("first_name") or "").strip()
                lname = (ap.get("last_name") or "").strip()
                full = f"{fname} {lname}".lower().strip()
                if full:
                    bdl_name_map[full] = ap

            # Seed any players missing from SportsDataIO (e.g. when SDIO 403s)
            for full_name, ap in bdl_name_map.items():
                if full_name not in name_to_player:
                    name_to_player[full_name] = {
                        "name": full_name,
                        "team": (ap.get("team") or {}).get("abbreviation", ""),
                        "position": ap.get("position", ""),
                    }

            # Fetch game logs for all known players (bulk)
            bdl_ids_needed = []
            name_to_bdl_id: dict[str, int] = {}
            for pname in name_to_player:
                bdl_p = bdl_name_map.get(pname)
                if not bdl_p:
                    parts = pname.split()
                    if parts:
                        last = parts[-1]
                        for bn, bp in bdl_name_map.items():
                            if bn.endswith(last) and len(last) >= 4:
                                bdl_p = bp
                                break
                if bdl_p:
                    bid = bdl_p.get("id")
                    if bid:
                        bdl_ids_needed.append(bid)
                        name_to_bdl_id[pname] = bid

            if bdl_ids_needed:
                bdl_ids_needed = bdl_ids_needed[:600]  # cap — covers all ~523 active players
                bulk_logs = await loop.run_in_executor(
                    None, bdl.get_bulk_game_logs, bdl_ids_needed
                )
                today = date.today()
                for pname, pf in name_to_player.items():
                    bid = name_to_bdl_id.get(pname)
                    if not bid or bid not in bulk_logs:
                        continue
                    logs = bulk_logs[bid]
                    if not logs:
                        continue
                    gl_feats = bdl.build_player_game_log_features(logs)
                    pf.update(gl_feats)
                    rest_feats = bdl.build_rest_features_from_logs(logs, today)
                    pf["rest_days"] = rest_feats.get("days_rest", pf.get("rest_days", 2))
                    pf["is_b2b"] = bool(rest_feats.get("is_b2b", 0))
                    pf["games_last_7"] = rest_feats.get("games_last_7", pf.get("games_last_7", 3))
                    pf["is_3_in_4"] = bool(rest_feats.get("is_3_in_4", 0))
                    opp_team = pf.get("opponent", "")
                    if opp_team:
                        for log in logs[:5]:
                            game = log.get("game", {})
                            team_id = log.get("team", {}).get("id")
                            home_id = game.get("home_team_id")
                            vis_id = game.get("visitor_team_id")
                            opp_id = vis_id if team_id == home_id else home_id
                            if opp_id:
                                mu_feats = bdl.build_matchup_features(logs, opp_id)
                                pf.update(mu_feats)
                                break

                logger.info(
                    f"NBA analysis: BDL enriched {len(bulk_logs)}/{len(bdl_ids_needed)} players"
                )

            # Map BDL season_avg_* → pts_pg/reb_pg/ast_pg etc.
            BDL_TO_PG = {
                "season_avg_pts": "pts_pg",
                "season_avg_reb": "reb_pg",
                "season_avg_ast": "ast_pg",
                "season_avg_fg3m": "three_pm_pg",
                "season_avg_stl": "stl_pg",
                "season_avg_blk": "blk_pg",
                "season_avg_tov": "tov_pg",
                "season_avg_min": "mpg",
            }
            for pf in name_to_player.values():
                for bdl_key, pg_key in BDL_TO_PG.items():
                    if bdl_key in pf and pf.get(bdl_key) is not None:
                        if not pf.get(pg_key):
                            pf[pg_key] = pf[bdl_key]

        except Exception as e:
            logger.warning(f"NBA analysis: BDL enrichment failed ({e})")

        # ── Step 3: Odds API game lines (spread, over_under) ─────────────────
        try:
            from app.services.odds_api import get_odds_api
            odds_client = get_odds_api()
            game_lines_dict = await odds_client.get_game_lines()
            game_lines_map: dict[str, dict] = {}
            for gl in game_lines_dict.values():
                home = (gl.get("home_team") or "").upper()
                away = (gl.get("away_team") or "").upper()
                spread_home = gl.get("spread_home", 0.0) or 0.0
                spread_away = gl.get("spread_away", 0.0) or 0.0
                total = gl.get("total", 220.0) or 220.0
                if home:
                    game_lines_map[home] = {"spread": spread_home, "over_under": total}
                if away:
                    game_lines_map[away] = {"spread": spread_away, "over_under": total}
            for pf in name_to_player.values():
                team = (pf.get("team") or "").upper()
                gl_entry = game_lines_map.get(team, {})
                if gl_entry:
                    pf["spread"] = gl_entry["spread"]
                    pf["over_under"] = gl_entry["over_under"]
            logger.info(f"NBA analysis: game lines for {len(game_lines_map)} teams")
        except Exception as e:
            logger.debug(f"NBA analysis: game lines failed ({e})")

        # ── Step 4: News sentiment ────────────────────────────────────────────
        try:
            from app.services.news_sentiment import get_news_sentiment
            ns_svc = get_news_sentiment()
            known_players = set(name_to_player.keys())
            player_sentiment = await loop.run_in_executor(
                None, ns_svc.build_player_sentiment, known_players, 48
            )
            for pname, sent in player_sentiment.items():
                pf = name_to_player.get(pname)
                if pf:
                    pf["news_sentiment"] = sent.get("news_sentiment", 0.0)
                    pf["injury_mentioned"] = sent.get("injury_mentioned", 0)
                    pf["rest_mentioned"] = sent.get("rest_mentioned", 0)
                    pf["hot_streak_mentioned"] = sent.get("hot_streak_mentioned", 0)
            mentioned = sum(1 for v in player_sentiment.values() if v.get("news_volume", 0) > 0)
            logger.info(f"NBA analysis: news sentiment for {mentioned}/{len(known_players)} players")
        except Exception as e:
            logger.debug(f"NBA analysis: news sentiment failed ({e})")

        result = {
            "name_to_player": name_to_player,
            "schedule": schedule,
            "built_at": datetime.utcnow(),
            "player_count": len(name_to_player),
        }

        _cache_date = date.today().isoformat()
        _cache_data = result

        logger.info(
            f"NBA analysis cache built: {len(name_to_player)} players, "
            f"schedule={len(schedule)} teams"
        )
        return result


_singleton = None


def get_nba_analysis_cache() -> "NBAAnalysisCacheService":
    global _singleton
    if _singleton is None:
        _singleton = NBAAnalysisCacheService()
    return _singleton


class NBAAnalysisCacheService:
    """Thin wrapper so callers can use dependency injection style."""

    async def get_players(self, force: bool = False) -> dict[str, dict]:
        data = await build_nba_analysis(force=force)
        return data.get("name_to_player", {})

    async def get_full(self, force: bool = False) -> dict[str, Any]:
        return await build_nba_analysis(force=force)
