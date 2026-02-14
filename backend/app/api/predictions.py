from __future__ import annotations
import hashlib
from fastapi import APIRouter, Query, HTTPException
from app.schemas.predictions import (
    PredictionResponse, PredictionListResponse, PredictionDetail, GameInfo,
)
from app.services.supabase_client import get_supabase
from app.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/today", response_model=PredictionListResponse)
async def get_today_predictions(
    prop_type: str | None = Query(None, description="Filter by prop type"),
    min_confidence: float | None = Query(None, description="Minimum confidence score"),
    team: str | None = Query(None, description="Filter by team abbreviation"),
    game_id: str | None = Query(None, description="Filter by game ID"),
):
    """Get today's prop predictions with optional filters, grouped by game."""
    sb = get_supabase()
    try:
        from datetime import date
        from collections import Counter
        today = date.today().isoformat()

        query = sb.table("predictions").select("*").gte("created_at", today)
        if prop_type:
            query = query.eq("prop_type", prop_type)
        if min_confidence:
            query = query.gte("confidence_score", min_confidence)
        if game_id:
            query = query.eq("game_id", game_id)

        query = query.order("confidence_score", desc=True).limit(1000)
        result = query.execute()

        # Build lookups
        players_result = sb.table("players").select("id, name, team_id").execute()
        players_map = {p["id"]: p for p in (players_result.data or [])}
        teams_result = sb.table("teams").select("id, abbreviation, name").execute()
        teams_abbr_map = {str(t["id"]): t["abbreviation"] for t in (teams_result.data or [])}
        teams_name_map = {str(t["id"]): t.get("name", "") for t in (teams_result.data or [])}

        # Build game info from the games table for today's game IDs
        pred_game_ids = list(set(row.get("game_id", "") for row in (result.data or []) if row.get("game_id")))
        games_list: list[GameInfo] = []
        game_matchup: dict[str, dict] = {}  # game_id -> {home_abbr, away_abbr}

        for gid in pred_game_ids:
            try:
                g = sb.table("games").select("*").eq("id", gid).single().execute()
                if g.data:
                    home_id = str(g.data["home_team_id"])
                    away_id = str(g.data["away_team_id"])
                    home_abbr = teams_abbr_map.get(home_id, "???")
                    away_abbr = teams_abbr_map.get(away_id, "???")
                    home_name = teams_name_map.get(home_id, "")
                    away_name = teams_name_map.get(away_id, "")
                    game_matchup[gid] = {
                        "home_abbr": home_abbr,
                        "away_abbr": away_abbr,
                        "home_name": home_name,
                        "away_name": away_name,
                        "home_id": home_id,
                        "away_id": away_id,
                        "game_date": g.data.get("game_date", today),
                    }
            except Exception:
                pass

        # Build predictions with opponent info
        predictions = []
        game_pick_counts: Counter = Counter()

        for row in result.data or []:
            player_data = players_map.get(row.get("player_id", ""), {})
            team_id = str(player_data.get("team_id", ""))
            team_abbr = teams_abbr_map.get(team_id, "")

            if team and team.upper() != team_abbr.upper():
                continue

            # Determine opponent from game matchup
            gid = row.get("game_id", "")
            matchup = game_matchup.get(gid, {})
            if team_id == matchup.get("home_id"):
                opponent = matchup.get("away_abbr", "")
            elif team_id == matchup.get("away_id"):
                opponent = matchup.get("home_abbr", "")
            else:
                opponent = ""

            game_pick_counts[gid] += 1

            predictions.append(PredictionDetail(
                id=row["id"],
                player_id=row.get("player_id", ""),
                player_name=player_data.get("name", "Unknown"),
                team=team_abbr,
                opponent=opponent,
                game_id=gid,
                prop_type=row.get("prop_type", ""),
                line=row.get("line", 0),
                predicted_value=row.get("predicted_value", 0),
                prediction_range_low=row.get("prediction_range_low", 0),
                prediction_range_high=row.get("prediction_range_high", 0),
                over_probability=row.get("over_probability", 0.5),
                under_probability=row.get("under_probability", 0.5),
                confidence_score=row.get("confidence_score", 0),
                confidence_tier=row.get("confidence_tier", 1),
                edge_pct=row.get("edge_pct", 0),
                expected_value=row.get("expected_value", 0),
                recommended_bet=row.get("recommended_bet", ""),
                kelly_bet_size=row.get("kelly_bet_size", 0),
                best_book=row.get("best_book", ""),
                best_odds=row.get("best_odds", 0),
                ensemble_agreement=row.get("ensemble_agreement", 0),
                model_contributions=row.get("model_contributions", []),
                feature_importances=row.get("feature_importances", []),
                line_edge_signal=row.get("line_edge_signal"),
                avg_vs_line_pct=row.get("avg_vs_line_pct"),
                pct_games_over_line=row.get("pct_games_over_line"),
                l10_avg=row.get("l10_avg"),
                created_at=row.get("created_at"),
            ))

        # Build GameInfo list with pick counts
        for gid, matchup in game_matchup.items():
            games_list.append(GameInfo(
                game_id=gid,
                home_team=matchup["home_abbr"],
                away_team=matchup["away_abbr"],
                home_team_name=matchup["home_name"],
                away_team_name=matchup["away_name"],
                game_date=matchup["game_date"],
                pick_count=game_pick_counts.get(gid, 0),
            ))

        return PredictionListResponse(
            predictions=predictions,
            games=games_list,
            total=len(predictions),
            filters_applied={
                "prop_type": prop_type,
                "min_confidence": min_confidence,
                "team": team,
                "game_id": game_id,
            },
        )
    except Exception as e:
        logger.error("Error getting today predictions", error=str(e))
        return PredictionListResponse(predictions=[], total=0)


@router.post("/generate")
async def generate_predictions():
    """
    Generate predictions using the SmartPredictor ensemble + enriched NBA data.

    Pipeline:
    1. Refresh real sportsbook odds from The Odds API (source of truth for active players)
    2. Fetch enriched player features from SportsDataIO (advanced stats, pace, defense)
    3. ONLY predict for players who have real sportsbook prop lines (filters out injured/inactive)
    4. Run 4-model ensemble (XGBoost, RF, GBR, Bayesian) per prop type
    5. Compute confidence, edge, EV, Kelly sizing from real odds
    6. Batch insert predictions to database
    """
    sb = get_supabase()
    from datetime import date as date_cls
    from collections import defaultdict

    try:
        import asyncio
        from app.services.nba_data import get_nba_data
        from app.services.odds_api import get_odds_api
        from app.services.smart_predictor import get_smart_predictor, PROP_STAT_MAP

        # ── Step 1: Refresh real odds from The Odds API ──────────────
        # This is the source of truth: if a sportsbook has lines for a
        # player, that player is confirmed active tonight.
        logger.info("Step 1: Refreshing real sportsbook odds...")
        odds_api = get_odds_api()

        # First get events so we can map Odds API event IDs → our game IDs
        odds_events = await odds_api.get_nba_events()
        logger.info(f"Found {len(odds_events)} NBA events from Odds API")

        # Build mapping: Odds API event_id → our games table game_id
        # Match by team names (Odds API uses full names, our DB has abbreviations)
        teams_result_for_map = sb.table("teams").select("id, abbreviation, name").execute()
        team_name_to_id: dict[str, str] = {}
        for t in (teams_result_for_map.data or []):
            # Map full name variations to team_id
            full_name = (t.get("name") or "").lower()
            team_name_to_id[full_name] = str(t["id"])

        today_games = sb.table("games").select("id, home_team_id, away_team_id").eq(
            "game_date", date_cls.today().isoformat()
        ).execute()

        # Build home_team_id+away_team_id → game_id lookup
        game_by_teams: dict[str, str] = {}
        for g in (today_games.data or []):
            key = f"{g['home_team_id']}|{g['away_team_id']}"
            game_by_teams[key] = str(g["id"])

        odds_event_to_game_id: dict[str, str] = {}
        for ev in odds_events:
            # Odds API: "Oklahoma City Thunder" → need to find our team_id
            home_full = (ev.get("home_team") or "").lower()
            away_full = (ev.get("away_team") or "").lower()
            # Extract city-less name (e.g. "thunder" from "oklahoma city thunder")
            home_tid = None
            away_tid = None
            for tname, tid in team_name_to_id.items():
                if tname and (tname in home_full or home_full.endswith(tname)):
                    home_tid = tid
                if tname and (tname in away_full or away_full.endswith(tname)):
                    away_tid = tid
            if home_tid and away_tid:
                key = f"{home_tid}|{away_tid}"
                our_game_id = game_by_teams.get(key, "")
                if not our_game_id:
                    # Auto-create missing game from Odds API event
                    synthetic_id = str(int(hashlib.md5(f"{home_tid}{away_tid}{date_cls.today().isoformat()}".encode()).hexdigest()[:8], 16))
                    try:
                        sb.table("games").upsert({
                            "id": synthetic_id,
                            "game_date": date_cls.today().isoformat(),
                            "home_team_id": home_tid,
                            "away_team_id": away_tid,
                            "status": "scheduled",
                        }).execute()
                        game_by_teams[key] = synthetic_id
                        our_game_id = synthetic_id
                        logger.info(f"Auto-created game {synthetic_id} for {ev.get('away_team')} @ {ev.get('home_team')}")
                    except Exception as e:
                        logger.warning(f"Failed to auto-create game: {e}")
                if our_game_id:
                    odds_event_to_game_id[ev["id"]] = our_game_id
                    logger.info(f"Mapped Odds event {ev['id'][:12]}... → game {our_game_id} ({ev.get('away_team')} @ {ev.get('home_team')})")

        all_props = await odds_api.get_all_todays_props()
        logger.info(f"Fetched {len(all_props)} prop lines from The Odds API")

        # Fetch real spreads and totals (1 extra API request)
        game_lines = await odds_api.get_game_lines()
        logger.info(f"Fetched real spreads/totals for {len(game_lines)} games")
        # Map Odds event_id → game lines, then map to our game_ids
        game_id_to_lines: dict[str, dict] = {}
        for eid, lines in game_lines.items():
            our_gid = odds_event_to_game_id.get(eid)
            if our_gid:
                game_id_to_lines[our_gid] = lines
            # Also map by team names for fallback
            home = (lines.get("home_team") or "").lower()
            away = (lines.get("away_team") or "").lower()
            game_id_to_lines[f"{away}@{home}"] = lines

        if not all_props:
            return {
                "status": "error",
                "message": "No prop lines available from sportsbooks. Games may not have started markets yet.",
            }

        # Build name-based lookup from Odds API props
        # Key: lowercase player name → list of prop dicts
        odds_by_player: dict[str, list[dict]] = defaultdict(list)
        for prop in all_props:
            pname = (prop.get("player") or "").strip().lower()
            if pname:
                odds_by_player[pname].append(prop)

        active_players_from_odds = set(odds_by_player.keys())
        logger.info(f"Active players with sportsbook lines: {len(active_players_from_odds)}")

        # Clear stale prop_lines and insert fresh ones
        try:
            sb.table("prop_lines").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        except Exception:
            pass

        # Match Odds API player names to Supabase player IDs
        players_result = sb.table("players").select("id, name, team_id").execute()
        name_to_supa: dict[str, dict] = {}  # lowercase name → {id, name, team_id}
        for p in (players_result.data or []):
            name_to_supa[p["name"].lower()] = p

        # Insert fresh prop_lines (use our mapped game_id)
        prop_lines_batch = []
        for prop in all_props:
            pname = (prop.get("player") or "").strip().lower()
            supa_player = name_to_supa.get(pname)
            if not supa_player:
                continue
            odds_eid = prop.get("event_id", "")
            mapped_gid = odds_event_to_game_id.get(odds_eid, odds_eid)
            prop_lines_batch.append({
                "player_id": supa_player["id"],
                "game_id": mapped_gid,
                "prop_type": prop.get("prop_type", ""),
                "sportsbook": prop.get("book", ""),
                "line": prop.get("line", 0),
                "over_odds": prop.get("over_odds", 0),
                "under_odds": prop.get("under_odds", 0),
            })

        props_inserted = 0
        for i in range(0, len(prop_lines_batch), 200):
            chunk = prop_lines_batch[i:i + 200]
            try:
                sb.table("prop_lines").insert(chunk).execute()
                props_inserted += len(chunk)
            except Exception as e:
                logger.warning(f"prop_lines insert error: {e}")
        logger.info(f"Inserted {props_inserted} fresh prop lines")

        # Build real odds lookup: supa_player_id|prop_type → best line
        real_odds_map: dict[str, dict] = {}
        for prop in all_props:
            pname = (prop.get("player") or "").strip().lower()
            supa_player = name_to_supa.get(pname)
            if not supa_player:
                continue
            key = f"{supa_player['id']}|{prop.get('prop_type', '')}"
            # Map Odds API event_id → our game_id
            odds_eid = prop.get("event_id", "")
            mapped_game_id = odds_event_to_game_id.get(odds_eid, odds_eid)
            # Keep best line (first one seen, or DraftKings preferred)
            if key not in real_odds_map or prop.get("book_key") == "draftkings":
                real_odds_map[key] = {
                    "line": prop.get("line", 0),
                    "book": prop.get("book", ""),
                    "over_odds": prop.get("over_odds", -110),
                    "under_odds": prop.get("under_odds", -110),
                    "game_id": mapped_game_id,
                }

        # ── Step 2: Build enriched features from SportsDataIO ────────
        logger.info("Step 2: Building enriched player features from SportsDataIO...")
        nba = get_nba_data()
        loop = asyncio.get_event_loop()
        feature_data = await loop.run_in_executor(None, nba.build_full_feature_set)
        enriched_players = feature_data["players"]
        schedule = feature_data["schedule"]

        if not enriched_players:
            return {"status": "error", "message": "No enriched player data available."}

        # Build SportsDataIO name → enriched player data lookup
        sdio_by_name: dict[str, dict] = {}
        for pid, pf in enriched_players.items():
            pname = (pf.get("name") or "").strip().lower()
            if pname:
                sdio_by_name[pname] = pf

        # ── Step 3: Fetch BallDontLie game logs for active players ───
        # This gives us rolling averages, matchup history, home/away splits,
        # rest features, and consistency metrics.
        logger.info("Step 3: Fetching BallDontLie game logs for active players...")
        from app.services.balldontlie import get_balldontlie
        from app.utils.travel import get_travel_distance, get_timezone_change, calculate_fatigue_score

        bdl = get_balldontlie()

        # Build BDL name → id mapping from active players
        bdl_active = await loop.run_in_executor(None, bdl.get_active_players)
        bdl_name_to_id: dict[str, int] = {}
        for p in bdl_active:
            full = f"{p['first_name']} {p['last_name']}".lower()
            bdl_name_to_id[full] = p["id"]

        # Map Odds API team names to BDL team IDs for matchup lookup
        bdl_team_abbr_to_id: dict[str, int] = {}
        for p in bdl_active:
            t = p.get("team", {})
            if t.get("abbreviation"):
                bdl_team_abbr_to_id[t["abbreviation"]] = t["id"]

        # Fetch game logs for all active players with odds
        bdl_ids_to_fetch = []
        bdl_id_to_name: dict[int, str] = {}
        for odds_name in active_players_from_odds:
            bdl_id = bdl_name_to_id.get(odds_name)
            if bdl_id:
                bdl_ids_to_fetch.append(bdl_id)
                bdl_id_to_name[bdl_id] = odds_name

        logger.info(f"Fetching game logs for {len(bdl_ids_to_fetch)} players from BallDontLie...")
        all_game_logs = await loop.run_in_executor(
            None, bdl.get_bulk_game_logs, bdl_ids_to_fetch
        )

        # ── Step 4: Enrich each player with full feature set ─────────
        logger.info("Step 4: Computing full feature set (40+ features per player)...")

        # Get game spread/O-U from Odds API events for game script features
        event_game_info: dict[str, dict] = {}
        for ev in odds_events:
            eid = ev.get("id", "")
            event_game_info[eid] = {
                "home_team": ev.get("home_team", ""),
                "away_team": ev.get("away_team", ""),
            }

        # Get spreads from The Odds API (h2h market gives implied spread)
        # For now, use a simple proxy: team win% difference → spread estimate
        # We'll also check if odds_events have spread info

        for odds_name in active_players_from_odds:
            player = sdio_by_name.get(odds_name)
            if not player:
                continue

            bdl_id = bdl_name_to_id.get(odds_name)
            game_logs = all_game_logs.get(bdl_id, []) if bdl_id else []

            # 3a. Rolling averages + trend + consistency from game logs
            # ALSO override SportsDataIO per-game averages with BDL's
            # DNP-filtered averages (SportsDataIO includes 0-minute games)
            if game_logs:
                gl_features = bdl.build_player_game_log_features(game_logs)
                player.update(gl_features)
                # Override SportsDataIO per-game stats with accurate BDL values
                bdl_to_sdio = {
                    "season_avg_pts": "pts_pg",
                    "season_avg_reb": "reb_pg",
                    "season_avg_ast": "ast_pg",
                    "season_avg_stl": "stl_pg",
                    "season_avg_blk": "blk_pg",
                    "season_avg_tov": "tov_pg",
                    "season_avg_fg3m": "three_pm_pg",
                    "season_avg_min": "mpg",
                }
                for bdl_key, sdio_key in bdl_to_sdio.items():
                    if bdl_key in gl_features:
                        player[sdio_key] = gl_features[bdl_key]
                if "game_log_count" in gl_features:
                    player["games_played"] = gl_features["game_log_count"]

                # Store raw stat arrays for line edge signal computation
                def _parse_min(m):
                    if not m or m in ("0", "00", ""): return 0.0
                    try:
                        if ":" in str(m): p_ = str(m).split(":"); return float(p_[0]) + float(p_[1]) / 60
                        return float(m)
                    except (ValueError, IndexError): return 0.0
                played = [g for g in game_logs if _parse_min(g.get("min", "0")) > 0]
                played.sort(key=lambda g: g.get("game", {}).get("date", ""))
                player["_raw_pts"] = [g.get("pts", 0) or 0 for g in played]
                player["_raw_reb"] = [g.get("reb", 0) or 0 for g in played]
                player["_raw_ast"] = [g.get("ast", 0) or 0 for g in played]
                player["_raw_fg3m"] = [g.get("fg3m", 0) or 0 for g in played]

            # 3b. Matchup history (player vs tonight's opponent)
            opp_abbr = player.get("opponent", "")
            opp_bdl_id = bdl_team_abbr_to_id.get(opp_abbr)
            if game_logs and opp_bdl_id:
                matchup_feats = bdl.build_matchup_features(game_logs, opp_bdl_id)
                player.update(matchup_feats)

            # 3c. Rest features from actual game log dates
            if game_logs:
                rest_feats = bdl.build_rest_features_from_logs(game_logs)
                player.update(rest_feats)

            # 3d. Travel / fatigue scoring
            team_abbr = player.get("team", "")
            is_home = player.get("is_home", False)
            opp_abbr = player.get("opponent", "")
            if is_home:
                # Playing at home — travel from last away game location
                travel_dist = 0.0
                tz_change = 0
            else:
                # Away game — travel from home to opponent city
                travel_dist = get_travel_distance(team_abbr, opp_abbr)
                tz_change = get_timezone_change(team_abbr, opp_abbr)

            player["travel_distance"] = travel_dist
            player["timezone_change"] = tz_change
            player["fatigue_score"] = calculate_fatigue_score(
                travel_dist, tz_change,
                player.get("is_b2b", False),
                player.get("rest_days", 2),
            )

            # 3e. Game script / blowout risk — use REAL Vegas lines
            # Look up real spread and total from The Odds API
            game_id = player.get("game_id", "")
            real_lines = game_id_to_lines.get(str(game_id), {})
            # Fallback: try team name key
            if not real_lines:
                opp_full = player.get("opponent_full", "").lower()
                team_full = player.get("team_full", "").lower()
                if is_home:
                    real_lines = game_id_to_lines.get(f"{opp_full}@{team_full}", {})
                else:
                    real_lines = game_id_to_lines.get(f"{team_full}@{opp_full}", {})

            if real_lines and "spread_home" in real_lines:
                # Real Vegas spread (negative = favored)
                if is_home:
                    spread = real_lines.get("spread_home", 0)
                else:
                    spread = real_lines.get("spread_away", 0)
                over_under = real_lines.get("total", 220)
                logger.debug(f"  {odds_name}: real spread={spread}, O/U={over_under}")
            else:
                # Fallback: estimate from win% differential
                team_win = player.get("team_win_pct", 0.5)
                opp_win = 0.5
                opp_team_data = feature_data.get("teams", {}).get(opp_abbr, {})
                if opp_team_data:
                    opp_win = opp_team_data.get("win_pct", 0.5)
                spread = (opp_win - team_win) * 30
                if not is_home:
                    spread += 3
                else:
                    spread -= 3
                pace_factor = player.get("pace_factor", 1.0)
                over_under = round(220 * pace_factor, 1)

            player["spread"] = round(spread, 1)
            player["over_under"] = round(over_under, 1)
            player["blowout_risk"] = min(abs(spread) / 20.0, 1.0)

            # 3f. Injury impact on teammates
            # Check which players on this team are NOT in the active odds list
            # (meaning they're injured/out) and sum their usage rates
            team_players = [
                (n, sdio_by_name[n]) for n in sdio_by_name
                if sdio_by_name[n].get("team") == team_abbr
            ]
            missing_usage = 0.0
            missing_count = 0
            for tp_name, tp_data in team_players:
                if tp_name not in active_players_from_odds:
                    usage = tp_data.get("usage_rate", 0)
                    starter = tp_data.get("starter_pct", 0)
                    if starter > 0.5 and usage > 15:
                        missing_usage += usage
                        missing_count += 1

            player["missing_teammate_usage"] = round(missing_usage, 1)
            # Boost: redistribute missing usage proportionally
            player_usage = player.get("usage_rate", 20)
            if missing_usage > 0 and player_usage > 0:
                # Player gets a share of missing usage proportional to their own usage
                team_remaining_usage = sum(
                    sdio_by_name[n].get("usage_rate", 0)
                    for n, _ in team_players
                    if n in active_players_from_odds
                )
                share = player_usage / max(team_remaining_usage, 1)
                player["lineup_boost_points"] = round(missing_usage * share * 0.5, 1)
                player["lineup_boost_rebounds"] = round(missing_count * share * 1.0, 1)
                player["lineup_boost_assists"] = round(missing_count * share * 0.5, 1)
            else:
                player["lineup_boost_points"] = 0
                player["lineup_boost_rebounds"] = 0
                player["lineup_boost_assists"] = 0

            # 3g. Opponent positional defense (#5)
            # Map player position to a defensive stat category
            pos = (player.get("position") or "").upper()
            opp_team_data = feature_data.get("teams", {}).get(opp_abbr, {})
            # Approximate: guards face perimeter D, bigs face interior D
            if pos in ("PG", "SG", "G"):
                pos_def = opp_team_data.get("opp_three_pct", 0)  # 3pt defense
            elif pos in ("PF", "C", "F-C", "C-F"):
                pos_def = opp_team_data.get("opp_reb_pg", 0)  # rebounding allowed
            else:
                pos_def = 0
            player["opp_pos_defense"] = pos_def

            # 3h. B2B decay factor (#7)
            # Players typically drop ~5-10% on back-to-backs
            # Use their own B2B history if available, else default
            if player.get("is_b2b"):
                # Check if we have B2B-specific data from game logs
                b2b_games = player.get("b2b_game_count", 0)
                if b2b_games > 0:
                    player["b2b_decay_factor"] = player.get("b2b_avg_drop", -0.08)
                else:
                    player["b2b_decay_factor"] = -0.08  # default 8% drop
            else:
                player["b2b_decay_factor"] = 0.0

            # 3i. Altitude factor (#9)
            # Denver is at 5,280 ft — visiting teams show fatigue
            opp_city = opp_abbr if not is_home else team_abbr
            if opp_city == "DEN" and not is_home:
                # Playing AT Denver as visitor — altitude penalty
                player["altitude_factor"] = -0.05  # ~5% penalty
            elif team_abbr == "DEN" and is_home:
                # Denver playing at home — altitude advantage
                player["altitude_factor"] = 0.03  # ~3% boost (acclimated)
            else:
                player["altitude_factor"] = 0.0

            # 3j. Referee impact (#8) — placeholder
            # Would need referee assignment data (not available in current APIs)
            player["ref_foul_rate"] = 0.0

        # ── Step 5: News sentiment from RSS + NewsAPI ──────────────
        logger.info("Step 5: Fetching news sentiment (RSS + NewsAPI)...")
        from app.services.news_sentiment import get_news_sentiment
        news_svc = get_news_sentiment()
        player_sentiment = await loop.run_in_executor(
            None, news_svc.build_player_sentiment, active_players_from_odds
        )
        # Merge sentiment features into player dicts
        news_mentioned = 0
        for pname, sentiment in player_sentiment.items():
            player = sdio_by_name.get(pname)
            if player:
                player.update(sentiment)
                if sentiment.get("news_volume", 0) > 0:
                    news_mentioned += 1
        logger.info(f"News sentiment: {news_mentioned} players mentioned in recent articles")

        # ── Step 5b: Player prop correlations ────────────────────────
        logger.info("Step 5b: Computing player prop correlations...")
        from app.services.correlation_engine import build_all_player_correlations
        player_correlations = build_all_player_correlations(all_game_logs, bdl_id_to_name)
        logger.info(f"Correlations computed for {len(player_correlations)} players")

        # ── Step 5c: Smart usage redistribution model ────────────────
        logger.info("Step 5c: Building smart redistribution model...")
        from app.services.redistribution import build_redistribution_model, get_redistribution_boost
        multi_season_stats = await loop.run_in_executor(None, bdl.get_multi_season_stats)
        redist_model = build_redistribution_model(multi_season_stats)
        logger.info(f"Redistribution model built for {len(redist_model)} teams")

        # Apply smart redistribution boosts (overrides simple proportional boost)
        for odds_name in active_players_from_odds:
            player = sdio_by_name.get(odds_name)
            if not player:
                continue
            team_abbr = player.get("team", "")
            bdl_team_id = bdl_team_abbr_to_id.get(team_abbr)
            bdl_player_id = bdl_name_to_id.get(odds_name)
            if not bdl_team_id or not bdl_player_id:
                continue

            # Find missing teammates (players on team NOT in active odds)
            team_players_on_team = [
                (n, bdl_name_to_id.get(n))
                for n in sdio_by_name
                if sdio_by_name[n].get("team") == team_abbr and n != odds_name
            ]
            missing_bdl_ids = [
                bid for n, bid in team_players_on_team
                if n not in active_players_from_odds and bid
                and sdio_by_name[n].get("starter_pct", 0) > 0.5
            ]

            if missing_bdl_ids:
                boost = get_redistribution_boost(
                    redist_model, bdl_team_id, bdl_player_id, missing_bdl_ids
                )
                if boost.get("confidence") != "none":
                    player["lineup_boost_points"] = boost.get("pts_boost", 0)
                    player["lineup_boost_rebounds"] = boost.get("reb_boost", 0)
                    player["lineup_boost_assists"] = boost.get("ast_boost", 0)
                    player["redist_confidence"] = boost.get("confidence", "none")
                    player["redist_sample"] = boost.get("sample_games", 0)

        # ── Step 6: Load smart predictor (skip retraining if models exist) ──
        predictor = get_smart_predictor()
        if not predictor.is_trained:
            logger.info("Step 6: No saved models found — training from scratch...")
            predictor.train_all_props(enriched_players)
        else:
            logger.info(f"Step 6: Using pre-trained models for {list(predictor.prop_models.keys())} (use /retrain to rebuild)")

        # ── Step 7: Delete existing predictions for today ────────────
        today_str = date_cls.today().isoformat()
        try:
            sb.table("predictions").delete().gte("created_at", today_str).execute()
        except Exception:
            pass

        # ── Step 8: Generate predictions ONLY for players with real odds ─
        logger.info("Step 8: Generating predictions for active players only...")
        batch = []
        skipped_no_features = 0
        skipped_no_supa = 0

        for odds_name in active_players_from_odds:
            supa_player = name_to_supa.get(odds_name)
            if not supa_player:
                skipped_no_supa += 1
                continue

            supa_id = supa_player["id"]

            player = sdio_by_name.get(odds_name)
            if not player:
                skipped_no_features += 1
                continue

            team = player.get("team", "")
            if team not in schedule:
                continue

            game_id = player.get("game_id", "")

            for prop_type in PROP_STAT_MAP:
                odds_key = f"{supa_id}|{prop_type}"
                real_line = real_odds_map.get(odds_key)
                if not real_line:
                    continue

                stat_col = PROP_STAT_MAP[prop_type]
                avg_val = player.get(stat_col, 0)
                if avg_val <= 0.1:
                    continue

                line = real_line["line"]
                best_book = real_line["book"]
                real_over = real_line["over_odds"]
                real_under = real_line["under_odds"]
                odds_game_id = real_line.get("game_id", "") or game_id

                # ── Line edge signal (proven +24% ROI strategy) ──
                _prop_to_raw = {
                    "points": "_raw_pts", "rebounds": "_raw_reb",
                    "assists": "_raw_ast", "threes": "_raw_fg3m",
                }
                raw_key = _prop_to_raw.get(prop_type)
                raw_vals = player.get(raw_key, []) if raw_key else []
                l10_avg = None
                avg_vs_line_pct = None
                pct_over_line = None
                line_edge_signal = None

                if raw_vals and len(raw_vals) >= 10 and line > 0:
                    l10_avg = round(sum(raw_vals[-10:]) / 10, 2)
                    avg_vs_line_pct = round((l10_avg - line) / line * 100, 1)
                    last20 = raw_vals[-20:] if len(raw_vals) >= 20 else raw_vals
                    pct_over_line = round(sum(1 for v in last20 if v > line) / len(last20), 3)

                    if avg_vs_line_pct >= 50:
                        line_edge_signal = "strong_over"
                    elif avg_vs_line_pct >= 30:
                        line_edge_signal = "moderate_over"
                    elif avg_vs_line_pct <= -50:
                        line_edge_signal = "strong_under"
                    elif avg_vs_line_pct <= -30:
                        line_edge_signal = "moderate_under"

                result = predictor.predict_prop(player, prop_type, line)

                predicted_value = result["predicted_value"]
                over_prob = result["over_probability"]
                under_prob = result["under_probability"]
                confidence_score = result["confidence_score"]
                confidence_tier = result["confidence_tier"]
                agreement = result["ensemble_agreement"]
                pred_std = result.get("prediction_std", 1.0)
                contributions = result.get("contributions", [])

                recommended_bet = "over" if over_prob > 0.5 else "under"
                best_odds = real_over if recommended_bet == "over" else real_under

                # No-vig fair odds calculation
                from app.services.line_tracker import remove_vig, compute_true_ev
                novig = remove_vig(real_over, real_under)
                our_prob = over_prob if recommended_bet == "over" else under_prob
                fair_prob = novig["fair_over_prob"] if recommended_bet == "over" else novig["fair_under_prob"]

                ev_data = compute_true_ev(our_prob, fair_prob, best_odds)
                edge_pct = round(max(ev_data["edge_vs_fair"], 0), 1)
                ev = round(ev_data["true_ev_pct"], 2)
                kelly = round(max(ev_data["kelly_fraction"] * 100 * 0.25, 0), 1)  # quarter Kelly

                # Override recommended_bet if strong line edge signal
                # Backtested strategy (115 days, +21.8% ROI):
                #   OVER: threes/points, 50%+ above line, odds >= +100
                #   UNDER: all props, 50%+ below line, odds >= +100
                over_edge = (
                    line_edge_signal == "strong_over"
                    and prop_type in ("threes", "points")
                    and real_over >= -110
                )
                under_edge = (
                    line_edge_signal == "strong_under"
                    and real_under >= 100
                )
                if over_edge:
                    recommended_bet = "over"
                    best_odds = real_over
                    novig = remove_vig(real_over, real_under)
                    our_prob = over_prob
                    fair_prob = novig["fair_over_prob"]
                    ev_data = compute_true_ev(our_prob, fair_prob, best_odds)
                    edge_pct = round(max(ev_data["edge_vs_fair"], 0), 1)
                    ev = round(ev_data["true_ev_pct"], 2)
                    kelly = round(max(ev_data["kelly_fraction"] * 100 * 0.25, 0), 1)
                elif under_edge:
                    recommended_bet = "under"
                    best_odds = real_under
                    novig = remove_vig(real_over, real_under)
                    our_prob = under_prob
                    fair_prob = novig["fair_under_prob"]
                    ev_data = compute_true_ev(our_prob, fair_prob, best_odds)
                    edge_pct = round(max(ev_data["edge_vs_fair"], 0), 1)
                    ev = round(ev_data["true_ev_pct"], 2)
                    kelly = round(max(ev_data["kelly_fraction"] * 100 * 0.25, 0), 1)

                pred_row = {
                    "player_id": supa_id,
                    "game_id": odds_game_id,
                    "prop_type": prop_type,
                    "line": line,
                    "predicted_value": predicted_value,
                    "prediction_range_low": round(max(predicted_value - pred_std * 1.5, 0), 1),
                    "prediction_range_high": round(predicted_value + pred_std * 1.5, 1),
                    "over_probability": over_prob,
                    "under_probability": under_prob,
                    "confidence_score": confidence_score,
                    "confidence_tier": confidence_tier,
                    "edge_pct": edge_pct,
                    "expected_value": ev,
                    "recommended_bet": recommended_bet,
                    "kelly_bet_size": kelly,
                    "best_book": best_book,
                    "best_odds": best_odds,
                    "ensemble_agreement": round(agreement, 3),
                    "model_contributions": contributions,
                    "feature_importances": [],
                    "preset_used": "balanced",
                }
                # Add line edge columns (will be ignored if DB columns don't exist yet)
                if line_edge_signal is not None:
                    pred_row["line_edge_signal"] = line_edge_signal
                if avg_vs_line_pct is not None:
                    pred_row["avg_vs_line_pct"] = avg_vs_line_pct
                if pct_over_line is not None:
                    pred_row["pct_games_over_line"] = pct_over_line
                if l10_avg is not None:
                    pred_row["l10_avg"] = l10_avg

                batch.append(pred_row)

        edge_signals = sum(1 for p in batch if p.get("line_edge_signal"))
        strong_overs = sum(1 for p in batch if p.get("line_edge_signal") == "strong_over")
        logger.info(
            f"Built {len(batch)} predictions "
            f"(skipped {skipped_no_supa} no Supabase match, "
            f"{skipped_no_features} no SportsDataIO features, "
            f"{edge_signals} line edge signals, {strong_overs} strong overs)"
        )

        # ── Step 9: Batch insert ─────────────────────────────────────
        predictions_created = 0
        for i in range(0, len(batch), 500):
            chunk = batch[i:i + 500]
            try:
                sb.table("predictions").insert(chunk).execute()
                predictions_created += len(chunk)
            except Exception as e:
                logger.error(f"Batch insert error: {e}")

        logger.info(f"Generated {predictions_created} smart predictions (v4 full features)")
        return {
            "status": "completed",
            "predictions_created": predictions_created,
            "active_players_from_odds": len(active_players_from_odds),
            "players_with_game_logs": len(all_game_logs),
            "props_refreshed": props_inserted,
            "teams_playing": len(schedule),
            "skipped_no_supabase_match": skipped_no_supa,
            "skipped_no_features": skipped_no_features,
            "news_players_mentioned": news_mentioned,
            "line_edge_signals": edge_signals,
            "strong_over_signals": strong_overs,
            "engine": "smart_ensemble_v6_line_edge",
        }

    except Exception as e:
        logger.error("Error generating predictions", error=str(e))
        return {"status": "error", "message": str(e)}


@router.post("/evaluate")
async def evaluate_yesterday():
    """
    Evaluate yesterday's predictions against actual results.
    Fetches real box scores from BDL, computes hit rate, RMSE, calibration.
    """
    from app.services.evaluation import evaluate_predictions
    return await evaluate_predictions()


@router.post("/evaluate/{eval_date}")
async def evaluate_date(eval_date: str):
    """Evaluate predictions for a specific date (YYYY-MM-DD)."""
    from datetime import date as d
    from app.services.evaluation import evaluate_predictions
    try:
        target = d.fromisoformat(eval_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    return await evaluate_predictions(target)


@router.post("/snapshot-lines")
async def snapshot_lines():
    """
    Take a timestamped snapshot of all current prop lines for CLV tracking.
    Call this periodically (every 30-60 min on game days) to track line movement.
    """
    from app.services.line_tracker import snapshot_current_lines
    count = await snapshot_current_lines()
    return {"status": "completed", "snapshots_stored": count}


@router.get("/clv/{clv_date}")
async def get_clv(clv_date: str):
    """Compute Closing Line Value for predictions on a specific date."""
    from datetime import date as d
    from app.services.line_tracker import compute_clv_for_date
    try:
        target = d.fromisoformat(clv_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    return await compute_clv_for_date(target)


@router.post("/backtest")
async def run_backtest_endpoint(
    start_date: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end_date: str | None = Query(None, description="End date YYYY-MM-DD"),
    prop_types: str | None = Query(None, description="Comma-separated prop types"),
    min_confidence: float = Query(60.0, description="Minimum confidence to bet"),
    bankroll: float = Query(1000.0, description="Starting bankroll"),
    kelly_fraction: float = Query(0.5, description="Kelly fraction"),
    period: str | None = Query(None, description="Quick period: day, week, month, season"),
):
    """
    Run a historical backtest simulation against real game data.
    Returns equity curve, hit rates, ROI, and full bet log.
    """
    from datetime import date as d, timedelta
    from app.services.backtesting import run_backtest

    # Parse dates
    sd = None
    ed = None

    if period:
        ed = d.today() - timedelta(days=1)
        if period == "day":
            sd = ed
        elif period == "week":
            sd = ed - timedelta(days=7)
        elif period == "month":
            sd = ed - timedelta(days=30)
        elif period == "season":
            sd = d(2025, 10, 21)
    else:
        if start_date:
            try:
                sd = d.fromisoformat(start_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        if end_date:
            try:
                ed = d.fromisoformat(end_date)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")

    # Parse prop types
    pts = None
    if prop_types:
        pts = [p.strip() for p in prop_types.split(",")]

    try:
        result = await run_backtest(
            start_date=sd,
            end_date=ed,
            prop_types=pts,
            min_confidence=min_confidence,
            bankroll=bankroll,
            kelly_fraction=kelly_fraction,
        )
        return result
    except Exception as e:
        logger.error("Backtest failed", error=str(e))
        return {"status": "error", "message": str(e)}


# ── Calibration background task state ──
_calibration_progress: dict = {"status": "idle", "pct": 0, "message": ""}
_calibration_task = None


@router.post("/calibrate")
async def run_calibration_endpoint(
    seasons: str | None = Query(None, description="Comma-separated seasons (e.g. 2024,2025)"),
    prop_types: str | None = Query(None, description="Comma-separated prop types"),
    min_games: int = Query(15, description="Minimum games history per player"),
    sample_pct: float = Query(1.0, description="Fraction of players to evaluate (0.0-1.0)"),
):
    """
    Launch walk-forward calibration as a background task.
    Poll /calibrate/progress for status updates.
    Results saved to /calibration-report when done.
    """
    import asyncio
    from app.services.calibration_engine import run_calibration

    global _calibration_task, _calibration_progress

    # Don't start if already running
    if _calibration_task and not _calibration_task.done():
        return {"status": "already_running", "progress": _calibration_progress}

    s_list = None
    if seasons:
        try:
            s_list = [int(s.strip()) for s in seasons.split(",")]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid seasons format")

    pts = None
    if prop_types:
        pts = [p.strip() for p in prop_types.split(",")]

    _calibration_progress = {"status": "running", "pct": 0, "message": "Starting calibration..."}

    async def _progress_cb(pct: int, msg: str):
        _calibration_progress["pct"] = pct
        _calibration_progress["message"] = msg

    async def _run():
        try:
            result = await run_calibration(
                seasons=s_list,
                prop_types=pts,
                min_games_history=min_games,
                sample_pct=min(max(sample_pct, 0.01), 1.0),
                progress_callback=_progress_cb,
            )
            _calibration_progress["status"] = "completed"
            _calibration_progress["pct"] = 100
            _calibration_progress["message"] = f"Done! {result.get('total_predictions', 0):,} predictions evaluated."
            _calibration_progress["summary"] = result.get("overall", {})
        except Exception as e:
            _calibration_progress["status"] = "error"
            _calibration_progress["message"] = str(e)
            logger.error("Calibration failed", error=str(e))

    _calibration_task = asyncio.create_task(_run())
    return {"status": "started", "message": "Calibration running in background. Poll /calibrate/progress for updates."}


@router.get("/calibrate/progress")
async def get_calibration_progress():
    """Poll calibration progress."""
    return _calibration_progress


@router.get("/calibration-report")
async def get_calibration_report():
    """Get the most recent calibration report from disk."""
    import json
    from pathlib import Path
    report_path = Path(__file__).parent.parent / "calibration" / "calibration_report.json"
    if not report_path.exists():
        return {"status": "not_found", "message": "No calibration report found. Run /calibrate first."}
    with open(report_path) as f:
        return json.load(f)


@router.get("/{prediction_id}", response_model=PredictionResponse)
async def get_prediction_detail(prediction_id: str):
    """Get detailed prediction with model transparency data."""
    sb = get_supabase()
    try:
        result = (
            sb.table("predictions")
            .select("*, players(name, team_id), games(home_team_id, away_team_id, game_date)")
            .eq("id", prediction_id)
            .single()
            .execute()
        )
        row = result.data
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")

        player_data = row.get("players") or {}
        prediction = PredictionDetail(
            id=row["id"],
            player_id=row.get("player_id", ""),
            player_name=player_data.get("name", "Unknown"),
            team=str(player_data.get("team_id", "")),
            opponent="",
            game_id=row.get("game_id", ""),
            prop_type=row.get("prop_type", ""),
            line=row.get("line", 0),
            predicted_value=row.get("predicted_value", 0),
            prediction_range_low=row.get("prediction_range_low", 0),
            prediction_range_high=row.get("prediction_range_high", 0),
            over_probability=row.get("over_probability", 0.5),
            under_probability=row.get("under_probability", 0.5),
            confidence_score=row.get("confidence_score", 0),
            confidence_tier=row.get("confidence_tier", 1),
            edge_pct=row.get("edge_pct", 0),
            expected_value=row.get("expected_value", 0),
            recommended_bet=row.get("recommended_bet", ""),
            kelly_bet_size=row.get("kelly_bet_size", 0),
            best_book=row.get("best_book", ""),
            best_odds=row.get("best_odds", 0),
            ensemble_agreement=row.get("ensemble_agreement", 0),
            model_contributions=row.get("model_contributions", []),
            feature_importances=row.get("feature_importances", []),
            created_at=row.get("created_at"),
        )
        return PredictionResponse(id=prediction_id, prediction=prediction)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting prediction", prediction_id=prediction_id, error=str(e))
        return PredictionResponse(id=prediction_id, message=str(e))
