"""
Smart Usage Redistribution Model.

Analyzes historical games where key players were missing to compute
real stat redistribution patterns per team. Instead of simple proportional
boosts, this measures the *actual* stat changes for remaining players
when specific teammates are out.

Example: When LeBron sits, AD's rebounds go up 3.2 but his assists drop 1.1.
         When Haliburton sits, Nembhard's assists jump from 4.2 to 7.8.
"""
from __future__ import annotations

from collections import defaultdict

from app.logging_config import get_logger

logger = get_logger(__name__)


def _parse_min(m: str) -> float:
    if not m or m in ("0", "00", ""):
        return 0.0
    try:
        if ":" in str(m):
            p = str(m).split(":")
            return float(p[0]) + float(p[1]) / 60
        return float(m)
    except (ValueError, IndexError):
        return 0.0


STAT_KEYS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]


def build_redistribution_model(season_stats: list[dict]) -> dict:
    """
    From ALL season box scores, build a redistribution model:
    1. Group stats by game
    2. For each game, identify which players played and which didn't
    3. Compare a player's stats in games WITH vs WITHOUT each teammate
    4. Compute the average stat delta per missing teammate

    Returns: {
        team_id: {
            missing_player_id: {
                remaining_player_id: {
                    "pts_delta": +2.3,
                    "reb_delta": +1.1,
                    "ast_delta": -0.5,
                    "games_with": 35,
                    "games_without": 8,
                }
            }
        }
    }
    """
    # Filter out DNP games
    played_stats = [s for s in season_stats if _parse_min(s.get("min", "0")) > 0]

    if not played_stats:
        return {}

    # Group by game_id → list of player stats
    games: dict[int, list[dict]] = defaultdict(list)
    for s in played_stats:
        game_id = s.get("game", {}).get("id")
        if game_id:
            games[game_id].append(s)

    # Group by team → all player IDs on that team
    team_players: dict[int, set[int]] = defaultdict(set)
    for s in played_stats:
        team_id = s.get("team", {}).get("id")
        player_id = s.get("player", {}).get("id")
        if team_id and player_id:
            team_players[team_id].add(player_id)

    # For each team, find games where specific players were missing
    # and compute stat deltas for remaining players
    model: dict[int, dict] = {}

    for team_id, all_pids in team_players.items():
        if len(all_pids) < 5:
            continue

        # Group this team's stats by game
        team_game_stats: dict[int, dict[int, dict]] = defaultdict(dict)
        for s in played_stats:
            if s.get("team", {}).get("id") != team_id:
                continue
            game_id = s.get("game", {}).get("id")
            pid = s.get("player", {}).get("id")
            if game_id and pid:
                team_game_stats[game_id][pid] = s

        # For each player on the team, split games into "with" and "without"
        # each other teammate
        team_model: dict[int, dict] = {}

        # Only analyze players who played 20+ games (starters/rotation)
        player_game_counts: dict[int, int] = defaultdict(int)
        for gid, players_in_game in team_game_stats.items():
            for pid in players_in_game:
                player_game_counts[pid] += 1

        key_players = {pid for pid, count in player_game_counts.items() if count >= 20}

        for missing_pid in key_players:
            missing_model: dict[int, dict] = {}

            for remaining_pid in key_players:
                if remaining_pid == missing_pid:
                    continue

                # Games where BOTH played
                stats_with: list[dict] = []
                # Games where remaining played but missing was OUT
                stats_without: list[dict] = []

                for gid, players_in_game in team_game_stats.items():
                    if remaining_pid not in players_in_game:
                        continue
                    if missing_pid in players_in_game:
                        stats_with.append(players_in_game[remaining_pid])
                    else:
                        stats_without.append(players_in_game[remaining_pid])

                if len(stats_with) < 10 or len(stats_without) < 2:
                    continue

                # Compute average stats with vs without
                deltas = {}
                for stat in STAT_KEYS:
                    avg_with = sum(s.get(stat, 0) or 0 for s in stats_with) / len(stats_with)
                    avg_without = sum(s.get(stat, 0) or 0 for s in stats_without) / len(stats_without)
                    deltas[f"{stat}_delta"] = round(avg_without - avg_with, 2)
                    deltas[f"{stat}_with"] = round(avg_with, 2)
                    deltas[f"{stat}_without"] = round(avg_without, 2)

                deltas["games_with"] = len(stats_with)
                deltas["games_without"] = len(stats_without)

                # Only store if there's a meaningful delta
                if any(abs(deltas.get(f"{s}_delta", 0)) > 0.5 for s in STAT_KEYS):
                    missing_model[remaining_pid] = deltas

            if missing_model:
                team_model[missing_pid] = missing_model

        if team_model:
            model[team_id] = team_model

    logger.info(f"Built redistribution model for {len(model)} teams")
    return model


def get_redistribution_boost(
    model: dict,
    team_id: int,
    player_id: int,
    missing_player_ids: list[int],
) -> dict:
    """
    Given a redistribution model, compute the expected stat boost
    for a player when specific teammates are missing.

    Returns: {
        "pts_boost": +3.5,
        "reb_boost": +1.2,
        "ast_boost": -0.3,
        "confidence": "high",  # based on sample size
        "missing_players_modeled": 2,
    }
    """
    team_model = model.get(team_id, {})
    if not team_model:
        return {"pts_boost": 0, "reb_boost": 0, "ast_boost": 0, "confidence": "none", "missing_players_modeled": 0}

    total_boosts: dict[str, float] = defaultdict(float)
    modeled_count = 0
    total_sample = 0

    for missing_pid in missing_player_ids:
        missing_data = team_model.get(missing_pid, {})
        player_data = missing_data.get(player_id, {})
        if player_data:
            for stat in STAT_KEYS:
                total_boosts[f"{stat}_boost"] = total_boosts.get(f"{stat}_boost", 0) + player_data.get(f"{stat}_delta", 0)
            modeled_count += 1
            total_sample += player_data.get("games_without", 0)

    # Confidence based on sample size
    if total_sample >= 10:
        confidence = "high"
    elif total_sample >= 5:
        confidence = "medium"
    elif modeled_count > 0:
        confidence = "low"
    else:
        confidence = "none"

    result = {
        "pts_boost": round(total_boosts.get("pts_boost", 0), 1),
        "reb_boost": round(total_boosts.get("reb_boost", 0), 1),
        "ast_boost": round(total_boosts.get("ast_boost", 0), 1),
        "fg3m_boost": round(total_boosts.get("fg3m_boost", 0), 1),
        "stl_boost": round(total_boosts.get("stl_boost", 0), 1),
        "blk_boost": round(total_boosts.get("blk_boost", 0), 1),
        "confidence": confidence,
        "missing_players_modeled": modeled_count,
        "sample_games": total_sample,
    }
    return result
