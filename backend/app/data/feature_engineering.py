from __future__ import annotations

"""
Feature engineering pipeline for NBA player prop predictions.
Transforms raw player/game data into ML-ready features.
"""

from datetime import date

import numpy as np
import pandas as pd

from app.logging_config import get_logger
from app.utils.travel import get_timezone_change, get_travel_distance

logger = get_logger(__name__)

# Time decay: 80% per season (exponential decay)
DECAY_RATE = 0.80
CURRENT_SEASON_YEAR = 2026


def compute_time_weight(season_id: str) -> float:
    """Compute time-decay weight for a season. Current = 1.0, last = 0.8, etc."""
    try:
        season_year = int(season_id)
    except ValueError:
        return 0.5
    years_ago = CURRENT_SEASON_YEAR - season_year
    return DECAY_RATE ** max(years_ago, 0)


def compute_rolling_averages(
    game_log: pd.DataFrame,
    stat_columns: list[str],
    windows: list[int] = [5, 10, 20],
) -> pd.DataFrame:
    """Compute rolling averages for specified stat columns over multiple windows."""
    result = game_log.copy()
    for col in stat_columns:
        for window in windows:
            result[f"{col}_last{window}"] = (
                result[col].rolling(window=window, min_periods=1).mean()
            )
    return result


def compute_home_away_splits(game_log: pd.DataFrame, stat_columns: list[str]) -> dict:
    """Compute home vs away averages for a player."""
    splits = {}
    if "is_home" not in game_log.columns:
        return splits

    home_games = game_log[game_log["is_home"]]
    away_games = game_log[not game_log["is_home"]]

    for col in stat_columns:
        splits[f"home_avg_{col}"] = home_games[col].mean() if len(home_games) > 0 else 0.0
        splits[f"away_avg_{col}"] = away_games[col].mean() if len(away_games) > 0 else 0.0
    return splits


def compute_matchup_features(
    player_game_log: pd.DataFrame,
    opponent_team_id: str,
) -> dict:
    """Compute how a player performs against a specific opponent."""
    vs_opponent = player_game_log[player_game_log["opponent_team_id"] == opponent_team_id]
    features = {}

    stat_cols = ["points", "rebounds", "assists", "three_pointers_made", "minutes"]
    for col in stat_cols:
        if col in vs_opponent.columns:
            features[f"vs_opp_avg_{col}"] = vs_opponent[col].mean() if len(vs_opponent) > 0 else 0.0
            features["vs_opp_games"] = len(vs_opponent)
    return features


def compute_rest_features(
    game_date: date,
    last_game_date: date | None,
    second_last_game_date: date | None,
) -> dict:
    """Compute rest-related features."""
    features = {}

    if last_game_date:
        days_rest = (game_date - last_game_date).days
        features["days_rest"] = days_rest
        features["is_back_to_back"] = 1 if days_rest <= 1 else 0
    else:
        features["days_rest"] = 7  # Default if unknown
        features["is_back_to_back"] = 0

    if second_last_game_date and last_game_date:
        features["is_3_in_4"] = 1 if (game_date - second_last_game_date).days <= 3 else 0
    else:
        features["is_3_in_4"] = 0

    return features


def compute_travel_features(
    last_game_team: str,
    current_game_team: str,
    is_home: bool,
) -> dict:
    """Compute travel distance and timezone change features."""
    if is_home:
        # Playing at home, travel from last game location to home
        distance = get_travel_distance(last_game_team, current_game_team)
    else:
        distance = get_travel_distance(last_game_team, current_game_team)

    tz_change = get_timezone_change(last_game_team, current_game_team)

    return {
        "travel_distance": distance,
        "timezone_change": tz_change,
    }


def compute_referee_features(
    referee_stats: dict | None,
    referee_player_stats: dict | None,
) -> dict:
    """Compute referee-related features."""
    features = {
        "ref_avg_fouls": 0.0,
        "ref_avg_pace": 0.0,
        "ref_home_win_pct": 0.5,
        "ref_player_avg_points": 0.0,
        "ref_player_avg_fta": 0.0,
        "ref_player_games": 0,
    }

    if referee_stats:
        features["ref_avg_fouls"] = referee_stats.get("avg_fouls_per_game", 0.0)
        features["ref_avg_pace"] = referee_stats.get("avg_pace", 0.0)
        features["ref_home_win_pct"] = referee_stats.get("home_win_pct", 0.5)

    if referee_player_stats:
        features["ref_player_avg_points"] = referee_player_stats.get("avg_points", 0.0)
        features["ref_player_avg_fta"] = referee_player_stats.get("avg_free_throws_attempted", 0.0)
        features["ref_player_games"] = referee_player_stats.get("games_together", 0)

    return features


def compute_lineup_impact(
    player_stats_with_teammate: dict | None,
    player_stats_without_teammate: dict | None,
    missing_teammates: list[str] | None = None,
) -> dict:
    """
    Compute the impact of missing teammates on a player's projected stats.
    Uses on/off splits and usage redistribution.
    """
    features = {
        "lineup_boost_points": 0.0,
        "lineup_boost_rebounds": 0.0,
        "lineup_boost_assists": 0.0,
        "lineup_boost_minutes": 0.0,
        "missing_teammate_count": len(missing_teammates) if missing_teammates else 0,
    }

    if not missing_teammates:
        return features

    # Simplified: each missing starter adds a small boost to remaining players
    # Full implementation would use actual on/off split data
    for _ in missing_teammates:
        features["lineup_boost_points"] += 1.5
        features["lineup_boost_rebounds"] += 0.5
        features["lineup_boost_assists"] += 0.5
        features["lineup_boost_minutes"] += 2.0

    return features


def compute_game_script_features(
    spread: float | None,
    over_under: float | None,
    opponent_pace: float | None,
    opponent_def_rating: float | None,
) -> dict:
    """Compute game environment features that affect minutes and stats."""
    features = {
        "spread": spread or 0.0,
        "over_under": over_under or 220.0,
        "is_heavy_favorite": 1 if spread and spread < -10 else 0,
        "is_heavy_underdog": 1 if spread and spread > 10 else 0,
        "blowout_risk": 0.0,
        "opponent_pace": opponent_pace or 100.0,
        "opponent_def_rating": opponent_def_rating or 110.0,
    }

    # Blowout risk: higher spread = more likely starters sit in 4th quarter
    if spread:
        features["blowout_risk"] = min(abs(spread) / 20.0, 1.0)

    return features


def build_feature_vector(
    player_season_stats: dict,
    rolling_stats: dict,
    home_away_splits: dict,
    matchup_features: dict,
    rest_features: dict,
    travel_features: dict,
    referee_features: dict,
    lineup_features: dict,
    game_script_features: dict,
    is_home: bool = True,
    is_starter: bool = True,
    is_playoff: bool = False,
    is_recently_traded: bool = False,
    is_rookie: bool = False,
) -> dict:
    """Combine all feature groups into a single feature vector."""
    features = {}

    # Season averages
    for key, value in player_season_stats.items():
        features[f"season_{key}"] = value

    # Rolling averages
    for key, value in rolling_stats.items():
        features[key] = value

    # Home/away splits
    for key, value in home_away_splits.items():
        features[key] = value

    # Matchup features
    for key, value in matchup_features.items():
        features[key] = value

    # Rest features
    for key, value in rest_features.items():
        features[key] = value

    # Travel features
    for key, value in travel_features.items():
        features[key] = value

    # Referee features
    for key, value in referee_features.items():
        features[key] = value

    # Lineup impact
    for key, value in lineup_features.items():
        features[key] = value

    # Game script
    for key, value in game_script_features.items():
        features[key] = value

    # Binary flags
    features["is_home"] = 1 if is_home else 0
    features["is_starter"] = 1 if is_starter else 0
    features["is_playoff"] = 1 if is_playoff else 0
    features["is_recently_traded"] = 1 if is_recently_traded else 0
    features["is_rookie"] = 1 if is_rookie else 0

    return features


def build_training_dataset(
    player_game_stats: pd.DataFrame,
    games: pd.DataFrame,
    target_stat: str = "points",
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Build a full training dataset from historical data.
    Returns (features_df, target_series, sample_weights).

    Supports two modes:
    - Per-game stats (game_id matches games table) → rolling averages + game context
    - Season aggregate stats (synthetic game_id) → cross-player features only
    """
    logger.info(f"Building training dataset for {target_stat}...")

    stat_columns = [
        "points", "rebounds", "assists", "steals", "blocks",
        "turnovers", "three_pointers_made", "minutes",
    ]

    # Try merging with games table
    df = player_game_stats.merge(
        games[["id", "game_date", "season_id", "home_team_id", "away_team_id",
               "spread", "over_under", "is_playoff"]],
        left_on="game_id",
        right_on="id",
        how="left",
        suffixes=("", "_game"),
    )

    has_game_dates = df["game_date"].notna().sum() > len(df) * 0.5

    if has_game_dates:
        # Full per-game mode with rolling averages
        return _build_pergame_dataset(df, stat_columns, target_stat)
    else:
        # Season aggregate mode — each row is one player-season
        return _build_aggregate_dataset(player_game_stats, stat_columns, target_stat)


def _build_pergame_dataset(
    df: pd.DataFrame,
    stat_columns: list[str],
    target_stat: str,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Build training data from per-game stats with rolling averages."""
    df = df.sort_values(["player_id", "game_date"])

    feature_rows = []
    targets = []
    weights = []

    for player_id, player_df in df.groupby("player_id"):
        if len(player_df) < 5:
            continue

        player_df = player_df.reset_index(drop=True)
        player_df = compute_rolling_averages(player_df, stat_columns)
        player_df["is_home"] = player_df["team_id"] == player_df["home_team_id"]

        for i in range(5, len(player_df)):
            row = player_df.iloc[i]
            prev_rows = player_df.iloc[:i]

            rolling_feats = {}
            for col in stat_columns:
                for w in [5, 10, 20]:
                    key = f"{col}_last{w}"
                    if key in row:
                        rolling_feats[key] = row[key]

            season_feats = {}
            for col in stat_columns:
                season_feats[f"avg_{col}"] = prev_rows[col].mean()

            game_date = pd.to_datetime(row["game_date"]).date() if isinstance(row["game_date"], str) else row["game_date"]
            last_date = pd.to_datetime(prev_rows.iloc[-1]["game_date"]).date() if isinstance(prev_rows.iloc[-1]["game_date"], str) else prev_rows.iloc[-1]["game_date"]
            second_last = None
            if len(prev_rows) >= 2:
                second_last = pd.to_datetime(prev_rows.iloc[-2]["game_date"]).date() if isinstance(prev_rows.iloc[-2]["game_date"], str) else prev_rows.iloc[-2]["game_date"]

            rest_feats = compute_rest_features(game_date, last_date, second_last)
            script_feats = compute_game_script_features(
                row.get("spread"), row.get("over_under"), None, None
            )

            features = {
                **season_feats,
                **rolling_feats,
                **rest_feats,
                **script_feats,
                "is_home": 1 if row.get("is_home") else 0,
                "is_starter": 1 if row.get("is_starter") else 0,
                "is_playoff": 1 if row.get("is_playoff") else 0,
                "games_played": i,
            }

            feature_rows.append(features)
            targets.append(row[target_stat])

            season_id = str(row.get("season_id", CURRENT_SEASON_YEAR))
            weights.append(compute_time_weight(season_id))

    if not feature_rows:
        logger.warning("No per-game training data generated")
        return pd.DataFrame(), pd.Series(dtype=float), np.array([])

    X = pd.DataFrame(feature_rows).fillna(0)
    y = pd.Series(targets, dtype=float)
    w = np.array(weights, dtype=float)
    logger.info(f"Per-game training dataset: {len(X)} samples, {len(X.columns)} features")
    return X, y, w


def _build_aggregate_dataset(
    stats_df: pd.DataFrame,
    stat_columns: list[str],
    target_stat: str,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Build training data from season aggregate stats.
    Each player-season row becomes a training sample.
    Features are the other stat columns; target is the specified stat.
    """
    available_cols = [c for c in stat_columns if c in stats_df.columns]
    if target_stat not in stats_df.columns:
        logger.warning(f"Target stat '{target_stat}' not in columns")
        return pd.DataFrame(), pd.Series(dtype=float), np.array([])

    df = stats_df.dropna(subset=[target_stat]).copy()
    df = df[df[target_stat] != 0]  # Filter out zero-stat rows

    if len(df) < 10:
        logger.warning(f"Too few samples for {target_stat}: {len(df)}")
        return pd.DataFrame(), pd.Series(dtype=float), np.array([])

    # Features: all stat columns except the target, plus derived features
    feature_cols = [c for c in available_cols if c != target_stat]
    feature_rows = []
    targets = []
    weights = []

    for _, row in df.iterrows():
        features = {}
        for col in feature_cols:
            features[f"avg_{col}"] = row.get(col, 0) or 0

        # Derived features
        fg_att = row.get("field_goals_attempted", 0) or 0
        fg_made = row.get("field_goals_made", 0) or 0
        ft_att = row.get("free_throws_attempted", 0) or 0
        ft_made = row.get("free_throws_made", 0) or 0
        features["fg_pct"] = fg_made / max(fg_att, 1)
        features["ft_pct"] = ft_made / max(ft_att, 1)
        features["is_starter"] = 1 if row.get("is_starter") else 0
        features["minutes_per_game"] = row.get("minutes", 0) or 0

        feature_rows.append(features)
        targets.append(row[target_stat])

        # Extract season from game_id if possible
        game_id = str(row.get("game_id", ""))
        season_year = CURRENT_SEASON_YEAR
        if game_id.startswith("season-"):
            parts = game_id.split("-")
            if len(parts) >= 2:
                try:
                    season_year = int(parts[1])
                except ValueError:
                    pass
        weights.append(compute_time_weight(str(season_year)))

    X = pd.DataFrame(feature_rows).fillna(0)
    y = pd.Series(targets, dtype=float)
    w = np.array(weights, dtype=float)
    logger.info(f"Aggregate training dataset: {len(X)} samples, {len(X.columns)} features")
    return X, y, w
