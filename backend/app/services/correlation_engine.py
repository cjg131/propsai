"""
Player Prop Correlation Engine.

Computes per-player correlations between prop types from game logs.
Powers smart parlay suggestions with real data-backed correlations.

Example: If LeBron scores 30+ points, his assists are also over 72% of the time.
This is data NO consumer betting tool provides.
"""
from __future__ import annotations

from collections import defaultdict

from app.logging_config import get_logger

logger = get_logger(__name__)

# Map prop types to BDL stat keys
STAT_KEYS = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "threes": "fg3m",
    "steals": "stl",
    "blocks": "blk",
}


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


def compute_player_correlations(game_logs: list[dict]) -> dict:
    """
    Compute pairwise correlations between all prop types for a single player.

    Returns: {
        "points|assists": {"correlation": 0.45, "sample_size": 42,
                           "both_over_pct": 0.68, "both_under_pct": 0.22},
        ...
    }
    """
    # Filter DNP games
    played = [g for g in game_logs if _parse_min(g.get("min", "0")) > 0]
    if len(played) < 10:
        return {}

    # Extract stat arrays
    stats: dict[str, list[float]] = {}
    for prop_type, key in STAT_KEYS.items():
        stats[prop_type] = [float(g.get(key, 0) or 0) for g in played]

    # Compute averages (used for over/under classification)
    avgs = {pt: sum(vals) / len(vals) for pt, vals in stats.items() if vals}

    correlations = {}
    prop_types = list(STAT_KEYS.keys())

    for i in range(len(prop_types)):
        for j in range(i + 1, len(prop_types)):
            pt1, pt2 = prop_types[i], prop_types[j]
            vals1 = stats.get(pt1, [])
            vals2 = stats.get(pt2, [])

            if len(vals1) != len(vals2) or len(vals1) < 10:
                continue

            n = len(vals1)
            avg1, avg2 = avgs.get(pt1, 0), avgs.get(pt2, 0)

            # Pearson correlation
            mean1 = sum(vals1) / n
            mean2 = sum(vals2) / n
            cov = sum((vals1[k] - mean1) * (vals2[k] - mean2) for k in range(n)) / n
            std1 = (sum((v - mean1) ** 2 for v in vals1) / n) ** 0.5
            std2 = (sum((v - mean2) ** 2 for v in vals2) / n) ** 0.5

            if std1 > 0 and std2 > 0:
                corr = cov / (std1 * std2)
            else:
                corr = 0

            # Both over/under percentages (relative to season average)
            both_over = sum(1 for k in range(n) if vals1[k] > avg1 and vals2[k] > avg2)
            both_under = sum(1 for k in range(n) if vals1[k] < avg1 and vals2[k] < avg2)
            one_over_one_under = n - both_over - both_under

            key = f"{pt1}|{pt2}"
            correlations[key] = {
                "prop1": pt1,
                "prop2": pt2,
                "correlation": round(corr, 3),
                "sample_size": n,
                "both_over_pct": round(both_over / n, 3),
                "both_under_pct": round(both_under / n, 3),
                "split_pct": round(one_over_one_under / n, 3),
            }

    return correlations


def find_correlated_parlays(
    player_correlations: dict[str, dict],
    predictions: list[dict],
    min_correlation: float = 0.3,
    min_confidence: float = 55.0,
) -> list[dict]:
    """
    Find parlay opportunities where props are positively correlated.

    player_correlations: {player_name: {prop_pair: correlation_data}}
    predictions: list of today's prediction dicts
    min_correlation: minimum Pearson r to consider correlated
    min_confidence: minimum confidence score for each leg

    Returns list of suggested parlays with correlation data.
    """
    # Group predictions by player
    by_player: dict[str, list[dict]] = defaultdict(list)
    for pred in predictions:
        name = pred.get("player_name", "").lower()
        if name and pred.get("confidence_score", 0) >= min_confidence:
            by_player[name].append(pred)

    parlays = []

    for player_name, player_preds in by_player.items():
        if len(player_preds) < 2:
            continue

        player_corrs = player_correlations.get(player_name, {})

        # Check all pairs of predictions for this player
        for i in range(len(player_preds)):
            for j in range(i + 1, len(player_preds)):
                p1, p2 = player_preds[i], player_preds[j]
                pt1, pt2 = p1.get("prop_type", ""), p2.get("prop_type", "")

                # Look up correlation
                key1 = f"{pt1}|{pt2}"
                key2 = f"{pt2}|{pt1}"
                corr_data = player_corrs.get(key1) or player_corrs.get(key2)

                if not corr_data:
                    continue

                corr = corr_data.get("correlation", 0)

                # Same-direction bets on positively correlated props
                same_direction = p1.get("recommended_bet") == p2.get("recommended_bet")
                if corr >= min_correlation and same_direction:
                    direction = p1.get("recommended_bet", "over")
                    if direction == "over":
                        hit_pct = corr_data.get("both_over_pct", 0)
                    else:
                        hit_pct = corr_data.get("both_under_pct", 0)

                    combined_conf = (
                        p1.get("confidence_score", 50) + p2.get("confidence_score", 50)
                    ) / 2

                    parlays.append({
                        "player_name": player_name.title(),
                        "legs": [
                            {
                                "prop_type": pt1,
                                "line": p1.get("line"),
                                "predicted": p1.get("predicted_value"),
                                "bet": p1.get("recommended_bet"),
                                "confidence": p1.get("confidence_score"),
                            },
                            {
                                "prop_type": pt2,
                                "line": p2.get("line"),
                                "predicted": p2.get("predicted_value"),
                                "bet": p2.get("recommended_bet"),
                                "confidence": p2.get("confidence_score"),
                            },
                        ],
                        "correlation": round(corr, 3),
                        "historical_hit_pct": round(hit_pct, 3),
                        "combined_confidence": round(combined_conf, 1),
                        "sample_size": corr_data.get("sample_size", 0),
                        "parlay_type": "same_game_correlated",
                    })

                # Opposite-direction bets on negatively correlated props
                elif corr <= -min_correlation and not same_direction:
                    combined_conf = (
                        p1.get("confidence_score", 50) + p2.get("confidence_score", 50)
                    ) / 2

                    parlays.append({
                        "player_name": player_name.title(),
                        "legs": [
                            {
                                "prop_type": pt1,
                                "line": p1.get("line"),
                                "predicted": p1.get("predicted_value"),
                                "bet": p1.get("recommended_bet"),
                                "confidence": p1.get("confidence_score"),
                            },
                            {
                                "prop_type": pt2,
                                "line": p2.get("line"),
                                "predicted": p2.get("predicted_value"),
                                "bet": p2.get("recommended_bet"),
                                "confidence": p2.get("confidence_score"),
                            },
                        ],
                        "correlation": round(corr, 3),
                        "historical_hit_pct": 0,
                        "combined_confidence": round(combined_conf, 1),
                        "sample_size": corr_data.get("sample_size", 0),
                        "parlay_type": "anti_correlated",
                    })

    # Sort by historical hit rate, then correlation strength
    parlays.sort(key=lambda p: (p["historical_hit_pct"], abs(p["correlation"])), reverse=True)
    return parlays


def build_all_player_correlations(all_game_logs: dict[int, list[dict]], bdl_id_to_name: dict[int, str]) -> dict[str, dict]:
    """
    Compute correlations for all players with game logs.
    Returns {player_name: {prop_pair: correlation_data}}.
    """
    all_correlations: dict[str, dict] = {}

    for bdl_id, logs in all_game_logs.items():
        name = bdl_id_to_name.get(bdl_id, "")
        if not name or len(logs) < 10:
            continue

        corrs = compute_player_correlations(logs)
        if corrs:
            all_correlations[name] = corrs

    logger.info(f"Computed correlations for {len(all_correlations)} players")
    return all_correlations
