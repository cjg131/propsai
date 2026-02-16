from __future__ import annotations

"""Kelly Criterion and bankroll management utilities."""


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def american_to_implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def kelly_criterion(
    win_probability: float,
    decimal_odds: float,
    fraction: float = 0.5,
) -> float:
    """
    Calculate Kelly Criterion bet size as a fraction of bankroll.

    Args:
        win_probability: Model's estimated probability of winning (0-1)
        decimal_odds: Decimal odds (e.g., 1.91 for -110)
        fraction: Kelly fraction (0.25=quarter, 0.5=half, 1.0=full)

    Returns:
        Optimal bet size as fraction of bankroll (0-1). Returns 0 if no edge.
    """
    b = decimal_odds - 1  # net odds (profit per $1 wagered)
    p = win_probability
    q = 1 - p

    if b <= 0 or p <= 0 or p >= 1:
        return 0.0

    kelly = (b * p - q) / b

    if kelly <= 0:
        return 0.0

    return kelly * fraction


def calculate_ev(
    win_probability: float,
    american_odds: int,
    stake: float = 1.0,
) -> float:
    """
    Calculate expected value of a bet.

    Returns:
        Expected value in dollars for the given stake.
    """
    decimal_odds = american_to_decimal(american_odds)
    profit_if_win = stake * (decimal_odds - 1)
    loss_if_lose = stake

    ev = (win_probability * profit_if_win) - ((1 - win_probability) * loss_if_lose)
    return round(ev, 4)


def calculate_edge(
    model_probability: float,
    american_odds: int,
) -> float:
    """
    Calculate the edge (advantage) over the sportsbook.

    Returns:
        Edge as a percentage. Positive = favorable bet.
    """
    implied_prob = american_to_implied_probability(american_odds)
    edge = model_probability - implied_prob
    return round(edge * 100, 2)


def confidence_to_tier(confidence: float) -> int:
    """
    Map a confidence score (0-1) to a 1-5 star tier.

    Tiers:
        1 star: 0.40 - 0.50
        2 star: 0.50 - 0.58
        3 star: 0.58 - 0.65
        4 star: 0.65 - 0.75
        5 star: 0.75+
    """
    if confidence >= 0.75:
        return 5
    elif confidence >= 0.65:
        return 4
    elif confidence >= 0.58:
        return 3
    elif confidence >= 0.50:
        return 2
    else:
        return 1


def kelly_to_units(
    kelly_fraction: float,
    bankroll: float,
    unit_size: float,
) -> float:
    """Convert Kelly fraction to number of units to bet."""
    bet_amount = bankroll * kelly_fraction
    if unit_size <= 0:
        return 0.0
    return round(bet_amount / unit_size, 1)
