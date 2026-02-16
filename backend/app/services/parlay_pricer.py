"""
Parlay fair-value pricing engine.
Takes Kalshi multi-game parlay markets, prices each leg using Odds API sharp lines,
and calculates the fair value of the full parlay to find mispricings.
"""
from __future__ import annotations

from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

# Fuzzy team name matching — maps common Kalshi names to Odds API names
# College basketball teams on Kalshi use short names like "Rutgers", "Illinois"
# Odds API uses full names like "Rutgers Scarlet Knights"

# We match by checking if the Kalshi team name appears as a substring of the Odds API team name
# or vice versa. For ambiguous cases, we use this explicit map.
TEAM_ALIASES: dict[str, list[str]] = {
    # College Basketball
    "southern illinois": ["siu", "southern ill"],
    "northern iowa": ["uni"],
    "saint peter's": ["st. peter's", "st peter's"],
    "loyola maryland": ["loyola md", "loyola-maryland"],
    "indiana st.": ["indiana state"],
    "illinois st.": ["illinois state"],
    "murray st.": ["murray state"],
    "youngstown st.": ["youngstown state"],
    "cleveland st.": ["cleveland state"],
    "oregon st.": ["oregon state"],
    "north texas": ["unt"],
    "south florida": ["usf"],
    "robert morris": ["rmu"],
    "sacred heart": ["shu"],
    "detroit mercy": ["detroit"],
    "florida atlantic": ["fau"],
    # Soccer — England
    "man city": ["manchester city"],
    "man united": ["manchester united", "man utd"],
    "spurs": ["tottenham", "tottenham hotspur"],
    "wolves": ["wolverhampton", "wolverhampton wanderers"],
    "brighton": ["brighton and hove albion", "brighton & hove albion"],
    "west ham": ["west ham united"],
    "newcastle": ["newcastle united"],
    "nottingham": ["nottingham forest"],
    "leicester": ["leicester city"],
    "leeds": ["leeds united"],
    "crystal palace": ["palace"],
    # Soccer — Spain
    "atletico": ["atletico madrid", "atlético madrid", "atl madrid", "atl. madrid"],
    "real madrid": ["real madrid cf"],
    "real sociedad": ["sociedad"],
    "real betis": ["betis"],
    "athletic bilbao": ["athletic club", "bilbao"],
    "celta vigo": ["celta"],
    # Soccer — Italy
    "inter milan": ["inter", "internazionale", "fc internazionale"],
    "ac milan": ["milan"],
    "as roma": ["roma"],
    "lazio": ["ss lazio"],
    # Soccer — Germany
    "bayern munich": ["bayern münchen", "fc bayern", "bayern"],
    "dortmund": ["borussia dortmund", "bvb"],
    "leverkusen": ["bayer leverkusen", "bayer 04"],
    "gladbach": ["borussia mönchengladbach", "borussia monchengladbach"],
    "rb leipzig": ["leipzig", "rasenballsport leipzig"],
    # Soccer — France
    "psg": ["paris saint-germain", "paris sg", "paris saint germain"],
    "marseille": ["olympique de marseille", "om"],
    "lyon": ["olympique lyonnais", "ol"],
    "monaco": ["as monaco"],
    "lille": ["losc lille", "losc"],
    # Soccer — UEFA / International
    "qarabag": ["qarabağ fk", "qarabag fk"],
    "bodoe/glimt": ["bodø/glimt", "bodo/glimt"],
    "club brugge": ["club bruges"],
    "benfica": ["sl benfica", "benfica lisbon"],
    "sporting": ["sporting cp", "sporting lisbon"],
    "galatasaray": ["galatasaray sk"],
    "fenerbahce": ["fenerbahçe"],
    "olympiacos": ["olympiakos piraeus", "olympiakos"],
    # Soccer — Americas
    "la galaxy": ["los angeles galaxy"],
    "lafc": ["los angeles fc"],
    "nycfc": ["new york city fc"],
    "red bulls": ["new york red bulls"],
    "inter miami": ["inter miami cf"],
    # Cricket
    "nepal": ["nepal cricket"],
    "namibia": ["namibia cricket"],
    "pakistan": ["pakistan cricket"],
    "scotland": ["scotland cricket"],
    # Tennis — last name matching
    "ben shelton": ["shelton"],
    "taylor fritz": ["fritz"],
    "emma navarro": ["navarro"],
    "clara tauson": ["tauson"],
    "juan manuel cerundolo": ["cerundolo"],
    "fabian marozsan": ["marozsan"],
    "ugo humbert": ["humbert"],
    "alexander bublik": ["bublik"],
    "pablo carreno busta": ["carreno busta"],
    "jakub mensik": ["mensik"],
    "jannik sinner": ["sinner"],
    "daniil medvedev": ["medvedev"],
    "carlos alcaraz": ["alcaraz"],
    "novak djokovic": ["djokovic"],
    "iga swiatek": ["swiatek"],
    "coco gauff": ["gauff"],
    "aryna sabalenka": ["sabalenka"],
}


def normalize_team(name: str) -> str:
    """Normalize a team name for matching."""
    return name.lower().strip().replace(".", "").replace("'", "'")


def teams_match(kalshi_name: str, odds_name: str) -> bool:
    """Check if a Kalshi team name matches an Odds API team name."""
    k = normalize_team(kalshi_name)
    o = normalize_team(odds_name)

    # Direct match
    if k == o:
        return True

    # Substring match (Kalshi "Rutgers" in Odds "Rutgers Scarlet Knights")
    if k in o or o in k:
        return True

    # Last-word match ("Rutgers" matches "Rutgers Scarlet Knights" via first word)
    k_words = k.split()
    o_words = o.split()
    if k_words and o_words:
        if k_words[0] == o_words[0] and len(k_words[0]) > 3:
            return True

    # Alias matching
    for canonical, aliases in TEAM_ALIASES.items():
        all_names = [canonical] + aliases
        k_match = any(normalize_team(n) == k or normalize_team(n) in k or k in normalize_team(n) for n in all_names)
        o_match = any(normalize_team(n) == o or normalize_team(n) in o or o in normalize_team(n) for n in all_names)
        if k_match and o_match:
            return True

    return False


def price_parlay_legs(
    legs: list[dict[str, Any]],
    odds_events: list[dict[str, Any]],
    sharp_books: set[str] | None = None,
) -> dict[str, Any]:
    """
    Price each leg of a Kalshi parlay using Odds API sharp lines.

    Returns:
        {
            "legs_priced": int,
            "legs_total": int,
            "fair_prob": float,  # Product of individual leg probs
            "legs": [...],  # Each leg with its sharp probability
            "confidence": float,  # How many legs we could price
        }
    """
    if sharp_books is None:
        sharp_books = {"pinnacle", "betfair_ex_eu", "betfair"}

    priced_legs: list[dict[str, Any]] = []
    unpriced_legs: list[dict[str, Any]] = []

    for leg in legs:
        leg_prob = _price_single_leg(leg, odds_events, sharp_books)
        if leg_prob is not None:
            priced_legs.append({**leg, "sharp_prob": leg_prob})
        else:
            unpriced_legs.append(leg)

    legs_total = len(legs)
    legs_priced = len(priced_legs)

    if legs_priced == 0:
        return {
            "legs_priced": 0,
            "legs_total": legs_total,
            "fair_prob": None,
            "legs": priced_legs,
            "unpriced_legs": unpriced_legs,
            "confidence": 0.0,
        }

    # Fair parlay probability = product of individual leg probabilities
    fair_prob = 1.0
    for leg in priced_legs:
        fair_prob *= leg["sharp_prob"]

    # If some legs are unpriced, we can't fully price the parlay
    # Estimate unpriced legs at 50% (maximum uncertainty)
    for _ in unpriced_legs:
        fair_prob *= 0.50

    confidence = legs_priced / legs_total if legs_total > 0 else 0.0

    return {
        "legs_priced": legs_priced,
        "legs_total": legs_total,
        "fair_prob": round(fair_prob, 6),
        "legs": priced_legs,
        "unpriced_legs": unpriced_legs,
        "confidence": round(confidence, 3),
    }


def _price_single_leg(
    leg: dict[str, Any],
    odds_events: list[dict[str, Any]],
    sharp_books: set[str],
) -> float | None:
    """
    Price a single parlay leg using Odds API data.
    Returns the sharp implied probability, or None if no match found.
    """
    leg_type = leg.get("type", "")
    direction = leg.get("direction", "yes")  # "yes" or "no"

    if leg_type == "moneyline":
        return _price_moneyline(leg, odds_events, sharp_books, direction)
    elif leg_type == "spread":
        return _price_spread(leg, odds_events, sharp_books, direction)
    elif leg_type == "total":
        return _price_total(leg, odds_events, sharp_books, direction)
    elif leg_type == "goals_total":
        return _price_total(leg, odds_events, sharp_books, direction)
    elif leg_type == "btts":
        # BTTS is harder to price — skip for now
        return None

    return None


def _price_moneyline(
    leg: dict[str, Any],
    odds_events: list[dict[str, Any]],
    sharp_books: set[str],
    direction: str,
) -> float | None:
    """Price a moneyline leg: 'yes TeamName' means team wins."""
    team = leg.get("team", "")
    if not team:
        return None

    for event in odds_events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")

        # Check if this team is in this event
        is_home = teams_match(team, home)
        is_away = teams_match(team, away)

        if not is_home and not is_away:
            continue

        # Found the event — get sharp h2h odds
        target_team = home if is_home else away

        probs = _extract_sharp_probs(event, "h2h", sharp_books)
        if not probs:
            continue

        # Find the probability for our team
        team_prob = None
        for outcome_name, prob in probs.items():
            if teams_match(target_team, outcome_name):
                team_prob = prob
                break

        if team_prob is None:
            continue

        # "yes TeamName" = team wins → prob = team_prob
        # "no TeamName" = team loses → prob = 1 - team_prob (but Kalshi "no" means we buy NO)
        if direction == "yes":
            return team_prob
        else:
            return 1.0 - team_prob

    return None


def _price_spread(
    leg: dict[str, Any],
    odds_events: list[dict[str, Any]],
    sharp_books: set[str],
    direction: str,
) -> float | None:
    """Price a spread leg: 'yes TeamName wins by over X.5 Points'."""
    team = leg.get("team", "")
    line = leg.get("line", 0)
    if not team or not line:
        return None

    for event in odds_events:
        home = event.get("home_team", "")
        away = event.get("away_team", "")

        is_home = teams_match(team, home)
        is_away = teams_match(team, away)

        if not is_home and not is_away:
            continue

        target_team = home if is_home else away

        # Get spread odds — the Odds API spread for the favorite is negative
        probs = _extract_sharp_spread_probs(event, sharp_books)
        if not probs:
            continue

        # Find matching spread
        for (outcome_name, point), prob in probs.items():
            if teams_match(target_team, outcome_name):
                # Kalshi "wins by over X.5" = spread of -X.5
                # Odds API spread: team at -3.5 means they need to win by 4+
                # Kalshi "wins by over 3.5 Points" = same as spread -3.5
                odds_spread = abs(point) if point else 0
                if abs(odds_spread - line) <= 1.0:  # Allow 1-point tolerance
                    if direction == "yes":
                        return prob
                    else:
                        return 1.0 - prob

    return None


def _price_total(
    leg: dict[str, Any],
    odds_events: list[dict[str, Any]],
    sharp_books: set[str],
    direction: str,
) -> float | None:
    """Price a total leg: 'yes Over X.5 points scored'."""
    line = leg.get("line", 0)
    if not line:
        return None

    # For totals, we need to find the right event
    # This is tricky for parlays since we don't know which game the total belongs to
    # We'll try to match by looking at the total line proximity

    for event in odds_events:
        probs = _extract_sharp_total_probs(event, sharp_books)
        if not probs:
            continue

        for (over_under, point), prob in probs.items():
            if abs(point - line) <= 2.0:  # Allow 2-point tolerance for totals
                if over_under.lower() == "over" and direction == "yes":
                    return prob
                elif over_under.lower() == "under" and direction == "no":
                    return prob
                elif over_under.lower() == "over" and direction == "no":
                    return 1.0 - prob
                elif over_under.lower() == "under" and direction == "yes":
                    return 1.0 - prob

    return None


def _extract_sharp_probs(
    event: dict[str, Any],
    market_key: str,
    sharp_books: set[str],
) -> dict[str, float]:
    """Extract sharp bookmaker probabilities for a market type."""
    probs: dict[str, list[float]] = {}

    for bm in event.get("bookmakers", []):
        if bm.get("key", "") not in sharp_books:
            continue

        for market in bm.get("markets", []):
            if market.get("key", "") != market_key:
                continue

            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price", 0)
                if isinstance(price, (int, float)) and abs(price) >= 100:
                    prob = _american_to_prob(int(price))
                    if name not in probs:
                        probs[name] = []
                    probs[name].append(prob)

    # Average across sharp books
    return {name: sum(ps) / len(ps) for name, ps in probs.items() if ps}


def _extract_sharp_spread_probs(
    event: dict[str, Any],
    sharp_books: set[str],
) -> dict[tuple[str, float], float]:
    """Extract sharp spread probabilities."""
    probs: dict[tuple[str, float], list[float]] = {}

    for bm in event.get("bookmakers", []):
        if bm.get("key", "") not in sharp_books:
            continue

        for market in bm.get("markets", []):
            if market.get("key", "") != "spreads":
                continue

            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price", 0)
                point = outcome.get("point", 0)
                if isinstance(price, (int, float)) and abs(price) >= 100:
                    prob = _american_to_prob(int(price))
                    key = (name, float(point))
                    if key not in probs:
                        probs[key] = []
                    probs[key].append(prob)

    return {k: sum(ps) / len(ps) for k, ps in probs.items() if ps}


def _extract_sharp_total_probs(
    event: dict[str, Any],
    sharp_books: set[str],
) -> dict[tuple[str, float], float]:
    """Extract sharp total probabilities."""
    probs: dict[tuple[str, float], list[float]] = {}

    for bm in event.get("bookmakers", []):
        if bm.get("key", "") not in sharp_books:
            continue

        for market in bm.get("markets", []):
            if market.get("key", "") != "totals":
                continue

            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")  # "Over" or "Under"
                price = outcome.get("price", 0)
                point = outcome.get("point", 0)
                if isinstance(price, (int, float)) and abs(price) >= 100:
                    prob = _american_to_prob(int(price))
                    key = (name, float(point))
                    if key not in probs:
                        probs[key] = []
                    probs[key].append(prob)

    return {k: sum(ps) / len(ps) for k, ps in probs.items() if ps}


def _american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)
