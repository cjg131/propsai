from __future__ import annotations

import sys
import types
import unittest


class _StructlogStub(types.SimpleNamespace):
    def get_logger(self, *_args, **_kwargs):
        return types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        )


sys.modules.setdefault(
    "structlog",
    _StructlogStub(
        configure=lambda **_kwargs: None,
        get_logger=lambda *_args, **_kwargs: types.SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
            debug=lambda *a, **k: None,
        ),
        contextvars=types.SimpleNamespace(merge_contextvars=lambda *_a, **_k: None),
        processors=types.SimpleNamespace(
            add_log_level=lambda *_a, **_k: None,
            StackInfoRenderer=lambda *_a, **_k: None,
            TimeStamper=lambda *_a, **_k: None,
            JSONRenderer=lambda *_a, **_k: None,
        ),
        dev=types.SimpleNamespace(
            set_exc_info=lambda *_a, **_k: None,
            ConsoleRenderer=lambda *_a, **_k: None,
        ),
        make_filtering_bound_logger=lambda *_a, **_k: None,
        PrintLoggerFactory=lambda *_a, **_k: None,
        stdlib=types.SimpleNamespace(BoundLogger=object),
    ),
)


class CrossMarketConsensusTests(unittest.TestCase):
    def test_spread_consensus_normalizes_per_line(self) -> None:
        from app.services.cross_market_sports import CrossMarketScanner

        scanner = CrossMarketScanner(odds_api_key="")
        event = {
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Home", "price": -110, "point": -3.5},
                                {"name": "Away", "price": -110, "point": 3.5},
                                {"name": "Home", "price": -105, "point": -4.5},
                                {"name": "Away", "price": -115, "point": 4.5},
                            ],
                        }
                    ],
                }
            ]
        }

        consensus = scanner.extract_sharp_consensus(event)["sharp_consensus"]

        self.assertAlmostEqual(consensus["spreads|Home|-3.5"] + consensus["spreads|Away|3.5"], 1.0, places=4)
        self.assertAlmostEqual(consensus["spreads|Home|-4.5"] + consensus["spreads|Away|4.5"], 1.0, places=4)

    def test_cross_market_team_match_uses_word_boundaries(self) -> None:
        from app.services.cross_market_sports import CrossMarketScanner

        scanner = CrossMarketScanner(odds_api_key="")

        self.assertFalse(scanner._team_matches("Heat", "Will the cheaters win tonight?"))
        self.assertTrue(scanner._team_matches("Miami Heat", "Will the Miami Heat win tonight?"))


class ParlaySpreadPricingTests(unittest.TestCase):
    def test_teams_match_does_not_confuse_real_madrid_with_real_sociedad(self) -> None:
        from app.services.parlay_pricer import teams_match

        self.assertFalse(teams_match("Real Madrid", "Real Sociedad"))

    def test_price_spread_requires_negative_line_for_margin_victory(self) -> None:
        from app.services.parlay_pricer import _price_spread

        leg = {"team": "Celtics", "line": 2.5}
        odds_events = [
            {
                "home_team": "Boston Celtics",
                "away_team": "Miami Heat",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "markets": [
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Boston Celtics", "price": -110, "point": 2.5},
                                    {"name": "Miami Heat", "price": -110, "point": -2.5},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        prob = _price_spread(leg, odds_events, {"pinnacle"}, "yes")
        self.assertIsNone(prob)

    def test_price_total_fails_closed_without_event_identity(self) -> None:
        from app.services.parlay_pricer import _price_total

        leg = {"line": 150.5}
        odds_events = [
            {
                "home_team": "Team A",
                "away_team": "Team B",
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "markets": [
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "price": -110, "point": 150.5},
                                    {"name": "Under", "price": -110, "point": 150.5},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]

        self.assertIsNone(_price_total(leg, odds_events, {"pinnacle"}, "yes"))


if __name__ == "__main__":
    unittest.main()
