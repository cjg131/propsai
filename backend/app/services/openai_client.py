from __future__ import annotations

from openai import AsyncOpenAI

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    def __init__(self):
        settings = get_settings()
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def generate_scouting_report(
        self,
        player_name: str,
        team: str,
        opponent: str,
        stats_summary: dict,
        matchup_data: dict | None = None,
        injury_info: dict | None = None,
        prop_lines: list[dict] | None = None,
    ) -> str:
        """Generate an AI scouting report for a player's upcoming game."""
        prompt = self._build_scouting_prompt(
            player_name, team, opponent, stats_summary,
            matchup_data, injury_info, prop_lines,
        )

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an elite NBA analytics expert and sports betting analyst. "
                            "Generate concise, data-driven scouting reports for player prop bets. "
                            "Focus on actionable insights: what props look good and why. "
                            "Be specific with numbers and trends. Keep it under 500 words."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )
            report = response.choices[0].message.content
            logger.info("Scouting report generated", player=player_name)
            return report or "Unable to generate report."
        except Exception as e:
            logger.error(
                "OpenAI scouting report error",
                player=player_name,
                error=str(e),
            )
            return f"Error generating scouting report: {str(e)}"

    def _build_scouting_prompt(
        self,
        player_name: str,
        team: str,
        opponent: str,
        stats: dict,
        matchup: dict | None,
        injury: dict | None,
        props: list[dict] | None,
    ) -> str:
        sections = [
            f"Generate a scouting report for {player_name} ({team}) vs {opponent}.",
            "",
            "## Season Stats",
        ]

        for key, value in stats.items():
            sections.append(f"- {key}: {value}")

        if matchup:
            sections.append("")
            sections.append("## Matchup Data")
            for key, value in matchup.items():
                sections.append(f"- {key}: {value}")

        if injury:
            sections.append("")
            sections.append("## Injury Info")
            for key, value in injury.items():
                sections.append(f"- {key}: {value}")

        if props:
            sections.append("")
            sections.append("## Available Prop Lines")
            for prop in props:
                sections.append(
                    f"- {prop.get('prop_type', 'unknown')}: "
                    f"{prop.get('line', 'N/A')} "
                    f"(best odds: {prop.get('best_odds', 'N/A')})"
                )

        sections.append("")
        sections.append(
            "Analyze the data and provide: "
            "1) Key factors for tonight's game, "
            "2) Which props have the most edge and why, "
            "3) Any concerns or red flags."
        )

        return "\n".join(sections)


_client: OpenAIClient | None = None


def get_openai_client() -> OpenAIClient:
    global _client
    if _client is None:
        _client = OpenAIClient()
    return _client
