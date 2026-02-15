from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Supabase
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_key: str = Field(..., description="Supabase service role key")

    # SportsDataIO
    sportsdataio_api_key: str = Field(
        ..., description="SportsDataIO API key for NBA stats and odds"
    )

    # OpenAI
    openai_api_key: str = Field(
        ..., description="OpenAI API key for GPT-4 scouting reports"
    )

    # The Odds API
    the_odds_api_key: str = Field(
        default="", description="The Odds API key for real sportsbook odds"
    )

    # The Odds API - Historical
    the_odds_api_historical_key: str = Field(
        default="", description="The Odds API key for historical odds data"
    )

    # BallDontLie API
    balldontlie_api_key: str = Field(
        default="", description="BallDontLie API key for game logs and injuries"
    )

    # NewsAPI
    newsapi_key: str = Field(
        default="", description="NewsAPI key for NBA news sentiment"
    )

    # Twitter/X API
    twitter_bearer_token: str = Field(
        default="", description="Twitter bearer token for injury news"
    )

    # Tomorrow.io
    tomorrow_io_api_key: str = Field(
        default="", description="Tomorrow.io API key for weather forecasts (free tier)"
    )

    # Visual Crossing
    visual_crossing_api_key: str = Field(
        default="", description="Visual Crossing API key for weather data ($35/month)"
    )

    # Kalshi
    kalshi_api_key_id: str = Field(
        default="", description="Kalshi API key ID (UUID)"
    )
    kalshi_private_key_path: str = Field(
        default="kalshi.key", description="Path to Kalshi RSA private key file"
    )

    # Sentry
    sentry_dsn: str = Field(default="", description="Sentry DSN for error tracking")

    # App Settings
    app_env: str = Field(default="development", description="Application environment")
    app_debug: bool = Field(default=True, description="Debug mode")
    cors_origins: str = Field(
        default="http://localhost:3000", description="Allowed CORS origins"
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    return Settings()
