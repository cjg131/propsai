from __future__ import annotations

"""
Data management service — orchestrates data loading, refreshing,
caching, API quota tracking, and model retraining.
"""

from datetime import date, datetime

from app.logging_config import get_logger
from app.services.injury_scraper import get_injury_scraper
from app.services.sportsdataio import get_sportsdataio
from app.services.supabase_client import get_supabase

logger = get_logger(__name__)


class DataManager:
    def __init__(self):
        self.supabase = get_supabase()
        self.sports_api = get_sportsdataio()
        self.injury_scraper = get_injury_scraper()

    # ---- API Quota Tracking ----

    async def track_api_usage(self, service: str, endpoint: str = "") -> None:
        """Track API usage for quota management."""
        today = date.today().isoformat()
        try:
            existing = (
                self.supabase.table("api_usage")
                .select("*")
                .eq("service", service)
                .eq("date", today)
                .execute()
            )
            if existing.data:
                self.supabase.table("api_usage").update(
                    {"requests_made": existing.data[0]["requests_made"] + 1}
                ).eq("id", existing.data[0]["id"]).execute()
            else:
                self.supabase.table("api_usage").insert(
                    {"service": service, "endpoint": endpoint, "date": today}
                ).execute()
        except Exception as e:
            logger.warning("Failed to track API usage", service=service, error=str(e))

    async def get_api_usage(self, service: str) -> dict:
        """Get current API usage for a service."""
        today = date.today().isoformat()
        try:
            result = (
                self.supabase.table("api_usage")
                .select("*")
                .eq("service", service)
                .eq("date", today)
                .execute()
            )
            if result.data:
                return result.data[0]
        except Exception:
            pass
        return {"service": service, "requests_made": 0, "date": today}

    # ---- Data Status ----

    async def get_data_status(self) -> dict:
        """Get current data pipeline status."""
        total_players = 0
        total_games = 0
        total_seasons = 0
        api_quota_used = 0
        model_last_trained = None

        # Each query is independent so one failure doesn't block others
        try:
            players = self.supabase.table("players").select("id", count="exact").execute()
            total_players = players.count or 0
        except Exception:
            pass

        try:
            games = self.supabase.table("games").select("id", count="exact").execute()
            total_games = games.count or 0
        except Exception:
            pass

        try:
            seasons = self.supabase.table("seasons").select("id", count="exact").execute()
            total_seasons = seasons.count or 0
        except Exception:
            pass

        try:
            training = (
                self.supabase.table("model_training_log")
                .select("*")
                .order("training_started_at", desc=True)
                .limit(1)
                .execute()
            )
            if training.data:
                model_last_trained = training.data[0]["training_completed_at"]
        except Exception:
            pass

        # Fallback: check saved model files on disk
        if not model_last_trained:
            from pathlib import Path
            smart_dir = Path(__file__).parent.parent / "models" / "artifacts" / "smart"
            if smart_dir.exists():
                joblib_files = list(smart_dir.glob("*.joblib"))
                if joblib_files:
                    latest_mtime = max(f.stat().st_mtime for f in joblib_files)
                    model_last_trained = datetime.fromtimestamp(latest_mtime).isoformat()

        try:
            sportsdataio_usage = await self.get_api_usage("sportsdataio")
            api_quota_used = sportsdataio_usage.get("requests_made", 0)
        except Exception:
            pass

        return {
            "last_refresh": None,
            "total_players": total_players,
            "total_games": total_games,
            "total_seasons": total_seasons,
            "api_quota_used": api_quota_used,
            "api_quota_limit": 1000,
            "model_last_trained": model_last_trained,
        }

    # ---- Data Refresh ----

    async def refresh_today_data(self) -> dict:
        """Refresh today's games, stats, injuries, and odds."""
        results = {"games": 0, "stats": 0, "injuries": 0, "odds": 0, "errors": []}

        try:
            # Refresh today's games
            today_str = date.today().isoformat()
            games = await self.sports_api.get_games_by_date(today_str)
            await self.track_api_usage("sportsdataio", "games_by_date")

            for game in games:
                try:
                    self.supabase.table("games").upsert({
                        "id": str(game.get("GameID", "")),
                        "game_date": game.get("Day", ""),
                        "home_team_id": str(game.get("HomeTeamID", "")),
                        "away_team_id": str(game.get("AwayTeamID", "")),
                        "home_score": game.get("HomeTeamScore"),
                        "away_score": game.get("AwayTeamScore"),
                        "status": game.get("Status", "scheduled"),
                        "over_under": game.get("OverUnder"),
                        "spread": game.get("PointSpread"),
                    }).execute()
                    results["games"] += 1
                except Exception as e:
                    results["errors"].append(f"Game {game.get('GameID')}: {str(e)}")

            # Refresh injuries
            injuries = await self.sports_api.get_injuries()
            await self.track_api_usage("sportsdataio", "injuries")

            for injury in injuries:
                try:
                    self.supabase.table("injury_reports").upsert({
                        "player_id": str(injury.get("PlayerID", "")),
                        "status": injury.get("Status", "unknown"),
                        "description": injury.get("Description", ""),
                        "source": "official",
                    }).execute()
                    results["injuries"] += 1
                except Exception as e:
                    results["errors"].append(f"Injury: {str(e)}")

            # Scrape additional injury news
            try:
                injury_news = await self.injury_scraper.get_all_injury_news()
                logger.info(
                    "Injury news scraped",
                    twitter=len(injury_news.get("twitter", [])),
                    rss=len(injury_news.get("rss", [])),
                )
            except Exception as e:
                results["errors"].append(f"Injury scraping: {str(e)}")

            logger.info("Data refresh completed", **results)
            return results

        except Exception as e:
            logger.error("Data refresh failed", error=str(e))
            results["errors"].append(str(e))
            return results

    # ---- Model Retraining ----

    async def retrain_models(self) -> dict:
        """Retrain all SmartPredictor models with latest enriched data."""
        from app.services.nba_data import get_nba_data
        from app.services.smart_predictor import SmartPredictor

        training_id = None
        try:
            # Log training start
            log_result = self.supabase.table("model_training_log").insert({
                "model_name": "smart_predictor",
                "status": "running",
            }).execute()
            if log_result.data:
                training_id = log_result.data[0]["id"]

            # Build enriched feature set from SportsDataIO + BDL
            nba = get_nba_data()
            feature_data = nba.build_full_feature_set()
            enriched_players = feature_data.get("players", [])

            if not enriched_players:
                raise ValueError("No enriched player data available. Refresh data first.")

            logger.info(f"Retraining SmartPredictor with {len(enriched_players)} enriched players")

            # Train all 7 prop types
            predictor = SmartPredictor()
            all_metrics = predictor.train_all_props(enriched_players)

            # Update training log
            if training_id:
                self.supabase.table("model_training_log").update({
                    "status": "completed",
                    "training_completed_at": datetime.utcnow().isoformat(),
                    "samples_used": len(enriched_players),
                    "parameters": {"model": "smart_predictor", "prop_types": list(all_metrics.keys())},
                }).eq("id", training_id).execute()

            # Update last trained timestamp in settings
            self.supabase.table("app_settings").update({
                "updated_at": datetime.utcnow().isoformat(),
            }).eq("id", "default").execute()

            logger.info("SmartPredictor retraining completed", metrics=all_metrics)
            return {"status": "completed", "metrics": all_metrics}

        except Exception as e:
            logger.error("Model retraining failed", error=str(e))
            if training_id:
                self.supabase.table("model_training_log").update({
                    "status": "failed",
                    "error_message": str(e),
                }).eq("id", training_id).execute()
            return {"status": "failed", "error": str(e)}

    # ---- Backup ----

    async def create_backup(self, format: str = "json") -> dict:
        """Create a data backup."""
        try:
            tables = [
                "players", "teams", "games", "player_game_stats",
                "predictions", "bets", "parlays", "app_settings",
                "model_presets",
            ]
            backup_data = {}
            for table in tables:
                result = self.supabase.table(table).select("*").execute()
                backup_data[table] = result.data or []

            # Log backup
            self.supabase.table("backup_log").insert({
                "backup_type": "manual",
                "format": format,
                "status": "completed",
            }).execute()

            logger.info("Backup created", format=format, tables=len(tables))
            return {"status": "completed", "tables": len(tables), "data": backup_data}

        except Exception as e:
            logger.error("Backup failed", error=str(e))
            return {"status": "failed", "error": str(e)}


_manager: DataManager | None = None


def get_data_manager() -> DataManager:
    global _manager
    if _manager is None:
        _manager = DataManager()
    return _manager
