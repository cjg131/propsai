"""
BallDontLie API client (All-Star tier).

Provides:
  - Per-game box scores (game logs) for rolling averages + matchup history
  - Player injuries
  - Active players list
"""
from __future__ import annotations

import json
import httpx
from datetime import date, timedelta
from pathlib import Path
from collections import defaultdict
from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

BDL_BASE = "https://api.balldontlie.io/v1"
CURRENT_SEASON = 2025  # BDL uses start year of season (2025-26 → 2025)
CACHE_DIR = Path(__file__).parent.parent / "cache"


class BallDontLieClient:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.balldontlie_api_key
        self.client = httpx.Client(timeout=30)
        self._season_cache: dict[int, list[dict]] = {}  # season -> data
        self._season_cache_date: dict[int, str] = {}  # season -> date string

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{BDL_BASE}/{path}"
        headers = {"Authorization": self.api_key}
        r = self.client.get(url, headers=headers, params=params or {})
        r.raise_for_status()
        return r.json()

    def _get_all_pages(self, path: str, params: dict | None = None, max_pages: int = 10) -> list[dict]:
        """
        Fetch all pages of a paginated endpoint with robust rate-limit handling.
        - 0.5s between requests (safe for All-Star tier)
        - On 429: wait 10s, retry up to 3 times per page
        - Logs progress every 25 pages
        """
        import time
        params = dict(params or {})
        params.setdefault("per_page", 100)
        all_data = []
        for page_num in range(max_pages):
            resp = None
            for attempt in range(4):
                try:
                    resp = self._get(path, params)
                    break
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        wait = 10 * (attempt + 1)
                        logger.info(f"BDL rate limited on page {page_num} (attempt {attempt+1}), waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.warning(f"BDL error on page {page_num}: {e}")
                        if attempt == 3:
                            resp = None
                        else:
                            time.sleep(2)

            if resp is None:
                logger.warning(f"BDL: giving up on page {page_num} after retries, returning {len(all_data)} rows")
                break

            all_data.extend(resp.get("data", []))
            cursor = resp.get("meta", {}).get("next_cursor")
            if not cursor:
                break
            params["cursor"] = cursor

            if page_num % 25 == 24:
                logger.info(f"BDL: fetched {len(all_data)} rows through page {page_num+1}...")

            time.sleep(0.5)
        return all_data

    # ------------------------------------------------------------------
    # Raw data fetchers
    # ------------------------------------------------------------------

    def get_injuries(self) -> list[dict]:
        """Get current injury report for all players."""
        try:
            data = self._get_all_pages("injuries")
            logger.info(f"BDL: fetched {len(data)} injuries")
            return data
        except Exception as e:
            logger.warning(f"BDL injuries fetch failed: {e}")
            return []

    def get_player_game_logs(self, bdl_player_id: int, season: int = CURRENT_SEASON) -> list[dict]:
        """Get all game logs for a single player this season."""
        try:
            data = self._get_all_pages("stats", {
                "player_ids[]": bdl_player_id,
                "season": season,
                "start_date": f"{season}-10-01",
            })
            return data
        except Exception as e:
            logger.warning(f"BDL game logs for player {bdl_player_id} failed: {e}")
            return []

    def get_team_game_logs_by_date(self, game_date: str) -> list[dict]:
        """Get all player stats for games on a specific date."""
        try:
            data = self._get_all_pages("stats", {
                "dates[]": game_date,
            })
            return data
        except Exception as e:
            logger.warning(f"BDL stats for date {game_date} failed: {e}")
            return []

    def get_bulk_game_logs(self, bdl_player_ids: list[int], season: int = CURRENT_SEASON) -> dict[int, list[dict]]:
        """
        Fetch game logs for multiple players.
        Returns {bdl_player_id: [game_log_dicts]}.
        BDL allows multiple player_ids[] in one request.
        """
        all_logs: dict[int, list[dict]] = defaultdict(list)

        # BDL supports up to ~30 player IDs per request
        batch_size = 25
        for i in range(0, len(bdl_player_ids), batch_size):
            batch = bdl_player_ids[i:i + batch_size]
            params: dict = {
                "season": season,
                "start_date": f"{season}-10-01",
                "per_page": 100,
            }
            # Add multiple player_ids
            for pid in batch:
                params.setdefault("player_ids[]", [])
                if isinstance(params["player_ids[]"], list):
                    params["player_ids[]"].append(pid)
                else:
                    params["player_ids[]"] = [params["player_ids[]"], pid]

            try:
                data = self._get_all_pages("stats", params, max_pages=20)
                for row in data:
                    pid = row.get("player", {}).get("id")
                    if pid:
                        all_logs[pid].append(row)
            except Exception as e:
                logger.warning(f"BDL bulk game logs batch failed: {e}")

        logger.info(f"BDL: fetched game logs for {len(all_logs)} players ({sum(len(v) for v in all_logs.values())} total games)")
        return dict(all_logs)

    # ------------------------------------------------------------------
    # File cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, season: int) -> tuple[Path, Path]:
        """Return (data_path, meta_path) for a given season's cache."""
        return (
            CACHE_DIR / f"bdl_season_{season}.json",
            CACHE_DIR / f"bdl_meta_{season}.json",
        )

    def _load_file_cache(self, season: int = CURRENT_SEASON) -> tuple[list[dict], dict]:
        """Load season stats from disk cache. Returns (data, meta)."""
        data_path, meta_path = self._cache_path(season)
        try:
            if data_path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text())
                data = json.loads(data_path.read_text())
                logger.info(
                    f"BDL: loaded {len(data)} rows for season {season} from file cache "
                    f"(last_game_date={meta.get('last_game_date','?')}, fetched={meta.get('saved_at','?')})"
                )
                return data, meta
        except Exception as e:
            logger.warning(f"BDL file cache load failed for season {season}: {e}")
        return [], {}

    def _save_file_cache(self, data: list[dict], season: int = CURRENT_SEASON) -> None:
        """Persist season stats to disk with metadata."""
        data_path, meta_path = self._cache_path(season)
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Find the latest game date in the data
            last_game_date = ""
            for row in data:
                d = row.get("game", {}).get("date", "")[:10]
                if d > last_game_date:
                    last_game_date = d
            data_path.write_text(json.dumps(data))
            meta_path.write_text(json.dumps({
                "season": season,
                "last_game_date": last_game_date,
                "row_count": len(data),
                "saved_at": date.today().isoformat(),
            }))
            logger.info(f"BDL: saved {len(data)} rows for season {season} (through {last_game_date})")
        except Exception as e:
            logger.warning(f"BDL file cache save failed: {e}")

    def _is_cache_complete(self, meta: dict, season: int) -> bool:
        """Check if a season's cache looks complete (not truncated)."""
        row_count = meta.get("row_count", 0)
        if season < CURRENT_SEASON:
            # Past seasons: ~30k rows for a full 82-game season
            return row_count >= 20000
        else:
            # Current season: estimate based on how far into the season we are
            days_into_season = (date.today() - date(season, 10, 20)).days
            expected_games = min(days_into_season * 0.6, 82) * 15  # ~15 teams play per day
            expected_rows = expected_games * 26  # ~26 box scores per game
            # Cache is "complete" if it has at least 80% of expected rows
            return row_count >= expected_rows * 0.8

    # ------------------------------------------------------------------
    # Season stats fetching (current + historical)
    # ------------------------------------------------------------------

    def get_season_stats(self, season: int = CURRENT_SEASON, max_pages: int = 400) -> list[dict]:
        """
        Fetch ALL player box scores for a season. Three-tier caching:
          1. In-memory (instant, lost on restart)
          2. File on disk (survives restarts)
          3. API fetch with robust retry (full or incremental)

        For the 2025-26 season with 55 games played: ~28,500 rows, ~285 pages.
        """
        today_str = date.today().isoformat()

        # Tier 1: In-memory cache (same day, same season)
        cached = self._season_cache.get(season, [])
        if cached and self._season_cache_date.get(season) == today_str:
            logger.info(f"BDL: using in-memory cache for season {season} ({len(cached)} rows)")
            return cached

        # Tier 2: File cache — but validate it's not truncated
        file_data, meta = self._load_file_cache(season)
        if file_data and self._is_cache_complete(meta, season):
            last_game = meta.get("last_game_date", "")
            saved_at = meta.get("saved_at", "")

            # If cache was saved today, use it as-is
            if saved_at == today_str:
                self._season_cache[season] = file_data
                self._season_cache_date[season] = today_str
                return file_data

            # Incremental: fetch only games after the last game date in cache
            if last_game and season == CURRENT_SEASON:
                fetch_from = (date.fromisoformat(last_game) + timedelta(days=1)).isoformat()
                if fetch_from <= today_str:
                    logger.info(f"BDL: incremental fetch from {fetch_from} (cache has {len(file_data)} rows through {last_game})...")
                    try:
                        new_data = self._get_all_pages("stats", {
                            "dates[]": [],  # not used, start_date handles it
                            "seasons[]": season,
                            "start_date": fetch_from,
                        }, max_pages=50)
                        if new_data:
                            file_data.extend(new_data)
                            logger.info(f"BDL: added {len(new_data)} new rows (total: {len(file_data)})")
                    except Exception as e:
                        logger.warning(f"BDL incremental fetch failed: {e}")

                self._season_cache[season] = file_data
                self._season_cache_date[season] = today_str
                self._save_file_cache(file_data, season)
                return file_data

        # Tier 3: Full fetch — cache missing, truncated, or stale
        if file_data and not self._is_cache_complete(meta, season):
            logger.warning(
                f"BDL: cache for season {season} looks truncated "
                f"({meta.get('row_count', 0)} rows, last_game={meta.get('last_game_date', '?')}). "
                f"Doing full re-fetch..."
            )

        logger.info(f"BDL: full fetch for season {season} (up to {max_pages} pages)...")
        try:
            data = self._get_all_pages("stats", {
                "seasons[]": season,
                "start_date": f"{season}-10-01",
            }, max_pages=max_pages)
            logger.info(f"BDL: fetched {len(data)} rows for season {season}")
            if len(data) > 500:
                self._season_cache[season] = data
                self._season_cache_date[season] = today_str
                self._save_file_cache(data, season)
            return data
        except Exception as e:
            logger.warning(f"BDL season stats fetch failed: {e}")
            # Return whatever we have (even truncated cache is better than nothing)
            if file_data:
                self._season_cache[season] = file_data
                return file_data
            return self._season_cache.get(season, [])

    def get_multi_season_stats(self, seasons: list[int] | None = None) -> list[dict]:
        """
        Fetch box scores for multiple seasons (for historical training).
        Default: current season + 4 prior seasons.
        Each season is cached independently on disk.
        """
        if seasons is None:
            seasons = list(range(CURRENT_SEASON - 4, CURRENT_SEASON + 1))

        all_data: list[dict] = []
        for season in seasons:
            logger.info(f"BDL: loading season {season}...")
            data = self.get_season_stats(season=season, max_pages=400)
            all_data.extend(data)
            logger.info(f"BDL: season {season}: {len(data)} rows (running total: {len(all_data)})")

        logger.info(f"BDL: multi-season total: {len(all_data)} rows across {len(seasons)} seasons")
        return all_data

    def background_prefetch(self, seasons: list[int] | None = None) -> dict:
        """
        Slowly pre-fetch season data with generous rate limiting.
        Designed to run as a background job.
        Returns {season: row_count} for each season fetched.
        """
        if seasons is None:
            seasons = [CURRENT_SEASON]

        results = {}
        for season in seasons:
            logger.info(f"BDL: background pre-fetch for season {season}...")
            data = self.get_season_stats(season=season, max_pages=400)
            results[season] = len(data)
            logger.info(f"BDL: season {season} pre-fetch complete: {len(data)} rows")

        return results

    def search_player(self, name: str) -> dict | None:
        """Search for a player by name."""
        try:
            resp = self._get("players", {"search": name, "per_page": 5})
            data = resp.get("data", [])
            if data:
                return data[0]
        except Exception:
            pass
        return None

    def get_active_players(self) -> list[dict]:
        """Get all active NBA players."""
        try:
            data = self._get_all_pages("players/active")
            logger.info(f"BDL: {len(data)} active players")
            return data
        except Exception as e:
            logger.warning(f"BDL active players failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Derived feature builders
    # ------------------------------------------------------------------

    def build_player_game_log_features(self, game_logs: list[dict]) -> dict:
        """
        From a player's game logs, compute:
        - Rolling averages (last 5, 10, 20 games)
        - Recent form (last 3 vs season)
        - Home/away splits
        - Trend direction (improving or declining)
        """
        if not game_logs:
            return {}

        def parse_minutes(min_str: str) -> float:
            if not min_str or min_str == "0" or min_str == "":
                return 0.0
            try:
                if ":" in str(min_str):
                    parts = str(min_str).split(":")
                    return float(parts[0]) + float(parts[1]) / 60
                return float(min_str)
            except (ValueError, IndexError):
                return 0.0

        # Filter out DNP games (0 minutes or "00" minutes)
        # These are games where the player was on the roster but didn't play
        played_logs = [
            g for g in game_logs
            if parse_minutes(g.get("min", "0")) > 0
        ]

        if not played_logs:
            return {"game_log_count": 0}

        # Sort by date ascending
        logs = sorted(played_logs, key=lambda g: g.get("game", {}).get("date", ""))

        # Extract stat arrays (only from games actually played)
        pts = [g.get("pts", 0) or 0 for g in logs]
        reb = [g.get("reb", 0) or 0 for g in logs]
        ast = [g.get("ast", 0) or 0 for g in logs]
        stl = [g.get("stl", 0) or 0 for g in logs]
        blk = [g.get("blk", 0) or 0 for g in logs]
        tov = [g.get("turnover", 0) or 0 for g in logs]
        fg3m = [g.get("fg3m", 0) or 0 for g in logs]
        mins = [parse_minutes(g.get("min", "0")) for g in logs]
        fga = [g.get("fga", 0) or 0 for g in logs]
        fta = [g.get("fta", 0) or 0 for g in logs]

        n = len(logs)
        features = {"game_log_count": n}

        stat_map = {
            "pts": pts, "reb": reb, "ast": ast, "stl": stl,
            "blk": blk, "tov": tov, "fg3m": fg3m, "min": mins,
            "fga": fga, "fta": fta,
        }

        for stat_name, values in stat_map.items():
            if not values:
                continue
            season_avg = sum(values) / len(values)
            features[f"season_avg_{stat_name}"] = round(season_avg, 2)

            # Rolling averages
            for window in [3, 5, 10]:
                if n >= window:
                    recent = values[-window:]
                    features[f"last{window}_{stat_name}"] = round(sum(recent) / len(recent), 2)
                else:
                    features[f"last{window}_{stat_name}"] = round(season_avg, 2)

            # Trend: last 5 vs season average (positive = improving)
            if n >= 5:
                last5_avg = sum(values[-5:]) / 5
                features[f"trend_{stat_name}"] = round(last5_avg - season_avg, 2)
            else:
                features[f"trend_{stat_name}"] = 0.0

        # Home/away splits
        home_pts, away_pts = [], []
        home_reb, away_reb = [], []
        home_ast, away_ast = [], []
        home_min, away_min = [], []

        for g in logs:
            game = g.get("game", {})
            team_id = g.get("team", {}).get("id")
            is_home = team_id == game.get("home_team_id")
            p, r, a, m = g.get("pts", 0) or 0, g.get("reb", 0) or 0, g.get("ast", 0) or 0, parse_minutes(g.get("min", "0"))
            if is_home:
                home_pts.append(p); home_reb.append(r); home_ast.append(a); home_min.append(m)
            else:
                away_pts.append(p); away_reb.append(r); away_ast.append(a); away_min.append(m)

        features["home_avg_pts"] = round(sum(home_pts) / max(len(home_pts), 1), 1)
        features["away_avg_pts"] = round(sum(away_pts) / max(len(away_pts), 1), 1)
        features["home_avg_reb"] = round(sum(home_reb) / max(len(home_reb), 1), 1)
        features["away_avg_reb"] = round(sum(away_reb) / max(len(away_reb), 1), 1)
        features["home_avg_ast"] = round(sum(home_ast) / max(len(home_ast), 1), 1)
        features["away_avg_ast"] = round(sum(away_ast) / max(len(away_ast), 1), 1)
        features["home_avg_min"] = round(sum(home_min) / max(len(home_min), 1), 1)
        features["away_avg_min"] = round(sum(away_min) / max(len(away_min), 1), 1)

        # Consistency (std dev — lower = more consistent = more predictable)
        import statistics
        if n >= 5:
            features["std_pts"] = round(statistics.stdev(pts), 2)
            features["std_reb"] = round(statistics.stdev(reb), 2)
            features["std_ast"] = round(statistics.stdev(ast), 2)
        else:
            features["std_pts"] = 0.0
            features["std_reb"] = 0.0
            features["std_ast"] = 0.0

        return features

    def build_matchup_features(self, game_logs: list[dict], opponent_team_id: int) -> dict:
        """
        Compute how a player performs against a specific opponent.
        Uses game logs to find past games vs this team.
        """
        def _parse_min(m: str) -> float:
            if not m or m in ('0','00',''):
                return 0.0
            try:
                if ':' in str(m):
                    p = str(m).split(':')
                    return float(p[0]) + float(p[1]) / 60
                return float(m)
            except (ValueError, IndexError):
                return 0.0

        vs_games = [
            g for g in game_logs
            if (g.get("game", {}).get("home_team_id") == opponent_team_id
                or g.get("game", {}).get("visitor_team_id") == opponent_team_id)
            and _parse_min(g.get("min", "0")) > 0
        ]

        features = {"vs_opp_games": len(vs_games)}

        if not vs_games:
            return features

        features["vs_opp_avg_pts"] = round(sum(g.get("pts", 0) or 0 for g in vs_games) / len(vs_games), 1)
        features["vs_opp_avg_reb"] = round(sum(g.get("reb", 0) or 0 for g in vs_games) / len(vs_games), 1)
        features["vs_opp_avg_ast"] = round(sum(g.get("ast", 0) or 0 for g in vs_games) / len(vs_games), 1)
        features["vs_opp_avg_fg3m"] = round(sum(g.get("fg3m", 0) or 0 for g in vs_games) / len(vs_games), 1)
        features["vs_opp_avg_stl"] = round(sum(g.get("stl", 0) or 0 for g in vs_games) / len(vs_games), 1)
        features["vs_opp_avg_blk"] = round(sum(g.get("blk", 0) or 0 for g in vs_games) / len(vs_games), 1)

        return features

    def build_rest_features_from_logs(self, game_logs: list[dict], today: date | None = None) -> dict:
        """
        Compute rest features from actual game log dates.
        More accurate than SportsDataIO schedule-based detection.
        """
        if today is None:
            today = date.today()

        features = {
            "days_rest": 3,
            "is_b2b": 0,
            "is_3_in_4": 0,
            "games_last_7": 0,
            "games_last_14": 0,
        }

        if not game_logs:
            return features

        def _parse_min_r(m: str) -> float:
            if not m or m in ('0','00',''):
                return 0.0
            try:
                if ':' in str(m):
                    p = str(m).split(':')
                    return float(p[0]) + float(p[1]) / 60
                return float(m)
            except (ValueError, IndexError):
                return 0.0

        # Filter out DNP games and sort by date descending
        sorted_logs = sorted(
            [g for g in game_logs if _parse_min_r(g.get("min", "0")) > 0],
            key=lambda g: g.get("game", {}).get("date", ""),
            reverse=True,
        )

        # Parse dates
        game_dates = []
        for g in sorted_logs:
            d = g.get("game", {}).get("date", "")
            if d:
                try:
                    game_dates.append(date.fromisoformat(d[:10]))
                except ValueError:
                    pass

        if not game_dates:
            return features

        last_game = game_dates[0]
        days_rest = (today - last_game).days
        features["days_rest"] = days_rest
        features["is_b2b"] = 1 if days_rest <= 1 else 0

        # 3 games in 4 nights
        four_nights_ago = today - timedelta(days=4)
        games_in_4 = sum(1 for d in game_dates if d >= four_nights_ago)
        features["is_3_in_4"] = 1 if games_in_4 >= 3 else 0

        # Games in last 7 and 14 days (workload)
        seven_ago = today - timedelta(days=7)
        fourteen_ago = today - timedelta(days=14)
        features["games_last_7"] = sum(1 for d in game_dates if d >= seven_ago)
        features["games_last_14"] = sum(1 for d in game_dates if d >= fourteen_ago)

        return features


_client: BallDontLieClient | None = None


def get_balldontlie() -> BallDontLieClient:
    global _client
    if _client is None:
        _client = BallDontLieClient()
    return _client
