"""
NBA News & Sentiment Service.

Aggregates news from:
  1. RSS feeds (ESPN, CBS Sports, Bleacher Report) — free, no key needed
  2. NewsAPI (newsapi.org) — free tier, 100 req/day

Produces per-player sentiment scores that feed into the prediction model:
  - news_sentiment: -1.0 (very negative) to +1.0 (very positive)
  - news_volume: number of recent mentions (more mentions = bigger story)
  - injury_mentioned: 1 if injury keywords found, 0 otherwise
  - rest_mentioned: 1 if rest/load management keywords found
  - trade_mentioned: 1 if trade rumors found
  - hot_streak_mentioned: 1 if hot streak / career high keywords found
"""
from __future__ import annotations

import re
import httpx
import feedparser
from datetime import datetime, timedelta
from collections import defaultdict

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

# ── RSS Feeds (free, no API key) ──────────────────────────────────────
NBA_RSS_FEEDS = [
    # ESPN NBA
    "https://www.espn.com/espn/rss/nba/news",
    # CBS Sports NBA
    "https://www.cbssports.com/rss/headlines/nba/",
    # Bleacher Report NBA
    "https://bleacherreport.com/articles/feed?tag_id=19",
    # Yahoo Sports NBA
    "https://sports.yahoo.com/nba/rss/",
    # Rotoworld / NBC Sports Edge (player news)
    "https://www.nbcsportsedge.com/basketball/nba/player-news?rss=true",
]

# ── Sentiment keyword dictionaries ────────────────────────────────────
POSITIVE_KEYWORDS = [
    "career high", "career-high", "breakout", "dominant", "erupts",
    "explodes", "hot streak", "on fire", "mvp", "all-star",
    "triple-double", "double-double", "season high", "season-high",
    "upgraded", "probable", "expected to play", "cleared to play",
    "returns to lineup", "back in lineup", "strong performance",
    "impressive", "outstanding", "stellar", "clutch",
]

NEGATIVE_KEYWORDS = [
    "injury", "injured", "out for", "ruled out", "doubtful",
    "questionable", "sidelined", "surgery", "sprain", "strain",
    "fracture", "concussion", "torn", "miss", "absence",
    "load management", "rest", "dnp", "did not play",
    "struggling", "slump", "cold streak", "worst",
    "trade", "traded", "trade rumors", "trade talks",
    "suspension", "suspended", "fine", "fined",
    "ejected", "ejection", "technical",
]

INJURY_KEYWORDS = [
    "injury", "injured", "out for", "ruled out", "doubtful",
    "questionable", "sidelined", "surgery", "sprain", "strain",
    "fracture", "concussion", "torn", "hamstring", "ankle",
    "knee", "shoulder", "back", "hip", "calf", "foot",
    "achilles", "acl", "mcl",
]

REST_KEYWORDS = [
    "load management", "rest", "resting", "dnp", "night off",
    "maintenance", "veteran rest",
]

TRADE_KEYWORDS = [
    "trade", "traded", "trade rumors", "trade talks", "deal",
    "acquisition", "waived", "released", "buyout",
]

HOT_KEYWORDS = [
    "career high", "career-high", "season high", "season-high",
    "hot streak", "on fire", "erupts", "explodes", "breakout",
    "triple-double", "40 points", "50 points", "dominant",
]


def _score_text(text: str) -> dict:
    """Score a piece of text for sentiment and keyword flags."""
    text_lower = text.lower()

    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)

    # Sentiment: -1 to +1
    total = pos_count + neg_count
    if total == 0:
        sentiment = 0.0
    else:
        sentiment = (pos_count - neg_count) / total

    return {
        "sentiment": round(sentiment, 2),
        "positive_hits": pos_count,
        "negative_hits": neg_count,
        "injury_mentioned": 1 if any(kw in text_lower for kw in INJURY_KEYWORDS) else 0,
        "rest_mentioned": 1 if any(kw in text_lower for kw in REST_KEYWORDS) else 0,
        "trade_mentioned": 1 if any(kw in text_lower for kw in TRADE_KEYWORDS) else 0,
        "hot_streak_mentioned": 1 if any(kw in text_lower for kw in HOT_KEYWORDS) else 0,
    }


def _extract_player_names(text: str, known_players: set[str]) -> list[str]:
    """Find which known player names appear in the text."""
    text_lower = text.lower()
    found = []
    for name in known_players:
        # Match full name or last name (for common references)
        if name in text_lower:
            found.append(name)
        else:
            # Try last name only (e.g. "James" for "lebron james")
            parts = name.split()
            if len(parts) >= 2:
                last = parts[-1]
                # Only match last name if it's 4+ chars to avoid false positives
                if len(last) >= 4 and re.search(r'\b' + re.escape(last) + r'\b', text_lower):
                    found.append(name)
    return found


class NewsSentimentService:
    def __init__(self):
        settings = get_settings()
        self.newsapi_key = getattr(settings, 'newsapi_key', '') or ''
        self.client = httpx.Client(timeout=15)

    def fetch_rss_articles(self, max_age_hours: int = 48) -> list[dict]:
        """Fetch recent articles from NBA RSS feeds."""
        articles = []
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

        for feed_url in NBA_RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:  # Max 20 per feed
                    # Parse date
                    published = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published = datetime(*entry.updated_parsed[:6])

                    # Skip old articles
                    if published and published < cutoff:
                        continue

                    title = getattr(entry, 'title', '') or ''
                    summary = getattr(entry, 'summary', '') or ''
                    text = f"{title} {summary}"

                    articles.append({
                        "title": title,
                        "summary": summary,
                        "text": text,
                        "source": feed_url.split('/')[2],
                        "published": published.isoformat() if published else "",
                        "url": getattr(entry, 'link', ''),
                    })
            except Exception as e:
                logger.debug(f"RSS feed error for {feed_url}: {e}")

        logger.info(f"RSS: fetched {len(articles)} recent NBA articles")
        return articles

    def fetch_newsapi_articles(self, max_age_hours: int = 48) -> list[dict]:
        """Fetch NBA news from NewsAPI."""
        if not self.newsapi_key:
            logger.debug("NewsAPI key not configured, skipping")
            return []

        articles = []
        from_date = (datetime.utcnow() - timedelta(hours=max_age_hours)).strftime("%Y-%m-%dT%H:%M:%S")

        try:
            resp = self.client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": "NBA basketball player",
                    "from": from_date,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 50,
                    "apiKey": self.newsapi_key,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            for art in data.get("articles", []):
                title = art.get("title", "") or ""
                desc = art.get("description", "") or ""
                content = art.get("content", "") or ""
                text = f"{title} {desc} {content}"

                articles.append({
                    "title": title,
                    "summary": desc,
                    "text": text,
                    "source": art.get("source", {}).get("name", ""),
                    "published": art.get("publishedAt", ""),
                    "url": art.get("url", ""),
                })

            logger.info(f"NewsAPI: fetched {len(articles)} NBA articles")
        except Exception as e:
            logger.warning(f"NewsAPI error: {e}")

        return articles

    def build_player_sentiment(
        self, known_players: set[str], max_age_hours: int = 48
    ) -> dict[str, dict]:
        """
        Build per-player sentiment features from all news sources.

        Args:
            known_players: Set of lowercase player names to look for
            max_age_hours: How far back to look for articles

        Returns:
            {player_name: {news_sentiment, news_volume, injury_mentioned, ...}}
        """
        # Fetch from all sources
        rss_articles = self.fetch_rss_articles(max_age_hours)
        newsapi_articles = self.fetch_newsapi_articles(max_age_hours)
        all_articles = rss_articles + newsapi_articles

        logger.info(f"Total articles to scan: {len(all_articles)} ({len(rss_articles)} RSS + {len(newsapi_articles)} NewsAPI)")

        # Score each article and attribute to players
        player_scores: dict[str, list[dict]] = defaultdict(list)

        for article in all_articles:
            text = article.get("text", "")
            if not text:
                continue

            mentioned_players = _extract_player_names(text, known_players)
            if not mentioned_players:
                continue

            score = _score_text(text)

            for player_name in mentioned_players:
                player_scores[player_name].append(score)

        # Aggregate per-player
        result: dict[str, dict] = {}
        for player_name, scores in player_scores.items():
            n = len(scores)
            avg_sentiment = sum(s["sentiment"] for s in scores) / n
            injury = max(s["injury_mentioned"] for s in scores)
            rest = max(s["rest_mentioned"] for s in scores)
            trade = max(s["trade_mentioned"] for s in scores)
            hot = max(s["hot_streak_mentioned"] for s in scores)

            result[player_name] = {
                "news_sentiment": round(avg_sentiment, 2),
                "news_volume": n,
                "injury_mentioned": injury,
                "rest_mentioned": rest,
                "trade_mentioned": trade,
                "hot_streak_mentioned": hot,
            }

        # Players with no mentions get neutral defaults
        for name in known_players:
            if name not in result:
                result[name] = {
                    "news_sentiment": 0.0,
                    "news_volume": 0,
                    "injury_mentioned": 0,
                    "rest_mentioned": 0,
                    "trade_mentioned": 0,
                    "hot_streak_mentioned": 0,
                }

        mentioned_count = sum(1 for v in result.values() if v["news_volume"] > 0)
        logger.info(f"News sentiment: {mentioned_count}/{len(known_players)} players mentioned in news")
        return result


_service: NewsSentimentService | None = None


def get_news_sentiment() -> NewsSentimentService:
    global _service
    if _service is None:
        _service = NewsSentimentService()
    return _service
