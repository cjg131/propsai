from __future__ import annotations
import feedparser
import tweepy
from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

NBA_REPORTERS = [
    "wojespn",
    "ShamsCharania",
    "ChrisBHaynes",
    "MarcJSpears",
    "KeithSmithNBA",
]

RSS_FEEDS = [
    "https://www.espn.com/espn/rss/nba/news",
    "https://www.cbssports.com/rss/headlines/nba/",
]


class InjuryScraper:
    def __init__(self):
        settings = get_settings()
        self.twitter_client: tweepy.Client | None = None
        if settings.twitter_bearer_token:
            self.twitter_client = tweepy.Client(
                bearer_token=settings.twitter_bearer_token
            )
            logger.info("Twitter client initialized for injury scraping")

    async def get_twitter_injury_news(self, max_results: int = 20) -> list[dict]:
        """Fetch recent injury-related tweets from NBA reporters."""
        if not self.twitter_client:
            logger.warning("Twitter client not configured, skipping Twitter scraping")
            return []

        results = []
        for reporter in NBA_REPORTERS:
            try:
                tweets = self.twitter_client.get_users_tweets(
                    id=reporter,
                    max_results=min(max_results, 10),
                    tweet_fields=["created_at", "text"],
                )
                if tweets.data:
                    for tweet in tweets.data:
                        text_lower = tweet.text.lower()
                        injury_keywords = [
                            "out", "injury", "injured", "questionable",
                            "doubtful", "day-to-day", "ruled out", "miss",
                            "sidelined", "surgery", "sprain", "strain",
                            "concussion", "rest", "load management",
                        ]
                        if any(kw in text_lower for kw in injury_keywords):
                            results.append({
                                "source": f"twitter:{reporter}",
                                "text": tweet.text,
                                "created_at": str(tweet.created_at),
                            })
            except Exception as e:
                logger.error(
                    "Twitter scraping error",
                    reporter=reporter,
                    error=str(e),
                )
        return results

    async def get_rss_injury_news(self) -> list[dict]:
        """Fetch injury news from RSS feeds."""
        results = []
        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    text_lower = (entry.get("title", "") + entry.get("summary", "")).lower()
                    injury_keywords = [
                        "injury", "injured", "out", "questionable",
                        "doubtful", "miss", "sidelined",
                    ]
                    if any(kw in text_lower for kw in injury_keywords):
                        results.append({
                            "source": f"rss:{feed_url}",
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", ""),
                            "link": entry.get("link", ""),
                            "published": entry.get("published", ""),
                        })
            except Exception as e:
                logger.error("RSS scraping error", feed_url=feed_url, error=str(e))
        return results

    async def get_all_injury_news(self) -> dict:
        """Get injury news from all sources."""
        twitter_news = await self.get_twitter_injury_news()
        rss_news = await self.get_rss_injury_news()
        return {
            "twitter": twitter_news,
            "rss": rss_news,
            "total": len(twitter_news) + len(rss_news),
        }


_scraper: InjuryScraper | None = None


def get_injury_scraper() -> InjuryScraper:
    global _scraper
    if _scraper is None:
        _scraper = InjuryScraper()
    return _scraper
