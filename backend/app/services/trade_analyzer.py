"""
Post-Trade Analysis Agent ("OpenClaw").

After each settled trade, analyzes wins and losses using GPT-5.2 to:
  1. Identify why losing trades failed
  2. Extract recurring patterns from losses
  3. Identify what's working from winning trades
  4. Store analysis for adaptive threshold tuning

Usage:
    analyzer = TradeAnalyzer()
    analyzer.init_db()
    await analyzer.analyze_trade(trade_dict, market_result="no")
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "trading_engine.db"

# Rate limiting: max analyses per hour
MAX_ANALYSES_PER_HOUR = 10
MAX_THESES_PER_HOUR = 30  # Theses are cheaper — allow more
MODEL = "gpt-5.2"
FALLBACK_MODEL = "gpt-4o-mini"


class TradeAnalyzer:
    """GPT-powered post-trade analysis engine."""

    def __init__(self) -> None:
        settings = get_settings()
        api_key = getattr(settings, "openai_api_key", "")
        self.client = AsyncOpenAI(api_key=api_key) if api_key else None
        self._analysis_timestamps: list[float] = []
        self._thesis_timestamps: list[float] = []
        self._init_db()

    def _init_db(self) -> None:
        """Create trade_reviews table if it doesn't exist."""
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS trade_reviews (
                id TEXT PRIMARY KEY,
                trade_id TEXT NOT NULL,
                strategy TEXT NOT NULL,
                ticker TEXT NOT NULL,
                outcome TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price_cents INTEGER DEFAULT 0,
                edge REAL DEFAULT 0,
                confidence REAL DEFAULT 0,
                signal_source TEXT DEFAULT '',
                signal_snapshot TEXT DEFAULT '',
                analysis TEXT DEFAULT '',
                patterns TEXT DEFAULT '',
                model_used TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _rate_limited(self) -> bool:
        """Check if we've exceeded the hourly analysis limit."""
        now = time.time()
        cutoff = now - 3600
        self._analysis_timestamps = [t for t in self._analysis_timestamps if t > cutoff]
        return len(self._analysis_timestamps) >= MAX_ANALYSES_PER_HOUR

    def _thesis_rate_limited(self) -> bool:
        """Check if we've exceeded the hourly thesis generation limit."""
        now = time.time()
        cutoff = now - 3600
        self._thesis_timestamps = [t for t in self._thesis_timestamps if t > cutoff]
        return len(self._thesis_timestamps) >= MAX_THESES_PER_HOUR

    async def generate_thesis(self, candidate: dict[str, Any]) -> str:
        """
        Generate a natural language pre-trade thesis using GPT-5.2.
        Called before a trade is placed. Returns a 2-3 sentence thesis string,
        or empty string if unavailable (rate limited, no API key, etc.).

        The thesis explains WHY this trade is being placed based on all signal data.
        """
        if not self.client or self._thesis_rate_limited():
            return ""

        strategy = candidate.get("strategy", "")
        title = candidate.get("title", "")
        side = candidate.get("side", "yes")
        edge = candidate.get("edge", 0)
        confidence = candidate.get("confidence", 0)
        our_prob = candidate.get("our_prob", 0)
        kalshi_prob = candidate.get("kalshi_prob", 0)
        signal_source = candidate.get("signal_source", "")
        price_cents = candidate.get("price_cents", 0)

        # Build signal detail string from candidate metadata
        signal_details: list[str] = []

        if strategy == "finance":
            fin_index = candidate.get("finance_index", "SP500")
            signal_details += [
                f"Index: {fin_index}",
                f"Signal source: {signal_source}",
                f"Our probability: {our_prob:.1%}",
                f"Kalshi implied: {kalshi_prob:.1%}",
                f"Edge: {edge:.1%}",
                f"Confidence: {confidence:.1%}",
                f"Market price: {price_cents}c",
                f"Side: {side.upper()} (betting market resolves {side})",
            ]
        elif strategy == "weather":
            signal_details += [
                f"Signal source: {signal_source}",
                f"Our probability: {our_prob:.1%}",
                f"Kalshi implied: {kalshi_prob:.1%}",
                f"Edge: {edge:.1%}",
                f"Confidence: {confidence:.1%} (forecast source agreement)",
                f"Side: {side.upper()}",
            ]
        elif strategy in ("sports", "nba_props"):
            signal_details += [
                f"Signal source: {signal_source}",
                f"Our probability: {our_prob:.1%}",
                f"Sharp book implied: {kalshi_prob:.1%}",
                f"Edge: {edge:.1%}",
                f"Confidence: {confidence:.1%}",
                f"Side: {side.upper()}",
            ]
        elif strategy == "crypto":
            signal_details += [
                f"Signal source: {signal_source}",
                f"Our probability (p_up): {our_prob:.1%}",
                f"Kalshi implied: {kalshi_prob:.1%}",
                f"Edge: {edge:.1%}",
                f"Confidence: {confidence:.1%}",
                f"Side: {side.upper()}",
            ]
        elif strategy == "econ":
            signal_details += [
                f"Signal source: {signal_source}",
                f"Our probability: {our_prob:.1%}",
                f"Kalshi implied: {kalshi_prob:.1%}",
                f"Edge: {edge:.1%}",
                f"Confidence: {confidence:.1%}",
                f"Side: {side.upper()}",
            ]

        signal_block = "\n".join(signal_details)

        prompt = f"""You are an automated Kalshi prediction market trading bot. You are about to place the following trade. Write a 2-3 sentence thesis explaining WHY this trade is being placed based on the signal data.

Market: {title}
Strategy: {strategy}

Signal Data:
{signal_block}

Write the thesis in plain English as if explaining to a human investor. Be specific about what the data shows and why it creates an edge. Do NOT use phrases like "the model predicts" — write as if you are the trading system explaining your reasoning. Keep it under 60 words."""

        try:
            model = MODEL
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a quantitative trading system explaining your trade decisions. "
                                "Be concise, specific, and data-driven. Write in first person as the system."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=120,
                )
            except Exception:
                model = FALLBACK_MODEL
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a quantitative trading system explaining your trade decisions. Be concise and data-driven.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=120,
                )

            self._thesis_timestamps.append(time.time())
            return (response.choices[0].message.content or "").strip()

        except Exception as e:
            logger.debug("Thesis generation failed", error=str(e))
            return ""

    def _get_trade_context(self, trade: dict[str, Any]) -> str:
        """Build context string from trade data for the LLM prompt."""
        parts = [
            f"Strategy: {trade.get('strategy', 'unknown')}",
            f"Ticker: {trade.get('ticker', '')}",
            f"Market: {trade.get('market_title', '')}",
            f"Side: {trade.get('side', '')}",
            f"Action: {trade.get('action', '')}",
            f"Contracts: {trade.get('count', 0)}",
            f"Entry Price: {trade.get('price_cents', 0)}c",
            f"Cost: ${trade.get('cost', 0):.2f}",
            f"Our Probability: {trade.get('our_prob', 0):.1%}",
            f"Kalshi Implied: {trade.get('kalshi_prob', 0):.1%}",
            f"Edge: {trade.get('edge', 0):.1%}",
            f"Signal Source: {trade.get('signal_source', '')}",
            f"Notes: {trade.get('notes', '')}",
            f"P&L: ${trade.get('pnl', 0):+.2f}",
            f"Result: {trade.get('result', '')}",
        ]
        thesis = trade.get("thesis", "")
        if thesis:
            parts.insert(3, f"Original Thesis: {thesis}")
        return "\n".join(parts)

    async def analyze_trade(
        self,
        trade: dict[str, Any],
        market_result: str = "",
    ) -> dict[str, Any] | None:
        """
        Analyze a settled trade using GPT-5.2.
        Returns analysis dict or None if skipped.
        """
        if not self.client:
            logger.debug("Trade analyzer: no OpenAI key configured")
            return None

        if self._rate_limited():
            logger.debug("Trade analyzer: rate limited")
            return None

        trade_id = trade.get("id", "")
        outcome = trade.get("result", market_result)
        pnl = trade.get("pnl", 0)
        is_loss = pnl < 0 or outcome == "loss"

        # Always analyze losses, sample 30% of wins
        if not is_loss:
            import random
            if random.random() > 0.30:
                return None

        context = self._get_trade_context(trade)

        if is_loss:
            prompt = f"""Analyze this LOSING trade from an automated Kalshi prediction market trading bot.

{context}

Provide a concise analysis (under 200 words):
1. Most likely reason this trade lost
2. What signal or data was missing/misinterpreted
3. One specific pattern to watch for (e.g., "low_volume_market", "counter_trend", "stale_signal")
4. Suggested threshold adjustment (tighter edge? higher confidence?)

Format the pattern as a single snake_case tag on its own line starting with "PATTERN: "."""
        else:
            prompt = f"""Analyze this WINNING trade from an automated Kalshi prediction market trading bot.

{context}

Provide a concise analysis (under 150 words):
1. What the signal got right
2. Key factors that made this trade successful
3. One pattern tag for what worked (e.g., "strong_consensus", "high_volume_edge")

Format the pattern as a single snake_case tag on its own line starting with "PATTERN: "."""

        try:
            model = MODEL
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert quantitative trading analyst reviewing automated trades "
                                "on Kalshi prediction markets. Be specific, data-driven, and actionable. "
                                "Focus on what the trading bot can learn to improve future performance."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=400,
                )
            except Exception:
                model = FALLBACK_MODEL
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are an expert quantitative trading analyst reviewing automated trades "
                                "on Kalshi prediction markets. Be specific, data-driven, and actionable."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=400,
                )

            analysis_text = response.choices[0].message.content or ""

            # Extract pattern tag
            patterns = []
            for line in analysis_text.split("\n"):
                if line.strip().upper().startswith("PATTERN:"):
                    tag = line.split(":", 1)[1].strip().lower().replace(" ", "_")
                    patterns.append(tag)

            self._analysis_timestamps.append(time.time())

            # Store in DB
            review_id = str(uuid.uuid4())[:12]
            review = {
                "id": review_id,
                "trade_id": trade_id,
                "strategy": trade.get("strategy", ""),
                "ticker": trade.get("ticker", ""),
                "outcome": "loss" if is_loss else "win",
                "side": trade.get("side", ""),
                "entry_price_cents": trade.get("price_cents", 0),
                "edge": trade.get("edge", 0),
                "confidence": 0,
                "signal_source": trade.get("signal_source", ""),
                "signal_snapshot": json.dumps({
                    "our_prob": trade.get("our_prob", 0),
                    "kalshi_prob": trade.get("kalshi_prob", 0),
                    "notes": trade.get("notes", ""),
                }),
                "analysis": analysis_text,
                "patterns": json.dumps(patterns),
                "model_used": model,
                "created_at": datetime.now(UTC).isoformat(),
            }

            conn = sqlite3.connect(str(DB_PATH))
            c = conn.cursor()
            c.execute(
                """INSERT INTO trade_reviews
                (id, trade_id, strategy, ticker, outcome, side, entry_price_cents,
                 edge, confidence, signal_source, signal_snapshot, analysis, patterns,
                 model_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    review_id, trade_id, review["strategy"], review["ticker"],
                    review["outcome"], review["side"], review["entry_price_cents"],
                    review["edge"], review["confidence"], review["signal_source"],
                    review["signal_snapshot"], review["analysis"], review["patterns"],
                    review["model_used"], review["created_at"],
                ),
            )
            conn.commit()
            conn.close()

            logger.info(
                "Trade analyzed",
                trade_id=trade_id,
                outcome=review["outcome"],
                patterns=patterns,
                model=model,
            )
            return review

        except Exception as e:
            logger.warning("Trade analysis failed", trade_id=trade_id, error=str(e))
            return None

    def get_recent_reviews(
        self, strategy: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Get recent trade reviews."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        if strategy:
            c.execute(
                "SELECT * FROM trade_reviews WHERE strategy = ? ORDER BY created_at DESC LIMIT ?",
                (strategy, limit),
            )
        else:
            c.execute(
                "SELECT * FROM trade_reviews ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )

        reviews = [dict(r) for r in c.fetchall()]
        conn.close()
        return reviews

    def get_pattern_summary(self) -> dict[str, Any]:
        """Get aggregated pattern counts from all reviews."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT outcome, patterns FROM trade_reviews")
        rows = c.fetchall()
        conn.close()

        loss_patterns: dict[str, int] = {}
        win_patterns: dict[str, int] = {}

        for row in rows:
            try:
                tags = json.loads(row["patterns"])
            except (json.JSONDecodeError, TypeError):
                continue

            target = loss_patterns if row["outcome"] == "loss" else win_patterns
            for tag in tags:
                target[tag] = target.get(tag, 0) + 1

        return {
            "loss_patterns": dict(sorted(loss_patterns.items(), key=lambda x: -x[1])),
            "win_patterns": dict(sorted(win_patterns.items(), key=lambda x: -x[1])),
            "total_reviews": len(rows),
            "losses_analyzed": sum(1 for r in rows if r["outcome"] == "loss"),
            "wins_analyzed": sum(1 for r in rows if r["outcome"] == "win"),
        }


# Singleton
_analyzer: TradeAnalyzer | None = None


def get_trade_analyzer() -> TradeAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = TradeAnalyzer()
    return _analyzer
