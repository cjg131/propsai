"""
Server-Sent Events (SSE) event bus for real-time dashboard streaming.

Publishes agent events (trades, settlements, position updates, log entries)
to connected SSE clients without polling.

Usage:
    from app.services.event_bus import get_event_bus

    bus = get_event_bus()
    bus.publish("trade", {"ticker": "KXHIGHTNYC...", "side": "yes", ...})

    # In SSE endpoint:
    async for event in bus.subscribe():
        yield f"data: {json.dumps(event)}\n\n"
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)


class EventBus:
    """Simple async pub/sub for SSE streaming."""

    def __init__(self, max_history: int = 50) -> None:
        self._subscribers: list[asyncio.Queue] = []
        self._history: list[dict[str, Any]] = []
        self._max_history = max_history

    def publish(self, event_type: str, data: dict[str, Any]) -> None:
        """Publish an event to all subscribers."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        # Store in history ring buffer
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Push to all subscriber queues
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        # Remove dead/full queues
        for q in dead:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    async def subscribe(self, include_history: bool = True) -> AsyncGenerator[dict[str, Any], None]:
        """Subscribe to the event stream. Yields events as they arrive."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)

        try:
            # Send recent history first so the client catches up
            if include_history:
                for event in self._history[-20:]:
                    yield event

            # Then stream live events
            while True:
                event = await q.get()
                yield event
        finally:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def get_recent(self, count: int = 20) -> list[dict[str, Any]]:
        """Get recent events (for initial page load)."""
        return self._history[-count:]

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Singleton
_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus
