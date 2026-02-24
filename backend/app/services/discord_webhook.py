import os
import logging

import httpx

logger = logging.getLogger(__name__)

async def send_discord_notification(title: str, message: str, color: int = 0x3498db) -> None:
    """Send a notification to Discord via Webhook."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return

    payload = {
        "embeds": [
            {
                "title": title,
                "description": message,
                "color": color
            }
        ]
    }

    try:
        timeout = httpx.Timeout(5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(webhook_url, json=payload)
            if response.status_code not in (200, 204):
                logger.warning(f"Failed to send Discord webhook: {response.status_code}")
    except Exception as e:
        logger.warning(f"Error sending Discord webhook: {e}")
