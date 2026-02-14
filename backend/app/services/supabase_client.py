from __future__ import annotations
from supabase import create_client, Client
from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

_client: Client | None = None


def get_supabase() -> Client:
    global _client
    if _client is None:
        settings = get_settings()
        _client = create_client(settings.supabase_url, settings.supabase_key)
        logger.info("Supabase client initialized")
    return _client
