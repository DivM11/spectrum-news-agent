from __future__ import annotations

from src.event_store.base import EventStore
from src.event_store.models import (
    ArticleMetadataRecord,
    EventRecord,
    LLMCallRecord,
    ToolCallRecord,
)
from src.event_store.null import NullEventStore
from src.event_store.sqlite import SQLiteEventStore


def create_event_store(config: dict) -> EventStore:
    cfg = config.get("event_store", {})
    if not cfg.get("enabled", True):
        return NullEventStore()
    backend = cfg.get("backend", "sqlite")
    if backend == "sqlite":
        db_path: str = cfg.get("sqlite", {}).get("db_path", "data/events.db")
        return SQLiteEventStore(db_path)
    return NullEventStore()


__all__ = [
    "EventStore",
    "NullEventStore",
    "SQLiteEventStore",
    "EventRecord",
    "LLMCallRecord",
    "ToolCallRecord",
    "ArticleMetadataRecord",
    "create_event_store",
]
