from __future__ import annotations

import os

from src.event_store.base import EventStore
from src.event_store.models import (
    ArticleMetadataRecord,
    EventRecord,
    LLMCallRecord,
    ToolCallRecord,
)
from src.event_store.null import NullEventStore
from src.event_store.postgres import PostgresEventStore
from src.event_store.sqlite import SQLiteEventStore


def create_event_store(config: dict) -> EventStore:
    cfg = config.get("event_store", {})
    if not cfg.get("enabled", True):
        return NullEventStore()
    backend = str(cfg.get("backend", "sqlite")).lower()
    if backend == "sqlite":
        db_path: str = cfg.get("sqlite", {}).get("db_path", "data/events.db")
        return SQLiteEventStore(db_path)
    if backend == "postgres":
        postgres_cfg = cfg.get("postgres", {})
        dsn = postgres_cfg.get("dsn") or os.getenv(postgres_cfg.get("dsn_env_var", "EVENT_STORE_DSN"))
        if not dsn:
            raise ValueError("event_store.postgres.dsn or dsn_env_var must be configured for postgres backend")
        return PostgresEventStore(dsn)
    return NullEventStore()


__all__ = [
    "EventStore",
    "NullEventStore",
    "PostgresEventStore",
    "SQLiteEventStore",
    "EventRecord",
    "LLMCallRecord",
    "ToolCallRecord",
    "ArticleMetadataRecord",
    "create_event_store",
]
