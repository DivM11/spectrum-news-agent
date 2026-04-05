from __future__ import annotations

import pytest

from src.event_store import create_event_store
from src.event_store.base import EventStore
from src.event_store.null import NullEventStore
from src.event_store.postgres import PostgresEventStore
from src.event_store.sqlite import SQLiteEventStore

def test_null_store_satisfies_protocol() -> None:
    store = NullEventStore()
    assert isinstance(store, EventStore)


def test_sqlite_store_satisfies_protocol(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "test.db"))
    assert isinstance(store, EventStore)
    store.close()


def test_create_event_store_sqlite_backend(tmp_path) -> None:
    store = create_event_store({"event_store": {"enabled": True, "backend": "sqlite", "sqlite": {"db_path": str(tmp_path / "test.db")}}})
    assert isinstance(store, SQLiteEventStore)
    store.close()


def test_create_event_store_postgres_requires_dsn() -> None:
    with pytest.raises(ValueError):
        create_event_store({"event_store": {"enabled": True, "backend": "postgres", "postgres": {}}})


def test_create_event_store_postgres_reads_dsn_env(monkeypatch) -> None:
    monkeypatch.setenv("EVENT_STORE_DSN", "postgresql://u:p@localhost:5432/db")

    store = create_event_store(
        {
            "event_store": {
                "enabled": True,
                "backend": "postgres",
                "postgres": {"dsn_env_var": "EVENT_STORE_DSN"},
            }
        }
    )

    assert isinstance(store, PostgresEventStore)
    store.close()
