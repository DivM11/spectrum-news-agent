from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from src.event_store.base import EventStore
from src.event_store.models import (
    ArticleMetadataRecord,
    EventRecord,
    LLMCallRecord,
    ToolCallRecord,
)
from src.event_store.null import NullEventStore
from src.event_store.sqlite import SQLiteEventStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds")


def _event(session_id: str = "sess-1", run_id: str = "run-1") -> EventRecord:
    return EventRecord(
        id=str(uuid.uuid4()),
        session_id=session_id,
        run_id=run_id,
        timestamp=_ts(),
        event_type="test",
        schema_version=1,
        payload={"key": "value"},
    )


def _llm_call(session_id: str = "sess-1", run_id: str = "run-1") -> LLMCallRecord:
    return LLMCallRecord(
        id=str(uuid.uuid4()),
        session_id=session_id,
        run_id=run_id,
        timestamp=_ts(),
        schema_version=1,
        task="orchestrator",
        model="test-model",
        prompt_tokens=100,
        completion_tokens=50,
        latency_ms=200.0,
        status="ok",
    )


def _tool_call(session_id: str = "sess-1", run_id: str = "run-1") -> ToolCallRecord:
    return ToolCallRecord(
        id=str(uuid.uuid4()),
        session_id=session_id,
        run_id=run_id,
        timestamp=_ts(),
        schema_version=1,
        tool_name="search_news",
        status="ok",
        duration_ms=100.0,
    )


def _article_meta(session_id: str = "sess-1", run_id: str = "run-1") -> ArticleMetadataRecord:
    return ArticleMetadataRecord(
        id=str(uuid.uuid4()),
        session_id=session_id,
        run_id=run_id,
        timestamp=_ts(),
        schema_version=1,
        url="https://reuters.com/test",
        source_name="Reuters",
        source_domain="reuters.com",
        bias="center",
        source_factuality="very_high",
        extraction_status="ok",
        word_count=500,
        llm_factuality_rating="high",
        llm_bias_rating="center",
    )


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_null_store_satisfies_protocol() -> None:
    store = NullEventStore()
    assert isinstance(store, EventStore)


def test_sqlite_store_satisfies_protocol(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "test.db"))
    assert isinstance(store, EventStore)
    store.close()


# ---------------------------------------------------------------------------
# NullEventStore
# ---------------------------------------------------------------------------


def test_null_store_all_ops_are_no_ops() -> None:
    store = NullEventStore()
    store.record_event(_event())
    store.record_llm_call(_llm_call())
    store.record_tool_call(_tool_call())
    store.record_article_metadata(_article_meta())
    assert store.query_llm_calls("sess-1") == []
    assert store.query_tool_calls("sess-1") == []
    store.close()


# ---------------------------------------------------------------------------
# SQLiteEventStore — write + read round-trips
# ---------------------------------------------------------------------------


def test_sqlite_record_and_query_llm_call(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "test.db"))
    record = _llm_call()
    store.record_llm_call(record)
    results = store.query_llm_calls("sess-1")
    assert len(results) == 1
    assert results[0].id == record.id
    assert results[0].task == "orchestrator"
    store.close()


def test_sqlite_query_llm_calls_by_run_id(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "test.db"))
    store.record_llm_call(_llm_call(run_id="run-A"))
    store.record_llm_call(_llm_call(run_id="run-B"))
    results = store.query_llm_calls("sess-1", run_id="run-A")
    assert len(results) == 1
    assert results[0].run_id == "run-A"
    store.close()


def test_sqlite_record_and_query_tool_call(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "test.db"))
    record = _tool_call()
    store.record_tool_call(record)
    results = store.query_tool_calls("sess-1")
    assert len(results) == 1
    assert results[0].tool_name == "search_news"
    store.close()


def test_sqlite_record_event(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "test.db"))
    store.record_event(_event())
    store.close()


def test_sqlite_record_article_metadata(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "test.db"))
    store.record_article_metadata(_article_meta())
    store.close()


def test_sqlite_duplicate_id_is_ignored(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "test.db"))
    record = _llm_call()
    store.record_llm_call(record)
    store.record_llm_call(record)  # duplicate — should not raise, just ignore
    results = store.query_llm_calls("sess-1")
    assert len(results) == 1
    store.close()


def test_sqlite_query_limit(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "test.db"))
    for _ in range(5):
        store.record_llm_call(_llm_call())
    results = store.query_llm_calls("sess-1", limit=3)
    assert len(results) == 3
    store.close()


def test_sqlite_query_different_session_returns_empty(tmp_path) -> None:
    store = SQLiteEventStore(str(tmp_path / "test.db"))
    store.record_llm_call(_llm_call(session_id="sess-A"))
    results = store.query_llm_calls("sess-B")
    assert results == []
    store.close()


# ---------------------------------------------------------------------------
# EventRecord / LLMCallRecord to_dict
# ---------------------------------------------------------------------------


def test_event_record_to_dict() -> None:
    rec = _event()
    d = rec.to_dict()
    assert d["event_type"] == "test"
    assert d["payload"] == {"key": "value"}


def test_llm_call_record_to_dict() -> None:
    rec = _llm_call()
    d = rec.to_dict()
    assert d["task"] == "orchestrator"
    assert d["status"] == "ok"
    assert d["error"] is None
