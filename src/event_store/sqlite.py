from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from src.event_store.models import (
    ArticleMetadataRecord,
    EventRecord,
    LLMCallRecord,
    ToolCallRecord,
)

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    payload TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    task TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL,
    duration_ms REAL NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS article_metadata (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    url TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_domain TEXT NOT NULL,
    bias TEXT NOT NULL,
    source_factuality TEXT NOT NULL,
    extraction_status TEXT NOT NULL,
    word_count INTEGER NOT NULL,
    llm_factuality_rating TEXT NOT NULL,
    llm_factuality_confidence REAL NOT NULL DEFAULT 0.5,
    llm_bias_rating TEXT NOT NULL,
    llm_bias_confidence REAL NOT NULL DEFAULT 0.5
);
"""


class SQLiteEventStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(_DDL)
        self._conn.commit()

    def record_event(self, event: EventRecord) -> None:
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?)",
                (
                    event.id,
                    event.session_id,
                    event.run_id,
                    event.timestamp,
                    event.event_type,
                    event.schema_version,
                    json.dumps(event.payload, separators=(",", ":")),
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception("Failed to record event", extra={"event_id": event.id})

    def record_llm_call(self, record: LLMCallRecord) -> None:
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO llm_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    record.session_id,
                    record.run_id,
                    record.timestamp,
                    record.schema_version,
                    record.task,
                    record.model,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.latency_ms,
                    record.status,
                    record.error,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception("Failed to record LLM call", extra={"record_id": record.id})

    def record_tool_call(self, record: ToolCallRecord) -> None:
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    record.session_id,
                    record.run_id,
                    record.timestamp,
                    record.schema_version,
                    record.tool_name,
                    record.status,
                    record.duration_ms,
                    record.error,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception("Failed to record tool call", extra={"record_id": record.id})

    def record_article_metadata(self, record: ArticleMetadataRecord) -> None:
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO article_metadata VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.id,
                    record.session_id,
                    record.run_id,
                    record.timestamp,
                    record.schema_version,
                    record.url,
                    record.source_name,
                    record.source_domain,
                    record.bias,
                    record.source_factuality,
                    record.extraction_status,
                    record.word_count,
                    record.llm_factuality_rating,
                    record.llm_factuality_confidence,
                    record.llm_bias_rating,
                    record.llm_bias_confidence,
                ),
            )
            self._conn.commit()
        except Exception:
            logger.exception("Failed to record article metadata", extra={"record_id": record.id})

    def query_llm_calls(self, session_id: str, run_id: str | None = None, limit: int = 100) -> list[LLMCallRecord]:
        sql = "SELECT * FROM llm_calls WHERE session_id=?"
        params: list[Any] = [session_id]
        if run_id is not None:
            sql += " AND run_id=?"
            params.append(run_id)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            LLMCallRecord(
                id=r[0],
                session_id=r[1],
                run_id=r[2],
                timestamp=r[3],
                schema_version=r[4],
                task=r[5],
                model=r[6],
                prompt_tokens=r[7],
                completion_tokens=r[8],
                latency_ms=r[9],
                status=r[10],
                error=r[11],
            )
            for r in rows
        ]

    def query_tool_calls(self, session_id: str, run_id: str | None = None, limit: int = 100) -> list[ToolCallRecord]:
        sql = "SELECT * FROM tool_calls WHERE session_id=?"
        params: list[Any] = [session_id]
        if run_id is not None:
            sql += " AND run_id=?"
            params.append(run_id)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [
            ToolCallRecord(
                id=r[0],
                session_id=r[1],
                run_id=r[2],
                timestamp=r[3],
                schema_version=r[4],
                tool_name=r[5],
                status=r[6],
                duration_ms=r[7],
                error=r[8],
            )
            for r in rows
        ]

    def close(self) -> None:
        self._conn.close()
