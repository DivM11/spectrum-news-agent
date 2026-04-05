from __future__ import annotations

import os

from agent_monitoring.models import (
    ArticleMetadataRecord as SharedArticleMetadataRecord,
    EventRecord as SharedEventRecord,
    LLMCallRecord as SharedLLMCallRecord,
    ToolCallRecord as SharedToolCallRecord,
)
from agent_monitoring.store.postgres import PostgresEventStore as SharedPostgresEventStore

from src.event_store.models import ArticleMetadataRecord, EventRecord, LLMCallRecord, ToolCallRecord


class PostgresEventStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._store: SharedPostgresEventStore | None = None

    def _shared_store(self) -> SharedPostgresEventStore:
        if self._store is None:
            self._store = SharedPostgresEventStore(
                self._dsn,
                app="spectrum-news-agent",
                service=os.getenv("SERVICE_NAME", "web"),
                environment=os.getenv("ENVIRONMENT", "local"),
                initialize_schema=True,
            )
        return self._store

    def record_event(self, event: EventRecord) -> None:
        self._shared_store().record_event(
            SharedEventRecord(
                id=event.id,
                app="spectrum-news-agent",
                service=os.getenv("SERVICE_NAME", "web"),
                environment=os.getenv("ENVIRONMENT", "local"),
                event_type=event.event_type,
                session_id=event.session_id,
                run_id=event.run_id,
                payload=event.payload,
                schema_version=event.schema_version,
                timestamp=event.timestamp,
            )
        )

    def record_llm_call(self, record: LLMCallRecord) -> None:
        self._shared_store().record_llm_call(
            SharedLLMCallRecord(
                id=record.id,
                app="spectrum-news-agent",
                service=os.getenv("SERVICE_NAME", "web"),
                environment=os.getenv("ENVIRONMENT", "local"),
                session_id=record.session_id,
                run_id=record.run_id,
                timestamp=record.timestamp,
                schema_version=record.schema_version,
                task=record.task,
                model=record.model,
                prompt_tokens=record.prompt_tokens,
                completion_tokens=record.completion_tokens,
                latency_ms=record.latency_ms,
                status=record.status,
                error=record.error,
            )
        )

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self._shared_store().record_tool_call(
            SharedToolCallRecord(
                id=record.id,
                app="spectrum-news-agent",
                service=os.getenv("SERVICE_NAME", "web"),
                environment=os.getenv("ENVIRONMENT", "local"),
                session_id=record.session_id,
                run_id=record.run_id,
                timestamp=record.timestamp,
                schema_version=record.schema_version,
                tool_name=record.tool_name,
                arguments=record.arguments,
                result_summary=record.result_summary,
                status=record.status,
                duration_ms=record.duration_ms,
                error=record.error,
            )
        )

    def record_article_metadata(self, record: ArticleMetadataRecord) -> None:
        self._shared_store().record_article_metadata(
            SharedArticleMetadataRecord(
                id=record.id,
                app="spectrum-news-agent",
                service=os.getenv("SERVICE_NAME", "web"),
                environment=os.getenv("ENVIRONMENT", "local"),
                session_id=record.session_id,
                run_id=record.run_id,
                timestamp=record.timestamp,
                schema_version=record.schema_version,
                url=record.url,
                source_name=record.source_name,
                source_domain=record.source_domain,
                bias=record.bias,
                source_factuality=record.source_factuality,
                extraction_status=record.extraction_status,
                word_count=record.word_count,
                llm_factuality_rating=record.llm_factuality_rating,
                llm_factuality_confidence=record.llm_factuality_confidence,
                llm_bias_rating=record.llm_bias_rating,
                llm_bias_confidence=record.llm_bias_confidence,
            )
        )

    def query_llm_calls(self, session_id: str, run_id: str | None = None, limit: int = 100) -> list[LLMCallRecord]:
        rows = self._shared_store().query_llm_calls(
            app="spectrum-news-agent",
            session_id=session_id,
            run_id=run_id,
            limit=limit,
        )
        return [
            LLMCallRecord(
                id=row.id,
                session_id=row.session_id,
                run_id=row.run_id or "",
                timestamp=row.timestamp,
                schema_version=row.schema_version,
                task=row.task or "unknown",
                model=row.model or "unknown",
                prompt_tokens=row.prompt_tokens or 0,
                completion_tokens=row.completion_tokens or 0,
                latency_ms=row.latency_ms or 0.0,
                status=row.status or "unknown",
                error=row.error,
            )
            for row in rows
        ]

    def query_tool_calls(self, session_id: str, run_id: str | None = None, limit: int = 100) -> list[ToolCallRecord]:
        rows = self._shared_store().query_tool_calls(
            app="spectrum-news-agent",
            session_id=session_id,
            run_id=run_id,
            limit=limit,
        )
        return [
            ToolCallRecord(
                id=row.id,
                session_id=row.session_id,
                run_id=row.run_id or "",
                timestamp=row.timestamp,
                schema_version=row.schema_version,
                tool_name=row.tool_name,
                arguments=row.arguments or {},
                result_summary=row.result_summary or "",
                status=row.status or "unknown",
                duration_ms=row.duration_ms or 0.0,
                error=row.error,
            )
            for row in rows
        ]

    def close(self) -> None:
        if self._store is not None:
            self._store.close()