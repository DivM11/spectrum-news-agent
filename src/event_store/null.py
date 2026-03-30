from __future__ import annotations

from src.event_store.models import (
    ArticleMetadataRecord,
    EventRecord,
    LLMCallRecord,
    ToolCallRecord,
)


class NullEventStore:
    """No-op event store. Use when monitoring is disabled."""

    def record_event(self, event: EventRecord) -> None:
        pass

    def record_llm_call(self, record: LLMCallRecord) -> None:
        pass

    def record_tool_call(self, record: ToolCallRecord) -> None:
        pass

    def record_article_metadata(self, record: ArticleMetadataRecord) -> None:
        pass

    def query_llm_calls(self, session_id: str, run_id: str | None = None, limit: int = 100) -> list[LLMCallRecord]:
        return []

    def query_tool_calls(self, session_id: str, run_id: str | None = None, limit: int = 100) -> list[ToolCallRecord]:
        return []

    def close(self) -> None:
        pass
