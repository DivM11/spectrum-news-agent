from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds")


def _uuid() -> str:
    return str(uuid.uuid4())


@dataclass
class EventRecord:
    run_id: str
    session_id: str
    event_type: str
    payload: dict[str, Any]
    id: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "schema_version": self.schema_version,
            "payload": self.payload,
        }


@dataclass
class LLMCallRecord:
    id: str
    session_id: str
    run_id: str
    timestamp: str
    schema_version: int
    task: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    status: str  # "ok" or "error"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "task": self.task,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class ToolCallRecord:
    run_id: str
    session_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    status: str = "ok"
    duration_ms: float = 0.0
    error: str | None = None
    id: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "tool_name": self.tool_name,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }


@dataclass
class ArticleMetadataRecord:
    run_id: str
    session_id: str
    url: str
    source_name: str
    source_domain: str
    bias: str
    source_factuality: str
    extraction_status: str
    word_count: int
    llm_factuality_rating: str
    llm_factuality_confidence: float = 0.5
    llm_bias_rating: str = "unknown"
    llm_bias_confidence: float = 0.5
    id: str = field(default_factory=_uuid)
    timestamp: str = field(default_factory=_now)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "url": self.url,
            "source_name": self.source_name,
            "source_domain": self.source_domain,
            "bias": self.bias,
            "source_factuality": self.source_factuality,
            "extraction_status": self.extraction_status,
            "word_count": self.word_count,
            "llm_factuality_rating": self.llm_factuality_rating,
            "llm_factuality_confidence": self.llm_factuality_confidence,
            "llm_bias_rating": self.llm_bias_rating,
            "llm_bias_confidence": self.llm_bias_confidence,
        }
