from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.agent_models import Context
from src.event_store.null import NullEventStore
from src.llm_service import LLMResponse
from src.tools.summarize_articles import summarize_articles_tool, tool_definition


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

@dataclass
class _ScriptedLLMService:
    """Returns scripted responses, cycling through the list."""

    responses: list[LLMResponse] = field(default_factory=list)
    _index: int = field(default=0, init=False, repr=False)

    def call(self, messages: list[dict], task: str = "orchestrator", tools: Any = None, plugins: Any = None) -> LLMResponse:
        if not self.responses:
            return LLMResponse(content="{}", tool_calls=[], raw_message={}, prompt_tokens=0, completion_tokens=0, model="m", annotations=[])
        response = self.responses[self._index % len(self.responses)]
        self._index += 1
        return response


def _summary_response() -> LLMResponse:
    content = json.dumps({
        "summary": "This article discusses important events.",
        "key_claims": ["Claim A", "Claim B"],
        "topics_covered": ["politics"],
    })
    return LLMResponse(content=content, tool_calls=[], raw_message={}, prompt_tokens=0, completion_tokens=0, model="m", annotations=[])


def _rating_response() -> LLMResponse:
    content = json.dumps({
        "factuality_rating": "high",
        "factuality_confidence": 0.9,
        "bias_rating": "center",
        "bias_confidence": 0.85,
    })
    return LLMResponse(content=content, tool_calls=[], raw_message={}, prompt_tokens=0, completion_tokens=0, model="m", annotations=[])


def _make_context_with_articles(bias_id: str = "center", count: int = 2) -> Context:
    ctx = Context()
    ctx.work_state["extracted_articles"] = {
        bias_id: [
            {
                "url": f"https://example.com/{i}",
                "title": f"Article {i}",
                "source_name": "Example",
                "source_domain": "example.com",
                "bias": bias_id,
                "source_factuality": "high",
                "full_text": f"Full text of article {i} with lots of content.",
                "word_count": 10,
                "extraction_status": "ok",
            }
            for i in range(count)
        ]
    }
    return ctx


# ---------------------------------------------------------------------------
# Tests for tool_definition
# ---------------------------------------------------------------------------

def test_tool_definition_structure() -> None:
    defn = tool_definition()
    assert defn["type"] == "function"
    assert defn["function"]["name"] == "summarize_articles"
    assert "bias_id" in defn["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Tests for summarize_articles_tool
# ---------------------------------------------------------------------------

def test_summarize_produces_summaries() -> None:
    svc = _ScriptedLLMService(responses=[_summary_response(), _rating_response()] * 4)
    ctx = _make_context_with_articles("center", 2)

    result = summarize_articles_tool(
        arguments={"bias_id": "center"},
        llm_service=svc,
        event_store=NullEventStore(),
        context=ctx,
        run_id="run-1",
        session_id="sess-1",
    )

    data = json.loads(result)
    assert data["count"] == 2
    assert len(data["summaries"]) == 2


def test_summarize_stores_in_work_state() -> None:
    svc = _ScriptedLLMService(responses=[_summary_response(), _rating_response()] * 4)
    ctx = _make_context_with_articles("left", 1)

    summarize_articles_tool(
        arguments={"bias_id": "left"},
        llm_service=svc,
        event_store=NullEventStore(),
        context=ctx,
        run_id="r",
        session_id="s",
    )

    assert "summaries" in ctx.work_state
    assert "left" in ctx.work_state["summaries"]


def test_summarize_picks_up_llm_summary_text() -> None:
    svc = _ScriptedLLMService(responses=[_summary_response(), _rating_response()])
    ctx = _make_context_with_articles("center", 1)

    result = summarize_articles_tool(
        arguments={"bias_id": "center"},
        llm_service=svc,
        event_store=NullEventStore(),
        context=ctx,
        run_id="r",
        session_id="s",
    )

    data = json.loads(result)
    s = data["summaries"][0]
    assert "This article discusses" in s["summary"]
    assert s["key_claims"] == ["Claim A", "Claim B"]


def test_summarize_picks_up_llm_ratings() -> None:
    svc = _ScriptedLLMService(responses=[_summary_response(), _rating_response()])
    ctx = _make_context_with_articles("center", 1)

    result = summarize_articles_tool(
        arguments={"bias_id": "center"},
        llm_service=svc,
        event_store=NullEventStore(),
        context=ctx,
        run_id="r",
        session_id="s",
    )

    data = json.loads(result)
    s = data["summaries"][0]
    assert s["llm_factuality_rating"] == "high"
    assert s["llm_factuality_confidence"] == pytest.approx(0.9)
    assert s["llm_bias_rating"] == "center"


def test_summarize_no_articles_returns_error() -> None:
    svc = _ScriptedLLMService()
    ctx = Context()

    result = summarize_articles_tool(
        arguments={"bias_id": "center"},
        llm_service=svc,
        event_store=NullEventStore(),
        context=ctx,
        run_id="r",
        session_id="s",
    )

    data = json.loads(result)
    assert "error" in data
    assert data["summaries"] == []


def test_summarize_graceful_on_llm_failure() -> None:
    """LLM errors should not crash; article should still appear with defaults."""
    @dataclass
    class _BrokenLLMService:
        call_count: int = 0

        def call(self, messages: list[dict], task: str = "orchestrator", tools: Any = None, plugins: Any = None) -> LLMResponse:
            self.call_count += 1
            raise RuntimeError("LLM unavailable")

    svc = _BrokenLLMService()
    ctx = _make_context_with_articles("right", 1)

    result = summarize_articles_tool(
        arguments={"bias_id": "right"},
        llm_service=svc,
        event_store=NullEventStore(),
        context=ctx,
        run_id="r",
        session_id="s",
    )

    data = json.loads(result)
    assert data["count"] == 1
    # Should fall back to defaults
    assert data["summaries"][0]["llm_factuality_rating"] == "mixed"


def test_summarize_records_to_event_store() -> None:
    @dataclass
    class _CapturingEventStore:
        records: list[Any] = field(default_factory=list)

        def record_event(self, *args: Any, **kwargs: Any) -> None:
            pass

        def record_llm_call(self, *args: Any, **kwargs: Any) -> None:
            pass

        def record_tool_call(self, *args: Any, **kwargs: Any) -> None:
            pass

        def record_article_metadata(self, record: Any) -> None:
            self.records.append(record)

        def query_llm_calls(self, *a: Any, **kw: Any) -> list[Any]:
            return []

        def query_tool_calls(self, *a: Any, **kw: Any) -> list[Any]:
            return []

        def close(self) -> None:
            pass

    svc = _ScriptedLLMService(responses=[_summary_response(), _rating_response()])
    ctx = _make_context_with_articles("center", 1)
    store = _CapturingEventStore()

    summarize_articles_tool(
        arguments={"bias_id": "center"},
        llm_service=svc,
        event_store=store,
        context=ctx,
        run_id="run-x",
        session_id="sess-y",
    )

    assert len(store.records) == 1
    assert store.records[0].run_id == "run-x"
