from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.agent_models import Context
from src.llm_service import LLMResponse
from src.sources import SourceRegistry
from src.tools.search_news import search_news_tool, tool_definition, _extract_domain


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

@dataclass
class _CaptureLLMService:
    """Records calls made to it; returns a pre-configured response."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    response: LLMResponse = field(
        default_factory=lambda: LLMResponse(
            content="Here are articles.",
            tool_calls=[],
            raw_message={},
            prompt_tokens=0,
            completion_tokens=0,
            model="test-model",
            annotations=[
                {"url": "https://reuters.com/article1", "title": "Story One"},
                {"url": "https://reuters.com/article2", "title": "Story Two"},
            ],
        )
    )

    def call(self, messages: list[dict], task: str = "orchestrator", tools: Any = None, plugins: Any = None) -> LLMResponse:
        self.calls.append({"messages": messages, "task": task, "plugins": plugins})
        return self.response


def _make_registry() -> SourceRegistry:
    config = {
        "biases": [{"id": "center", "label": "Center"}],
        "sources": [
            {"name": "Reuters", "domain": "reuters.com", "bias": "center", "factuality": "very_high"},
        ],
    }
    return SourceRegistry(config)


def _make_context() -> Context:
    return Context()


# ---------------------------------------------------------------------------
# Tests for tool_definition
# ---------------------------------------------------------------------------

def test_tool_definition_structure() -> None:
    defn = tool_definition()
    assert defn["type"] == "function"
    assert defn["function"]["name"] == "search_news"
    params = defn["function"]["parameters"]
    assert "bias_id" in params["properties"]
    assert "topics" in params["properties"]


# ---------------------------------------------------------------------------
# Tests for search_news_tool
# ---------------------------------------------------------------------------

def test_search_news_returns_articles() -> None:
    svc = _CaptureLLMService()
    registry = _make_registry()
    ctx = _make_context()

    result = search_news_tool(
        arguments={"bias_id": "center", "topics": ["economy"]},
        config={"agent": {"articles_per_bias": 5}},
        llm_service=svc,
        source_registry=registry,
        context=ctx,
    )
    data = json.loads(result)
    assert data["bias_id"] == "center"
    assert len(data["articles"]) == 2


def test_search_news_stores_in_work_state() -> None:
    svc = _CaptureLLMService()
    registry = _make_registry()
    ctx = _make_context()

    search_news_tool(
        arguments={"bias_id": "center", "topics": ["climate"]},
        config={},
        llm_service=svc,
        source_registry=registry,
        context=ctx,
    )

    assert "articles_found" in ctx.work_state
    assert "center" in ctx.work_state["articles_found"]


def test_search_news_respects_max_results() -> None:
    many_annotations = [
        {"url": f"https://reuters.com/a{i}", "title": f"Story {i}"}
        for i in range(10)
    ]
    svc = _CaptureLLMService(
        response=LLMResponse(
            content="",
            tool_calls=[],
            raw_message={},
            prompt_tokens=0,
            completion_tokens=0,
            model="m",
            annotations=many_annotations,
        )
    )
    registry = _make_registry()
    ctx = _make_context()

    result = search_news_tool(
        arguments={"bias_id": "center", "topics": ["news"], "max_results": 3},
        config={},
        llm_service=svc,
        source_registry=registry,
        context=ctx,
    )
    data = json.loads(result)
    assert len(data["articles"]) == 3


def test_search_news_deduplicates_urls() -> None:
    duplicate_annotations = [
        {"url": "https://reuters.com/same", "title": "Same Story"},
        {"url": "https://reuters.com/same", "title": "Same Story Duplicate"},
    ]
    svc = _CaptureLLMService(
        response=LLMResponse(
            content="",
            tool_calls=[],
            raw_message={},
            prompt_tokens=0,
            completion_tokens=0,
            model="m",
            annotations=duplicate_annotations,
        )
    )
    registry = _make_registry()
    ctx = _make_context()

    result = search_news_tool(
        arguments={"bias_id": "center", "topics": ["news"]},
        config={},
        llm_service=svc,
        source_registry=registry,
        context=ctx,
    )
    data = json.loads(result)
    assert len(data["articles"]) == 1


def test_search_news_unknown_bias_returns_error() -> None:
    svc = _CaptureLLMService()
    registry = _make_registry()
    ctx = _make_context()

    result = search_news_tool(
        arguments={"bias_id": "nonexistent", "topics": ["news"]},
        config={},
        llm_service=svc,
        source_registry=registry,
        context=ctx,
    )
    data = json.loads(result)
    assert "error" in data
    assert data["articles"] == []


def test_search_news_passes_plugin_with_domains() -> None:
    svc = _CaptureLLMService(
        response=LLMResponse(content="", tool_calls=[], raw_message={}, prompt_tokens=0, completion_tokens=0, model="m", annotations=[])
    )
    registry = _make_registry()
    ctx = _make_context()

    search_news_tool(
        arguments={"bias_id": "center", "topics": ["tech"]},
        config={"web_search": {"engine": "exa"}},
        llm_service=svc,
        source_registry=registry,
        context=ctx,
    )

    assert svc.calls
    plugins = svc.calls[0]["plugins"]
    assert plugins is not None
    assert plugins[0]["engine"] == "exa"
    assert "reuters.com" in plugins[0]["include_domains"]


# ---------------------------------------------------------------------------
# Tests for _extract_domain helper
# ---------------------------------------------------------------------------

def test_extract_domain_strips_www() -> None:
    assert _extract_domain("https://www.reuters.com/article") == "reuters.com"


def test_extract_domain_no_www() -> None:
    assert _extract_domain("https://apnews.com/story") == "apnews.com"


def test_extract_domain_empty_fallback() -> None:
    result = _extract_domain("")
    assert isinstance(result, str)


def test_search_news_handles_search_llm_failure() -> None:
    """When the search LLM call raises, search_news returns a graceful error JSON instead of propagating."""
    @dataclass
    class _FailingLLMService:
        def call(self, messages: list, task: str = "orchestrator", tools: Any = None, plugins: Any = None) -> LLMResponse:
            raise RuntimeError("API unavailable")

    registry = _make_registry()
    ctx = _make_context()

    result = search_news_tool(
        arguments={"bias_id": "center", "topics": ["news"]},
        config={},
        llm_service=_FailingLLMService(),
        source_registry=registry,
        context=ctx,
    )
    data = json.loads(result)
    assert "error" in data
    assert data["articles"] == []
    assert ctx.work_state["articles_found"]["center"] == []


def test_search_news_handles_url_citation_annotation_format() -> None:
    """OpenRouter Exa returns {type: url_citation, url_citation: {url, title}} — must be unwrapped."""
    nested_annotations = [
        {
            "type": "url_citation",
            "url_citation": {
                "url": "https://reuters.com/article-nested",
                "title": "Nested Story",
                "start_index": 0,
                "end_index": 50,
            },
        },
    ]
    svc = _CaptureLLMService(
        response=LLMResponse(
            content="Here are articles.",
            tool_calls=[],
            raw_message={},
            prompt_tokens=0,
            completion_tokens=0,
            model="m",
            annotations=nested_annotations,
        )
    )
    registry = _make_registry()
    ctx = _make_context()

    result = search_news_tool(
        arguments={"bias_id": "center", "topics": ["economy"]},
        config={},
        llm_service=svc,
        source_registry=registry,
        context=ctx,
    )
    data = json.loads(result)
    assert len(data["articles"]) == 1
    assert data["articles"][0]["url"] == "https://reuters.com/article-nested"
    assert data["articles"][0]["title"] == "Nested Story"
