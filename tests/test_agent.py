from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.agent import NewsAgent, _MAX_TOOL_CALLS
from src.agent_models import AgentResult, Context
from src.article_extractor import ArticleExtractor
from src.event_store.null import NullEventStore
from src.llm_service import LLMResponse
from src.report_compiler import ReportCompiler
from src.sources import SourceRegistry


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _make_registry() -> SourceRegistry:
    config = {
        "biases": [{"id": "center", "label": "Center"}],
        "sources": [
            {"name": "Reuters", "domain": "reuters.com", "bias": "center", "factuality": "very_high"},
        ],
    }
    return SourceRegistry(config)


def _no_tool_response(message: str = "All done.") -> LLMResponse:
    return LLMResponse(
        content=message,
        tool_calls=[],
        raw_message={"role": "assistant", "content": message},
        prompt_tokens=10,
        completion_tokens=5,
        model="test-model",
    )


def _tool_response(tool_name: str, arguments: dict[str, Any]) -> LLMResponse:
    tc = {
        "id": "call-1",
        "type": "function",
        "function": {"name": tool_name, "arguments": json.dumps(arguments)},
    }
    return LLMResponse(
        content=None,
        tool_calls=[tc],
        raw_message={"role": "assistant", "content": None, "tool_calls": [tc]},
        prompt_tokens=10,
        completion_tokens=5,
        model="test-model",
    )


@dataclass
class _ScriptedLLMService:
    """Returns responses from a queue; last one is repeated indefinitely."""

    responses: list[LLMResponse] = field(default_factory=list)
    _index: int = field(default=0, init=False, repr=False)

    def call(self, messages: list[dict], task: str = "orchestrator", tools: Any = None, plugins: Any = None) -> LLMResponse:
        if not self.responses:
            return _no_tool_response()
        resp = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        return resp


@dataclass
class _StubExtractor:
    def extract(self, url: str) -> Any:
        pass

    def extract_batch(self, urls: list[str]) -> list[Any]:
        return []


def _make_agent(llm_service: Any, registry: SourceRegistry | None = None) -> NewsAgent:
    return NewsAgent(
        config={"agent": {"articles_per_bias": 2, "system_prompt": "System {date} {topics} {biases} {articles_per_bias}"}},
        llm_service=llm_service,
        source_registry=registry or _make_registry(),
        article_extractor=_StubExtractor(),  # type: ignore[arg-type]
        report_compiler=ReportCompiler(),
        event_store=NullEventStore(),
        output_dir="output",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_agent_run_returns_agent_result() -> None:
    svc = _ScriptedLLMService(responses=[_no_tool_response("Done!")])
    agent = _make_agent(svc)

    result = agent.run(topics=["economy"])

    assert isinstance(result, AgentResult)
    assert result.error is None


def test_agent_run_no_tool_calls_finishes_quickly() -> None:
    svc = _ScriptedLLMService(responses=[_no_tool_response()])
    agent = _make_agent(svc)

    result = agent.run(topics=["tech"])

    assert result.error is None
    assert result.metadata["tool_invocations"] == 0


def test_agent_run_executes_known_tool() -> None:
    # First call returns a search_news tool call, second returns final answer
    svc = _ScriptedLLMService(
        responses=[
            _tool_response("search_news", {"bias_id": "center", "topics": ["tech"]}),
            _no_tool_response("Done"),
        ]
    )
    agent = _make_agent(svc)

    result = agent.run(topics=["tech"])

    # 1 tool call was made (search_news)
    assert result.metadata["tool_invocations"] == 1


def test_agent_run_handles_unknown_tool_gracefully() -> None:
    svc = _ScriptedLLMService(
        responses=[
            _tool_response("nonexistent_tool", {}),
            _no_tool_response("Done"),
        ]
    )
    agent = _make_agent(svc)

    result = agent.run(topics=["news"])

    assert result.error is None
    assert result.metadata["tool_invocations"] == 1


def test_agent_run_respects_max_tool_calls() -> None:
    """Agent must stop after _MAX_TOOL_CALLS even if LLM keeps requesting tools."""
    always_tool_svc = _ScriptedLLMService(
        responses=[_tool_response("search_news", {"bias_id": "center", "topics": ["x"]})] * (_MAX_TOOL_CALLS + 5)
    )
    agent = _make_agent(always_tool_svc)

    result = agent.run(topics=["economy"])

    assert result.metadata["tool_invocations"] <= _MAX_TOOL_CALLS


def test_agent_run_records_session_and_run_id() -> None:
    svc = _ScriptedLLMService(responses=[_no_tool_response()])
    agent = _make_agent(svc)

    result = agent.run(topics=["news"])

    assert result.session_id != ""
    assert result.run_id != ""
    assert result.session_id != result.run_id


def test_agent_run_passes_biases_to_context() -> None:
    svc = _ScriptedLLMService(responses=[_no_tool_response()])
    registry = SourceRegistry({
        "biases": [{"id": "left", "label": "Left"}, {"id": "right", "label": "Right"}],
        "sources": [
            {"name": "L", "domain": "l.com", "bias": "left", "factuality": "mixed"},
            {"name": "R", "domain": "r.com", "bias": "right", "factuality": "mixed"},
        ],
    })
    agent = _make_agent(svc, registry)

    result = agent.run(topics=["news"], biases=["left"])

    assert result.error is None


def test_agent_run_llm_exception_sets_error() -> None:
    @dataclass
    class _ErrorLLMService:
        def call(self, messages: list[dict], task: str = "o", tools: Any = None, plugins: Any = None) -> LLMResponse:
            raise RuntimeError("LLM API down")

    agent = _make_agent(_ErrorLLMService())

    result = agent.run(topics=["news"])

    assert result.error is not None
    assert "LLM API down" in result.error


def test_agent_uses_all_registry_biases_by_default() -> None:
    registry = SourceRegistry({
        "biases": [{"id": "left", "label": "Left"}, {"id": "center", "label": "Center"}],
        "sources": [
            {"name": "L", "domain": "l.com", "bias": "left", "factuality": "high"},
            {"name": "C", "domain": "c.com", "bias": "center", "factuality": "very_high"},
        ],
    })
    svc = _ScriptedLLMService(responses=[_no_tool_response()])
    agent = _make_agent(svc, registry)

    result = agent.run(topics=["news"])

    # Should complete without error — didn't restrict biases
    assert result.error is None


def test_agent_fallback_compile_runs_when_llm_skips_compile_report() -> None:
    """If the LLM never calls compile_report, the agent must still generate the report file."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        # LLM only fires one search_news call then gives a final answer without calling compile_report
        svc = _ScriptedLLMService(
            responses=[
                _tool_response("search_news", {"bias_id": "center", "topics": ["tech"]}),
                _no_tool_response("I found some articles but will not call compile_report."),
            ]
        )
        agent = _make_agent(svc)
        agent._output_dir = tmpdir

        result = agent.run(topics=["tech"])

        # Fallback compilation must have written a file
        assert result.output_path is not None
        assert os.path.exists(result.output_path)
