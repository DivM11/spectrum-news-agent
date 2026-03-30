from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.agent import NewsAgent
from src.agent_models import AgentResult, Context
from src.article_extractor import ArticleExtractor
from src.event_store.null import NullEventStore
from src.report_compiler import ReportCompiler
from src.sources import SourceRegistry


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


def _make_registry(biases: tuple[str, ...] = ("center",)) -> SourceRegistry:
    return SourceRegistry({
        "biases": [{"id": b, "label": b.capitalize()} for b in biases],
        "sources": [
            {"name": b[0].upper(), "domain": f"{b}.com", "bias": b, "factuality": "high"}
            for b in biases
        ],
    })


def _make_agent(
    registry: SourceRegistry | None = None,
    output_dir: str = "output",
) -> NewsAgent:
    return NewsAgent(
        config={"agent": {"articles_per_bias": 2}},
        llm_service=None,  # type: ignore[arg-type]  # patched — pipeline calls are stubbed
        source_registry=registry or _make_registry(),
        article_extractor=ArticleExtractor(),
        report_compiler=ReportCompiler(),
        event_store=NullEventStore(),
        output_dir=output_dir,
    )


class _NoopRunner:
    """Pipeline runner that does nothing (no-op for tests)."""

    def run(self, bias_ids: list[str], ctx: Context, processor: Any) -> None:
        pass


def _patch_noop(
    monkeypatch: pytest.MonkeyPatch,
    compile_result: dict[str, Any] | None = None,
) -> None:
    """Stub out the pipeline runner and compile_report_tool."""
    monkeypatch.setattr("src.agent.ThreadedPipelineRunner", lambda **_kw: _NoopRunner())
    result = compile_result or {"output_path": None, "total_articles": 0, "summary": {}}
    monkeypatch.setattr(
        "src.tools.compile_report.compile_report_tool",
        lambda **_kw: json.dumps(result),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_run_returns_agent_result(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_noop(monkeypatch)
    result = _make_agent().run(topics=["economy"])
    assert isinstance(result, AgentResult)
    assert result.error is None


def test_agent_run_records_session_and_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_noop(monkeypatch)
    result = _make_agent().run(topics=["news"])
    assert result.session_id != ""
    assert result.run_id != ""
    assert result.session_id != result.run_id


def test_agent_run_passes_selected_biases_to_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _make_registry(biases=("left", "right"))
    biases_received: list[str] = []

    class _CapturingRunner:
        def run(self, bias_ids: list[str], ctx: Context, processor: Any) -> None:
            biases_received.extend(bias_ids)

    monkeypatch.setattr("src.agent.ThreadedPipelineRunner", lambda **_kw: _CapturingRunner())
    monkeypatch.setattr(
        "src.tools.compile_report.compile_report_tool",
        lambda **_kw: json.dumps({"output_path": None, "total_articles": 0}),
    )
    _make_agent(registry=registry).run(topics=["news"], biases=["left"])
    assert biases_received == ["left"]


def test_agent_run_uses_all_registry_biases_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _make_registry(biases=("left", "center", "right"))
    biases_received: list[str] = []

    class _CapturingRunner:
        def run(self, bias_ids: list[str], ctx: Context, processor: Any) -> None:
            biases_received.extend(bias_ids)

    monkeypatch.setattr("src.agent.ThreadedPipelineRunner", lambda **_kw: _CapturingRunner())
    monkeypatch.setattr(
        "src.tools.compile_report.compile_report_tool",
        lambda **_kw: json.dumps({"output_path": None, "total_articles": 0}),
    )
    _make_agent(registry=registry).run(topics=["news"])
    assert set(biases_received) == {"left", "center", "right"}


def test_agent_run_always_calls_compile_report(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        compile_calls: list[dict[str, Any]] = []

        def _spy_compile(**kw: Any) -> str:
            compile_calls.append(kw)
            return json.dumps({"output_path": os.path.join(tmpdir, "out.md"), "total_articles": 0})

        monkeypatch.setattr("src.agent.ThreadedPipelineRunner", lambda **_kw: _NoopRunner())
        monkeypatch.setattr("src.tools.compile_report.compile_report_tool", _spy_compile)
        _make_agent(output_dir=tmpdir).run(topics=["news"])
        assert len(compile_calls) == 1


def test_agent_run_pipeline_error_sets_result_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ErrorRunner:
        def run(self, bias_ids: list[str], ctx: Context, processor: Any) -> None:
            raise RuntimeError("pipeline boom")

    monkeypatch.setattr("src.agent.ThreadedPipelineRunner", lambda **_kw: _ErrorRunner())
    result = _make_agent().run(topics=["news"])
    assert result.error is not None
    assert "pipeline boom" in result.error


def test_agent_run_output_path_and_total_from_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_path = "/tmp/test-report.md"
    monkeypatch.setattr("src.agent.ThreadedPipelineRunner", lambda **_kw: _NoopRunner())
    monkeypatch.setattr(
        "src.tools.compile_report.compile_report_tool",
        lambda **_kw: json.dumps({"output_path": expected_path, "total_articles": 7}),
    )
    result = _make_agent().run(topics=["news"])
    assert result.output_path == expected_path
    assert result.total_articles == 7

