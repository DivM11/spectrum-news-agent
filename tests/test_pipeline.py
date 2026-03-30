from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

import src.tools.extract_articles as _extract_mod
import src.tools.search_news as _search_mod
import src.tools.summarize_articles as _summarize_mod
from src.agent_models import Context
from src.pipeline import (
    BiasProcessorProtocol,
    NewsBiasPipeline,
    PipelineRunnerProtocol,
    SequentialPipelineRunner,
    ThreadedPipelineRunner,
)


# ---------------------------------------------------------------------------
# Stub processors shared by runner tests
# ---------------------------------------------------------------------------


@dataclass
class _RecordingProcessor:
    """Records which bias IDs were processed."""

    processed: list[str] = field(default_factory=list)

    def process(self, bias_id: str, ctx: Context) -> None:
        self.processed.append(bias_id)


@dataclass
class _DelayProcessor:
    """Simulates slow processing to verify concurrency."""

    delay: float = 0.05
    completed: list[str] = field(default_factory=list)

    def process(self, bias_id: str, ctx: Context) -> None:
        time.sleep(self.delay)
        self.completed.append(bias_id)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_sequential_runner_satisfies_protocol() -> None:
    assert isinstance(SequentialPipelineRunner(), PipelineRunnerProtocol)


def test_threaded_runner_satisfies_protocol() -> None:
    assert isinstance(ThreadedPipelineRunner(), PipelineRunnerProtocol)


# ---------------------------------------------------------------------------
# SequentialPipelineRunner
# ---------------------------------------------------------------------------


def test_sequential_runner_processes_all_biases() -> None:
    proc = _RecordingProcessor()
    SequentialPipelineRunner().run(["left", "center", "right"], Context(), proc)
    assert set(proc.processed) == {"left", "center", "right"}


def test_sequential_runner_preserves_order() -> None:
    proc = _RecordingProcessor()
    SequentialPipelineRunner().run(["a", "b", "c"], Context(), proc)
    assert proc.processed == ["a", "b", "c"]


def test_sequential_runner_empty_list_is_noop() -> None:
    proc = _RecordingProcessor()
    SequentialPipelineRunner().run([], Context(), proc)
    assert proc.processed == []


# ---------------------------------------------------------------------------
# ThreadedPipelineRunner
# ---------------------------------------------------------------------------


def test_threaded_runner_processes_all_biases() -> None:
    proc = _RecordingProcessor()
    ThreadedPipelineRunner().run(["left", "center", "right"], Context(), proc)
    assert set(proc.processed) == {"left", "center", "right"}


def test_threaded_runner_empty_list_is_noop() -> None:
    proc = _RecordingProcessor()
    ThreadedPipelineRunner().run([], Context(), proc)
    assert proc.processed == []


def test_threaded_runner_is_concurrent() -> None:
    """3 biases × 0.05s each: parallel should be <0.12s total."""
    proc = _DelayProcessor(delay=0.05)
    t0 = time.monotonic()
    ThreadedPipelineRunner(max_workers=3).run(["left", "center", "right"], Context(), proc)
    elapsed = time.monotonic() - t0
    assert set(proc.completed) == {"left", "center", "right"}
    assert elapsed < 0.12, f"Expected parallel <0.12s; got {elapsed:.3f}s"


def test_threaded_runner_continues_after_processor_error() -> None:
    """A failing processor for one bias must not block the others."""
    successful: list[str] = []

    class _PartiallyFailing:
        def process(self, bias_id: str, ctx: Context) -> None:
            if bias_id == "error":
                raise RuntimeError("Intentional failure")
            successful.append(bias_id)

    ThreadedPipelineRunner(max_workers=3).run(
        ["left", "error", "right"], Context(), _PartiallyFailing()
    )
    assert set(successful) == {"left", "right"}


# ---------------------------------------------------------------------------
# NewsBiasPipeline — unit tests with patched tool modules
# ---------------------------------------------------------------------------


_FAKE_ARTICLE = {
    "url": "http://t.com/a",
    "title": "Test Article",
    "source_domain": "t.com",
    "source_name": "Test",
}


def _make_pipeline(topics: list[str] | None = None) -> NewsBiasPipeline:
    from src.article_extractor import ArticleExtractor
    from src.event_store.null import NullEventStore
    from src.sources import SourceRegistry

    registry = SourceRegistry({
        "biases": [{"id": "left", "label": "Left"}],
        "sources": [{"name": "T", "domain": "t.com", "bias": "left", "factuality": "high"}],
    })
    return NewsBiasPipeline(
        config={"agent": {"articles_per_bias": 2}},
        llm_service=None,  # type: ignore[arg-type]  # patched away
        source_registry=registry,
        article_extractor=ArticleExtractor(),
        event_store=NullEventStore(),
        topics=topics or ["economy"],
        run_id="r1",
        session_id="s1",
    )


def test_pipeline_satisfies_processor_protocol() -> None:
    assert isinstance(_make_pipeline(), BiasProcessorProtocol)


def test_pipeline_calls_search_then_extract_then_summarize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []

    monkeypatch.setattr(
        _search_mod,
        "search_news_tool",
        lambda **_: (call_order.append("search"), json.dumps({"articles": [_FAKE_ARTICLE]}))[1],
    )
    monkeypatch.setattr(
        _extract_mod,
        "extract_articles_tool",
        lambda **_: (call_order.append("extract"), json.dumps({"extracted": [], "total": 1, "ok": 0}))[1],
    )
    monkeypatch.setattr(
        _summarize_mod,
        "summarize_articles_tool",
        lambda **_: (call_order.append("summarize"), json.dumps({"summaries": [], "count": 0}))[1],
    )

    _make_pipeline().process("left", Context())
    assert call_order == ["search", "extract", "summarize"]


def test_pipeline_skips_extract_and_summarize_when_no_articles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_order: list[str] = []

    monkeypatch.setattr(
        _search_mod,
        "search_news_tool",
        lambda **_: (call_order.append("search"), json.dumps({"articles": []}))[1],
    )
    monkeypatch.setattr(
        _extract_mod,
        "extract_articles_tool",
        lambda **_: (call_order.append("extract"), "{}")[1],
    )

    _make_pipeline().process("left", Context())
    assert "extract" not in call_order
    assert call_order == ["search"]


def test_pipeline_passes_search_articles_to_extract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received_articles: list[Any] = []

    monkeypatch.setattr(
        _search_mod,
        "search_news_tool",
        lambda **_: json.dumps({"articles": [_FAKE_ARTICLE]}),
    )
    monkeypatch.setattr(
        _extract_mod,
        "extract_articles_tool",
        lambda **kw: (
            received_articles.extend(kw["arguments"]["articles"]),
            json.dumps({"extracted": [], "total": 0, "ok": 0}),
        )[1],
    )
    monkeypatch.setattr(
        _summarize_mod,
        "summarize_articles_tool",
        lambda **_: json.dumps({"summaries": [], "count": 0}),
    )

    _make_pipeline().process("left", Context())
    assert received_articles == [_FAKE_ARTICLE]


def test_pipeline_passes_bias_id_to_all_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bias_ids_seen: dict[str, str] = {}

    monkeypatch.setattr(
        _search_mod,
        "search_news_tool",
        lambda **kw: (
            bias_ids_seen.__setitem__("search", kw["arguments"]["bias_id"]),
            json.dumps({"articles": [_FAKE_ARTICLE]}),
        )[1],
    )
    monkeypatch.setattr(
        _extract_mod,
        "extract_articles_tool",
        lambda **kw: (
            bias_ids_seen.__setitem__("extract", kw["arguments"]["bias_id"]),
            json.dumps({"extracted": [], "total": 0, "ok": 0}),
        )[1],
    )
    monkeypatch.setattr(
        _summarize_mod,
        "summarize_articles_tool",
        lambda **kw: (
            bias_ids_seen.__setitem__("summarize", kw["arguments"]["bias_id"]),
            json.dumps({"summaries": [], "count": 0}),
        )[1],
    )

    _make_pipeline().process("right", Context())
    assert bias_ids_seen == {"search": "right", "extract": "right", "summarize": "right"}
