from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.agent_models import Context
from src.report_compiler import ReportCompiler
from src.schemas import ArticleRecord, ArticleSummary, BiasReport, ReportOutput, DailyReport
from src.sources import SourceRegistry
from src.tools.compile_report import compile_report_tool, tool_definition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> SourceRegistry:
    config = {
        "biases": [
            {"id": "left", "label": "Left"},
            {"id": "center", "label": "Center"},
        ],
        "sources": [
            {"name": "Left Source", "domain": "left.com", "bias": "left", "factuality": "high"},
            {"name": "Center Source", "domain": "center.com", "bias": "center", "factuality": "very_high"},
        ],
    }
    return SourceRegistry(config)


def _make_summary_dict(url: str = "https://example.com", bias: str = "center") -> dict[str, Any]:
    return {
        "article": {
            "url": url,
            "title": "Test Article",
            "source_name": "Example",
            "source_domain": "example.com",
            "bias": bias,
            "source_factuality": "high",
            "full_text": "Full article text content.",
            "word_count": 5,
            "extraction_status": "ok",
        },
        "summary": "A short summary.",
        "llm_factuality_rating": "high",
        "llm_factuality_confidence": 0.9,
        "llm_bias_rating": "center",
        "llm_bias_confidence": 0.8,
        "key_claims": ["Claim"],
        "topics_covered": ["politics"],
    }


def _make_context(summaries: dict[str, list[dict[str, Any]]]) -> Context:
    ctx = Context()
    ctx.work_state["summaries"] = summaries
    return ctx


# ---------------------------------------------------------------------------
# Tests for tool_definition
# ---------------------------------------------------------------------------

def test_tool_definition_structure() -> None:
    defn = tool_definition()
    assert defn["type"] == "function"
    assert defn["function"]["name"] == "compile_report"
    assert "topics" in defn["function"]["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Tests for compile_report_tool
# ---------------------------------------------------------------------------

def test_compile_report_saves_file(tmp_path: Path) -> None:
    registry = _make_registry()
    compiler = ReportCompiler()
    ctx = _make_context({
        "center": [_make_summary_dict(bias="center")],
        "left": [],
    })

    result = compile_report_tool(
        arguments={"topics": ["politics"], "output_dir": str(tmp_path)},
        report_compiler=compiler,
        source_registry=registry,
        context=ctx,
        date="2024-03-15",
        run_id="run-1",
        session_id="sess-1",
        output_dir=str(tmp_path),
    )

    data = json.loads(result)
    assert "output_path" in data
    assert Path(data["output_path"]).exists()


def test_compile_report_total_articles() -> None:
    registry = _make_registry()
    compiler = ReportCompiler()
    ctx = _make_context({
        "center": [_make_summary_dict(url="https://a.com"), _make_summary_dict(url="https://b.com")],
        "left": [_make_summary_dict(url="https://c.com", bias="left")],
    })

    result = compile_report_tool(
        arguments={"topics": ["tech"]},
        report_compiler=compiler,
        source_registry=registry,
        context=ctx,
        date="2024-03-15",
        run_id="r",
        session_id="s",
        output_dir="output",
    )

    data = json.loads(result)
    assert data["total_articles"] == 3


def test_compile_report_no_summaries_generates_empty_report(tmp_path: Path) -> None:
    registry = _make_registry()
    compiler = ReportCompiler()
    ctx = Context()

    result = compile_report_tool(
        arguments={"topics": ["tech"], "output_dir": str(tmp_path)},
        report_compiler=compiler,
        source_registry=registry,
        context=ctx,
        date="2024-03-15",
        run_id="r",
        session_id="s",
        output_dir=str(tmp_path),
    )

    data = json.loads(result)
    assert "output_path" in data
    assert data["total_articles"] == 0


def test_compile_report_stores_in_work_state(tmp_path: Path) -> None:
    registry = _make_registry()
    compiler = ReportCompiler()
    ctx = _make_context({"center": [_make_summary_dict()]})

    compile_report_tool(
        arguments={"topics": ["news"], "output_dir": str(tmp_path)},
        report_compiler=compiler,
        source_registry=registry,
        context=ctx,
        date="2024-01-01",
        run_id="r",
        session_id="s",
        output_dir=str(tmp_path),
    )

    assert "report_output" in ctx.work_state


def test_compile_report_includes_summary_json(tmp_path: Path) -> None:
    registry = _make_registry()
    compiler = ReportCompiler()
    ctx = _make_context({"center": [_make_summary_dict()]})

    result = compile_report_tool(
        arguments={"topics": ["news"], "output_dir": str(tmp_path)},
        report_compiler=compiler,
        source_registry=registry,
        context=ctx,
        date="2024-01-01",
        run_id="run-x",
        session_id="sess-y",
        output_dir=str(tmp_path),
    )

    data = json.loads(result)
    assert "summary" in data
    assert data["summary"]["run_id"] == "run-x"


def test_compile_report_handles_missing_bias(tmp_path: Path) -> None:
    """Biases in registry but not in summaries still produce BiasReport with 0 articles."""
    registry = _make_registry()
    compiler = ReportCompiler()
    # Only center has summaries, left has none
    ctx = _make_context({"center": [_make_summary_dict()]})

    result = compile_report_tool(
        arguments={"topics": ["news"], "output_dir": str(tmp_path)},
        report_compiler=compiler,
        source_registry=registry,
        context=ctx,
        date="2024-01-01",
        run_id="r",
        session_id="s",
        output_dir=str(tmp_path),
    )

    data = json.loads(result)
    bias_counts = {b["bias_id"]: b["article_count"] for b in data["summary"]["bias_reports"]}
    assert bias_counts["center"] == 1
    assert bias_counts["left"] == 0
