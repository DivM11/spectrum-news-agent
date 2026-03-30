from __future__ import annotations

import pytest

from src.report_compiler import ReportCompiler
from src.schemas import ArticleRecord, ArticleSummary, BiasReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_article_record(
    url: str = "https://example.com/article",
    title: str = "Test Article",
    source_factuality: str = "high",
    full_text: str | None = "Full text of the article.",
) -> ArticleRecord:
    return ArticleRecord(
        url=url,
        title=title,
        source_name="Example News",
        source_domain="example.com",
        bias="center",
        source_factuality=source_factuality,
        full_text=full_text,
        word_count=len((full_text or "").split()),
        extraction_status="ok" if full_text else "failed",
    )


def _make_summary(article: ArticleRecord | None = None) -> ArticleSummary:
    if article is None:
        article = _make_article_record()
    return ArticleSummary(
        article=article,
        summary="This is a short summary.",
        llm_factuality_rating="high",
        llm_factuality_confidence=0.9,
        llm_bias_rating="center",
        llm_bias_confidence=0.8,
        key_claims=["Claim A", "Claim B"],
        topics_covered=["politics"],
    )


def _make_bias_report(bias_id: str = "center", n: int = 2) -> BiasReport:
    articles = [_make_summary(_make_article_record(url=f"https://example.com/{i}")) for i in range(n)]
    return BiasReport(
        bias_id=bias_id,
        bias_label=bias_id.capitalize(),
        articles=articles,
        source_count=n,
        avg_source_factuality_score=0.8,
    )


# ---------------------------------------------------------------------------
# Tests for ReportCompiler.compile
# ---------------------------------------------------------------------------

def test_compile_returns_report_output() -> None:
    compiler = ReportCompiler()
    bias_reports = [_make_bias_report()]
    output = compiler.compile(bias_reports, ["politics"], "2024-01-15", "run-1", "sess-1")

    assert output.daily_report.date == "2024-01-15"
    assert output.daily_report.total_articles == 2
    assert output.big_doc_markdown != ""


def test_compile_sets_daily_report_fields() -> None:
    compiler = ReportCompiler()
    bias_reports = [_make_bias_report("left", 3), _make_bias_report("right", 1)]
    output = compiler.compile(bias_reports, ["economy", "climate"], "2024-06-01", "run-2", "sess-2")

    dr = output.daily_report
    assert dr.topics == ["economy", "climate"]
    assert dr.run_id == "run-2"
    assert dr.session_id == "sess-2"
    assert dr.total_articles == 4


def test_compile_output_path_initially_empty() -> None:
    compiler = ReportCompiler()
    output = compiler.compile([], [], "2024-01-01", "r", "s")
    assert output.output_path == ""


# ---------------------------------------------------------------------------
# Tests for _build_big_doc
# ---------------------------------------------------------------------------

def test_big_doc_contains_date() -> None:
    compiler = ReportCompiler()
    output = compiler.compile([_make_bias_report()], [], "2025-03-22", "r", "s")
    assert "2025-03-22" in output.big_doc_markdown


def test_big_doc_contains_bias_label() -> None:
    compiler = ReportCompiler()
    report = _make_bias_report("left")
    report.bias_label = "Left"
    output = compiler.compile([report], [], "2024-01-01", "r", "s")
    assert "Left" in output.big_doc_markdown


def test_big_doc_contains_article_title() -> None:
    compiler = ReportCompiler()
    article = _make_article_record(title="Unique Test Article Title")
    summary = _make_summary(article)
    bias_report = BiasReport(bias_id="center", bias_label="Center", articles=[summary])
    output = compiler.compile([bias_report], [], "2024-01-01", "r", "s")
    assert "Unique Test Article Title" in output.big_doc_markdown


def test_big_doc_includes_full_text_in_details() -> None:
    compiler = ReportCompiler()
    article = _make_article_record(full_text="Complete original article body text.")
    summary = _make_summary(article)
    bias_report = BiasReport(bias_id="center", bias_label="Center", articles=[summary])
    output = compiler.compile([bias_report], [], "2024-01-01", "r", "s")
    assert "Complete original article body text." in output.big_doc_markdown
    assert "<details>" in output.big_doc_markdown


def test_big_doc_no_full_text_skips_details() -> None:
    compiler = ReportCompiler()
    article = _make_article_record(full_text=None)
    summary = _make_summary(article)
    bias_report = BiasReport(bias_id="center", bias_label="Center", articles=[summary])
    output = compiler.compile([bias_report], [], "2024-01-01", "r", "s")
    assert "<details>" not in output.big_doc_markdown


def test_big_doc_fallback_summary_when_extraction_fails() -> None:
    compiler = ReportCompiler()
    article = _make_article_record(full_text=None)
    summary = ArticleSummary(
        article=article,
        summary="",  # empty — extraction failed, no LLM summary
        llm_factuality_rating="mixed",
        llm_factuality_confidence=0.3,
        llm_bias_rating="center",
        llm_bias_confidence=0.4,
    )
    bias_report = BiasReport(bias_id="center", bias_label="Center", articles=[summary])
    output = compiler.compile([bias_report], [], "2024-01-01", "r", "s")
    assert "could not be extracted" in output.big_doc_markdown


def test_big_doc_empty_bias_report() -> None:
    compiler = ReportCompiler()
    bias_report = BiasReport(bias_id="center", bias_label="Center", articles=[])
    output = compiler.compile([bias_report], [], "2024-01-01", "r", "s")
    assert "No articles collected" in output.big_doc_markdown


# ---------------------------------------------------------------------------
# Tests for save
# ---------------------------------------------------------------------------

def test_save_writes_file(tmp_path: pytest.TempPathFactory) -> None:
    compiler = ReportCompiler()
    output = compiler.compile([_make_bias_report()], [], "2024-08-10", "r", "s")
    saved = compiler.save(output, str(tmp_path))

    from pathlib import Path
    p = Path(saved)
    assert p.exists()
    assert p.name == "2024-08-10.md"
    assert p.read_text(encoding="utf-8") == output.big_doc_markdown


def test_save_creates_directory(tmp_path: pytest.TempPathFactory) -> None:
    compiler = ReportCompiler()
    output = compiler.compile([], [], "2024-01-01", "r", "s")
    nested = str(tmp_path / "nested" / "dir")
    saved = compiler.save(output, nested)

    from pathlib import Path
    assert Path(saved).exists()


def test_save_sets_output_path(tmp_path: pytest.TempPathFactory) -> None:
    compiler = ReportCompiler()
    output = compiler.compile([], [], "2024-01-01", "r", "s")
    assert output.output_path == ""
    compiler.save(output, str(tmp_path))
    assert output.output_path != ""


# ---------------------------------------------------------------------------
# Tests for build_summary_json
# ---------------------------------------------------------------------------

def test_build_summary_json_structure() -> None:
    compiler = ReportCompiler()
    bias_reports = [_make_bias_report()]
    output = compiler.compile(bias_reports, ["tech"], "2024-02-01", "run-x", "sess-y")
    j = compiler.build_summary_json(output.daily_report)

    assert j["date"] == "2024-02-01"
    assert j["topics"] == ["tech"]
    assert j["run_id"] == "run-x"
    assert j["session_id"] == "sess-y"
    assert isinstance(j["bias_reports"], list)
    assert len(j["bias_reports"]) == 1
    assert j["bias_reports"][0]["article_count"] == 2
