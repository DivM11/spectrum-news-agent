from __future__ import annotations

from src.schemas import (
    EXTRACTION_STATUS_FAILED,
    EXTRACTION_STATUS_OK,
    FACTUALITY_SCORES,
    ArticleRecord,
    ArticleSummary,
    BiasReport,
    DailyReport,
    ReportOutput,
)


# ---------------------------------------------------------------------------
# FACTUALITY_SCORES
# ---------------------------------------------------------------------------


def test_factuality_scores_very_high_is_one() -> None:
    assert FACTUALITY_SCORES["very_high"] == 1.0


def test_factuality_scores_very_low_is_zero() -> None:
    assert FACTUALITY_SCORES["very_low"] == 0.0


def test_factuality_scores_ordering() -> None:
    assert FACTUALITY_SCORES["very_high"] > FACTUALITY_SCORES["high"]
    assert FACTUALITY_SCORES["high"] > FACTUALITY_SCORES["mostly_factual"]
    assert FACTUALITY_SCORES["mostly_factual"] > FACTUALITY_SCORES["mixed"]
    assert FACTUALITY_SCORES["mixed"] > FACTUALITY_SCORES["low"]
    assert FACTUALITY_SCORES["low"] > FACTUALITY_SCORES["very_low"]


# ---------------------------------------------------------------------------
# ArticleRecord
# ---------------------------------------------------------------------------


def _article_record(**overrides) -> ArticleRecord:
    defaults = dict(
        url="https://reuters.com/article/1",
        title="Test Article",
        source_name="Reuters",
        source_domain="reuters.com",
        bias="center",
        source_factuality="very_high",
        full_text="Full article text here.",
        word_count=4,
        extraction_status=EXTRACTION_STATUS_OK,
    )
    defaults.update(overrides)
    return ArticleRecord(**defaults)


def test_article_record_fields() -> None:
    rec = _article_record()
    assert rec.url == "https://reuters.com/article/1"
    assert rec.source_name == "Reuters"
    assert rec.word_count == 4
    assert rec.extraction_status == EXTRACTION_STATUS_OK


def test_article_record_allows_none_full_text() -> None:
    rec = _article_record(full_text=None, extraction_status=EXTRACTION_STATUS_FAILED)
    assert rec.full_text is None
    assert rec.extraction_status == EXTRACTION_STATUS_FAILED


# ---------------------------------------------------------------------------
# ArticleSummary
# ---------------------------------------------------------------------------


def test_article_summary_defaults() -> None:
    rec = _article_record()
    summary = ArticleSummary(
        article=rec,
        summary="A brief summary.",
        llm_factuality_rating="high",
        llm_factuality_confidence=0.9,
        llm_bias_rating="center",
        llm_bias_confidence=0.8,
    )
    assert summary.key_claims == []
    assert summary.topics_covered == []


def test_article_summary_with_claims() -> None:
    rec = _article_record()
    summary = ArticleSummary(
        article=rec,
        summary="Summary text.",
        llm_factuality_rating="mixed",
        llm_factuality_confidence=0.5,
        llm_bias_rating="left",
        llm_bias_confidence=0.7,
        key_claims=["Claim A", "Claim B"],
        topics_covered=["US Politics"],
    )
    assert len(summary.key_claims) == 2
    assert "US Politics" in summary.topics_covered


# ---------------------------------------------------------------------------
# BiasReport
# ---------------------------------------------------------------------------


def test_bias_report_defaults() -> None:
    report = BiasReport(bias_id="center", bias_label="Center / Neutral")
    assert report.articles == []
    assert report.source_count == 0
    assert abs(report.avg_source_factuality_score) < 0.01


# ---------------------------------------------------------------------------
# DailyReport / ReportOutput
# ---------------------------------------------------------------------------


def test_daily_report_fields() -> None:
    daily = DailyReport(
        date="2026-03-28",
        topics=["Politics"],
        bias_reports=[],
        total_articles=0,
        run_id="run-1",
        session_id="sess-1",
    )
    assert daily.date == "2026-03-28"
    assert daily.total_articles == 0


def test_report_output_fields() -> None:
    daily = DailyReport(
        date="2026-03-28",
        topics=[],
        bias_reports=[],
        total_articles=0,
        run_id="r",
        session_id="s",
    )
    out = ReportOutput(daily_report=daily, big_doc_markdown="# Report", output_path="/tmp/report.md")
    assert out.output_path == "/tmp/report.md"
    assert out.big_doc_markdown.startswith("# Report")
