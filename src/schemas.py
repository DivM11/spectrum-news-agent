from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Factuality scale (source-level, from MBFC)
# ---------------------------------------------------------------------------
FACTUALITY_SCORES: dict[str, float] = {
    "very_high": 1.0,
    "high": 0.8,
    "mostly_factual": 0.6,
    "mixed": 0.4,
    "low": 0.2,
    "very_low": 0.0,
}

EXTRACTION_STATUS_OK = "ok"
EXTRACTION_STATUS_FAILED = "failed"
EXTRACTION_STATUS_PAYWALLED = "paywalled"
EXTRACTION_STATUS_TIMEOUT = "timeout"

LLM_FACTUALITY_HIGH = "high"
LLM_FACTUALITY_MIXED = "mixed"
LLM_FACTUALITY_LOW = "low"


@dataclass
class ArticleRecord:
    url: str
    title: str
    source_name: str
    source_domain: str
    bias: str
    source_factuality: str  # MBFC factuality label from SourceRegistry
    full_text: str | None  # Extracted via newspaper4k; None if extraction failed
    word_count: int
    extraction_status: str  # EXTRACTION_STATUS_* constant


@dataclass
class ArticleSummary:
    article: ArticleRecord
    summary: str
    llm_factuality_rating: str  # LLM_FACTUALITY_* constant
    llm_factuality_confidence: float  # 0.0–1.0
    llm_bias_rating: str  # e.g. "left", "center", "right"
    llm_bias_confidence: float  # 0.0–1.0
    key_claims: list[str] = field(default_factory=list)
    topics_covered: list[str] = field(default_factory=list)


@dataclass
class BiasReport:
    bias_id: str
    bias_label: str
    articles: list[ArticleSummary] = field(default_factory=list)
    source_count: int = 0
    avg_source_factuality_score: float = 0.0  # Numeric average of MBFC ratings


@dataclass
class DailyReport:
    date: str  # ISO date: YYYY-MM-DD
    topics: list[str]
    bias_reports: list[BiasReport]
    total_articles: int
    run_id: str
    session_id: str


@dataclass
class ReportOutput:
    daily_report: DailyReport
    big_doc_markdown: str  # Full exhaustive markdown document
    output_path: str  # Path to saved markdown file
