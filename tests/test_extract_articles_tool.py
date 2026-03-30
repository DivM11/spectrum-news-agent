from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.agent_models import Context
from src.article_extractor import ArticleExtractionResult, EXTRACTION_STATUS_OK, EXTRACTION_STATUS_FAILED
from src.tools.extract_articles import extract_articles_tool, tool_definition


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

@dataclass
class _CaptureExtractor:
    results: list[ArticleExtractionResult] = field(default_factory=list)

    def extract_batch(self, urls: list[str]) -> list[ArticleExtractionResult]:
        # Return a result per URL, cycling through preset results or using default
        out = []
        for i, url in enumerate(urls):
            if i < len(self.results):
                result = self.results[i]
            else:
                result = ArticleExtractionResult(
                    url=url,
                    title=f"Title for {url}",
                    full_text="Some text content.",
                    word_count=3,
                    status=EXTRACTION_STATUS_OK,
                )
            # Override URL to match input
            result = ArticleExtractionResult(
                url=url,
                title=result.title,
                full_text=result.full_text,
                word_count=result.word_count,
                status=result.status,
            )
            out.append(result)
        return out


def _make_context() -> Context:
    return Context()


# ---------------------------------------------------------------------------
# Tests for tool_definition
# ---------------------------------------------------------------------------

def test_tool_definition_structure() -> None:
    defn = tool_definition()
    assert defn["type"] == "function"
    assert defn["function"]["name"] == "extract_articles"
    params = defn["function"]["parameters"]
    assert "bias_id" in params["properties"]
    assert "articles" in params["properties"]


# ---------------------------------------------------------------------------
# Tests for extract_articles_tool
# ---------------------------------------------------------------------------

def test_extract_records_all_urls() -> None:
    extractor = _CaptureExtractor()
    ctx = _make_context()

    articles = [
        {"url": "https://a.com/1", "title": "A1", "source_name": "A", "source_domain": "a.com"},
        {"url": "https://b.com/2", "title": "B2", "source_name": "B", "source_domain": "b.com"},
    ]

    result = extract_articles_tool(
        arguments={"bias_id": "center", "articles": articles},
        article_extractor=extractor,
        context=ctx,
    )

    data = json.loads(result)
    assert data["total"] == 2
    assert data["ok"] == 2


def test_extract_stores_in_work_state() -> None:
    extractor = _CaptureExtractor()
    ctx = _make_context()

    articles = [{"url": "https://x.com/1", "source_domain": "x.com"}]
    extract_articles_tool(
        arguments={"bias_id": "left", "articles": articles},
        article_extractor=extractor,
        context=ctx,
    )

    assert "extracted_articles" in ctx.work_state
    assert "left" in ctx.work_state["extracted_articles"]


def test_extract_applies_factuality_map() -> None:
    extractor = _CaptureExtractor()
    ctx = _make_context()

    articles = [{"url": "https://x.com/1", "source_domain": "x.com", "source_name": "X"}]
    result = extract_articles_tool(
        arguments={
            "bias_id": "left",
            "articles": articles,
            "source_factuality_map": {"x.com": "very_high"},
        },
        article_extractor=extractor,
        context=ctx,
    )

    data = json.loads(result)
    assert data["extracted"][0]["source_factuality"] == "very_high"


def test_extract_failed_article_counted() -> None:
    failed_result = ArticleExtractionResult(
        url="https://fail.com",
        title="",
        full_text=None,
        word_count=0,
        status=EXTRACTION_STATUS_FAILED,
    )
    extractor = _CaptureExtractor(results=[failed_result])
    ctx = _make_context()

    articles = [{"url": "https://fail.com", "source_domain": "fail.com"}]
    result = extract_articles_tool(
        arguments={"bias_id": "right", "articles": articles},
        article_extractor=extractor,
        context=ctx,
    )

    data = json.loads(result)
    assert data["total"] == 1
    assert data["ok"] == 0


def test_extract_empty_articles_returns_error() -> None:
    extractor = _CaptureExtractor()
    ctx = _make_context()

    result = extract_articles_tool(
        arguments={"bias_id": "center", "articles": []},
        article_extractor=extractor,
        context=ctx,
    )

    data = json.loads(result)
    assert "error" in data


def test_extract_default_factuality_is_mixed() -> None:
    extractor = _CaptureExtractor()
    ctx = _make_context()

    articles = [{"url": "https://unknown.com/1", "source_domain": "unknown.com"}]
    result = extract_articles_tool(
        arguments={"bias_id": "center", "articles": articles},
        article_extractor=extractor,
        context=ctx,
    )

    data = json.loads(result)
    assert data["extracted"][0]["source_factuality"] == "mixed"


def test_extract_accumulates_across_calls() -> None:
    extractor = _CaptureExtractor()
    ctx = _make_context()

    extract_articles_tool(
        arguments={"bias_id": "left", "articles": [{"url": "https://a.com", "source_domain": "a.com"}]},
        article_extractor=extractor,
        context=ctx,
    )
    extract_articles_tool(
        arguments={"bias_id": "right", "articles": [{"url": "https://b.com", "source_domain": "b.com"}]},
        article_extractor=extractor,
        context=ctx,
    )

    assert "left" in ctx.work_state["extracted_articles"]
    assert "right" in ctx.work_state["extracted_articles"]
