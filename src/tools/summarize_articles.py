from __future__ import annotations

import json
import logging
from typing import Any

from src.agent_models import Context
from src.event_store.base import EventStore
from src.event_store.models import ArticleMetadataRecord
from src.llm_service import LLMServiceProtocol
from src.llm_validation import clamp_float, extract_json_object, require_field
from src.schemas import ArticleRecord, ArticleSummary, FACTUALITY_SCORES

logger = logging.getLogger(__name__)

_TOOL_NAME = "summarize_articles"

_SUMMARY_PROMPT_TEMPLATE = """\
You are an expert journalism analyst. Summarize the following news article clearly and concisely.
Identify the key claims made. Focus on verifiable facts.

Article title: {title}
Source: {source_name}
URL: {url}

Article text:
{full_text}

Respond with a JSON object:
{{
  "summary": "<2-4 sentence summary>",
  "key_claims": ["claim1", "claim2", ...],
  "topics_covered": ["topic1", "topic2", ...]
}}
"""

_RATING_PROMPT_TEMPLATE = """\
You are a media bias and factuality analyst. Rate the following news article.

Article title: {title}
Source: {source_name} (known bias: {bias})
Summary: {summary}
Key claims: {key_claims}

Rate the article on:
1. Factuality: how factually accurate and evidence-based it is.
2. Bias: the political leaning of the article's framing.

Respond with a JSON object:
{{
  "factuality_rating": "<high|mixed|low>",
  "factuality_confidence": <0.0-1.0>,
  "bias_rating": "<left|center|right|unknown>",
  "bias_confidence": <0.0-1.0>
}}
"""


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Summarize and rate extracted articles using LLM. "
                "Produces per-article summaries with factuality and bias ratings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bias_id": {
                        "type": "string",
                        "description": "The bias category to process.",
                    },
                },
                "required": ["bias_id"],
            },
        },
    }


def summarize_articles_tool(
    arguments: dict[str, Any],
    llm_service: LLMServiceProtocol,
    event_store: EventStore,
    context: Context,
    run_id: str,
    session_id: str,
) -> str:
    bias_id: str = arguments["bias_id"]

    extracted = context.work_state.get("extracted_articles", {}).get(bias_id, [])
    if not extracted:
        msg = f"No extracted articles found for bias '{bias_id}'"
        logger.warning(msg)
        return json.dumps({"bias_id": bias_id, "summaries": [], "error": msg})

    summaries: list[dict[str, Any]] = []

    for article_dict in extracted:
        title = article_dict.get("title") or article_dict.get("url", "?")
        logger.info(
            "Summarising article",
            extra={"bias_id": bias_id, "title": title[:80]},
        )
        summary_dict = _summarize_one(article_dict, bias_id, llm_service, event_store, run_id, session_id)
        if summary_dict is not None:
            summaries.append(summary_dict)

    if "summaries" not in context.work_state:
        context.work_state["summaries"] = {}
    context.work_state["summaries"][bias_id] = summaries

    logger.info(
        "summarize_articles completed",
        extra={"bias_id": bias_id, "count": len(summaries)},
    )
    return json.dumps({"bias_id": bias_id, "count": len(summaries), "summaries": summaries})


def _summarize_one(
    article_dict: dict[str, Any],
    bias_id: str,
    llm_service: LLMServiceProtocol,
    event_store: EventStore,
    run_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    url = article_dict.get("url", "")
    title = article_dict.get("title", url)
    source_name = article_dict.get("source_name", "")
    full_text = article_dict.get("full_text") or ""
    source_factuality = article_dict.get("source_factuality", "mixed")

    # Summarize
    summary_text = ""
    key_claims: list[str] = []
    topics_covered: list[str] = []

    if full_text:
        try:
            summary_prompt = _SUMMARY_PROMPT_TEMPLATE.format(
                title=title,
                source_name=source_name,
                url=url,
                full_text=full_text[:8000],  # Truncate to avoid token overflow
            )
            summary_response = llm_service.call(
                messages=[{"role": "user", "content": summary_prompt}],
                task="summarizer",
            )
            parsed = extract_json_object(summary_response.content)
            if parsed:
                summary_text = str(parsed.get("summary", ""))
                key_claims = list(parsed.get("key_claims", []))
                topics_covered = list(parsed.get("topics_covered", []))
        except Exception as exc:
            logger.warning("Summary LLM call failed", extra={"url": url, "error": str(exc)})

    # Rate
    llm_factuality_rating = "mixed"
    llm_factuality_confidence = 0.5
    llm_bias_rating = "unknown"
    llm_bias_confidence = 0.5

    try:
        rating_prompt = _RATING_PROMPT_TEMPLATE.format(
            title=title,
            source_name=source_name,
            bias=bias_id,
            summary=summary_text or "(no summary available)",
            key_claims=json.dumps(key_claims),
        )
        rating_response = llm_service.call(
            messages=[{"role": "user", "content": rating_prompt}],
            task="rater",
        )
        parsed_rating = extract_json_object(rating_response.content)
        if parsed_rating:
            llm_factuality_rating = str(parsed_rating.get("factuality_rating", "mixed"))
            llm_factuality_confidence = clamp_float(parsed_rating.get("factuality_confidence", 0.5), 0.0, 1.0)
            llm_bias_rating = str(parsed_rating.get("bias_rating", "unknown"))
            llm_bias_confidence = clamp_float(parsed_rating.get("bias_confidence", 0.5), 0.0, 1.0)
    except Exception as exc:
        logger.warning("Rating LLM call failed", extra={"url": url, "error": str(exc)})

    record = ArticleMetadataRecord(
        run_id=run_id,
        session_id=session_id,
        url=url,
        source_name=source_name,
        source_domain=article_dict.get("source_domain", ""),
        bias=bias_id,
        source_factuality=source_factuality,
        extraction_status=article_dict.get("extraction_status", "failed"),
        word_count=int(article_dict.get("word_count", 0)),
        llm_factuality_rating=llm_factuality_rating,
        llm_factuality_confidence=llm_factuality_confidence,
        llm_bias_rating=llm_bias_rating,
        llm_bias_confidence=llm_bias_confidence,
    )
    event_store.record_article_metadata(record)

    return {
        "article": {
            "url": url,
            "title": title,
            "source_name": source_name,
            "source_domain": article_dict.get("source_domain", ""),
            "bias": bias_id,
            "source_factuality": source_factuality,
            "full_text": full_text or None,
            "word_count": int(article_dict.get("word_count", 0)),
            "extraction_status": article_dict.get("extraction_status", "failed"),
        },
        "summary": summary_text,
        "llm_factuality_rating": llm_factuality_rating,
        "llm_factuality_confidence": llm_factuality_confidence,
        "llm_bias_rating": llm_bias_rating,
        "llm_bias_confidence": llm_bias_confidence,
        "key_claims": key_claims,
        "topics_covered": topics_covered,
    }
