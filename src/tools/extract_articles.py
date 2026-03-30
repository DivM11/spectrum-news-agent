from __future__ import annotations

import json
import logging
from typing import Any

from src.agent_models import Context
from src.article_extractor import ArticleExtractor
from src.schemas import ArticleRecord

logger = logging.getLogger(__name__)

_TOOL_NAME = "extract_articles"


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Extract full text from a list of article URLs using newspaper4k. "
                "Returns extraction status (ok/failed/paywalled/timeout) and word counts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bias_id": {
                        "type": "string",
                        "description": "The bias category these articles belong to.",
                    },
                    "articles": {
                        "type": "array",
                        "description": "List of articles with url, title, source_name, source_domain.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "title": {"type": "string"},
                                "source_name": {"type": "string"},
                                "source_domain": {"type": "string"},
                            },
                            "required": ["url"],
                        },
                    },
                    "source_factuality_map": {
                        "type": "object",
                        "description": "Mapping from source_domain to MBFC factuality label.",
                    },
                },
                "required": ["bias_id", "articles"],
            },
        },
    }


def extract_articles_tool(
    arguments: dict[str, Any],
    article_extractor: ArticleExtractor,
    context: Context,
) -> str:
    bias_id: str = arguments["bias_id"]
    articles_meta: list[dict[str, Any]] = arguments.get("articles", [])
    source_factuality_map: dict[str, str] = arguments.get("source_factuality_map", {})

    urls = [a["url"] for a in articles_meta]
    if not urls:
        msg = "No URLs provided"
        logger.warning(msg, extra={"bias_id": bias_id})
        return json.dumps({"bias_id": bias_id, "extracted": [], "error": msg})

    extraction_results = article_extractor.extract_batch(urls)

    # Build index from URL → meta
    meta_by_url = {a["url"]: a for a in articles_meta}

    article_records: list[dict[str, Any]] = []
    for result in extraction_results:
        meta = meta_by_url.get(result.url, {})
        domain = meta.get("source_domain", "")
        factuality = source_factuality_map.get(domain, "mixed")

        record = ArticleRecord(
            url=result.url,
            title=result.title or meta.get("title", result.url),
            source_name=meta.get("source_name", domain),
            source_domain=domain,
            bias=bias_id,
            source_factuality=factuality,
            full_text=result.full_text,
            word_count=result.word_count,
            extraction_status=result.status,
        )
        article_records.append(
            {
                "url": record.url,
                "title": record.title,
                "source_name": record.source_name,
                "source_domain": record.source_domain,
                "bias": record.bias,
                "source_factuality": record.source_factuality,
                "full_text": record.full_text,
                "word_count": record.word_count,
                "extraction_status": record.extraction_status,
            }
        )

    # Update work state (setdefault is atomic in CPython — safe for concurrent bias calls)
    context.work_state.setdefault("extracted_articles", {})[bias_id] = article_records

    ok_count = sum(1 for r in article_records if r["extraction_status"] == "ok")
    logger.info(
        "extract_articles completed",
        extra={"bias_id": bias_id, "total": len(article_records), "ok": ok_count},
    )
    return json.dumps(
        {"bias_id": bias_id, "total": len(article_records), "ok": ok_count, "extracted": article_records}
    )
