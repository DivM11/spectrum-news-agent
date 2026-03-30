from __future__ import annotations

import json
import logging
from typing import Any

from src.agent_models import Context
from src.llm_service import LLMServiceProtocol
from src.sources import SourceRegistry

logger = logging.getLogger(__name__)

_TOOL_NAME = "search_news"


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Search for recent news articles from sources matching the specified political bias. "
                "Returns a list of articles with URLs, titles, and source domains."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bias_id": {
                        "type": "string",
                        "description": "The bias category to search (e.g. 'left', 'center', 'right').",
                    },
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of topics to search for.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of articles to return.",
                        "default": 5,
                    },
                },
                "required": ["bias_id", "topics"],
            },
        },
    }


def search_news_tool(
    arguments: dict[str, Any],
    config: dict[str, Any],
    llm_service: LLMServiceProtocol,
    source_registry: SourceRegistry,
    context: Context,
) -> str:
    bias_id: str = arguments["bias_id"]
    topics: list[str] = arguments["topics"]
    max_results: int = int(arguments.get("max_results", config.get("agent", {}).get("articles_per_bias", 5)))

    domains = source_registry.get_domains_by_bias(bias_id)
    if not domains:
        error_msg = f"No sources found for bias '{bias_id}'"
        logger.warning(error_msg)
        return json.dumps({"error": error_msg, "articles": []})

    topic_query = " OR ".join(topics)
    messages = [
        {
            "role": "user",
            "content": (
                f"Search for news articles about the following topics: {topic_query}. "
                f"Return {max_results} recent and relevant articles. "
                f"Focus on factual reporting and include article titles and URLs."
            ),
        }
    ]

    plugins = [
        {
            "id": "web",
            "engine": config.get("web_search", {}).get("engine", "exa"),
            "include_domains": domains,
            "max_results": max_results,
        }
    ]

    try:
        response = llm_service.call(messages=messages, task="search", plugins=plugins)
    except Exception as exc:
        logger.error("Search LLM call failed", extra={"bias_id": bias_id, "error": str(exc)})
        context.work_state.setdefault("articles_found", {})[bias_id] = []
        return json.dumps({"bias_id": bias_id, "articles": [], "error": f"Search LLM failed: {exc}"})

    articles: list[dict[str, str]] = []

    if response.annotations:
        for annotation in response.annotations:
            # OpenRouter Exa returns {"type": "url_citation", "url_citation": {"url": ..., "title": ...}}
            if annotation.get("type") == "url_citation" and "url_citation" in annotation:
                inner = annotation["url_citation"]
                url = inner.get("url", "")
                title = inner.get("title", url)
            else:
                # Flat format {"url": ..., "title": ...}
                url = annotation.get("url", "")
                title = annotation.get("title", url)
            domain = _extract_domain(url)
            source_info = source_registry.get_source_by_domain(domain)
            articles.append(
                {
                    "url": url,
                    "title": title,
                    "source_domain": domain,
                    "source_name": source_info.name if source_info else domain,
                }
            )

    # Deduplicate by URL, preserve order
    seen: set[str] = set()
    unique_articles: list[dict[str, str]] = []
    for a in articles:
        if a["url"] not in seen:
            seen.add(a["url"])
            unique_articles.append(a)

    unique_articles = unique_articles[:max_results]

    # Update shared work state
    if "articles_found" not in context.work_state:
        context.work_state["articles_found"] = {}
    context.work_state["articles_found"][bias_id] = unique_articles

    logger.info(
        "search_news completed",
        extra={"bias_id": bias_id, "articles_found": len(unique_articles)},
    )
    return json.dumps({"bias_id": bias_id, "articles": unique_articles})


def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        host = parsed.netloc.lower()
        # Remove www. prefix
        return host.removeprefix("www.")
    except Exception:
        return url
