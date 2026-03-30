"""
Parallelized news collection pipeline.

Architecture (SOLID – Dependency Inversion):

  BiasProcessorProtocol   — abstract: process one bias category (search → extract → summarize)
  PipelineRunnerProtocol  — abstract: run a processor over multiple bias categories

Concrete implementations:

  NewsBiasPipeline        — BiasProcessorProtocol: calls the three tool functions in order
  SequentialPipelineRunner — PipelineRunnerProtocol: one bias at a time (debug / testing)
  ThreadedPipelineRunner  — PipelineRunnerProtocol: all biases concurrently via ThreadPoolExecutor

Thread-safety: each bias writes to its own key in ctx.work_state, so concurrent
writes are safe as long as setdefault() is used for outer-dict creation.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol, runtime_checkable

from src.agent_models import Context
from src.article_extractor import ArticleExtractor
from src.event_store.base import EventStore
from src.llm_service import LLMServiceProtocol
from src.sources import SourceRegistry
from src.tools import extract_articles as _t_extract
from src.tools import search_news as _t_search
from src.tools import summarize_articles as _t_summarize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols (Dependency Inversion — callers depend on abstractions, not concretions)
# ---------------------------------------------------------------------------


@runtime_checkable
class BiasProcessorProtocol(Protocol):
    """Runs the full data pipeline for a single bias category."""

    def process(self, bias_id: str, ctx: Context) -> None: ...


@runtime_checkable
class PipelineRunnerProtocol(Protocol):
    """Orchestrates a BiasProcessor over multiple bias categories."""

    def run(self, bias_ids: list[str], ctx: Context, processor: BiasProcessorProtocol) -> None: ...


# ---------------------------------------------------------------------------
# Concrete runners
# ---------------------------------------------------------------------------


class SequentialPipelineRunner:
    """Processes bias categories one at a time. Useful for debugging and testing."""

    def run(self, bias_ids: list[str], ctx: Context, processor: BiasProcessorProtocol) -> None:
        for bias_id in bias_ids:
            processor.process(bias_id, ctx)


class ThreadedPipelineRunner:
    """Processes all bias categories concurrently using a thread pool."""

    def __init__(self, max_workers: int | None = None) -> None:
        self._max_workers = max_workers

    def run(self, bias_ids: list[str], ctx: Context, processor: BiasProcessorProtocol) -> None:
        if not bias_ids:
            return
        workers = self._max_workers or len(bias_ids)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(processor.process, bias_id, ctx): bias_id
                for bias_id in bias_ids
            }
            for future in as_completed(futures):
                bias_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.error(
                        "Bias pipeline failed",
                        extra={"bias_id": bias_id, "error": str(exc)},
                    )


# ---------------------------------------------------------------------------
# Concrete pipeline for a single bias
# ---------------------------------------------------------------------------


class NewsBiasPipeline:
    """
    Concrete BiasProcessor.

    For a single bias category, runs in sequence:
      1. search_news       — finds article URLs via search LLM + Exa web search
      2. extract_articles  — downloads and parses full article text via newspaper4k
      3. summarize_articles — summarises and rates each article (parallelised internally)

    Thread-safe: writes are keyed by bias_id in ctx.work_state, so concurrent
    calls for different biases never collide.
    """

    def __init__(
        self,
        config: dict[str, Any],
        llm_service: LLMServiceProtocol,
        source_registry: SourceRegistry,
        article_extractor: ArticleExtractor,
        event_store: EventStore,
        topics: list[str],
        run_id: str,
        session_id: str,
    ) -> None:
        self._config = config
        self._llm_service = llm_service
        self._source_registry = source_registry
        self._article_extractor = article_extractor
        self._event_store = event_store
        self._topics = topics
        self._run_id = run_id
        self._session_id = session_id

    def process(self, bias_id: str, ctx: Context) -> None:
        articles_per_bias: int = self._config.get("agent", {}).get("articles_per_bias", 5)

        # 1. Search
        logger.info("Pipeline: searching news", extra={"bias_id": bias_id})
        search_raw = _t_search.search_news_tool(
            arguments={
                "bias_id": bias_id,
                "topics": self._topics,
                "max_results": articles_per_bias,
            },
            config=self._config,
            llm_service=self._llm_service,
            source_registry=self._source_registry,
            context=ctx,
        )
        articles: list[dict[str, Any]] = json.loads(search_raw).get("articles", [])
        if not articles:
            logger.warning(
                "No articles found — skipping extract and summarise",
                extra={"bias_id": bias_id},
            )
            return

        # 2. Extract
        logger.info(
            "Pipeline: extracting articles",
            extra={"bias_id": bias_id, "count": len(articles)},
        )
        _t_extract.extract_articles_tool(
            arguments={
                "bias_id": bias_id,
                "articles": articles,
                "source_factuality_map": self._source_registry.get_factuality_map(),
            },
            article_extractor=self._article_extractor,
            context=ctx,
        )

        # 3. Summarise (article-level parallelism handled inside the tool)
        logger.info("Pipeline: summarising articles", extra={"bias_id": bias_id})
        _t_summarize.summarize_articles_tool(
            arguments={"bias_id": bias_id},
            llm_service=self._llm_service,
            event_store=self._event_store,
            context=ctx,
            run_id=self._run_id,
            session_id=self._session_id,
        )
