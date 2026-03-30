from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from typing import Any, Callable

from src.agent_models import AgentContext, AgentResult, Context
from src.article_extractor import ArticleExtractor
from src.event_store.base import EventStore
from src.event_store.models import EventRecord
from src.llm_service import LLMServiceProtocol
from src.pipeline import NewsBiasPipeline, ThreadedPipelineRunner
from src.report_compiler import ReportCompiler
from src.sources import SourceRegistry
from src.tools import compile_report as _t_compile

logger = logging.getLogger(__name__)


class NewsAgent:
    """
    Orchestrates the news collection pipeline.

    All bias categories are processed concurrently:
        search_news → extract_articles → summarize_articles  (per bias, in parallel)

    Once all biases are done, compile_report consolidates the results.
    """

    def __init__(
        self,
        config: dict[str, Any],
        llm_service: LLMServiceProtocol,
        source_registry: SourceRegistry,
        article_extractor: ArticleExtractor,
        report_compiler: ReportCompiler,
        event_store: EventStore,
        output_dir: str = "output",
    ) -> None:
        self._config = config
        self._llm_service = llm_service
        self._source_registry = source_registry
        self._article_extractor = article_extractor
        self._report_compiler = report_compiler
        self._event_store = event_store
        self._output_dir = output_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, topics: list[str], biases: list[str] | None = None, on_progress: Callable[[str], None] | None = None) -> AgentResult:
        run_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        today = date.today().isoformat()

        if biases is None:
            biases = self._source_registry.get_all_biases()

        agent_context = AgentContext(
            topics=tuple(topics),
            biases=tuple(biases),
            articles_per_bias=self._config.get("agent", {}).get("articles_per_bias", 5),
            session_id=session_id,
            run_id=run_id,
            date=today,
        )
        result = AgentResult(session_id=session_id, run_id=run_id, topics=list(topics))
        ctx = Context()

        logger.info(
            "NewsAgent run started",
            extra={"run_id": run_id, "topics": topics, "biases": biases},
        )

        self._event_store.record_event(
            EventRecord(
                run_id=run_id,
                session_id=session_id,
                event_type="run_started",
                payload={"topics": topics, "biases": biases},
            )
        )

        try:
            self._run_pipeline(agent_context, ctx, result, on_progress)
        except Exception as exc:
            logger.error("NewsAgent run failed", extra={"run_id": run_id, "error": str(exc)})
            result.error = str(exc)

        self._event_store.record_event(
            EventRecord(
                run_id=run_id,
                session_id=session_id,
                event_type="run_finished",
                payload={"error": result.error, "total_articles": result.total_articles},
            )
        )

        logger.info(
            "NewsAgent run finished",
            extra={"run_id": run_id, "total_articles": result.total_articles},
        )
        return result

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        agent_context: AgentContext,
        ctx: Context,
        result: AgentResult,
        on_progress: Callable[[str], None] | None,
    ) -> None:
        biases = list(agent_context.biases)
        if on_progress:
            label = "y" if len(biases) == 1 else "ies"
            on_progress(f"🔄 Processing **{len(biases)}** bias categor{label} in parallel…")

        pipeline = NewsBiasPipeline(
            config=self._config,
            llm_service=self._llm_service,
            source_registry=self._source_registry,
            article_extractor=self._article_extractor,
            event_store=self._event_store,
            topics=list(agent_context.topics),
            run_id=agent_context.run_id,
            session_id=agent_context.session_id,
        )
        runner = ThreadedPipelineRunner(max_workers=len(biases))
        runner.run(biases, ctx, pipeline)

        if on_progress:
            on_progress("📊 Compiling final report…")

        compile_raw = _t_compile.compile_report_tool(
            arguments={"topics": list(agent_context.topics)},
            report_compiler=self._report_compiler,
            source_registry=self._source_registry,
            context=ctx,
            date=agent_context.date,
            run_id=agent_context.run_id,
            session_id=agent_context.session_id,
            output_dir=self._output_dir,
        )
        compile_output = json.loads(compile_raw)
        result.output_path = compile_output.get("output_path")
        result.total_articles = compile_output.get("total_articles", 0)
        result.biases_processed = list(ctx.work_state.get("summaries", {}).keys())
        result.metadata["bias_count"] = len(biases)
