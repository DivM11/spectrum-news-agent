from __future__ import annotations

import json
import logging
import uuid
from datetime import date
from typing import Any, Callable

from src.agent_models import AgentContext, AgentResult, Context
from src.article_extractor import ArticleExtractor
from src.event_store.base import EventStore
from src.event_store.models import EventRecord, ToolCallRecord
from src.llm_service import LLMServiceProtocol
from src.report_compiler import ReportCompiler
from src.sources import SourceRegistry
from src.tools import compile_report as _t_compile
from src.tools import extract_articles as _t_extract
from src.tools import search_news as _t_search
from src.tools import summarize_articles as _t_summarize

logger = logging.getLogger(__name__)

_MAX_TOOL_CALLS = 50


class NewsAgent:
    """
    Orchestrator that drives the news collection pipeline.

    The agent uses a tool-call loop with the orchestrator LLM.  Each
    iteration the LLM can request one of four tools:
        search_news, extract_articles, summarize_articles, compile_report

    The loop continues until the LLM produces a final text message (no
    tool calls) or until the safety cap is reached.
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

        self._tool_definitions = [
            _t_search.tool_definition(),
            _t_extract.tool_definition(),
            _t_summarize.tool_definition(),
            _t_compile.tool_definition(),
        ]

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
            self._run_loop(agent_context, ctx, result, on_progress=on_progress)
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
    # Internal loop
    # ------------------------------------------------------------------

    def _run_loop(self, agent_context: AgentContext, ctx: Context, result: AgentResult, on_progress: Callable[[str], None] | None = None) -> None:
        system_prompt = self._build_system_prompt(agent_context)
        ctx.add_user_message(self._build_user_prompt(agent_context))

        messages_with_system = [{"role": "system", "content": system_prompt}] + ctx.messages

        while ctx.tool_invocations < _MAX_TOOL_CALLS:
            response = self._llm_service.call(
                messages=messages_with_system,
                task="orchestrator",
                tools=self._tool_definitions,
            )

            if not response.tool_calls:
                # Final answer — no more tool calls
                ctx.add_assistant_message(response.content)
                break

            # Process each tool call in the response
            raw_assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
                "tool_calls": response.tool_calls,
            }
            ctx.add_assistant_tool_call_message(raw_assistant_msg)

            for tool_call in response.tool_calls:
                tool_result = self._execute_tool(tool_call, agent_context, ctx, on_progress=on_progress)
                ctx.add_tool_result(tool_call["id"], tool_result)
                ctx.tool_invocations += 1

            # Rebuild messages list for next iteration
            messages_with_system = [{"role": "system", "content": system_prompt}] + ctx.messages

        # Ensure compile_report always runs even if the LLM skipped it (e.g. all tool calls failed)
        if "report_output" not in ctx.work_state:
            logger.warning(
                "LLM did not call compile_report — running fallback compilation",
                extra={"run_id": agent_context.run_id},
            )
            try:
                fallback = _t_compile.compile_report_tool(
                    arguments={"topics": list(agent_context.topics)},
                    report_compiler=self._report_compiler,
                    source_registry=self._source_registry,
                    context=ctx,
                    date=agent_context.date,
                    run_id=agent_context.run_id,
                    session_id=agent_context.session_id,
                    output_dir=self._output_dir,
                )
                ctx.work_state["report_output"] = json.loads(fallback)
            except Exception as exc:
                logger.error("Fallback compile_report failed", extra={"error": str(exc)})

        # Materialise final result
        report_output = ctx.work_state.get("report_output", {})
        result.output_path = report_output.get("output_path")
        result.total_articles = report_output.get("total_articles", 0)
        result.biases_processed = list(ctx.work_state.get("summaries", {}).keys())
        result.metadata["tool_invocations"] = ctx.tool_invocations

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _execute_tool(
        self,
        tool_call: dict[str, Any],
        agent_context: AgentContext,
        ctx: Context,
        on_progress: Callable[[str], None] | None = None,
    ) -> str:
        name: str = tool_call.get("function", {}).get("name", "")
        raw_args: str = tool_call.get("function", {}).get("arguments", "{}")

        try:
            arguments: dict[str, Any] = json.loads(raw_args)
        except json.JSONDecodeError:
            return json.dumps({"error": f"Invalid JSON arguments for tool '{name}'"})

        logger.info("Tool call", extra={"tool": name, "run_id": agent_context.run_id})
        if on_progress:
            on_progress(_progress_label(name, arguments))

        try:
            result = self._dispatch(name, arguments, agent_context, ctx)
        except Exception as exc:
            logger.error("Tool call failed", extra={"tool": name, "error": str(exc)})
            result = json.dumps({"error": f"Tool '{name}' raised: {exc}"})

        self._event_store.record_tool_call(
            ToolCallRecord(
                run_id=agent_context.run_id,
                session_id=agent_context.session_id,
                tool_name=name,
                arguments=arguments,
                result_summary=result[:500],
            )
        )

        return result

    def _dispatch(
        self,
        name: str,
        arguments: dict[str, Any],
        agent_context: AgentContext,
        ctx: Context,
    ) -> str:
        if name == "search_news":
            return _t_search.search_news_tool(
                arguments=arguments,
                config=self._config,
                llm_service=self._llm_service,
                source_registry=self._source_registry,
                context=ctx,
            )
        if name == "extract_articles":
            return _t_extract.extract_articles_tool(
                arguments=arguments,
                article_extractor=self._article_extractor,
                context=ctx,
            )
        if name == "summarize_articles":
            return _t_summarize.summarize_articles_tool(
                arguments=arguments,
                llm_service=self._llm_service,
                event_store=self._event_store,
                context=ctx,
                run_id=agent_context.run_id,
                session_id=agent_context.session_id,
            )
        if name == "compile_report":
            return _t_compile.compile_report_tool(
                arguments=arguments,
                report_compiler=self._report_compiler,
                source_registry=self._source_registry,
                context=ctx,
                date=agent_context.date,
                run_id=agent_context.run_id,
                session_id=agent_context.session_id,
                output_dir=self._output_dir,
            )
        return json.dumps({"error": f"Unknown tool: '{name}'"})

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_system_prompt(self, agent_context: AgentContext) -> str:
        template: str = self._config.get("agent", {}).get("system_prompt", _DEFAULT_SYSTEM_PROMPT)
        bias_list = ", ".join(agent_context.biases)
        return template.format(
            date=agent_context.date,
            topics=", ".join(agent_context.topics),
            biases=bias_list,
            articles_per_bias=agent_context.articles_per_bias,
        )

    def _build_user_prompt(self, agent_context: AgentContext) -> str:
        return (
            f"Please collect {agent_context.articles_per_bias} news articles per bias category "
            f"on the following topics: {', '.join(agent_context.topics)}. "
            f"Bias categories to cover: {', '.join(agent_context.biases)}. "
            f"For each category: search for articles, extract their full text, summarize and rate them, "
            f"then compile the final report. Today's date is {agent_context.date}."
        )


_DEFAULT_SYSTEM_PROMPT = (
    "You are a sophisticated news analysis agent. Today is {date}. "
    "Your task is to collect, extract, summarize, and rate news articles from sources across the political spectrum. "
    "Topics to cover: {topics}. Bias categories: {biases}. Target {articles_per_bias} articles per bias category. "
    "Use the provided tools in order: search_news → extract_articles → summarize_articles → compile_report. "
    "Always compile the report last. Do not skip any bias category unless no articles were found."
)


def _progress_label(tool_name: str, arguments: dict[str, Any]) -> str:
    """Return a human-readable progress line for the Streamlit UI."""
    bias = arguments.get("bias_id", "")
    bias_tag = f" — **{bias}**" if bias else ""
    if tool_name == "search_news":
        return f"🔍 Searching news{bias_tag}"
    if tool_name == "extract_articles":
        count = len(arguments.get("articles", []))
        return f"📄 Extracting {count} article(s){bias_tag}"
    if tool_name == "summarize_articles":
        return f"✍️ Summarising & rating articles{bias_tag}"
    if tool_name == "compile_report":
        return "📊 Compiling final report…"
    return f"⚙️ {tool_name}"
