from __future__ import annotations

import json
import logging
from typing import Any

from src.agent_models import Context
from src.report_compiler import ReportCompiler
from src.schemas import ArticleRecord, ArticleSummary, BiasReport, FACTUALITY_SCORES
from src.sources import SourceRegistry

logger = logging.getLogger(__name__)

_TOOL_NAME = "compile_report"


def tool_definition() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": _TOOL_NAME,
            "description": (
                "Compile all bias summaries into a final daily report. "
                "Builds a JSON summary and an exhaustive markdown document, then saves to disk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The topics that were searched.",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save the markdown output file.",
                        "default": "output",
                    },
                },
                "required": ["topics"],
            },
        },
    }


def compile_report_tool(
    arguments: dict[str, Any],
    report_compiler: ReportCompiler,
    source_registry: SourceRegistry,
    context: Context,
    date: str,
    run_id: str,
    session_id: str,
    output_dir: str = "output",
) -> str:
    topics: list[str] = arguments.get("topics", [])
    output_dir = arguments.get("output_dir", output_dir)

    summaries_by_bias: dict[str, list[dict[str, Any]]] = context.work_state.get("summaries", {})
    if not summaries_by_bias:
        logger.warning("No summaries in work state — compiling empty report")
    else:
        total = sum(len(v) for v in summaries_by_bias.values())
        logger.info(
            "compile_report started",
            extra={"biases": list(summaries_by_bias.keys()), "total_articles": total},
        )

    bias_reports: list[BiasReport] = []

    for bias_id in source_registry.get_all_biases():
        bias_label = source_registry.get_bias_label(bias_id)
        summary_dicts = summaries_by_bias.get(bias_id, [])

        article_summaries: list[ArticleSummary] = []
        factuality_scores: list[float] = []

        for s in summary_dicts:
            art_dict = s.get("article", {})
            record = ArticleRecord(
                url=art_dict.get("url", ""),
                title=art_dict.get("title", ""),
                source_name=art_dict.get("source_name", ""),
                source_domain=art_dict.get("source_domain", ""),
                bias=art_dict.get("bias", bias_id),
                source_factuality=art_dict.get("source_factuality", "mixed"),
                full_text=art_dict.get("full_text"),
                word_count=int(art_dict.get("word_count", 0)),
                extraction_status=art_dict.get("extraction_status", "failed"),
            )
            article_summaries.append(
                ArticleSummary(
                    article=record,
                    summary=s.get("summary", ""),
                    llm_factuality_rating=s.get("llm_factuality_rating", "mixed"),
                    llm_factuality_confidence=float(s.get("llm_factuality_confidence", 0.5)),
                    llm_bias_rating=s.get("llm_bias_rating", "unknown"),
                    llm_bias_confidence=float(s.get("llm_bias_confidence", 0.5)),
                    key_claims=list(s.get("key_claims", [])),
                    topics_covered=list(s.get("topics_covered", [])),
                )
            )
            score = FACTUALITY_SCORES.get(record.source_factuality, 0.4)
            factuality_scores.append(score)

        avg_factuality = sum(factuality_scores) / len(factuality_scores) if factuality_scores else 0.0

        bias_reports.append(
            BiasReport(
                bias_id=bias_id,
                bias_label=bias_label,
                articles=article_summaries,
                source_count=len(article_summaries),
                avg_source_factuality_score=avg_factuality,
            )
        )

    report_output = report_compiler.compile(
        bias_reports=bias_reports,
        topics=topics,
        date=date,
        run_id=run_id,
        session_id=session_id,
    )
    saved_path = report_compiler.save(report_output, output_dir)

    summary_json = report_compiler.build_summary_json(report_output.daily_report)

    context.work_state["report_output"] = {
        "output_path": saved_path,
        "total_articles": report_output.daily_report.total_articles,
        "bias_count": len(bias_reports),
    }

    logger.info("compile_report completed", extra={"path": saved_path})
    return json.dumps(
        {
            "output_path": saved_path,
            "total_articles": report_output.daily_report.total_articles,
            "summary": summary_json,
        }
    )
