from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.schemas import ArticleSummary, BiasReport, DailyReport, FACTUALITY_SCORES, ReportOutput

logger = logging.getLogger(__name__)

_FACTUALITY_LABEL_MAP: dict[str, str] = {
    "very_high": "Very High",
    "high": "High",
    "mostly_factual": "Mostly Factual",
    "mixed": "Mixed",
    "low": "Low",
    "very_low": "Very Low",
}


class ReportCompiler:
    def compile(
        self,
        bias_reports: list[BiasReport],
        topics: list[str],
        date: str,
        run_id: str,
        session_id: str,
    ) -> ReportOutput:
        total_articles = sum(len(r.articles) for r in bias_reports)

        daily = DailyReport(
            date=date,
            topics=topics,
            bias_reports=bias_reports,
            total_articles=total_articles,
            run_id=run_id,
            session_id=session_id,
        )

        big_doc = self._build_big_doc(daily)
        return ReportOutput(daily_report=daily, big_doc_markdown=big_doc, output_path="")

    def save(self, output: ReportOutput, output_dir: str) -> str:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{output.daily_report.date}.md"
        path = directory / filename
        path.write_text(output.big_doc_markdown, encoding="utf-8")
        output.output_path = str(path)
        logger.info("Report saved", extra={"path": str(path)})
        return str(path)

    def _build_big_doc(self, daily: DailyReport) -> str:
        lines: list[str] = []
        lines.append(f"# Spectrum News Report — {daily.date}")
        lines.append(f"**Topics**: {', '.join(daily.topics)}")
        lines.append(f"**Total articles**: {daily.total_articles}")
        lines.append(f"**Run ID**: {daily.run_id}")
        lines.append("")
        lines.append("---")
        lines.append("")

        for bias_report in daily.bias_reports:
            lines.append(f"## {bias_report.bias_label}")
            lines.append(
                f"*{len(bias_report.articles)} articles | "
                f"Avg source factuality: {bias_report.avg_source_factuality_score:.2f}*"
            )
            lines.append("")

            if not bias_report.articles:
                lines.append("*No articles collected for this bias category.*")
                lines.append("")
                continue

            for i, summary in enumerate(bias_report.articles, 1):
                lines.extend(self._render_article(i, summary))

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _render_article(self, index: int, summary: ArticleSummary) -> list[str]:
        art = summary.article
        source_fact_label = _FACTUALITY_LABEL_MAP.get(art.source_factuality, art.source_factuality)
        summary_body = summary.summary or "*Article text could not be extracted — no summary available.*"
        lines: list[str] = [
            f"### {index}. {art.title or art.url or '(unknown)'}",
            f"**Source**: [{art.source_name or art.source_domain or 'Unknown'}]({art.url})  ",
            f"**Source Factuality (MBFC)**: {source_fact_label}  ",
            f"**LLM Factuality Rating**: {summary.llm_factuality_rating.capitalize()} "
            f"(confidence: {summary.llm_factuality_confidence:.0%})  ",
            f"**LLM Bias Rating**: {summary.llm_bias_rating.capitalize()} "
            f"(confidence: {summary.llm_bias_confidence:.0%})  ",
            f"**Word Count**: {art.word_count}  ",
            f"**Extraction Status**: {art.extraction_status}",
            "",
            "#### Summary",
            summary_body,
            "",
        ]

        if summary.key_claims:
            lines.append("#### Key Claims")
            for claim in summary.key_claims:
                lines.append(f"- {claim}")
            lines.append("")

        if summary.topics_covered:
            lines.append(f"**Topics covered**: {', '.join(summary.topics_covered)}")
            lines.append("")

        if art.full_text:
            lines.append("<details>")
            lines.append("<summary>Full Article Text</summary>")
            lines.append("")
            lines.append(art.full_text)
            lines.append("")
            lines.append("</details>")
            lines.append("")

        lines.append("---")
        lines.append("")
        return lines

    def build_summary_json(self, daily: DailyReport) -> dict[str, Any]:
        return {
            "date": daily.date,
            "topics": daily.topics,
            "run_id": daily.run_id,
            "session_id": daily.session_id,
            "total_articles": daily.total_articles,
            "bias_reports": [
                {
                    "bias_id": r.bias_id,
                    "bias_label": r.bias_label,
                    "article_count": len(r.articles),
                    "avg_source_factuality_score": r.avg_source_factuality_score,
                    "articles": [
                        {
                            "url": s.article.url,
                            "title": s.article.title,
                            "source_name": s.article.source_name,
                            "source_factuality": s.article.source_factuality,
                            "llm_factuality_rating": s.llm_factuality_rating,
                            "llm_factuality_confidence": s.llm_factuality_confidence,
                            "llm_bias_rating": s.llm_bias_rating,
                            "llm_bias_confidence": s.llm_bias_confidence,
                            "summary": s.summary,
                            "key_claims": s.key_claims,
                            "topics_covered": s.topics_covered,
                        }
                        for s in r.articles
                    ],
                }
                for r in daily.bias_reports
            ],
        }
