"""
Streamlit web frontend for Spectrum News Agent.

Main window  — important config (topics, bias categories) + analysis results.
Sidebar      — optional config (articles per bias, model info).
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import streamlit as st

from src.app_helpers import parse_topics
from src.logging_config import setup_logging

st.set_page_config(
    page_title="Spectrum News Agent",
    page_icon="📰",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Infrastructure (cached so it survives re-renders)
# ---------------------------------------------------------------------------


@st.cache_resource
def _load_infrastructure(config_path: str) -> tuple[dict[str, Any], Any]:
    """Load config + SourceRegistry once; re-runs only when config_path changes."""
    from src.config import load_config
    from src.sources import SourceRegistry

    path = Path(config_path) if config_path.strip() else None
    cfg = load_config(path)
    registry = SourceRegistry(cfg)
    return cfg, registry


# ---------------------------------------------------------------------------
# Sidebar — optional / technical settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Options")

    articles_per_bias = st.slider(
        "Articles per bias category",
        min_value=1,
        max_value=10,
        value=5,
        help="How many articles to collect per political bias category.",
    )

    st.divider()

    # API key status
    api_key_present = bool(os.environ.get("OPENROUTER_API_KEY", "").strip())
    if api_key_present:
        st.success("OpenRouter API key detected", icon="🔑")
    else:
        st.error("OPENROUTER_API_KEY not set. The agent will fail at runtime.", icon="🚨")


# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------

try:
    base_config, source_registry = _load_infrastructure("")
except Exception as exc:
    st.error(f"Failed to load configuration: {exc}")
    st.stop()

setup_logging(base_config)

# Show model selection in sidebar once config is loaded
with st.sidebar:
    st.divider()
    models_cfg: dict[str, Any] = base_config.get("openrouter", {}).get("models", {})
    _label_map = {
        "orchestrator": "Orchestrator",
        "search": "Search",
        "summarizer": "Summarizer",
        "rater": "Rater",
    }
    selected_models: dict[str, str] = {}
    with st.expander("🤖 Model Selection", expanded=False):
        for _task_key, _task_label in _label_map.items():
            _mdl = models_cfg.get(_task_key, {})
            _choices = _mdl.get("choices", [_mdl.get("id", "")])
            _default = _mdl.get("id", _choices[0] if _choices else "")
            _idx = _choices.index(_default) if _default in _choices else 0
            selected_models[_task_key] = st.selectbox(
                _task_label,
                options=_choices,
                index=_idx,
                key=f"model__{_task_key}",
            )

all_bias_configs: list[dict[str, str]] = source_registry.get_all_bias_configs()
output_dir: str = base_config.get("agent", {}).get("output_dir", "output")

# ---------------------------------------------------------------------------
# Main window — header
# ---------------------------------------------------------------------------

st.title("📰 Spectrum News Agent")
st.caption(
    "Collects news articles from left, center, and right-leaning sources, "
    "then summarises each article and rates it for factuality and political bias using LLMs."
)

st.divider()

# ---------------------------------------------------------------------------
# Topics — primary input
# ---------------------------------------------------------------------------

st.subheader("Topics")
topics_raw = st.text_area(
    "Enter the news topics to analyse — one per line (max 5)",
    value="AI regulation\nclimate policy",
    height=130,
    help="Each non-blank line becomes one search topic. Extra topics beyond the first 5 are ignored.",
)

# ---------------------------------------------------------------------------
# Bias category selector
# ---------------------------------------------------------------------------

st.subheader("Bias categories to include")

bias_cols = st.columns(len(all_bias_configs))
selected_bias_ids: list[str] = []

for col, bias_cfg in zip(bias_cols, all_bias_configs):
    with col:
        if st.checkbox(bias_cfg["label"], value=True, key=f"bias__{bias_cfg['id']}"):
            selected_bias_ids.append(bias_cfg["id"])

st.divider()

run_btn = st.button("🔍 Run Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Analysis execution
# ---------------------------------------------------------------------------

if run_btn:
    topics = parse_topics(topics_raw)

    if not topics:
        st.error("Please enter at least one topic.")
        st.stop()

    if not selected_bias_ids:
        st.error("Please select at least one bias category.")
        st.stop()

    if not api_key_present:
        st.error("Set the OPENROUTER_API_KEY environment variable before running.")
        st.stop()

    # Deep-copy so sidebar overrides don't mutate the cached config
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("agent", {})["articles_per_bias"] = articles_per_bias
    # Apply user-selected models
    for _task_key, _model_id in selected_models.items():
        cfg.setdefault("openrouter", {}).setdefault("models", {}).setdefault(_task_key, {})["id"] = _model_id

    result = None
    with st.status("Running analysis…", expanded=True) as status_widget:
        st.write(f"**Topics:** {', '.join(topics)}")
        st.write(f"**Bias categories:** {', '.join(selected_bias_ids)}")
        st.write(f"**Articles per category:** {articles_per_bias}")

        try:
            from src.agent import NewsAgent
            from src.article_extractor import ArticleExtractor
            from src.event_store import create_event_store
            from src.llm_service import MultiModelLLMService
            from src.report_compiler import ReportCompiler

            event_store = create_event_store(cfg)
            llm_service = MultiModelLLMService(config=cfg, event_store=event_store)
            article_extractor = ArticleExtractor()
            report_compiler = ReportCompiler()

            agent = NewsAgent(
                config=cfg,
                llm_service=llm_service,
                source_registry=source_registry,
                article_extractor=article_extractor,
                report_compiler=report_compiler,
                event_store=event_store,
                output_dir=output_dir,
            )

            def _on_progress(msg: str) -> None:
                st.write(msg)

            result = agent.run(topics=topics, biases=selected_bias_ids, on_progress=_on_progress)
            event_store.close()

            if result.error:
                status_widget.update(label="Analysis failed", state="error")
                st.error(f"Agent error: {result.error}")
            else:
                status_widget.update(label="Analysis complete!", state="complete")

        except Exception as exc:
            status_widget.update(label="Analysis failed", state="error")
            st.error(f"Unexpected error: {exc}")
            st.stop()

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------

    if result and not result.error:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total articles", result.total_articles)
        col_b.metric("Bias categories processed", len(result.biases_processed))
        col_c.metric("Topics analysed", len(topics))

        if result.output_path:
            st.caption(f"Report saved → `{result.output_path}`")

        report_path = Path(result.output_path) if result.output_path else None
        if report_path and report_path.exists():
            st.divider()
            st.subheader("Analysis Report")
            report_md = report_path.read_text(encoding="utf-8")
            st.markdown(report_md, unsafe_allow_html=True)
        else:
            st.warning(
                "The report file was not found. The agent may not have reached the compile_report step. "
                "Check the logs for details."
            )
