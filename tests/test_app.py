"""Tests for the Streamlit frontend (app.py) and its pure helpers.

Structure
---------
1. Pure-logic tests — exercise ``src.app_helpers.parse_topics`` with no Streamlit involved.
2. UI tests — use ``streamlit.testing.v1.AppTest`` to run the full app script and
   assert on the rendered widget tree.

Mocking strategy
----------------
* ``_load_infrastructure`` calls ``load_config`` + ``SourceRegistry``.  Both work against
  the real ``config.yml`` in the Docker test environment so no patching is needed there.
* Agent infrastructure (``NewsAgent``, ``MultiModelLLMService``, ``create_event_store``,
  ``ArticleExtractor``, ``ReportCompiler``) is patched for any test that clicks the
  "Run Analysis" button, so no real API calls are made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from streamlit.testing.v1 import AppTest

from src.agent_models import AgentResult
from src.app_helpers import parse_topics

# Absolute path so the tests find app.py regardless of the working directory.
_APP_PATH = str(Path(__file__).parent.parent / "app.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    *,
    error: str | None = None,
    total_articles: int = 4,
    biases_processed: list[str] | None = None,
    output_path: str | None = None,
) -> AgentResult:
    result = AgentResult(session_id="sess", run_id="run", topics=["AI regulation"])
    result.error = error
    result.total_articles = total_articles
    result.biases_processed = biases_processed or ["left", "center"]
    result.output_path = output_path
    return result


def _agent_patches(mock_result: AgentResult) -> tuple:
    """Return a tuple of patch context managers for all agent infrastructure."""
    mock_agent = MagicMock()
    mock_agent.run.return_value = mock_result
    return (
        patch("src.agent.NewsAgent", return_value=mock_agent),
        patch("src.event_store.create_event_store", return_value=MagicMock()),
        patch("src.llm_service.MultiModelLLMService", return_value=MagicMock()),
        patch("src.article_extractor.ArticleExtractor", return_value=MagicMock()),
        patch("src.report_compiler.ReportCompiler", return_value=MagicMock()),
    )


# ---------------------------------------------------------------------------
# 1. Pure-logic tests — parse_topics
# ---------------------------------------------------------------------------

def test_parse_topics_basic() -> None:
    assert parse_topics("AI regulation\nclimate policy") == ["AI regulation", "climate policy"]


def test_parse_topics_caps_at_five() -> None:
    raw = "\n".join(f"topic {i}" for i in range(8))
    result = parse_topics(raw)
    assert len(result) == 5
    assert result[0] == "topic 0"
    assert result[4] == "topic 4"


def test_parse_topics_strips_whitespace() -> None:
    assert parse_topics("  economy  \n  healthcare  ") == ["economy", "healthcare"]


def test_parse_topics_filters_blank_lines() -> None:
    assert parse_topics("AI\n\n\nclimate\n") == ["AI", "climate"]


def test_parse_topics_empty_string_returns_empty_list() -> None:
    assert parse_topics("") == []


def test_parse_topics_whitespace_only_returns_empty_list() -> None:
    assert parse_topics("   \n   \n   ") == []


def test_parse_topics_single_topic() -> None:
    assert parse_topics("immigration") == ["immigration"]


def test_parse_topics_exactly_five_preserved() -> None:
    raw = "\n".join(f"topic {i}" for i in range(5))
    assert len(parse_topics(raw)) == 5


# ---------------------------------------------------------------------------
# 2. UI tests — AppTest
# ---------------------------------------------------------------------------

class TestAppRendering:
    """The app should render without errors given a valid config."""

    def test_renders_without_exception(self) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.run()
        assert not at.exception

    def test_shows_title(self) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.run()
        assert any("Spectrum News Agent" in t.value for t in at.title)

    def test_shows_topics_text_area(self) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.run()
        assert len(at.text_area) >= 1

    def test_text_area_has_default_topics(self) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.run()
        assert at.text_area[0].value.strip() != ""

    def test_shows_run_button(self) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.run()
        buttons_with_run = [b for b in at.button if "Run Analysis" in b.label]
        assert len(buttons_with_run) == 1

    def test_shows_bias_checkboxes(self) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.run()
        # config.yml defines 3 bias categories
        assert len(at.checkbox) == 3

    def test_bias_checkboxes_default_to_true(self) -> None:
        at = AppTest.from_file(_APP_PATH)
        at.run()
        assert all(cb.value for cb in at.checkbox)


class TestSidebarApiKeyStatus:
    """The sidebar should reflect whether OPENROUTER_API_KEY is set."""

    def test_shows_warning_when_key_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        at = AppTest.from_file(_APP_PATH)
        at.run()
        assert len(at.sidebar.error) >= 1

    def test_shows_success_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        at = AppTest.from_file(_APP_PATH)
        at.run()
        assert len(at.sidebar.success) >= 1

    def test_no_main_window_error_on_load(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        at = AppTest.from_file(_APP_PATH)
        at.run()
        # The API key error is in the sidebar only — all at.error entries should be sidebar entries
        assert len(at.error) == len(at.sidebar.error)


class TestRunValidation:
    """Clicking Run Analysis with invalid inputs should show errors, not call the agent."""

    def test_empty_topics_shows_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        at = AppTest.from_file(_APP_PATH)
        at.run()
        at.text_area[0].set_value("")
        at.button[0].click()
        at.run()
        assert any("topic" in e.value.lower() for e in at.error)

    def test_no_biases_selected_shows_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        at = AppTest.from_file(_APP_PATH)
        at.run()
        for cb in at.checkbox:
            cb.set_value(False)
        at.button[0].click()
        at.run()
        assert any("bias" in e.value.lower() for e in at.error)

    def test_missing_api_key_shows_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        at = AppTest.from_file(_APP_PATH)
        at.run()
        # Topics and biases are valid by default; only the key is missing
        at.button[0].click()
        at.run()
        assert any("OPENROUTER_API_KEY" in e.value for e in at.error)

    def test_empty_topics_does_not_call_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        mock_agent = MagicMock()
        with patch("src.agent.NewsAgent", return_value=mock_agent):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.text_area[0].set_value("")
            at.button[0].click()
            at.run()
        mock_agent.run.assert_not_called()


class TestSuccessfulRun:
    """When the agent succeeds the UI should show metrics and an optional report."""

    def _run_with_mocked_agent(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_result: AgentResult,
    ) -> AppTest:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        patches = _agent_patches(mock_result)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.button[0].click()
            at.run()
        return at

    def test_shows_total_articles_metric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _make_result(total_articles=7)
        at = self._run_with_mocked_agent(monkeypatch, result)
        assert any(m.value == "7" for m in at.metric)

    def test_shows_biases_processed_metric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _make_result(biases_processed=["left", "center", "right"])
        at = self._run_with_mocked_agent(monkeypatch, result)
        assert any(m.value == "3" for m in at.metric)

    def test_shows_topics_analysed_metric(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _make_result()
        at = self._run_with_mocked_agent(monkeypatch, result)
        # Default text area has 2 topics
        assert any(m.value == "2" for m in at.metric)

    def test_three_metrics_rendered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _make_result()
        at = self._run_with_mocked_agent(monkeypatch, result)
        assert len(at.metric) >= 3

    def test_no_error_messages_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _make_result()
        at = self._run_with_mocked_agent(monkeypatch, result)
        assert len(at.error) == 0


class TestFailedRun:
    """When the agent returns an error the UI should report it clearly."""

    def test_shows_agent_error_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        result = _make_result(error="LLM API unavailable")
        patches = _agent_patches(result)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.button[0].click()
            at.run()
        assert any("LLM API unavailable" in e.value for e in at.error)

    def test_no_metrics_shown_on_agent_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        result = _make_result(error="timeout")
        patches = _agent_patches(result)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.button[0].click()
            at.run()
        assert len(at.metric) == 0

    def test_shows_unexpected_exception_as_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("connection refused")
        with (
            patch("src.agent.NewsAgent", return_value=mock_agent),
            patch("src.event_store.create_event_store", return_value=MagicMock()),
            patch("src.llm_service.MultiModelLLMService", return_value=MagicMock()),
            patch("src.article_extractor.ArticleExtractor", return_value=MagicMock()),
            patch("src.report_compiler.ReportCompiler", return_value=MagicMock()),
        ):
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.button[0].click()
            at.run()
        assert any("connection refused" in e.value for e in at.error)


class TestReportDisplay:
    """The Markdown report file should be read and rendered when it exists."""

    def test_report_rendered_when_file_exists(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        report_file = tmp_path / "2026-03-28.md"
        report_file.write_text("# Daily Report\n\nSome content here.", encoding="utf-8")

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        result = _make_result(output_path=str(report_file))
        patches = _agent_patches(result)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.button[0].click()
            at.run()
        # The report subheader should be present
        assert any("Analysis Report" in h.value for h in at.subheader)

    def test_warning_shown_when_report_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        nonexistent = str(tmp_path / "does_not_exist.md")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        result = _make_result(output_path=nonexistent)
        patches = _agent_patches(result)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            at = AppTest.from_file(_APP_PATH)
            at.run()
            at.button[0].click()
            at.run()
        assert len(at.warning) >= 1
