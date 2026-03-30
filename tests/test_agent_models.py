from __future__ import annotations

from src.agent_models import AgentContext, AgentResult, Context


# ---------------------------------------------------------------------------
# AgentContext (frozen dataclass)
# ---------------------------------------------------------------------------


def test_agent_context_fields() -> None:
    ctx = AgentContext(
        topics=("Politics", "Economy"),
        biases=("left", "center"),
        articles_per_bias=5,
        session_id="sess-1",
        run_id="run-1",
        date="2026-03-28",
    )
    assert ctx.topics == ("Politics", "Economy")
    assert ctx.articles_per_bias == 5


def test_agent_context_is_immutable() -> None:
    import pytest

    ctx = AgentContext(
        topics=("Tech",),
        biases=("center",),
        articles_per_bias=3,
        session_id="s",
        run_id="r",
        date="2026-01-01",
    )
    with pytest.raises((AttributeError, TypeError)):
        ctx.session_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------


def test_agent_result_defaults() -> None:
    result = AgentResult(session_id="sess-1", run_id="run-1")
    assert result.topics == []
    assert result.biases_processed == []
    assert result.total_articles == 0
    assert result.output_path is None
    assert result.error is None


def test_agent_result_mutable() -> None:
    result = AgentResult(session_id="s", run_id="r")
    result.total_articles = 15
    result.output_path = "/tmp/report.md"
    assert result.total_articles == 15
    assert result.output_path == "/tmp/report.md"


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def test_context_starts_empty() -> None:
    ctx = Context()
    assert ctx.messages == []
    assert ctx.work_state == {}
    assert ctx.tool_invocations == 0


def test_context_add_user_message() -> None:
    ctx = Context()
    ctx.add_user_message("Hello")
    assert len(ctx.messages) == 1
    assert ctx.messages[0] == {"role": "user", "content": "Hello"}


def test_context_add_assistant_message() -> None:
    ctx = Context()
    ctx.add_assistant_message("I will help.")
    assert ctx.messages[0]["role"] == "assistant"


def test_context_add_tool_result() -> None:
    ctx = Context()
    ctx.add_tool_result(tool_call_id="tc-1", content="Done")
    msg = ctx.messages[0]
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "tc-1"
    assert msg["content"] == "Done"


def test_context_add_assistant_tool_call_message() -> None:
    ctx = Context()
    raw = {"role": "assistant", "content": None, "tool_calls": [{"id": "tc-1"}]}
    ctx.add_assistant_tool_call_message(raw)
    assert ctx.messages[0] is raw


def test_context_work_state_mutation() -> None:
    ctx = Context()
    ctx.work_state["found_articles"] = [{"url": "https://example.com"}]
    assert len(ctx.work_state["found_articles"]) == 1
