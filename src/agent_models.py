from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentContext:
    """Immutable snapshot of agent run parameters. Used for logging and auditing."""

    topics: tuple[str, ...]
    biases: tuple[str, ...]
    articles_per_bias: int
    session_id: str
    run_id: str
    date: str


@dataclass
class AgentResult:
    """Mutable output holder populated by the agent run."""

    session_id: str
    run_id: str
    topics: list[str] = field(default_factory=list)
    biases_processed: list[str] = field(default_factory=list)
    total_articles: int = 0
    output_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Context:
    """
    Mutable state carrier for a single agent run.
    Owned by the agent, mutated by tool calls via work_state.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    work_state: dict[str, Any] = field(default_factory=dict)
    tool_invocations: int = 0

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })

    def add_assistant_tool_call_message(self, raw_message: dict[str, Any]) -> None:
        """Append the raw assistant message containing tool_calls from the LLM."""
        self.messages.append(raw_message)
