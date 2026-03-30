from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pytest

from src.event_store.models import LLMCallRecord
from src.event_store.null import NullEventStore
from src.llm_service import LLMResponse, LLMServiceProtocol, MultiModelLLMService


# ---------------------------------------------------------------------------
# Helpers / Mock classes
# ---------------------------------------------------------------------------


def _config(orchestrator_id: str = "test-model", summarizer_id: str = "test-summarizer") -> dict[str, Any]:
    return {
        "openrouter": {
            "api": {
                "key": "sk-test",
                "base_url": "https://openrouter.ai/api/v1",
                "http_referer": "",
                "x_title": "",
            },
            "models": {
                "orchestrator": {"id": orchestrator_id, "max_tokens": 100, "temperature": 0.2},
                "summarizer": {"id": summarizer_id, "max_tokens": 200, "temperature": 0.3},
                "rater": {"id": "test-rater", "max_tokens": 100, "temperature": 0.1},
                "search": {"id": "test-search", "max_tokens": 100, "temperature": 0.3},
            },
        },
        "event_store": {"schema_version": 1},
    }


class CaptureLLMCallStore(NullEventStore):
    def __init__(self) -> None:
        self.calls: list[LLMCallRecord] = []

    def record_llm_call(self, record: LLMCallRecord) -> None:
        self.calls.append(record)


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_multi_model_service_satisfies_protocol() -> None:
    service = MultiModelLLMService(_config(), NullEventStore())
    assert isinstance(service, LLMServiceProtocol)


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def test_resolve_model_known_task(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MultiModelLLMService(_config(), NullEventStore())
    model_cfg = service._resolve_model("summarizer")
    assert model_cfg["id"] == "test-summarizer"


def test_resolve_model_unknown_task_falls_back_to_orchestrator(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MultiModelLLMService(_config(), NullEventStore())
    model_cfg = service._resolve_model("nonexistent_task")
    assert model_cfg["id"] == "test-model"


# ---------------------------------------------------------------------------
# LLM call recording
# ---------------------------------------------------------------------------


def test_call_records_llm_call_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    store = CaptureLLMCallStore()
    service = MultiModelLLMService(_config(), store, session_id="sess-1", run_id="run-1")

    def fake_create(**kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion('{"result": "ok"}')

    monkeypatch.setattr(service._client.chat.completions, "create", fake_create)

    service.call([{"role": "user", "content": "test"}], task="orchestrator")

    assert len(store.calls) == 1
    assert store.calls[0].task == "orchestrator"
    assert store.calls[0].status == "ok"
    assert store.calls[0].session_id == "sess-1"


def test_call_records_error_status_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    store = CaptureLLMCallStore()
    service = MultiModelLLMService(_config(), store, session_id="sess-1", run_id="run-1")

    def fail_create(**kwargs: Any) -> None:
        raise RuntimeError("API down")

    monkeypatch.setattr(service._client.chat.completions, "create", fail_create)

    with pytest.raises(RuntimeError, match="API down"):
        service.call([{"role": "user", "content": "test"}], task="orchestrator")

    assert len(store.calls) == 1
    assert store.calls[0].status == "error"
    assert "API down" in (store.calls[0].error or "")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_call_returns_llm_response(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MultiModelLLMService(_config(), NullEventStore())

    def fake_create(**kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion("Hello response")

    monkeypatch.setattr(service._client.chat.completions, "create", fake_create)

    response = service.call([{"role": "user", "content": "hi"}], task="orchestrator")

    assert isinstance(response, LLMResponse)
    assert response.content == "Hello response"
    assert response.tool_calls == []


def test_call_with_tool_calls_parses_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MultiModelLLMService(_config(), NullEventStore())

    class _FakeFunction:
        name = "search_news"
        arguments = '{"bias": "left", "topics": ["Politics"]}'

    class _FakeTc:
        id = "tc-1"
        type = "function"
        function = _FakeFunction()

    class _FakeMsgWithTc:
        content = None
        tool_calls = [_FakeTc()]

    class _FakeChoiceWithTc:
        message = _FakeMsgWithTc()

    class _FakeCompletionWithTc:
        choices = [_FakeChoiceWithTc()]
        usage = _FakeUsage()

    monkeypatch.setattr(service._client.chat.completions, "create", lambda **k: _FakeCompletionWithTc())

    response = service.call([{"role": "user", "content": "go"}], task="orchestrator")
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["function"]["name"] == "search_news"
