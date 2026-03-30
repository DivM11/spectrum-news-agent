from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from openai import OpenAI

from src.event_store.base import EventStore
from src.event_store.models import LLMCallRecord
from src.event_store.null import NullEventStore

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[dict[str, Any]]
    raw_message: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    model: str
    annotations: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.annotations is None:
            self.annotations = []


@runtime_checkable
class LLMServiceProtocol(Protocol):
    def call(
        self,
        messages: list[dict[str, Any]],
        task: str,
        tools: list[dict[str, Any]] | None,
        plugins: list[dict[str, Any]] | None,
    ) -> LLMResponse: ...


class MultiModelLLMService:
    """
    Routes LLM calls to task-specific models configured in config.yml.
    All calls go through OpenRouter (OpenAI-compatible API).
    """

    def __init__(
        self,
        config: dict[str, Any],
        event_store: EventStore | None = None,
        session_id: str = "",
        run_id: str = "",
    ) -> None:
        self._config = config
        self._event_store: EventStore = event_store or NullEventStore()
        self._session_id = session_id
        self._run_id = run_id
        self._schema_version = config.get("event_store", {}).get("schema_version", _SCHEMA_VERSION)

        or_cfg = config["openrouter"]
        api_key: str = or_cfg["api"].get("key", "")
        base_url: str = or_cfg["api"]["base_url"]
        http_referer: str = or_cfg["api"].get("http_referer", "")
        x_title: str = or_cfg["api"].get("x_title", "")

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={
                "HTTP-Referer": http_referer,
                "X-Title": x_title,
            },
        )
        self._models: dict[str, Any] = or_cfg.get("models", {})

    def call(
        self,
        messages: list[dict[str, Any]],
        task: str = "orchestrator",
        tools: list[dict[str, Any]] | None = None,
        plugins: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        model_cfg = self._resolve_model(task)
        model_id: str = model_cfg["id"]
        max_tokens: int = model_cfg.get("max_tokens", 2048)
        temperature: float = model_cfg.get("temperature", 0.3)

        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if plugins:
            kwargs["extra_body"] = {"plugins": plugins}

        start = time.monotonic()
        status = "ok"
        error_msg: str | None = None
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            logger.exception("LLM call failed", extra={"task": task, "model": model_id})
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000.0
            self._record(
                task=task,
                model=model_id,
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0)
                if status == "ok"
                else 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0)
                if status == "ok"
                else 0,
                latency_ms=latency_ms,
                status=status,
                error=error_msg,
            )

        choice = response.choices[0]
        message = choice.message

        raw_message: dict[str, Any] = {"role": "assistant", "content": message.content}

        tool_calls: list[dict[str, Any]] = []
        if message.tool_calls:
            tcs = []
            for tc in message.tool_calls:
                tcs.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
            raw_message["tool_calls"] = tcs
            tool_calls = tcs

        # Extract web-search annotations if present (OpenRouter Exa plugin)
        annotations: list[dict[str, Any]] = []
        try:
            raw_msg_annotations = getattr(message, "annotations", None)
            if raw_msg_annotations:
                annotations = [
                    a if isinstance(a, dict)
                    else (a.model_dump() if hasattr(a, "model_dump") else vars(a))
                    for a in raw_msg_annotations
                ]
        except Exception:
            annotations = []
        if not annotations:
            try:
                raw_resp = getattr(response, "__dict__", {})
                for key in ("annotations", "web_search_citations", "citations"):
                    ann = raw_resp.get(key) or getattr(response, key, None)
                    if ann and isinstance(ann, list):
                        annotations = [
                            a if isinstance(a, dict)
                            else (a.model_dump() if hasattr(a, "model_dump") else vars(a))
                            for a in ann
                        ]
                        if annotations:
                            break
            except Exception:
                pass

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            raw_message=raw_message,
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            model=model_id,
            annotations=annotations,
        )

    def _resolve_model(self, task: str) -> dict[str, Any]:
        if task in self._models:
            return self._models[task]  # type: ignore[return-value]
        # Fallback to orchestrator
        return self._models.get("orchestrator", {"id": "openai/gpt-4o-mini", "max_tokens": 1024})  # type: ignore[return-value]

    def _record(
        self,
        task: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        status: str,
        error: str | None,
    ) -> None:
        try:
            record = LLMCallRecord(
                id=str(uuid.uuid4()),
                session_id=self._session_id,
                run_id=self._run_id,
                timestamp=datetime.utcnow().isoformat(timespec="milliseconds"),
                schema_version=self._schema_version,
                task=task,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                status=status,
                error=error,
            )
            self._event_store.record_llm_call(record)
        except Exception:
            logger.exception("Failed to record LLM call to event store")
