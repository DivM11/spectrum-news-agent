from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from typing import Any

_session_id_var: ContextVar[str] = ContextVar("session_id", default="")
_run_id_var: ContextVar[str] = ContextVar("run_id", default="")


def set_session_id(session_id: str) -> None:
    _session_id_var.set(session_id)


def set_run_id(run_id: str) -> None:
    _run_id_var.set(run_id)


def get_session_id() -> str:
    return _session_id_var.get()


def get_run_id() -> str:
    return _run_id_var.get()


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        session_id = _session_id_var.get()
        run_id = _run_id_var.get()
        if session_id:
            payload["session_id"] = session_id
        if run_id:
            payload["run_id"] = run_id
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Merge any extra fields passed via extra={}
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, separators=(",", ":"), default=str)


def setup_logging(config: dict[str, Any]) -> None:
    log_cfg = config.get("logging", {})
    level_name: str = log_cfg.get("level", "INFO")
    fmt: str = log_cfg.get("format", "json")

    level = getattr(logging, level_name.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)

    if fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))

    root.addHandler(handler)
