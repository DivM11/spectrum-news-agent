from __future__ import annotations

import json
import logging

import pytest

from src.logging_config import (
    get_run_id,
    get_session_id,
    set_run_id,
    set_session_id,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(level: str = "INFO", fmt: str = "json") -> dict:
    return {"logging": {"level": level, "format": fmt, "include_timestamps": True}}


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


def test_setup_logging_sets_root_level() -> None:
    setup_logging(_cfg("WARNING"))
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_clears_existing_handlers() -> None:
    root = logging.getLogger()
    root.addHandler(logging.NullHandler())
    setup_logging(_cfg())
    # After setup there should be exactly one handler (our StreamHandler)
    assert len(root.handlers) == 1


def test_setup_logging_json_format_produces_valid_json(capfd: pytest.CaptureFixture) -> None:
    setup_logging(_cfg("DEBUG", "json"))
    logger = logging.getLogger("test.json")
    logger.info("hello world")
    captured = capfd.readouterr()
    parsed = json.loads(captured.err)
    assert parsed["message"] == "hello world"
    assert parsed["level"] == "INFO"


def test_setup_logging_plain_format_does_not_crash(capfd: pytest.CaptureFixture) -> None:
    setup_logging(_cfg("INFO", "plain"))
    logger = logging.getLogger("test.plain")
    logger.info("plain message")
    captured = capfd.readouterr()
    assert "plain message" in captured.err


# ---------------------------------------------------------------------------
# session_id / run_id context vars
# ---------------------------------------------------------------------------


def test_set_and_get_session_id() -> None:
    set_session_id("sess-abc")
    assert get_session_id() == "sess-abc"


def test_set_and_get_run_id() -> None:
    set_run_id("run-xyz")
    assert get_run_id() == "run-xyz"


def test_session_id_appears_in_json_log(capfd: pytest.CaptureFixture) -> None:
    setup_logging(_cfg())
    set_session_id("sess-visible")
    set_run_id("")
    logger = logging.getLogger("test.ctx")
    logger.info("ctx test")
    captured = capfd.readouterr()
    parsed = json.loads(captured.err)
    assert parsed.get("session_id") == "sess-visible"


def test_run_id_absent_when_empty(capfd: pytest.CaptureFixture) -> None:
    setup_logging(_cfg())
    set_run_id("")
    set_session_id("")
    logger = logging.getLogger("test.empty")
    logger.info("no ids")
    captured = capfd.readouterr()
    parsed = json.loads(captured.err)
    assert "run_id" not in parsed
    assert "session_id" not in parsed
