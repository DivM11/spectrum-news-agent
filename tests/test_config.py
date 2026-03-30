from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import _deep_get, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_config(tmp_path: Path, key_env_var: str = "OPENROUTER_API_KEY") -> Path:
    p = tmp_path / "config.yml"
    p.write_text(
        f"""
app:
  title: "Test Agent"
openrouter:
  api:
    key_env_var: "{key_env_var}"
    base_url: "https://openrouter.ai/api/v1"
  models:
    orchestrator:
      id: "test-orchestrator"
      max_tokens: 100
      temperature: 0.2
    search:
      id: "test-search"
      max_tokens: 100
      temperature: 0.3
    summarizer:
      id: "test-summarizer"
      max_tokens: 200
      temperature: 0.3
    rater:
      id: "test-rater"
      max_tokens: 100
      temperature: 0.1
agent:
  articles_per_bias: 3
  max_tool_rounds: 10
  max_topics: 5
  system_prompt: "test"
event_store:
  backend: "sqlite"
  enabled: true
  schema_version: 1
  sqlite:
    db_path: "data/test.db"
biases: []
sources: []
default_topics: []
output:
  directory: "output"
"""
    )
    return p


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_returns_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg_path = _minimal_config(tmp_path)
    result = load_config(cfg_path)
    assert isinstance(result, dict)


def test_load_config_injects_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-injected")
    cfg_path = _minimal_config(tmp_path)
    result = load_config(cfg_path)
    assert result["openrouter"]["api"]["key"] == "sk-injected"


def test_load_config_missing_env_var_sets_empty_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg_path = _minimal_config(tmp_path)
    result = load_config(cfg_path)
    assert result["openrouter"]["api"]["key"] == ""


def test_load_config_missing_required_key_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yml"
    p.write_text("app:\n  title: test\n")
    with pytest.raises(ValueError, match="Missing required config key"):
        load_config(p)


def test_load_config_preserves_nested_structure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    cfg_path = _minimal_config(tmp_path)
    result = load_config(cfg_path)
    assert result["openrouter"]["models"]["search"]["id"] == "test-search"
    assert result["openrouter"]["models"]["summarizer"]["id"] == "test-summarizer"
    assert result["openrouter"]["models"]["rater"]["id"] == "test-rater"
    assert result["agent"]["articles_per_bias"] == 3


# ---------------------------------------------------------------------------
# _deep_get
# ---------------------------------------------------------------------------


def test_deep_get_existing_key() -> None:
    data = {"a": {"b": {"c": 42}}}
    assert _deep_get(data, "a.b.c") == 42


def test_deep_get_missing_key_returns_none() -> None:
    data = {"a": {"b": 1}}
    assert _deep_get(data, "a.x.y") is None


def test_deep_get_top_level_key() -> None:
    data = {"foo": "bar"}
    assert _deep_get(data, "foo") == "bar"


def test_deep_get_empty_dict() -> None:
    assert _deep_get({}, "a.b") is None
