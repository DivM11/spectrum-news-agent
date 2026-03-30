from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


_CONFIG_PATH = Path(__file__).parent.parent / "config.yml"

_REQUIRED_KEYS = [
    "openrouter.api.key_env_var",
    "openrouter.models.orchestrator.id",
    "openrouter.models.summarizer.id",
    "agent.articles_per_bias",
    "event_store.backend",
]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _CONFIG_PATH
    cfg = OmegaConf.load(path)

    api_key_env = OmegaConf.select(cfg, "openrouter.api.key_env_var")
    if api_key_env:
        api_key = os.environ.get(api_key_env, "")
        OmegaConf.update(cfg, "openrouter.api.key", api_key, merge=True)

    result: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    _validate(result)
    return result


def _validate(config: dict[str, Any]) -> None:
    for key_path in _REQUIRED_KEYS:
        value = _deep_get(config, key_path)
        if value is None:
            raise ValueError(f"Missing required config key: {key_path}")


def _deep_get(config: dict[str, Any], key_path: str) -> Any:
    parts = key_path.split(".")
    current: Any = config
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current
