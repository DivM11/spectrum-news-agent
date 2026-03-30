from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """
    Extract the first JSON object from a string (LLM response).
    Returns None if no valid JSON object is found.
    """
    # Try direct parse first
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # Look for ```json ... ``` code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    # Greedy scan for first { ... }
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    return None
    return None


def extract_json_list(text: str) -> list[Any] | None:
    """
    Extract the first JSON array from a string.
    Returns None if no valid JSON array is found.
    """
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            return json.loads(stripped)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])  # type: ignore[no-any-return]
                except json.JSONDecodeError:
                    return None
    return None


def require_field(data: dict[str, Any], field: str, expected_type: type) -> Any:
    """
    Extract a required field from a parsed dict, raising ValueError on failure.
    Use this to validate LLM JSON responses at system boundaries.
    """
    if field not in data:
        raise ValueError(f"LLM response missing required field: '{field}'")
    value = data[field]
    if not isinstance(value, expected_type):
        raise ValueError(
            f"LLM response field '{field}' expected {expected_type.__name__}, got {type(value).__name__}"
        )
    return value


def clamp_float(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    """Coerce and clamp a value to [low, high]. Returns low on failure."""
    try:
        f = float(value)
        return max(low, min(high, f))
    except (TypeError, ValueError):
        return low
