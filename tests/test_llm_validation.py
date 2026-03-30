from __future__ import annotations

import pytest

from src.llm_validation import clamp_float, extract_json_list, extract_json_object, require_field


# ---------------------------------------------------------------------------
# extract_json_object
# ---------------------------------------------------------------------------


def test_extract_json_object_direct() -> None:
    result = extract_json_object('{"key": "value"}')
    assert result == {"key": "value"}


def test_extract_json_object_with_code_fence() -> None:
    text = '```json\n{"rating": "high"}\n```'
    result = extract_json_object(text)
    assert result == {"rating": "high"}


def test_extract_json_object_embedded_in_prose() -> None:
    text = 'Here is the result: {"score": 0.9} as requested.'
    result = extract_json_object(text)
    assert result == {"score": 0.9}


def test_extract_json_object_no_json_returns_none() -> None:
    result = extract_json_object("No JSON here at all.")
    assert result is None


def test_extract_json_object_malformed_returns_none() -> None:
    result = extract_json_object("{malformed: json}")
    assert result is None


def test_extract_json_object_nested() -> None:
    result = extract_json_object('{"a": {"b": [1, 2, 3]}}')
    assert result == {"a": {"b": [1, 2, 3]}}


# ---------------------------------------------------------------------------
# extract_json_list
# ---------------------------------------------------------------------------


def test_extract_json_list_direct() -> None:
    result = extract_json_list('[1, 2, 3]')
    assert result == [1, 2, 3]


def test_extract_json_list_with_code_fence() -> None:
    text = "```json\n[\"a\", \"b\"]\n```"
    result = extract_json_list(text)
    assert result == ["a", "b"]


def test_extract_json_list_embedded() -> None:
    text = 'Results: ["claim1", "claim2"] end.'
    result = extract_json_list(text)
    assert result == ["claim1", "claim2"]


def test_extract_json_list_no_list_returns_none() -> None:
    result = extract_json_list("just text")
    assert result is None


# ---------------------------------------------------------------------------
# require_field
# ---------------------------------------------------------------------------


def test_require_field_present_and_correct_type() -> None:
    data = {"rating": "high"}
    value = require_field(data, "rating", str)
    assert value == "high"


def test_require_field_missing_raises_value_error() -> None:
    with pytest.raises(ValueError, match="missing required field"):
        require_field({}, "missing_key", str)


def test_require_field_wrong_type_raises_value_error() -> None:
    data = {"score": "not-a-float"}
    with pytest.raises(ValueError, match="expected float"):
        require_field(data, "score", float)


# ---------------------------------------------------------------------------
# clamp_float
# ---------------------------------------------------------------------------


def test_clamp_float_within_range() -> None:
    assert abs(clamp_float(0.5) - 0.5) < 0.001


def test_clamp_float_above_max() -> None:
    assert abs(clamp_float(1.5) - 1.0) < 0.001


def test_clamp_float_below_min() -> None:
    assert abs(clamp_float(-0.5) - 0.0) < 0.001


def test_clamp_float_string_coercion() -> None:
    assert abs(clamp_float("0.75") - 0.75) < 0.001


def test_clamp_float_invalid_returns_low() -> None:
    assert abs(clamp_float("not-a-number") - 0.0) < 0.001


def test_clamp_float_custom_range() -> None:
    assert abs(clamp_float(5.0, low=0.0, high=3.0) - 3.0) < 0.001
