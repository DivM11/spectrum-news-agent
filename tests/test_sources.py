from __future__ import annotations

import pytest

from src.sources import SourceInfo, SourceRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config(sources: list[dict] | None = None, biases: list[dict] | None = None) -> dict:
    return {
        "sources": sources or _default_sources(),
        "biases": biases or _default_biases(),
    }


def _default_biases() -> list[dict]:
    return [
        {"id": "left", "label": "Left / Progressive"},
        {"id": "center", "label": "Center / Neutral"},
        {"id": "right", "label": "Right / Conservative"},
    ]


def _default_sources() -> list[dict]:
    return [
        {"name": "Reuters", "domain": "reuters.com", "bias": "center", "factuality": "very_high"},
        {"name": "AP", "domain": "apnews.com", "bias": "center", "factuality": "very_high"},
        {"name": "HuffPost", "domain": "huffpost.com", "bias": "left", "factuality": "mostly_factual"},
        {"name": "Fox News", "domain": "foxnews.com", "bias": "right", "factuality": "mixed"},
    ]


# ---------------------------------------------------------------------------
# SourceRegistry construction
# ---------------------------------------------------------------------------


def test_registry_loads_sources() -> None:
    registry = SourceRegistry(_config())
    assert len(registry.all_sources) == 4


def test_registry_empty_config() -> None:
    registry = SourceRegistry({"sources": [], "biases": []})
    assert registry.all_sources == []
    assert registry.get_all_biases() == []


# ---------------------------------------------------------------------------
# get_sources_by_bias
# ---------------------------------------------------------------------------


def test_get_sources_by_bias_center() -> None:
    registry = SourceRegistry(_config())
    sources = registry.get_sources_by_bias("center")
    assert len(sources) == 2
    names = {s.name for s in sources}
    assert "Reuters" in names
    assert "AP" in names


def test_get_sources_by_bias_unknown_returns_empty() -> None:
    registry = SourceRegistry(_config())
    assert registry.get_sources_by_bias("unknown_bias") == []


# ---------------------------------------------------------------------------
# get_domains_by_bias
# ---------------------------------------------------------------------------


def test_get_domains_by_bias() -> None:
    registry = SourceRegistry(_config())
    domains = registry.get_domains_by_bias("center")
    assert "reuters.com" in domains
    assert "apnews.com" in domains


def test_get_domains_by_bias_empty_for_unknown() -> None:
    registry = SourceRegistry(_config())
    assert registry.get_domains_by_bias("nonexistent") == []


# ---------------------------------------------------------------------------
# get_source_by_domain
# ---------------------------------------------------------------------------


def test_get_source_by_domain_found() -> None:
    registry = SourceRegistry(_config())
    source = registry.get_source_by_domain("reuters.com")
    assert source is not None
    assert source.name == "Reuters"
    assert source.factuality == "very_high"


def test_get_source_by_domain_not_found_returns_none() -> None:
    registry = SourceRegistry(_config())
    assert registry.get_source_by_domain("unknown.com") is None


# ---------------------------------------------------------------------------
# get_all_biases / get_bias_label / is_valid_bias
# ---------------------------------------------------------------------------


def test_get_all_biases_returns_list_of_ids() -> None:
    registry = SourceRegistry(_config())
    biases = registry.get_all_biases()
    assert len(biases) == 3
    assert "left" in biases
    assert "center" in biases
    assert "right" in biases


def test_get_bias_label_known() -> None:
    registry = SourceRegistry(_config())
    assert registry.get_bias_label("center") == "Center / Neutral"


def test_get_bias_label_unknown_returns_id() -> None:
    registry = SourceRegistry(_config())
    assert registry.get_bias_label("fancy_new_bias") == "fancy_new_bias"


def test_is_valid_bias_true() -> None:
    registry = SourceRegistry(_config())
    assert registry.is_valid_bias("left") is True


def test_is_valid_bias_false() -> None:
    registry = SourceRegistry(_config())
    assert registry.is_valid_bias("extremist") is False


# ---------------------------------------------------------------------------
# SourceInfo is frozen
# ---------------------------------------------------------------------------


def test_source_info_is_frozen() -> None:
    info = SourceInfo(name="Test", domain="test.com", bias="center", factuality="high")
    with pytest.raises((AttributeError, TypeError)):
        info.name = "Other"  # type: ignore[misc]
