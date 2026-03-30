from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceInfo:
    name: str
    domain: str
    bias: str
    factuality: str


class SourceRegistry:
    """
    Loads sources and bias definitions from config. Purely data-driven:
    adding a new source or bias requires only a config.yml change.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._sources: list[SourceInfo] = [
            SourceInfo(
                name=s["name"],
                domain=s["domain"],
                bias=s["bias"],
                factuality=s["factuality"],
            )
            for s in config.get("sources", [])
        ]
        self._biases: list[dict[str, str]] = [
            {"id": b["id"], "label": b["label"]}
            for b in config.get("biases", [])
        ]
        self._by_bias: dict[str, list[SourceInfo]] = {}
        for source in self._sources:
            self._by_bias.setdefault(source.bias, []).append(source)
        self._by_domain: dict[str, SourceInfo] = {s.domain: s for s in self._sources}

    def get_sources_by_bias(self, bias_id: str) -> list[SourceInfo]:
        return self._by_bias.get(bias_id, [])

    def get_domains_by_bias(self, bias_id: str) -> list[str]:
        return [s.domain for s in self.get_sources_by_bias(bias_id)]

    def get_source_by_domain(self, domain: str) -> SourceInfo | None:
        return self._by_domain.get(domain)

    def get_all_biases(self) -> list[str]:
        return [b["id"] for b in self._biases]

    def get_all_bias_configs(self) -> list[dict[str, str]]:
        return list(self._biases)

    def get_bias_label(self, bias_id: str) -> str:
        for b in self._biases:
            if b["id"] == bias_id:
                return b["label"]
        return bias_id

    def is_valid_bias(self, bias_id: str) -> bool:
        return any(b["id"] == bias_id for b in self._biases)

    @property
    def all_sources(self) -> list[SourceInfo]:
        return list(self._sources)

    def get_factuality_map(self) -> dict[str, str]:
        """Return a domain → MBFC factuality label mapping for all sources."""
        return {s.domain: s.factuality for s in self._sources}
