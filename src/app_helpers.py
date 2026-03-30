"""Pure helper functions for app.py that are testable without a Streamlit context."""

from __future__ import annotations


def parse_topics(raw: str) -> list[str]:
    """Parse the raw text-area string into a capped, stripped list of topic strings.

    Rules:
    - Each non-blank line becomes one topic.
    - Leading/trailing whitespace is stripped from every line.
    - Blank (or whitespace-only) lines are ignored.
    - At most 5 topics are returned (extra lines are silently truncated).
    """
    return [t.strip() for t in raw.strip().splitlines() if t.strip()][:5]
