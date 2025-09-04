from __future__ import annotations

"""Utility for rotating prompt strategies after repeated failures."""

from typing import Dict, List

# Ordered list of available prompt templates.
TEMPLATES: List[str] = [
    "strict_fix",
    "delete_rebuild",
    "comment_refactor",
    "unit_test_rewrite",
]

# Track consecutive failures for each strategy.
failure_counts: Dict[str, int] = {name: 0 for name in TEMPLATES}
# Consecutive failures allowed for each strategy before rotating. The default
# is a single attempt but the mapping allows per-strategy customisation.
failure_limits: Dict[str, int] = {name: 1 for name in TEMPLATES}


def next_strategy(current: str, reason: str) -> str:
    """Record a failure for ``current`` and return the next strategy.

    Parameters
    ----------
    current:
        The strategy that has just failed.
    reason:
        Free form text describing the failure; stored for diagnostic
        purposes but otherwise unused.

    Returns
    -------
    str
        The strategy to attempt next in the ordered template list.
    """

    failure_counts[current] = failure_counts.get(current, 0) + 1
    limit = failure_limits.get(current, 1)
    if failure_counts[current] < limit:
        return current
    try:
        idx = TEMPLATES.index(current)
    except ValueError:
        idx = -1
    return TEMPLATES[(idx + 1) % len(TEMPLATES)]


__all__ = ["TEMPLATES", "failure_counts", "next_strategy"]
