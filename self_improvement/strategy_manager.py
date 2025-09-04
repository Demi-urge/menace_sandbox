"""Strategy rotation helper for self-improvement tasks.

This lightweight module keeps an ordered list of strategy templates and
provides :func:`next_strategy` to rotate through them after failures.  Certain
keywords found in the failure reason trigger specific strategies while unknown
reasons simply advance to the next entry in the rotation order.
"""

from __future__ import annotations

from typing import Dict, List

# Ordered list of available strategies.  The first entry is considered the
# default starting strategy.
STRATEGIES: List[str] = [
    "strict_fix",
    "delete_rebuild",
    "comment_refactor",
    "unit_test_rewrite",
]

# Mapping of lowercase keywords to strategy names.  When the failure reason
# contains one of these tokens the associated strategy is selected directly.
_KEYWORD_MAP: Dict[str, str] = {
    "test": "unit_test_rewrite",
    "comment": "comment_refactor",
    "refactor": "comment_refactor",
    "delete": "delete_rebuild",
    "rebuild": "delete_rebuild",
}

_index = 0


def next_strategy(failure_reason: str | None = None) -> str:
    """Return the next strategy to attempt.

    Parameters
    ----------
    failure_reason:
        Optional text describing why the previous attempt failed.  When the
        reason contains one of the keywords from ``_KEYWORD_MAP`` the
        associated strategy becomes the next suggestion.  Unrecognised reasons
        simply rotate to the next strategy in :data:`STRATEGIES`.

    Returns
    -------
    str
        The name of the strategy that should be attempted next.
    """

    global _index

    if failure_reason:
        reason = failure_reason.lower()
        for key, strategy in _KEYWORD_MAP.items():
            if key in reason:
                _index = STRATEGIES.index(strategy)
                break
        else:
            _index = (_index + 1) % len(STRATEGIES)
    else:
        _index = (_index + 1) % len(STRATEGIES)
    return STRATEGIES[_index]


__all__ = ["STRATEGIES", "next_strategy"]

