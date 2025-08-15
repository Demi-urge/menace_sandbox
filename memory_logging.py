from __future__ import annotations

"""Helper functions for consistent GPT memory logging."""

from typing import Any, Sequence

from gpt_memory import STANDARD_TAGS


def _normalise_tags(tags: Sequence[str] | None) -> list[str]:
    """Return tags filtered against :data:`STANDARD_TAGS`.

    Tags are lowercased and any values not present in ``STANDARD_TAGS`` are
    discarded so that callers cannot accidentally introduce new labels.
    """

    return [t.lower() for t in (tags or []) if t.lower() in STANDARD_TAGS]


def log_with_tags(
    memory: Any,
    prompt: str,
    response: str,
    tags: Sequence[str] | None = None,
) -> Any:
    """Record a prompt/response pair ensuring tags are standardised.

    The helper transparently supports both ``log_interaction`` and ``store``
    methods.  When neither is available the call is ignored.
    """

    clean = _normalise_tags(tags)
    if hasattr(memory, "log_interaction"):
        return memory.log_interaction(prompt, response, clean)
    if hasattr(memory, "store"):
        return memory.store(prompt, response, clean)
    return None
