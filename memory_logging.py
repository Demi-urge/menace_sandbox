from __future__ import annotations

"""Helper functions for consistent GPT memory logging."""

from typing import Any, Sequence
import logging

from gpt_memory import STANDARD_TAGS


logger = logging.getLogger(__name__)


def _normalise_tags(tags: Sequence[str] | None) -> list[str]:
    """Return tags filtered against :data:`STANDARD_TAGS`.

    Tags are lowercased and validated against ``STANDARD_TAGS``. Unknown tags
    are discarded and reported while duplicates are removed. This prevents
    callers from accidentally introducing new labels.
    """

    clean: list[str] = []
    invalid: list[str] = []
    for tag in tags or []:
        normalised = tag.lower()
        if normalised in STANDARD_TAGS:
            if normalised not in clean:
                clean.append(normalised)
        else:
            invalid.append(tag)
    if invalid:
        logger.warning("discarding unknown tags: %s", invalid)
    return clean


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
