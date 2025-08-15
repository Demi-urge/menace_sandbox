from __future__ import annotations

"""Helper functions for consistent GPT memory logging."""

from typing import Any, Sequence
import logging
import re

from gpt_memory import STANDARD_TAGS


logger = logging.getLogger(__name__)

# Simple regex based heuristics for inferring canonical tags from responses.
_TAG_PATTERNS = {
    "feedback": re.compile(r"\b(feedback|success|failure)\b", re.I),
    # match "patch" even when followed by an underscore (e.g. "patch_id")
    "improvement_path": re.compile(r"\b(improvement|improve|patch)\b|patch_id", re.I),
    "error_fix": re.compile(r"\b(error|fix|bug)\b", re.I),
    "insight": re.compile(r"\b(insight|idea|observation)\b", re.I),
}


def _infer_tags(text: str) -> list[str]:
    """Return inferred standard tags based on ``text`` heuristics.

    Only a single tag is returned.  When multiple patterns match the text the
    result is considered ambiguous and an empty list is returned.
    """

    matches = [tag for tag, pat in _TAG_PATTERNS.items() if pat.search(text)]
    return matches if len(matches) == 1 else []


def _normalise_tags(tags: Sequence[str] | None, response: str | None = None) -> list[str]:
    """Return tags filtered against :data:`STANDARD_TAGS`.

    Tags are lowercased and validated against ``STANDARD_TAGS``. Unknown tags
    are discarded and reported while duplicates are removed. This prevents
    callers from accidentally introducing new labels.  When no valid tags are
    supplied an attempt is made to infer a single standard tag from the
    ``response`` text.
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
    if not clean and response:
        inferred = _infer_tags(response.lower())
        if inferred:
            clean.extend(inferred)
    return clean


def log_with_tags(
    memory: Any,
    prompt: str,
    response: str,
    tags: Sequence[str] | None = None,
) -> Any:
    """Record a prompt/response pair ensuring tags are standardised.

    Callers may omit ``tags`` entirely, in which case a best-effort tag is
    inferred from the ``response`` text using lightweight regex heuristics.  The
    helper transparently supports both ``log_interaction`` and ``store``
    methods.  When neither is available the call is ignored.
    """

    clean = _normalise_tags(tags, response)
    if hasattr(memory, "log_interaction"):
        return memory.log_interaction(prompt, response, clean)
    if hasattr(memory, "store"):
        return memory.store(prompt, response, clean)
    return None
