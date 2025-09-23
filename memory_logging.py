from __future__ import annotations

"""Helper functions for consistent GPT memory logging."""

from typing import Any, Sequence
import logging
import os
import re
import threading
import time

from menace_sandbox.gpt_memory import STANDARD_TAGS


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- refresh opts
_AUTO_REFRESH = os.getenv("GPT_AUTO_REFRESH_INSIGHTS", "1").lower() not in {
    "0",
    "false",
    "no",
}
_REFRESH_INTERVAL = 5.0  # seconds
_refresh_lock = threading.Lock()
_last_refresh = 0.0


def _schedule_refresh(knowledge: Any) -> None:
    """Run ``knowledge.update_insights`` in the background."""

    def _run() -> None:
        try:
            knowledge.update_insights()
        except Exception:  # pragma: no cover - defensive
            logger.exception("failed to refresh insights")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def _maybe_refresh(memory: Any) -> None:
    """Refresh insights if the environment and cooldown permit."""

    if not _AUTO_REFRESH:
        return
    knowledge = getattr(memory, "knowledge", None)
    update = getattr(knowledge, "update_insights", None)
    if not callable(update):
        return
    global _last_refresh
    now = time.monotonic()
    with _refresh_lock:
        if now - _last_refresh < _REFRESH_INTERVAL:
            return
        _last_refresh = now
    _schedule_refresh(knowledge)

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
        # Preserve key-style tags such as ``module:name`` or ``module.action``
        # verbatim to allow free-form categorisation.
        if ":" in tag or "." in tag:
            if tag not in clean:
                clean.append(tag)
            continue
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


def ensure_tags(key: str, tags: Sequence[str] | None = None) -> list[str]:
    """Return ``tags`` augmented with ``module`` and ``action`` markers.

    ``key`` is expected to follow ``"module.action"`` format. The returned list
    includes ``module:{module}`` and ``action:{action}`` entries. When the key is
    malformed a warning is emitted and the original ``tags`` are simply
    normalised.
    """

    clean = _normalise_tags(tags)
    if "." not in key:
        logger.warning("key '%s' missing module.action format", key)
        return clean
    module, action = key.split(".", 1)
    mod_tag = f"module:{module}"
    act_tag = f"action:{action}"
    if mod_tag not in clean:
        clean.append(mod_tag)
    if act_tag not in clean:
        clean.append(act_tag)
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
    if not any(":" in t or "." in t for t in clean):
        logger.warning("missing module/action tags: %s", tags)
    result: Any = None
    if hasattr(memory, "log_interaction"):
        result = memory.log_interaction(prompt, response, clean)
    elif hasattr(memory, "store"):
        result = memory.store(prompt, response, clean)
    else:
        return None
    _maybe_refresh(memory)
    return result
