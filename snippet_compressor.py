from __future__ import annotations

"""Helpers to condense diff, code and log snippets for prompt generation.

This module relies on lightweight micro-models to summarise snippets and the
:mod:`redaction_utils` helpers to strip potentially sensitive tokens.  Each
returned field is capped to ``max_length`` characters so prompts remain
compact.
"""

from typing import Any, Dict

from .redaction_utils import redact_text


def _summarize_diff(before: str, after: str) -> str:
    """Best-effort wrapper around :mod:`micro_models.diff_summarizer`."""

    try:  # pragma: no cover - optional dependency
        from micro_models.diff_summarizer import summarize_diff as _summ
    except Exception:  # pragma: no cover - summariser may be missing
        return after or before
    try:  # pragma: no cover - defensive
        return _summ(before, after)
    except Exception:
        return after or before


def _inject(prompt: str, prefix: str) -> str:
    """Return ``prompt`` prefixed with ``prefix`` when available."""

    try:  # pragma: no cover - optional dependency
        from micro_models.prefix_injector import inject_prefix as _inject_prefix
    except Exception:  # pragma: no cover - fall back to simple concatenation
        if prefix:
            return prefix + "\n\n" + prompt
        return prompt
    try:  # pragma: no cover - defensive
        return _inject_prefix(prompt, prefix, 1.0)
    except Exception:
        if prefix:
            return prefix + "\n\n" + prompt
        return prompt


_DEF_MAX_LENGTH = 200


def _truncate(text: str, limit: int) -> str:
    """Return ``text`` truncated to ``limit`` characters."""

    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def compress_snippets(meta: Dict[str, Any], *, max_length: int = _DEF_MAX_LENGTH) -> Dict[str, str]:
    """Return redacted and summarised ``diff``, ``snippet`` and ``test_log``.

    Parameters
    ----------
    meta:
        Metadata dictionary potentially containing ``before``, ``after``,
        ``diff``, ``snippet``/``code`` and ``test_log`` fields.
    max_length:
        Maximum length for each returned field.
    """

    out: Dict[str, str] = {}

    before = meta.get("before") or ""
    after = meta.get("after") or ""
    diff = meta.get("diff") or ""
    if before or after or diff:
        summary = _summarize_diff(before, after)
        if not summary:
            summary = diff
        out["diff"] = _truncate(redact_text(summary), max_length)

    code = meta.get("snippet") or meta.get("code")
    if isinstance(code, str) and code.strip():
        prefix = _summarize_diff("", code)
        code = _inject(code, prefix)
        out["snippet"] = _truncate(redact_text(code), max_length)

    log = meta.get("test_log")
    if isinstance(log, str) and log.strip():
        prefix = _summarize_diff("", log)
        log = _inject(log, prefix)
        out["test_log"] = _truncate(redact_text(log), max_length)

    return out


__all__ = ["compress_snippets"]
