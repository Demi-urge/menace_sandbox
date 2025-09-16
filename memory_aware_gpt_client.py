from __future__ import annotations

"""Compatibility module for legacy memory-aware client helpers.

The :func:`ask_with_memory` helper has been removed.  Callers should build
prompts directly via :meth:`ContextBuilder.build_prompt` and forward the result
to ``client.generate(...)``.
"""

from typing import Any

_REMOVAL_MESSAGE = (
    "ask_with_memory has been removed; use ContextBuilder.build_prompt() "
    "followed by client.generate(...)."
)


def __getattr__(name: str) -> Any:
    """Provide a helpful error when legacy attributes are requested."""

    if name == "ask_with_memory":
        raise AttributeError(_REMOVAL_MESSAGE)
    raise AttributeError(name)
