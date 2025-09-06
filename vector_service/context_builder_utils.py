"""Utilities for constructing ContextBuilder instances with default databases."""

from __future__ import annotations

from typing import Any

from .context_builder import ContextBuilder


def get_default_context_builder(**kwargs: Any) -> ContextBuilder:
    """Return a :class:`ContextBuilder` preconfigured with default databases."""
    return ContextBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
        **kwargs,
    )
