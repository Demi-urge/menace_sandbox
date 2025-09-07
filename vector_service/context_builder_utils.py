"""Utilities for constructing :class:`~vector_service.context_builder.ContextBuilder`.

The real :class:`ContextBuilder` already provides sensible defaults for all of
its parameters.  This module simply exposes a tiny convenience wrapper that
forwards any keyword arguments to the constructor.  Callers can therefore rely
on the defaults defined by :class:`ContextBuilder` and
``ContextBuilderConfig`` or override them by passing ``**kwargs``.
"""

from __future__ import annotations

from typing import Any

from .context_builder import ContextBuilder


def get_default_context_builder(**kwargs: Any) -> ContextBuilder:
    """Return a ready-to-use :class:`ContextBuilder`.

    Parameters
    ----------
    **kwargs:
        Additional keyword arguments forwarded directly to
        :class:`ContextBuilder`.  Common options include ``retriever``,
        ``max_tokens`` and other tuning parameters.  Unspecified arguments fall
        back to the defaults provided by :class:`ContextBuilder` and
        ``ContextBuilderConfig``.

    Returns
    -------
    ContextBuilder
        An instance configured with the standard defaults.
    """

    return ContextBuilder(**kwargs)
