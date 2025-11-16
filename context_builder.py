"""Shared helpers for consistent ContextBuilder error handling."""

from __future__ import annotations

from typing import Any, NoReturn
import logging

from .context_builder_util import create_context_builder as _create_context_builder


class PromptBuildError(RuntimeError):
    """Raised when prompt construction via ``ContextBuilder`` fails."""


def handle_failure(
    message: str, exc: BaseException, *, logger: logging.Logger | None = None
) -> NoReturn:
    """Log ``message`` and raise :class:`PromptBuildError` chained from ``exc``.

    Parameters
    ----------
    message:
        Human readable description of the failure.
    exc:
        The exception raised while attempting to build a prompt.
    logger:
        Optional logger used for the error message.  When omitted a module level
        logger tied to ``context_builder`` is used so that callers without a
        bespoke logger still emit diagnostics.
    """

    log = logger or logging.getLogger(__name__)
    log.exception(message)
    raise PromptBuildError(message) from exc


def create_context_builder(*args: Any, **kwargs: Any):
    """Proxy to :func:`context_builder_util.create_context_builder`.

    Historically :mod:`menace_sandbox.context_builder` exposed
    ``create_context_builder`` directly.  The implementation now lives in
    :mod:`menace_sandbox.context_builder_util`, so this thin wrapper preserves
    the public API for callers that still import from the original module.
    """

    return _create_context_builder(*args, **kwargs)


__all__ = ["PromptBuildError", "handle_failure", "create_context_builder"]

