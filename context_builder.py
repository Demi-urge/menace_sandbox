"""Shared helpers for consistent ContextBuilder error handling."""

from __future__ import annotations

from typing import NoReturn
import logging


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


__all__ = ["PromptBuildError", "handle_failure"]

