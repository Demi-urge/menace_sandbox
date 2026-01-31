"""Convenience wrappers for structured logging."""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from logging_wrapper import wrap_with_logging as _wrap_with_logging

P = ParamSpec("P")
R = TypeVar("R")


def wrap_with_logging(
    callable_obj: Callable[P, R],
    config: dict | None = None,
) -> Callable[P, R]:
    """Wrap a callable with deterministic structured logging.

    The wrapper delegates to :func:`logging_wrapper.wrap_with_logging` to apply
    logging without mutating inputs, retrying calls, or introducing
    nondeterministic side effects beyond emitting log records.

    Args:
        callable_obj: The callable to wrap.
        config: Optional logging configuration overrides.

    Returns:
        A callable that preserves the original signature and behavior.
    """
    return _wrap_with_logging(callable_obj, config=config)


__all__ = ["wrap_with_logging"]
