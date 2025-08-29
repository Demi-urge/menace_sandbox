"""Utility helpers for self-improvement package.

This module provides internal helpers used across the self-improvement
submodules to dynamically load optional dependencies and execute calls with
retry semantics.
"""
from __future__ import annotations

import importlib
import logging
import time
from typing import Any, Callable

from ..metrics_exporter import self_improvement_failure_total


def _load_callable(module: str, attr: str) -> Callable[..., Any]:
    """Dynamically import ``attr`` from ``module`` with logging."""
    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr)
    except (ImportError, AttributeError) as exc:  # pragma: no cover - best effort logging
        logger = logging.getLogger(__name__)
        logger.exception("missing dependency %s.%s", module, attr)
        self_improvement_failure_total.labels(reason="missing_dependency").inc()
        raise RuntimeError(f"{module} dependency is required for {attr}") from exc


def _call_with_retries(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    delay: float = 0.1,
    **kwargs: Any,
) -> Any:
    """Execute ``func`` with retry semantics."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - exercised in tests
            last_exc = exc
            logging.getLogger(__name__).warning(
                "call to %s failed on attempt %s/%s",
                getattr(func, "__name__", repr(func)),
                attempt,
                retries,
                exc_info=True,
            )
            if attempt < retries:
                time.sleep(delay * attempt)
    assert last_exc is not None
    self_improvement_failure_total.labels(reason="call_retry_failure").inc()
    raise last_exc
