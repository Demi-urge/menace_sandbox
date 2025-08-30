"""Utility helpers for self-improvement package.

This module provides internal helpers used across the self-improvement
submodules to dynamically load optional dependencies and execute calls with
retry semantics.
"""
from __future__ import annotations

import importlib
import logging
import time
import asyncio
import inspect
import random
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from ..metrics_exporter import self_improvement_failure_total
from ..sandbox_settings import SandboxSettings


def _load_callable(module: str, attr: str) -> Callable[..., Any]:
    """Dynamically import ``attr`` from ``module`` with logging.

    When the dependency is missing a stub is returned. The stub carries a
    structured error object and, depending on :class:`SandboxSettings`, may
    attempt to lazily retry the import on first use.
    """

    try:
        mod = importlib.import_module(module)
        return getattr(mod, attr)
    except (ImportError, AttributeError) as exc:  # pragma: no cover - best effort logging
        logger = logging.getLogger(__name__)
        logger.exception("missing dependency %s.%s", module, attr)
        self_improvement_failure_total.labels(reason="missing_dependency").inc()

        @dataclass
        class MissingDependencyError:
            module: str
            attr: str
            exc: Exception

        error = MissingDependencyError(module, attr, exc)
        settings = SandboxSettings()

        if not getattr(settings, "retry_optional_dependencies", False):
            def _stub(*_args: Any, **_kwargs: Any) -> Any:
                raise RuntimeError(f"{module} dependency is required for {attr}")

            _stub.error = error
            return _stub

        def _retry_stub(*args: Any, **kwargs: Any) -> Any:
            def _load_and_call() -> Any:
                mod_inner = importlib.import_module(module)
                func = getattr(mod_inner, attr)
                return func(*args, **kwargs)

            return _call_with_retries(
                _load_and_call,
                retries=getattr(settings, "sandbox_max_retries", 3) or 3,
                delay=getattr(settings, "sandbox_retry_delay", 0.1),
            )

        _retry_stub.error = error
        return _retry_stub


def _call_with_retries(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    delay: float = 0.1,
    backoff: Callable[[int, float], float] | None = None,
    jitter: float = 0.0,
    max_delay: float | None = None,
    logger: logging.Logger | None = None,
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Execute ``func`` with retry semantics.

    ``func`` may be synchronous or return an awaitable. Delays between retries
    can be customised via ``backoff`` (which receives the attempt number and
    base ``delay``), bounded by ``max_delay`` and extended with a random
    ``jitter``. Logging supports an optional ``context`` dictionary which is
    attached to each log record via :class:`logging.LoggerAdapter`.
    """

    logger = logger or logging.getLogger(__name__)
    if context:
        logger = logging.LoggerAdapter(logger, context)
    backoff = backoff or (lambda attempt, base: base * attempt)

    def _run(result: Awaitable[Any] | Any) -> Any:
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return _run(func(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - exercised in tests
            last_exc = exc
            logger.warning(
                "call to %s failed on attempt %s/%s",
                getattr(func, "__name__", repr(func)),
                attempt,
                retries,
                exc_info=True,
            )
            if attempt < retries:
                sleep_for = backoff(attempt, delay)
                if jitter:
                    sleep_for += random.uniform(0, jitter)
                if max_delay is not None:
                    sleep_for = min(sleep_for, max_delay)
                time.sleep(sleep_for)
    assert last_exc is not None
    self_improvement_failure_total.labels(reason="call_retry_failure").inc()
    raise last_exc
