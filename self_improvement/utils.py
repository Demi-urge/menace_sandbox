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
import threading
from functools import lru_cache
from typing import Any, Callable, Awaitable

from ..metrics_exporter import self_improvement_failure_total
from ..sandbox_settings import SandboxSettings


_diagnostics_lock = threading.Lock()
_diagnostics = {
    "cache_hits": 0,
    "cache_misses": 0,
}


@lru_cache(maxsize=None)
def _import_callable(module: str, attr: str) -> Callable[..., Any]:
    mod = importlib.import_module(module)
    return getattr(mod, attr)


def _load_callable(
    module: str,
    attr: str,
) -> Callable[..., Any]:
    """Dynamically import ``attr`` from ``module`` with logging and caching.

    Successful imports are cached for future lookups. When the dependency is
    missing the function now fails fast with a descriptive ``RuntimeError``
    instead of attempting automatic installation or returning a stub.
    """

    logger = logging.getLogger(__name__)

    try:
        func = _import_callable(module, attr)
        with _diagnostics_lock:
            info = _import_callable.cache_info()
            _diagnostics["cache_hits"] = info.hits
            _diagnostics["cache_misses"] = info.misses
        _load_callable.diagnostics = _diagnostics
        return func
    except (ImportError, AttributeError) as exc:  # pragma: no cover - best effort logging
        logger.exception("missing dependency %s.%s", module, attr)
        self_improvement_failure_total.labels(reason="missing_dependency").inc()
        guide = f"Install it via `pip install {module.split('.')[0]}` to use {attr}"
        _load_callable.diagnostics = _diagnostics
        raise RuntimeError(
            f"{module} dependency is required for {attr}. {guide}"
        ) from exc


def _call_with_retries(
    func: Callable[..., Any],
    *args: Any,
    retries: int = 3,
    delay: float = 0.1,
    backoff: Callable[[int, float], float] | None = None,
    jitter: float | None = None,
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

    settings = SandboxSettings()
    if backoff is None:
        factor = getattr(settings, "sandbox_retry_backoff_multiplier", 1.0)
        backoff = lambda attempt, base: base * attempt * factor
    if jitter is None:
        jitter = getattr(settings, "sandbox_retry_jitter", 0.0)

    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    async def _async_call() -> Any:
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                result = func(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
                return result
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
                    await asyncio.sleep(sleep_for)
        assert last_exc is not None
        self_improvement_failure_total.labels(reason="call_retry_failure").inc()
        raise last_exc

    if loop_running:
        return _async_call()

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            return result
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
