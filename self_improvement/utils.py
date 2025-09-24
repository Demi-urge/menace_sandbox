"""Utility helpers for self-improvement package.

This module provides internal helpers used across the self-improvement
submodules to dynamically load optional dependencies and execute calls with
retry semantics.

Imports performed via :func:`_load_callable` are memoised using an
``functools.lru_cache`` limited to 128 entries to avoid unbounded memory
growth. The cache state can be reset manually with
``clear_import_cache()`` when modules change at runtime or during tests.

The retry helper implements exponential backoff with optional jitter and a
maximum delay. Delays start at ``base`` and grow as ``base * 2 **
(``attempt`` - 1)`` for successive attempts. Each delay is randomly jittered
by up to ``jitter`` percent (for example ``jitter=0.1`` applies ±10% noise)
before being capped by ``max_delay`` when provided.
"""
from __future__ import annotations

import importlib
import logging
import time
import asyncio
import inspect
import random
import threading
import shutil
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable

try:  # pragma: no cover - prefer package-relative import when available
    from ..dynamic_path_router import resolve_dir, get_project_root
except ImportError:  # pragma: no cover - support flat execution layout
    from dynamic_path_router import resolve_dir, get_project_root  # type: ignore

try:  # pragma: no cover - prefer package-relative import when available
    from ..metrics_exporter import self_improvement_failure_total
except ImportError:  # pragma: no cover - support flat execution layout
    from metrics_exporter import self_improvement_failure_total  # type: ignore

try:  # pragma: no cover - prefer package-relative import when available
    from ..sandbox_settings import SandboxSettings
except ImportError:  # pragma: no cover - support flat execution layout
    from sandbox_settings import SandboxSettings  # type: ignore


_diagnostics_lock = threading.Lock()
_diagnostics = {
    "cache_hits": 0,
    "cache_misses": 0,
}


@lru_cache(maxsize=128)
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


def clear_import_cache() -> None:
    """Clear cached imports and reset diagnostic counters.

    ``_import_callable`` uses an :func:`functools.lru_cache` to memoise
    import lookups. This helper exposes a manual way to wipe that cache and
    update the exported diagnostics, which is useful for tests or when
    dependencies change at runtime.
    """

    _import_callable.cache_clear()
    with _diagnostics_lock:
        _diagnostics["cache_hits"] = 0
        _diagnostics["cache_misses"] = 0
    _load_callable.diagnostics = _diagnostics


def remove_import_cache_files(base: str | Path | None = None) -> None:
    """Delete ``__pycache__`` directories to free disk space."""

    root = resolve_dir(base) if base else get_project_root()
    for pycache in root.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache)
        except FileNotFoundError:
            continue
        except Exception:  # pragma: no cover - best effort
            logging.getLogger(__name__).debug(
                "failed to remove cache directory %s", pycache, exc_info=True
            )


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
    ``jitter`` percentage. Logging supports an optional ``context`` dictionary
    which is attached to each log record via :class:`logging.LoggerAdapter`.
    """

    logger = logger or logging.getLogger(__name__)
    if context:
        logger = logging.LoggerAdapter(logger, context)

    settings = SandboxSettings()
    if backoff is None:
        factor = getattr(settings, "sandbox_retry_backoff_multiplier", 1.0)

        def default_backoff(attempt: int, base: float, *, _factor=factor) -> float:
            return base * (2 ** (attempt - 1)) * _factor

        backoff = default_backoff
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
                        sleep_for *= random.uniform(1 - jitter, 1 + jitter)
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
                    sleep_for *= random.uniform(1 - jitter, 1 + jitter)
                if max_delay is not None:
                    sleep_for = min(sleep_for, max_delay)
                time.sleep(sleep_for)
    assert last_exc is not None
    self_improvement_failure_total.labels(reason="call_retry_failure").inc()
    raise last_exc

