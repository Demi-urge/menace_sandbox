from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

import logging
import asyncio

try:  # pragma: no cover - optional dependency not critical for tests
    from .. import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - fallback when running as script
    import metrics_exporter as _me  # type: ignore

F = TypeVar("F", bound=Callable[..., Any])

# Gauges for tracking basic metrics per function
_CALL_COUNT = _me.Gauge(
    "vector_service_calls_total", "Number of times a function is invoked", ["function"],
)
_LATENCY_GAUGE = _me.Gauge(
    "vector_service_latency_seconds", "Execution time of functions", ["function"],
)
_RESULT_SIZE_GAUGE = _me.Gauge(
    "vector_service_result_size", "Size of results returned by functions", ["function"],
)


def _result_size(result: Any) -> int:
    if hasattr(result, "__len__"):
        try:
            return len(result)  # type: ignore[arg-type]
        except Exception:
            return 0
    return 0


def log_and_measure(func: F) -> F:
    """Log start/end timestamps and emit metrics for ``func``."""

    logger = logging.getLogger(func.__module__)
    name = func.__qualname__

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            session_id = kwargs.get("session_id", "")
            start = time.time()
            logger.info("%s start", name, extra={"session_id": session_id, "timestamp": start})
            try:
                result = await func(*args, **kwargs)
            except Exception:
                end = time.time()
                latency = end - start
                _CALL_COUNT.labels(name).inc()
                _LATENCY_GAUGE.labels(name).set(latency)
                logger.exception(
                    "%s error", name,
                    extra={"session_id": session_id, "timestamp": end, "latency": latency, "result_size": 0},
                )
                raise

            end = time.time()
            latency = end - start
            size = _result_size(result)
            _CALL_COUNT.labels(name).inc()
            _LATENCY_GAUGE.labels(name).set(latency)
            _RESULT_SIZE_GAUGE.labels(name).set(size)
            logger.info(
                "%s end", name,
                extra={"session_id": session_id, "timestamp": end, "latency": latency, "result_size": size},
            )
            return result

        return awrapper  # type: ignore[return-value]

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        session_id = kwargs.get("session_id", "")
        start = time.time()
        logger.info("%s start", name, extra={"session_id": session_id, "timestamp": start})
        try:
            result = func(*args, **kwargs)
        except Exception:
            end = time.time()
            latency = end - start
            _CALL_COUNT.labels(name).inc()
            _LATENCY_GAUGE.labels(name).set(latency)
            logger.exception(
                "%s error", name,
                extra={"session_id": session_id, "timestamp": end, "latency": latency, "result_size": 0},
            )
            raise

        end = time.time()
        latency = end - start
        size = _result_size(result)
        _CALL_COUNT.labels(name).inc()
        _LATENCY_GAUGE.labels(name).set(latency)
        _RESULT_SIZE_GAUGE.labels(name).set(size)
        logger.info(
            "%s end", name,
            extra={"session_id": session_id, "timestamp": end, "latency": latency, "result_size": size},
        )
        return result

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Backwards compatibility helpers

def log_and_time(func: F) -> F:  # pragma: no cover - compatibility shim
    """Alias for :func:`log_and_measure` for older call sites."""

    return log_and_measure(func)


def track_metrics(func: F) -> F:  # pragma: no cover - compatibility shim
    """No-op wrapper retained for backwards compatibility."""

    return func


__all__ = ["log_and_measure", "log_and_time", "track_metrics"]
