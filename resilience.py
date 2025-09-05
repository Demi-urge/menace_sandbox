from __future__ import annotations

"""Resilience helpers providing retries, circuit breaking and structured errors."""

from dataclasses import dataclass
import time
import threading
import logging
from typing import Callable, Iterable, Type, TypeVar


class ResilienceError(Exception):
    """Base class for resilience related failures."""


class RetryError(ResilienceError):
    """Raised when all retry attempts fail."""


class CircuitOpenError(ResilienceError):
    """Raised when an operation is attempted while the circuit is open."""


class PublishError(ResilienceError):
    """Raised when an event cannot be published after retries."""


T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    delay: float = 1.0,
    exceptions: Iterable[Type[BaseException]] | Type[BaseException] = Exception,
    logger: logging.Logger | None = None,
    max_delay: float = 60.0,
    delays: Iterable[float] | None = None,
) -> T:
    """Call ``func`` with retries using exponential or explicit backoff.

    When ``delays`` is provided it specifies the exact sleep durations between
    attempts.  Otherwise exponential backoff starting at ``delay`` is used.
    """
    if isinstance(exceptions, type):
        exc_types: tuple[Type[BaseException], ...] = (exceptions,)
    else:
        exc_types = tuple(exceptions)

    if delays is not None:
        schedule = list(delays)
        attempts = len(schedule)
    else:
        schedule = []
        backoff = delay

    for i in range(attempts):
        try:
            return func()
        except exc_types as exc:
            if i == attempts - 1:
                raise RetryError(str(exc)) from exc
            log = logger or logging
            log.warning("retry %s/%s after error: %s", i + 1, attempts, exc)
            if delays is not None:
                time.sleep(schedule[i])
            else:
                time.sleep(backoff)
                backoff = min(backoff * 2, max_delay)
    raise RetryError("exhausted retries without success")


@dataclass
class CircuitBreaker:
    """Simple circuit breaker controlling access to fragile operations."""

    max_failures: int = 5
    reset_timeout: float = 60.0
    _failures: int = 0
    _opened_until: float = 0.0
    _lock: threading.Lock = threading.Lock()

    def call(self, func: Callable[[], T]) -> T:
        """Invoke ``func`` respecting circuit state."""
        with self._lock:
            if time.time() < self._opened_until:
                raise CircuitOpenError("circuit open")
        try:
            result = func()
        except Exception:
            with self._lock:
                self._failures += 1
                if self._failures >= self.max_failures:
                    self._opened_until = time.time() + self.reset_timeout
                    self._failures = 0
            raise
        else:
            with self._lock:
                self._failures = 0
            return result


__all__ = [
    "ResilienceError",
    "RetryError",
    "CircuitOpenError",
    "PublishError",
    "retry_with_backoff",
    "CircuitBreaker",
]
