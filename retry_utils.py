from __future__ import annotations

"""Retry helpers for network and IO operations."""

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Type


def retry(exc: Type[BaseException], attempts: int = 3, delay: float = 1.0) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Simple retry decorator with exponential backoff."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            backoff = delay
            for i in range(attempts):
                try:
                    return func(*args, **kwargs)
                except exc as e:
                    if i == attempts - 1:
                        raise
                    logging.warning(
                        "retry %s/%s after error: %s", i + 1, attempts, e
                    )
                    time.sleep(backoff)
                    backoff *= 2

        return wrapper

    return decorator


def publish_with_retry(bus: Any, topic: str, event: object, *, attempts: int = 3, delay: float = 1.0) -> bool:
    """Publish ``event`` with retries on failure."""
    backoff = delay
    for i in range(attempts):
        try:
            bus.publish(topic, event)
            return True
        except Exception as exc:  # pragma: no cover - best effort
            logging.warning("publish attempt %s/%s failed: %s", i + 1, attempts, exc)
            if i < attempts - 1:
                time.sleep(backoff)
                backoff *= 2
    return False


from typing import Iterable, TypeVar

T = TypeVar("T")


def with_retry(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    delay: float = 1.0,
    exc: Iterable[Type[BaseException]] | Type[BaseException] = Exception,
    logger: logging.Logger | None = None,
    jitter: float = 0.0,
) -> T:
    """Call ``func`` with retry and exponential backoff."""
    if isinstance(exc, type):
        exceptions: tuple[Type[BaseException], ...] = (exc,)
    else:
        exceptions = tuple(exc)
    backoff = delay
    for i in range(attempts):
        try:
            return func()
        except exceptions as e:
            if i == attempts - 1:
                raise
            log = logger or logging
            log.warning("retry %s/%s after error: %s", i + 1, attempts, e)
            sleep = backoff
            if jitter:
                sleep += random.uniform(0, backoff * jitter)
            time.sleep(sleep)
            backoff *= 2
    # should not reach here
    raise RuntimeError("with_retry exhausted without returning")


__all__ = ["retry", "publish_with_retry", "with_retry"]
