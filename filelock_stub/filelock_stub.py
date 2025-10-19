"""Lightweight stub of the :mod:`filelock` package for test environments."""
from __future__ import annotations

import contextlib
import threading


class Timeout(Exception):
    """Fallback timeout exception matching :mod:`filelock` interface."""


class FileLock:
    """Minimal in-memory lock compatible with :class:`filelock.FileLock`."""

    def __init__(self, *_: object, **__: object) -> None:
        self._lock = threading.Lock()
        self.locked = False

    def acquire(self, timeout: float | None = None) -> bool:
        locked = self._lock.acquire(timeout=timeout) if timeout else self._lock.acquire()
        self.locked = locked
        return locked

    def release(self) -> None:
        if self.locked:
            self.locked = False
            self._lock.release()

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@contextlib.contextmanager
def open_locked(*_args: object, **_kwargs: object):
    lock = FileLock()
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()
