"""Lightweight stub of the :mod:`filelock` package for test environments."""
from __future__ import annotations

import contextlib
import threading


class Timeout(Exception):
    """Fallback timeout exception matching :mod:`filelock` interface."""


class FileLock:
    """Minimal in-memory lock compatible with :class:`filelock.FileLock`."""

    def __init__(self, lock_file: str | None = None, timeout: float | None = None, *_: object, **__: object) -> None:
        self._lock = threading.Lock()
        self.locked = False
        self.lock_file = lock_file
        self.timeout = -1 if timeout is None else timeout

    @property
    def is_locked(self) -> bool:
        return self.locked

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
