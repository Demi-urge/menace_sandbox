"""Fallback shim for the optional :mod:`filelock` dependency."""
from filelock_stub.filelock_stub import FileLock, Timeout, open_locked  # noqa: F401
