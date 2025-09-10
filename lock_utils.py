import os
import time
import logging
import errno
from contextlib import suppress
from filelock import FileLock, Timeout

try:
    from .fcntl_compat import flock, LOCK_EX, LOCK_NB
except Exception:  # pragma: no cover - allow running as script
    from fcntl_compat import flock, LOCK_EX, LOCK_NB

try:  # pragma: no cover - platform specific
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover - posix
    msvcrt = None  # type: ignore

LOCK_TIMEOUT = float(os.getenv("LOCK_TIMEOUT", "3600"))
logger = logging.getLogger(__name__)


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    return True


def is_lock_stale(path: str, *, timeout: float | None = None) -> bool:
    if timeout is None:
        timeout = LOCK_TIMEOUT
    try:
        with open(path, "r") as fh:
            data = fh.read().strip().split(",")
        pid = int(data[0])
        ts = float(data[1]) if len(data) > 1 else 0.0
    except Exception:
        return True
    if not _pid_running(pid):
        return True
    if ts and time.time() - ts > timeout:
        return True
    return False


class _ContextFileLock(FileLock):
    """FileLock variant whose ``acquire`` acts as a context manager."""

    class _Guard:
        def __init__(self, lock: FileLock) -> None:
            self.lock = lock

        def __enter__(self) -> FileLock:
            return self.lock

        def __exit__(self, exc_type, exc, tb) -> None:
            if self.lock.is_locked:
                try:
                    self.lock.release()
                except Exception as exc:
                    logger.exception("failed to release file lock: %s", exc)

    def is_lock_stale(self, *, timeout: float | None = None) -> bool:
        """Return ``True`` if the lock file was created by a dead process."""
        lock_path = getattr(self, "lock_file", None)
        if not lock_path or not os.path.exists(lock_path):
            return False
        return is_lock_stale(lock_path, timeout=timeout)

    def acquire(self, timeout: float | None = None, poll_interval: float = 0.05):  # type: ignore[override]
        lock_path = getattr(self, "lock_file", None)
        if lock_path and os.path.exists(lock_path) and self.is_lock_stale(timeout=timeout or LOCK_TIMEOUT):
            with suppress(Exception):
                os.remove(lock_path)

        if not hasattr(self, "_context"):
            super().acquire(timeout=timeout)
            if lock_path:
                with suppress(Exception):
                    tmp = lock_path + ".tmp"
                    with open(tmp, "w") as fh:
                        fh.write(f"{os.getpid()},{time.time()}")
                    os.replace(tmp, lock_path)
            return self._Guard(self)

        if msvcrt or os.name != "nt":
            if timeout is None:
                timeout = self._context.timeout
            start = time.perf_counter()
            while True:
                fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, getattr(self._context, "mode", 0o644))
                with suppress(PermissionError):
                    if hasattr(os, "fchmod"):
                        os.fchmod(fd, getattr(self._context, "mode", 0o644))
                try:
                    if os.name == "nt":
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    else:
                        flock(fd, LOCK_EX | LOCK_NB)
                    self._context.lock_file_fd = fd
                    break
                except OSError as exc:  # pragma: no cover - race conditions
                    os.close(fd)
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                        raise
                    if timeout >= 0 and time.perf_counter() - start >= timeout:
                        raise Timeout(lock_path)
                    time.sleep(poll_interval)
            if lock_path:
                with suppress(Exception):
                    tmp = lock_path + ".tmp"
                    with open(tmp, "w") as fh:
                        fh.write(f"{os.getpid()},{time.time()}")
                    os.replace(tmp, lock_path)
            return self._Guard(self)
        else:  # pragma: no cover - fallback
            result = super().acquire(timeout=timeout, poll_interval=poll_interval)
            if lock_path:
                with suppress(Exception):
                    tmp = lock_path + ".tmp"
                    with open(tmp, "w") as fh:
                        fh.write(f"{os.getpid()},{time.time()}")
                    os.replace(tmp, lock_path)
            return result

    def release(self, *args, **kwargs) -> None:  # type: ignore[override]
        try:
            super().release(*args, **kwargs)
        finally:
            try:
                os.remove(self.lock_file)
            except FileNotFoundError:
                logger.warning("lock file already removed: %s", self.lock_file)
            except Exception as exc:
                logger.exception("failed to remove lock file: %s", exc)


class SandboxLock(_ContextFileLock):
    """File lock used throughout the sandbox with cross-platform support.

    This class behaves as a regular context manager (``with lock:``) while also
    supporting the explicit ``with lock.acquire():`` style used by some parts of
    the codebase.  It delegates all heavy lifting to :class:`_ContextFileLock`,
    which performs platform specific locking using ``fcntl`` on POSIX systems
    and ``msvcrt`` on Windows.
    """

    def __enter__(self) -> "SandboxLock":
        # ``_ContextFileLock.acquire`` returns a guard object that releases the
        # lock on exit.  Store it so ``__exit__`` can delegate correctly.
        self._guard = super().acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        guard = getattr(self, "_guard", None)
        if guard is not None:
            guard.__exit__(exc_type, exc, tb)
            self._guard = None
        else:  # pragma: no cover - defensive fallback
            super().release()


__all__ = [
    "SandboxLock",
    "_ContextFileLock",
    "is_lock_stale",
    "LOCK_TIMEOUT",
    "Timeout",
]
