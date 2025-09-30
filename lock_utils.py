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

LOCK_TIMEOUT = float(os.getenv("LOCK_TIMEOUT", "60"))
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
        resolved_timeout: float
        if timeout is None or timeout < 0:
            context_timeout = None
            if hasattr(self, "_context"):
                context_timeout = getattr(self._context, "timeout", None)
            if context_timeout is None or context_timeout < 0:
                context_timeout = getattr(self, "timeout", None)
            if context_timeout is None or context_timeout < 0:
                resolved_timeout = LOCK_TIMEOUT
            else:
                resolved_timeout = context_timeout
        else:
            resolved_timeout = timeout

        lock_path = getattr(self, "lock_file", None)
        if lock_path and os.path.exists(lock_path) and self.is_lock_stale(timeout=resolved_timeout):
            with suppress(Exception):
                os.remove(lock_path)

        context = getattr(self, "_context", None)

        if context is None:
            super().acquire(timeout=resolved_timeout)
            if lock_path:
                with suppress(Exception):
                    tmp = lock_path + ".tmp"
                    with open(tmp, "w") as fh:
                        fh.write(f"{os.getpid()},{time.time()}")
                    os.replace(tmp, lock_path)
            return self._Guard(self)

        if not (msvcrt or os.name != "nt"):
            result = super().acquire(timeout=resolved_timeout, poll_interval=poll_interval)
            if lock_path:
                with suppress(Exception):
                    tmp = lock_path + ".tmp"
                    with open(tmp, "w") as fh:
                        fh.write(f"{os.getpid()},{time.time()}")
                    os.replace(tmp, lock_path)
            return result

        context.lock_counter += 1
        try:
            if self.is_locked:
                return self._Guard(self)

            start = time.perf_counter()
            while True:
                fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, getattr(context, "mode", 0o644))
                with suppress(PermissionError):
                    if hasattr(os, "fchmod"):
                        os.fchmod(fd, getattr(context, "mode", 0o644))
                try:
                    if os.name == "nt":
                        file_end = os.lseek(fd, 0, os.SEEK_END)
                        if file_end == 0:
                            os.write(fd, b"\0")
                        os.lseek(fd, 0, os.SEEK_SET)
                        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    else:
                        flock(fd, LOCK_EX | LOCK_NB)
                    context.lock_file_fd = fd
                    break
                except OSError as exc:  # pragma: no cover - race conditions
                    os.close(fd)
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                        raise
                    if resolved_timeout >= 0 and time.perf_counter() - start >= resolved_timeout:
                        raise Timeout(lock_path)
                    time.sleep(poll_interval)

            if lock_path:
                with suppress(Exception):
                    tmp = lock_path + ".tmp"
                    with open(tmp, "w") as fh:
                        fh.write(f"{os.getpid()},{time.time()}")
                    os.replace(tmp, lock_path)
            return self._Guard(self)
        except BaseException:
            context.lock_counter = max(0, context.lock_counter - 1)
            raise

    def release(self, *args, **kwargs) -> None:  # type: ignore[override]
        try:
            super().release(*args, **kwargs)
        finally:
            lock_path = getattr(self, "lock_file", None)
            if not lock_path:
                return

            # ``WindowsFileLock`` already removes the lock file as part of
            # its ``_release`` implementation.  Attempting another removal is
            # racy because a waiting process may acquire the lock immediately
            # after it is released, keeping the file open and triggering noisy
            # ``PermissionError`` warnings.  Skip the manual cleanup on
            # Windows and let the next owner manage the lifecycle instead.
            if os.name == "nt":  # pragma: win32 cover - behaviour validated in CI
                return

            for attempt in range(5):
                try:
                    os.remove(lock_path)
                    return
                except FileNotFoundError:
                    logger.warning("lock file already removed: %s", lock_path)
                    return
                except PermissionError:
                    # Windows may keep the lock file open momentarily after
                    # releasing ``msvcrt`` locks.  Retry a few times before
                    # giving up to avoid noisy log spam on that platform.
                    if os.name == "nt":
                        time.sleep(0.1 * (attempt + 1))
                        continue
                    logger.exception("failed to remove lock file: %s", lock_path)
                    return
                except Exception as exc:
                    logger.exception("failed to remove lock file: %s", exc)
                    return

            logger.warning("failed to remove lock file after retries: %s", lock_path)


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
        timeout = getattr(self, "timeout", None)
        if timeout is None or timeout < 0:
            timeout = LOCK_TIMEOUT
        self._guard = super().acquire(timeout=timeout)
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
