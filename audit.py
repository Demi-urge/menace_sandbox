from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import io
from contextlib import closing
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock

from audit_utils import configure_audit_sqlite_connection
from dynamic_path_router import resolve_dir
from fcntl_compat import LOCK_EX, LOCK_NB, LOCK_UN, flock


# Default log file within the repository.  Resolved lazily to avoid running
# project root discovery (which may shell out to ``git``) during import.
DEFAULT_LOG_PATH: Path | None = None


def _audit_file_mode_enabled() -> bool:
    value = os.getenv("AUDIT_FILE_MODE")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _file_logging_disabled() -> bool:
    value = os.getenv("DB_AUDIT_DISABLE_FILE")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MAX_BYTES = _env_int("DB_AUDIT_LOG_MAX_BYTES", 10 * 1024 * 1024)
BACKUP_COUNT = _env_int("DB_AUDIT_LOG_BACKUPS", 5)
LOCK_TIMEOUT = _env_float("DB_AUDIT_LOCK_TIMEOUT", 1.0)
BOOTSTRAP_LOCK_TIMEOUT = _env_float("DB_AUDIT_BOOTSTRAP_LOCK_TIMEOUT", 0.2)
LOCK_RETRY_DELAY = _env_float("DB_AUDIT_LOCK_RETRY_DELAY", 0.05)
TIMING_ENABLED = _env_flag("DB_AUDIT_TIMING", False)


_write_lock = Lock()
_logger_lock = Lock()
_loggers: dict[tuple[Path, bool], logging.Logger] = {}
_module_logger = logging.getLogger(__name__)
_timing_first_call_logged = False


def _acquire_lock(fd: int, *, timeout: float | None) -> bool:
    """Attempt to acquire an exclusive lock on *fd* within *timeout* seconds."""

    deadline = None if timeout is None or timeout <= 0 else time.monotonic() + timeout
    delay = LOCK_RETRY_DELAY if LOCK_RETRY_DELAY > 0 else 0.01
    while True:
        try:
            flock(fd, LOCK_EX | LOCK_NB)
            return True
        except BlockingIOError:
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(delay)
        except OSError:
            return False


class _LockedRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that locks the file during writes."""

    def __init__(self, *args: object, lock_timeout: float = LOCK_TIMEOUT, **kwargs: object):
        super().__init__(*args, **kwargs)
        self.lock_timeout = lock_timeout

    def _open(self):  # type: ignore[override]
        raw = open(self.baseFilename, "a+b")
        return io.TextIOWrapper(raw, encoding=self.encoding or "utf-8", newline="")

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        with _write_lock:
            if self.stream is None:
                self.stream = self._open()
            fd = self.stream.fileno()
            if not _acquire_lock(fd, timeout=self.lock_timeout):
                _module_logger.warning("audit log file locked; skipping emit")
                return
            try:
                if self.shouldRollover(record):
                    self.doRollover()
                logging.FileHandler.emit(self, record)
            finally:
                flock(fd, LOCK_UN)


def _default_log_path() -> Path:
    """Return the default log path, initialising it lazily."""

    global DEFAULT_LOG_PATH

    if DEFAULT_LOG_PATH is None:
        DEFAULT_LOG_PATH = (resolve_dir("logs") / "shared_db_access.log").resolve()

    return Path(DEFAULT_LOG_PATH)


def _emit_timing(
    stage: str,
    *,
    start: float | None,
    bootstrap_safe: bool,
    log_path: Path | None,
    first_call: bool,
    **details: object,
) -> None:
    if not TIMING_ENABLED or start is None:
        return

    payload: dict[str, object] = {
        "stage": stage,
        "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
        "bootstrap_safe": bootstrap_safe,
        "first_call": first_call,
    }
    if log_path is not None:
        payload["log_path"] = str(log_path)
    if details:
        payload.update(details)
    try:
        _module_logger.info("audit timing %s", json.dumps(payload, sort_keys=True))
    except Exception:  # pragma: no cover - diagnostic helper
        _module_logger.info("audit timing %s", payload)


def _open_state_file(path: Path, *, bootstrap_safe: bool) -> io.BufferedRandom | None:
    """Return a handle to the audit state file or ``None`` when unavailable."""

    state_path = Path(f"{path}.state")

    if bootstrap_safe:
        # Avoid creating new directories/files during bootstrap, which can block on
        # networked filesystems or restricted environments.  If the audit artefacts
        # are not already present, skip file logging instead of risking stalls.
        if not path.parent.exists() or not state_path.exists():
            return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _module_logger.warning(
            "audit log directory unavailable for %s: %s", path.parent, exc
        )
        return None

    if os.name == "nt" and not os.access(path.parent, os.W_OK):
        _module_logger.warning(
            "audit log directory is not writable on Windows: %s", path.parent
        )
        return None

    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(state_path, flags, 0o600)
        try:
            return io.BufferedRandom(os.fdopen(fd, "r+b", buffering=0))
        except Exception:
            os.close(fd)
            raise
    except OSError:
        _module_logger.warning(
            "audit state file unavailable for %s; skipping audit write", state_path,
            exc_info=not bootstrap_safe,
        )
        return None


def _get_logger(path: Path, *, bootstrap_safe: bool = False) -> logging.Logger:
    with _logger_lock:
        key = (path, bootstrap_safe)
        logger = _loggers.get(key)
        if logger is None:
            logger = logging.getLogger(f"db_audit_{path}")
            logger.setLevel(logging.INFO)
            lock_timeout = (
                BOOTSTRAP_LOCK_TIMEOUT if bootstrap_safe else LOCK_TIMEOUT
            )
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                _module_logger.warning(
                    "audit log directory unavailable for %s: %s", path.parent, exc
                )
                _loggers[key] = _module_logger
                return _module_logger
            handler = _LockedRotatingFileHandler(
                path,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
                lock_timeout=lock_timeout,
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            logger.propagate = False
            _loggers[key] = logger
        return logger


def _connection_database_path(conn: sqlite3.Connection) -> str:
    """Return the filesystem path backing *conn*'s ``main`` database."""

    with closing(conn.execute("PRAGMA database_list")) as cur:
        for _, name, path in cur.fetchall():
            if name == "main":
                return path or ":memory:"
    return ":memory:"


def log_db_access(
    action: str,
    table_name: str,
    row_count: int,
    menace_id: str,
    *,
    log_path: Path | None = None,
    db_conn: sqlite3.Connection | None = None,
    log_to_db: bool = False,
    bootstrap_safe: bool = False,
) -> None:
    """Record a database access event to a JSONL file and optional SQLite table.

    Parameters
    ----------
    action:
        Operation performed (e.g. ``"read"`` or ``"write"``).
    table_name:
        Name of the table accessed.
    row_count:
        Number of rows affected by the action.
    menace_id:
        Identifier of the menace instance performing the operation.
    log_path:
        Optional path of the log file. Defaults to ``logs/shared_db_access.log``
        within the repository.
    db_conn:
        Optional sqlite3 connection used to persist the record into the
        ``shared_db_audit`` table when ``log_to_db`` is ``True``.
    log_to_db:
        When ``True`` the entry is also written to the SQLite audit mirror
        referenced by ``db_conn``.  Defaults to ``False`` to avoid blocking
        active writers when the audit database is locked.
    bootstrap_safe:
        Use a shortened lock timeout and skip logging when the audit file is
        contended. This applies to both the state and JSONL files so bootstrap
        or read-only flows never block on audit writes, emitting a warning when
        contention is detected.
    """

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "table": table_name,
        "rows": row_count,
        "menace_id": menace_id,
    }

    timing_start = time.perf_counter() if TIMING_ENABLED else None
    global _timing_first_call_logged
    timing_first_call = False
    if TIMING_ENABLED and not _timing_first_call_logged:
        timing_first_call = True
        _timing_first_call_logged = True

    resolved_path: Path | None = None

    if not _file_logging_disabled():
        # Determine log path and ensure directory exists
        resolved_path = Path(log_path).resolve() if log_path is not None else _default_log_path()
        _emit_timing(
            "path_resolved",
            start=timing_start,
            bootstrap_safe=bootstrap_safe,
            log_path=resolved_path,
            first_call=timing_first_call,
        )

        lock_timeout = BOOTSTRAP_LOCK_TIMEOUT if bootstrap_safe else LOCK_TIMEOUT
        state_file = _open_state_file(resolved_path, bootstrap_safe=bootstrap_safe)
        if state_file is None:
            _emit_timing(
                "state_file_unavailable",
                start=timing_start,
                bootstrap_safe=bootstrap_safe,
                log_path=resolved_path,
                first_call=timing_first_call,
            )
        else:
            try:
                with state_file as sf:
                    fd = sf.fileno()
                    if not _acquire_lock(fd, timeout=lock_timeout):
                        _module_logger.warning(
                            "skipping audit log write for %s: lock held elsewhere", resolved_path
                        )
                        _emit_timing(
                            "state_lock_contended",
                            start=timing_start,
                            bootstrap_safe=bootstrap_safe,
                            log_path=resolved_path,
                            first_call=timing_first_call,
                            timeout=lock_timeout,
                        )
                        return
                    _emit_timing(
                        "state_lock_acquired",
                        start=timing_start,
                        bootstrap_safe=bootstrap_safe,
                        log_path=resolved_path,
                        first_call=timing_first_call,
                        timeout=lock_timeout,
                    )
                    try:
                        sf.seek(0)
                        prev_hash_bytes = sf.read()
                        prev_hash = (
                            prev_hash_bytes.decode().strip() if prev_hash_bytes else "0" * 64
                        )
                        data = json.dumps(record, sort_keys=True)
                        new_hash = hashlib.sha256((prev_hash + data).encode()).hexdigest()
                        record["hash"] = new_hash
                        logger = _get_logger(resolved_path, bootstrap_safe=bootstrap_safe)
                        logger.info(json.dumps(record, sort_keys=True))
                        sf.seek(0)
                        sf.truncate()
                        sf.write(new_hash.encode())
                        sf.flush()
                        if not bootstrap_safe:
                            os.fsync(fd)
                    finally:
                        flock(fd, LOCK_UN)
            except OSError:
                # Logging failures are non-fatal
                _module_logger.warning(
                    "audit file write failed; continuing without audit entry", exc_info=True
                )

    if _audit_file_mode_enabled():
        _emit_timing(
            "file_mode_enabled",
            start=timing_start,
            bootstrap_safe=bootstrap_safe,
            log_path=resolved_path,
            first_call=timing_first_call,
        )
        return

    if not log_to_db or db_conn is None:
        _emit_timing(
            "complete",
            start=timing_start,
            bootstrap_safe=bootstrap_safe,
            log_path=resolved_path,
            first_call=timing_first_call,
            db_logging=False,
        )
        return

    if getattr(db_conn, "_closed", False):
        _emit_timing(
            "complete",
            start=timing_start,
            bootstrap_safe=bootstrap_safe,
            log_path=resolved_path,
            first_call=timing_first_call,
            db_logging=False,
        )
        return

    initial_tx = db_conn.in_transaction
    try:
        if not initial_tx:
            configure_audit_sqlite_connection(db_conn)
        base_cursor = sqlite3.Connection.cursor(db_conn, factory=sqlite3.Cursor)
        with closing(base_cursor) as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS shared_db_audit (
                    action TEXT,
                    "table" TEXT,
                    rows INTEGER,
                    menace_id TEXT,
                    timestamp TEXT
                )
                """
            )
            cur.execute(
                'INSERT INTO shared_db_audit (action, "table", rows, menace_id, timestamp)'
                ' VALUES (?, ?, ?, ?, ?)',
                (action, table_name, row_count, menace_id, record["timestamp"]),
            )
        if not initial_tx:
            db_conn.commit()
    except sqlite3.Error as exc:
        _module_logger.debug("failed to persist shared_db_audit entry: %s", exc)
    finally:
        _emit_timing(
            "complete",
            start=timing_start,
            bootstrap_safe=bootstrap_safe,
            log_path=resolved_path,
            first_call=timing_first_call,
            db_logging=True,
        )


__all__ = ["log_db_access"]
