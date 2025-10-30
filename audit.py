from __future__ import annotations

import json
import os
import sqlite3
import logging
from contextlib import closing, suppress
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from threading import Lock

from fcntl_compat import LOCK_EX, LOCK_UN, flock
from dynamic_path_router import resolve_dir
import hashlib
from audit_utils import safe_write_audit


# Default log file within the repository.  Resolved lazily to avoid running
# project root discovery (which may shell out to ``git``) during import.
DEFAULT_LOG_PATH: Path | None = None


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


MAX_BYTES = _env_int("DB_AUDIT_LOG_MAX_BYTES", 10 * 1024 * 1024)
BACKUP_COUNT = _env_int("DB_AUDIT_LOG_BACKUPS", 5)


_write_lock = Lock()
_logger_lock = Lock()
_loggers: dict[Path, logging.Logger] = {}
_module_logger = logging.getLogger(__name__)


class _LockedRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that locks the file during writes."""

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        with _write_lock:
            if self.stream is None:
                self.stream = self._open()
            fd = self.stream.fileno()
            flock(fd, LOCK_EX)
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


def _get_logger(path: Path) -> logging.Logger:
    with _logger_lock:
        logger = _loggers.get(path)
        if logger is None:
            logger = logging.getLogger(f"db_audit_{path}")
            logger.setLevel(logging.INFO)
            handler = _LockedRotatingFileHandler(
                path,
                maxBytes=MAX_BYTES,
                backupCount=BACKUP_COUNT,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            logger.propagate = False
            _loggers[path] = logger
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
        ``shared_db_audit`` table.
    """

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "table": table_name,
        "rows": row_count,
        "menace_id": menace_id,
    }

    # Determine log path and ensure directory exists
    if log_path is not None:
        path = Path(log_path).resolve()
    else:
        path = _default_log_path()
    state_path = Path(f"{path}.state")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        with state_path.open("a+") as sf:
            fd = sf.fileno()
            flock(fd, LOCK_EX)
            sf.seek(0)
            prev_hash = sf.read().strip() or "0" * 64
            data = json.dumps(record, sort_keys=True)
            new_hash = hashlib.sha256((prev_hash + data).encode()).hexdigest()
            record["hash"] = new_hash
            logger = _get_logger(path)
            logger.info(json.dumps(record, sort_keys=True))
            sf.seek(0)
            sf.truncate()
            sf.write(new_hash)
            sf.flush()
            os.fsync(fd)
            flock(fd, LOCK_UN)
    except OSError:
        # Logging failures are non-fatal
        pass

    db_path: str | None = None
    if db_conn is not None:
        if getattr(db_conn, "_closed", False):
            return
        try:
            db_path = _connection_database_path(db_conn)
        except sqlite3.Error as exc:
            _module_logger.debug("failed to resolve audit database path: %s", exc)
            db_path = None

    if db_path and db_path not in (":memory:", ""):
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        try:
            def _persist(conn: sqlite3.Connection) -> None:
                with closing(conn.execute("PRAGMA journal_mode=WAL;")) as pragma:
                    pragma.fetchall()
                try:
                    with closing(conn.cursor()) as cur:
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
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

            safe_write_audit(db_path, _persist, logger=_module_logger)
        except (sqlite3.Error, SystemError) as exc:
            _module_logger.debug("failed to persist shared_db_audit entry: %s", exc)


__all__ = ["log_db_access"]
