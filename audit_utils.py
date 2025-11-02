"""Utility helpers shared by the audit logging modules."""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Callable

LOGGER = logging.getLogger(__name__)


def configure_audit_sqlite_connection(conn: sqlite3.Connection) -> None:
    """Configure *conn* for concurrent audit access.

    Enabling WAL mode ensures readers are not blocked by writers, while the
    busy timeout and synchronous level keep contention manageable for the
    high-frequency audit workload.
    """

    try:
        with closing(conn.execute("PRAGMA journal_mode=WAL;")) as pragma:
            pragma.fetchall()
    except sqlite3.OperationalError:
        LOGGER.debug("Failed to enable WAL mode for audit DB", exc_info=True)
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")


def _audit_file_mode_enabled() -> bool:
    value = os.getenv("AUDIT_FILE_MODE")
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def safe_write_audit(
    db_path: str | Path,
    write_fn: Callable[[sqlite3.Connection], None],
    *,
    timeout: float | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Persist audit data retrying transient SQLite locks with backoff."""

    log = logger or LOGGER

    if _audit_file_mode_enabled():
        log.debug(
            "AUDIT_FILE_MODE enabled; skipping SQLite audit persistence for %s", db_path
        )
        return

    connect_path = str(db_path)
    connect_kwargs: dict[str, float] = {}
    if timeout is not None:
        connect_kwargs["timeout"] = timeout

    for attempt in range(10):
        try:
            with sqlite3.connect(connect_path, **connect_kwargs) as conn:
                configure_audit_sqlite_connection(conn)
                write_fn(conn)
            return
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc).lower():
                time.sleep(0.05 * (attempt + 1))
                continue
            raise
    log.warning("audit write dropped: DB locked too long")


__all__ = ["configure_audit_sqlite_connection", "safe_write_audit"]
