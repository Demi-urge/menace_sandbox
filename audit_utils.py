"""Utility helpers shared by the audit logging modules."""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Callable


LOGGER = logging.getLogger(__name__)


def safe_write_audit(
    db_path: str | Path,
    write_fn: Callable[[sqlite3.Connection], None],
    *,
    timeout: float | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Persist audit data retrying transient SQLite locks with backoff."""

    log = logger or LOGGER
    connect_path = str(db_path)
    connect_kwargs: dict[str, float] = {}
    if timeout is not None:
        connect_kwargs["timeout"] = timeout

    for attempt in range(10):
        try:
            with sqlite3.connect(connect_path, **connect_kwargs) as conn:
                with closing(conn.execute("PRAGMA journal_mode=WAL;")) as pragma:
                    pragma.fetchall()
                conn.execute("PRAGMA busy_timeout = 5000;")
                write_fn(conn)
            return
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc).lower():
                time.sleep(0.05 * (attempt + 1))
                continue
            raise
    log.warning("audit write dropped: DB locked too long")


__all__ = ["safe_write_audit"]

