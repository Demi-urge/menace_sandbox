from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock

from fcntl_compat import LOCK_EX, LOCK_UN, flock


# Default log file within the repository
DEFAULT_LOG_PATH = Path(__file__).resolve().parent / "logs" / "shared_db_access.log"
DEFAULT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_write_lock = Lock()


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
    path = Path(log_path) if log_path is not None else DEFAULT_LOG_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(record, sort_keys=True)
        with _write_lock:
            with path.open("a", encoding="utf-8") as fh:
                flock(fh.fileno(), LOCK_EX)
                try:
                    fh.write(data)
                    fh.write("\n")
                finally:
                    flock(fh.fileno(), LOCK_UN)
    except OSError:
        # Logging failures are non-fatal
        pass

    if db_conn is not None:
        try:
            cur = db_conn.cursor()
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
            db_conn.commit()
        except sqlite3.Error:
            # Database logging is best-effort
            pass


__all__ = ["log_db_access"]
