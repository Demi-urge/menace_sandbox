from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock

from fcntl_compat import LOCK_EX, LOCK_UN, flock
import db_router


_write_lock = Lock()


def log_db_access(
    action: str,
    table_name: str,
    row_count: int,
    menace_id: str,
    *,
    log_to_db: bool = False,
    log_path: str = "logs/shared_db_access.log",
) -> None:
    """Append a database access record to ``log_path`` and optionally to SQLite.

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
    log_to_db:
        When ``True`` the entry is inserted into the ``db_access_audit`` table
        using the local database connection.
    log_path:
        File to append the log line to. Defaults to ``"logs/shared_db_access.log"``.
    """

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "table": table_name,
        "rows": row_count,
        "menace_id": menace_id,
    }

    path = Path(log_path)
    data = json.dumps(record, sort_keys=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
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

    if log_to_db:
        try:
            router = db_router.GLOBAL_ROUTER or db_router.init_db_router(menace_id)
            conn = router.local_conn
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS db_access_audit (
                    timestamp TEXT,
                    action TEXT,
                    table_name TEXT,
                    row_count INTEGER,
                    menace_id TEXT
                )
                """
            )
            cur.execute(
                "INSERT INTO db_access_audit (timestamp, action, table_name, row_count, menace_id)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    record["timestamp"],
                    action,
                    table_name,
                    row_count,
                    menace_id,
                ),
            )
            conn.commit()
        except Exception:
            # Database logging is best-effort
            pass


__all__ = ["log_db_access"]
