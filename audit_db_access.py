from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from threading import Lock

from fcntl_compat import flock, LOCK_EX, LOCK_UN

DB_ACCESS_LOG_PATH = Path(os.getenv("DB_ACCESS_LOG_PATH", "logs/shared_db_access.log"))
DB_ACCESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

_write_lock = Lock()


def log_db_access(action: str, table_name: str, row_count: int, menace_id: str) -> None:
    """Append a database access record to ``DB_ACCESS_LOG_PATH``.

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
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "table": table_name,
        "rows": row_count,
        "menace_id": menace_id,
    }
    data = json.dumps(record, sort_keys=True)

    with _write_lock:
        with DB_ACCESS_LOG_PATH.open("a", encoding="utf-8") as fh:
            flock(fh.fileno(), LOCK_EX)
            fh.write(data)
            fh.write("\n")
            flock(fh.fileno(), LOCK_UN)


__all__ = ["log_db_access", "DB_ACCESS_LOG_PATH"]
