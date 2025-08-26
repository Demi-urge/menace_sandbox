from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from threading import Lock

from fcntl_compat import flock, LOCK_EX, LOCK_UN


_write_lock = Lock()


def log_db_access(
    action: str,
    table_name: str,
    row_count: int,
    menace_id: str,
    log_path: str = "shared_db_access.log",
) -> None:
    """Append a database access record to ``log_path``.

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
        File to append the log line to. Defaults to ``"shared_db_access.log"``.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "table": table_name,
        "rows": row_count,
        "menace_id": menace_id,
    }
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(record, sort_keys=True)

    with _write_lock:
        with path.open("a", encoding="utf-8") as fh:
            flock(fh.fileno(), LOCK_EX)
            fh.write(data)
            fh.write("\n")
            flock(fh.fileno(), LOCK_UN)


__all__ = ["log_db_access"]
