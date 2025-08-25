from __future__ import annotations

"""Utilities for buffering database writes to disk.

Records are appended to per-table JSONL files under a configurable queue
directory.  File locking ensures that writes are atomic across threads and
processes.  This is a lightweight helper used to collect writes that may be
replayed later by another process.
"""

from pathlib import Path
import json
import os
from threading import Lock
from typing import Mapping, Iterable, Any

from fcntl import flock, LOCK_EX, LOCK_UN

# Default directory where queue files are stored.  Callers may override via the
# ``MENACE_QUEUE_DIR`` environment variable.
DEFAULT_QUEUE_DIR = Path(os.getenv("MENACE_QUEUE_DIR", "./queue"))

# In-process lock to guard concurrent writers.  ``flock`` handles cross-process
# safety but a threading lock avoids interleaving within a process.
_write_lock = Lock()


def append_to_queue(
    table_name: str,
    values: Mapping[str, Any],
    menace_id: str,
    *,
    hash_fields: Iterable[str] | None = None,
    queue_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Append a record to the table-specific queue file.

    Parameters
    ----------
    table_name:
        Name of the target table.
    values:
        Mapping of column names to values for the pending row.
    menace_id:
        Identifier of the menace instance generating the record.
    hash_fields:
        Iterable of field names describing how to deduplicate the record.  The
        names are stored verbatim without additional processing.
    queue_dir:
        Optional directory where queue files are stored.  Defaults to the path
        defined by ``MENACE_QUEUE_DIR`` or ``./queue``.
    """
    payload = {
        "table": table_name,
        "values": dict(values),
        "source_menace_id": menace_id,
        "hash_fields": list(hash_fields or []),
    }

    base = Path(queue_dir) if queue_dir is not None else DEFAULT_QUEUE_DIR
    base.mkdir(parents=True, exist_ok=True)
    file_path = base / f"{table_name}_queue.jsonl"

    record = json.dumps(payload, ensure_ascii=False) + "\n"

    with _write_lock:
        with file_path.open("a", encoding="utf-8") as fh:
            flock(fh.fileno(), LOCK_EX)
            fh.write(record)
            flock(fh.fileno(), LOCK_UN)


def buffer_shared_insert(
    table_name: str,
    values: Mapping[str, Any],
    hash_fields: Iterable[str] | None = None,
    *,
    queue_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Helper for queuing inserts into shared database tables.

    The ``source_menace_id`` is populated from the ``MENACE_ID`` environment
    variable and the resulting record is delegated to :func:`append_to_queue`.
    """
    menace_id = os.getenv("MENACE_ID", "")
    append_to_queue(
        table_name,
        values,
        menace_id,
        hash_fields=hash_fields,
        queue_dir=queue_dir,
    )
