from __future__ import annotations

"""Thread-safe persistent queue for deferred database writes.

The module provides two lightweight helpers for queuing writes destined for a
shared database:

* :func:`queue_insert` – legacy helper that writes to per-table queue files.
* :func:`append_record` – new helper that appends records to per-menace queue
  files.  These files may later be processed by a background worker.

Queue files live under the directory specified by the ``DB_QUEUE_DIR"
environment variable (``logs/queue`` by default).  ``fcntl_compat`` file locks
are used to avoid interleaving writes from concurrent threads or processes.
"""

from pathlib import Path
import json
import os
from threading import Lock
from typing import Iterable, Iterator, List, Mapping, Any

from fcntl_compat import flock, LOCK_EX, LOCK_UN
from db_dedup import compute_content_hash

# Base directory for queue files.  Allow overriding via the ``DB_QUEUE_DIR``
# environment variable.  By default queue files are stored under ``logs/queue``.
DEFAULT_QUEUE_DIR = Path(os.getenv("DB_QUEUE_DIR", "logs/queue"))

# Global lock for in-process thread safety.  File locks handle cross-process
# coordination.
_write_lock = Lock()


def _queue_file(table: str, queue_path: str | Path | None) -> Path:
    """Return the path to the queue file for *table*, ensuring directories exist."""
    base = Path(queue_path) if queue_path is not None else DEFAULT_QUEUE_DIR
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{table}_queue.jsonl"


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    """Append *record* as JSON to *path* with file locking."""
    data = json.dumps(record, sort_keys=True)
    with _write_lock:
        with path.open("a", encoding="utf-8") as fh:
            flock(fh.fileno(), LOCK_EX)
            fh.write(data)
            fh.write("\n")
            flock(fh.fileno(), LOCK_UN)


def append_record(
    table: str,
    payload: Mapping[str, Any],
    menace_id: str,
    queue_dir: Path | None = None,
) -> None:
    """Append ``payload`` for ``table`` to a menace-specific queue file.

    Parameters
    ----------
    table:
        Target table name for the write.
    payload:
        Mapping of column names to values.
    menace_id:
        Identifier of the menace instance producing the record.
    queue_dir:
        Optional base directory for queue files.  If omitted the directory is
        derived from :data:`DEFAULT_QUEUE_DIR` which may be overridden via the
        ``DB_QUEUE_DIR`` environment variable.
    """

    base = queue_dir if queue_dir is not None else DEFAULT_QUEUE_DIR
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{menace_id}.jsonl"

    record = {
        "table": table,
        "data": dict(payload),
        "source_menace_id": menace_id,
    }

    _write_record(path, record)


def iter_queue_files(queue_dir: Path | None = None) -> Iterator[Path]:
    """Yield menace-specific queue files within ``queue_dir``."""

    base = queue_dir if queue_dir is not None else DEFAULT_QUEUE_DIR
    if not base.exists():
        return
    for path in sorted(base.glob("*.jsonl")):
        # Skip legacy per-table queue files
        if path.name.endswith("_queue.jsonl"):
            continue
        if path.is_file():
            yield path


def read_queue(path: Path) -> List[dict]:
    """Return all records from ``path`` as a list of dictionaries."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def remove_processed_lines(path: Path, processed: int) -> None:
    """Remove the first ``processed`` lines from ``path``."""

    if processed <= 0 or not path.exists():
        return

    with _write_lock:
        with path.open("r+", encoding="utf-8") as fh:
            flock(fh.fileno(), LOCK_EX)
            lines = fh.readlines()
            fh.seek(0)
            fh.writelines(lines[processed:])
            fh.truncate()
            flock(fh.fileno(), LOCK_UN)


def queue_insert(
    table: str,
    values: Mapping[str, Any],
    hash_fields: Iterable[str],
    queue_path: str | Path | None = None,
) -> str:
    """Queue an ``INSERT`` operation for ``table``.

    Parameters
    ----------
    table:
        Target table name.
    values:
        Mapping of column names to values for the row.
    hash_fields:
        Iterable of field names from ``values`` used to compute a content hash
        for deduplication.
    queue_path:
        Optional directory where queue files are stored.  Defaults to
        ``logs/queue`` as defined by :data:`DEFAULT_QUEUE_DIR`.

    Returns
    -------
    str
        The computed content hash.
    """
    missing = [field for field in hash_fields if field not in values]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise KeyError(f"Missing fields for hashing: {missing_list}")

    payload = {key: values[key] for key in hash_fields}
    content_hash = compute_content_hash(payload)

    record = {
        "table": table,
        "op": "insert",
        "data": dict(values),
        "content_hash": content_hash,
        "source_menace_id": os.getenv("MENACE_ID", ""),
    }

    file_path = _queue_file(table, queue_path)
    _write_record(file_path, record)
    return content_hash
