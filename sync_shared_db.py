"""Synchronise queued writes with a shared SQLite database.

The script scans a queue directory for ``*_queue.jsonl`` files. Each line in
these files represents an ``INSERT`` operation produced by
``db_write_queue.queue_insert``.  Records are attempted against the shared
database using :func:`db_dedup.insert_if_unique`.  Successful inserts – or rows
that already exist – are removed from the queue.  Failures are moved to a
``<table>_queue.error.jsonl`` file for inspection.

The utility may run continuously with a polling interval or execute a single
iteration via ``--once``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable

from db_dedup import insert_if_unique
from db_write_queue import DEFAULT_QUEUE_DIR
from fcntl_compat import LOCK_EX, LOCK_UN, flock


logger = logging.getLogger(__name__)


def _table_from_path(path: Path) -> str:
    """Return table name for a queue file path."""

    stem = path.stem  # e.g. ``foo_queue``
    if stem.endswith("_queue"):
        stem = stem[: -len("_queue")]
    return stem


def process_queue_file(path: Path, *, conn: sqlite3.Connection) -> None:
    """Process pending operations from ``path``.

    Each line in *path* is treated independently.  Successful inserts are
    removed from the queue file.  Lines that raise an exception or are malformed
    are appended to ``<table>_queue.error.jsonl``.
    """

    error_path = path.with_name(f"{path.stem}.error.jsonl")
    with path.open("r+", encoding="utf-8") as fh:
        flock(fh.fileno(), LOCK_EX)
        lines = fh.readlines()
        fh.seek(0)
        fh.truncate()

        for raw in lines:
            line = raw.rstrip("\n")
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:  # pragma: no cover - logged then moved to error file
                logger.exception("Malformed queue record in %s", path.name)
                _append_lines(error_path, [raw])
                continue

            table = record.get("table") or _table_from_path(path)
            data = record.get("data", {})
            menace_id = record.get("source_menace_id", "")
            hash_fields = list(data.keys())

            try:
                insert_if_unique(
                    table,
                    data,
                    hash_fields,
                    menace_id,
                    logger=logger,
                    conn=conn,
                )
                conn.commit()
            except Exception:  # pragma: no cover - logged then moved to error file
                logger.exception("Failed to insert queued record for %s", table)
                _append_lines(error_path, [json.dumps(record, sort_keys=True) + "\n"])

        flock(fh.fileno(), LOCK_UN)


def _append_lines(path: Path, lines: Iterable[str]) -> None:
    """Append *lines* to *path* creating directories if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", default=str(DEFAULT_QUEUE_DIR))
    parser.add_argument("--db-path", default="menace.db")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true", help="Run a single iteration")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    conn = sqlite3.connect(args.db_path)
    try:
        queue_dir = Path(args.queue_dir)
        while True:
            for file in queue_dir.glob("*_queue.jsonl"):
                process_queue_file(file, conn=conn)
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

