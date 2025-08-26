"""Synchronise queued writes with a shared SQLite database.

The script scans a queue directory for ``*_queue.jsonl`` files. Each line in
these files represents an ``INSERT`` operation produced by
``db_write_queue.queue_insert``.  Records are attempted against the shared
database using :func:`db_dedup.insert_if_unique`.  Successful inserts – or rows
that already exist – are removed from the queue.  Malformed records are moved to
``<table>_queue.error.jsonl`` while failed inserts are appended to
``<table>_queue.failed.jsonl`` along with error details.

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

from db_dedup import compute_content_hash, insert_if_unique
from db_write_queue import DEFAULT_QUEUE_DIR
from fcntl_compat import LOCK_EX, LOCK_UN, flock


logger = logging.getLogger(__name__)


def _table_from_path(path: Path) -> str:
    """Return table name for a queue file path."""

    stem = path.stem  # e.g. ``foo_queue``
    if stem.endswith("_queue"):
        stem = stem[: -len("_queue")]
    return stem


def process_queue_file(path: Path, *, conn: sqlite3.Connection, max_retries: int) -> None:
    """Process pending operations from ``path``.

    Each line in *path* is treated independently.  Successful inserts are
    removed from the queue file.  Lines that raise an exception or are malformed
    are appended to ``<table>_queue.error.jsonl``.  Failed inserts are appended
    to ``<table>_queue.failed.jsonl`` with error details.  Records may be
    retried up to ``max_retries`` times before being left in the failed queue.
    """

    error_path = path.with_name(f"{path.stem}.error.jsonl")
    failed_path = path.with_name(f"{path.stem}.failed.jsonl")
    with path.open("r+", encoding="utf-8") as fh:
        flock(fh.fileno(), LOCK_EX)
        lines = fh.readlines()
        fh.seek(0)
        fh.truncate()

        remaining: list[str] = []

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
            retries = int(record.get("retries", 0))
            hash_fields = record.get("hash_fields") or data.pop("hash_fields", None)
            hash_fields = list(hash_fields or data.keys())

            try:
                payload = {k: data[k] for k in hash_fields}
                content_hash = compute_content_hash(payload)
                existing = conn.execute(
                    f"SELECT id FROM {table} WHERE content_hash=?",
                    (content_hash,),
                ).fetchone()
                if existing:
                    logger.info(
                        "queue_insert_duplicate",
                        extra={
                            "event": "duplicate",
                            "table": table,
                            "menace_id": menace_id,
                            "id": int(existing[0]),
                        },
                    )
                    continue

                inserted_id = insert_if_unique(
                    table,
                    data,
                    hash_fields,
                    menace_id,
                    logger=logger,
                    conn=conn,
                )
                conn.commit()
                logger.info(
                    "queue_insert_success",
                    extra={
                        "event": "success",
                        "table": table,
                        "menace_id": menace_id,
                        "id": inserted_id,
                    },
                )
            except Exception as exc:  # pragma: no cover - logged then moved to failed file
                conn.rollback()
                retries += 1
                logger.error(
                    "queue_insert_failure",
                    extra={
                        "event": "failure",
                        "table": table,
                        "menace_id": menace_id,
                        "error": str(exc),
                        "retries": retries,
                    },
                )
                failed_entry = {"record": record, "error": str(exc)}
                _append_lines(
                    failed_path, [json.dumps(failed_entry, sort_keys=True) + "\n"]
                )
                if retries < max_retries:
                    record["retries"] = retries
                    remaining.append(json.dumps(record, sort_keys=True) + "\n")

        fh.writelines(remaining)
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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed inserts",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    conn = sqlite3.connect(args.db_path)
    try:
        queue_dir = Path(args.queue_dir)
        while True:
            for file in queue_dir.glob("*_queue.jsonl"):
                process_queue_file(file, conn=conn, max_retries=args.max_retries)
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

