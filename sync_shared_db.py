#!/usr/bin/env python3
"""Synchronise queued database writes with the shared database.

This utility scans a directory for ``*.jsonl`` queue files.  Each line in a
queue file is expected to be a JSON mapping of the form::

    {"table": "<table>", "record": {...}, "menace_id": "<id>"}

The ``record`` mapping is inserted into ``table`` using
``db_dedup.insert_if_unique``.  The hash used for deduplication is computed from
all fields in ``record``.  If the hash already exists, the insert is skipped and
the queue entry is discarded.  Failed inserts are moved to
``queue.failed.jsonl`` for later inspection.

The script may run once via ``--once`` or loop with a configurable polling
``--interval``.  When ``--watch`` is provided and the optional ``watchdog``
package is available, filesystem events trigger immediate processing.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from db_dedup import compute_content_hash, insert_if_unique
from db_write_queue import DEFAULT_QUEUE_DIR
from fcntl_compat import LOCK_EX, LOCK_UN, flock


logger = logging.getLogger(__name__)


@dataclass
class Stats:
    """Simple container for processing statistics."""

    processed: int = 0
    duplicates: int = 0
    failures: int = 0

    def as_dict(self) -> dict[str, int]:  # pragma: no cover - trivial
        return {
            "processed": self.processed,
            "duplicates": self.duplicates,
            "failures": self.failures,
        }


def _append_lines(path: Path, lines: Iterable[str]) -> None:
    """Append ``lines`` to ``path`` creating directories if required."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line)


def process_queue_file(path: Path, *, conn: sqlite3.Connection) -> Stats:
    """Process all records from ``path`` returning :class:`Stats`.

    Each line in ``path`` is handled independently.  Successful inserts and
    detected duplicates are removed from the queue file.  Lines that raise an
    exception are moved to ``queue.failed.jsonl`` along with error details.
    """

    failed_path = path.parent / "queue.failed.jsonl"
    stats = Stats()

    with path.open("r+", encoding="utf-8") as fh:
        flock(fh.fileno(), LOCK_EX)
        lines = fh.readlines()
        fh.seek(0)
        fh.truncate()

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except Exception as exc:  # pragma: no cover - logged then skipped
                stats.failures += 1
                _append_lines(
                    failed_path,
                    [json.dumps({"record": raw.rstrip("\n"), "error": str(exc)}) + "\n"],
                )
                continue

            table = payload.get("table")
            record = payload.get("record") or payload.get("data", {})
            menace_id = payload.get("menace_id") or payload.get("source_menace_id", "")

            if not table or not isinstance(record, dict):
                stats.failures += 1
                _append_lines(
                    failed_path,
                    [
                        json.dumps(
                            {"record": payload, "error": "missing table or record"},
                            sort_keys=True,
                        )
                        + "\n"
                    ],
                )
                continue

            hash_fields = list(record.keys())
            content_hash = compute_content_hash({k: record[k] for k in hash_fields})

            try:
                existing = conn.execute(
                    f"SELECT id FROM {table} WHERE content_hash=?", (content_hash,)
                ).fetchone()
                if existing:
                    stats.duplicates += 1
                    logger.info(
                        "duplicate", extra={"table": table, "menace_id": menace_id, "id": existing[0]}
                    )
                    continue

                insert_if_unique(
                    table,
                    record,
                    hash_fields,
                    menace_id,
                    logger=logger,
                    conn=conn,
                )
                conn.commit()
                stats.processed += 1
                logger.info(
                    "inserted",
                    extra={"table": table, "menace_id": menace_id, "hash": content_hash},
                )
            except Exception as exc:  # pragma: no cover - logged then moved to failed
                conn.rollback()
                stats.failures += 1
                logger.error(
                    "failed",
                    extra={"table": table, "menace_id": menace_id, "error": str(exc)},
                )
                _append_lines(
                    failed_path,
                    [
                        json.dumps({"record": payload, "error": str(exc)}, sort_keys=True) + "\n"
                    ],
                )

        flock(fh.fileno(), LOCK_UN)

    return stats


def _sync_once(queue_dir: Path, conn: sqlite3.Connection) -> Stats:
    """Process all queue files under ``queue_dir`` returning cumulative stats."""

    stats = Stats()
    if not queue_dir.exists():
        return stats

    for file in sorted(queue_dir.glob("*.jsonl")):
        if file.name == "queue.failed.jsonl":
            continue
        file_stats = process_queue_file(file, conn=conn)
        stats.processed += file_stats.processed
        stats.duplicates += file_stats.duplicates
        stats.failures += file_stats.failures
    return stats


def _run_polling(queue_dir: Path, conn: sqlite3.Connection, interval: float, once: bool) -> None:
    while True:
        stats = _sync_once(queue_dir, conn)
        logger.info("sync_stats", extra=stats.as_dict())
        if once:
            break
        time.sleep(interval)


def _run_watch(queue_dir: Path, conn: sqlite3.Connection, interval: float, once: bool) -> None:
    try:  # pragma: no cover - optional dependency
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except Exception:  # pragma: no cover - optional dependency
        logger.warning("watchdog not available, falling back to polling")
        _run_polling(queue_dir, conn, interval, once)
        return

    class Handler(FileSystemEventHandler):  # pragma: no cover - depends on filesystem events
        def on_created(self, event):
            if not event.is_directory and event.src_path.endswith(".jsonl"):
                _sync_once(queue_dir, conn)

        def on_modified(self, event):
            if not event.is_directory and event.src_path.endswith(".jsonl"):
                _sync_once(queue_dir, conn)

    observer = Observer()
    observer.schedule(Handler(), str(queue_dir))
    observer.start()
    try:
        _sync_once(queue_dir, conn)
        if once:
            return
        while True:
            time.sleep(interval)
    finally:  # pragma: no cover - cleanup
        observer.stop()
        observer.join()


def main() -> None:  # pragma: no cover - CLI entry point
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", default=str(DEFAULT_QUEUE_DIR))
    parser.add_argument("--db-path", default="menace.db")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true", help="Process queues once and exit")
    parser.add_argument("--watch", action="store_true", help="Watch for filesystem events")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    conn = sqlite3.connect(args.db_path)
    queue_dir = Path(args.queue_dir)
    try:
        if args.watch:
            _run_watch(queue_dir, conn, args.interval, args.once)
        else:
            _run_polling(queue_dir, conn, args.interval, args.once)
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    main()

