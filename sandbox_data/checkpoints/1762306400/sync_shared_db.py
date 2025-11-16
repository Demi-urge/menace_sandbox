#!/usr/bin/env python3
"""Synchronise queued database writes with the shared database.

This utility scans a directory for ``*.jsonl`` queue files.  Each line in a
queue file is expected to be a JSON mapping similar to::

    {"table": "<table>", "data": {...}, "hash": "<hash>",
     "hash_fields": ["<field>", ...], "menace_id": "<id>"}

Older entries may use ``record`` instead of ``data`` or omit the ``hash`` and
``hash_fields`` keys.  When processing an entry the pre-computed ``hash`` is
first checked against previously processed hashes recorded in ``processed.log``
and the destination table.  Successful inserts append the hash to this log so
that re-running the daemon after a crash skips already committed rows.  Failed
inserts are moved to ``queue.failed.jsonl`` for later inspection.

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
from typing import Iterable, Callable

from db_dedup import compute_content_hash, insert_if_unique
from db_write_queue import remove_processed_lines
from env_config import SHARED_QUEUE_DIR, SYNC_INTERVAL
from fcntl_compat import LOCK_EX, LOCK_UN, flock


logger = logging.getLogger(__name__)

_LogDBAccess = Callable[..., None]
_log_db_access_fn: _LogDBAccess | None = None


def _ensure_log_db_access() -> _LogDBAccess:
    """Return :func:`audit.log_db_access`, importing it on demand."""

    global _log_db_access_fn

    if _log_db_access_fn is None:
        from audit import log_db_access as _impl

        _log_db_access_fn = _impl
    return _log_db_access_fn


def _log_db_access(
    action: str, table: str, rows: int, menace_id: str, **kwargs: object
) -> None:
    """Record database access without importing :mod:`audit` during module load."""

    _ensure_log_db_access()(action, table, rows, menace_id, **kwargs)


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


def replay_failed(queue_dir: Path) -> None:
    """Requeue entries from ``queue.failed.jsonl`` if it exists."""

    failed_path = queue_dir / "queue.failed.jsonl"
    if not failed_path.exists():
        return

    try:
        lines = failed_path.read_text(encoding="utf-8").splitlines(True)
    except Exception:  # pragma: no cover - unable to read file
        return

    # Preserve failed log for manual inspection
    bak_path = failed_path.with_suffix(failed_path.suffix + ".bak")
    failed_path.rename(bak_path)

    for raw in lines:
        try:
            entry = json.loads(raw)
        except Exception:  # pragma: no cover - skip unparsable
            continue
        record = entry.get("record")
        if isinstance(record, str):
            try:
                record = json.loads(record)
            except Exception:
                continue
        if not isinstance(record, dict):
            continue
        menace_id = record.get("menace_id") or record.get("source_menace_id", "")
        target = queue_dir / (f"{menace_id}.jsonl" if menace_id else "replay.jsonl")
        _append_lines(target, [json.dumps(record, sort_keys=True) + "\n"])


def process_queue_file(
    path: Path,
    *,
    conn: sqlite3.Connection,
    preserve_empty: bool = True,
) -> Stats:
    """Process all records from ``path`` returning :class:`Stats`.

    Each line in ``path`` is handled independently.  Successful inserts and
    detected duplicates are removed from the queue file.  Lines that raise an
    exception are moved to ``queue.failed.jsonl`` along with error details.  A
    ``processed.log`` file records hashes of successfully committed rows so that
    rerunning after a crash avoids duplicating work.
    """

    _ensure_log_db_access()

    failed_path = path.parent / "queue.failed.jsonl"
    processed_log = path.parent / "processed.log"
    stats = Stats()

    processed_hashes: set[str] = set()
    if processed_log.exists():
        try:
            with processed_log.open("r", encoding="utf-8") as pf:
                processed_hashes = {line.strip() for line in pf if line.strip()}
        except Exception:  # pragma: no cover - unreadable log
            processed_hashes = set()

    processed_lines = 0

    with path.open("r", encoding="utf-8") as fh:
        flock(fh.fileno(), LOCK_EX)
        lines = fh.readlines()
        flock(fh.fileno(), LOCK_UN)

    for raw in lines:
        line = raw.strip()
        if not line:
            processed_lines += 1
            continue

        try:
            payload = json.loads(line)
        except Exception as exc:  # pragma: no cover - logged then skipped
            stats.failures += 1
            _append_lines(
                failed_path,
                [json.dumps({"record": raw.rstrip("\n"), "error": str(exc)}) + "\n"],
            )
            processed_lines += 1
            continue

        table = payload.get("table")
        record = payload.get("record") or payload.get("data", {})
        menace_id = payload.get("menace_id") or payload.get("source_menace_id", "")
        hash_fields = payload.get("hash_fields") or list(record.keys())
        content_hash = (
            payload.get("hash")
            or payload.get("content_hash")
            or compute_content_hash({k: record[k] for k in hash_fields})
        )

        if not table or not isinstance(record, dict):
            stats.failures += 1
            _append_lines(
                failed_path,
                [
                    json.dumps(
                        {"record": payload, "error": "missing table or record"},
                        sort_keys=True,
                    )
                    + "\n",
                ],
            )
            processed_lines += 1
            continue

        if content_hash in processed_hashes:
            stats.duplicates += 1
            _log_db_access("write", table, 0, menace_id)
            processed_lines += 1
            continue

        try:
            existing = conn.execute(
                f"SELECT id FROM {table} WHERE content_hash=?", (content_hash,)
            ).fetchone()
            _log_db_access("read", table, 1 if existing else 0, menace_id)
            if existing:
                stats.duplicates += 1
                logger.info(
                    "duplicate",
                    extra={"table": table, "menace_id": menace_id, "id": existing[0]},
                )
                _log_db_access("write", table, 0, menace_id)
                _append_lines(processed_log, [content_hash + "\n"])
                processed_hashes.add(content_hash)
                processed_lines += 1
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
            _log_db_access("write", table, 1, menace_id)
            stats.processed += 1
            _append_lines(processed_log, [content_hash + "\n"])
            processed_hashes.add(content_hash)
            logger.info(
                "inserted",
                extra={"table": table, "menace_id": menace_id, "hash": content_hash},
            )
        except Exception as exc:  # pragma: no cover - logged then moved to failed
            conn.rollback()
            stats.failures += 1
            logger.error(
                "failed", extra={"table": table, "menace_id": menace_id, "error": str(exc)}
            )
            _append_lines(
                failed_path,
                [json.dumps({"record": payload, "error": str(exc)}, sort_keys=True) + "\n"],
            )

        processed_lines += 1

    remove_processed_lines(path, processed_lines)
    if preserve_empty and not path.exists():
        path.touch()
    return stats


def _sync_once(queue_dir: Path, conn: sqlite3.Connection, preserve_empty: bool = False) -> Stats:
    """Process all queue files under ``queue_dir`` returning cumulative stats."""

    stats = Stats()
    if not queue_dir.exists():
        return stats

    failed_path = queue_dir / "queue.failed.jsonl"

    for file in sorted(queue_dir.glob("*.jsonl")):
        if file.name == "queue.failed.jsonl":
            continue
        try:
            file_stats = process_queue_file(file, conn=conn, preserve_empty=preserve_empty)
        except Exception as exc:  # pragma: no cover - logged then recorded
            _append_lines(
                failed_path,
                [
                    json.dumps(
                        {"record": {"file": str(file)}, "error": str(exc)}, sort_keys=True
                    )
                    + "\n",
                ],
            )
            stats.failures += 1
            logger.error(
                "file_failed", extra={"file": str(file), "error": str(exc)}
            )
            continue
        stats.processed += file_stats.processed
        stats.duplicates += file_stats.duplicates
        stats.failures += file_stats.failures
    return stats


def _run_polling(queue_dir: Path, conn: sqlite3.Connection, interval: float, once: bool) -> None:
    while True:
        stats = _sync_once(queue_dir, conn, preserve_empty=once)
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
    parser.add_argument("--queue-dir", default=SHARED_QUEUE_DIR)
    parser.add_argument("--db-path", default="menace.db")
    parser.add_argument("--interval", type=float, default=SYNC_INTERVAL)
    parser.add_argument("--once", action="store_true", help="Process queues once and exit")
    parser.add_argument(
        "--replay-failed", action="store_true", help="Requeue entries from queue.failed.jsonl"
    )
    parser.add_argument("--watch", action="store_true", help="Watch for filesystem events")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    conn = sqlite3.connect(args.db_path)  # noqa: SQL001
    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.replay_failed:
            replay_failed(queue_dir)
        if args.watch:
            _run_watch(queue_dir, conn, args.interval, args.once)
        else:
            _run_polling(queue_dir, conn, args.interval, args.once)
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    main()
