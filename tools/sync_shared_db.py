#!/usr/bin/env python3
"""Synchronise queued writes with the shared database.

This utility scans ``*_queue.jsonl`` files produced by :mod:`db_write_buffer`.
Each line represents a record destined for a table in the shared database.
Processed lines are removed from the queue file.  Failed lines remain and are
logged for later inspection.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

from db_dedup import compute_content_hash
from db_router import DBRouter
from lock_utils import _ContextFileLock

logger = logging.getLogger(__name__)


def _iter_queue_files(queue_dir: Path) -> Iterable[Path]:
    """Yield queue files under *queue_dir*."""

    yield from sorted(queue_dir.glob("*_queue.jsonl"))


def _insert_record(router: DBRouter, table: str, values: dict, hash_fields: list[str]) -> None:
    """Insert ``values`` into ``table`` if unique."""

    conn = router.shared_conn
    columns = ", ".join(values.keys())
    placeholders = ", ".join("?" for _ in values)

    if hash_fields:
        payload = {k: values[k] for k in hash_fields}
        values = dict(values)
        values["content_hash"] = compute_content_hash(payload)
        columns = ", ".join(values.keys())
        placeholders = ", ".join("?" for _ in values)

    conn.execute(
        f"INSERT OR IGNORE INTO {table} ({columns}) VALUES ({placeholders})",
        tuple(values.values()),
    )


def _process_file(path: Path, router: DBRouter, db_lock: _ContextFileLock) -> None:
    """Process queued records in ``path``."""

    file_lock = _ContextFileLock(str(path) + ".lock")
    with file_lock.acquire():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return

        remaining: list[str] = []
        for raw in lines:
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
                table = payload["table"]
                values = payload["values"]
                hash_fields = payload.get("hash_fields", [])
                with db_lock.acquire():
                    conn = router.shared_conn
                    conn.execute("BEGIN")
                    try:
                        _insert_record(router, table, values, hash_fields)
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
            except Exception:  # pragma: no cover - logging
                logger.exception("failed to process record", extra={"file": str(path)})
                remaining.append(raw)

        if remaining:
            with path.open("w", encoding="utf-8") as fh:
                for line in remaining:
                    fh.write(line.rstrip("\n") + "\n")
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def sync_once(queue_dir: Path, router: DBRouter, db_lock: _ContextFileLock) -> None:
    """Process all queue files once."""

    if not queue_dir.exists():
        return

    for file in _iter_queue_files(queue_dir):
        _process_file(file, router, db_lock)


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", default=os.getenv("MENACE_QUEUE_DIR", "./queue"))
    parser.add_argument("--shared-db", default="./shared/global.db")
    parser.add_argument("--menace-id", default=os.getenv("MENACE_ID", ""))
    parser.add_argument(
        "--interval", type=float, default=float(os.getenv("SYNC_DB_INTERVAL", "30"))
    )
    parser.add_argument("--once", action="store_true", help="Process queues once and exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    queue_dir = Path(args.queue_dir)
    router = DBRouter(args.menace_id, "./_local.db", args.shared_db)
    db_lock = _ContextFileLock(args.shared_db + ".lock")

    while True:
        sync_once(queue_dir, router, db_lock)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":  # pragma: no cover
    main()
