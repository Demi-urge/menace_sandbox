from __future__ import annotations

"""Synchronise queued database writes with a shared database.

This utility processes JSONL queue files produced by :mod:`db_write_queue` and
attempts to flush their contents into the target database using
:func:`db_dedup.insert_if_unique`.

The script watches a directory for ``*_queue.jsonl`` files. Each iteration it
locks the file, processes queued entries and removes those that were committed
successfully. Failed records are left in place for a future retry.

Usage
-----
python sync_shared_db.py --db-url sqlite:///menace.db --queue-dir sandbox_data/queues
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import logging
import os
import signal
from threading import Event
from typing import Any, Dict

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.engine import Engine

from db_dedup import insert_if_unique
from db_write_queue import DEFAULT_QUEUE_DIR
from fcntl_compat import flock, LOCK_EX, LOCK_UN

logger = logging.getLogger(__name__)


def _load_table(engine: Engine, name: str, cache: Dict[str, Table]) -> Table:
    """Load table metadata, caching results."""
    if name not in cache:
        meta = MetaData()
        cache[name] = Table(name, meta, autoload_with=engine)
    return cache[name]


def process_queue_file(path: Path, *, engine: Engine) -> None:
    """Process queued operations from *path*.

    Lines that fail to process are written back to the file for retry.
    """
    keep: list[str] = []
    with path.open("r+", encoding="utf-8") as fh:
        flock(fh.fileno(), LOCK_EX)
        lines = fh.readlines()
        fh.seek(0)
        fh.truncate(0)
        for line in lines:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                logger.exception("Malformed queue record in %s", path.name)
                keep.append(line)
                continue

            if record.get("op") != "insert":
                logger.warning("Unsupported operation %s", record.get("op"))
                keep.append(line)
                continue

            table = record.get("table")
            data = record.get("data", {})
            menace_id = record.get("source_menace_id", "")
            hash_fields = list(data.keys())

            try:
                tbl = _load_table(engine, table, process_queue_file._tbl_cache)
                insert_if_unique(tbl, data, hash_fields, menace_id, logger=logger, engine=engine)
            except Exception:
                logger.exception("Failed to insert queued record into %s", table)
                keep.append(line)

        for line in keep:
            fh.write(line)
            fh.write("\n")
        flock(fh.fileno(), LOCK_UN)


process_queue_file._tbl_cache: Dict[str, Table] = {}


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL", "sqlite:///menace.db"))
    parser.add_argument("--queue-dir", default=str(DEFAULT_QUEUE_DIR))
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    engine = create_engine(args.db_url)

    queue_dir = Path(args.queue_dir)
    stop = Event()

    def _handle_signal(*_: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    mtimes: dict[Path, float] = {}
    while not stop.is_set():
        processed = False
        for file in queue_dir.glob("*_queue.jsonl"):
            mtime = file.stat().st_mtime
            if mtimes.get(file) == mtime:
                continue
            process_queue_file(file, engine=engine)
            mtimes[file] = file.stat().st_mtime
            processed = True
        if not processed:
            stop.wait(args.interval)

    engine.dispose()


if __name__ == "__main__":  # pragma: no cover
    main()
