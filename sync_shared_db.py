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


MAX_RETRIES = 3


def _log(event: str, **data: Any) -> None:
    """Emit a structured JSON log entry."""

    logger.info("%s", json.dumps({"event": event, **data}))


def process_queue_file(path: Path, *, engine: Engine) -> None:
    """Process queued operations from *path* with temporary file safety.

    Lines that fail to process are written back for retry.  Records exceeding
    ``MAX_RETRIES`` are moved to a ``.failed`` file for manual inspection.
    """

    temp_path = path.with_suffix(path.suffix + ".tmp")
    failed_path = path.with_suffix(".failed.jsonl")
    keep: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        flock(fh.fileno(), LOCK_EX)
        lines = fh.readlines()
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
                keep.append(json.dumps(record, sort_keys=True))
                continue

            table = record.get("table")
            data = record.get("data", {})
            menace_id = record.get("source_menace_id", "")
            hash_fields = list(data.keys())

            try:
                tbl = _load_table(engine, table, process_queue_file._tbl_cache)
                insert_if_unique(
                    tbl, data, hash_fields, menace_id, logger=logger, engine=engine
                )
                _log("commit", table=table, content_hash=record.get("content_hash"))
            except Exception as exc:
                fail_count = int(record.get("fail_count", 0)) + 1
                record["fail_count"] = fail_count
                if fail_count >= MAX_RETRIES:
                    with failed_path.open("a", encoding="utf-8") as fail_fh:
                        fail_fh.write(json.dumps(record, sort_keys=True))
                        fail_fh.write("\n")
                    _log(
                        "rollback",
                        table=table,
                        content_hash=record.get("content_hash"),
                        error=str(exc),
                        action="moved_to_failed",
                        fail_count=fail_count,
                    )
                else:
                    keep.append(json.dumps(record, sort_keys=True))
                    _log(
                        "rollback",
                        table=table,
                        content_hash=record.get("content_hash"),
                        error=str(exc),
                        fail_count=fail_count,
                    )

        with temp_path.open("w", encoding="utf-8") as tmp_fh:
            for line in keep:
                tmp_fh.write(line)
                tmp_fh.write("\n")
        os.replace(temp_path, path)
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
