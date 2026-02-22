from __future__ import annotations

"""Utility to purge stale queue log files.

Removes ``*_queue.jsonl.tmp`` and ``*_queue.failed.jsonl`` files older than a
specified retention period.
"""

from argparse import ArgumentParser
from pathlib import Path
import logging
import time
from typing import Iterable

from db_write_queue import DEFAULT_QUEUE_DIR


def _stale_files(queue_dir: Path) -> Iterable[Path]:
    patterns = ("*_queue.jsonl.tmp", "*_queue.failed.jsonl")
    for pattern in patterns:
        yield from queue_dir.glob(pattern)


def cleanup(queue_dir: Path, days: int) -> None:
    """Remove stale temporary and failed queue files."""
    cutoff = time.time() - days * 86400
    for path in _stale_files(queue_dir):
        if path.stat().st_mtime < cutoff:
            try:
                path.unlink()
                logging.info("removed %s", path)
            except FileNotFoundError:  # pragma: no cover - race condition
                pass


def main() -> None:  # pragma: no cover - CLI wrapper
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", default=str(DEFAULT_QUEUE_DIR))
    parser.add_argument("--days", type=int, default=7, help="Retention period in days")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    cleanup(Path(args.queue_dir), args.days)


if __name__ == "__main__":  # pragma: no cover
    main()
