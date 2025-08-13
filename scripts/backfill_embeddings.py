#!/usr/bin/env python3
"""Backfill vector embeddings for selected databases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

# Ensure repository root on sys.path when executed directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_database import BotDB
from task_handoff_bot import WorkflowDB
from chatgpt_enhancement_bot import EnhancementDB
from error_bot import ErrorDB
from research_aggregator_bot import InfoDB


DB_CLASSES = {
    "bot": BotDB,
    "workflow": WorkflowDB,
    "enhancement": EnhancementDB,
    "error": ErrorDB,
    "info": InfoDB,
}


def main(databases: Iterable[str], backend: str, batch_size: int = 100) -> None:
    """Backfill embeddings for ``databases`` using ``backend``."""

    for name in databases:
        db_cls = DB_CLASSES[name]
        db = db_cls(vector_backend=backend)
        db.backfill_embeddings(batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for selected databases",
    )
    parser.add_argument(
        "--db",
        choices=DB_CLASSES.keys(),
        nargs="*",
        default=list(DB_CLASSES.keys()),
        help="Databases to process (default: all)",
    )
    parser.add_argument(
        "--backend",
        choices=["annoy", "faiss"],
        default="annoy",
        help="Vector backend to use (default: annoy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )
    args = parser.parse_args()
    main(args.db, args.backend, batch_size=args.batch_size)

