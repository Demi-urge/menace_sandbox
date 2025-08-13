#!/usr/bin/env python3
"""Backfill vector embeddings for all databases."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root on sys.path when executed directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_database import BotDB
from task_handoff_bot import WorkflowDB
from chatgpt_enhancement_bot import EnhancementDB
from error_bot import ErrorDB
from research_aggregator_bot import InfoDB


def main(batch_size: int = 100) -> None:
    BotDB().backfill_embeddings(batch_size=batch_size)
    WorkflowDB().backfill_embeddings(batch_size=batch_size)
    EnhancementDB().backfill_embeddings(batch_size=batch_size)
    ErrorDB().backfill_embeddings(batch_size=batch_size)
    InfoDB().backfill_embeddings(batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for all databases"
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    args = parser.parse_args()
    main(batch_size=args.batch_size)
