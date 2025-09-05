from __future__ import annotations

"""Command line interface to retrain prompt formatting preferences."""

import argparse
import json
import os
import sys
from typing import Iterable

from dynamic_path_router import resolve_path
from prompt_memory_trainer import PromptMemoryTrainer

_BASE = resolve_path(".")
DEFAULT_STATE_PATH = _BASE / os.getenv(
    "PROMPT_MEMORY_WEIGHTS_PATH", "config/prompt_memory_weights.json"
)
DEFAULT_DB_PATH = _BASE / os.getenv("PROMPT_STYLE_DB_PATH", "prompt_styles.db")


# ---------------------------------------------------------------------------

def cli(argv: Iterable[str] | None = None) -> int:
    """Entry point for command line execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-path",
        type=str,
        default=str(DEFAULT_STATE_PATH),
        help="File storing persistent style weights",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="SQLite database for prompt style statistics",
    )
    parser.add_argument(
        "--records",
        type=str,
        help="JSON file containing new records to append before retraining",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    trainer = PromptMemoryTrainer(
        state_path=resolve_path(args.state_path),
        db_path=resolve_path(args.db_path),
    )
    if args.records:
        records_path = resolve_path(args.records)
        with records_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            records = [data]
        else:
            records = list(data)
        trainer.append_records(records)
    else:
        trainer.train()
    print(f"Saved weights to {args.state_path} and stats to {args.db_path}")
    return 0


# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI glue
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
