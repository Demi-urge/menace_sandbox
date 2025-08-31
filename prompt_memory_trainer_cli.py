from __future__ import annotations

"""Command line interface to retrain prompt formatting preferences."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

from prompt_memory_trainer import PromptMemoryTrainer

DEFAULT_STATE_PATH = Path(
    os.getenv(
        "PROMPT_MEMORY_WEIGHTS_PATH",
        Path(__file__).resolve().parent / "config" / "prompt_memory_weights.json",
    )
)
DEFAULT_DB_PATH = Path(
    os.getenv(
        "PROMPT_STYLE_DB_PATH",
        Path(__file__).resolve().parent / "prompt_styles.db",
    )
)


# ---------------------------------------------------------------------------

def cli(argv: Iterable[str] | None = None) -> int:
    """Entry point for command line execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state-path",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="File storing persistent style weights",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite database for prompt style statistics",
    )
    parser.add_argument(
        "--records",
        type=Path,
        help="JSON file containing new records to append before retraining",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    trainer = PromptMemoryTrainer(state_path=args.state_path, db_path=args.db_path)
    if args.records:
        with args.records.open("r", encoding="utf-8") as fh:
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
