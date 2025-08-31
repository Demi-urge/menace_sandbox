from __future__ import annotations

"""Command line interface to retrain prompt formatting preferences."""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from prompt_memory_trainer import PromptMemoryTrainer

DEFAULT_OUTPUT = Path(
    os.getenv(
        "PROMPT_MEMORY_WEIGHTS_PATH",
        Path(__file__).resolve().parent / "config" / "prompt_memory_weights.json",
    )
)


# ---------------------------------------------------------------------------

def cli(argv: Iterable[str] | None = None) -> int:
    """Entry point for command line execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="File to store computed style weights",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    trainer = PromptMemoryTrainer()
    trainer.train()
    trainer.save_weights(args.output)
    print(f"Saved weights to {args.output}")
    return 0


# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> None:  # pragma: no cover - CLI glue
    sys.exit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
