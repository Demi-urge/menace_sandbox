"""Backward compatible CLI wrapper around :mod:`vector_service.stack_ingest`."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .stack_ingest import build_arg_parser, main as _main


def main(argv: Sequence[str] | None = None) -> int:
    """Delegate to :func:`vector_service.stack_ingest.main` for CLI execution."""

    # Reuse the parser from the new module but maintain legacy flag aliases.
    parser = build_arg_parser()
    parser.add_argument("--db", dest="cache", help="Legacy alias for --cache", default=argparse.SUPPRESS)

    # Parse arguments using the extended parser, then forward to the new entry point.
    args = parser.parse_args(argv)

    # The new entry point expects an ``argparse.Namespace``; reuse its logic.
    return _main(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

