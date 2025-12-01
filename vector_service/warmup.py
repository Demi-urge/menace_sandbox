from __future__ import annotations

"""CLI for warming vector service caches without triggering full bootstrap."""

import argparse
import logging
from typing import Sequence

from .lazy_bootstrap import warmup_vector_service


def _flag(parser: argparse.ArgumentParser, name: str, *, default: bool = False, help: str) -> None:
    parser.add_argument(
        f"--{name}",
        action="store_true",
        default=default,
        help=help,
    )


def cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _flag(parser, "download-model", default=False, help="Download the bundled embedding model if missing")
    _flag(parser, "hydrate-handlers", default=False, help="Instantiate SharedVectorService to prime vectorizers")
    _flag(parser, "start-scheduler", default=False, help="Start the embedding scheduler using env configuration")
    _flag(parser, "all", default=False, help="Enable every warmup action")
    args = parser.parse_args(argv)

    download = args.all or args.download_model
    hydrate = args.all or args.hydrate_handlers or args.download_model
    scheduler = args.all or args.start_scheduler

    logger = logging.getLogger(__name__)
    warmup_vector_service(
        download_model=download,
        hydrate_handlers=hydrate,
        start_scheduler=scheduler,
        logger=logger,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
