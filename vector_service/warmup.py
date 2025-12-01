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
    _flag(parser, "probe-model", default=False, help="Log whether the bundled embedding model is present without downloading")
    _flag(parser, "hydrate-handlers", default=False, help="Instantiate SharedVectorService to prime vectorizers")
    _flag(parser, "run-vectorise", default=False, help="Execute a single vectorise call after handler hydration")
    _flag(parser, "start-scheduler", default=False, help="Start the embedding scheduler using env configuration")
    _flag(parser, "light", default=False, help="Validate scheduler/model presence without hydrating handlers")
    _flag(parser, "all", default=False, help="Enable every warmup action")
    args = parser.parse_args(argv)

    if args.all:
        download = True
        probe = False
        hydrate = True
        vectorise = True
        scheduler = True
    elif args.light:
        download = False
        probe = True
        hydrate = False
        vectorise = False
        scheduler = True
    else:
        download = args.download_model
        probe = args.probe_model
        hydrate = args.hydrate_handlers
        vectorise = args.run_vectorise
        scheduler = args.start_scheduler

    logger = logging.getLogger(__name__)
    warmup_vector_service(
        download_model=download,
        probe_model=probe,
        hydrate_handlers=hydrate,
        start_scheduler=scheduler,
        run_vectorise=vectorise,
        logger=logger,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(cli(argv))


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
