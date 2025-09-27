from __future__ import annotations

import argparse
import logging
from typing import Optional

from vector_service.stack_ingestion import StackDatasetStreamer


logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    """Drive Stack ingestion from the command line."""

    _configure_logging()

    parser = argparse.ArgumentParser(description="Ingest BigCode Stack embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Maximum chunks to embed")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continue streaming until interrupted",
    )
    args = parser.parse_args(argv)

    streamer = StackDatasetStreamer.from_environment()
    high_water = streamer.metadata_store.last_cursor(streamer.config.split)
    if high_water:
        logger.info("resuming Stack ingestion from cursor %s", high_water)
    else:
        logger.info("starting Stack ingestion from dataset head")

    count = streamer.process(limit=args.limit, continuous=args.continuous)
    logger.info("Stack ingestion embedded %s chunks", count)
    return count


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())

