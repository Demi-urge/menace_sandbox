"""CLI for running embedding backfills across databases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repository root on ``sys.path`` when executed directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from vector_service import EmbeddingBackfill, VectorServiceError


def main(*, session_id: str, backend: str, batch_size: int, dbs: list[str] | None) -> None:
    try:
        EmbeddingBackfill(batch_size=batch_size, backend=backend).run(
            session_id=session_id, dbs=dbs
        )
    except VectorServiceError:
        raise


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill embeddings for all known databases",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Identifier used for metrics aggregation",
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
    parser.add_argument(
        "--db",
        action="append",
        dest="dbs",
        help="Restrict to a specific database class (can be used multiple times)",
    )
    args = parser.parse_args()
    main(
        session_id=args.session_id,
        backend=args.backend,
        batch_size=args.batch_size,
        dbs=args.dbs,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    cli()

