"""CLI for streaming embeddings from The Stack dataset."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Sequence

from config import StackDatasetConfig, get_config
from vector_service.stack_ingestor import StackIngestor

LOGGER = logging.getLogger(__name__)

_HF_ENV_KEYS = (
    "STACK_HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_API_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_TOKEN",
)


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _resolve_hf_token() -> str | None:
    for key in _HF_ENV_KEYS:
        token = os.environ.get(key)
        if token:
            if key != "HUGGINGFACE_TOKEN":
                os.environ.setdefault("HUGGINGFACE_TOKEN", token)
            return token
    return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream embeddings from The Stack dataset")
    parser.add_argument("--dataset", default="bigcode/the-stack-dedup", help="Dataset identifier")
    parser.add_argument("--split", default="train", help="Dataset split to stream")
    parser.add_argument("--languages", nargs="*", default=None, help="Languages to include")
    parser.add_argument("--chunk-lines", type=int, default=None, help="Maximum lines per embedded chunk")
    parser.add_argument("--max-lines", type=int, default=None, help="Maximum lines retained per file")
    parser.add_argument("--max-bytes", type=int, default=None, help="Maximum bytes retained per file")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--db", default="stack_embeddings.db", help="SQLite database path for metadata")
    parser.add_argument("--namespace", default="stack", help="Namespace for vector entries")
    parser.add_argument(
        "--index-path",
        default=None,
        help="Path for the dedicated Stack vector index (defaults to <db>.index)",
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Vector index backend (faiss, annoy, chroma, qdrant - defaults to annoy)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on processed files")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress checkpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not _truthy(os.environ.get("STACK_STREAMING", "1")):
        LOGGER.warning("STACK_STREAMING disabled - exiting without processing")
        return 0

    cfg = get_config()
    stack_cfg: StackDatasetConfig = getattr(cfg, "stack_dataset", StackDatasetConfig())
    if not stack_cfg.enabled:
        LOGGER.info("Stack dataset ingestion disabled via configuration")
        return 0

    languages = args.languages
    if languages is None:
        languages = sorted(stack_cfg.allowed_languages)

    chunk_lines = args.chunk_lines if args.chunk_lines is not None else stack_cfg.chunk_size
    max_lines = args.max_lines if args.max_lines is not None else stack_cfg.max_lines_per_document

    auth_token = _resolve_hf_token()
    if not auth_token:
        LOGGER.info(
            "No Hugging Face credentials found (set STACK_HF_TOKEN if required); proceeding unauthenticated"
        )

    ingestor = StackIngestor(
        dataset_name=args.dataset,
        split=args.split,
        languages=languages,
        max_lines=max_lines,
        max_bytes=args.max_bytes,
        chunk_lines=chunk_lines,
        batch_size=args.batch_size,
        namespace=args.namespace,
        metadata_path=args.db,
        index_path=args.index_path,
        vector_backend=args.backend,
        use_auth_token=auth_token,
    )
    ingestor.ingest(resume=args.resume, limit=args.limit)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

