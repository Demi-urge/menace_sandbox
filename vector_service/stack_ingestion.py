from __future__ import annotations

"""Streaming ingestion pipeline for The Stack dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import uuid

try:  # pragma: no cover - optional dependency
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - dataset ingestion optional in tests
    load_dataset = None  # type: ignore

from config import StackDatasetConfig, get_config

from .vectorizer import SharedVectorService

LOGGER = logging.getLogger(__name__)

_HF_ENV_KEYS = (
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


class SQLiteVectorStore:
    """Minimal SQLite-based vector store dedicated to Stack embeddings."""

    def __init__(self, path: str | Path, namespace: str = "stack") -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.table = f"{namespace}_embeddings"
        self.progress_table = f"{namespace}_progress"
        self.conn = sqlite3.connect(self.path)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                origin_db TEXT,
                vector TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.progress_table} (
                file_id TEXT PRIMARY KEY,
                processed_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    # VectorStore protocol -------------------------------------------------
    def add(
        self,
        kind: str,
        record_id: str,
        vector: Sequence[float],
        *,
        origin_db: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if kind != self.namespace:
            raise ValueError(f"unexpected kind '{kind}', expected '{self.namespace}'")
        payload = json.dumps(list(vector))
        meta = json.dumps(dict(metadata or {}))
        cur = self.conn.cursor()
        cur.execute(
            f"""
            INSERT OR REPLACE INTO {self.table} (id, kind, origin_db, vector, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (record_id, kind, origin_db, payload, meta),
        )
        self.conn.commit()

    def query(self, vector: Sequence[float], top_k: int = 5) -> list[tuple[str, float]]:  # pragma: no cover - unused
        raise NotImplementedError("Querying Stack embeddings is not supported by the ingestion pipeline")

    def load(self) -> None:  # pragma: no cover - no-op for SQLite store
        return

    # Progress helpers -----------------------------------------------------
    def has_processed(self, file_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute(f"SELECT 1 FROM {self.progress_table} WHERE file_id = ?", (file_id,))
        return cur.fetchone() is not None

    def mark_processed(self, file_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            f"INSERT OR REPLACE INTO {self.progress_table} (file_id, processed_at) VALUES (?, ?)",
            (file_id, time.time()),
        )
        self.conn.commit()

    def reset(self) -> None:
        cur = self.conn.cursor()
        cur.execute(f"DELETE FROM {self.table}")
        cur.execute(f"DELETE FROM {self.progress_table}")
        self.conn.commit()


@dataclass
class StackIngestionService:
    """Stream and embed documents from The Stack dataset."""

    languages: Sequence[str] | None = None
    chunk_size: int = 2048
    dataset_name: str = "bigcode/the-stack-dedup"
    split: str = "train"
    namespace: str = "stack"
    db_path: str | Path = "stack_embeddings.db"
    use_auth_token: str | None = None
    max_lines_per_document: int = 0

    def __post_init__(self) -> None:
        self.languages = tuple(lang.lower() for lang in (self.languages or ()))
        if self.max_lines_per_document < 0:
            raise ValueError("max_lines_per_document must be non-negative")
        self.store = SQLiteVectorStore(self.db_path, namespace=self.namespace)
        self.vector_service = SharedVectorService(vector_store=self.store)
        LOGGER.debug("StackIngestionService initialised", extra={"languages": self.languages})

    # Public API -----------------------------------------------------------
    def run(self, *, resume: bool = False, limit: int | None = None) -> None:
        if not resume:
            LOGGER.info("Resetting existing Stack embeddings at %s", self.db_path)
            self.store.reset()
        dataset_iter = self._stream_dataset()
        processed = 0
        for sample in dataset_iter:
            if limit is not None and processed >= limit:
                break
            try:
                if self._handle_sample(sample):
                    processed += 1
            except KeyboardInterrupt:  # pragma: no cover - manual interruption
                raise
            except Exception as exc:
                LOGGER.exception("Failed to process sample: %s", exc)
        LOGGER.info("Stack ingestion completed: processed %d files", processed)

    # Internal helpers -----------------------------------------------------
    def _stream_dataset(self) -> Iterator[MutableMapping[str, object]]:
        if load_dataset is None:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "datasets package not available - install `datasets` to stream The Stack"
            )
        LOGGER.info(
            "Loading dataset %s split %s (streaming)",
            self.dataset_name,
            self.split,
        )
        kwargs: Dict[str, object] = {"streaming": True, "split": self.split}
        if self.use_auth_token:
            kwargs["use_auth_token"] = self.use_auth_token
        return iter(load_dataset(self.dataset_name, **kwargs))

    def _handle_sample(self, sample: Mapping[str, object]) -> bool:
        language = str(sample.get("language", "")).lower()
        if self.languages and language not in self.languages:
            return False
        repo = str(sample.get("repo_name", ""))
        path = str(sample.get("path", ""))
        file_id = self._file_identifier(repo, path, sample)
        if self.store.has_processed(file_id):
            LOGGER.debug("Skipping previously processed file: %s", file_id)
            return False
        content = sample.get("content")
        if not isinstance(content, str) or not content.strip():
            LOGGER.debug("Skipping empty content for file: %s", file_id)
            self.store.mark_processed(file_id)
            return True
        if self.max_lines_per_document:
            lines = content.splitlines()
            if len(lines) > self.max_lines_per_document:
                LOGGER.debug(
                    "Truncating %s to %d lines before embedding",
                    file_id,
                    self.max_lines_per_document,
                )
                content = "\n".join(lines[: self.max_lines_per_document])
        chunk_count = 0
        for chunk_index, start, end, chunk in self._chunk_content(content):
            record_id = self._chunk_identifier(file_id, chunk_index)
            metadata = {
                "repo": repo,
                "path": path,
                "language": language,
                "chunk_index": chunk_index,
                "start": start,
                "end": end,
            }
            self.vector_service.vectorise_and_store(
                self.namespace,
                record_id,
                {"text": chunk},
                origin_db=self.namespace,
                metadata=metadata,
            )
            chunk_count += 1
        self.store.mark_processed(file_id)
        LOGGER.info(
            "Embedded %d chunks for %s (%s)",
            chunk_count,
            path or file_id,
            language,
        )
        return True

    def _chunk_content(self, content: str) -> Iterable[tuple[int, int, int, str]]:
        size = max(1, int(self.chunk_size))
        length = len(content)
        chunk_index = 0
        for start in range(0, length, size):
            end = min(start + size, length)
            chunk = content[start:end]
            yield chunk_index, start, end, chunk
            chunk_index += 1

    @staticmethod
    def _file_identifier(repo: str, path: str, sample: Mapping[str, object]) -> str:
        base = f"{repo}:{path}" if repo or path else str(sample.get("id", ""))
        if not base:
            base = str(uuid.uuid4())
        return base

    @staticmethod
    def _chunk_identifier(file_id: str, chunk_index: int) -> str:
        return f"{file_id}::chunk-{chunk_index}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream embeddings from The Stack dataset")
    parser.add_argument(
        "--languages",
        nargs="*",
        default=None,
        help="Languages to include (defaults to stack_dataset.allowed_languages)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Maximum characters per chunk (defaults to stack_dataset.chunk_size)",
    )
    parser.add_argument("--split", default="train", help="Dataset split to stream")
    parser.add_argument("--dataset", default="bigcode/the-stack-dedup", help="Dataset identifier")
    parser.add_argument("--db", default="stack_embeddings.db", help="SQLite database path")
    parser.add_argument("--namespace", default="stack", help="Vector store namespace")
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

    auth_token = _resolve_hf_token()
    if not auth_token:
        LOGGER.warning(
            "No Hugging Face credentials found (set HUGGINGFACE_TOKEN); skipping Stack ingestion"
        )
        return 0

    languages = args.languages
    if languages is None:
        languages = sorted(stack_cfg.allowed_languages)

    chunk_size = args.chunk_size if args.chunk_size is not None else stack_cfg.chunk_size

    service = StackIngestionService(
        languages=languages,
        chunk_size=chunk_size,
        dataset_name=args.dataset,
        split=args.split,
        namespace=args.namespace,
        db_path=args.db,
        use_auth_token=auth_token,
        max_lines_per_document=stack_cfg.max_lines_per_document,
    )
    service.run(resume=args.resume, limit=args.limit)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
