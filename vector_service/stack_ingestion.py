from __future__ import annotations

"""Streaming ingestion pipeline for The Stack dataset."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import argparse
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

from chunking import CodeChunk, split_into_chunks
from code_vectorizer import CodeVectorizer
from config import StackDatasetConfig, get_config
from vector_service.embed_utils import EMBED_DIM
from vector_service.vector_store import VectorStore, create_vector_store

from .vectorizer import SharedVectorService

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


class SQLiteMetadataStore:
    """Track Stack ingestion metadata and progress in SQLite."""

    def __init__(self, path: str | Path, namespace: str = "stack") -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.metadata_table = f"{namespace}_metadata"
        self.progress_table = f"{namespace}_progress"
        self.conn = sqlite3.connect(self.path)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.metadata_table} (
                embedding_id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL,
                repo TEXT,
                path TEXT,
                language TEXT,
                license TEXT,
                chunk_index INTEGER NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                token_count INTEGER,
                updated_at REAL NOT NULL
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

    def upsert_chunk_metadata(
        self,
        *,
        embedding_id: str,
        file_id: str,
        repo: str,
        path: str,
        language: str,
        license: str,
        chunk_index: int,
        start_line: int | None,
        end_line: int | None,
        token_count: int | None,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            f"""
            INSERT OR REPLACE INTO {self.metadata_table} (
                embedding_id,
                file_id,
                repo,
                path,
                language,
                license,
                chunk_index,
                start_line,
                end_line,
                token_count,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embedding_id,
                file_id,
                repo or None,
                path or None,
                language or None,
                license or None,
                chunk_index,
                start_line,
                end_line,
                token_count,
                time.time(),
            ),
        )
        self.conn.commit()

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
        cur.execute(f"DELETE FROM {self.metadata_table}")
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
    index_path: str | Path | None = None
    vector_backend: str | None = None
    use_auth_token: str | None = None
    max_lines_per_document: int = 0
    max_chunk_lines: int | None = None

    def __post_init__(self) -> None:
        self.languages = tuple(lang.lower() for lang in (self.languages or ()))
        if self.max_lines_per_document < 0:
            raise ValueError("max_lines_per_document must be non-negative")
        if self.max_chunk_lines is None and self.max_lines_per_document:
            self.max_chunk_lines = self.max_lines_per_document
        self.metadata_store = SQLiteMetadataStore(self.db_path, namespace=self.namespace)
        self.index_path = Path(self.index_path or Path(self.db_path).with_suffix(".index"))
        self._vector_store: VectorStore | None = None
        self.vector_service: SharedVectorService | None = None
        self.code_vectorizer = CodeVectorizer()
        LOGGER.debug(
            "StackIngestionService initialised",
            extra={
                "languages": self.languages,
                "index_path": str(self.index_path),
                "backend": self.vector_backend or "annoy",
            },
        )

    # Public API -----------------------------------------------------------
    def run(self, *, resume: bool = False, limit: int | None = None) -> None:
        self._initialise_vector_pipeline(reset=not resume)
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
        language = str(sample.get("language") or "").lower()
        if self.languages and language not in self.languages:
            return False
        repo = str(sample.get("repo_name") or "")
        path = str(sample.get("path") or "")
        file_id = self._file_identifier(repo, path, sample)
        if self.metadata_store.has_processed(file_id):
            LOGGER.debug("Skipping previously processed file: %s", file_id)
            return False
        content = sample.get("content")
        if not isinstance(content, str) or not content.strip():
            LOGGER.debug("Skipping empty content for file: %s", file_id)
            self.metadata_store.mark_processed(file_id)
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
        license_info = str(sample.get("license") or "")
        for chunk_index, chunk in self._chunk_content(content):
            record_id = self._chunk_identifier(file_id, chunk_index)
            vector = self._embed_chunk(chunk)
            metadata = {
                "repo": repo,
                "path": path,
                "language": language,
                "license": license_info,
                "chunk_index": chunk_index,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "token_count": chunk.token_count,
            }
            if self._vector_store is None:
                raise RuntimeError("Vector store not initialised")
            self._vector_store.add(
                self.namespace,
                record_id,
                vector,
                origin_db=self.namespace,
                metadata=metadata,
            )
            self.metadata_store.upsert_chunk_metadata(
                embedding_id=record_id,
                file_id=file_id,
                repo=repo,
                path=path,
                language=language,
                license=license_info,
                chunk_index=chunk_index,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                token_count=chunk.token_count,
            )
            chunk_count += 1

        self.metadata_store.mark_processed(file_id)
        LOGGER.info(
            "Embedded %d chunks for %s (%s)",
            chunk_count,
            path or file_id,
            language,
        )
        return True

    def _chunk_content(self, content: str) -> Iterable[tuple[int, CodeChunk]]:
        token_limit = max(1, int(self.chunk_size))
        chunks = split_into_chunks(content, token_limit)
        for idx, chunk in enumerate(chunks):
            text = chunk.text
            if self.max_chunk_lines:
                lines = text.splitlines()
                if len(lines) > self.max_chunk_lines:
                    text = "\n".join(lines[: self.max_chunk_lines])
                    approx_tokens = len(text.split())
                    chunk = CodeChunk(
                        start_line=chunk.start_line,
                        end_line=min(chunk.end_line, chunk.start_line + self.max_chunk_lines - 1),
                        text=text,
                        hash=chunk.hash,
                        token_count=approx_tokens,
                    )
            yield idx, chunk

    def _embed_chunk(self, chunk: CodeChunk) -> Sequence[float]:
        record = {"content": chunk.text}
        if self.vector_service is not None:
            try:
                return self.vector_service.vectorise("code", record)
            except Exception as exc:
                LOGGER.debug("SharedVectorService fallback for chunk failed: %s", exc)
        return self.code_vectorizer.transform(record)

    @staticmethod
    def _file_identifier(repo: str, path: str, sample: Mapping[str, object]) -> str:
        base = f"{repo}:{path}" if repo or path else str(sample.get("id", ""))
        if not base:
            base = str(uuid.uuid4())
        return base

    @staticmethod
    def _chunk_identifier(file_id: str, chunk_index: int) -> str:
        return f"{file_id}::chunk-{chunk_index}"

    def _initialise_vector_pipeline(self, *, reset: bool) -> None:
        if reset:
            LOGGER.info("Resetting existing Stack embeddings at %s", self.db_path)
            self.metadata_store.reset()
            self._reset_vector_index()
        backend = (self.vector_backend or "annoy").lower()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._vector_store = create_vector_store(
            EMBED_DIM,
            self.index_path,
            backend=backend,
            metric="angular",
        )
        self.vector_service = SharedVectorService(vector_store=self._vector_store)

    def _reset_vector_index(self) -> None:
        path = Path(self.index_path)
        candidates = {
            path,
            path.with_suffix(path.suffix + ".meta"),
            path.with_suffix(path.suffix + ".meta.json"),
            path.with_suffix(".meta"),
            path.with_suffix(".meta.json"),
        }
        for candidate in candidates:
            try:
                if candidate.exists():
                    candidate.unlink()
            except Exception:
                LOGGER.debug("Failed to remove vector index file: %s", candidate)


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
        help="Maximum tokens per chunk (defaults to stack_dataset.chunk_size)",
    )
    parser.add_argument("--split", default="train", help="Dataset split to stream")
    parser.add_argument("--dataset", default="bigcode/the-stack-dedup", help="Dataset identifier")
    parser.add_argument("--db", default="stack_embeddings.db", help="SQLite database path")
    parser.add_argument("--namespace", default="stack", help="Vector store namespace")
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

    auth_token = _resolve_hf_token()
    if not auth_token:
        LOGGER.info(
            "No Hugging Face credentials found (set STACK_HF_TOKEN if required); proceeding unauthenticated"
        )

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
        index_path=args.index_path,
        vector_backend=args.backend,
        use_auth_token=auth_token,
        max_lines_per_document=stack_cfg.max_lines_per_document,
        max_chunk_lines=stack_cfg.max_lines_per_document,
    )
    service.run(resume=args.resume, limit=args.limit)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
