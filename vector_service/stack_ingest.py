"""Streaming ingestion pipeline for The Stack dataset.

This module provides a high level façade for streaming repository files from
`bigcode/the-stack-dedup`, chunking them into manageable snippets and
generating embeddings via :class:`~vector_service.vectorizer.SharedVectorService`.

Only the resulting embeddings and their associated metadata are cached in a
lightweight SQLite (or optional DuckDB) database.  Raw source content is
discarded immediately after each batch so the cache never stores the original
code and temporary artefacts are cleaned up eagerly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Sequence

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import tempfile
import time

try:  # pragma: no cover - optional dependency for runtime environments
    import duckdb  # type: ignore
except Exception:  # pragma: no cover - duckdb is optional
    duckdb = None  # type: ignore

try:  # pragma: no cover - runtime dependency
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - tests provide a stub
    load_dataset = None  # type: ignore

from config import StackDatasetConfig, get_config
from vector_service import SharedVectorService

LOGGER = logging.getLogger(__name__)

_HF_ENV_KEYS = (
    "STACK_HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_API_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_TOKEN",
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


def _normalise_language(language: str) -> str:
    return str(language or "").strip().lower()


@dataclass
class StackFile:
    """Representation of a single repository file from The Stack."""

    file_id: str
    repo: str
    path: str
    language: str
    content: str

    @property
    def byte_size(self) -> int:
        return len(self.content.encode("utf-8"))

    @property
    def line_count(self) -> int:
        if not self.content:
            return 0
        return self.content.count("\n") + 1


@dataclass
class StackChunk:
    """Chunked snippet originating from :class:`StackFile`."""

    file_id: str
    index: int
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def line_count(self) -> int:
        if not self.text:
            return 0
        return self.text.count("\n") + 1


# ---------------------------------------------------------------------------
# Metadata cache
# ---------------------------------------------------------------------------


class StackMetadataStore:
    """Persist embedding metadata (and progress) without raw source code."""

    def __init__(self, path: str | Path, *, namespace: str = "stack", use_duckdb: bool | None = None) -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.backend = self._resolve_backend(use_duckdb)
        self.conn = self._connect()
        self._init_schema()

    # Setup -----------------------------------------------------------------
    def _resolve_backend(self, use_duckdb: bool | None) -> str:
        if use_duckdb is True:
            if duckdb is None:
                raise RuntimeError("duckdb requested but not available")
            return "duckdb"
        if use_duckdb is False:
            return "sqlite"
        if self.path.suffix.lower() == ".duckdb" and duckdb is not None:
            return "duckdb"
        return "sqlite"

    def _connect(self):  # pragma: no cover - trivial wrappers
        if self.backend == "duckdb":
            return duckdb.connect(str(self.path))  # type: ignore[arg-type]
        return sqlite3.connect(self.path)

    @property
    def metadata_table(self) -> str:
        return f"{self.namespace}_embeddings"

    @property
    def progress_table(self) -> str:
        return f"{self.namespace}_progress"

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
                file_lines INTEGER,
                chunk_lines INTEGER,
                chunk_index INTEGER,
                chunk_hash TEXT,
                vector BLOB,
                created_at REAL NOT NULL
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.progress_table} (
                file_id TEXT PRIMARY KEY,
                repo TEXT,
                path TEXT,
                language TEXT,
                processed_at REAL NOT NULL
            )
            """
        )
        self.conn.commit()

    # Progress ---------------------------------------------------------------
    def has_processed(self, file_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute(f"SELECT 1 FROM {self.progress_table} WHERE file_id = ?", (file_id,))
        return cur.fetchone() is not None

    def mark_processed(self, file: StackFile) -> None:
        cur = self.conn.cursor()
        cur.execute(
            f"INSERT OR REPLACE INTO {self.progress_table} (file_id, repo, path, language, processed_at) VALUES (?, ?, ?, ?, ?)",
            (file.file_id, file.repo, file.path, file.language, time.time()),
        )
        self.conn.commit()

    def reset(self) -> None:
        cur = self.conn.cursor()
        cur.execute(f"DELETE FROM {self.metadata_table}")
        cur.execute(f"DELETE FROM {self.progress_table}")
        self.conn.commit()

    # Metadata ---------------------------------------------------------------
    def upsert_embedding(
        self,
        *,
        embedding_id: str,
        file: StackFile,
        chunk: StackChunk,
        vector: Sequence[float],
    ) -> None:
        payload = json.dumps([float(x) for x in vector]).encode("utf-8")
        if self.backend == "sqlite":
            payload = sqlite3.Binary(payload)
        cur = self.conn.cursor()
        cur.execute(
            f"""
            INSERT OR REPLACE INTO {self.metadata_table} (
                embedding_id,
                file_id,
                repo,
                path,
                language,
                file_lines,
                chunk_lines,
                chunk_index,
                chunk_hash,
                vector,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embedding_id,
                file.file_id,
                file.repo,
                file.path,
                file.language,
                file.line_count,
                chunk.line_count,
                chunk.index,
                chunk.sha256,
                payload,
                time.time(),
            ),
        )
        self.conn.commit()


# ---------------------------------------------------------------------------
# Dataset streaming
# ---------------------------------------------------------------------------


class StackDatasetStream:
    """Wrapper around :func:`datasets.load_dataset` yielding :class:`StackFile`."""

    def __init__(
        self,
        dataset_name: str = "bigcode/the-stack-dedup",
        *,
        split: str = "train",
        languages: Sequence[str] | None = None,
        max_lines: int | None = None,
        max_bytes: int | None = None,
        use_auth_token: str | None = None,
        dataset_loader=load_dataset,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.languages = tuple(_normalise_language(lang) for lang in (languages or ())) or None
        self.max_lines = max_lines if (max_lines or 0) > 0 else None
        self.max_bytes = max_bytes if (max_bytes or 0) > 0 else None
        self.use_auth_token = use_auth_token
        self.dataset_loader = dataset_loader

    def __iter__(self) -> Iterator[StackFile]:
        return self.iter_files()

    def iter_files(self) -> Iterator[StackFile]:
        if self.dataset_loader is None:  # pragma: no cover - runtime dependency missing
            raise RuntimeError("datasets package not available; install `datasets` to stream The Stack")

        kwargs: MutableMapping[str, object] = {"streaming": True, "split": self.split}
        if self.use_auth_token:
            kwargs["use_auth_token"] = self.use_auth_token

        LOGGER.info("Loading dataset %s (split=%s)", self.dataset_name, self.split)
        dataset = self.dataset_loader(self.dataset_name, **kwargs)

        iterator: Iterable[Mapping[str, object]]
        if isinstance(dataset, Mapping):
            iterator = dataset[self.split]  # type: ignore[index]
        else:
            iterator = dataset

        for sample in iterator:
            file = self._coerce_sample(sample)
            if file is None:
                continue
            yield file

    # Helpers ----------------------------------------------------------------
    def _coerce_sample(self, sample: Mapping[str, object]) -> StackFile | None:
        language = _normalise_language(str(sample.get("language") or ""))
        if self.languages and language not in self.languages:
            return None

        content = sample.get("content")
        if not isinstance(content, str) or not content.strip():
            return None

        text = self._truncate_content(content)

        repo = str(sample.get("repo_name") or sample.get("repo") or "").strip()
        path = str(sample.get("path") or sample.get("file_path") or "").strip()
        file_id = self._derive_identifier(repo, path, sample)

        return StackFile(
            file_id=file_id,
            repo=repo,
            path=path,
            language=language,
            content=text,
        )

    def _truncate_content(self, content: str) -> str:
        text = content
        if self.max_lines is not None and self.max_lines > 0:
            lines = text.splitlines()
            text = "\n".join(lines[: self.max_lines])
        if self.max_bytes is not None and self.max_bytes > 0:
            encoded = text.encode("utf-8")
            if len(encoded) > self.max_bytes:
                text = encoded[: self.max_bytes].decode("utf-8", errors="ignore")
        return text

    @staticmethod
    def _derive_identifier(repo: str, path: str, sample: Mapping[str, object]) -> str:
        base = f"{repo}:{path}".strip(":")
        if base:
            return base
        identifier = str(sample.get("id") or sample.get("hexsha") or "").strip()
        if identifier:
            return identifier
        digest = hashlib.sha1(repr(sorted(sample.items())).encode("utf-8")).hexdigest()
        return digest


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------


class StackIngestor:
    """Stream Stack files, chunk them and persist embeddings + metadata."""

    def __init__(
        self,
        *,
        dataset_name: str = "bigcode/the-stack-dedup",
        split: str = "train",
        languages: Sequence[str] | None = None,
        max_lines: int | None = None,
        max_bytes: int | None = None,
        chunk_lines: int = 256,
        batch_size: int = 16,
        record_kind: str = "stack_code",
        metadata_path: str | Path = "stack_embeddings.db",
        namespace: str = "stack",
        use_auth_token: str | None = None,
        dataset_loader=load_dataset,
        vector_service: SharedVectorService | None = None,
        metadata_store: StackMetadataStore | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if chunk_lines < 1:
            raise ValueError("chunk_lines must be positive")

        self.chunk_lines = int(chunk_lines)
        self.batch_size = int(batch_size)
        self.record_kind = record_kind
        self.vector_service = vector_service or SharedVectorService()

        self.stream = StackDatasetStream(
            dataset_name,
            split=split,
            languages=languages,
            max_lines=max_lines,
            max_bytes=max_bytes,
            use_auth_token=use_auth_token,
            dataset_loader=dataset_loader,
        )

        self.metadata_store = metadata_store or StackMetadataStore(metadata_path, namespace=namespace)

    # Public API -------------------------------------------------------------
    def ingest(self, *, resume: bool = False, limit: int | None = None) -> int:
        if not resume:
            LOGGER.info("Resetting Stack metadata cache at %s", self.metadata_store.path)
            self.metadata_store.reset()

        processed = 0
        for file in self.stream:
            if limit is not None and processed >= limit:
                break
            if self.metadata_store.has_processed(file.file_id):
                LOGGER.debug("Skipping already processed file: %s", file.file_id)
                continue

            chunk_count = self._process_file(file)
            if chunk_count:
                processed += 1
            self.metadata_store.mark_processed(file)

        LOGGER.info("Completed Stack ingestion for %d files", processed)
        return processed

    # Internal helpers ------------------------------------------------------
    def _process_file(self, file: StackFile) -> int:
        chunks = list(self._chunk_file(file))
        if not chunks:
            return 0

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            with tempfile.TemporaryDirectory() as tmpdir:
                LOGGER.debug("Processing batch of %d chunks (tmp=%s)", len(batch), tmpdir)
                self._embed_batch(file, batch)
            for chunk in batch:
                chunk.text = ""

        file.content = ""
        return 1

    def _chunk_file(self, file: StackFile) -> Iterator[StackChunk]:
        lines = file.content.splitlines()
        chunk_size = max(1, self.chunk_lines)
        for index in range(0, len(lines), chunk_size):
            segment = "\n".join(lines[index : index + chunk_size])
            if not segment.strip():
                continue
            yield StackChunk(file.file_id, index // chunk_size, segment)

    def _embed_batch(self, file: StackFile, chunks: Sequence[StackChunk]) -> None:
        for chunk in chunks:
            record_id = f"{file.file_id}::chunk-{chunk.index}"
            record = {
                "text": chunk.text,
                "repo": file.repo,
                "path": file.path,
                "language": file.language,
                "chunk_index": chunk.index,
            }
            metadata = {
                "repo": file.repo,
                "path": file.path,
                "language": file.language,
                "file_lines": file.line_count,
                "chunk_lines": chunk.line_count,
                "chunk_index": chunk.index,
                "chunk_hash": chunk.sha256,
            }
            vector = self.vector_service.vectorise_and_store(
                self.record_kind,
                record_id,
                record,
                origin_db=self.record_kind,
                metadata=metadata,
            )
            self.metadata_store.upsert_embedding(embedding_id=record_id, file=file, chunk=chunk, vector=vector)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------


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
    parser.add_argument("--cache", default="stack_embeddings.db", help="SQLite/DuckDB cache path")
    parser.add_argument("--namespace", default="stack", help="Namespace for cache tables")
    parser.add_argument("--record-kind", default="stack_code", help="Record kind used for SharedVectorService")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on processed files")
    parser.add_argument("--resume", action="store_true", help="Resume from previous progress checkpoint")
    return parser


def main(argv: Sequence[str] | argparse.Namespace | None = None) -> int:
    parser = build_arg_parser()
    if isinstance(argv, argparse.Namespace):
        args = argv
    else:
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

    languages = args.languages if args.languages is not None else sorted(stack_cfg.languages)
    chunk_lines = args.chunk_lines if args.chunk_lines is not None else stack_cfg.chunk_lines
    max_lines = args.max_lines if args.max_lines is not None else stack_cfg.max_lines
    max_bytes = args.max_bytes if args.max_bytes is not None else stack_cfg.max_bytes

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
        max_bytes=max_bytes,
        chunk_lines=chunk_lines,
        batch_size=args.batch_size,
        record_kind=args.record_kind,
        metadata_path=args.cache,
        namespace=args.namespace,
        use_auth_token=auth_token,
    )

    ingestor.ingest(resume=args.resume, limit=args.limit)
    return 0


__all__ = [
    "StackChunk",
    "StackDatasetStream",
    "StackFile",
    "StackIngestor",
    "StackMetadataStore",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

