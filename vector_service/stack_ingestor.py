"""Streaming ingestion helpers for The Stack dataset.

This module provides a light-weight pipeline that streams code files from the
`bigcode/the-stack-dedup` dataset, embeds the resulting snippets and persists
only anonymised metadata alongside the vectors.  The focus is on keeping memory
usage predictable (by batching embeddings) and avoiding the storage of raw
source code which is immediately discarded after the embeddings are computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import hashlib
import logging
import sqlite3
import time

try:  # pragma: no cover - optional dependency for runtime environments
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - tests provide a stub
    load_dataset = None  # type: ignore

from vector_service.embed_utils import EMBED_DIM, get_text_embeddings
from vector_service.vector_store import VectorStore, create_vector_store

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StackFile:
    """Simple container representing a single file from The Stack."""

    file_id: str
    repo: str
    path: str
    language: str
    license: str | None
    content: str

    @property
    def byte_size(self) -> int:
        return len(self.content.encode("utf-8"))

    @property
    def line_count(self) -> int:
        if not self.content:
            return 0
        return self.content.count("\n") + 1


@dataclass(frozen=True)
class StackChunk:
    """Represents a chunked snippet derived from a :class:`StackFile`."""

    file_id: str
    index: int
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Metadata persistence
# ---------------------------------------------------------------------------


class StackMetadataStore:
    """Persist embedding metadata and ingestion progress in SQLite."""

    def __init__(self, path: str | Path, *, namespace: str = "stack") -> None:
        self.path = Path(path)
        self.namespace = namespace
        self.conn = sqlite3.connect(self.path)
        self._init_schema()

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
                chunk_hash TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                created_at REAL NOT NULL
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

    # Progress ---------------------------------------------------------
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

    # Metadata ---------------------------------------------------------
    def upsert_embedding(
        self,
        *,
        embedding_id: str,
        file_id: str,
        repo: str,
        path: str,
        language: str,
        chunk_hash: str,
        chunk_index: int,
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
                chunk_hash,
                chunk_index,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embedding_id,
                file_id,
                repo or None,
                path or None,
                language or None,
                chunk_hash,
                chunk_index,
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
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.languages = tuple(lang.lower() for lang in (languages or ())) or None
        self.max_lines = max_lines if (max_lines or 0) > 0 else None
        self.max_bytes = max_bytes if (max_bytes or 0) > 0 else None
        self.use_auth_token = use_auth_token

    # Public API -------------------------------------------------------
    def __iter__(self) -> Iterator[StackFile]:
        return self.iter_files()

    def iter_files(self) -> Iterator[StackFile]:
        if load_dataset is None:  # pragma: no cover - runtime dependency missing
            raise RuntimeError("datasets package not available; install `datasets` to stream The Stack")

        kwargs: MutableMapping[str, object] = {"streaming": True, "split": self.split}
        if self.use_auth_token:
            kwargs["use_auth_token"] = self.use_auth_token

        LOGGER.info("Loading dataset %s (split=%s)", self.dataset_name, self.split)
        dataset = load_dataset(self.dataset_name, **kwargs)
        iterator: Iterable[Mapping[str, object]]
        if isinstance(dataset, Mapping):
            # Some dataset versions return a mapping of splits; pick the requested one.
            iterator = dataset[self.split]  # type: ignore[index]
        else:
            iterator = dataset

        for sample in iterator:
            file = self._coerce_sample(sample)
            if file is None:
                continue
            yield file

    # Internal helpers -------------------------------------------------
    def _coerce_sample(self, sample: Mapping[str, object]) -> StackFile | None:
        language = str(sample.get("language") or "").strip().lower()
        if self.languages and language not in self.languages:
            return None
        content = sample.get("content")
        if not isinstance(content, str) or not content.strip():
            return None

        truncated = self._truncate_content(content)
        repo = str(sample.get("repo_name") or "")
        path = str(sample.get("path") or "")
        license_info = sample.get("license")
        license_str = str(license_info) if license_info is not None else None
        file_id = self._file_identifier(repo, path, sample)

        return StackFile(
            file_id=file_id,
            repo=repo,
            path=path,
            language=language,
            license=license_str,
            content=truncated,
        )

    def _truncate_content(self, content: str) -> str:
        text = content
        if self.max_lines is not None:
            lines = text.splitlines()
            text = "\n".join(lines[: self.max_lines])
        if self.max_bytes is not None:
            encoded = text.encode("utf-8")
            if len(encoded) > self.max_bytes:
                text = encoded[: self.max_bytes].decode("utf-8", errors="ignore")
        return text

    @staticmethod
    def _file_identifier(repo: str, path: str, sample: Mapping[str, object]) -> str:
        base = f"{repo}:{path}".strip(":")
        if not base:
            base = str(sample.get("id") or "")
        if not base:
            base = hashlib.sha1(repr(sorted(sample.items())).encode("utf-8")).hexdigest()
        return base


# ---------------------------------------------------------------------------
# Ingestion pipeline
# ---------------------------------------------------------------------------


EmbeddingFn = Callable[[List[str]], List[List[float]]]


@dataclass
class StackIngestor:
    """Stream files from The Stack and persist embeddings and metadata."""

    dataset_name: str = "bigcode/the-stack-dedup"
    split: str = "train"
    languages: Sequence[str] | None = None
    max_lines: int | None = None
    max_bytes: int | None = None
    chunk_lines: int = 200
    batch_size: int = 16
    namespace: str = "stack"
    metadata_path: str | Path = "stack_embeddings.db"
    index_path: str | Path | None = None
    vector_backend: str | None = None
    use_auth_token: str | None = None
    embedding_fn: Callable[..., List[List[float]]] = get_text_embeddings
    vector_store: VectorStore | None = None
    metadata_store: StackMetadataStore | None = None

    def __post_init__(self) -> None:
        self.languages = tuple(lang.lower() for lang in (self.languages or ())) or None
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.chunk_lines < 1:
            raise ValueError("chunk_lines must be positive")
        meta_path = Path(self.metadata_path)
        self.metadata_store = self.metadata_store or StackMetadataStore(meta_path, namespace=self.namespace)
        if self.index_path is None:
            self.index_path = Path(meta_path).with_suffix(".index")
        else:
            self.index_path = Path(self.index_path)
        self.vector_store = self.vector_store or create_vector_store(
            EMBED_DIM,
            self.index_path,
            backend=self.vector_backend,
            metric="angular",
        )
        self.stream = StackDatasetStream(
            self.dataset_name,
            split=self.split,
            languages=self.languages,
            max_lines=self.max_lines,
            max_bytes=self.max_bytes,
            use_auth_token=self.use_auth_token,
        )

    # Public API -------------------------------------------------------
    def ingest(self, *, resume: bool = False, limit: int | None = None) -> int:
        """Stream the dataset and persist embeddings.

        Parameters
        ----------
        resume:
            When ``True`` previously processed files are preserved and ingestion
            resumes from the recorded checkpoint.  Otherwise the metadata and
            vector index are reset before processing starts.
        limit:
            Optional maximum number of files to process.  Useful for smoke tests
            or to bound runtime during development.
        Returns
        -------
        int
            The number of files processed during this run.
        """

        if not resume:
            LOGGER.info("Resetting Stack ingestion state")
            self.metadata_store.reset()  # type: ignore[union-attr]
            self._reset_vector_index()

        processed = 0
        for file in self.stream:
            if limit is not None and processed >= limit:
                break
            if self.metadata_store.has_processed(file.file_id):  # type: ignore[union-attr]
                LOGGER.debug("Skipping already processed file: %s", file.file_id)
                continue
            chunk_count = self._process_file(file)
            if chunk_count:
                processed += 1
            self.metadata_store.mark_processed(file.file_id)  # type: ignore[union-attr]

        LOGGER.info("Completed Stack ingestion for %d files", processed)
        return processed

    # Internal helpers -------------------------------------------------
    def _process_file(self, file: StackFile) -> int:
        chunks = list(self._chunk_file(file))
        if not chunks:
            return 0

        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            texts = [chunk.text for chunk in batch]
            embeddings = self._embed_batch(texts)
            for chunk, vector in zip(batch, embeddings):
                embedding_id = f"{file.file_id}::chunk-{chunk.index}"
                metadata = {
                    "repo": file.repo,
                    "path": file.path,
                    "language": file.language,
                    "hash": chunk.sha256,
                }
                self.vector_store.add(  # type: ignore[union-attr]
                    self.namespace,
                    embedding_id,
                    vector,
                    origin_db=self.namespace,
                    metadata=metadata,
                )
                self.metadata_store.upsert_embedding(  # type: ignore[union-attr]
                    embedding_id=embedding_id,
                    file_id=file.file_id,
                    repo=file.repo,
                    path=file.path,
                    language=file.language,
                    chunk_hash=chunk.sha256,
                    chunk_index=chunk.index,
                )
        # Wipe chunk texts eagerly to help the GC drop references quickly.
        chunks.clear()
        return 1

    def _chunk_file(self, file: StackFile) -> Iterator[StackChunk]:
        lines = file.content.splitlines()
        chunk_size = max(1, int(self.chunk_lines))
        for idx in range(0, len(lines), chunk_size):
            segment = "\n".join(lines[idx : idx + chunk_size])
            if not segment.strip():
                continue
            yield StackChunk(file.file_id, idx // chunk_size, segment)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        kwargs = {}
        return self.embedding_fn(texts, **kwargs)

    def _reset_vector_index(self) -> None:
        if self.index_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        candidates = {
            self.index_path,
            self.index_path.with_suffix(self.index_path.suffix + ".meta"),
            self.index_path.with_suffix(self.index_path.suffix + ".meta.json"),
            self.index_path.with_suffix(".meta"),
            self.index_path.with_suffix(".meta.json"),
        }
        for candidate in candidates:
            try:
                if candidate.exists():
                    candidate.unlink()
            except Exception:
                LOGGER.debug("Failed to remove vector index candidate %s", candidate)


__all__ = [
    "StackChunk",
    "StackDatasetStream",
    "StackFile",
    "StackIngestor",
    "StackMetadataStore",
]

