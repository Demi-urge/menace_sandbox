from __future__ import annotations

"""Streaming ingestion for BigCode's The Stack dataset.

This module exposes :class:`StackDatasetStreamer` which streams records from the
``bigcode/the-stack-v2-dedup`` dataset, chunks source files into line-limited
segments and persists only their embeddings plus structured metadata. State is
tracked in a lightweight SQLite catalogue so ingestion can resume after
interruption without re-embedding previously processed chunks.
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, TYPE_CHECKING

from dynamic_path_router import resolve_path

try:  # pragma: no cover - optional heavy dependency
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - fallback when datasets unavailable
    load_dataset = None  # type: ignore

from .vector_store import VectorStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .vectorizer import SharedVectorService as SharedVectorServiceType
else:  # pragma: no cover - runtime fallback type alias
    SharedVectorServiceType = Any  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


def _lookup_hf_token() -> str | None:
    """Return any configured Hugging Face token from config or the environment."""

    candidates: list[str | None] = []
    try:  # pragma: no cover - configuration access is optional in tests
        from config import CONFIG  # type: ignore

        candidates.append(getattr(CONFIG, "huggingface_token", None))
        for attr in ("sandbox", "sandbox_settings", "sandbox_config"):
            section = getattr(CONFIG, attr, None)
            if section is not None:
                candidates.append(getattr(section, "huggingface_token", None))
    except Exception:
        pass

    env_fallbacks = [
        os.environ.get("HUGGINGFACE_API_TOKEN"),
        os.environ.get("HF_TOKEN"),
        os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    ]
    candidates.extend(env_fallbacks)
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def _coerce_languages(value: Iterable[str] | str | None) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
    else:
        items = [str(part).strip() for part in value]
    return {item for item in items if item}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class StackIngestionConfig:
    """Runtime configuration for streaming ingestion."""

    dataset_name: str = "bigcode/the-stack-v2-dedup"
    split: str = "train"
    allowed_languages: set[str] = field(default_factory=set)
    max_lines: int = 200
    vector_dim: int = 768
    vector_backend: str = "annoy"
    vector_metric: str = "angular"
    vector_store_path: Path = field(
        default_factory=lambda: resolve_path("vector_service") / "stack_vectors.db"
    )
    metadata_path: Path = field(
        default_factory=lambda: resolve_path("vector_service") / "stack_metadata.db"
    )
    cache_dir: Path = field(
        default_factory=lambda: resolve_path("vector_service") / "stack_cache"
    )
    batch_size: int | None = None
    retry_attempts: int = 3
    retry_initial: float = 1.0
    retry_backoff: float = 2.0
    idle_sleep: float = 5.0
    token: str | None = None
    streaming_enabled: bool = True

    @classmethod
    def from_environment(cls, **overrides: Any) -> "StackIngestionConfig":
        """Build configuration using environment hints and overrides."""

        allowed = overrides.pop("allowed_languages", None)
        if allowed is None:
            allowed = _coerce_languages(os.environ.get("STACK_LANGUAGES"))
        max_lines = int(os.environ.get("STACK_MAX_LINES", overrides.pop("max_lines", 200)))
        vector_dim = int(os.environ.get("STACK_VECTOR_DIM", overrides.pop("vector_dim", 768)))
        backend = os.environ.get("STACK_VECTOR_BACKEND", overrides.pop("vector_backend", "annoy"))
        metric = os.environ.get("STACK_VECTOR_METRIC", overrides.pop("vector_metric", "angular"))
        dataset_name = os.environ.get(
            "STACK_DATASET", overrides.pop("dataset_name", "bigcode/the-stack-v2-dedup")
        )
        split = os.environ.get("STACK_SPLIT", overrides.pop("split", "train"))
        batch_size_env = os.environ.get("STACK_BATCH_SIZE")
        batch_size = (
            int(batch_size_env)
            if batch_size_env is not None
            else overrides.pop("batch_size", None)
        )
        streaming_enabled = _env_flag("STACK_STREAMING", overrides.pop("streaming_enabled", True))

        vector_path_env = os.environ.get("STACK_VECTOR_PATH")
        metadata_path_env = os.environ.get("STACK_METADATA_PATH")
        cache_dir_env = os.environ.get("STACK_CACHE_DIR")

        cfg = cls(
            dataset_name=dataset_name,
            split=split,
            allowed_languages=_coerce_languages(allowed),
            max_lines=max_lines,
            vector_dim=vector_dim,
            vector_backend=backend,
            vector_metric=metric,
            batch_size=batch_size,
            streaming_enabled=streaming_enabled,
            token=overrides.pop("token", None) or _lookup_hf_token(),
        )
        if vector_path_env:
            cfg.vector_store_path = Path(vector_path_env)
        if metadata_path_env:
            cfg.metadata_path = Path(metadata_path_env)
        if cache_dir_env:
            cfg.cache_dir = Path(cache_dir_env)
        for field_name, value in overrides.items():
            if hasattr(cfg, field_name):
                setattr(cfg, field_name, value)
        cfg.vector_store_path = Path(cfg.vector_store_path)
        cfg.metadata_path = Path(cfg.metadata_path)
        cfg.cache_dir = Path(cfg.cache_dir)
        cfg.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        return cfg


@dataclass(slots=True)
class StackRecord:
    repo: str
    path: str
    language: str
    content: str
    identifier: str


@dataclass(slots=True)
class StackChunk:
    repo: str
    path: str
    language: str
    start_line: int
    end_line: int
    checksum: str
    text: str

    def release(self) -> None:
        self.text = ""


@dataclass(slots=True)
class StackIngestionMetrics:
    files_seen: int = 0
    files_skipped: int = 0
    chunks_embedded: int = 0
    chunks_skipped: int = 0
    errors: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "files_seen": self.files_seen,
            "files_skipped": self.files_skipped,
            "chunks_embedded": self.chunks_embedded,
            "chunks_skipped": self.chunks_skipped,
            "errors": self.errors,
        }


class SQLiteVectorStore:
    """Minimal SQLite-backed :class:`VectorStore` implementation."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()
        self._initialise()

    def _initialise(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    origin_db TEXT,
                    repo TEXT,
                    path TEXT,
                    language TEXT,
                    vector TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    # ``VectorStore`` protocol compliance ---------------------------------
    def add(
        self,
        kind: str,
        record_id: str,
        vector: Iterable[float],
        *,
        origin_db: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        meta = dict(metadata or {})
        payload = {
            "kind": kind,
            "id": record_id,
            "origin_db": origin_db,
            "repo": str(meta.get("repo", "")),
            "path": str(meta.get("path", "")),
            "language": str(meta.get("language", "")),
            "vector": json.dumps([float(x) for x in vector]),
        }
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (id, kind, origin_db, repo, path, language, vector)
                VALUES (:id, :kind, :origin_db, :repo, :path, :language, :vector)
                """,
                payload,
            )

    def query(self, vector: Iterable[float], top_k: int = 5) -> list[tuple[str, float]]:
        # Nearest neighbour search is out-of-scope for ingestion tests; return empty results.
        return []

    def load(self) -> None:  # pragma: no cover - trivial method for protocol compatibility
        self._initialise()

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()


class StackMetadataStore:
    """Lightweight SQLite catalogue for chunk metadata and dataset cursors."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._initialise()

    def _initialise(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    checksum TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    path TEXT NOT NULL,
                    language TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cursors (
                    split TEXT PRIMARY KEY,
                    cursor TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_repo_path ON chunks(repo, path)"
            )

    def has_chunk(self, checksum: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM chunks WHERE checksum = ? LIMIT 1", (checksum,)
        )
        row = cur.fetchone()
        cur.close()
        return row is not None

    def record_chunk(self, chunk: StackChunk) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO chunks (checksum, repo, path, language, start_line, end_line)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.checksum,
                    chunk.repo,
                    chunk.path,
                    chunk.language,
                    chunk.start_line,
                    chunk.end_line,
                ),
            )

    def update_cursor(self, split: str, cursor: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO cursors (split, cursor)
                VALUES (?, ?)
                ON CONFLICT(split) DO UPDATE SET cursor = excluded.cursor,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (split, cursor),
            )

    def last_cursor(self, split: str) -> str | None:
        cur = self._conn.execute(
            "SELECT cursor FROM cursors WHERE split = ?", (split,)
        )
        row = cur.fetchone()
        cur.close()
        if row:
            return str(row[0])
        return None

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._conn.close()


class StackDatasetStreamer:
    """Stream ``the-stack`` dataset, embedding chunks into a dedicated store."""

    @classmethod
    def from_environment(cls, **overrides: Any) -> "StackDatasetStreamer":
        config = StackIngestionConfig.from_environment(**overrides)
        return cls(config=config)

    def __init__(
        self,
        config: StackIngestionConfig | None = None,
        *,
        metadata_store: StackMetadataStore | None = None,
        vector_service: SharedVectorServiceType | None = None,
        dataset_loader: Callable[..., Iterable[Dict[str, Any]]] | None = None,
        vector_store_factory: Callable[[StackIngestionConfig], VectorStore] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or StackIngestionConfig.from_environment()
        self.metadata_store = metadata_store or StackMetadataStore(self.config.metadata_path)
        self._dataset_loader = dataset_loader or load_dataset
        if self._dataset_loader is None:
            raise RuntimeError("datasets.load_dataset is unavailable; install datasets")
        self._vector_store_factory = vector_store_factory or self._build_vector_store
        self._vector_store = self._vector_store_factory(self.config)
        if vector_service is None:
            try:
                from .vectorizer import SharedVectorService as _SharedVectorService
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("SharedVectorService unavailable") from exc
            self.vector_service = _SharedVectorService(vector_store=self._vector_store)
        else:
            self.vector_service = vector_service
        self.metrics = StackIngestionMetrics()
        self._sleep = sleep or time.sleep
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process(self, *, limit: int | None = None, continuous: bool = False) -> int:
        """Stream dataset chunks, returning the number of embedded chunks."""

        if not self._is_streaming_enabled():
            logger.info("stack dataset streaming disabled via STACK_STREAMING flag")
            return 0

        total_embedded = 0
        while not self._stop_event.is_set():
            start = time.time()
            embedded = self._process_once(limit=limit)
            total_embedded += embedded
            elapsed = time.time() - start
            logger.info(
                "stack ingestion iteration processed %s chunks (skipped=%s, files=%s) in %.2fs",
                embedded,
                self.metrics.chunks_skipped,
                self.metrics.files_seen,
                elapsed,
            )
            if not continuous or (limit is not None and embedded >= limit):
                break
            if embedded == 0 and self.config.batch_size is not None:
                # Batch completed with no new data; avoid tight loop
                self._sleep(self.config.idle_sleep)
            if not self._is_streaming_enabled():
                break
        return total_embedded

    async def process_async(self, *, limit: int | None = None, continuous: bool = False) -> int:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.process(limit=limit, continuous=continuous)
        )

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------
    def _build_vector_store(self, config: StackIngestionConfig) -> VectorStore:
        return SQLiteVectorStore(config.vector_store_path)

    def _is_streaming_enabled(self) -> bool:
        if not self.config.streaming_enabled:
            return False
        return _env_flag("STACK_STREAMING", True)

    def _process_once(self, *, limit: int | None = None) -> int:
        dataset_iter = self._load_dataset_with_retry()
        embedded = 0
        for index, record in enumerate(dataset_iter):
            stack_record = self._coerce_record(record, position=index)
            if stack_record is None:
                self.metrics.files_skipped += 1
                continue
            self.metrics.files_seen += 1
            try:
                embedded += self._process_record(stack_record, limit=limit, so_far=embedded)
            except Exception:  # pragma: no cover - unexpected runtime failure
                self.metrics.errors += 1
                logger.exception(
                    "failed to process stack record %s:%s", stack_record.repo, stack_record.path
                )
            self.metadata_store.update_cursor(self.config.split, stack_record.identifier)
            if limit is not None and embedded >= limit:
                break
        return embedded

    def _load_dataset_with_retry(self) -> Iterable[Dict[str, Any]]:
        attempt = 0
        delay = self.config.retry_initial
        while True:
            try:
                dataset = self._dataset_loader(
                    self.config.dataset_name,
                    split=self.config.split,
                    streaming=True,
                    token=self.config.token,
                    cache_dir=str(self.config.cache_dir),
                    keep_in_memory=False,
                )
                return dataset
            except Exception as exc:
                attempt += 1
                if attempt >= self.config.retry_attempts:
                    logger.exception("stack dataset load failed after %s attempts", attempt)
                    raise
                logger.warning(
                    "stack dataset load failed (attempt %s/%s): %s", attempt, self.config.retry_attempts, exc
                )
                self._sleep(delay)
                delay *= self.config.retry_backoff

    def _coerce_record(self, record: Dict[str, Any], *, position: int) -> StackRecord | None:
        content = record.get("content")
        language = str(record.get("language", "")).strip()
        if not content or not isinstance(content, str):
            return None
        if self.config.allowed_languages and language not in self.config.allowed_languages:
            return None
        path_raw = str(record.get("path", ""))
        repo = str(record.get("repo_name", record.get("repo", "unknown")))
        identifier = str(record.get("id", f"{repo}:{path_raw}:{position}"))
        path = self._normalise_path(path_raw)
        return StackRecord(repo=repo, path=path, language=language or "unknown", content=content, identifier=identifier)

    def _normalise_path(self, raw: str) -> str:
        if not raw:
            return "unknown"
        try:
            path = Path(raw)
            normalised = path.as_posix()
        except Exception:
            normalised = raw.replace("\\", "/")
        normalised = normalised.lstrip("./")
        while normalised.startswith("/"):
            normalised = normalised[1:]
        return normalised or "unknown"

    def _process_record(self, record: StackRecord, *, limit: int | None, so_far: int) -> int:
        embedded = 0
        for chunk in self._chunk_record(record):
            if self.metadata_store.has_chunk(chunk.checksum):
                self.metrics.chunks_skipped += 1
                continue
            self._embed_chunk(chunk)
            self.metadata_store.record_chunk(chunk)
            chunk.release()
            embedded += 1
            self.metrics.chunks_embedded += 1
            if limit is not None and so_far + embedded >= limit:
                break
        return embedded

    def _chunk_record(self, record: StackRecord) -> Iterator[StackChunk]:
        lines = record.content.splitlines()
        if not lines:
            return
        max_lines = max(self.config.max_lines, 1)
        for start in range(0, len(lines), max_lines):
            chunk_lines = lines[start : start + max_lines]
            text = "\n".join(chunk_lines).strip()
            if not text:
                continue
            start_line = start + 1
            end_line = start + len(chunk_lines)
            checksum = hashlib.sha1(
                f"{record.repo}:{record.path}:{start_line}:{end_line}:{text}".encode("utf-8")
            ).hexdigest()
            yield StackChunk(
                repo=record.repo,
                path=record.path,
                language=record.language,
                start_line=start_line,
                end_line=end_line,
                checksum=checksum,
                text=text,
            )

    def _embed_chunk(self, chunk: StackChunk) -> None:
        metadata = {"path": chunk.path, "repo": chunk.repo, "language": chunk.language}
        record = {"text": chunk.text, "language": chunk.language, "path": chunk.path}
        self.vector_service.vectorise_and_store(
            "stack", chunk.checksum, record, origin_db="stack", metadata=metadata
        )


# ---------------------------------------------------------------------------
# Helper entry points
# ---------------------------------------------------------------------------


_BACKGROUND_LOCK = threading.Lock()
_BACKGROUND_THREAD: threading.Thread | None = None


def ensure_background_task(**kwargs: Any) -> bool:
    """Start a background ingestion thread when enabled by environment."""

    global _BACKGROUND_THREAD
    if not _env_flag("STACK_STREAMING", False):
        return False
    with _BACKGROUND_LOCK:
        if _BACKGROUND_THREAD and _BACKGROUND_THREAD.is_alive():
            return True

        def _run() -> None:
            try:
                streamer = StackDatasetStreamer.from_environment(**kwargs)
                batch = streamer.config.batch_size
                streamer.process(limit=batch, continuous=batch is None)
            except Exception:  # pragma: no cover - background best effort
                logger.exception("stack background ingestion failed")

        thread = threading.Thread(target=_run, name="stack-streaming", daemon=True)
        thread.start()
        _BACKGROUND_THREAD = thread
        return True


def run_stack_ingestion_async(*, limit: int | None = None, continuous: bool = False) -> asyncio.Future[int]:
    """Convenience wrapper returning an asyncio Task for stack ingestion."""

    loop = asyncio.get_event_loop()
    streamer = StackDatasetStreamer.from_environment()
    return loop.create_task(streamer.process_async(limit=limit, continuous=continuous))


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for manual dataset ingestion."""

    parser = argparse.ArgumentParser(description="Stream BigCode stack embeddings")
    parser.add_argument("--limit", type=int, default=None, help="Maximum chunks per run")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continue streaming indefinitely instead of stopping after one pass",
    )
    args = parser.parse_args(argv)

    streamer = StackDatasetStreamer.from_environment()
    count = streamer.process(limit=args.limit, continuous=args.continuous)
    logger.info("stack ingestion completed: %s chunks embedded", count)
    return count


__all__ = [
    "StackDatasetStreamer",
    "StackIngestionConfig",
    "StackMetadataStore",
    "StackIngestionMetrics",
    "SQLiteVectorStore",
    "ensure_background_task",
    "run_stack_ingestion_async",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
