from __future__ import annotations

"""Streaming ingestion for BigCode's The Stack dataset.

This module exposes :class:`StackDatasetStreamer` which streams records from the
``bigcode/the-stack-v2-dedup`` dataset, chunks source files into line-limited
segments and persists only their embeddings plus structured metadata. State is
tracked in a lightweight SQLite catalogue so ingestion can resume after
interruption without re-embedding previously processed chunks. Embeddings are
persisted through :class:`~vector_service.vectorizer.SharedVectorService` into a
dedicated FAISS/Annoy index so only vectors and metadata ever hit disk.
"""

import argparse
import asyncio
import contextlib
import hashlib
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

from .stack_snippet_cache import StackSnippetCache
from .vector_store import VectorStore, create_vector_store

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

    try:  # pragma: no cover - sandbox settings optional during tests
        from sandbox_settings import SandboxSettings  # type: ignore

        settings = SandboxSettings()
        candidates.append(getattr(settings, "huggingface_token", None))
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


def _to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    for attr in ("model_dump", "dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                data = method()  # type: ignore[misc]
            except TypeError:
                data = method(exclude_none=False)  # type: ignore[misc]
            return dict(data)
    return dict(getattr(value, "__dict__", {}))


def _stack_config_defaults() -> Dict[str, Any]:
    try:  # pragma: no cover - optional configuration dependency
        from config import ContextBuilderConfig  # type: ignore
    except Exception:
        return {}

    try:
        cfg = ContextBuilderConfig()
    except Exception:
        return {}

    stack_cfg = getattr(cfg, "stack", None) or getattr(cfg, "stack_dataset", None)
    if stack_cfg is None:
        return {}

    stack_dict = _to_dict(stack_cfg)
    overrides: Dict[str, Any] = {}

    dataset_name = stack_dict.get("dataset_name")
    if dataset_name:
        overrides["dataset_name"] = dataset_name
    split = stack_dict.get("split")
    if split:
        overrides["split"] = split

    ingestion = _to_dict(stack_dict.get("ingestion"))
    cache_cfg = _to_dict(stack_dict.get("cache"))

    languages = stack_dict.get("languages") or ingestion.get("languages")
    if languages:
        overrides["allowed_languages"] = languages
    max_lines = stack_dict.get("max_lines") or ingestion.get("max_document_lines")
    if max_lines is not None:
        overrides["max_lines"] = max_lines
    chunk_overlap = stack_dict.get("chunk_overlap") or ingestion.get("chunk_overlap")
    if chunk_overlap is not None:
        overrides["chunk_overlap"] = chunk_overlap
    streaming = stack_dict.get("streaming")
    if streaming is None:
        streaming = ingestion.get("streaming")
    if streaming is not None:
        overrides["streaming_enabled"] = streaming

    if cache_cfg.get("data_dir"):
        overrides["cache_dir"] = cache_cfg.get("data_dir")
    if cache_cfg.get("index_path"):
        overrides["vector_store_path"] = cache_cfg.get("index_path")
    if cache_cfg.get("metadata_path"):
        overrides["metadata_path"] = cache_cfg.get("metadata_path")
    if cache_cfg.get("document_cache"):
        overrides["document_cache"] = cache_cfg.get("document_cache")

    return {key: value for key, value in overrides.items() if value is not None}


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
    chunk_overlap: int = 20
    vector_dim: int = 768
    vector_backend: str = "faiss"
    vector_metric: str = "angular"
    vector_store_path: Path = field(
        default_factory=lambda: resolve_path("vector_service") / "stack.faiss"
    )
    metadata_path: Path = field(
        default_factory=lambda: resolve_path("vector_service") / "stack_embeddings.db"
    )
    cache_dir: Path = field(
        default_factory=lambda: resolve_path("vector_service") / "stack_cache"
    )
    document_cache: Path = field(
        default_factory=lambda: resolve_path("chunk_summary_cache")
        / "stack_documents"
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

        defaults = _stack_config_defaults()
        merged: Dict[str, Any] = dict(defaults)
        merged.update(overrides)

        allowed_cfg = merged.pop("allowed_languages", None)
        allowed_env = os.environ.get("STACK_LANGUAGES")
        allowed = _coerce_languages(allowed_env if allowed_env is not None else allowed_cfg)

        max_lines_default = merged.pop("max_lines", 200)
        max_lines_env = os.environ.get("STACK_MAX_LINES")
        max_lines = int(max_lines_env) if max_lines_env is not None else int(max_lines_default)

        chunk_default = merged.pop("chunk_overlap", 20)
        chunk_env = os.environ.get("STACK_CHUNK_OVERLAP")
        chunk_overlap = (
            int(chunk_env)
            if chunk_env is not None
            else int(chunk_default)
        )

        vector_dim = int(os.environ.get("STACK_VECTOR_DIM", merged.pop("vector_dim", 768)))
        backend = os.environ.get("STACK_VECTOR_BACKEND", merged.pop("vector_backend", "faiss"))
        metric = os.environ.get("STACK_VECTOR_METRIC", merged.pop("vector_metric", "angular"))
        dataset_name = os.environ.get(
            "STACK_DATASET", merged.pop("dataset_name", "bigcode/the-stack-v2-dedup")
        )
        split = os.environ.get("STACK_SPLIT", merged.pop("split", "train"))
        batch_size_env = os.environ.get("STACK_BATCH_SIZE")
        batch_size = (
            int(batch_size_env)
            if batch_size_env is not None
            else merged.pop("batch_size", None)
        )
        streaming_default = bool(merged.pop("streaming_enabled", True))
        streaming_enabled = _env_flag("STACK_STREAMING", streaming_default)

        vector_path_env = os.environ.get("STACK_VECTOR_PATH")
        metadata_path_env = os.environ.get("STACK_METADATA_PATH")
        cache_dir_env = os.environ.get("STACK_CACHE_DIR") or os.environ.get("STACK_DATA_DIR")
        document_cache_env = os.environ.get("STACK_DOCUMENT_CACHE")

        vector_override = merged.pop("vector_store_path", None)
        metadata_override = merged.pop("metadata_path", None)
        cache_override = merged.pop("cache_dir", None)
        document_override = merged.pop("document_cache", None)

        cfg = cls(
            dataset_name=dataset_name,
            split=split,
            allowed_languages=allowed,
            max_lines=max_lines,
            chunk_overlap=chunk_overlap,
            vector_dim=vector_dim,
            vector_backend=backend,
            vector_metric=metric,
            batch_size=batch_size,
            streaming_enabled=streaming_enabled,
            token=merged.pop("token", None) or _lookup_hf_token(),
        )

        if cfg.max_lines < 1:
            cfg.max_lines = 1
        if cfg.chunk_overlap < 0:
            cfg.chunk_overlap = 0
        if cfg.chunk_overlap >= cfg.max_lines:
            cfg.chunk_overlap = max(cfg.max_lines - 1, 0)

        if vector_path_env:
            cfg.vector_store_path = Path(vector_path_env)
        elif vector_override:
            cfg.vector_store_path = Path(vector_override)
        if metadata_path_env:
            cfg.metadata_path = Path(metadata_path_env)
        elif metadata_override:
            cfg.metadata_path = Path(metadata_override)
        if cache_dir_env:
            cfg.cache_dir = Path(cache_dir_env)
        elif cache_override:
            cfg.cache_dir = Path(cache_override)
        if document_cache_env:
            cfg.document_cache = Path(document_cache_env)
        elif document_override:
            cfg.document_cache = Path(document_override)

        for field_name, value in merged.items():
            if hasattr(cfg, field_name):
                setattr(cfg, field_name, value)
        cfg.vector_store_path = Path(cfg.vector_store_path)
        cfg.metadata_path = Path(cfg.metadata_path)
        cfg.cache_dir = Path(cfg.cache_dir)
        cfg.document_cache = Path(cfg.document_cache)
        cfg.vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        cfg.document_cache.mkdir(parents=True, exist_ok=True)
        return cfg


@dataclass(slots=True)
class StackRecord:
    repo: str
    path: str
    language: str
    license: str
    content: str
    identifier: str


@dataclass(slots=True)
class StackChunk:
    repo: str
    path: str
    language: str
    license: str
    start_line: int
    end_line: int
    vector_id: str
    summary_hash: str
    text: str
    snippet_pointer: str | None = None

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
                    license TEXT DEFAULT 'unknown',
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    summary_hash TEXT NOT NULL,
                    snippet_path TEXT DEFAULT '',
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
        self._ensure_column("license", "TEXT DEFAULT 'unknown'")
        self._ensure_column("summary_hash", "TEXT DEFAULT '' NOT NULL")
        self._ensure_column("snippet_path", "TEXT DEFAULT ''")

    def _ensure_column(self, column: str, definition: str) -> None:
        cur = self._conn.execute("PRAGMA table_info(chunks)")
        existing = {row[1] for row in cur.fetchall()}
        cur.close()
        if column not in existing:
            with self._lock, self._conn:
                self._conn.execute(f"ALTER TABLE chunks ADD COLUMN {column} {definition}")

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
                INSERT OR IGNORE INTO chunks (
                    checksum,
                    repo,
                    path,
                    language,
                    license,
                    start_line,
                    end_line,
                    summary_hash,
                    snippet_path
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.vector_id,
                    chunk.repo,
                    chunk.path,
                    chunk.language,
                    chunk.license,
                    chunk.start_line,
                    chunk.end_line,
                    chunk.summary_hash,
                    chunk.snippet_pointer or "",
                ),
            )
            if chunk.snippet_pointer:
                self._conn.execute(
                    "UPDATE chunks SET snippet_path = ? WHERE checksum = ?",
                    (chunk.snippet_pointer, chunk.vector_id),
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

    def clear_cursor(self, split: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM cursors WHERE split = ?", (split,))

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
        self._snippet_cache = StackSnippetCache(self.config.document_cache)
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
            throughput = (embedded / elapsed) if elapsed > 0 else 0.0
            logger.info(
                "stack ingestion iteration processed %s chunks (skipped=%s, files=%s) in %.2fs (%.2f chunks/s)",
                embedded,
                self.metrics.chunks_skipped,
                self.metrics.files_seen,
                elapsed,
                throughput,
            )
            self._log_throughput_warning(throughput)
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
        return create_vector_store(
            dim=config.vector_dim,
            path=config.vector_store_path,
            backend=config.vector_backend,
            metric=config.vector_metric,
        )

    def _is_streaming_enabled(self) -> bool:
        if not self.config.streaming_enabled:
            return False
        return _env_flag("STACK_STREAMING", True)

    def _log_throughput_warning(self, throughput: float) -> None:
        threshold_raw = os.environ.get("STACK_THROUGHPUT_WARN")
        if not threshold_raw:
            return
        try:
            threshold = float(threshold_raw)
        except ValueError:
            logger.debug("invalid STACK_THROUGHPUT_WARN value; ignoring")
            return
        if throughput < threshold:
            logger.warning(
                "stack ingestion throughput %.2f chunks/s below threshold %.2f",
                throughput,
                threshold,
            )

    def _process_once(self, *, limit: int | None = None) -> int:
        dataset_iter = self._load_dataset_with_retry()
        embedded = 0
        resume_cursor = self.metadata_store.last_cursor(self.config.split)
        cursor_seen = resume_cursor is None or resume_cursor == ""
        cursor_found = cursor_seen
        for index, record in enumerate(dataset_iter):
            identifier = self._extract_identifier(record, position=index)
            if not cursor_seen and identifier == resume_cursor:
                cursor_seen = True
                cursor_found = True
                self.metadata_store.update_cursor(self.config.split, identifier)
                continue
            stack_record = self._coerce_record(record, position=index, identifier=identifier)
            if stack_record is None:
                self.metrics.files_skipped += 1
                if identifier:
                    self.metadata_store.update_cursor(self.config.split, identifier)
                continue
            self.metrics.files_seen += 1
            try:
                embedded += self._process_record(stack_record, limit=limit, so_far=embedded)
            except Exception:  # pragma: no cover - unexpected runtime failure
                self.metrics.errors += 1
                logger.exception(
                    "failed to process stack record %s:%s", stack_record.repo, stack_record.path
                )
            self.metadata_store.update_cursor(self.config.split, identifier)
            if limit is not None and embedded >= limit:
                break
        if resume_cursor and not cursor_found:
            logger.warning("stack resume cursor %s not found; resetting", resume_cursor)
            self.metadata_store.clear_cursor(self.config.split)
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

    def _extract_identifier(self, record: Dict[str, Any], *, position: int) -> str:
        repo = str(record.get("repo_name", record.get("repo", "unknown")))
        path_raw = str(record.get("path", ""))
        identifier = record.get("id")
        if identifier is None:
            identifier = f"{repo}:{path_raw}:{position}"
        return str(identifier)

    def _coerce_record(
        self, record: Dict[str, Any], *, position: int, identifier: str
    ) -> StackRecord | None:
        content = record.get("content")
        language = str(record.get("language", "")).strip()
        license_name = str(record.get("license", "unknown")).strip() or "unknown"
        if not content or not isinstance(content, str):
            return None
        if self.config.allowed_languages and language not in self.config.allowed_languages:
            return None
        path_raw = str(record.get("path", ""))
        repo = str(record.get("repo_name", record.get("repo", "unknown")))
        path = self._normalise_path(path_raw)
        return StackRecord(
            repo=repo,
            path=path,
            language=language or "unknown",
            license=license_name,
            content=content,
            identifier=str(identifier),
        )

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
            if self.metadata_store.has_chunk(chunk.vector_id):
                self.metrics.chunks_skipped += 1
                continue
            pointer = self._embed_chunk(chunk)
            if pointer:
                chunk.snippet_pointer = pointer
            self.metadata_store.record_chunk(chunk)
            chunk.release()
            embedded += 1
            self.metrics.chunks_embedded += 1
            if limit is not None and so_far + embedded >= limit:
                break
        record.content = ""
        return embedded

    def _chunk_record(self, record: StackRecord) -> Iterator[StackChunk]:
        lines = record.content.splitlines()
        if not lines:
            return
        max_lines = max(self.config.max_lines, 1)
        overlap = max(min(self.config.chunk_overlap, max_lines - 1), 0)
        step = max(max_lines - overlap, 1)
        for start in range(0, len(lines), step):
            chunk_lines = lines[start : start + max_lines]
            text = "\n".join(chunk_lines).strip()
            if not text:
                continue
            start_line = start + 1
            end_line = start + len(chunk_lines)
            vector_id = hashlib.sha1(
                f"{record.repo}:{record.path}:{start_line}:{end_line}:{text}".encode("utf-8")
            ).hexdigest()
            summary_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
            yield StackChunk(
                repo=record.repo,
                path=record.path,
                language=record.language,
                license=record.license,
                start_line=start_line,
                end_line=end_line,
                vector_id=vector_id,
                summary_hash=summary_hash,
                text=text,
            )

    def _embed_chunk(self, chunk: StackChunk) -> str | None:
        metadata = {
            "path": chunk.path,
            "repo": chunk.repo,
            "language": chunk.language,
            "license": chunk.license,
            "summary_hash": chunk.summary_hash,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        }
        record = {"text": chunk.text, "language": chunk.language, "path": chunk.path}
        self.vector_service.vectorise_and_store(
            "code",
            chunk.vector_id,
            record,
            origin_db="stack",
            metadata=metadata,
        )
        pointer: str | None = None
        if hasattr(self, "_snippet_cache") and self._snippet_cache is not None:
            try:
                snippet, pointer = self._snippet_cache.store(
                    chunk.summary_hash, chunk.text
                )
            except Exception:
                logger.exception("failed to persist stack snippet for %s", chunk.vector_id)
                pointer = None
        return pointer


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
    "ensure_background_task",
    "run_stack_ingestion_async",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
