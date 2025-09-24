"""Mixin providing embedding storage and vector search backends.

This module offers :class:`EmbeddableDBMixin` which can be mixed into a
class managing a SQLite database.  The mixin stores embedding vectors in an
Annoy or FAISS index on disk and keeps companion metadata in a JSON file.  A
 lazily loaded `SentenceTransformer` model is provided for text-to-vector
encoding, allowing subclasses to embed arbitrary records.

Subclasses must provide a ``self.conn`` database connection and override
:meth:`vector` to return an embedding for a record.  To support
:meth:`backfill_embeddings`, subclasses should also implement
:meth:`iter_records` yielding ``(record_id, record, kind)`` tuples.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_PACKAGE_NAME = "menace_sandbox"


def _bootstrap_package() -> None:
    """Ensure the package layout is available when executed as a script."""

    this_file = Path(__file__).resolve()
    package_root = this_file.parent
    repo_root = package_root.parent

    module = sys.modules.get(__name__)
    if module is None:  # pragma: no cover - defensive fallback
        module = ModuleType(__name__)
        module.__file__ = str(this_file)
        sys.modules[__name__] = module

    if __package__ in {None, ""}:  # pragma: no cover - script execution path
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        globals()["__package__"] = _PACKAGE_NAME

    pkg_module = sys.modules.get(_PACKAGE_NAME)
    if pkg_module is None:
        try:
            pkg_module = importlib.import_module(_PACKAGE_NAME)
        except ModuleNotFoundError:
            spec = importlib.util.spec_from_file_location(
                _PACKAGE_NAME,
                package_root / "__init__.py",
                submodule_search_locations=[str(package_root)],
            )
            if spec and spec.loader:
                pkg_module = importlib.util.module_from_spec(spec)
                sys.modules[_PACKAGE_NAME] = pkg_module
                spec.loader.exec_module(pkg_module)
            else:  # pragma: no cover - fallback when spec cannot be created
                pkg_module = ModuleType(_PACKAGE_NAME)
                pkg_module.__file__ = str(package_root / "__init__.py")
                pkg_module.__path__ = [str(package_root)]
                sys.modules[_PACKAGE_NAME] = pkg_module
        else:
            pkg_module = sys.modules[_PACKAGE_NAME]

    pkg_root_str = str(package_root)
    pkg_path = getattr(pkg_module, "__path__", None)
    if pkg_path is None:
        pkg_module.__path__ = [pkg_root_str]
    else:
        try:
            existing_paths = list(pkg_path)
        except TypeError:  # pragma: no cover - exotic path container
            pkg_module.__path__ = [pkg_root_str]
        else:
            if pkg_root_str not in existing_paths:
                try:
                    pkg_path.insert(0, pkg_root_str)
                except Exception:  # pragma: no cover - immutable path container
                    pkg_module.__path__ = [pkg_root_str, *existing_paths]

    sys.modules.setdefault(f"{_PACKAGE_NAME}.embeddable_db_mixin", module)
    sys.modules.setdefault("embeddable_db_mixin", module)


_bootstrap_package()

_MODULE_CACHE: dict[str, ModuleType] = {}


def _load_internal_module(name: str) -> ModuleType:
    """Import ``name`` from this package falling back to flat modules."""

    cached = _MODULE_CACHE.get(name)
    if cached is not None:
        return cached

    qualified = f"{_PACKAGE_NAME}.{name}"
    try:
        module = importlib.import_module(qualified)
    except ModuleNotFoundError as primary_exc:
        if primary_exc.name not in {qualified, _PACKAGE_NAME}:
            raise
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as secondary_exc:
            raise secondary_exc from primary_exc

    _MODULE_CACHE[name] = module
    return module

from datetime import datetime
from time import perf_counter
from typing import Any, Dict, Iterator, List, Sequence, Tuple
import hashlib
import json
import logging
import re
from security.secret_redactor import redact
from analysis.semantic_diff_filter import find_semantic_risks
from governed_embeddings import governed_embed
from chunking import split_into_chunks, summarize_snippet
from vector_service.text_preprocessor import generalise, PreprocessingConfig

# Lightweight license detection based on SPDX‑style fingerprints.  This avoids
# embedding content that is under GPL or non‑commercial restrictions.
from compliance.license_fingerprint import (
    check as license_check,
    fingerprint as license_fingerprint,
)

try:  # pragma: no cover - optional dependency
    from annoy import AnnoyIndex
except Exception:  # pragma: no cover - Annoy not installed
    AnnoyIndex = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except Exception:  # pragma: no cover - FAISS not installed
    faiss = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - NumPy not installed
    np = None  # type: ignore

_metrics_exporter = _load_internal_module("metrics_exporter")
_EMBED_STORE_LAST = _metrics_exporter.embedding_store_latency_seconds
_EMBED_STORE_TOTAL = _metrics_exporter.embedding_store_seconds_total
_EMBED_STALE = _metrics_exporter.embedding_stale_cost_seconds
_EMBED_TOKENS = _metrics_exporter.embedding_tokens_total
_EMBED_WALL_TOTAL = _metrics_exporter.embedding_wall_seconds_total
_EMBED_WALL_LAST = _metrics_exporter.embedding_wall_time_seconds

_vector_metrics_db = _load_internal_module("vector_metrics_db")
VectorMetricsDB = _vector_metrics_db.VectorMetricsDB

_embedding_stats_db = _load_internal_module("embedding_stats_db")
EmbeddingStatsDB = _embedding_stats_db.EmbeddingStatsDB

_data_bot = _load_internal_module("data_bot")
MetricsDB = _data_bot.MetricsDB

logger = logging.getLogger(__name__)


_VEC_METRICS = VectorMetricsDB()
_EMBED_STATS_DB = EmbeddingStatsDB("metrics.db")


def log_embedding_metrics(
    db_name: str,
    tokens: int,
    wall_time: float,
    store_latency: float,
    *,
    vector_id: str = "",
) -> None:
    """Log embedding metrics to Prometheus and persistent storage."""

    try:
        _EMBED_TOKENS.inc(tokens)
        _EMBED_WALL_TOTAL.inc(wall_time)
        _EMBED_STORE_TOTAL.inc(store_latency)
        _EMBED_WALL_LAST.set(wall_time)
        _EMBED_STORE_LAST.set(store_latency)
    except Exception:  # pragma: no cover - best effort
        pass
    try:
        _EMBED_STATS_DB.log(
            db_name=db_name,
            tokens=tokens,
            wall_ms=wall_time * 1000,
            store_ms=store_latency * 1000,
        )
        _VEC_METRICS.log_embedding(
            db=db_name,
            tokens=tokens,
            wall_time_ms=wall_time * 1000,
            store_time_ms=store_latency * 1000,
            vector_id=vector_id,
        )
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed to persist embedding metrics")


class EmbeddableDBMixin:
    """Add embedding storage and similarity search to a database class."""

    def __init__(
        self,
        *,
        index_path: str | Path = "embeddings.ann",
        metadata_path: str | Path = "embeddings.json",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_version: int = 1,
        backend: str = "annoy",
    ) -> None:
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        if metadata_path.name == "embeddings.json" and index_path.name != "embeddings.ann":
            # Derive metadata file alongside the provided index path to avoid
            # cross-database interference when tests supply unique index files
            metadata_path = index_path.with_suffix(".json")
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name
        self.embedding_version = embedding_version
        self.backend = backend

        self._model = None
        self._index: Any | None = None
        self._vector_dim = 0
        self._id_map: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._last_embedding_tokens = 0
        self._last_embedding_time = 0.0
        self._last_chunk_meta: Dict[str, Any] = {}

        self.load_index()

    # ------------------------------------------------------------------
    # model helpers
    @property
    def model(self):
        """Lazily loaded `SentenceTransformer` instance."""
        if self._model is None:  # pragma: no cover - heavy dependency
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode_text(self, text: str) -> List[float]:
        """Encode ``text`` using the SentenceTransformer model."""

        start = perf_counter()
        vec = governed_embed(text, self.model)
        self._last_embedding_time = perf_counter() - start
        tokens = 0
        if vec is not None:
            try:  # pragma: no cover - optional dependency
                tokenizer = getattr(self.model, "tokenizer", None)
                if tokenizer:
                    tokens = len(tokenizer.encode(redact(text)))
            except Exception:
                tokens = 0
        self._last_embedding_tokens = tokens
        return vec or []

    def _split_and_summarise(
        self,
        text: str,
        *,
        config: "PreprocessingConfig" | None = None,
        db_key: str | None = None,
    ) -> str:
        """Split ``text`` into sentences, filter, chunk and summarise.

        ``config`` may override the behaviour for sentence splitting,
        chunk sizes and semantic risk filtering.  When ``config`` is ``None``
        a configuration registered for ``db_key`` will be used if available.
        The resulting condensed text is returned while ``self._last_chunk_meta``
        records ``chunk_count`` and ``chunk_hashes`` for traceability.
        """

        from vector_service.text_preprocessor import get_config

        cfg = config or get_config(db_key or self.__class__.__name__.lower())

        if not isinstance(text, str):
            self._last_chunk_meta = {"chunk_count": 0, "chunk_hashes": []}
            return text

        if cfg.split_sentences:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        else:
            sentences = [text]
        if not sentences:
            self._last_chunk_meta = {"chunk_count": 0, "chunk_hashes": []}
            return ""

        alerts = find_semantic_risks(sentences) if cfg.filter_semantic_risks else []
        risky = {line for line, _, _ in alerts}
        filtered = [s for s in sentences if s not in risky]
        joined = "\n".join(filtered)

        size = cfg.chunk_size or 400
        try:
            chunks = split_into_chunks(joined, size)
        except Exception:  # pragma: no cover - fallback if chunking fails
            chunks = []
            if joined:
                chunks.append(type("C", (), {"text": joined})())

        # ``ContextBuilder`` is a heavy dependency so we import lazily and fall
        # back to a tiny stub when unavailable.  The builder is cached on the
        # instance to avoid repeated initialisation across calls.
        builder = getattr(self, "_summary_builder", None)
        if builder is None:
            try:  # pragma: no cover - best effort builder creation
                from context_builder_util import create_context_builder

                builder = create_context_builder()
            except Exception as exc:  # pragma: no cover - builder unavailable
                logger.warning("context builder unavailable: %s", exc)

                class _DummyBuilder:
                    def build(self, _: str) -> str:
                        return ""

                builder = _DummyBuilder()
            self._summary_builder = builder

        # Try to initialise a lightweight local LLM client.  This is optional
        # and failures are logged but otherwise ignored.
        llm = getattr(self, "_summary_llm", None)
        if llm is None:
            try:  # pragma: no cover - optional dependency
                from local_client import OllamaClient

                llm = OllamaClient()
            except Exception:
                try:  # pragma: no cover - secondary fallback
                    from local_client import VLLMClient

                    llm = VLLMClient()
                except Exception as exc:
                    logger.debug("no local LLM available: %s", exc)
                    llm = None
            self._summary_llm = llm

        summaries: List[str] = []
        chunk_hashes: List[str] = []
        for ch in chunks:
            digest = hashlib.sha256(ch.text.encode("utf-8")).hexdigest()
            chunk_hashes.append(digest)
            try:
                summary = summarize_snippet(ch.text, llm, context_builder=builder)
                if not summary:
                    raise ValueError("empty summary")
            except Exception as exc:  # pragma: no cover - summariser issues
                logger.exception("summary generation failed for %s", digest, exc_info=exc)
                summary = ch.text
            summary = generalise(summary, config=cfg, db_key=db_key)
            if summary:
                summaries.append(summary)

        self._last_chunk_meta = {
            "chunk_count": len(chunks),
            "chunk_hashes": chunk_hashes,
        }
        return " ".join(s for s in summaries if s)

    def _prepare_text_for_embedding(
        self,
        text: str,
        *,
        chunk_tokens: int | None = None,
        config: "PreprocessingConfig" | None = None,
        db_key: str | None = None,
    ) -> str:
        """Backward compatible wrapper for older callers.

        Older callers may still pass ``chunk_tokens`` to specify the desired
        chunk size.  Newer code should pass a :class:`PreprocessingConfig`
        instance to fully control splitting behaviour.
        """

        if config is None and chunk_tokens is not None:
            config = PreprocessingConfig(chunk_size=chunk_tokens)
        return self._split_and_summarise(text, config=config, db_key=db_key)

    def _extract_last_updated(self, record: Any) -> str | None:
        """Best-effort extraction of a last-updated timestamp from ``record``.

        Many database records expose their modification time under different
        keys.  This helper normalises a handful of common field names and
        returns an ISO formatted string when found.  Subclasses can override
        this method for custom behaviour.
        """

        if isinstance(record, dict):
            for key in (
                "last_updated",
                "last_modification_date",
                "updated_at",
                "updated",
                "modified_at",
                "modified",
            ):
                val = record.get(key)
                if not val:
                    continue
                if isinstance(val, datetime):
                    return val.isoformat()
                try:
                    return str(val)
                except Exception:  # pragma: no cover - defensive
                    return None
        return None

    # ------------------------------------------------------------------
    # methods expected to be overridden
    def vector(self, record: Any) -> List[float]:
        """Return an embedding vector for ``record``.

        Textual records are expected to be preprocessed via
        :meth:`_split_and_summarise` before being passed here. Subclasses can
        override this for non-textual records.
        """

        if isinstance(record, str):
            return self.encode_text(record)
        raise NotImplementedError

    def iter_records(self) -> Iterator[Tuple[Any, Any, str]]:
        """Yield ``(record_id, record, kind)`` tuples for backfilling.

        Override in subclasses that use :meth:`backfill_embeddings`.
        """

        raise NotImplementedError

    def license_text(self, record: Any) -> str | None:
        """Return textual content to scan for license violations.

        Subclasses can override this to extract text from structured
        records. By default, if ``record`` is a string it is returned as-is
        otherwise ``None`` is returned, skipping the license check.
        """

        return record if isinstance(record, str) else None

    # ------------------------------------------------------------------
    # index persistence
    def load_index(self) -> None:
        """Load the vector index and metadata from disk if available."""
        if self.metadata_path.exists():
            data = json.loads(self.metadata_path.read_text())
            self._id_map = data.get("id_map", [])
            self._metadata = data.get("metadata", {})
            self._vector_dim = data.get("vector_dim", 0)
            if not self._id_map:
                self._id_map = list(self._metadata.keys())
        if self.backend == "annoy":
            if AnnoyIndex and self.index_path.exists() and self._vector_dim:
                self._index = AnnoyIndex(self._vector_dim, "angular")
                self._index.load(str(self.index_path))
            elif AnnoyIndex and self._metadata:
                self._rebuild_index()
        elif self.backend == "faiss":
            if faiss and self.index_path.exists() and self._vector_dim:
                self._index = faiss.read_index(str(self.index_path))
            elif faiss and self._metadata:
                self._rebuild_index()

    def save_index(self) -> None:
        """Persist vector index and metadata to disk."""
        if self._index is None:
            return
        if self.backend == "annoy":
            if not AnnoyIndex:
                return
            self._index.save(str(self.index_path))
        elif self.backend == "faiss":
            if not faiss:
                return
            faiss.write_index(self._index, str(self.index_path))
        data = {
            "id_map": self._id_map,
            "metadata": self._metadata,
            "vector_dim": self._vector_dim,
        }
        self.metadata_path.write_text(json.dumps(data, indent=2))

    def _rebuild_index(self) -> None:
        """Rebuild vector index from stored metadata."""
        if not self._metadata:
            self._index = None
            return
        self._vector_dim = len(next(iter(self._metadata.values()))["vector"])
        if self.backend == "annoy":
            if not AnnoyIndex:
                self._index = None
                return
            self._index = AnnoyIndex(self._vector_dim, "angular")
            for i, rid in enumerate(self._id_map):
                vec = self._metadata[rid]["vector"]
                self._index.add_item(i, vec)
            self._index.build(10)
        elif self.backend == "faiss":
            if not faiss or not np:
                self._index = None
                return
            self._index = faiss.IndexFlatIP(self._vector_dim)
            vectors = [self._metadata[rid]["vector"] for rid in self._id_map]
            if vectors:
                arr = np.array(vectors, dtype="float32")
                self._index.add(arr)

    # ------------------------------------------------------------------
    # public API
    def add_embedding(
        self,
        record_id: Any,
        record: Any,
        kind: str,
        *,
        source_id: str = "",
        chunk_meta: Dict[str, Any] | None = None,
    ) -> None:
        """Embed ``record`` and store the vector and metadata."""
        last_updated = self._extract_last_updated(record)
        text = self.license_text(record)
        if text is None and isinstance(record, str):
            text = record
        if text:
            lic = license_check(text)
            if lic:
                try:  # pragma: no cover - best effort
                    hash_ = license_fingerprint(text)
                    log_fn = getattr(self, "log_license_violation", None)
                    if callable(log_fn):
                        log_fn("", lic, hash_)
                except Exception:  # pragma: no cover - best effort
                    logger.exception(
                        "failed to log license violation for %s", record_id
                    )
                rid = str(record_id)
                self._metadata[rid] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "embedding_version": self.embedding_version,
                    "kind": kind,
                    "source_id": source_id,
                    "redacted": False,
                    "license": lic,
                }
                if last_updated:
                    self._metadata[rid]["last_updated"] = last_updated
                log_embedding_metrics(
                    self.__class__.__name__, 0, 0.0, 0.0, vector_id=str(record_id)
                )
                logger.warning(
                    "skipping embedding for %s due to license %s", record_id, lic
                )
                return
            alerts = find_semantic_risks(text.splitlines())
            if alerts:
                rid = str(record_id)
                self._metadata[rid] = {
                    "created_at": datetime.utcnow().isoformat(),
                    "embedding_version": self.embedding_version,
                    "kind": kind,
                    "source_id": source_id,
                    "redacted": False,
                    "semantic_risks": alerts,
                }
                if last_updated:
                    self._metadata[rid]["last_updated"] = last_updated
                log_embedding_metrics(
                    self.__class__.__name__, 0, 0.0, 0.0, vector_id=str(record_id)
                )
                logger.warning(
                    "skipping embedding for %s due to semantic risks", record_id
                )
                for line, msg, score in alerts:
                    logger.warning("semantic risk %.2f for %s: %s", score, line, msg)
                return
        record = redact(record) if isinstance(record, str) else record
        if isinstance(record, str):
            if chunk_meta is None:
                record = self._split_and_summarise(record)
                chunk_meta = getattr(self, "_last_chunk_meta", {})
        if chunk_meta is None:
            chunk_meta = {"chunk_count": 0, "chunk_hashes": []}

        start = perf_counter()
        vec = self.vector(record)
        chunk_meta = getattr(self, "_last_chunk_meta", chunk_meta)
        wall_time = perf_counter() - start
        tokens = getattr(self, "_last_embedding_tokens", 0)
        if not tokens and isinstance(record, str):  # pragma: no cover - best effort
            try:
                tokenizer = getattr(self.model, "tokenizer", None)
                if tokenizer:
                    tokens = len(tokenizer.encode(record))
            except Exception:  # pragma: no cover - best effort
                tokens = 0
        self._last_embedding_tokens = tokens
        self._last_embedding_time = wall_time

        rid = str(record_id)
        if rid not in self._metadata:
            self._id_map.append(rid)
        self._metadata[rid] = {
            "vector": list(vec),
            "created_at": datetime.utcnow().isoformat(),
            "embedding_version": self.embedding_version,
            "kind": kind,
            "source_id": source_id,
            "redacted": True,
            "record": record,
            **(chunk_meta or {}),
            "_last_chunk_meta": chunk_meta or {},
        }
        if last_updated:
            self._metadata[rid]["last_updated"] = last_updated
        self._rebuild_index()
        save_start = perf_counter()
        self.save_index()
        index_latency = perf_counter() - save_start
        log_embedding_metrics(
            self.__class__.__name__,
            tokens,
            wall_time,
            index_latency,
            vector_id=str(record_id),
        )

    def get_vector(self, record_id: Any) -> List[float] | None:
        """Return the stored embedding vector for ``record_id`` if present."""

        meta = self._metadata.get(str(record_id))
        if meta:
            self._record_staleness(str(record_id), meta.get("created_at"))
            return list(meta["vector"])
        return None

    def try_add_embedding(
        self,
        record_id: Any,
        record: Any,
        kind: str,
        *,
        source_id: str = "",
    ) -> None:
        """Best-effort variant of :meth:`add_embedding`."""

        try:
            self.add_embedding(record_id, record, kind, source_id=source_id)
        except Exception:  # pragma: no cover - best effort
            logging.exception("embedding hook failed for %s", record_id)

    def update_embedding_version(
        self, record_id: Any, *, embedding_version: int | None = None
    ) -> None:
        """Update ``embedding_version`` metadata for ``record_id``."""

        rid = str(record_id)
        if rid not in self._metadata:
            return
        version = (
            embedding_version if embedding_version is not None else self.embedding_version
        )
        self._metadata[rid]["embedding_version"] = int(version)
        self._metadata[rid]["created_at"] = datetime.utcnow().isoformat()
        self.save_index()

    def update_embedding_versions(
        self, record_ids: Sequence[Any], *, embedding_version: int | None = None
    ) -> None:
        """Bulk update ``embedding_version`` for multiple records."""

        version = (
            embedding_version if embedding_version is not None else self.embedding_version
        )
        now = datetime.utcnow().isoformat()
        updated = False
        for rid in map(str, record_ids):
            if rid in self._metadata:
                self._metadata[rid]["embedding_version"] = int(version)
                self._metadata[rid]["created_at"] = now
                updated = True
        if updated:
            self.save_index()

    def needs_refresh(self, record_id: Any, record: Any | None = None) -> bool:
        """Return ``True`` if ``record_id`` requires re-embedding.

        A record is considered stale when no metadata is stored, the
        ``embedding_version`` has changed, or the supplied ``record`` carries a
        different ``last_updated`` timestamp to that stored in metadata.  When
        ``record`` is ``None`` the check falls back to version mismatches only.
        """

        rid = str(record_id)
        meta = self._metadata.get(rid)
        if not meta:
            return True
        try:
            if int(meta.get("embedding_version", 0)) != int(self.embedding_version):
                return True
        except Exception:
            return True
        if record is not None:
            current = self._extract_last_updated(record)
            stored = meta.get("last_updated")
            if current and stored != current:
                return True
        return False

    # internal ---------------------------------------------------------
    def _record_staleness(self, rid: str, created_at: str | None) -> None:
        """Log how stale an embedding is when accessed."""
        if not created_at:
            return
        try:
            age = (datetime.utcnow() - datetime.fromisoformat(created_at)).total_seconds()
        except Exception:
            return
        origin = getattr(self, "origin_db", self.__class__.__name__)
        if _EMBED_STALE:
            try:
                _EMBED_STALE.labels(origin_db=origin).set(age)
            except ValueError:  # pragma: no cover - labels not configured
                _EMBED_STALE.set(age)
        try:
            MetricsDB().log_embedding_staleness(origin, rid, age)
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to persist embedding staleness")

    def search_by_vector(
        self, vector: Sequence[float], top_k: int = 10
    ) -> List[Tuple[Any, float]]:
        """Return ``top_k`` records most similar to ``vector``."""

        if self._index is None:
            self.load_index()
        if self._index is None:
            return []
        if self.backend == "annoy":
            ids, dists = self._index.get_nns_by_vector(
                list(vector), top_k, include_distances=True
            )
            results: List[Tuple[Any, float]] = []
            for i, d in zip(ids, dists):
                if i < len(self._id_map):
                    rid = self._id_map[i]
                    meta = self._metadata.get(rid)
                    if not meta or not meta.get("redacted"):
                        continue
                    results.append((rid, float(d)))
                    self._record_staleness(rid, meta.get("created_at"))
            return results
        elif self.backend == "faiss":
            if not faiss or not np:
                return []
            vec = np.array([list(vector)], dtype="float32")
            dists, ids = self._index.search(vec, top_k)
            results: List[Tuple[Any, float]] = []
            for idx, dist in zip(ids[0], dists[0]):
                if 0 <= idx < len(self._id_map):
                    rid = self._id_map[idx]
                    meta = self._metadata.get(rid)
                    if not meta or not meta.get("redacted"):
                        continue
                    results.append((rid, float(dist)))
                    self._record_staleness(rid, meta.get("created_at"))
            return results
        return []

    def backfill_embeddings(self) -> None:
        """Generate embeddings for all records lacking them."""
        for record_id, record, kind in self.iter_records():
            rid = str(record_id)
            if not self.needs_refresh(record_id, record):
                continue
            text = self.license_text(record)
            if text is None and isinstance(record, str):
                text = record
            if text:
                lic = license_check(text)
                if lic:
                    hash_ = license_fingerprint(text)
                    log_fn = getattr(self, "log_license_violation", None)
                    if callable(log_fn):
                        try:  # pragma: no cover - best effort
                            log_fn("", lic, hash_)
                        except Exception:  # pragma: no cover - best effort
                            logger.exception(
                                "failed to log license violation for %s", record_id
                            )
                    self._metadata[rid] = {
                        "created_at": datetime.utcnow().isoformat(),
                        "embedding_version": self.embedding_version,
                        "kind": kind,
                        "source_id": "",
                        "redacted": False,
                        "license": lic,
                    }
                    last_updated = self._extract_last_updated(record)
                    if last_updated:
                        self._metadata[rid]["last_updated"] = last_updated
                    logger.warning(
                        "skipping embedding for %s due to license %s", record_id, lic
                    )
                    continue
                alerts = find_semantic_risks(text.splitlines())
                if alerts:
                    self._metadata[rid] = {
                        "created_at": datetime.utcnow().isoformat(),
                        "embedding_version": self.embedding_version,
                        "kind": kind,
                        "source_id": "",
                        "redacted": False,
                        "semantic_risks": alerts,
                    }
                    last_updated = self._extract_last_updated(record)
                    if last_updated:
                        self._metadata[rid]["last_updated"] = last_updated
                    logger.warning(
                        "skipping embedding for %s due to semantic risks", record_id
                    )
                    for line, msg, score in alerts:
                        logger.warning("semantic risk %.2f for %s: %s", score, line, msg)
                    continue
            record = redact(record) if isinstance(record, str) else record
            chunk_meta: Dict[str, Any] | None = None
            if isinstance(record, str):
                record = self._split_and_summarise(record)
                chunk_meta = getattr(self, "_last_chunk_meta", {})
            self.add_embedding(record_id, record, kind, chunk_meta=chunk_meta)
