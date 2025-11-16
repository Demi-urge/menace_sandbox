from __future__ import annotations

"""Retrieval helpers for Stack dataset embeddings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence

import contextlib
import logging
import math
import os
import sqlite3

from dynamic_path_router import resolve_path

from governed_retrieval import govern_retrieval, redact_dict
from redaction_utils import redact_dict as pii_redact_dict
from redaction_utils import redact_text as pii_redact_text
from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST

from patch_safety import PatchSafety

try:  # pragma: no cover - optional dependency during import
    from .stack_snippet_cache import StackSnippetCache
    from .vector_store import VectorStore, create_vector_store
except Exception:  # pragma: no cover - fallback when relative import fails
    from vector_service.stack_snippet_cache import StackSnippetCache  # type: ignore
    from vector_service.vector_store import VectorStore, create_vector_store  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .vectorizer import SharedVectorService as SharedVectorServiceType
else:  # pragma: no cover - runtime fallback
    SharedVectorServiceType = Any  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

_DEFAULT_LICENSE_DENYLIST = set(_LICENSE_DENYLIST.values())


def _normalise_path(path: str | None) -> str:
    if not path:
        return "unknown"
    normalised = path.replace("\\", "/")
    normalised = normalised.strip()
    normalised = normalised.lstrip("./")
    while normalised.startswith("/"):
        normalised = normalised[1:]
    return normalised or "unknown"


def _to_float_list(values: Sequence[Any]) -> List[float]:
    return [float(v) for v in values]


@dataclass
class StackRetrieverConfig:
    """Runtime configuration used when constructing the retriever."""

    vector_path: Path = field(
        default_factory=lambda: resolve_path("vector_service") / "stack_vectors"
    )
    metadata_path: Path = field(
        default_factory=lambda: resolve_path("vector_service") / "stack_metadata.db"
    )
    document_cache: Path = field(
        default_factory=lambda: resolve_path("chunk_summary_cache") / "stack_documents"
    )
    backend: str = "annoy"
    metric: str = "angular"
    vector_dim: int = 768
    similarity_metric: str = "cosine"
    similarity_threshold: float = 0.0

    @classmethod
    def from_environment(cls, **overrides: Any) -> "StackRetrieverConfig":
        vector_path_override = overrides.pop("vector_path", None)
        metadata_path_override = overrides.pop("metadata_path", None)
        document_cache_override = overrides.pop("document_cache", None)
        backend_override = overrides.pop("backend", None)
        metric_override = overrides.pop("metric", None)
        similarity_metric_override = overrides.pop("similarity_metric", None)
        threshold_override = overrides.pop("similarity_threshold", None)
        dim_override = overrides.pop("vector_dim", overrides.pop("dim", None))

        env_vector_path = os.environ.get("STACK_VECTOR_PATH")
        env_metadata_path = os.environ.get("STACK_METADATA_PATH")
        env_document_cache = os.environ.get("STACK_DOCUMENT_CACHE")
        env_backend = os.environ.get("STACK_VECTOR_BACKEND")
        env_metric = os.environ.get("STACK_VECTOR_METRIC")
        env_dim = os.environ.get("STACK_VECTOR_DIM")

        if vector_path_override is not None:
            vector_path = Path(vector_path_override)
        elif env_vector_path:
            vector_path = Path(env_vector_path)
        else:
            vector_path = resolve_path("vector_service") / "stack_vectors"

        if metadata_path_override is not None:
            metadata_path = Path(metadata_path_override)
        elif env_metadata_path:
            metadata_path = Path(env_metadata_path)
        else:
            metadata_path = resolve_path("vector_service") / "stack_metadata.db"

        if document_cache_override is not None:
            document_cache = Path(document_cache_override)
        elif env_document_cache:
            document_cache = Path(env_document_cache)
        else:
            document_cache = (
                resolve_path("chunk_summary_cache") / "stack_documents"
            )

        backend = backend_override or env_backend or "annoy"
        metric = metric_override or env_metric or "angular"
        if env_dim is not None:
            try:
                vector_dim = int(env_dim)
            except Exception:
                vector_dim = int(dim_override or 768)
        else:
            vector_dim = int(dim_override or 768)
        similarity_metric = str(similarity_metric_override or "cosine")
        if threshold_override is None:
            threshold = 0.0
        else:
            threshold = float(threshold_override)

        return cls(
            vector_path=vector_path,
            metadata_path=metadata_path,
            document_cache=document_cache,
            backend=str(backend),
            metric=str(metric),
            vector_dim=int(vector_dim),
            similarity_metric=similarity_metric,
            similarity_threshold=float(threshold),
        )


@dataclass
class StackRetriever:
    """Query Stack dataset embeddings from a local :class:`VectorStore`."""

    vector_store: VectorStore | None = None
    vector_service: SharedVectorServiceType | None = None
    metadata_path: str | Path | None = None
    config: StackRetrieverConfig | None = None
    document_cache: str | Path | None = None
    top_k: int = 5
    similarity_metric: str = "cosine"
    similarity_threshold: float = 0.0
    max_alert_severity: float = 1.0
    max_alerts: int = 5
    license_denylist: set[str] = field(
        default_factory=lambda: set(_DEFAULT_LICENSE_DENYLIST)
    )
    roi_tag_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._settings = self.config or self._load_config()
        if self.metadata_path is None:
            self.metadata_path = self._settings.metadata_path
        else:
            self.metadata_path = Path(self.metadata_path)
        self.similarity_metric = self.similarity_metric or self._settings.similarity_metric
        if not self.similarity_threshold:
            self.similarity_threshold = self._settings.similarity_threshold
        self._vector_store: VectorStore | None = self.vector_store
        self._vector_service: SharedVectorServiceType | None = self.vector_service
        self._metadata_conn: sqlite3.Connection | None = None
        self._id_to_index: Dict[str, int] = {}
        self.patch_safety = PatchSafety()
        self.patch_safety.max_alert_severity = self.max_alert_severity
        self.patch_safety.max_alerts = self.max_alerts
        self.patch_safety.license_denylist = set(self.license_denylist)
        self._store_error_logged = False
        self._metadata_error_logged = False
        cache_path = self.document_cache or getattr(self._settings, "document_cache", None)
        if cache_path:
            self.document_cache = Path(cache_path)
            self.document_cache.mkdir(parents=True, exist_ok=True)
            self._snippet_cache: StackSnippetCache | None = StackSnippetCache(
                self.document_cache
            )
        else:
            self.document_cache = None
            self._snippet_cache = None
        if self._vector_store is not None:
            self._index_store_metadata()

    # ------------------------------------------------------------------
    def _load_config(self) -> StackRetrieverConfig:
        overrides: Dict[str, Any] = {}
        try:  # pragma: no cover - configuration optional
            from config import ContextBuilderConfig  # type: ignore

            cfg = ContextBuilderConfig()
            stack_cfg = getattr(cfg, "stack", None)
            if stack_cfg is not None:
                for attr in (
                    "vector_path",
                    "metadata_path",
                    "document_cache",
                    "backend",
                    "metric",
                    "vector_dim",
                    "similarity_metric",
                    "similarity_threshold",
                ):
                    if hasattr(stack_cfg, attr):
                        overrides[attr] = getattr(stack_cfg, attr)
                if hasattr(stack_cfg, "top_k") and isinstance(stack_cfg.top_k, int):
                    self.top_k = stack_cfg.top_k
        except Exception:
            pass
        return StackRetrieverConfig.from_environment(**overrides)

    # ------------------------------------------------------------------
    def _ensure_vector_service(self) -> SharedVectorServiceType | None:
        if self._vector_service is not None:
            return self._vector_service
        try:
            from .vectorizer import SharedVectorService as _SharedVectorService  # type: ignore
        except Exception:
            try:  # pragma: no cover - absolute import fallback
                from vector_service.vectorizer import SharedVectorService as _SharedVectorService  # type: ignore
            except Exception:
                logger.exception("failed to initialise SharedVectorService for stack retriever")
                self._vector_service = None
                return None
        try:
            self._vector_service = _SharedVectorService()
        except Exception:
            logger.exception("failed to initialise SharedVectorService for stack retriever")
            self._vector_service = None
        return self._vector_service

    # ------------------------------------------------------------------
    def _ensure_store(self) -> VectorStore | None:
        if self._vector_store is not None:
            return self._vector_store
        try:
            self._vector_store = create_vector_store(
                self._settings.vector_dim,
                self._settings.vector_path,
                backend=self._settings.backend,
                metric=self._settings.metric,
            )
        except Exception:
            if not self._store_error_logged:
                logger.exception("failed to initialise stack vector store")
                self._store_error_logged = True
            self._vector_store = None
            return None
        self._index_store_metadata()
        return self._vector_store

    # ------------------------------------------------------------------
    def _index_store_metadata(self) -> None:
        store = self._vector_store
        if store is None:
            return
        ids = getattr(store, "ids", None)
        if isinstance(ids, list):
            self._id_to_index = {str(identifier): idx for idx, identifier in enumerate(ids)}
        else:
            self._id_to_index = {}

    # ------------------------------------------------------------------
    def _ensure_metadata(self) -> sqlite3.Connection | None:
        if self._metadata_conn is not None:
            return self._metadata_conn
        if self.metadata_path is None:
            return None
        try:
            conn = sqlite3.connect(str(self.metadata_path))
            conn.row_factory = sqlite3.Row
            self._metadata_conn = conn
        except Exception:
            if not self._metadata_error_logged:
                logger.exception("failed to open stack metadata store")
                self._metadata_error_logged = True
            self._metadata_conn = None
        return self._metadata_conn

    # ------------------------------------------------------------------
    def warm_cache(self) -> bool:
        store = self._ensure_store()
        conn = self._ensure_metadata()
        return store is not None and conn is not None

    # ------------------------------------------------------------------
    def embed_query(self, query: str) -> List[float] | None:
        service = self._ensure_vector_service()
        if service is None:
            return None
        try:
            vec = service.vectorise("stack", {"text": query})
        except Exception:
            logger.exception("stack query embedding failed")
            return None
        if hasattr(vec, "tolist"):
            vec = vec.tolist()  # type: ignore[attr-defined]
        if not isinstance(vec, (list, tuple)):
            return None
        return _to_float_list(vec)

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        keywords: Iterable[str] | None = None,
        exclude_tags: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        embedding = self.embed_query(query)
        if embedding is None:
            return []
        return self.retrieve(
            embedding,
            k=k or self.top_k,
            keywords=keywords,
            exclude_tags=exclude_tags,
        )

    # ------------------------------------------------------------------
    def _lookup_metadata(self, checksum: str) -> Dict[str, Any]:
        conn = self._ensure_metadata()
        if conn is None:
            return {}
        try:
            cur = conn.execute(
                """
                SELECT repo, path, language, start_line, end_line, summary_hash, snippet_path
                FROM chunks WHERE checksum=?
                """,
                (checksum,),
            )
            row = cur.fetchone()
            cur.close()
        except Exception:
            logger.exception("failed to fetch stack metadata for checksum %s", checksum)
            return {}
        if row is None:
            return {}
        data = dict(row)
        start = data.get("start_line")
        end = data.get("end_line")
        if isinstance(start, int) and isinstance(end, int) and end >= start:
            data["size"] = int(end - start + 1)
        return data

    # ------------------------------------------------------------------
    def _similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        metric = (self.similarity_metric or "cosine").lower()
        if metric == "inner_product":
            return float(sum(x * y for x, y in zip(a, b)))
        if metric == "cosine":
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            if not na or not nb:
                return 0.0
            return float(sum(x * y for x, y in zip(a, b)) / (na * nb))
        raise ValueError(f"unsupported similarity metric: {self.similarity_metric}")

    # ------------------------------------------------------------------
    def _distance_to_similarity(self, distance: float) -> float:
        metric = (self._settings.metric or "angular").lower()
        if metric in {"angular", "cosine"}:
            return 1.0 / (1.0 + float(distance))
        if metric in {"l2", "euclidean"}:
            return 1.0 / (1.0 + float(distance))
        return 1.0 / (1.0 + float(distance))

    # ------------------------------------------------------------------
    def _extract_store_entry(
        self, vector_id: str
    ) -> tuple[List[float] | None, Mapping[str, Any], str, str]:
        store = self._ensure_store()
        if store is None:
            return None, {}, "stack", ""
        idx = self._id_to_index.get(vector_id)
        vectors = getattr(store, "vectors", [])
        meta_entries = getattr(store, "meta", [])
        text = ""
        origin = "stack"
        metadata: Mapping[str, Any] = {}
        vector: List[float] | None = None
        if isinstance(idx, int) and idx >= 0:
            if idx < len(vectors):
                vec = vectors[idx]
                if isinstance(vec, (list, tuple)):
                    vector = _to_float_list(vec)
                else:
                    vector = None
            else:
                vector = None
            if idx < len(meta_entries):
                entry = meta_entries[idx]
                if isinstance(entry, Mapping):
                    origin = str(entry.get("origin_db", "stack"))
                    metadata = entry.get("metadata") or {}
                    if isinstance(metadata, Mapping):
                        metadata = dict(metadata)
                    else:
                        metadata = {}
                    text = str(metadata.get("text") or metadata.get("snippet") or "")
        # Fallback when index lookup failed but vector is still accessible
        if vector is None and isinstance(vector_id, str):
            try:
                idx = getattr(store, "ids", []).index(vector_id)  # type: ignore[arg-type]
            except Exception:
                idx = -1
            if idx >= 0:
                if idx < len(vectors):
                    vec = vectors[idx]
                    if isinstance(vec, (list, tuple)):
                        vector = _to_float_list(vec)
                if not metadata and idx < len(meta_entries):
                    entry = meta_entries[idx]
                    if isinstance(entry, Mapping):
                        origin = str(entry.get("origin_db", "stack"))
                        raw_meta = entry.get("metadata") or {}
                        if isinstance(raw_meta, Mapping):
                            metadata = dict(raw_meta)
                        text = str(metadata.get("text") or metadata.get("snippet") or "")
        return vector, metadata, origin, text

    # ------------------------------------------------------------------
    def _should_skip(
        self,
        metadata: Mapping[str, Any],
        *,
        excluded_tags: set[str],
        keywords: List[str],
    ) -> bool:
        if excluded_tags:
            roi_tag = metadata.get("roi_tag")
            if roi_tag is not None and str(roi_tag) in excluded_tags:
                return True
            roi_tags = metadata.get("roi_tags")
            if isinstance(roi_tags, (list, set, tuple)):
                if any(str(tag) in excluded_tags for tag in roi_tags):
                    return True
        if keywords:
            haystacks = []
            for key in ("path", "repo", "language", "text", "summary"):
                value = metadata.get(key)
                if value:
                    haystacks.append(str(value).lower())
            if haystacks and not any(
                any(keyword in hay for hay in haystacks) for keyword in keywords
            ):
                return True
        return False

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query_embedding: Sequence[float],
        *,
        k: int | None = None,
        keywords: Iterable[str] | None = None,
        exclude_tags: Iterable[str] | None = None,
        similarity_threshold: float | None = None,
    ) -> List[Dict[str, Any]]:
        store = self._ensure_store()
        if store is None:
            logger.warning("stack vector store unavailable; returning no results")
            return []
        top_k = k or self.top_k
        try:
            hits = list(store.query(query_embedding, top_k=top_k))
        except Exception:
            logger.exception("stack vector query failed")
            return []
        threshold = (
            self.similarity_threshold if similarity_threshold is None else similarity_threshold
        )
        keyword_list = [str(k).lower() for k in keywords or [] if str(k).strip()]
        excluded = {str(tag) for tag in (exclude_tags or []) if str(tag)}
        results: List[Dict[str, Any]] = []
        for vector_id, distance in hits:
            candidate_vec, raw_meta, origin, text = self._extract_store_entry(str(vector_id))
            similarity = 0.0
            if candidate_vec is not None:
                similarity = self._similarity(query_embedding, candidate_vec)
            else:
                similarity = self._distance_to_similarity(distance)
            similarity = max(0.0, min(1.0, similarity if math.isfinite(similarity) else 0.0))
            if threshold and similarity < threshold:
                continue
            lookup_meta = self._lookup_metadata(str(vector_id))
            merged_meta: Dict[str, Any] = {}
            if isinstance(raw_meta, Mapping):
                merged_meta.update(raw_meta)
            merged_meta.update({k: v for k, v in lookup_meta.items() if v is not None})
            merged_meta.setdefault("path", merged_meta.get("file_path"))
            merged_meta["path"] = _normalise_path(str(merged_meta.get("path", "")))
            if "repo" in merged_meta:
                merged_meta["repo"] = str(merged_meta["repo"])
            merged_meta.setdefault("language", lookup_meta.get("language"))
            merged_meta.setdefault("start_line", lookup_meta.get("start_line"))
            merged_meta.setdefault("end_line", lookup_meta.get("end_line"))
            if lookup_meta.get("summary_hash") and "summary_hash" not in merged_meta:
                merged_meta["summary_hash"] = lookup_meta.get("summary_hash")
            if lookup_meta.get("snippet_path") and "snippet_path" not in merged_meta:
                merged_meta["snippet_path"] = lookup_meta.get("snippet_path")
            if "size" not in merged_meta and "start_line" in merged_meta and "end_line" in merged_meta:
                try:
                    start = int(merged_meta["start_line"])
                    end = int(merged_meta["end_line"])
                    if end >= start:
                        merged_meta["size"] = end - start + 1
                except Exception:
                    pass
            merged_meta.setdefault("checksum", str(vector_id))
            merged_meta.setdefault("identifier", str(vector_id))
            merged_meta.setdefault("origin", origin)
            merged_meta.setdefault("license", merged_meta.get("license_name"))
            merged_meta.setdefault("redacted", True)
            snippet = str(merged_meta.get("summary") or merged_meta.get("snippet") or text or "")
            snippet_pointer = merged_meta.get("snippet_path") or merged_meta.get("snippet_pointer")
            summary_hash = merged_meta.get("summary_hash")
            cached_snippet: str | None = None
            if self._snippet_cache is not None and not snippet.strip():
                if snippet_pointer:
                    cached_snippet = self._snippet_cache.load_by_pointer(
                        str(snippet_pointer)
                    )
                if (
                    (not cached_snippet or not cached_snippet.strip())
                    and summary_hash
                ):
                    cached_snippet = self._snippet_cache.load(str(summary_hash))
            if cached_snippet and cached_snippet.strip():
                snippet = cached_snippet
                merged_meta.setdefault("snippet", cached_snippet)
            elif snippet.strip():
                merged_meta.setdefault("snippet", snippet)
            snippet = pii_redact_text(str(snippet or ""))
            governed = govern_retrieval(
                snippet,
                merged_meta,
                None,
                max_alert_severity=self.max_alert_severity,
            )
            if governed is None:
                continue
            governed_meta, reason = governed
            if governed_meta.get("license") in self.license_denylist:
                continue
            passed, risk_score, _ = self.patch_safety.evaluate(
                governed_meta,
                governed_meta,
                origin=origin,
            )
            if not passed:
                continue
            if self._should_skip(governed_meta, excluded_tags=excluded, keywords=keyword_list):
                continue
            roi_tag = governed_meta.get("roi_tag")
            if roi_tag is not None:
                similarity = max(
                    similarity - float(self.roi_tag_weights.get(str(roi_tag), 0.0)), 0.0
                )
            roi_tags = governed_meta.get("roi_tags")
            if isinstance(roi_tags, (list, tuple, set)):
                penalties = [
                    float(self.roi_tag_weights.get(str(tag), 0.0))
                    for tag in roi_tags
                    if str(tag) in self.roi_tag_weights
                ]
                if penalties:
                    similarity = max(similarity - max(penalties), 0.0)
            governed_meta["score"] = float(similarity)
            item: Dict[str, Any] = {
                "identifier": str(vector_id),
                "checksum": str(vector_id),
                "score": float(similarity),
                "text": snippet,
                "metadata": dict(governed_meta),
                "repo": governed_meta.get("repo"),
                "path": governed_meta.get("path"),
                "language": governed_meta.get("language"),
                "license": governed_meta.get("license"),
            }
            if reason:
                item["reason"] = reason
            item["metadata"]["risk_score"] = risk_score
            results.append(redact_dict(pii_redact_dict(item)))
        results.sort(key=lambda entry: entry.get("score", 0.0), reverse=True)
        return results

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._metadata_conn is not None:
            with contextlib.suppress(Exception):
                self._metadata_conn.close()
            self._metadata_conn = None


__all__ = ["StackRetriever", "StackRetrieverConfig"]
