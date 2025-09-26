from __future__ import annotations

"""Wrapper around :class:`universal_retriever.UniversalRetriever`.

The real retriever object lives in :mod:`universal_retriever` and returns
``ResultBundle`` instances.  The :class:`Retriever` below normalises those
results into plain dictionaries and provides a tiny layer of retry and
heuristic fallbacks used across the code base.
"""

from dataclasses import dataclass, field
import time
import asyncio
import math
import os
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Sequence

import urllib.request

from retrieval_cache import RetrievalCache

from redaction_utils import redact_dict as pii_redact_dict, redact_text as pii_redact_text
from governed_retrieval import govern_retrieval, redact, redact_dict
from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
try:  # pragma: no cover - optional dependency for metrics
    from . import metrics_exporter as _me  # noqa: F401  # type: ignore
except Exception:  # pragma: no cover - fallback when running as script
    import metrics_exporter as _me  # noqa: F401  # type: ignore

from .patch_logger import _VECTOR_RISK  # type: ignore
from patch_safety import PatchSafety
from .decorators import log_and_measure
from .exceptions import MalformedPromptError, RateLimitError, VectorServiceError

try:  # pragma: no cover - optional dependency
    from code_database import PatchHistoryDB  # type: ignore
except Exception:  # pragma: no cover
    PatchHistoryDB = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from vector_metrics_db import VectorMetricsDB  # type: ignore
except Exception:  # pragma: no cover
    VectorMetricsDB = None  # type: ignore

_DEFAULT_LICENSE_DENYLIST = set(_LICENSE_DENYLIST.values())


try:  # pragma: no cover - optional dependency
    from universal_retriever import UniversalRetriever  # type: ignore
except Exception:  # pragma: no cover - fallback when not available
    UniversalRetriever = None  # type: ignore


@dataclass
class FallbackResult(Sequence[Dict[str, Any]]):
    """Container returned when search falls back to heuristics.

    The object behaves like a sequence of result dictionaries while also
    exposing the ``reason`` and the confidence of the failed lookup.
    """

    reason: str
    results: List[Dict[str, Any]]
    confidence: float = 0.0

    def __iter__(self):  # pragma: no cover - simple delegation
        return iter(self.results)

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        return len(self.results)

    def __getitem__(self, item: int) -> Dict[str, Any]:  # pragma: no cover
        return self.results[item]


@dataclass
class Retriever:
    """Facade around :class:`UniversalRetriever` with caching and retries.

    When the underlying retriever exposes ``retrieve_with_confidence`` the
    confidence score is used to determine whether to retry the search with
    broader parameters.  If all attempts fail a :class:`FallbackResult` is
    returned containing the reason and heuristic hits.
    """

    context_builder: Any
    retriever: UniversalRetriever | None = None
    top_k: int = 5
    similarity_threshold: float = 0.1
    retriever_kwargs: Dict[str, Any] = field(default_factory=dict)
    content_filtering: bool = field(default=True)
    use_fts_fallback: bool = True
    cache: RetrievalCache | None = field(default_factory=RetrievalCache)
    max_alert_severity: float = 1.0
    max_alerts: int = 5
    license_denylist: set[str] = field(
        default_factory=lambda: set(_DEFAULT_LICENSE_DENYLIST)
    )
    patch_safety: PatchSafety = field(default_factory=PatchSafety)
    risk_penalty: float = 1.0
    roi_tag_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not hasattr(self.context_builder, "roi_tag_penalties"):
            raise TypeError(
                "context_builder must define 'roi_tag_penalties' for Retriever"
            )
        if not self.roi_tag_weights:
            try:
                self.roi_tag_weights = getattr(
                    self.context_builder, "roi_tag_penalties", {}
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _get_retriever(self) -> UniversalRetriever:
        if self.retriever is None:
            if UniversalRetriever is None:  # pragma: no cover - defensive
                raise RuntimeError("UniversalRetriever unavailable")
            self.retriever = UniversalRetriever(**self.retriever_kwargs)
        return self.retriever

    # ------------------------------------------------------------------
    def reload_reliability_scores(self) -> None:
        """Refresh retriever reliability statistics."""

        try:
            self._get_retriever().reload_reliability_scores()
        except AttributeError:
            pass

    # ------------------------------------------------------------------
    def _parse_hits(
        self,
        hits: Iterable[Any],
        *,
        max_alert_severity: float = 1.0,
        max_alerts: int | None = None,
        license_denylist: set[str] | None = None,
        exclude_tags: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        filtered = 0
        denylist = license_denylist or self.license_denylist
        alert_limit = max_alerts if max_alerts is not None else self.max_alerts
        ps = self.patch_safety
        ps.max_alert_severity = max_alert_severity
        ps.max_alerts = alert_limit
        ps.license_denylist = denylist
        excluded = set(exclude_tags or [])
        for h in hits:
            meta = getattr(h, "metadata", {})
            if not isinstance(meta, dict) or not meta.get("redacted"):
                continue
            if hasattr(h, "to_dict"):
                item = h.to_dict()
                item.setdefault("record_id", getattr(h, "record_id", None))
                meta = item.get("metadata", meta)
            else:  # pragma: no cover - very defensive
                item = {
                    "origin_db": getattr(h, "origin_db", ""),
                    "record_id": getattr(h, "record_id", None),
                    "score": getattr(h, "score", 0.0),
                    "reason": getattr(h, "reason", ""),
                    "metadata": meta,
                }
            # Pre-filter based on metadata-only safety signals so all
            # modalities are subject to the same limits.
            passed, _, _ = ps.evaluate(meta)
            if not passed:
                filtered += 1
                continue
            text = str(item.get("text") or "")
            governed = govern_retrieval(
                text, meta, item.get("reason"), max_alert_severity=max_alert_severity
            )
            if governed is None:
                filtered += 1
                continue
            meta, reason = governed
            origin = item.get("origin_db") or meta.get("origin") or ""
            passed, risk_score, _ = ps.evaluate(meta, meta, origin=origin)
            if not passed:
                filtered += 1
                continue
            item["metadata"] = meta
            if reason is not None:
                item["reason"] = reason
            item["license"] = meta.get("license")
            fp = meta.get("license_fingerprint")
            item["license_fingerprint"] = fp
            item["semantic_alerts"] = meta.get("semantic_alerts")
            item["alignment_severity"] = meta.get("alignment_severity")
            if isinstance(meta, dict):
                meta["risk_score"] = risk_score
            item["risk_score"] = risk_score
            item = redact_dict(pii_redact_dict(item))
            if fp is not None:
                item["license_fingerprint"] = fp
                if isinstance(item.get("metadata"), dict):
                    item["metadata"]["license_fingerprint"] = fp
            penalty = 0.0
            sev = meta.get("alignment_severity")
            if sev is not None:
                try:
                    penalty += float(sev)
                except Exception:
                    pass
            lic = meta.get("license")
            if lic in denylist or _LICENSE_DENYLIST.get(fp) in denylist:
                penalty += 1.0
            total_penalty = penalty + risk_score * self.risk_penalty
            score = max(float(item.get("score", 0.0)) - total_penalty, 0.0)
            roi_tag = meta.get("roi_tag")
            if roi_tag is not None:
                score = max(score - self.roi_tag_weights.get(str(roi_tag), 0.0), 0.0)
                item["roi_tag"] = roi_tag
                if isinstance(item.get("metadata"), dict):
                    item["metadata"]["roi_tag"] = roi_tag
            tags = meta.get("tags") if isinstance(meta, dict) else None
            tag_set = (
                {str(t) for t in tags}
                if isinstance(tags, (list, tuple, set))
                else {str(tags)} if tags else set()
            )
            if excluded and tag_set & excluded:
                continue
            item["score"] = score
            results.append(item)
        if filtered:
            try:
                _VECTOR_RISK.labels("filtered").inc(filtered)
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------
    def _fallback(
        self,
        query: str,
        limit: int | None = None,
        *,
        max_alert_severity: float = 1.0,
        max_alerts: int | None = None,
        license_denylist: set[str] | None = None,
        exclude_tags: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        query = redact(pii_redact_text(query))
        try:  # pragma: no cover - best effort import
            from code_database import CodeDB
        except Exception:
            return []
        try:
            rows = CodeDB().search_fts(query, limit or self.top_k)
        except Exception:
            return []
        results: List[Dict[str, Any]] = []
        filtered = 0
        denylist = license_denylist or self.license_denylist
        alert_limit = max_alerts if max_alerts is not None else self.max_alerts
        ps = self.patch_safety
        ps.max_alert_severity = max_alert_severity
        ps.max_alerts = alert_limit
        ps.license_denylist = denylist
        excluded = set(exclude_tags or [])
        for row in rows:
            text = str(row.get("code") or row.get("summary") or "")
            governed = govern_retrieval(text, max_alert_severity=max_alert_severity)
            if governed is None:
                filtered += 1
                continue
            meta, reason = governed
            passed, _ = ps.evaluate(meta)
            if not passed:
                filtered += 1
                continue
            fp = meta.get("license_fingerprint")
            lic = meta.get("license")
            alerts = meta.get("semantic_alerts")
            item = {
                "origin_db": row.get("origin_db", "code"),
                "record_id": row.get("id") or row.get("record_id"),
                "score": 0.0,
                "metadata": meta,
                "text": text,
                "license": lic,
                "license_fingerprint": fp,
                "semantic_alerts": alerts,
                "alignment_severity": meta.get("alignment_severity"),
            }
            if reason is not None:
                item["reason"] = reason
            item = redact_dict(pii_redact_dict(item))
            if fp is not None:
                item["license_fingerprint"] = fp
                if isinstance(item.get("metadata"), dict):
                    item["metadata"]["license_fingerprint"] = fp
            tags = meta.get("tags") if isinstance(meta, dict) else None
            tag_set = (
                {str(t) for t in tags}
                if isinstance(tags, (list, tuple, set))
                else {str(tags)} if tags else set()
            )
            if excluded and tag_set & excluded:
                continue
            penalty = 0.0
            sev = meta.get("alignment_severity")
            if sev is not None:
                try:
                    penalty += float(sev)
                except Exception:
                    pass
            if lic in denylist or _LICENSE_DENYLIST.get(fp) in denylist:
                penalty += 1.0
            item["score"] = max(float(item.get("score", 0.0)) - penalty, 0.0)
            results.append(item)
        if filtered:
            try:
                _VECTOR_RISK.labels("filtered").inc(filtered)
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------

    @log_and_measure
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        session_id: str = "",
        similarity_threshold: float | None = None,
        dbs: Sequence[str] | None = None,
        use_fts_fallback: bool | None = None,
        max_alert_severity: float | None = None,
        exclude_tags: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]] | FallbackResult:
        """Perform semantic search and return normalised results.

        The method first attempts ``retrieve_with_confidence`` when available.
        If the returned confidence is below ``similarity_threshold`` or no
        results are produced, the search is retried once with ``top_k`` doubled.
        After failing again a :class:`FallbackResult` is returned instead of
        plain hits.
        """

        if not isinstance(query, str) or not query.strip():
            raise MalformedPromptError("query must be a non-empty string")

        query = redact(pii_redact_text(query))
        k = top_k or self.top_k
        thresh = (
            similarity_threshold if similarity_threshold is not None else self.similarity_threshold
        )
        sev_limit = (
            max_alert_severity if max_alert_severity is not None else self.max_alert_severity
        )
        cached: List[Dict[str, Any]] | None = None
        if self.cache:
            cached = self.cache.get(query, dbs)
            if cached is not None:
                return cached

        retriever = self._get_retriever()

        attempts = 2
        backoff = 1.0
        hits: List[Any] = []
        confidence = 0.0
        for attempt in range(attempts):
            try:
                if dbs is None and hasattr(retriever, "retrieve_with_confidence"):
                    hits, confidence, _ = retriever.retrieve_with_confidence(
                        query, top_k=k
                    )  # type: ignore[attr-defined]
                else:
                    hits, _, _ = retriever.retrieve(  # type: ignore[arg-type]
                        query, top_k=k, dbs=dbs
                    )
                    confidence = (
                        max(getattr(h, "score", 0.0) for h in hits) if hits else 0.0
                    )
            except AttributeError:  # pragma: no cover - compatibility fallback
                try:
                    hits = retriever.search(query)[:k]  # type: ignore[assignment]
                    confidence = (
                        max(getattr(h, "score", 0.0) for h in hits) if hits else 0.0
                    )
                except Exception as exc:
                    raise VectorServiceError("vector search failed") from exc
            except Exception as exc:  # pragma: no cover - best effort
                msg = str(exc).lower()
                if ("rate" in msg and "limit" in msg) or "429" in msg:
                    if attempt < attempts - 1:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise RateLimitError("vector search rate limited") from exc
                raise VectorServiceError("vector search failed") from exc

            if hits and confidence >= thresh:
                results = self._parse_hits(
                    hits,
                    max_alert_severity=sev_limit,
                    max_alerts=self.max_alerts,
                    license_denylist=self.license_denylist,
                    exclude_tags=exclude_tags,
                )
                if results:
                    if self.cache:
                        self.cache.set(query, dbs, results)
                    return results

            # Broaden parameters and retry once
            k *= 2

        if self.cache:
            cached = self.cache.get(query, dbs)
            if cached is not None:
                return cached
        reason = "no results" if not hits else "low confidence"
        fts_hits: List[Dict[str, Any]] = []
        if use_fts_fallback if use_fts_fallback is not None else self.use_fts_fallback:
            fts_hits = fts_search(query, dbs=dbs, limit=k)
        fb_hits = self._fallback(
            query,
            limit=k,
            max_alert_severity=sev_limit,
            max_alerts=self.max_alerts,
            license_denylist=self.license_denylist,
            exclude_tags=exclude_tags,
        )
        merged: List[Dict[str, Any]] = []
        seen = set()
        for item in fts_hits + fb_hits:
            key = (item.get("origin_db"), item.get("record_id"))
            if key in seen:
                continue
            merged.append(item)
            seen.add(key)
        return FallbackResult(reason, merged, confidence)

    # ------------------------------------------------------------------

    @log_and_measure
    async def search_async(
        self,
        query: str,
        *,
        top_k: int | None = None,
        session_id: str = "",
        similarity_threshold: float | None = None,
        dbs: Sequence[str] | None = None,
        use_fts_fallback: bool | None = None,
        max_alert_severity: float | None = None,
        exclude_tags: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]] | FallbackResult:
        """Asynchronous wrapper for :meth:`search`.

        Executes the synchronous :meth:`search` implementation in a separate
        thread so it can be awaited without blocking the event loop.
        """
        query = redact(pii_redact_text(query))
        return await asyncio.to_thread(
            self.search.__wrapped__,
            self,
            query,
            top_k=top_k,
            session_id=session_id,
            similarity_threshold=similarity_threshold,
            dbs=dbs,
            use_fts_fallback=use_fts_fallback,
            max_alert_severity=max_alert_severity,
            exclude_tags=exclude_tags,
        )

    # ------------------------------------------------------------------
    def error_frequency(self, error_id: int, scope: str | None = None) -> float:
        """Expose raw error frequency metric from the underlying service."""

        try:
            return float(self._get_retriever()._error_frequency(int(error_id), scope))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def workflow_usage(self, wf: Any) -> float:
        """Expose workflow usage count for a workflow record."""

        try:
            return float(self._get_retriever()._workflow_usage(wf))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def bot_deploy_freq(self, bot_id: int) -> float:
        """Return deployment frequency for the given bot id."""

        try:
            return float(self._get_retriever()._bot_deploy_freq(int(bot_id)))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def search_bots(
        self, query: str, *, top_k: int | None = None
    ) -> List[Dict[str, Any]] | FallbackResult:
        """Convenience wrapper to search only bot definitions."""

        return self.search(query, top_k=top_k, dbs=["bot"])

    def search_workflows(
        self, query: str, *, top_k: int | None = None
    ) -> List[Dict[str, Any]] | FallbackResult:
        """Convenience wrapper to search only workflow specifications."""

        return self.search(query, top_k=top_k, dbs=["workflow"])

    def search_enhancements(
        self, query: str, *, top_k: int | None = None
    ) -> List[Dict[str, Any]] | FallbackResult:
        """Search only enhancement records."""

        return self.search(query, top_k=top_k, dbs=["enhancement"])

    def search_errors(
        self, query: str, *, top_k: int | None = None
    ) -> List[Dict[str, Any]] | FallbackResult:
        """Search only error telemetry entries."""

        return self.search(query, top_k=top_k, dbs=["error"])


if TYPE_CHECKING:  # pragma: no cover
    from .vector_store import VectorStore
    from .vectorizer import SharedVectorService


@dataclass
class StackRetriever:
    """Query Stack embeddings stored in a dedicated index."""

    context_builder: Any
    stack_index: "VectorStore" | None = None
    vector_store: "VectorStore" | None = None
    metadata_db_path: str | Path | None = None
    vector_service: "SharedVectorService" | None = None
    namespace: str = "stack"
    top_k: int = 5
    metric: str = "cosine"
    max_lines: int = 200
    max_alert_severity: float = 1.0
    max_alerts: int = 5
    languages: Iterable[str] | None = None
    min_similarity: float | None = None
    min_score: float | None = None
    license_denylist: set[str] = field(
        default_factory=lambda: set(_DEFAULT_LICENSE_DENYLIST)
    )
    risk_penalty: float = 1.0
    roi_tag_weights: Dict[str, float] = field(default_factory=dict)
    patch_safety: PatchSafety | None = None

    def __post_init__(self) -> None:
        if not hasattr(self.context_builder, "roi_tag_penalties"):
            raise TypeError(
                "context_builder must define 'roi_tag_penalties' for StackRetriever"
            )
        if not self.roi_tag_weights:
            try:
                self.roi_tag_weights = dict(
                    getattr(self.context_builder, "roi_tag_penalties", {})
                )
            except Exception:
                self.roi_tag_weights = {}
        if self.patch_safety is None:
            ps = getattr(self.context_builder, "patch_safety", None)
            self.patch_safety = ps if isinstance(ps, PatchSafety) else PatchSafety()
        normalised = {
            str(lic).strip()
            for lic in (self.license_denylist or set())
            if isinstance(lic, str) and lic.strip()
        }
        self.license_denylist = set(normalised)
        self._license_denylist_lower = {lic.lower() for lic in self.license_denylist}
        if self.vector_service is None:
            try:  # pragma: no cover - heavy dependency handled lazily
                from .vectorizer import SharedVectorService  # type: ignore
            except Exception:  # pragma: no cover - fallback when running as script
                from vector_service.vectorizer import (  # type: ignore
                    SharedVectorService,
                )
            self.vector_service = SharedVectorService()
        if isinstance(self.metadata_db_path, (str, Path)):
            try:
                self.metadata_db_path = Path(self.metadata_db_path)
            except Exception:
                self.metadata_db_path = None
        self._metadata_table = f"{self.namespace}_metadata"
        self._metadata_conn: sqlite3.Connection | None = None
        if self.stack_index is None and self.vector_service is not None:
            self.stack_index = getattr(self.vector_service, "vector_store", None)
        # Backwards compatibility for callers using the old ``vector_store`` name
        vector_store_alias = getattr(self, "vector_store", None)
        if vector_store_alias is not None and self.stack_index is None:
            self.stack_index = vector_store_alias
        elif vector_store_alias is None:
            setattr(self, "vector_store", self.stack_index)
        self.metric = str(self.metric or "cosine").lower()
        self.embedder = getattr(self.vector_service, "text_embedder", None)
        self.set_languages(self.languages)

    # ------------------------------------------------------------------
    def set_languages(self, languages: Iterable[str] | None) -> None:
        normalised = {
            str(lang).strip().lower()
            for lang in (languages or [])
            if isinstance(lang, str) and lang.strip()
        }
        self.languages = normalised or None

    # ------------------------------------------------------------------
    def embed_query(self, query: str) -> List[float]:
        if not isinstance(query, str) or not query.strip():
            return []
        if self.vector_service is None:
            return []
        try:
            vec = self.vector_service.vectorise("text", {"text": query})
        except Exception:
            return []
        coerced: List[float] = []
        try:
            coerced = [float(x) for x in vec]
        except Exception:
            return []
        return coerced

    # ------------------------------------------------------------------
    def _coerce_vector(self, query: str | Sequence[float]) -> List[float] | None:
        if isinstance(query, str):
            return self.embed_query(query)
        try:
            return [float(x) for x in query]  # type: ignore[arg-type]
        except Exception:
            return None

    def _trim_snippet(self, text: str) -> str:
        if not text:
            return ""
        if self.max_lines and self.max_lines > 0:
            lines = text.splitlines()
            if len(lines) > self.max_lines:
                return "\n".join(lines[: self.max_lines])
        return text

    def _similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        metric = (self.metric or "cosine").lower()
        if metric == "inner_product":
            return sum(float(x) * float(y) for x, y in zip(a, b))
        # default to cosine similarity
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        norm_a = math.sqrt(sum(float(x) ** 2 for x in a))
        norm_b = math.sqrt(sum(float(x) ** 2 for x in b))
        if not norm_a or not norm_b:
            return 0.0
        return dot / (norm_a * norm_b)

    def _to_unit_interval(self, score: float) -> float:
        if (self.metric or "cosine").lower() == "inner_product":
            return (score + 1.0) / 2.0
        return 1.0 / (1.0 + math.exp(-score))

    def _normalise_distance(self, dist: float, backend: str) -> float:
        if "qdrant" in backend:
            return max(0.0, min(1.0, float(dist)))
        return 1.0 / (1.0 + float(dist))

    def _compute_similarity(
        self,
        query_vec: Sequence[float],
        stored_vec: Sequence[float],
        distance: float,
        backend: str,
    ) -> float:
        if stored_vec:
            return self._to_unit_interval(self._similarity(query_vec, stored_vec))
        return self._normalise_distance(distance, backend)

    def _extract_entry(
        self, store: "VectorStore", record_id: str
    ) -> tuple[Dict[str, Any], Sequence[float], Dict[str, Any]]:
        ids = list(getattr(store, "ids", []) or [])
        meta = list(getattr(store, "meta", []) or [])
        vectors = getattr(store, "vectors", None)
        rid = str(record_id)
        idx = -1
        if rid in ids:
            idx = ids.index(rid)
        elif record_id in ids:
            idx = ids.index(record_id)
        entry: Dict[str, Any] = {"id": rid}
        vec: Sequence[float] = []
        raw_meta: Dict[str, Any] = {}
        if idx >= 0:
            if idx < len(meta) and isinstance(meta[idx], dict):
                entry = dict(meta[idx])
                raw = entry.get("metadata", {})
                raw_meta = dict(raw) if isinstance(raw, dict) else {}
            if vectors is not None and idx < len(vectors):
                try:
                    vec = [float(x) for x in vectors[idx]]  # type: ignore[assignment]
                except Exception:
                    vec = []
        entry.setdefault("id", rid)
        return entry, vec, raw_meta

    def _get_metadata_connection(self) -> sqlite3.Connection | None:
        if not isinstance(self.metadata_db_path, Path):
            return None
        if not self.metadata_db_path.exists():
            return None
        if self._metadata_conn is None:
            try:
                conn = sqlite3.connect(
                    str(self.metadata_db_path), check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                self._metadata_conn = conn
            except Exception:
                self._metadata_conn = None
        return self._metadata_conn

    def _load_metadata(self, record_id: str) -> Dict[str, Any]:
        conn = self._get_metadata_connection()
        if conn is None:
            return {}
        try:
            cur = conn.execute(
                f"""
                SELECT repo, path, language, license, chunk_index, start_line,
                       end_line, token_count
                FROM {self._metadata_table}
                WHERE embedding_id = ?
                """,
                (record_id,),
            )
            row = cur.fetchone()
        except Exception:
            return {}
        if row is None:
            return {}
        if isinstance(row, sqlite3.Row):
            data = {key: row[key] for key in row.keys()}
        else:  # pragma: no cover - defensive fallback for custom cursors
            cols = [desc[0] for desc in getattr(cur, "description", [])]
            data = {col: row[idx] for idx, col in enumerate(cols)}
        return {k: v for k, v in data.items() if v is not None}

    def _prepare_metadata(
        self,
        raw: Dict[str, Any],
        record_id: str,
        db_meta: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        if isinstance(db_meta, dict):
            for key, value in db_meta.items():
                if value is None:
                    continue
                meta[key] = value
        if isinstance(raw, dict):
            for key, value in raw.items():
                if value is None or key in meta:
                    continue
                meta[key] = value
        meta.setdefault("record_id", record_id)
        meta.setdefault("origin", self.namespace)
        meta.setdefault("redacted", True)
        return meta

    # ------------------------------------------------------------------
    def get_index_path(self) -> Path | None:
        """Best-effort resolution of the backing vector index path."""

        store = self.stack_index or getattr(self, "vector_store", None)
        candidate = getattr(store, "path", None) or getattr(store, "index_path", None)
        if isinstance(candidate, Path):
            return candidate
        if isinstance(candidate, str):
            try:
                return Path(candidate)
            except Exception:
                return None
        if isinstance(self.metadata_db_path, Path):
            return self.metadata_db_path.with_suffix(".index")
        return None

    # ------------------------------------------------------------------
    def get_metadata_path(self) -> Path | None:
        """Return the SQLite database path storing Stack metadata when known."""

        path = getattr(self, "metadata_db_path", None)
        if isinstance(path, Path):
            return path
        if isinstance(path, str):
            try:
                return Path(path)
            except Exception:
                return None
        return None

    # ------------------------------------------------------------------
    def is_index_stale(self) -> bool:
        """Return ``True`` when no Stack embeddings appear to be available."""

        meta_path = self.get_metadata_path()
        if meta_path is None or not meta_path.exists():
            return True

        store = self.stack_index or getattr(self, "vector_store", None)
        ids = getattr(store, "ids", None)
        if isinstance(ids, list) and ids:
            return False

        tables = {self._metadata_table, f"{self.namespace}_embeddings"}
        try:
            conn = sqlite3.connect(str(meta_path))
        except Exception:
            return True
        try:
            tables = {t for t in tables if t}
            for table in tables:
                try:
                    cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
                except Exception:
                    continue
                row = cur.fetchone()
                if row and row[0]:
                    try:
                        if int(row[0]) > 0:
                            return False
                    except Exception:
                        return False
        finally:
            try:
                conn.close()
            except Exception:
                pass
        return True

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str | Sequence[float],
        *,
        k: int | None = None,
        exclude_tags: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        vector = self._coerce_vector(query)
        if not vector:
            return []
        store = self.stack_index
        if store is None:
            return []
        max_hits = max(0, int(k if k is not None else self.top_k))
        if max_hits == 0:
            return []
        try:
            raw_hits = store.query(vector, top_k=max(max_hits * 3, max_hits))
        except Exception:
            return []
        excluded = {str(tag) for tag in (exclude_tags or []) if tag is not None}
        backend = store.__class__.__name__.lower()
        results: List[Dict[str, Any]] = []
        filtered = 0
        for record_id, distance in raw_hits:
            entry, stored_vec, raw_meta = self._extract_entry(store, record_id)
            entry_type = str(
                entry.get("type")
                or entry.get("kind")
                or raw_meta.get("origin")
                or self.namespace
            ).lower()
            if entry_type and entry_type != self.namespace.lower():
                continue
            rid = str(entry.get("id") or record_id)
            db_meta = self._load_metadata(rid)
            meta = self._prepare_metadata(raw_meta, rid, db_meta)
            license_value = str(meta.get("license") or "").strip()
            if license_value:
                meta["license"] = license_value
                if license_value.lower() in self._license_denylist_lower:
                    continue
            fp_value = meta.get("license_fingerprint")
            if fp_value is not None:
                mapped_license = _LICENSE_DENYLIST.get(str(fp_value))
                if mapped_license and mapped_license.lower() in self._license_denylist_lower:
                    continue
            tags = meta.get("tags")
            tag_set = (
                {str(t) for t in tags}
                if isinstance(tags, (list, tuple, set))
                else {str(tags)}
                if tags
                else set()
            )
            if excluded and tag_set & excluded:
                continue
            language_value = str(meta.get("language") or "").strip().lower()
            if self.languages:
                if not language_value or language_value not in self.languages:
                    continue
                meta["language"] = language_value
            snippet_source = (
                meta.get("summary")
                or meta.get("snippet")
                or meta.get("content")
                or meta.get("text")
                or ""
            )
            if not isinstance(snippet_source, str):
                snippet_source = str(snippet_source or "")
            snippet_source = snippet_source.strip()
            if not snippet_source:
                continue
            snippet = self._trim_snippet(snippet_source)
            truncated_lines: int | None = None
            if (
                snippet != snippet_source
                and self.max_lines
                and isinstance(self.max_lines, int)
                and self.max_lines > 0
            ):
                try:
                    truncated_lines = int(self.max_lines)
                except Exception:
                    try:
                        truncated_lines = int(float(self.max_lines))
                    except Exception:
                        truncated_lines = None
                if truncated_lines is not None:
                    meta["stack_truncated_lines"] = truncated_lines
            if not snippet:
                continue
            meta["summary"] = snippet
            for key in ("snippet", "content", "text"):
                if key in meta:
                    meta.pop(key, None)
            governed = govern_retrieval(
                snippet,
                meta,
                entry.get("reason"),
                max_alert_severity=self.max_alert_severity,
            )
            if governed is None:
                filtered += 1
                continue
            meta, reason = governed
            if truncated_lines is not None:
                meta.setdefault("stack_truncated_lines", truncated_lines)
            if self.languages and language_value:
                meta.setdefault("language", language_value)
            ps = self.patch_safety or PatchSafety()
            ps.max_alert_severity = self.max_alert_severity
            ps.max_alerts = self.max_alerts
            ps.license_denylist = self.license_denylist
            passed, risk_score, _ = ps.evaluate(meta, meta, origin=self.namespace)
            if not passed:
                filtered += 1
                continue
            penalty = float(risk_score) * float(self.risk_penalty)
            severity = meta.get("alignment_severity")
            if severity is not None:
                try:
                    penalty += float(severity)
                except Exception:
                    pass
            lic = str(meta.get("license") or "").strip()
            fp = meta.get("license_fingerprint")
            mapped = _LICENSE_DENYLIST.get(str(fp))
            if lic and lic.lower() in self._license_denylist_lower:
                penalty += 1.0
            elif mapped and mapped.lower() in self._license_denylist_lower:
                penalty += 1.0
            similarity = self._compute_similarity(
                vector, stored_vec, float(distance), backend
            )
            if self.min_similarity is not None and similarity < float(self.min_similarity):
                continue
            score = max(similarity - penalty, 0.0)
            roi_tag = meta.get("roi_tag")
            if roi_tag is not None:
                score = max(
                    score - self.roi_tag_weights.get(str(roi_tag), 0.0),
                    0.0,
                )
            if self.min_score is not None and score < float(self.min_score):
                continue
            meta["risk_score"] = risk_score
            meta.setdefault("license_fingerprint", fp)
            item = {
                "origin_db": self.namespace,
                "record_id": rid,
                "text": snippet,
                "score": score,
                "similarity": similarity,
                "metadata": meta,
                "redacted": True,
            }
            if reason is not None:
                item["reason"] = reason
            item = redact_dict(pii_redact_dict(item))
            results.append(item)
            if len(results) >= max_hits:
                break
        if filtered:
            try:
                _VECTOR_RISK.labels("filtered").inc(filtered)
            except Exception:
                pass
        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results


@dataclass
class PatchRetriever:
    """Lightweight retriever that queries a local :class:`VectorStore`.

    The retriever embeds natural language queries using
    :class:`~vector_service.vectorizer.SharedVectorService` and looks up the
    closest patch vectors from the configured :class:`VectorStore`.
    """

    context_builder: Any
    store: VectorStore | None = None
    vector_service: SharedVectorService | None = None
    top_k: int = 5
    metric: str | None = None
    enhancement_weight: float = 1.0
    vector_metrics: VectorMetricsDB | None = None
    roi_tag_weights: Dict[str, float] = field(default_factory=dict)
    service_url: str | None = None

    def __post_init__(self) -> None:
        if self.service_url is None:
            self.service_url = os.environ.get("VECTOR_SERVICE_URL")

        if self.vector_service is None and not self.service_url:
            try:
                from .vectorizer import SharedVectorService  # type: ignore
            except Exception:  # pragma: no cover - fallback to absolute import
                from vector_service.vectorizer import SharedVectorService  # type: ignore
            self.vector_service = SharedVectorService()
        backend = "annoy"
        path = "vectors.index"
        dim = 0
        metric = "cosine"
        try:  # pragma: no cover - configuration optional in tests
            from config import CONFIG
            cfg = getattr(CONFIG, "vector_store", None)
            vec_cfg = getattr(CONFIG, "vector", None)
            if cfg is not None:
                backend = getattr(cfg, "backend", backend)
                path = getattr(cfg, "path", path)
                metric = getattr(cfg, "metric", metric)
            if vec_cfg is not None:
                dim = getattr(vec_cfg, "dimensions", dim)
        except Exception:
            pass
        if self.store is None and not self.service_url:
            try:
                from .vector_store import create_vector_store  # type: ignore
            except Exception:  # pragma: no cover - fallback
                from vector_service.vector_store import create_vector_store  # type: ignore
            self.store = create_vector_store(dim or 0, path, backend=backend)
        if not self.metric and not self.service_url:
            self.metric = str(metric).lower()

        if self.vector_metrics is None and VectorMetricsDB is not None and not self.service_url:
            try:
                self.vector_metrics = VectorMetricsDB()
            except Exception:
                self.vector_metrics = None
        if not hasattr(self.context_builder, "roi_tag_penalties"):
            raise TypeError(
                "context_builder must define 'roi_tag_penalties' for PatchRetriever"
            )
        if not self.roi_tag_weights:
            try:
                self.roi_tag_weights = getattr(
                    self.context_builder, "roi_tag_penalties", {}
                )
            except Exception:
                pass

    def reload_from_config(self) -> None:
        """Reload configuration for backend and metric."""
        try:
            from config import CONFIG
            cfg = getattr(CONFIG, "vector_store", None)
            vec_cfg = getattr(CONFIG, "vector", None)
        except Exception:
            return
        if cfg is None or vec_cfg is None:
            return
        backend = getattr(cfg, "backend", None)
        path = getattr(cfg, "path", None)
        metric = getattr(cfg, "metric", None)
        try:
            from .vector_store import create_vector_store  # type: ignore
        except Exception:  # pragma: no cover - fallback
            from vector_service.vector_store import create_vector_store  # type: ignore
        if backend and path:
            self.store = create_vector_store(vec_cfg.dimensions, path, backend=backend)
        if metric:
            self.metric = str(metric).lower()

    # ------------------------------------------------------------------
    def _similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Return similarity score for vectors ``a`` and ``b``.

        The computation depends on :attr:`metric` which can be either
        ``"cosine"`` (the default) or ``"inner_product"``.  Any other value
        results in a :class:`ValueError` to surface misconfiguration early.
        """

        metric = (self.metric or "cosine").lower()
        if metric == "inner_product":
            return float(sum(x * y for x, y in zip(a, b)))
        if metric == "cosine":
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if not na or not nb:
                return 0.0
            return float(sum(x * y for x, y in zip(a, b)) / (na * nb))
        raise ValueError(f"unsupported metric: {self.metric}")

    def _to_unit_interval(self, score: float) -> float:
        if (self.metric or "cosine").lower() == "cosine":
            return (score + 1.0) / 2.0
        return 1.0 / (1.0 + math.exp(-score))

    def _normalise_distance(self, dist: float, backend: str) -> float:
        if "qdrant" in backend:
            return max(0.0, min(1.0, float(dist)))
        return 1.0 / (1.0 + float(dist))

    # ------------------------------------------------------------------

    @log_and_measure
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        exclude_tags: Iterable[str] | None = None,
    ) -> List[Dict[str, Any]]:
        if self.service_url:
            data = json.dumps(
                {
                    "kind": "text",
                    "record": {"text": query},
                    "top_k": top_k or self.top_k,
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                f"{self.service_url.rstrip('/')}/search",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                hits = payload.get("data", [])
                results: List[Dict[str, Any]] = []
                excluded = set(exclude_tags or [])
                for item in hits:
                    if not isinstance(item, dict):
                        continue
                    md = item.get("metadata") or {}
                    tags = md.get("roi_tags") if isinstance(md, dict) else None
                    if isinstance(tags, list) and excluded.intersection(tags):
                        continue
                    results.append(item)
                return results
            except Exception:
                return []

        if self.store is None or self.vector_service is None:
            return []
        vec = self.vector_service.vectorise("text", {"text": query})
        ids = getattr(self.store, "ids", [])
        vectors = getattr(self.store, "vectors", [])
        meta = getattr(self.store, "meta", [])
        backend = self.store.__class__.__name__.lower()
        results: List[Dict[str, Any]] = []
        excluded = set(exclude_tags or [])
        for vid, dist in self.store.query(vec, top_k=top_k or self.top_k):
            md: Dict[str, Any] = {}
            text_val = ""
            origin = "patch"
            similarity: float
            if vid in ids:
                idx = ids.index(vid)
                vec2 = vectors[idx] if idx < len(vectors) else []
                raw = self._similarity(vec, vec2)
                similarity = self._to_unit_interval(raw)
                m = meta[idx] if idx < len(meta) else {}
                md = m.get("metadata", {}) if isinstance(m, dict) else {}
                text_val = md.get("diff") or md.get("text") or ""
                origin = m.get("origin_db", "patch")
            else:
                similarity = self._normalise_distance(dist, backend)

            enh: float | None = None
            if isinstance(md, dict):
                try:
                    enh = float(md.get("enhancement_score"))
                except Exception:
                    enh = None
            if (enh is None) and PatchHistoryDB is not None:
                try:
                    pid = int(vid)
                    rec = PatchHistoryDB().get(pid)  # type: ignore[operator]
                    if rec is not None:
                        enh = float(getattr(rec, "enhancement_score", 0.0))
                except Exception:
                    pass
            if (enh is None) and self.vector_metrics is not None:
                try:
                    cur = self.vector_metrics.conn.execute(
                        "SELECT enhancement_score FROM patch_metrics WHERE patch_id=?",
                        (str(vid),),
                    )
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        enh = float(row[0])
                except Exception:
                    pass
            if enh is None:
                enh = 0.0

            score = similarity
            if self.enhancement_weight:
                score *= 1.0 + max(0.0, enh) * self.enhancement_weight
            roi_tag = md.get("roi_tag") if isinstance(md, dict) else None
            if roi_tag is not None:
                score = max(score - self.roi_tag_weights.get(str(roi_tag), 0.0), 0.0)
                if isinstance(md, dict):
                    md["roi_tag"] = roi_tag

            tags = md.get("tags") if isinstance(md, dict) else None
            tag_set = (
                {str(t) for t in tags}
                if isinstance(tags, (list, tuple, set))
                else {str(tags)} if tags else set()
            )
            if excluded and tag_set & excluded:
                continue
            item = {
                "origin_db": origin,
                "record_id": str(vid),
                "score": score,
                "similarity": similarity,
                "text": text_val,
                "metadata": md,
            }
            if roi_tag is not None:
                item["roi_tag"] = roi_tag
            if enh:
                item["enhancement_score"] = enh
            results.append(item)
        results.sort(key=lambda r: r.get("score", 0.0), reverse=True)
        return results


_patch_retriever: PatchRetriever | None = None


def _get_patch_retriever(builder: Any) -> PatchRetriever:
    """Return a configured :class:`PatchRetriever` instance."""

    global _patch_retriever
    if _patch_retriever is None or _patch_retriever.context_builder is not builder:
        _patch_retriever = PatchRetriever(context_builder=builder)
    return _patch_retriever


def search_patches(
    query: str,
    top_k: int = 5,
    *,
    exclude_tags: Iterable[str] | None = None,
    context_builder: Any,
) -> List[Dict[str, Any]]:
    """Retrieve patch examples for ``query``.

    This adapter wraps :class:`PatchRetriever.search` and initialises the
    retriever using application configuration, exposing a single entry point
    for modules that only need patch lookups.  The similarity metric
    (``cosine`` or ``inner_product``) and the underlying :class:`VectorStore`
    are selected via configuration.
    """

    return _get_patch_retriever(context_builder).search(
        query, top_k=top_k, exclude_tags=exclude_tags
    )


def fts_search(
    query: str,
    *,
    dbs: Sequence[str] | None = None,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Run SQLite FTS queries via :class:`DBRouter`.

    The helper is intentionally lightweight and returns an empty list on any
    failure so callers can use it opportunistically.
    """

    try:  # pragma: no cover - optional dependency
        from db_router import DBRouter
    except Exception:
        return []

    try:
        router = DBRouter()
        return router.search_fts(query, dbs=dbs, limit=limit)
    except Exception:
        return []


__all__ = [
    "Retriever",
    "PatchRetriever",
    "StackRetriever",
    "FallbackResult",
    "fts_search",
    "search_patches",
]