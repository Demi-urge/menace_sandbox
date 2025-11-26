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
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Sequence
import copy
import contextlib

import urllib.request

from retrieval_cache import RetrievalCache

from redaction_utils import redact_dict as pii_redact_dict, redact_text as pii_redact_text
from governed_retrieval import govern_retrieval, redact, redact_dict
from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
from .registry import _resolve_bootstrap_fast
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

try:  # pragma: no cover - optional dependency for stack integration
    from .stack_retriever import StackRetriever as _StackDatasetRetriever  # type: ignore
except Exception:  # pragma: no cover - stack dataset optional
    _StackDatasetRetriever = None  # type: ignore


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
    bootstrap_fast: bool | None = None

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
    bootstrap_fast: bool | None = None

    def __post_init__(self) -> None:
        requested_fast = self.bootstrap_fast
        resolved_fast, bootstrap_context, defaulted_fast = _resolve_bootstrap_fast(
            requested_fast
        )
        if self.bootstrap_fast is None:
            self.bootstrap_fast = resolved_fast
        if bootstrap_context and resolved_fast:
            logger.info(
                "patch_retriever.bootstrap_fast.active",
                extra={
                    "bootstrap_fast_defaulted": defaulted_fast,
                    "bootstrap_context": True,
                },
            )

        if self.service_url is None:
            self.service_url = os.environ.get("VECTOR_SERVICE_URL")

        if self.vector_service is None and not self.service_url:
            try:
                from .vectorizer import SharedVectorService  # type: ignore
            except Exception:  # pragma: no cover - fallback to absolute import
                from vector_service.vectorizer import SharedVectorService  # type: ignore
            self.vector_service = SharedVectorService(
                bootstrap_fast=requested_fast
            )
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
                self.vector_metrics = VectorMetricsDB(
                    bootstrap_fast=self.bootstrap_fast
                )
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


class StackRetriever:
    """Lightweight adapter around the Stack dataset embedding store."""

    def __init__(
        self,
        *,
        backend: Any | None = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        max_alert_severity: float = 1.0,
        max_alerts: int = 5,
        license_denylist: Iterable[str] | None = None,
        roi_tag_weights: Mapping[str, float] | None = None,
        **backend_kwargs: Any,
    ) -> None:
        self.top_k = int(top_k)
        self.similarity_threshold = float(similarity_threshold)
        self.max_alert_severity = float(max_alert_severity)
        self.max_alerts = int(max_alerts)
        self.license_denylist: set[str] = set(license_denylist or _DEFAULT_LICENSE_DENYLIST)
        self.roi_tag_weights: Dict[str, float] = dict(roi_tag_weights or {})
        self._backend_kwargs = dict(backend_kwargs)
        self._backend: Any | None = backend

    def _sync_backend(self, backend: Any) -> None:
        for attr in ("top_k", "similarity_threshold", "max_alert_severity", "max_alerts"):
            if hasattr(backend, attr):
                setattr(backend, attr, getattr(self, attr))
        if hasattr(backend, "license_denylist"):
            setattr(backend, "license_denylist", set(self.license_denylist))
        if hasattr(backend, "roi_tag_weights"):
            setattr(backend, "roi_tag_weights", dict(self.roi_tag_weights))

    def _get_backend(self) -> Any:
        backend = self._backend
        if backend is None:
            if _StackDatasetRetriever is None:  # pragma: no cover - defensive
                raise RuntimeError("Stack retriever backend unavailable")
            kwargs = dict(self._backend_kwargs)
            kwargs.setdefault("top_k", self.top_k)
            kwargs.setdefault("similarity_threshold", self.similarity_threshold)
            kwargs.setdefault("max_alert_severity", self.max_alert_severity)
            kwargs.setdefault("max_alerts", self.max_alerts)
            kwargs.setdefault("license_denylist", set(self.license_denylist))
            kwargs.setdefault("roi_tag_weights", dict(self.roi_tag_weights))
            backend = _StackDatasetRetriever(**kwargs)  # type: ignore[misc]
            self._backend = backend
        self._sync_backend(backend)
        return backend

    def embed_query(self, query: str) -> List[float] | None:
        backend = self._get_backend()
        embed_fn = getattr(backend, "embed_query", None)
        if not callable(embed_fn):
            return None
        try:
            vec = embed_fn(query)
        except Exception:  # pragma: no cover - passthrough for backend errors
            return None
        if hasattr(vec, "tolist"):
            vec = vec.tolist()  # type: ignore[attr-defined]
        if isinstance(vec, (list, tuple)):
            return [float(x) for x in vec]
        return None

    def warm_cache(self) -> bool:
        backend = self._get_backend()
        fn = getattr(backend, "warm_cache", None)
        if callable(fn):
            try:
                return bool(fn())
            except Exception:  # pragma: no cover - defensive
                return False
        return False

    def close(self) -> None:
        backend = self._backend
        if backend is None:
            return
        close_fn = getattr(backend, "close", None)
        if callable(close_fn):
            with contextlib.suppress(Exception):
                close_fn()

    def retrieve(
        self,
        query_embedding: Sequence[float],
        k: int | None = None,
        *,
        languages: Iterable[str] | None = None,
        max_lines: int | None = None,
    ) -> List[Dict[str, Any]]:
        backend = self._get_backend()
        top_k = int(k) if k is not None else self.top_k
        try:
            raw_hits = backend.retrieve(
                list(query_embedding),
                k=top_k,
                similarity_threshold=self.similarity_threshold,
            )
        except TypeError:
            raw_hits = backend.retrieve(list(query_embedding), k=top_k)
        except Exception:  # pragma: no cover - backend failure
            return []

        allowed_languages = {
            str(lang).strip().lower()
            for lang in (languages or [])
            if str(lang).strip()
        }
        line_cap = max_lines if max_lines is not None and max_lines > 0 else None
        results: List[Dict[str, Any]] = []
        for hit in raw_hits or []:
            if not isinstance(hit, Mapping):
                continue
            score = float(hit.get("score", 0.0))
            metadata = dict(copy.deepcopy(hit.get("metadata", {})) or {})
            language = str(
                metadata.get("language")
                or metadata.get("lang")
                or hit.get("language")
                or ""
            ).strip()
            if allowed_languages:
                language_key = language.lower() if language else ""
                if language_key not in allowed_languages:
                    continue
            start = metadata.get("start_line")
            end = metadata.get("end_line")
            size = metadata.get("size")
            try:
                if size is None and isinstance(start, int) and isinstance(end, int):
                    size = end - start + 1
            except Exception:
                size = None
            if line_cap is not None and size is not None and size > line_cap:
                continue
            snippet = (
                hit.get("summary")
                or metadata.get("summary")
                or hit.get("text")
                or metadata.get("text")
                or ""
            )
            snippet = pii_redact_text(str(snippet))
            result_meta = dict(metadata)
            if language:
                result_meta.setdefault("language", language)
            if size is not None:
                result_meta.setdefault("size", int(size))
            result_meta.setdefault("source", "stack")
            result_meta.setdefault("redacted", True)
            result_meta.setdefault("score", score)
            result: Dict[str, Any] = {
                "score": score,
                "metadata": result_meta,
                "text": snippet,
                "origin_db": hit.get("origin_db", "stack"),
            }
            identifier = (
                hit.get("record_id")
                or hit.get("identifier")
                or result_meta.get("identifier")
                or result_meta.get("checksum")
            )
            if identifier is not None:
                result.setdefault("identifier", identifier)
                result.setdefault("record_id", identifier)
            for key in ("checksum", "repo", "path", "license"):
                if key in hit and key not in result:
                    result[key] = hit[key]
                if key not in result and key in result_meta:
                    result[key] = result_meta[key]
            result["metadata"] = redact_dict(pii_redact_dict(result_meta))
            result = redact_dict(pii_redact_dict(result))
            results.append(result)
        results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return results


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
    "FallbackResult",
    "StackRetriever",
    "fts_search",
    "search_patches",
]