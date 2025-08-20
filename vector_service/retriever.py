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
from typing import Any, Dict, Iterable, List, Sequence

from retrieval_cache import RetrievalCache

from redaction_utils import redact_dict as pii_redact_dict, redact_text as pii_redact_text
from governed_retrieval import govern_retrieval, redact, redact_dict
from compliance.license_fingerprint import DENYLIST as _LICENSE_DENYLIST
try:  # pragma: no cover - optional dependency for metrics
    from . import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - fallback when running as script
    import metrics_exporter as _me  # type: ignore

_FILTERED_RESULTS = _me.Gauge(
    "retriever_filtered_vectors_total",
    "Vectors filtered due to high alert severity",
)
_DISALLOWED_LICENSES = set(_LICENSE_DENYLIST.values())
from .decorators import log_and_measure
from .exceptions import MalformedPromptError, RateLimitError, VectorServiceError


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

    retriever: UniversalRetriever | None = None
    top_k: int = 5
    similarity_threshold: float = 0.1
    retriever_kwargs: Dict[str, Any] = field(default_factory=dict)
    content_filtering: bool = field(default=True)
    use_fts_fallback: bool = True
    cache: RetrievalCache | None = field(default_factory=RetrievalCache)
    max_alert_severity: float = 1.0

    # ------------------------------------------------------------------
    def _get_retriever(self) -> UniversalRetriever:
        if self.retriever is None:
            if UniversalRetriever is None:  # pragma: no cover - defensive
                raise RuntimeError("UniversalRetriever unavailable")
            self.retriever = UniversalRetriever(**self.retriever_kwargs)
        return self.retriever

    # ------------------------------------------------------------------
    def _parse_hits(self, hits: Iterable[Any], *, max_alert_severity: float = 1.0) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        filtered = 0
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
            sev = meta.get("alignment_severity")
            if sev is not None:
                try:
                    if float(sev) > max_alert_severity:
                        filtered += 1
                        continue
                except Exception:
                    pass
            lic = meta.get("license")
            fp_meta = meta.get("license_fingerprint")
            if fp_meta in _LICENSE_DENYLIST or lic in _DISALLOWED_LICENSES:
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
            lic = meta.get("license")
            fp = meta.get("license_fingerprint")
            if fp in _LICENSE_DENYLIST or lic in _DISALLOWED_LICENSES:
                filtered += 1
                continue
            item["metadata"] = meta
            if reason is not None:
                item["reason"] = reason
            item["license"] = lic
            item["license_fingerprint"] = fp
            item["semantic_alerts"] = meta.get("semantic_alerts")
            item["alignment_severity"] = meta.get("alignment_severity")
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
            if lic in _DISALLOWED_LICENSES:
                penalty += 1.0
            item["score"] = max(float(item.get("score", 0.0)) - penalty, 0.0)
            results.append(item)
        if filtered:
            try:
                _FILTERED_RESULTS.inc(filtered)
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------
    def _fallback(
        self, query: str, limit: int | None = None, *, max_alert_severity: float = 1.0
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
        for row in rows:
            text = str(row.get("code") or row.get("summary") or "")
            governed = govern_retrieval(text, max_alert_severity=max_alert_severity)
            if governed is None:
                filtered += 1
                continue
            meta, reason = governed
            fp = meta.get("license_fingerprint")
            item = {
                "origin_db": row.get("origin_db", "code"),
                "record_id": row.get("id") or row.get("record_id"),
                "score": 0.0,
                "metadata": meta,
                "text": text,
                "license": meta.get("license"),
                "license_fingerprint": fp,
                "semantic_alerts": meta.get("semantic_alerts"),
                "alignment_severity": meta.get("alignment_severity"),
            }
            if reason is not None:
                item["reason"] = reason
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
            if lic in _DISALLOWED_LICENSES:
                penalty += 1.0
            item["score"] = max(float(item.get("score", 0.0)) - penalty, 0.0)
            results.append(item)
        if filtered:
            try:
                _FILTERED_RESULTS.inc(filtered)
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
                    hits, confidence, _ = retriever.retrieve_with_confidence(  # type: ignore[attr-defined]
                        query, top_k=k
                    )
                else:
                    hits, _, _ = retriever.retrieve(query, top_k=k, dbs=dbs)  # type: ignore[arg-type]
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
                results = self._parse_hits(hits, max_alert_severity=sev_limit)
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
        fb_hits = self._fallback(query, limit=k, max_alert_severity=sev_limit)
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
        )

    # ------------------------------------------------------------------
    def error_frequency(self, error_id: int) -> float:
        """Expose raw error frequency metric from the underlying service."""

        try:
            return float(self._get_retriever()._error_frequency(int(error_id)))
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


def fts_search(
    query: str,
    *,
    dbs: Sequence[str] | None = None,
    limit: int | None = None,
) -> List[Dict[str, Any]]:
    """Run SQLite FTS queries via :class:`DatabaseRouter`.

    The helper is intentionally lightweight and returns an empty list on any
    failure so callers can use it opportunistically.
    """

    try:  # pragma: no cover - optional dependency
        from database_router import DatabaseRouter
    except Exception:
        return []

    try:
        router = DatabaseRouter()
        return router.search_fts(query, dbs=dbs, limit=limit)
    except Exception:
        return []


__all__ = ["Retriever", "FallbackResult", "fts_search"]

