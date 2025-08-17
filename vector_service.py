from __future__ import annotations

"""Vector service wrappers adding logging and metrics.

This module exposes small facades around existing retrieval and logging
components.  Each wrapper performs structured logging, exports metrics via
:mod:`metrics_exporter` and supports dependency injection for easy testing.

It mirrors the behaviour of :mod:`semantic_service` but is intentionally
light‑weight so other parts of the sandbox can depend on it without pulling in
heavy modules.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Callable, TypeVar, Union
import functools
import logging
import time
from logging_utils import get_logger, log_record

try:  # optional dependency
    from universal_retriever import UniversalRetriever, ResultBundle  # type: ignore
except Exception:  # pragma: no cover - fallback when running in isolation
    UniversalRetriever = None  # type: ignore
    ResultBundle = object  # type: ignore

try:
    from context_builder import ContextBuilder as _LegacyContextBuilder  # type: ignore
except Exception:  # pragma: no cover
    _LegacyContextBuilder = None  # type: ignore

try:
    from vector_metrics_db import VectorMetricsDB  # type: ignore
except Exception:  # pragma: no cover
    VectorMetricsDB = None  # type: ignore

try:
    from embeddable_db_mixin import EmbeddableDBMixin  # type: ignore
except Exception:  # pragma: no cover
    EmbeddableDBMixin = object  # type: ignore

try:
    import metrics_exporter as _me  # type: ignore
except Exception:  # pragma: no cover - when running inside package
    from . import metrics_exporter as _me  # type: ignore

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class VectorServiceError(RuntimeError):
    """Base exception for vector service failures."""


class RateLimitError(VectorServiceError):
    """Raised when the underlying service rate limits requests."""


class MalformedPromptError(VectorServiceError):
    """Raised when input prompts are malformed or empty."""


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])

_SUCCESS_COUNT = _me.Gauge(
    "vector_service_success_total",
    "Number of successful vector service calls",
    ["function"],
)
_FAILURE_COUNT = _me.Gauge(
    "vector_service_failure_total",
    "Number of failed vector service calls",
    ["function"],
)
_LATENCY_GAUGE = _me.Gauge(
    "vector_service_latency_seconds",
    "Execution time of vector service functions",
    ["function"],
)


def _result_size(result: Any) -> int:
    if hasattr(result, "__len__"):
        try:
            return len(result)  # type: ignore[arg-type]
        except Exception:
            return 0
    return 0


def log_and_metric(func: F) -> F:
    """Log call metadata and update Prometheus counters.

    The decorator records wall time and increments success/failure counters
    while logging structured information via :mod:`logging_utils`.
    """

    logger = get_logger(func.__module__)
    name = func.__qualname__

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        session_id = kwargs.get("session_id", "")
        start = time.time()

        req_meta = {
            "args": [repr(a) for a in args[1:]],  # skip ``self``
            "kwargs": {k: repr(v) for k, v in kwargs.items() if k != "session_id"},
        }

        try:
            result = func(*args, **kwargs)
        except Exception:
            latency = time.time() - start
            try:
                _FAILURE_COUNT.labels(name).inc()
                _LATENCY_GAUGE.labels(name).set(latency)
            except Exception:
                pass
            logger.exception(
                "%s failed", func.__qualname__,
                extra=log_record(
                    session_id=session_id,
                    latency=latency,
                    request=req_meta,
                    result_size=0,
                ),
            )
            raise

        latency = time.time() - start
        size = _result_size(result)
        try:
            _SUCCESS_COUNT.labels(name).inc()
            _LATENCY_GAUGE.labels(name).set(latency)
        except Exception:
            pass
        logger.info(
            "%s executed", func.__qualname__,
            extra=log_record(
                session_id=session_id,
                latency=latency,
                request=req_meta,
                response={"result_size": size},
            ),
        )
        return result

    return wrapper  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


@dataclass
class FallbackResult(Sequence[Dict[str, Any]]):
    """Container returned when semantic search falls back.

    Behaves like a sequence of result dictionaries while exposing the
    ``reason`` for the fallback and the confidence of the failed lookup.
    """

    reason: str
    results: List[Dict[str, Any]]
    confidence: float = 0.0

    def __iter__(self):  # pragma: no cover - delegation
        return iter(self.results)

    def __len__(self) -> int:  # pragma: no cover - delegation
        return len(self.results)

    def __getitem__(self, item: int) -> Dict[str, Any]:  # pragma: no cover
        return self.results[item]


@dataclass
class Retriever:
    """Facade around :class:`UniversalRetriever` with caching and retries.

    The retriever attempts ``retrieve_with_confidence`` when available and
    retries once with broader parameters when the returned confidence is below
    ``score_threshold``.  If all attempts fail a :class:`FallbackResult` is
    returned containing the reason and heuristic hits.
    """

    retriever: UniversalRetriever | None = None
    fallback_retriever: UniversalRetriever | None = None
    top_k: int = 5
    score_threshold: float = 0.0
    _cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def _get_retriever(self) -> UniversalRetriever:
        if self.retriever is None:
            if UniversalRetriever is None:  # pragma: no cover - defensive
                raise RuntimeError("UniversalRetriever unavailable")
            self.retriever = UniversalRetriever()
        return self.retriever

    def _get_fallback(self) -> UniversalRetriever | None:
        if self.fallback_retriever is not None:
            return self.fallback_retriever
        return None

    def _normalise(self, hits: Iterable[Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for h in hits:
            try:
                results.append(
                    {
                        "origin_db": getattr(h, "origin_db", ""),
                        "record_id": getattr(h, "record_id", ""),
                        "score": float(getattr(h, "score", 0.0)),
                        "metadata": getattr(h, "metadata", {}),
                        "reason": getattr(h, "reason", ""),
                    }
                )
            except Exception:
                continue
        return results

    def _retrieve(
        self, retriever: UniversalRetriever, query: str, k: int
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Run the underlying retriever and return hits with confidence."""

        try:
            if hasattr(retriever, "retrieve_with_confidence"):
                hits, confidence, _ = retriever.retrieve_with_confidence(  # type: ignore[attr-defined]
                    query, top_k=k
                )
            else:
                hits, _, _ = retriever.retrieve(query, top_k=k)  # type: ignore[arg-type]
                confidence = (
                    max(getattr(h, "score", 0.0) for h in hits) if hits else 0.0
                )
        except Exception as exc:
            msg = str(exc).lower()
            if "rate limit" in msg or "too many requests" in msg or "429" in msg:
                raise RateLimitError(str(exc)) from exc
            raise VectorServiceError(str(exc)) from exc
        normalised = self._normalise(hits)
        return normalised, float(confidence)

    def _fallback(self, reason: str) -> List[Dict[str, Any]]:
        return [
            {
                "origin_db": "heuristic",
                "record_id": None,
                "score": 0.0,
                "metadata": {},
                "reason": reason,
            }
        ]

    @log_and_metric
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        session_id: str = "",
        min_score: float | None = None,
        include_confidence: bool = False,
    ) -> Union[
        List[Dict[str, Any]],
        Tuple[List[Dict[str, Any]], float],
        FallbackResult,
        Tuple[FallbackResult, float],
    ]:
        """Perform semantic search and optionally return a confidence score.

        A primary retrieval is attempted and, when no hits are produced or the
        confidence falls below ``min_score`` (or ``score_threshold``), the
        search is retried once with broader parameters.  After exhausting these
        attempts a :class:`FallbackResult` is returned documenting the reason
        for the fallback.
        """

        if not isinstance(query, str) or not query.strip():
            raise MalformedPromptError("query must be a non-empty string")

        k = top_k or self.top_k
        threshold = self.score_threshold if min_score is None else min_score

        primary = self._get_retriever()
        secondary = self._get_fallback() or primary

        attempts = [(primary, k), (secondary, k * 2)]
        last_results: List[Dict[str, Any]] = []
        confidence = 0.0
        for retr, kk in attempts:
            results, confidence = self._retrieve(retr, query, kk)
            if results and confidence >= threshold:
                self._cache[query] = results
                if include_confidence:
                    return results, confidence
                return results
            last_results = results

        cached = self._cache.get(query)
        if cached is not None:
            if include_confidence:
                return cached, confidence
            return cached

        reason = "no results" if not last_results else "low confidence"
        fb_hits = self._fallback(reason)
        fb_result = FallbackResult(reason, fb_hits, confidence)
        if include_confidence:
            return fb_result, confidence
        return fb_result


@dataclass
class ErrorResult:
    """Standardised error structure returned by :meth:`ContextBuilder.build`."""

    error: str
    message: str


@dataclass
class ContextBuilder:
    """Thin wrapper around :func:`context_builder.ContextBuilder`.

    When downstream APIs raise ``ValueError`` or rate-limit the request the
    method returns an :class:`ErrorResult` instead of bubbling up raw
    exceptions.  All other exceptions are wrapped in :class:`VectorServiceError`.
    """

    builder: _LegacyContextBuilder | None = None

    def _get_builder(self) -> _LegacyContextBuilder:
        if self.builder is None:
            if _LegacyContextBuilder is None:  # pragma: no cover - defensive
                raise RuntimeError("ContextBuilder unavailable")
            self.builder = _LegacyContextBuilder()
        return self.builder

    @log_and_metric
    def build(
        self, task_description: str, *, session_id: str = "", **kwargs: Any
    ) -> Union[str, ErrorResult]:
        """Build a context string for ``task_description``.

        ``ValueError`` and rate‑limit responses from the downstream builder are
        captured and returned as :class:`ErrorResult` instances.  All other
        exceptions are wrapped in :class:`VectorServiceError`.
        """

        if not isinstance(task_description, str) or not task_description.strip():
            raise MalformedPromptError("task_description must be a non-empty string")
        builder = self._get_builder()
        try:
            return builder.build_context(task_description, **kwargs)
        except ValueError as exc:
            return ErrorResult("value_error", str(exc))
        except Exception as exc:  # pragma: no cover - best effort
            msg = str(exc).lower()
            if "rate limit" in msg or "too many requests" in msg or "429" in msg:
                return ErrorResult("rate_limited", str(exc))
            raise VectorServiceError(str(exc)) from exc


@dataclass
class PatchLogger:
    """Record patch outcomes in :class:`VectorMetricsDB` or legacy metrics DB."""

    metrics_db: Any | None = None
    vector_metrics: VectorMetricsDB | None = None

    def _parse_vectors(self, ids: Iterable[str]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for vid in ids:
            if ":" in vid:
                db, vec = vid.split(":", 1)
            else:
                db, vec = "", vid
            pairs.append((db, vec))
        return pairs

    def _get_db(self) -> VectorMetricsDB:
        if self.vector_metrics is None:
            if VectorMetricsDB is None:  # pragma: no cover - defensive
                raise VectorServiceError("VectorMetricsDB unavailable")
            self.vector_metrics = VectorMetricsDB()
        return self.vector_metrics

    @log_and_metric
    def track_contributors(
        self,
        vector_ids: Sequence[str],
        result: bool,
        *,
        patch_id: str = "",
        session_id: str = "",
    ) -> None:
        if not all(isinstance(v, str) for v in vector_ids):
            raise MalformedPromptError("vector_ids must be a sequence of strings")
        pairs = self._parse_vectors(vector_ids)
        if self.metrics_db is not None:
            try:
                self.metrics_db.log_patch_outcome(
                    patch_id, result, pairs, session_id=session_id
                )
            except Exception as exc:  # pragma: no cover - best effort
                raise VectorServiceError(str(exc)) from exc
            return
        db = self._get_db()
        try:
            db.update_outcome(
                session_id,
                pairs,
                contribution=0.0,
                patch_id=patch_id,
                win=result,
                regret=not result,
            )
        except Exception as exc:  # pragma: no cover - best effort
            raise VectorServiceError(str(exc)) from exc


class EmbeddingBackfill:
    """Trigger embedding backfills on all known database classes."""

    def __init__(self, batch_size: int = 100, backend: str = "annoy") -> None:
        self.batch_size = batch_size
        self.backend = backend

    def _load_known_dbs(self) -> List[type]:
        modules = [
            "bot_database",
            "task_handoff_bot",
            "error_bot",
            "information_db",
            "chatgpt_enhancement_bot",
            "research_aggregator_bot",
        ]
        for name in modules:  # pragma: no cover - best effort imports
            try:
                __import__(name)
            except Exception:
                pass
        try:
            subclasses = [
                cls
                for cls in EmbeddableDBMixin.__subclasses__()  # type: ignore[attr-defined]
                if hasattr(cls, "backfill_embeddings")
            ]
        except Exception:  # pragma: no cover - defensive
            subclasses = []
        return subclasses

    @log_and_metric
    def _process_db(
        self,
        db: EmbeddableDBMixin,
        *,
        batch_size: int,
        session_id: str = "",
    ) -> None:
        try:
            db.backfill_embeddings(batch_size=batch_size)  # type: ignore[call-arg]
        except TypeError:
            db.backfill_embeddings()  # type: ignore[call-arg]

    @log_and_metric
    def run(
        self,
        *,
        session_id: str = "",
        batch_size: int | None = None,
        backend: str | None = None,
    ) -> None:
        bs = batch_size if batch_size is not None else self.batch_size
        be = backend or self.backend
        subclasses = self._load_known_dbs()
        logger = logging.getLogger(__name__)
        total = len(subclasses)
        for idx, cls in enumerate(subclasses, 1):
            try:
                db = cls(vector_backend=be)  # type: ignore[call-arg]
            except Exception:  # pragma: no cover - fallback
                try:
                    db = cls()  # type: ignore[call-arg]
                except Exception:
                    continue
            logger.info(
                "Backfilling %s (%d/%d)",
                cls.__name__,
                idx,
                total,
                extra={"session_id": session_id},
            )
            try:
                self._process_db(db, batch_size=bs, session_id=session_id)
            except Exception:  # pragma: no cover - best effort
                continue


__all__ = [
    "Retriever",
    "FallbackResult",
    "ContextBuilder",
    "ErrorResult",
    "PatchLogger",
    "EmbeddingBackfill",
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
]
