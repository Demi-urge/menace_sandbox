from __future__ import annotations

"""High level semantic retrieval helpers.

This module groups together small wrappers around existing components so that
callers can depend on a unified interface.  The provided classes deliberately
thin and proxy to their counterparts in other modules.

Classes
-------
Retriever
    Convenience facade around :class:`universal_retriever.UniversalRetriever`.
ContextBuilder
    Lightweight wrapper exposing ``build`` around the historic
    :mod:`context_builder` module.
PatchLogger
    Helper for recording patch outcomes in :class:`data_bot.MetricsDB` and
    :class:`vector_metrics_db.VectorMetricsDB`.
EmbeddingBackfill
    Utility to backfill embeddings for all known databases inheriting from
    :class:`embeddable_db_mixin.EmbeddableDBMixin`.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .decorators import log_and_time, track_metrics
from .exceptions import (
    MalformedPromptError,
    RateLimitError,
    VectorServiceError,
)

try:  # Optional dependencies are imported lazily to avoid heavy startup cost
    from universal_retriever import UniversalRetriever, ResultBundle  # type: ignore
except Exception:  # pragma: no cover - fallback when running in isolation
    UniversalRetriever = None  # type: ignore
    ResultBundle = object  # type: ignore

try:
    from context_builder import ContextBuilder as _LegacyContextBuilder  # type: ignore
except Exception:  # pragma: no cover
    _LegacyContextBuilder = None  # type: ignore

try:
    from data_bot import MetricsDB  # type: ignore
except Exception:  # pragma: no cover
    MetricsDB = None  # type: ignore

try:
    from vector_metrics_db import VectorMetricsDB  # type: ignore
except Exception:  # pragma: no cover
    VectorMetricsDB = None  # type: ignore

try:
    from embeddable_db_mixin import EmbeddableDBMixin  # type: ignore
except Exception:  # pragma: no cover
    EmbeddableDBMixin = object  # type: ignore


@dataclass
class Retriever:
    """Facade around :class:`UniversalRetriever`.

    Parameters
    ----------
    retriever:
        Optional pre‑instantiated :class:`UniversalRetriever`.  When omitted a
        default instance is constructed on first use.
    top_k:
        Number of results to return from :meth:`search`.
    """

    retriever: UniversalRetriever | None = None
    top_k: int = 5
    _cache: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def _get_retriever(self) -> UniversalRetriever:
        if self.retriever is None:
            if UniversalRetriever is None:  # pragma: no cover - defensive
                raise RuntimeError("UniversalRetriever unavailable")
            self.retriever = UniversalRetriever()
        return self.retriever

    @log_and_time
    @track_metrics
    def search(
        self, query: str, *, top_k: int | None = None, session_id: str = ""
    ) -> List[Dict[str, Any]]:
        """Perform semantic search and return normalised results.

        The underlying :class:`UniversalRetriever` returns ``ResultBundle``
        objects.  This method converts them into serialisable dictionaries with
        a consistent shape:

        ``{"origin_db": ..., "record_id": ..., "score": ..., "metadata": ..., "reason": ...}``

        Parameters
        ----------
        query:
            Search query string.
        top_k:
            Optional limit for number of results.
        session_id:
            Optional identifier used for structured logging.
        """

        if not isinstance(query, str) or not query.strip():
            raise MalformedPromptError("query must be a non-empty string")

        k = top_k or self.top_k
        retriever = self._get_retriever()
        attempts = 3
        backoff = 1.0
        hits: List[Any] = []
        for attempt in range(attempts):
            try:
                try:
                    hits, _, _ = retriever.retrieve(query, top_k=k)  # type: ignore[arg-type]
                except AttributeError:  # pragma: no cover - older interface
                    hits = retriever.search(query)[:k]  # type: ignore[assignment]
                break
            except Exception as exc:  # pragma: no cover - best effort
                msg = str(exc).lower()
                if ("rate" in msg and "limit" in msg) or "429" in msg:
                    if attempt < attempts - 1:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise RateLimitError("vector search rate limited") from exc
                raise VectorServiceError("vector search failed") from exc

        if not hits:
            cached = self._cache.get(query)
            if cached is not None:
                return cached
            return [
                {
                    "origin_db": "heuristic",
                    "record_id": None,
                    "score": 0.0,
                    "metadata": {},
                    "reason": "no results",
                }
            ]

        top_score = max(getattr(h, "score", 0.0) for h in hits)
        if top_score < 0.1:
            cached = self._cache.get(query)
            if cached is not None:
                return cached
            return [
                {
                    "origin_db": "heuristic",
                    "record_id": None,
                    "score": 0.0,
                    "metadata": {},
                    "reason": "low similarity",
                }
            ]

        results: List[Dict[str, Any]] = []
        for h in hits:
            if isinstance(h, ResultBundle):
                item = h.to_dict()
                item["record_id"] = getattr(h, "record_id", None)
            elif hasattr(h, "to_dict"):
                item = h.to_dict()
                item.setdefault("record_id", getattr(h, "record_id", None))
            else:  # pragma: no cover - unlikely
                meta = getattr(h, "metadata", {})
                item = {
                    "origin_db": getattr(h, "origin_db", ""),
                    "record_id": getattr(h, "record_id", None),
                    "score": getattr(h, "score", 0.0),
                    "reason": getattr(h, "reason", ""),
                    "metadata": meta,
                }
            results.append(item)
        self._cache[query] = results
        return results


class ContextBuilder:
    """Wrapper exposing a ``build`` method compatible with existing code."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _LegacyContextBuilder is None:  # pragma: no cover - defensive
            raise RuntimeError("context_builder module unavailable")
        self._builder = _LegacyContextBuilder(*args, **kwargs)
        self._cache: Dict[str, str] = {}

    @log_and_time
    @track_metrics
    def build(self, task_description: str, **kwargs: Any) -> str:
        """Return a compact JSON context for ``task_description``."""

        if not isinstance(task_description, str) or not task_description.strip():
            raise MalformedPromptError("task_description must be a non-empty string")

        retriever = getattr(self._builder, "retriever", None)
        if retriever is None:  # pragma: no cover - defensive
            raise VectorServiceError("retriever unavailable")

        attempts = 3
        backoff = 1.0
        hits: List[Any] = []
        for attempt in range(attempts):
            try:
                hits, _, _ = retriever.retrieve(task_description, top_k=1)
                break
            except Exception as exc:  # pragma: no cover - best effort
                msg = str(exc).lower()
                if ("rate" in msg and "limit" in msg) or "429" in msg:
                    if attempt < attempts - 1:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise RateLimitError("vector search rate limited") from exc
                raise VectorServiceError("vector search failed") from exc

        if not hits or max(getattr(h, "score", 0.0) for h in hits) < 0.1:
            cached = self._cache.get(task_description)
            if cached is not None:
                return cached
            return json.dumps({"note": "insufficient context"})

        context = ""
        backoff = 1.0
        for attempt in range(attempts):
            try:
                context = self._builder.build_context(task_description, **kwargs)
                break
            except Exception as exc:  # pragma: no cover - best effort
                msg = str(exc).lower()
                if ("rate" in msg and "limit" in msg) or "429" in msg:
                    if attempt < attempts - 1:
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise RateLimitError("vector search rate limited") from exc
                raise VectorServiceError("context build failed") from exc

        self._cache[task_description] = context
        return context


class PatchLogger:
    """Record patch outcomes for contributing vectors."""

    def __init__(
        self,
        metrics_db: MetricsDB | None = None,
        vector_metrics: VectorMetricsDB | None = None,
    ) -> None:
        self.metrics_db = metrics_db or (MetricsDB() if MetricsDB is not None else None)
        self.vector_metrics = vector_metrics or (
            VectorMetricsDB() if VectorMetricsDB is not None else None
        )

    def _parse_vectors(self, vector_ids: Iterable[str]) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        for vid in vector_ids:
            if ":" in vid:
                origin, vec_id = vid.split(":", 1)
            else:
                origin, vec_id = "", vid
            pairs.append((origin, vec_id))
        return pairs

    @log_and_time
    @track_metrics
    def track_contributors(
        self, vector_ids: Sequence[str], result: bool, *, patch_id: str = "", session_id: str = ""
    ) -> None:
        """Log patch outcome for vectors contributing to a patch.

        Parameters
        ----------
        vector_ids:
            Iterable of vector identifiers.  Entries may be plain IDs or
            ``"origin_db:vector_id"`` pairs.
        result:
            ``True`` for a successful patch, ``False`` otherwise.
        patch_id:
            Optional identifier of the patch being logged.
        session_id:
            Retrieval session identifier linking patches to retrieval events.
        """

        pairs = self._parse_vectors(vector_ids)
        if self.metrics_db is not None:
            try:  # pragma: no cover - best effort
                self.metrics_db.log_patch_outcome(
                    patch_id or "", result, pairs, session_id=session_id
                )
            except Exception:
                pass
        elif self.vector_metrics is not None:
            try:  # pragma: no cover - best effort
                self.vector_metrics.update_outcome(
                    session_id, pairs, contribution=0.0, patch_id=patch_id, win=result, regret=not result
                )
            except Exception:
                pass


class EmbeddingBackfill:
    """Trigger embedding backfills on all known database classes."""

    def __init__(self, batch_size: int = 100, backend: str = "annoy") -> None:
        self.batch_size = batch_size
        self.backend = backend

    def _load_known_dbs(self) -> List[type]:
        """Import common database modules and return discovered classes.

        Importing ensures their classes register as subclasses of
        :class:`EmbeddableDBMixin`.
        """

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
                cls for cls in EmbeddableDBMixin.__subclasses__()
                if hasattr(cls, "backfill_embeddings")
            ]
        except Exception:  # pragma: no cover - defensive
            subclasses = []
        return subclasses

    @log_and_time
    @track_metrics
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

    @log_and_time
    @track_metrics
    def run(
        self,
        *,
        session_id: str = "",
        batch_size: int | None = None,
        backend: str | None = None,
    ) -> None:
        """Backfill embeddings for all ``EmbeddableDBMixin`` subclasses."""

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
    "ContextBuilder",
    "PatchLogger",
    "EmbeddingBackfill",
    "VectorServiceError",
    "RateLimitError",
    "MalformedPromptError",
]
