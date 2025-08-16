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

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .decorators import log_and_time, track_metrics

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

        k = top_k or self.top_k
        retriever = self._get_retriever()
        try:
            hits, _, _ = retriever.retrieve(query, top_k=k)  # type: ignore[arg-type]
        except AttributeError:  # pragma: no cover - older interface
            hits = retriever.search(query)[:k]  # type: ignore[assignment]
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
        return results


class ContextBuilder:
    """Wrapper exposing a ``build`` method compatible with existing code."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if _LegacyContextBuilder is None:  # pragma: no cover - defensive
            raise RuntimeError("context_builder module unavailable")
        self._builder = _LegacyContextBuilder(*args, **kwargs)

    @log_and_time
    @track_metrics
    def build(self, task_description: str, **kwargs: Any) -> str:
        """Return a compact JSON context for ``task_description``."""

        return self._builder.build_context(task_description, **kwargs)


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

    def __init__(self, batch_size: int = 100) -> None:
        self.batch_size = batch_size

    def _load_known_dbs(self) -> None:
        """Best-effort import of common database modules.

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

    @log_and_time
    @track_metrics
    def run(self, *, session_id: str = "") -> None:
        """Backfill embeddings for all ``EmbeddableDBMixin`` subclasses."""

        self._load_known_dbs()
        try:
            subclasses = list(EmbeddableDBMixin.__subclasses__())
        except Exception:  # pragma: no cover - defensive
            subclasses = []
        for cls in subclasses:
            if not hasattr(cls, "backfill_embeddings"):
                continue
            try:
                db = cls()  # type: ignore[call-arg]
                db.backfill_embeddings(self.batch_size)
            except Exception:  # pragma: no cover - best effort
                continue


__all__ = [
    "Retriever",
    "ContextBuilder",
    "PatchLogger",
    "EmbeddingBackfill",
]
