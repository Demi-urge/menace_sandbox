"""High level orchestration layer around vector retrieval.

The :class:`CognitionLayer` combines the retriever, context builder,
patch logger and metrics database into a single convenience facade.  It
exposes a small API used by services that want to perform a retrieval
request and later record the outcome of a patch based on that retrieval.

An optional :class:`roi_tracker.ROITracker` instance can be supplied to
record ROI deltas for the origin databases contributing to a patch::

    from roi_tracker import ROITracker
    from vector_service.cognition_layer import CognitionLayer

    tracker = ROITracker()
    layer = CognitionLayer(roi_tracker=tracker)
    ctx, sid = layer.query("What is ROI?")
    # ... apply patch ...
    layer.record_patch_outcome(sid, True, contribution=1.0)

This automatically forwards the patch outcome to ``tracker`` so ROI
histories stay up to date.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .retriever import Retriever
from .context_builder import ContextBuilder
from .patch_logger import PatchLogger
from vector_metrics_db import VectorMetricsDB
from .decorators import log_and_measure

try:  # pragma: no cover - optional dependency
    from roi_tracker import ROITracker  # type: ignore
except Exception:  # pragma: no cover - ROI tracking optional
    ROITracker = None  # type: ignore


class CognitionLayer:
    """Tie together retrieval, context building and patch logging.

    Parameters
    ----------
    roi_tracker:
        Optional :class:`roi_tracker.ROITracker` instance used to update
        ROI histories when recording patch outcomes.  If not provided and
        the tracker can be imported, a default instance is created.
    """

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        context_builder: ContextBuilder | None = None,
        patch_logger: PatchLogger | None = None,
        vector_metrics: VectorMetricsDB | None = None,
        roi_tracker: ROITracker | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.vector_metrics = vector_metrics or VectorMetricsDB()
        self.context_builder = context_builder or ContextBuilder(
            retriever=self.retriever
        )
        self.roi_tracker = roi_tracker or (ROITracker() if ROITracker is not None else None)
        self.patch_logger = patch_logger or PatchLogger(
            vector_metrics=self.vector_metrics,
            roi_tracker=self.roi_tracker,
        )
        # Keep track of vectors by session so outcomes can be recorded later
        self._session_vectors: Dict[str, List[Tuple[str, str, float]]] = {}
        self._retrieval_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    def reload_ranker_model(self, model_path: str | "Path") -> None:
        """Reload ranking model on retriever and context builder."""

        try:
            self.retriever.reload_ranker_model(model_path)  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            from pathlib import Path

            try:  # pragma: no cover - package relative import
                from .. import retrieval_ranker as _rr  # type: ignore
            except Exception:  # pragma: no cover - fallback
                import retrieval_ranker as _rr  # type: ignore

            self.context_builder.ranking_model = _rr.load_model(Path(model_path))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def reload_reliability_scores(self) -> None:
        """Refresh retriever reliability statistics."""

        try:
            self.retriever.reload_reliability_scores()  # type: ignore[attr-defined]
        except Exception:
            pass

    # ------------------------------------------------------------------
    @log_and_measure
    def query(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        session_id: str = "",
    ) -> Tuple[str, str]:
        """Retrieve context for *prompt* and store vector metrics.

        Returns a tuple of ``(context, session_id)``.  The ``session_id`` can
        later be passed to :meth:`record_patch_outcome` to update metrics once
        the patch succeeds or fails.
        """

        context, sid, vectors, stats = self.context_builder.build_context(
            prompt,
            top_k=top_k,
            include_vectors=True,
            session_id=session_id,
            return_stats=True,
        )

        self._session_vectors[sid] = vectors
        # Prepare metadata mapping for later use by PatchLogger
        meta: Dict[str, Dict[str, Any]] = {}
        for rank, (origin, vec_id, score) in enumerate(vectors, start=1):
            key = f"{origin}:{vec_id}" if origin else vec_id
            meta[key] = {}
            try:  # Best effort metrics logging
                self.vector_metrics.log_retrieval(
                    origin,
                    tokens=stats["tokens"],
                    wall_time_ms=stats["wall_time_ms"],
                    hit=True,
                    rank=rank,
                    contribution=0.0,
                    prompt_tokens=stats["prompt_tokens"],
                    session_id=sid,
                    vector_id=vec_id,
                    similarity=score,
                    context_score=score,
                    age=0.0,
                )
            except Exception:
                pass
        self._retrieval_meta[sid] = meta
        return context, sid

    # ------------------------------------------------------------------
    @log_and_measure
    async def query_async(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        session_id: str = "",
    ) -> Tuple[str, str]:
        """Asynchronous wrapper for :meth:`query`."""

        context, sid, vectors, stats = await self.context_builder.build_async(
            prompt,
            top_k=top_k,
            include_vectors=True,
            session_id=session_id,
            return_stats=True,
        )

        self._session_vectors[sid] = vectors
        meta: Dict[str, Dict[str, Any]] = {}
        for rank, (origin, vec_id, score) in enumerate(vectors, start=1):
            key = f"{origin}:{vec_id}" if origin else vec_id
            meta[key] = {}
            try:  # Best effort metrics logging
                self.vector_metrics.log_retrieval(
                    origin,
                    tokens=stats["tokens"],
                    wall_time_ms=stats["wall_time_ms"],
                    hit=True,
                    rank=rank,
                    contribution=0.0,
                    prompt_tokens=stats["prompt_tokens"],
                    session_id=sid,
                    vector_id=vec_id,
                    similarity=score,
                    context_score=score,
                    age=0.0,
                )
            except Exception:
                pass
        self._retrieval_meta[sid] = meta
        return context, sid

    # ------------------------------------------------------------------
    def record_patch_outcome(
        self,
        session_id: str,
        success: bool,
        *,
        patch_id: str = "",
        contribution: float | None = None,
    ) -> None:
        """Forward patch outcome to :class:`PatchLogger`.

        ``session_id`` must match the value returned from :meth:`query`.  The
        stored vector identifiers will be passed to
        :meth:`PatchLogger.track_contributors`.  When an
        :class:`roi_tracker.ROITracker` is configured, ROI metrics for each
        origin database are updated automatically based on ``contribution`` or
        similarity scores.
        """

        vectors = self._session_vectors.pop(session_id, [])
        meta = self._retrieval_meta.pop(session_id, None)
        if not vectors:
            return
        vec_ids = [(f"{o}:{vid}", score) for o, vid, score in vectors]
        self.patch_logger.track_contributors(
            vec_ids,
            success,
            patch_id=patch_id,
            session_id=session_id,
            contribution=contribution,
            retrieval_metadata=meta,
        )

        if self.roi_tracker is not None:
            try:  # pragma: no cover - best effort
                cur = self.vector_metrics.conn.execute(
                    """
                    SELECT db, tokens, contribution, hit
                      FROM vector_metrics
                     WHERE session_id=? AND event_type='retrieval'
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
                roi_after = sum(float(r[2] or 0.0) for r in rows)
                retrieval_metrics = [
                    {
                        "origin_db": str(db),
                        "tokens": float(tokens or 0.0),
                        "hit": bool(hit),
                    }
                    for db, tokens, _contrib, hit in rows
                ]
                self.roi_tracker.update(
                    0.0,
                    roi_after,
                    retrieval_metrics=retrieval_metrics,
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    async def record_patch_outcome_async(
        self,
        session_id: str,
        success: bool,
        *,
        patch_id: str = "",
        contribution: float | None = None,
    ) -> None:
        """Asynchronous wrapper for :meth:`record_patch_outcome`."""

        vectors = self._session_vectors.pop(session_id, [])
        meta = self._retrieval_meta.pop(session_id, None)
        if not vectors:
            return
        vec_ids = [(f"{o}:{vid}", score) for o, vid, score in vectors]
        await self.patch_logger.track_contributors_async(
            vec_ids,
            success,
            patch_id=patch_id,
            session_id=session_id,
            contribution=contribution,
            retrieval_metadata=meta,
        )


__all__ = ["CognitionLayer"]
