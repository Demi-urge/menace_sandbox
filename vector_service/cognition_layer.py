"""High level orchestration layer around vector retrieval.

The :class:`CognitionLayer` combines the retriever, context builder,
patch logger and metrics database into a single convenience facade.  It
exposes a small API used by services that want to perform a retrieval
request and later record the outcome of a patch based on that retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .retriever import Retriever
from .context_builder import ContextBuilder
from .patch_logger import PatchLogger
from vector_metrics_db import VectorMetricsDB
from .decorators import log_and_measure


class CognitionLayer:
    """Tie together retrieval, context building and patch logging."""

    def __init__(
        self,
        *,
        retriever: Retriever | None = None,
        context_builder: ContextBuilder | None = None,
        patch_logger: PatchLogger | None = None,
        vector_metrics: VectorMetricsDB | None = None,
    ) -> None:
        self.retriever = retriever or Retriever()
        self.vector_metrics = vector_metrics or VectorMetricsDB()
        self.context_builder = context_builder or ContextBuilder(
            retriever=self.retriever
        )
        self.patch_logger = patch_logger or PatchLogger(
            vector_metrics=self.vector_metrics
        )
        # Keep track of vectors by session so outcomes can be recorded later
        self._session_vectors: Dict[str, List[Tuple[str, str, float]]] = {}
        self._retrieval_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}

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

        context, sid, vectors = self.context_builder.build_context(
            prompt,
            top_k=top_k,
            include_vectors=True,
            session_id=session_id,
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
                    tokens=0,
                    wall_time_ms=0.0,
                    hit=True,
                    rank=rank,
                    contribution=0.0,
                    prompt_tokens=0,
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
        :meth:`PatchLogger.track_contributors`.
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


__all__ = ["CognitionLayer"]
