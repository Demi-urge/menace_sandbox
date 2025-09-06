"""High level helpers wrapping vector retrieval, feedback logging and model maintenance.

This module provides a small convenience API for bots and workflows that
need to build a cognitive context for a query, later log feedback about the
resulting patch, reload ranking models, refresh reliability metrics and
inspect ROI statistics.  Internally it wires together the vector service
:class:`ContextBuilder`, :class:`PatchSafety` and :class:`ROITracker` through
:class:`vector_service.cognition_layer.CognitionLayer`.

Example
-------
>>> from cognition_layer import (
...     build_cognitive_context,
...     log_feedback,
...     reload_ranker_model,
...     get_roi_stats,
... )
>>> context, sid = build_cognitive_context("improve error handling")
>>> log_feedback(sid, success=True)
>>> reload_ranker_model()
>>> stats = get_roi_stats()
"""

from __future__ import annotations

from typing import Any, Tuple

try:
    from vector_service import ContextBuilder, get_default_context_builder
except ImportError:  # pragma: no cover - fallback when helper missing
    from vector_service import ContextBuilder  # type: ignore

    def get_default_context_builder(**kwargs):  # type: ignore
        return ContextBuilder(**kwargs)
from vector_service.cognition_layer import CognitionLayer as _CognitionLayer
from patch_safety import PatchSafety
from roi_tracker import ROITracker

__all__ = [
    "build_cognitive_context",
    "build_cognitive_context_async",
    "log_feedback",
    "log_feedback_async",
    "reload_ranker_model",
    "reload_reliability_scores",
    "get_roi_stats",
]

# Global components shared by all calls.  They ensure every query flows
# through retrieval, ranking, patch safety and ROI tracking.
_roi_tracker = ROITracker()
_patch_safety = PatchSafety()
_context_builder = get_default_context_builder(
    roi_tracker=_roi_tracker, patch_safety=_patch_safety
)
_layer = _CognitionLayer(context_builder=_context_builder, roi_tracker=_roi_tracker)


def build_cognitive_context(query: str, **kwargs: Any) -> Tuple[str, str]:
    """Return context and session id for *query*.

    Parameters
    ----------
    query:
        Natural language description of the desired context.
    **kwargs:
        Additional keyword arguments forwarded to
        :meth:`vector_service.cognition_layer.CognitionLayer.query`.
    """

    return _layer.query(query, **kwargs)


async def build_cognitive_context_async(query: str, **kwargs: Any) -> Tuple[str, str]:
    """Asynchronously return context and session id for *query*.

    Parameters
    ----------
    query:
        Natural language description of the desired context.
    **kwargs:
        Additional keyword arguments forwarded to
        :meth:`vector_service.cognition_layer.CognitionLayer.query_async`.
    """

    return await _layer.query_async(query, **kwargs)


def log_feedback(
    session_id: str,
    success: bool,
    *,
    patch_id: str = "",
    contribution: float | None = None,
) -> None:
    """Record feedback for a previously built context.

    This forwards the outcome to the underlying
    :class:`vector_service.cognition_layer.CognitionLayer` which updates the
    ranking model and ROI metrics.
    """

    _layer.record_patch_outcome(
        session_id,
        success,
        patch_id=patch_id,
        contribution=contribution,
    )


async def log_feedback_async(
    session_id: str,
    success: bool,
    *,
    patch_id: str = "",
    contribution: float | None = None,
) -> None:
    """Record feedback asynchronously for a previously built context.

    This forwards the outcome to the underlying
    :class:`vector_service.cognition_layer.CognitionLayer` which updates the
    ranking model and ROI metrics.
    """

    await _layer.record_patch_outcome_async(
        session_id,
        success,
        patch_id=patch_id,
        contribution=contribution,
    )


def reload_ranker_model(
    model_path: str | "Path" | None = None,
    *,
    roi_delta: float | None = None,
    risk_penalty: float | None = None,
) -> None:
    """Reload the retrieval ranking model used by the cognition layer.

    Parameters
    ----------
    model_path:
        Optional path to the ranking model on disk. When omitted the path is
        read from ``retrieval_ranker.json``.
    roi_delta:
        Optional ROI delta that can trigger an asynchronous retrain.
    risk_penalty:
        Optional risk penalty that can trigger an asynchronous retrain.
    """

    _layer.reload_ranker_model(
        model_path, roi_delta=roi_delta, risk_penalty=risk_penalty
    )


def reload_reliability_scores() -> None:
    """Refresh retriever reliability statistics."""

    _layer.reload_reliability_scores()


def get_roi_stats() -> dict[str, dict[str, dict[str, float]]]:
    """Return latest ROI statistics grouped by origin type."""

    return _layer.roi_stats()
