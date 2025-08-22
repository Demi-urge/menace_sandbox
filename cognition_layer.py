"""High level helpers wrapping vector retrieval and feedback logging.

This module provides a small convenience API for bots and workflows that
need to build a cognitive context for a query and later log feedback about
the resulting patch.  Internally it wires together the vector service
:class:`ContextBuilder`, :class:`PatchSafety` and :class:`ROITracker` through
:class:`vector_service.cognition_layer.CognitionLayer`.
"""

from __future__ import annotations

from typing import Any, Tuple

from vector_service.context_builder import ContextBuilder
from vector_service.cognition_layer import CognitionLayer as _CognitionLayer
from patch_safety import PatchSafety
from roi_tracker import ROITracker

__all__ = [
    "build_cognitive_context",
    "build_cognitive_context_async",
    "log_feedback",
    "log_feedback_async",
]

# Global components shared by all calls.  They ensure every query flows
# through retrieval, ranking, patch safety and ROI tracking.
_roi_tracker = ROITracker()
_patch_safety = PatchSafety()
_context_builder = ContextBuilder(roi_tracker=_roi_tracker, patch_safety=_patch_safety)
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
