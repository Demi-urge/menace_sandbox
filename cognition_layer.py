"""High level helpers wrapping vector retrieval, feedback logging and model maintenance.

This module provides a small convenience API for bots and workflows that
need to build a cognitive context for a query, later log feedback about the
resulting patch, reload ranking models, refresh reliability metrics and
inspect ROI statistics.  Internally it wires together the vector service
:class:`ContextBuilder`, :class:`PatchSafety` and :class:`ROITracker` through
:class:`vector_service.cognition_layer.CognitionLayer`.  The module advertises
the active bootstrap placeholder at import time so downstream helpers reuse the
shared promise rather than triggering a cascading
``prepare_pipeline_for_bootstrap`` chain during cognition warm-ups. Maintain
the broker-first bootstrap pattern referenced in
``docs/bootstrap_troubleshooting.md`` when updating these helpers.

Example
-------
>>> from cognition_layer import (
...     build_cognitive_context,
...     log_feedback,
...     reload_ranker_model,
...     get_roi_stats,
... )
>>> builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
>>> context, sid = build_cognitive_context("improve error handling", context_builder=builder)
>>> log_feedback(sid, success=True, context_builder=builder)
>>> reload_ranker_model(context_builder=builder)
>>> stats = get_roi_stats(context_builder=builder)
"""

from __future__ import annotations

from typing import Any, Tuple

from vector_service.context_builder import ContextBuilder
from coding_bot_interface import advertise_bootstrap_placeholder, get_active_bootstrap_pipeline
from bootstrap_gate import resolve_bootstrap_placeholders
from bootstrap_helpers import ensure_bootstrapped
from vector_service.cognition_layer import CognitionLayer as _CognitionLayer
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

# Global ROI tracker shared by all calls so ROI metrics accumulate across
# retrieval sessions.  Context builders are supplied by callers to avoid
# implicit global state.
_roi_tracker = ROITracker()
_BOOTSTRAP_PLACEHOLDER_PIPELINE: object | None = None
_BOOTSTRAP_PLACEHOLDER_MANAGER: object | None = None
_BOOTSTRAP_GATE_TIMEOUT = 12.0


def _bootstrap_placeholders() -> tuple[object, object]:
    """Resolve bootstrap placeholders after the readiness gate clears."""

    global _BOOTSTRAP_PLACEHOLDER_PIPELINE, _BOOTSTRAP_PLACEHOLDER_MANAGER
    if None not in (_BOOTSTRAP_PLACEHOLDER_PIPELINE, _BOOTSTRAP_PLACEHOLDER_MANAGER):
        return _BOOTSTRAP_PLACEHOLDER_PIPELINE, _BOOTSTRAP_PLACEHOLDER_MANAGER

    pipeline, manager, broker = resolve_bootstrap_placeholders(
        timeout=_BOOTSTRAP_GATE_TIMEOUT,
        description="CognitionLayer bootstrap gate",
    )
    _BOOTSTRAP_PLACEHOLDER_PIPELINE, _BOOTSTRAP_PLACEHOLDER_MANAGER = advertise_bootstrap_placeholder(
        dependency_broker=broker,
        pipeline=pipeline,
        manager=manager,
    )
    return _BOOTSTRAP_PLACEHOLDER_PIPELINE, _BOOTSTRAP_PLACEHOLDER_MANAGER


def _get_layer(builder: ContextBuilder) -> _CognitionLayer:
    """Return cognition layer bound to *builder*.

    The layer instance is cached on ``builder`` to avoid recreating heavy
    components across calls.
    """

    # Guard against recursive bootstrap from cognition entrypoints; rely on the
    # shared readiness snapshot before building the layer cache.
    ensure_bootstrapped()
    _bootstrap_placeholders()
    layer = getattr(builder, "_cognition_layer", None)
    if layer is None:
        layer = _CognitionLayer(context_builder=builder, roi_tracker=_roi_tracker)
        try:
            setattr(builder, "_cognition_layer", layer)
        except Exception:  # pragma: no cover - builder may be immutable
            pass
    return layer


def build_cognitive_context(
    query: str,
    *,
    context_builder: ContextBuilder,
    **kwargs: Any,
) -> Tuple[str, str]:
    """Return context and session id for *query*.

    Parameters
    ----------
    query:
        Natural language description of the desired context.
    **kwargs:
        Additional keyword arguments forwarded to
        :meth:`vector_service.cognition_layer.CognitionLayer.query`.
    """

    layer = _get_layer(context_builder)
    return layer.query(query, **kwargs)


async def build_cognitive_context_async(
    query: str,
    *,
    context_builder: ContextBuilder,
    **kwargs: Any,
) -> Tuple[str, str]:
    """Asynchronously return context and session id for *query*.

    Parameters
    ----------
    query:
        Natural language description of the desired context.
    **kwargs:
        Additional keyword arguments forwarded to
        :meth:`vector_service.cognition_layer.CognitionLayer.query_async`.
    """

    layer = _get_layer(context_builder)
    return await layer.query_async(query, **kwargs)


def log_feedback(
    session_id: str,
    success: bool,
    *,
    patch_id: str = "",
    contribution: float | None = None,
    context_builder: ContextBuilder,
) -> None:
    """Record feedback for a previously built context.

    This forwards the outcome to the underlying
    :class:`vector_service.cognition_layer.CognitionLayer` which updates the
    ranking model and ROI metrics.
    """

    layer = _get_layer(context_builder)
    layer.record_patch_outcome(
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
    context_builder: ContextBuilder,
) -> None:
    """Record feedback asynchronously for a previously built context.

    This forwards the outcome to the underlying
    :class:`vector_service.cognition_layer.CognitionLayer` which updates the
    ranking model and ROI metrics.
    """

    layer = _get_layer(context_builder)
    await layer.record_patch_outcome_async(
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
    context_builder: ContextBuilder,
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

    layer = _get_layer(context_builder)
    layer.reload_ranker_model(
        model_path, roi_delta=roi_delta, risk_penalty=risk_penalty
    )


def reload_reliability_scores(*, context_builder: ContextBuilder) -> None:
    """Refresh retriever reliability statistics."""

    layer = _get_layer(context_builder)
    layer.reload_reliability_scores()


def get_roi_stats(
    *, context_builder: ContextBuilder,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return latest ROI statistics grouped by origin type."""

    layer = _get_layer(context_builder)
    return layer.roi_stats()
