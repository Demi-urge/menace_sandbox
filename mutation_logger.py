from __future__ import annotations

"""Thread-safe mutation logging helpers."""

import logging
from threading import Lock, Thread
from contextlib import contextmanager
from typing import Optional, Dict, Generator

from .retry_utils import publish_with_retry

from .evolution_history_db import (
    EvolutionEvent,
    EvolutionHistoryDB,
)

try:  # optional dependency
    from .unified_event_bus import UnifiedEventBus  # type: ignore
except Exception:  # pragma: no cover - bus optional
    UnifiedEventBus = None  # type: ignore

_logger = logging.getLogger(__name__)

# Path for the evolution history database
_DB_PATH = "evolution_history.db"

# Create a single shared database instance for logging mutations.
_history_db = EvolutionHistoryDB(_DB_PATH)

_lock = Lock()

_event_bus: Optional[UnifiedEventBus]
if UnifiedEventBus is not None:
    try:
        _event_bus = UnifiedEventBus()
    except Exception:  # pragma: no cover - optional runtime dependency
        _event_bus = None
else:  # pragma: no cover - bus not available
    _event_bus = None


def set_event_bus(bus: Optional[UnifiedEventBus]) -> None:
    """Override the global event bus instance.

    This allows tests or applications to provide a shared bus.
    """
    global _event_bus
    _event_bus = bus


def log_mutation(
    change: str,
    reason: str,
    trigger: str,
    performance: float,
    workflow_id: int,
    before_metric: float = 0.0,
    after_metric: float = 0.0,
    parent_id: int | None = None,
) -> int:
    """Record a mutation event and return its row id.

    ``roi`` is computed automatically from ``before_metric`` and
    ``after_metric``.
    """
    roi = after_metric - before_metric
    event = EvolutionEvent(
        action=change,
        before_metric=before_metric,
        after_metric=after_metric,
        roi=roi,
        reason=reason,
        trigger=trigger,
        performance=performance,
        workflow_id=workflow_id,
        parent_event_id=parent_id,
    )
    with _lock:
        event_id = _history_db.add(event)
    if _event_bus is not None:
        payload = {"event_id": event_id, **event.__dict__}

        def _publish() -> None:
            try:
                publish_with_retry(_event_bus, "mutation_recorded", payload, delay=0.1)
            except Exception as exc:  # pragma: no cover - best effort
                _logger.error("failed publishing mutation_recorded: %s", exc)

        Thread(target=_publish, daemon=True).start()
    return event_id


def log_workflow_evolution(
    workflow_id: int,
    variant: str,
    baseline_roi: float,
    variant_roi: float,
    *,
    mutation_id: int | None = None,
) -> int:
    """Record evaluation of a workflow variant and return the mutation id."""

    roi_delta = variant_roi - baseline_roi
    if mutation_id is None:
        mutation_id = log_mutation(
            change=variant,
            reason="variant",
            trigger="workflow_evolution",
            performance=roi_delta,
            workflow_id=workflow_id,
            before_metric=baseline_roi,
            after_metric=variant_roi,
        )
    with _lock:
        _history_db.log_workflow_evolution(
            workflow_id=workflow_id,
            variant=variant,
            baseline_roi=baseline_roi,
            variant_roi=variant_roi,
            roi_delta=roi_delta,
            mutation_id=mutation_id,
        )
    return mutation_id


def record_mutation_outcome(
    event_id: int,
    after_metric: float,
    roi: float,
    performance: float,
) -> None:
    """Record outcome metrics for an existing mutation event."""

    with _lock:
        _history_db.record_outcome(
            event_id,
            after_metric=after_metric,
            roi=roi,
            performance=performance,
        )


@contextmanager
def log_context(
    *,
    change: str,
    reason: str,
    trigger: str,
    workflow_id: int,
    before_metric: float = 0.0,
    performance: float = 0.0,
    after_metric: float | None = None,
    parent_id: int | None = None,
) -> Generator[Dict[str, float | int], None, None]:
    """Context manager that logs a mutation and records its outcome.

    Example
    -------
    >>> with log_context(change="patch", reason="test", trigger="unit", workflow_id=0) as rec:
    ...     rec["after_metric"] = 1.0
    ...     rec["performance"] = 1.0

    Parameters
    ----------
    change, reason, trigger, workflow_id, before_metric, performance, after_metric, parent_id
        Parameters forwarded to :func:`log_mutation`.

    Yields
    ------
    dict
        Mutable mapping where callers may store ``after_metric``, ``performance`` and
        ``roi`` values which will be persisted on exit.
    """

    event_id = log_mutation(
        change=change,
        reason=reason,
        trigger=trigger,
        performance=performance,
        workflow_id=workflow_id,
        before_metric=before_metric,
        after_metric=after_metric if after_metric is not None else before_metric,
        parent_id=parent_id,
    )
    data: Dict[str, float | int] = {"event_id": event_id, "before_metric": before_metric}
    try:
        yield data
    finally:
        after = float(data.get("after_metric", after_metric if after_metric is not None else before_metric))
        perf = float(data.get("performance", after - before_metric))
        roi = float(data.get("roi", after - before_metric))
        record_mutation_outcome(
            event_id,
            after_metric=after,
            roi=roi,
            performance=perf,
        )


def build_lineage(workflow_id: int) -> list[dict]:
    """Return the lineage tree for *workflow_id*."""
    with _lock:
        return _history_db.lineage_tree(workflow_id)


__all__ = [
    "log_mutation",
    "record_mutation_outcome",
    "build_lineage",
    "set_event_bus",
    "log_context",
    "log_workflow_evolution",
]
