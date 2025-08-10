from __future__ import annotations

"""Thread-safe mutation logging helpers."""

import logging
from threading import Lock, Thread
from typing import Optional
import sqlite3

from .retry_utils import publish_with_retry

from .evolution_history_db import EvolutionHistoryDB, EvolutionEvent

try:  # optional dependency
    from .unified_event_bus import UnifiedEventBus  # type: ignore
except Exception:  # pragma: no cover - bus optional
    UnifiedEventBus = None  # type: ignore

_logger = logging.getLogger(__name__)

# Path for the evolution history database
_DB_PATH = "evolution_history.db"

# Create a single shared database instance and reopen the connection with
# ``check_same_thread=False`` to allow access across threads.
_history_db = EvolutionHistoryDB(_DB_PATH)
try:  # ensure the connection is usable from multiple threads
    _history_db.conn.close()
    _history_db.conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
except Exception:  # pragma: no cover - fallback for unusual environments
    pass

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


def build_lineage(workflow_id: int) -> list[dict]:
    """Return the lineage tree for *workflow_id*."""
    with _lock:
        return _history_db.lineage_tree(workflow_id)


__all__ = [
    "log_mutation",
    "record_mutation_outcome",
    "build_lineage",
    "set_event_bus",
]
