from __future__ import annotations

"""Thread-safe mutation logging helpers."""

import logging
from threading import Lock
from typing import Optional
import sqlite3

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


def log_mutation(
    change: str,
    reason: str,
    trigger: str,
    performance: float,
    workflow_id: int,
    parent_id: int | None = None,
) -> int:
    """Record a mutation event and return its row id."""
    event = EvolutionEvent(
        action=change,
        before_metric=0.0,
        after_metric=0.0,
        roi=0.0,
        reason=reason,
        trigger=trigger,
        performance=performance,
        workflow_id=workflow_id,
        parent_event_id=parent_id,
    )
    with _lock:
        event_id = _history_db.add(event)
    if _event_bus is not None:
        try:
            payload = {"event_id": event_id, **event.__dict__}
            _event_bus.publish("mutation_recorded", payload)
        except Exception as exc:  # pragma: no cover - best effort
            _logger.error("failed publishing mutation_recorded: %s", exc)
    return event_id


def build_lineage(workflow_id: int) -> list[dict]:
    """Return the lineage tree for *workflow_id*."""
    with _lock:
        return _history_db.lineage_tree(workflow_id)


__all__ = ["log_mutation", "build_lineage"]
