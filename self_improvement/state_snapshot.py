from __future__ import annotations

"""Deprecated snapshot helpers.

The functionality previously implemented in this module now lives in
:smod:`snapshot_tracker`.  The old entry points are retained as thin wrappers
for backwards compatibility but will be removed in a future release.
"""

from warnings import warn

from .snapshot_tracker import (
    Snapshot,
    SnapshotTracker,
    capture,
    compute_delta,
    save_checkpoint,
    get_best_checkpoint,
    downgrade_counts,
    record_downgrade,
)

warn(
    "state_snapshot is deprecated; use snapshot_tracker instead",
    DeprecationWarning,
    stacklevel=2,
)

# Compatibility aliases -----------------------------------------------------

capture_snapshot = capture

def delta(a: Snapshot, b: Snapshot):
    """Backward compatible alias for :func:`compute_delta`."""
    return compute_delta(a, b)

__all__ = [
    "Snapshot",
    "SnapshotTracker",
    "capture",
    "compute_delta",
    "capture_snapshot",
    "delta",
    "save_checkpoint",
    "get_best_checkpoint",
    "downgrade_counts",
    "record_downgrade",
]
