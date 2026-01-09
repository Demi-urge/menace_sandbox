from __future__ import annotations

"""Orchestration helpers for the self-improvement engine."""

from .orphan_handling import (
    integrate_orphans,
    integrate_orphans_sync,
    post_round_orphan_scan,
    post_round_orphan_scan_sync,
)
from .meta_planning import (
    self_improvement_cycle,
    start_self_improvement_cycle,
    stop_self_improvement_cycle,
)

__all__ = [
    "integrate_orphans",
    "integrate_orphans_sync",
    "post_round_orphan_scan",
    "post_round_orphan_scan_sync",
    "self_improvement_cycle",
    "start_self_improvement_cycle",
    "stop_self_improvement_cycle",
]
