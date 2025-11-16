from __future__ import annotations

"""Orchestration helpers for the self-improvement engine."""

from .orphan_handling import integrate_orphans, post_round_orphan_scan
from .meta_planning import (
    self_improvement_cycle,
    start_self_improvement_cycle,
    stop_self_improvement_cycle,
)

__all__ = [
    "integrate_orphans",
    "post_round_orphan_scan",
    "self_improvement_cycle",
    "start_self_improvement_cycle",
    "stop_self_improvement_cycle",
]
