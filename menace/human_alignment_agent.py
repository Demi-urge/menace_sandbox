from __future__ import annotations

"""Compatibility shim for :mod:`human_alignment_agent`."""

from human_alignment_agent import (
    HumanAlignmentAgent,
    SandboxSettings,
    flag_improvement,
    log_violation,
)

__all__ = [
    "HumanAlignmentAgent",
    "SandboxSettings",
    "flag_improvement",
    "log_violation",
]
