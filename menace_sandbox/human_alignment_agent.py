from __future__ import annotations

"""Compatibility shim for :mod:`menace.human_alignment_agent`."""

from menace.human_alignment_agent import (
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
