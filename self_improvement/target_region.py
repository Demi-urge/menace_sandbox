from __future__ import annotations

"""Lightweight representation of a source code region.

This module defines :class:`TargetRegion`, a minimal data container used to
identify the portion of a file implicated in a failure.  The dataclass is kept
intentionally small so it can be serialised and passed through various layers
without incurring heavy dependencies.
"""

from dataclasses import dataclass


@dataclass
class TargetRegion:
    """Contiguous region in a source file."""

    file: str
    start_line: int
    end_line: int
    function: str


__all__ = ["TargetRegion"]
