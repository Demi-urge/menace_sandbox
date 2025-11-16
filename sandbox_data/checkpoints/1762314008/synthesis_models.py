"""Shared dataclasses for synthesis-related tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataRequest:
    """Request for additional data from Stage 2 bots."""

    table: str
    field: str
    reason: str
    priority: int = 1


@dataclass
class SynthesisTask:
    """Actionable task for Stage 4 planning."""

    description: str
    urgency: int
    complexity: int
    category: str


__all__ = ["DataRequest", "SynthesisTask"]
