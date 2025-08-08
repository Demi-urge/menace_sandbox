from __future__ import annotations

"""Centralised error taxonomy for Menace."""

from enum import Enum


class ErrorType(str, Enum):
    """High level categories for error classification."""

    UNKNOWN = "unknown"
    SEMANTIC_BUG = "semantic_bug"
    RUNTIME_FAULT = "runtime_fault"
    DEPENDENCY_MISMATCH = "dependency_mismatch"
    LOGIC_MISFIRE = "logic_misfire"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


__all__ = ["ErrorType"]
