"""Exception hierarchy for Menace errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MenaceError(Exception):
    """Base exception for expected Menace failures.

    Attributes:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """

    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)


@dataclass
class ConfigError(MenaceError):
    """Raised when configuration is missing, invalid, or inconsistent."""


@dataclass
class WorkflowValidationError(MenaceError):
    """Raised when a workflow definition fails validation rules."""


@dataclass
class WorkflowExecutionError(MenaceError):
    """Raised when a workflow cannot execute or terminates unexpectedly."""


@dataclass
class EvaluatorError(MenaceError):
    """Raised when evaluation logic cannot compute a deterministic result."""


@dataclass
class OrchestratorError(MenaceError):
    """Raised when orchestration fails due to scheduling or execution errors."""


@dataclass
class LoggingError(MenaceError):
    """Raised when logging or audit capture cannot be completed."""
