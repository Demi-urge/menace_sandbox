"""Exception hierarchy for Menace errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MenaceError(Exception):
    """Base exception for expected Menace failures.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """

    message: str
    details: dict[str, Any] | None = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, serializable view of the error."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": dict(self.details or {}),
        }


@dataclass
class ConfigError(MenaceError):
    """Raised when configuration is missing, invalid, or inconsistent."""


@dataclass
class WorkflowError(MenaceError):
    """Raised when a workflow cannot start or complete as expected."""


@dataclass
class ValidationError(MenaceError):
    """Raised when input, payload, or schema validation fails."""


@dataclass
class PatchRuleError(ValidationError):
    """Raised when a patch rule definition is malformed or unsupported."""


@dataclass
class PatchAnchorError(ValidationError):
    """Raised when a patch anchor is missing or resolves ambiguously."""


@dataclass
class PatchConflictError(ValidationError):
    """Raised when patch edits contradict or overlap each other."""


@dataclass
class EvaluationError(MenaceError):
    """Raised when evaluation logic cannot compute a deterministic result."""


@dataclass
class OrchestratorError(MenaceError):
    """Raised when orchestration fails due to scheduling or execution errors."""


@dataclass
class WorkflowValidationError(ValidationError):
    """Raised when a workflow definition fails validation rules."""


@dataclass
class WorkflowExecutionError(WorkflowError):
    """Raised when a workflow cannot execute or terminates unexpectedly."""


@dataclass
class EvaluatorError(EvaluationError):
    """Raised when evaluation logic cannot compute a deterministic result."""


@dataclass
class LoggingError(MenaceError):
    """Raised when logging or audit capture cannot be completed."""
