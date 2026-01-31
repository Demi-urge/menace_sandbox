"""Exception hierarchy for Menace errors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class MenaceError(Exception):
    """Base exception for expected Menace failures.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """

    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, serializable view of the error."""
        details = dict(self.details or {})
        payload: dict[str, Any] = {
            "type": self.__class__.__name__,
            "error_type": self.__class__.__name__,
            "message": self.message,
            "rule_id": details.get("rule_id"),
            "rule_index": details.get("rule_index"),
            "details": details,
        }
        location = _extract_location(details)
        if location:
            payload["location"] = location
        return payload


def _extract_location(details: Mapping[str, Any]) -> dict[str, Any]:
    """Extract optional location context from error details."""
    location: dict[str, Any] = {}
    line_offsets = details.get("line_offsets")
    if isinstance(line_offsets, Mapping):
        location.update(line_offsets)
    for key in ("line", "lineno"):
        if key in details:
            location["line"] = details[key]
            break
    for key in ("column", "col_offset", "col"):
        if key in details:
            location["column"] = details[key]
            break
    if "column" not in location and "offset" in details:
        location["column"] = details["offset"]
    if "span" in details:
        location["span"] = details["span"]
    return location


@dataclass
class ConfigError(MenaceError):
    """Raised when configuration is missing, invalid, or inconsistent.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class WorkflowError(MenaceError):
    """Raised when a workflow cannot start or complete as expected.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class ValidationError(MenaceError):
    """Raised when input, payload, or schema validation fails.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class PatchRuleError(ValidationError):
    """Raised when a patch rule definition is malformed or unsupported.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class PatchAnchorError(ValidationError):
    """Raised when a patch anchor is missing or resolves ambiguously.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class PatchConflictError(ValidationError):
    """Raised when patch edits contradict or overlap each other.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class EvaluationError(MenaceError):
    """Raised when evaluation logic cannot compute a deterministic result.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class PatchSyntaxError(ValidationError):
    """Raised when a generated patch introduces syntax errors.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class PatchValidationError(ValidationError):
    """Raised when patch validation fails unexpectedly.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class PatchParseError(EvaluationError):
    """Raised when a generated patch cannot be parsed or validated.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class OrchestratorError(MenaceError):
    """Raised when orchestration fails due to scheduling or execution errors.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class WorkflowValidationError(ValidationError):
    """Raised when a workflow definition fails validation rules.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class WorkflowExecutionError(WorkflowError):
    """Raised when a workflow cannot execute or terminates unexpectedly.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class EvaluatorError(EvaluationError):
    """Raised when evaluation logic cannot compute a deterministic result.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """


@dataclass
class LoggingError(MenaceError):
    """Raised when logging or audit capture cannot be completed.

    Args:
        message: Human-readable error message describing the failure.
        details: Optional structured context describing the failure.
    """
