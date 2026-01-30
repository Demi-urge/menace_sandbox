"""Typed exceptions and serialization helpers for Menace."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from menace.infra.logging import log_event


@dataclass
class MenaceError(Exception):
    """Base exception for expected Menace failures.

    Raised when a domain-specific failure occurs that should be handled
    gracefully rather than propagated as an untyped exception.

    Attributes:
        code: Stable error code for programmatic handling.
        message: Human-readable error message.
        details: Optional structured metadata describing the failure.
    """

    message: str
    code: str = "menace_error"
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_error_dict(self) -> Dict[str, Any]:
        """Return a structured error payload."""
        return {
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass
class ConfigError(MenaceError):
    """Raised when configuration is missing, invalid, or inconsistent.

    This is used when a configuration file, environment variable, or
    runtime setting cannot be parsed or validated.
    """

    code: str = "config_error"


@dataclass
class WorkflowError(MenaceError):
    """Raised when a workflow definition is malformed or incomplete.

    This is used when a workflow cannot be parsed, validated, or
    connected to required dependencies.
    """

    code: str = "workflow_error"


@dataclass
class EvaluationError(MenaceError):
    """Raised when evaluation logic cannot compute a deterministic result.

    This is used when an evaluation step fails due to missing inputs,
    invalid metrics, or non-deterministic behavior.
    """

    code: str = "evaluation_error"


@dataclass
class OrchestratorError(MenaceError):
    """Raised when the orchestrator cannot execute a workflow.

    This is used when orchestration fails due to scheduling errors,
    execution failures, or unsupported workflow states.
    """

    code: str = "orchestrator_error"


def serialize_exception(
    error: Exception,
    *,
    status: str = "error",
    data: Optional[Mapping[str, Any]] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Serialize an exception into the standard error schema.

    Args:
        error: The exception to serialize.
        status: Status string for the response payload.
        data: Optional response data to include.
        meta: Optional metadata to include.

    Returns:
        Serialized error payload in the ``{status, data, errors, meta}`` schema.
    """

    if isinstance(error, MenaceError):
        error_payload = error.to_error_dict()
    else:
        error_payload = {
            "code": "unhandled_exception",
            "message": str(error),
            "details": {},
        }

    payload = {
        "status": status,
        "data": dict(data) if data is not None else None,
        "errors": [error_payload],
        "meta": dict(meta) if meta is not None else {},
    }

    log_event(
        level="error",
        message="Serialized exception payload",
        metadata={
            "error_code": error_payload.get("code"),
            "status": status,
        },
    )

    return payload
