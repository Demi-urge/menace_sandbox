"""Typed exceptions for Menace workflows."""

from __future__ import annotations

from typing import Any, Dict, Optional


class MenaceError(Exception):
    """Base exception for expected Menace failures.

    Context payload:
        details: Optional mapping with structured context for error handling.
    """

    def __init__(self, message: str, *, code: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}

    def to_error_dict(self) -> Dict[str, Any]:
        """Return a structured error payload."""
        return {
            "message": str(self),
            "code": self.code,
            "details": dict(self.details),
        }


class ValidationError(MenaceError):
    """Raised when inputs fail validation checks.

    Context payload:
        details: Includes field names, expected types, and received values.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "validation_error",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ConfigurationError(MenaceError):
    """Raised when configuration is missing or invalid.

    Context payload:
        details: Includes config keys, invalid values, or schema hints.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="configuration_error", details=details)


class WorkflowDefinitionError(MenaceError):
    """Raised when a workflow definition is malformed or incomplete.

    Context payload:
        details: Includes workflow_id, missing fields, and schema expectations.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="workflow_definition_error", details=details)


class EvaluationError(MenaceError):
    """Raised when evaluation logic cannot compute a deterministic result.

    Context payload:
        details: Includes evaluation inputs and any computed intermediates.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="evaluation_error", details=details)


class OrchestratorExecutionError(MenaceError):
    """Raised when orchestrator execution fails or receives invalid workflows.

    Context payload:
        details: Includes workflow index, workflow_id, or execution metadata.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="orchestrator_execution_error", details=details)


class InputValidationError(ValidationError):
    """Raised when inputs fail validation checks.

    Context payload:
        details: Includes field names, expected types, and received values.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="input_validation_error", details=details)


class MissingFieldError(ValidationError):
    """Raised when a required field is missing or None.

    Context payload:
        details: Includes "field" for the missing field name.
    """

    def __init__(self, field_name: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        message = f"Missing required field: {field_name}"
        merged_details = {"field": field_name}
        if details:
            merged_details.update(details)
        super().__init__(message, code="missing_field", details=merged_details)


class MalformedInputError(ValidationError):
    """Raised when input types or structures are malformed.

    Context payload:
        details: Includes expected schema and received_type/value_type.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="malformed_input", details=details)


class CalculationError(EvaluationError):
    """Raised when deterministic calculations cannot be completed.

    Context payload:
        details: Includes input values that prevented the calculation.
    """

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="calculation_error", details=details)
