"""Typed exceptions for Menace workflows."""

from __future__ import annotations

from typing import Any, Dict, Optional


class MenaceError(Exception):
    """Base exception for expected Menace failures."""

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


class InputValidationError(MenaceError):
    """Raised when inputs fail validation checks."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="input_validation_error", details=details)


class MissingFieldError(MenaceError):
    """Raised when a required field is missing or None."""

    def __init__(self, field_name: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        message = f"Missing required field: {field_name}"
        merged_details = {"field": field_name}
        if details:
            merged_details.update(details)
        super().__init__(message, code="missing_field", details=merged_details)


class MalformedInputError(MenaceError):
    """Raised when input types or structures are malformed."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="malformed_input", details=details)


class CalculationError(MenaceError):
    """Raised when deterministic calculations cannot be completed."""

    def __init__(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message, code="calculation_error", details=details)
