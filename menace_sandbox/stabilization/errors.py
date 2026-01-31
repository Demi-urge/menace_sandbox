"""Stabilization-specific error types."""

from __future__ import annotations

from menace.errors import PatchRuleError, PatchValidationError


class MenaceValidationError(PatchValidationError):
    """Raised when patch validation fails in the stabilization layer."""


class MenaceRuleSchemaError(PatchRuleError):
    """Raised when a patch rule definition fails schema validation."""


class RoiDeltaValidationError(ValueError):
    """Raised when ROI delta inputs are invalid."""

    def __init__(self, message: str, details: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_record(self) -> dict[str, object]:
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


__all__ = [
    "MenaceValidationError",
    "MenaceRuleSchemaError",
    "RoiDeltaValidationError",
]
