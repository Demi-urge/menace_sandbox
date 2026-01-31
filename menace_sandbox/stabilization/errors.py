"""Stabilization-specific error types."""

from __future__ import annotations

from menace.errors import PatchRuleError, PatchValidationError


class MenaceValidationError(PatchValidationError):
    """Raised when patch validation fails in the stabilization layer."""


class MenaceRuleSchemaError(PatchRuleError):
    """Raised when a patch rule definition fails schema validation."""


__all__ = [
    "MenaceValidationError",
    "MenaceRuleSchemaError",
]
