"""Stabilization utilities for normalizing LLM outputs."""

from .pipeline import stabilize_completion
from .response_schemas import (
    ValidationError,
    normalize_error_response,
    normalize_mvp_response,
    normalize_patch_apply,
    normalize_patch_code_validation,
    normalize_patch_validation,
    normalize_preset_batch,
)
from .roi import compute_roi_delta
from .errors import MenaceRuleSchemaError, MenaceValidationError
from .patch_validator import (
    PatchValidationLimits,
    validate_patch,
    validate_patch_text,
)

__all__ = [
    "ValidationError",
    "normalize_error_response",
    "normalize_mvp_response",
    "normalize_patch_apply",
    "normalize_patch_code_validation",
    "normalize_patch_validation",
    "normalize_preset_batch",
    "stabilize_completion",
    "compute_roi_delta",
    "MenaceRuleSchemaError",
    "MenaceValidationError",
    "PatchValidationLimits",
    "validate_patch",
    "validate_patch_text",
]
