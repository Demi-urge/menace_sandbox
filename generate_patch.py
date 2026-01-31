"""Generate deterministic patches by delegating to the sandbox generator."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from menace.errors import PatchAnchorError, PatchRuleError
from menace_sandbox.patch_generator import generate_patch as _generate_patch


def _validate_inputs(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
    validate_syntax: bool | None,
) -> None:
    if not isinstance(source, str):
        raise PatchRuleError(
            "source must be a string",
            details={"field": "source", "expected": "str", "actual_type": type(source).__name__},
        )
    if not source:
        raise PatchRuleError(
            "source must not be empty",
            details={"field": "source", "expected": "non-empty string"},
        )
    if not isinstance(error_report, Mapping):
        raise PatchRuleError(
            "error_report must be a mapping",
            details={
                "field": "error_report",
                "expected": "mapping",
                "actual_type": type(error_report).__name__,
            },
        )
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)):
        raise PatchRuleError(
            "rules must be a sequence",
            details={"field": "rules", "expected": "sequence", "actual_type": type(rules).__name__},
        )
    if validate_syntax is not None and not isinstance(validate_syntax, bool):
        raise PatchRuleError(
            "validate_syntax must be a boolean when provided",
            details={
                "field": "validate_syntax",
                "expected": "bool",
                "actual_type": type(validate_syntax).__name__,
            },
        )
    for index, rule in enumerate(rules):
        if not isinstance(rule, Mapping):
            raise PatchRuleError(
                "rule must be a mapping",
                details={"rule_index": index, "rule": rule, "expected": "mapping"},
            )


def _raise_for_anchor_error(result: Mapping[str, Any]) -> None:
    errors = result.get("errors")
    if not isinstance(errors, Sequence):
        return
    for error in errors:
        if not isinstance(error, Mapping):
            continue
        if error.get("type") == "PatchAnchorError":
            raise PatchAnchorError(
                error.get("message", "missing patch anchor"),
                details=dict(error.get("details") or {}),
            )


def generate_patch(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]],
    *,
    validate_syntax: bool | None = None,
) -> dict[str, object]:
    """Generate a deterministic patch payload by applying explicit rules.

    Args:
        source: The original source content to modify.
        error_report: Structured metadata about the error context.
        rules: Patch rules to apply as mappings.
        validate_syntax: Explicit override for syntax validation.

    Returns:
        A structured payload containing status, data, errors, and meta fields.
    """
    _validate_inputs(source, error_report, rules, validate_syntax)
    result = _generate_patch(
        source,
        error_report,
        rules,
        validate_syntax=validate_syntax,
    )
    _raise_for_anchor_error(result)
    return result


__all__ = ["generate_patch", "PatchAnchorError", "PatchRuleError"]
