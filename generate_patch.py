"""Deterministic patch generation wrapper."""

from __future__ import annotations

from menace.errors import PatchAnchorError, PatchRuleError
from menace_sandbox import patch_generator


_ALLOWED_RULE_TYPES = {"replace", "insert_after", "delete_regex"}


def generate_patch(
    source: str,
    error_report: dict[str, object],
    rules: list[dict[str, object]],
    *,
    validate_syntax: bool | None = None,
) -> dict[str, object]:
    """Generate a deterministic patch payload using explicit rules.

    This wrapper enforces rule schema constraints and delegates to the
    deterministic rule application logic in menace_sandbox.patch_generator.
    """
    _validate_inputs(source, error_report, rules)
    return patch_generator.generate_patch(
        source,
        error_report,
        rules,
        validate_syntax=validate_syntax,
    )


def _validate_inputs(
    source: str,
    error_report: dict[str, object],
    rules: list[dict[str, object]],
) -> None:
    """Validate top-level inputs and rule schemas."""
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
    if not isinstance(error_report, dict):
        raise PatchRuleError(
            "error_report must be a dict",
            details={
                "field": "error_report",
                "expected": "dict",
                "actual_type": type(error_report).__name__,
            },
        )
    if not isinstance(rules, list):
        raise PatchRuleError(
            "rules must be a list",
            details={"field": "rules", "expected": "list", "actual_type": type(rules).__name__},
        )
    if not rules:
        raise PatchRuleError("rules must not be empty", details={"field": "rules"})
    for index, rule in enumerate(rules):
        _validate_rule(rule, index)


def _validate_rule(rule: dict[str, object], index: int) -> None:
    """Ensure each rule has a supported schema."""
    if not isinstance(rule, dict):
        raise PatchRuleError(
            "rule must be a dict",
            details={"rule_index": index, "expected": "dict", "actual_type": type(rule).__name__},
        )
    rule_type = rule.get("type")
    if not isinstance(rule_type, str) or not rule_type:
        raise PatchRuleError(
            "rule type is required",
            details={"rule_index": index, "field": "type"},
        )
    if rule_type not in _ALLOWED_RULE_TYPES:
        raise PatchRuleError(
            "Unknown rule type",
            details={"rule_index": index, "field": "type", "value": rule_type},
        )
    _require_non_empty_str(rule, "id", index)
    _require_non_empty_str(rule, "description", index)
    _require_meta(rule, index)
    if rule_type == "replace":
        _require_non_empty_str(rule, "anchor", index)
        _require_non_empty_str(rule, "replacement", index)
        _require_anchor_kind(rule, index)
    elif rule_type == "insert_after":
        _require_non_empty_str(rule, "anchor", index)
        _require_non_empty_str(rule, "content", index)
        _require_anchor_kind(rule, index)
    elif rule_type == "delete_regex":
        _require_non_empty_str(rule, "pattern", index)
        _require_flags(rule, index)


def _require_meta(rule: dict[str, object], index: int) -> None:
    meta = rule.get("meta")
    if meta is None:
        raise PatchRuleError(
            "meta is required",
            details={"rule_index": index, "field": "meta"},
        )
    if not isinstance(meta, dict):
        raise PatchRuleError(
            "meta must be a dict",
            details={
                "rule_index": index,
                "field": "meta",
                "actual_type": type(meta).__name__,
            },
        )


def _require_anchor_kind(rule: dict[str, object], index: int) -> None:
    anchor_kind = rule.get("anchor_kind")
    if not isinstance(anchor_kind, str) or not anchor_kind:
        raise PatchRuleError(
            "anchor_kind is required",
            details={"rule_index": index, "field": "anchor_kind"},
        )


def _require_flags(rule: dict[str, object], index: int) -> None:
    flags = rule.get("flags")
    if flags is None:
        raise PatchRuleError(
            "flags are required",
            details={"rule_index": index, "field": "flags"},
        )
    if isinstance(flags, str):
        return
    if isinstance(flags, list) and all(isinstance(flag, str) for flag in flags):
        return
    raise PatchRuleError(
        "flags must be a string or list of strings",
        details={"rule_index": index, "field": "flags", "actual_type": type(flags).__name__},
    )


def _require_non_empty_str(rule: dict[str, object], field: str, index: int) -> str:
    value = rule.get(field)
    if not isinstance(value, str) or not value.strip():
        raise PatchRuleError(
            f"{field} is required",
            details={"rule_index": index, "field": field},
        )
    return value


__all__ = ["generate_patch", "PatchAnchorError", "PatchRuleError"]
