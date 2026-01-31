"""Generate deterministic patches by delegating to the sandbox generator."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from menace_sandbox.patch_generator import (
    DeleteRegexRule,
    InsertAfterRule,
    ReplaceRule,
    generate_patch as _generate_patch,
)

Rule = ReplaceRule | InsertAfterRule | DeleteRegexRule


def generate_patch(
    source: str,
    error_report: Mapping[str, Any],
    rules: Sequence[Mapping[str, Any]] | Sequence[Rule],
    *,
    validate_syntax: bool | None = None,
) -> dict[str, object]:
    """Generate a deterministic patch payload by applying explicit rules.

    Args:
        source: The original source content to modify.
        error_report: Structured metadata about the error context.
        rules: Patch rules to apply as mappings or rule dataclasses.
        validate_syntax: Explicit override for syntax validation.

    Returns:
        A structured payload containing status, data, errors, and meta fields.
    """
    return _generate_patch(
        source,
        error_report,
        rules,
        validate_syntax=validate_syntax,
    )


__all__ = ["generate_patch"]
