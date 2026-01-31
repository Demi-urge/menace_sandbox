"""Top-level re-exports for deterministic patch validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from menace_sandbox.stabilization.patch_validator import PatchValidationLimits
from menace_sandbox.stabilization.patch_validator import (
    validate_patch as _validate_patch,
)
from menace_sandbox.stabilization.patch_validator import (
    validate_patch_text as _validate_patch_text,
)


def validate_patch(
    original: str,
    patched: str,
    rules: Sequence[Mapping[str, object]] | Mapping[str, object],
    *,
    module_name: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate a patch deterministically using static AST and compile checks.

    This wrapper forwards to the stabilization validator, which parses the
    original and patched sources via ``compile``/``ast.parse`` without executing
    code or importing dynamic modules.

    Returns:
        Dict[str, object]: Structured validation output with ``status``, ``data``,
        ``errors``, and ``meta`` keys.
    """

    return _validate_patch(
        original,
        patched,
        rules,
        module_name=module_name,
        metadata=metadata,
    )


def validate_patch_text(
    patch_text: str,
    *,
    limits: PatchValidationLimits | None = None,
) -> dict[str, object]:
    """Validate a unified diff patch deterministically.

    The underlying validator performs static checks on the patch text and
    returns a structured payload with ``valid``, ``flags``, and ``context``
    describing the result.

    Returns:
        Dict[str, object]: Normalized patch text validation payload.
    """

    return _validate_patch_text(patch_text, limits=limits)


__all__ = ["validate_patch", "validate_patch_text", "PatchValidationLimits"]
