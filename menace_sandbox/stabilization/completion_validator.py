"""Offline-friendly completion validation utilities.

The original Menace deployment performs extensive linting using internal
services. When running the sandbox locally we still want confidence that a
language-model completion contains syntactically valid Python while remaining
lightweight and dependency free.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["ValidationResult", "ValidationError", "validate_completion"]

_MAX_COMPLETION_LENGTH = 50_000
_CODE_FENCE_PATTERN = re.compile(r"^```(?:python)?\n(?P<body>.*)```$", re.DOTALL)


class ValidationError(ValueError):
    """Raised when a completion fails validation."""


@dataclass(slots=True)
class ValidationResult:
    """Result object describing the outcome of :func:`validate_completion`."""

    ok: bool
    text: str = ""
    reason: Optional[str] = None
    diagnostics: List[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        """Raise :class:`ValidationError` when ``ok`` is ``False``."""

        if not self.ok:
            raise ValidationError(self.reason or "completion did not pass validation")


def _normalise_payload(payload: str) -> str:
    """Strip Markdown fences and surrounding whitespace from *payload*."""

    text = payload.strip()
    match = _CODE_FENCE_PATTERN.match(text)
    if match:
        text = match.group("body").strip()
    return text


def _lint_python(text: str) -> List[str]:
    """Return lightweight diagnostics for *text* without failing fast."""

    diagnostics: List[str] = []
    try:
        ast.parse(text)
    except SyntaxError as exc:  # pragma: no cover - exercised in integration
        diagnostics.append(f"syntax error at line {exc.lineno}: {exc.msg}")
    except RecursionError as exc:  # pragma: no cover - defensive
        diagnostics.append(f"recursion error: {exc}")
    except Exception as exc:  # pragma: no cover - defensive catch-all
        diagnostics.append(f"unexpected parse failure: {exc}")
    return diagnostics


def validate_completion(output: str) -> ValidationResult:
    """Validate ``output`` and return a :class:`ValidationResult` instance."""

    if not isinstance(output, str):
        return ValidationResult(False, reason="completion must be a string")

    text = _normalise_payload(output)
    if not text:
        return ValidationResult(False, reason="empty completion")

    if len(text) > _MAX_COMPLETION_LENGTH:
        return ValidationResult(False, reason="completion exceeds maximum length")

    diagnostics = _lint_python(text)
    if diagnostics:
        return ValidationResult(
            False,
            text=text,
            reason=diagnostics[0],
            diagnostics=diagnostics,
        )

    return ValidationResult(True, text=text)
