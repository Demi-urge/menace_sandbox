from __future__ import annotations
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional


@dataclass
class ValidationResult:
    """Outcome from :func:`validate_completion`."""

    ok: bool
    text: str = ""
    reason: Optional[str] = None


def validate_completion(output: str) -> ValidationResult:
    """Check ``output`` for basic syntax validity.

    Parameters
    ----------
    output:
        Raw completion string returned by an LLM.
    """

    text = output.strip()
    if not text:
        return ValidationResult(False, reason="empty result")
    try:
        ast.parse(text)
    except Exception as exc:  # pragma: no cover - syntax error branch
        return ValidationResult(False, reason=f"syntax error: {exc}")
    return ValidationResult(True, text=text)


__all__ = ["ValidationResult", "validate_completion"]

