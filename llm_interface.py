"""Lightweight LLM interface definitions.

This module defines simple dataclasses for LLM prompts and results as well as
an ``LLMClient`` protocol for generating responses.  The goal is to provide a
minimal interface that different LLM backends can implement without pulling in
heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass(slots=True)
class Prompt:
    """Input to an LLM generation call."""

    text: str
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)

    # The original codebase often treated prompts as raw strings.  To ease the
    # transition to the structured :class:`Prompt` object, the dataclass mimics
    # ``str`` behaviour for common operations.  This allows existing callers
    # that perform string operations such as ``in`` checks or ``.index`` calls to
    # continue working without modification while still exposing the structured
    # fields.

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.text

    def __eq__(self, other: object) -> bool:  # pragma: no cover - defensive
        if isinstance(other, Prompt):
            return (
                self.text == other.text
                and self.examples == other.examples
                and self.metadata == other.metadata
            )
        if isinstance(other, str):
            return self.text == other
        return False

    def __contains__(self, item: str) -> bool:  # pragma: no cover - delegation
        return item in self.text

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - delegation
        return getattr(self.text, name)

    def __add__(self, other: object):  # pragma: no cover - delegation
        if isinstance(other, Prompt):
            return self.text + other.text
        if isinstance(other, str):
            return self.text + other
        return NotImplemented

    def __radd__(self, other: object):  # pragma: no cover - delegation
        if isinstance(other, str):
            return other + self.text
        if isinstance(other, Prompt):
            return other.text + self.text
        return NotImplemented


@dataclass(slots=True)
class LLMResult:
    """Result returned by an :class:`LLMClient`."""

    raw: Dict[str, object] = field(default_factory=dict)
    text: str = ""


class LLMClient(Protocol):
    """Protocol describing the minimal LLM client interface."""

    def generate(self, prompt: Prompt) -> LLMResult:  # pragma: no cover - interface
        """Generate a response for *prompt*.

        Implementations should return an :class:`LLMResult` with the model's
        response as ``text`` and any raw payload in ``raw``.
        """


__all__ = ["Prompt", "LLMResult", "LLMClient"]
