"""Lightweight LLM interface definitions.

This module defines simple dataclasses for LLM prompts and results as well as
an ``LLMClient`` protocol for generating responses.  The goal is to provide a
minimal interface that different LLM backends can implement without pulling in
heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol


@dataclass(slots=True)
class Prompt:
    """Input to an LLM generation call."""

    text: str
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)


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
