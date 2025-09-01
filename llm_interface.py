"""Lightweight LLM interface definitions.

The project only needs a very small slice of functionality from whatever
language model backend is in use.  To keep the dependency surface minimal this
module defines tiny dataclasses for exchanging prompts and results together with
a base :class:`LLMClient` that backends can inherit from.  The class exposes a
single :py:meth:`generate` method returning an :class:`LLMResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, List


@dataclass(slots=True)
class Prompt:
    """Input to an LLM generation call.

    ``text`` contains the assembled prompt shown to the model while ``examples``
    stores any illustrative snippets that were used to build the prompt.
    Additional information such as ROI metrics or tone preferences can be
    attached via the ``metadata`` mapping.
    """

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
    """Result returned by an :class:`LLMClient`.

    ``text`` holds the raw string produced by the model.  ``parsed`` optionally
    stores a structured representation of that string (for example JSON
    decoded from ``text``).  ``raw`` can be used by clients to stash any
    transport specific payload such as HTTP responses.
    """

    text: str = ""
    parsed: Any | None = None
    raw: Dict[str, object] = field(default_factory=dict)


class LLMClient(ABC):
    """Base class describing the minimal LLM client interface."""

    @abstractmethod
    def generate(self, prompt: Prompt) -> LLMResult:  # pragma: no cover - interface
        """Generate a response for *prompt*.

        Implementations should return an :class:`LLMResult` with the model's
        response in ``text`` and may optionally populate ``parsed`` and
        ``raw`` with backend-specific data.
        """


__all__ = ["Prompt", "LLMResult", "LLMClient"]
