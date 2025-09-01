from __future__ import annotations

"""Common prompt related data structures."""

from dataclasses import dataclass, field
from typing import List


@dataclass(init=False)
class Prompt:
    """Container describing an LLM prompt.

    Parameters
    ----------
    system:
        Optional system message describing high level behaviour.
    user:
        The main user request shown to the model.
    examples:
        Optional few shot examples or snippets used to prime the model.
    vector_confidence:
        Average confidence score returned by vector retrieval, if any.
    tags:
        Arbitrary tags associated with the prompt such as ROI categories.
    """

    system: str = ""
    user: str = ""
    examples: List[str] = field(default_factory=list)
    vector_confidence: float | None = None
    tags: List[str] = field(default_factory=list)

    def __init__(
        self,
        user: str = "",
        *,
        system: str = "",
        examples: List[str] | None = None,
        vector_confidence: float | None = None,
        tags: List[str] | None = None,
        text: str | None = None,
    ) -> None:
        if text is not None and not user:
            user = text
        self.system = system
        self.user = user
        self.examples = list(examples) if examples else []
        self.vector_confidence = vector_confidence
        self.tags = list(tags) if tags else []

    # ------------------------------------------------------------------
    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.user

    def __contains__(self, item: object) -> bool:  # pragma: no cover - delegation
        return str(item) in self.user

    def __getattr__(self, name: str):  # pragma: no cover - delegation
        return getattr(self.user, name)

    # Backwards compatibility properties ---------------------------------
    @property
    def text(self) -> str:
        """Alias returning the user portion of the prompt."""
        return self.user

    @property
    def vector_confidences(self) -> List[float]:
        """Legacy list-style confidence accessor."""
        return [self.vector_confidence] if self.vector_confidence is not None else []

    @property
    def outcome_tags(self) -> List[str]:
        """Legacy accessor mapping to :attr:`tags`."""
        return self.tags


__all__ = ["Prompt"]
