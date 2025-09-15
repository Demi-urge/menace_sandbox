from __future__ import annotations

"""Common prompt related data structures."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    origin: str = ""

    def __init__(
        self,
        user: str = "",
        *,
        system: str = "",
        examples: List[str] | None = None,
        vector_confidence: float | None = None,
        tags: List[str] | None = None,
        outcome_tags: List[str] | None = None,
        vector_confidences: List[float] | None = None,
        text: str | None = None,
        metadata: Dict[str, Any] | None = None,
        origin: str | None = None,
    ) -> None:
        if text is not None and not user:
            user = text
        meta = dict(metadata or {})
        self.system = system
        self.user = user
        self.examples = list(examples) if examples else []
        if vector_confidence is not None:
            self.vector_confidence = vector_confidence
        elif vector_confidences:
            self.vector_confidence = vector_confidences[0]
        elif "vector_confidence" in meta:
            self.vector_confidence = meta.get("vector_confidence")
        elif meta.get("vector_confidences"):
            self.vector_confidence = meta.get("vector_confidences")[0]
        else:
            self.vector_confidence = None
        if tags is not None:
            self.tags = list(tags)
        elif outcome_tags is not None:
            self.tags = list(outcome_tags)
        elif meta.get("tags") is not None:
            self.tags = list(meta.get("tags"))
        elif meta.get("outcome_tags") is not None:
            self.tags = list(meta.get("outcome_tags"))
        else:
            self.tags = []
        self.metadata = meta
        self.origin = origin or meta.get("origin", "")

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
        if "vector_confidences" in self.metadata:
            return list(self.metadata["vector_confidences"])
        return [self.vector_confidence] if self.vector_confidence is not None else []

    @property
    def outcome_tags(self) -> List[str]:
        """Legacy accessor mapping to :attr:`tags`."""
        return self.tags


__all__ = ["Prompt"]
