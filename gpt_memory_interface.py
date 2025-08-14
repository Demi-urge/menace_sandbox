"""Protocol defining a minimal GPT memory interface."""

from __future__ import annotations

from typing import Any, List, Protocol, Sequence


class GPTMemoryInterface(Protocol):
    """Common operations supported by GPT memory backends."""

    def store(self, key: str, data: str, tags: Sequence[str] | None = None) -> int | None:
        """Persist ``data`` under ``key`` with optional ``tags``."""
        ...

    def retrieve(
        self, query: str, limit: int = 5, tags: Sequence[str] | None = None
    ) -> List[Any]:
        """Return stored records matching ``query``."""
        ...

    def log_interaction(
        self, prompt: str, response: str, tags: Sequence[str] | None = None
    ) -> int | None:
        """Record a prompt/response pair."""
        ...

    def search_context(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> List[Any]:
        """Search stored interactions related to ``query``."""
        ...
