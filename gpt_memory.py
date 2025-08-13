"""Simple GPT memory interface backed by :class:`MenaceMemoryManager`.

This module provides a thin wrapper that exposes two convenience APIs for
logging model interactions and retrieving context for follow up prompts.

Each logged entry stores the original ``prompt`` and ``response`` along with
placeholder metadata fields that can later be populated with human or
automatic feedback, error fixes, and suggested improvement paths.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

try:  # pragma: no cover - allow use outside package context
    from .menace_memory_manager import MenaceMemoryManager, MemoryEntry
except Exception:  # pragma: no cover - fallback when run as script
    from menace_memory_manager import MenaceMemoryManager, MemoryEntry  # type: ignore


class GPTMemory:
    """Persist and query conversation snippets for GPT style models."""

    def __init__(self, manager: MenaceMemoryManager | None = None) -> None:
        self.manager = manager or MenaceMemoryManager()

    # ------------------------------------------------------------------
    def log_interaction(self, prompt: str, response: str, tags: Iterable[str]) -> None:
        """Store a prompt/response pair with default metadata placeholders.

        Parameters
        ----------
        prompt:
            The user prompt supplied to the model.
        response:
            The model's textual response.
        tags:
            An iterable of tag strings associated with this interaction.
        """

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            # Metadata fields kept empty initially; they can be updated later
            "metadata": {
                "feedback": [],
                "error_fixes": [],
                "improvement_paths": [],
            },
        }
        entry = MemoryEntry(
            key=prompt[:100],  # use a truncated prompt as the key
            data=json.dumps(payload),
            version=1,
            tags=",".join(tags),
        )
        self.manager.log(entry)

    # ------------------------------------------------------------------
    def search_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Return stored interactions matching ``query``.

        The search leverages :class:`MenaceMemoryManager`'s text search which
        falls back to a simple ``LIKE`` lookup when FTS is unavailable.
        Results are returned as deserialised dictionaries containing the
        original prompt, response and metadata.
        """

        results = self.manager.search(query, limit)
        contexts: List[Dict[str, Any]] = []
        for entry in results:
            try:
                contexts.append(json.loads(entry.data))
            except json.JSONDecodeError:
                contexts.append({"prompt": entry.key, "response": entry.data, "metadata": {}})
        return contexts


__all__ = ["GPTMemory"]
