"""Helper functions for retrieving knowledge from GPT memory.

This module wraps :class:`gpt_memory.GPTMemoryManager` and
:class:`gpt_knowledge_service.GPTKnowledgeService` to provide simple
functions for bots to fetch feedback, error fixes and other insights.
"""

from __future__ import annotations

from gpt_memory import GPTMemoryManager
try:  # pragma: no cover - allow flat imports
    from .gpt_knowledge_service import GPTKnowledgeService
except Exception:  # pragma: no cover - fallback for flat layout
    from gpt_knowledge_service import GPTKnowledgeService  # type: ignore

try:  # pragma: no cover - allow flat imports
    from .log_tags import FEEDBACK, ERROR_FIX, IMPROVEMENT_PATH
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, ERROR_FIX, IMPROVEMENT_PATH  # type: ignore


# ---------------------------------------------------------------------------
# Raw interaction retrieval helpers

def get_feedback(manager: GPTMemoryManager, key: str, *, limit: int = 5):
    """Return past feedback entries related to ``key``.

    This searches the memory for interactions tagged with ``FEEDBACK``.
    """

    return manager.search_context(
        key, tags=[FEEDBACK], limit=limit, use_embeddings=False
    )


def get_error_fixes(manager: GPTMemoryManager, key: str, *, limit: int = 5):
    """Return past error fix suggestions related to ``key``."""

    return manager.search_context(
        key, tags=[ERROR_FIX], limit=limit, use_embeddings=False
    )


def get_improvement_paths(manager: GPTMemoryManager, key: str, *, limit: int = 5):
    """Return previous improvement path suggestions for ``key``."""

    return manager.search_context(
        key, tags=[IMPROVEMENT_PATH], limit=limit, use_embeddings=False
    )


# ---------------------------------------------------------------------------
# Insight retrieval helpers

def recent_feedback(service: GPTKnowledgeService) -> str:
    """Return the latest summarised feedback insight."""

    return service.get_recent_insights(FEEDBACK)


def recent_error_fix(service: GPTKnowledgeService) -> str:
    """Return the latest summarised error fix insight."""

    return service.get_recent_insights(ERROR_FIX)


def recent_improvement_path(service: GPTKnowledgeService) -> str:
    """Return the latest summarised improvement path insight."""

    return service.get_recent_insights(IMPROVEMENT_PATH)
