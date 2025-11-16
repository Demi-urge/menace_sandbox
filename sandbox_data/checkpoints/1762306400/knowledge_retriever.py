"""Helper functions for retrieving knowledge from GPT memory.

This module wraps :class:`gpt_memory.GPTMemoryManager` and
:class:`gpt_knowledge_service.GPTKnowledgeService` to provide simple
functions for bots to fetch feedback, error fixes and other insights.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import TYPE_CHECKING
from governed_retrieval import govern_retrieval
from secret_redactor import redact_secrets, redact_secrets_dict
try:  # pragma: no cover - allow flat imports
    from .gpt_knowledge_service import GPTKnowledgeService
except Exception:  # pragma: no cover - fallback for flat layout
    from gpt_knowledge_service import GPTKnowledgeService  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - used for static analysis only
    from menace_sandbox.gpt_memory import GPTMemoryManager

try:  # pragma: no cover - allow flat imports
    from .log_tags import FEEDBACK, ERROR_FIX, IMPROVEMENT_PATH
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import FEEDBACK, ERROR_FIX, IMPROVEMENT_PATH  # type: ignore


__all__ = [
    "get_feedback",
    "get_error_fixes",
    "get_improvement_paths",
    "recent_feedback",
    "recent_error_fix",
    "recent_improvement_path",
]

# ---------------------------------------------------------------------------
# Raw interaction retrieval helpers


def _govern(entries):
    governed = []
    for e in entries:
        data = redact_secrets_dict(asdict(e))
        text = f"{data.get('prompt', '')} {data.get('response', '')}".strip()
        meta_in = {"tags": list(data.get("tags", []))}
        governed_meta = govern_retrieval(text, meta_in)
        if governed_meta is None:
            continue
        meta, _ = governed_meta
        data["tags"] = meta.pop("tags", data.get("tags", []))
        data["metadata"] = meta
        governed.append(replace(e, **data))
    return governed

def get_feedback(
    manager: "GPTMemoryManager", key: str, *, limit: int = 5, use_embeddings: bool = True
):
    """Return past feedback entries related to ``key``.

    This searches the memory for interactions tagged with ``FEEDBACK``.  When
    ``use_embeddings`` is ``True`` semantic search is employed.
    """

    res = manager.search_context(
        key, tags=[FEEDBACK], limit=limit, use_embeddings=use_embeddings
    )
    return _govern(res)


def get_error_fixes(
    manager: "GPTMemoryManager", key: str, *, limit: int = 5, use_embeddings: bool = True
):
    """Return past error fix suggestions related to ``key``."""

    res = manager.search_context(
        key, tags=[ERROR_FIX], limit=limit, use_embeddings=use_embeddings
    )
    return _govern(res)


def get_improvement_paths(
    manager: "GPTMemoryManager", key: str, *, limit: int = 5, use_embeddings: bool = True
):
    """Return previous improvement path suggestions for ``key``."""

    res = manager.search_context(
        key, tags=[IMPROVEMENT_PATH], limit=limit, use_embeddings=use_embeddings
    )
    return _govern(res)


# ---------------------------------------------------------------------------
# Insight retrieval helpers

def recent_feedback(service: GPTKnowledgeService) -> str:
    """Return the latest summarised feedback insight."""

    return redact_secrets(service.get_recent_insights(FEEDBACK))


def recent_error_fix(service: GPTKnowledgeService) -> str:
    """Return the latest summarised error fix insight."""

    return redact_secrets(service.get_recent_insights(ERROR_FIX))


def recent_improvement_path(service: GPTKnowledgeService) -> str:
    """Return the latest summarised improvement path insight."""

    return redact_secrets(service.get_recent_insights(IMPROVEMENT_PATH))
