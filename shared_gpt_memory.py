from __future__ import annotations

"""Shared GPT memory manager instance for all ChatGPT clients."""

from typing import Any, Optional, cast

try:  # pragma: no cover - prefer package-relative import when packaged
    from .gpt_memory import GPTMemoryManager  # type: ignore
except Exception:  # pragma: no cover - fall back to flat layout
    try:
        from gpt_memory import GPTMemoryManager  # type: ignore
    except Exception:  # pragma: no cover - expose a permissive placeholder
        GPTMemoryManager = cast(Any, object)  # type: ignore[assignment]

try:  # pragma: no cover - prefer package-relative import when packaged
    from .shared_knowledge_module import LOCAL_KNOWLEDGE_MODULE  # type: ignore
except Exception:  # pragma: no cover - fall back to flat layout
    try:
        from shared_knowledge_module import LOCAL_KNOWLEDGE_MODULE  # type: ignore
    except Exception:  # pragma: no cover - degrade gracefully
        LOCAL_KNOWLEDGE_MODULE = None  # type: ignore[assignment]


def _resolve_manager() -> Optional[GPTMemoryManager]:
    """Return the shared memory manager instance if available."""

    manager = None
    if LOCAL_KNOWLEDGE_MODULE is not None:
        manager = getattr(LOCAL_KNOWLEDGE_MODULE, "memory", None)
    if manager is not None:
        return cast(GPTMemoryManager, manager)
    try:
        return GPTMemoryManager("gpt_memory.db")  # type: ignore[call-arg]
    except Exception:
        return None


# Single global GPT memory instance reused across bots when available
GPT_MEMORY_MANAGER: Optional[GPTMemoryManager] = _resolve_manager()

__all__ = ["GPT_MEMORY_MANAGER"]
