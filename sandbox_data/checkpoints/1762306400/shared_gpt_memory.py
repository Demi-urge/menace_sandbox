"""Shared GPT memory manager instance for all ChatGPT clients."""

import sys
from typing import Any, Optional, cast

try:  # pragma: no cover - prefer package-relative import when packaged
    from .gpt_memory import GPTMemoryManager  # type: ignore
except Exception:  # pragma: no cover - fall back to flat layout
    try:
        from gpt_memory import GPTMemoryManager  # type: ignore
    except Exception:  # pragma: no cover - expose a permissive placeholder
        GPTMemoryManager = cast(Any, object)  # type: ignore[assignment]


def _resolve_manager() -> Optional[GPTMemoryManager]:
    """Return the shared memory manager instance if available."""

    manager = _local_manager_from_cache()
    if manager is not None:
        return manager
    try:
        return GPTMemoryManager("gpt_memory.db")  # type: ignore[call-arg]
    except Exception:
        return None


def _local_manager_from_cache() -> Optional[GPTMemoryManager]:
    """Look for a pre-initialised manager exposed by the knowledge module."""

    module_names = (
        "menace_sandbox.shared_knowledge_module",
        "shared_knowledge_module",
    )
    for name in module_names:
        module = sys.modules.get(name)
        if not module:
            continue
        local_module = getattr(module, "LOCAL_KNOWLEDGE_MODULE", None)
        if local_module is None:
            continue
        manager = getattr(local_module, "memory", None)
        if manager is not None:
            return cast(GPTMemoryManager, manager)
    return None


# Single global GPT memory instance reused across bots when available
GPT_MEMORY_MANAGER: Optional[GPTMemoryManager] = _resolve_manager()

__all__ = ["GPT_MEMORY_MANAGER"]
