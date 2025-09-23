from __future__ import annotations

"""Shared GPT memory manager instance for all ChatGPT clients."""

from menace_sandbox.gpt_memory import GPTMemoryManager
from shared_knowledge_module import LOCAL_KNOWLEDGE_MODULE

# Single global GPT memory instance reused across bots
GPT_MEMORY_MANAGER: GPTMemoryManager = LOCAL_KNOWLEDGE_MODULE.memory

__all__ = ["GPT_MEMORY_MANAGER"]
