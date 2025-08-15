from __future__ import annotations

"""Shared GPT memory manager instance for all ChatGPT clients."""

import os

from gpt_memory import GPTMemoryManager
from local_knowledge_module import init_local_knowledge

# Single global instance reused across bots
GPT_MEMORY_MANAGER: GPTMemoryManager = init_local_knowledge(
    os.getenv("GPT_MEMORY_DB", "gpt_memory.db")
).memory

__all__ = ["GPT_MEMORY_MANAGER"]
