from __future__ import annotations

"""Shared GPT memory manager instance for all ChatGPT clients."""

from gpt_memory import GPTMemoryManager

# Single global instance reused across bots
GPT_MEMORY_MANAGER = GPTMemoryManager()

__all__ = ["GPT_MEMORY_MANAGER"]
