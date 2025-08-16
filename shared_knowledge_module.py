from __future__ import annotations

"""Process-wide LocalKnowledgeModule shared across components.

This module exposes a single :data:`LOCAL_KNOWLEDGE_MODULE` instance which
combines :class:`gpt_memory.GPTMemoryManager` with
:class:`gpt_knowledge_service.GPTKnowledgeService`.  The underlying database
location can be customised through the ``GPT_MEMORY_DB`` environment variable.
"""

import os
from pathlib import Path

from local_knowledge_module import init_local_knowledge, LocalKnowledgeModule

# Resolve database path from environment or fall back to default location.
_MEM_DB = Path(os.getenv("GPT_MEMORY_DB", "gpt_memory.db"))

# Public singleton instance reused by all modules.
LOCAL_KNOWLEDGE_MODULE: LocalKnowledgeModule = init_local_knowledge(_MEM_DB)

__all__ = ["LOCAL_KNOWLEDGE_MODULE", "LocalKnowledgeModule"]
