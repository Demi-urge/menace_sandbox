from __future__ import annotations

import logging
from typing import Set

logger = logging.getLogger(__name__)

# In-memory record of bots that have been paused for review.
_PAUSED_BOTS: Set[str] = set()

def pause_bot(bot_id: str) -> bool:
    """Mark *bot_id* as paused/pending review.

    The operation is idempotent; subsequent calls for the same bot have no
    effect. Returns ``True`` when the bot was newly paused and ``False`` when it
    was already paused.
    """
    if bot_id in _PAUSED_BOTS:
        logger.debug("bot '%s' already paused", bot_id)
        return False
    _PAUSED_BOTS.add(bot_id)
    logger.info("bot '%s' paused for review", bot_id)
    return True

def is_paused(bot_id: str) -> bool:
    """Return ``True`` if *bot_id* has been paused."""
    return bot_id in _PAUSED_BOTS

def reset() -> None:
    """Clear paused bot state (for tests)."""
    _PAUSED_BOTS.clear()
