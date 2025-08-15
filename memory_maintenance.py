"""Background maintenance utilities for :mod:`gpt_memory`."""

from __future__ import annotations

import os
import threading
from typing import Dict, Mapping, TYPE_CHECKING

from gpt_memory import GPTMemoryManager

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from gpt_knowledge_service import GPTKnowledgeService
from logging_utils import get_logger

logger = get_logger(__name__)

# Interval in seconds between maintenance cycles
DEFAULT_INTERVAL = float(os.getenv("GPT_MEMORY_COMPACT_INTERVAL", "3600"))


def _load_prune_limit() -> int | None:
    """Return the max rows per tag parsed from ``GPT_MEMORY_MAX_ROWS``."""

    try:
        limit = int(os.getenv("GPT_MEMORY_MAX_ROWS", "0"))
    except ValueError:
        return None
    return limit if limit > 0 else None


def _load_retention_rules() -> Dict[str, int]:
    """Parse per-tag retention limits from environment variables."""
    rules: Dict[str, int] = {}
    # Combined mapping: "tag1=10,tag2=5"
    env_map = os.getenv("GPT_MEMORY_RETENTION", "")
    for item in env_map.split(","):
        if not item.strip():
            continue
        tag, _, num = item.partition("=")
        try:
            rules[tag.strip()] = int(num.strip())
        except ValueError:
            continue
    # Prefix variables: GPT_MEMORY_RETENTION_<TAG>=N
    prefix = "GPT_MEMORY_RETENTION_"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            tag = key[len(prefix) :].lower()
            try:
                rules[tag] = int(value)
            except ValueError:
                continue
    return rules


def prune_memory(manager: GPTMemoryManager, max_rows: int) -> int:
    """Convenience wrapper around :meth:`GPTMemoryManager.prune_old_entries`."""

    removed = manager.prune_old_entries(max_rows)
    logger.debug("memory maintenance pruned %d rows", removed)
    return removed


class MemoryMaintenance:
    """Background thread that periodically compacts GPT memory."""

    def __init__(
        self,
        manager: GPTMemoryManager,
        *,
        interval: float | None = None,
        retention: Mapping[str, int] | None = None,
        max_rows: int | None = None,
        knowledge_service: GPTKnowledgeService | None = None,
    ) -> None:
        self.manager = manager
        self.interval = float(interval or DEFAULT_INTERVAL)
        self.retention: Dict[str, int] = (
            dict(retention) if retention is not None else _load_retention_rules()
        )
        self.max_rows = max_rows if max_rows is not None else _load_prune_limit()
        self.knowledge_service = knowledge_service
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if not self.retention and not self.max_rows:
            logger.debug(
                "memory maintenance disabled – no retention or pruning rules configured"
            )
            return
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------ internal
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                if self.retention:
                    removed = self.manager.compact(self.retention)
                    logger.debug("memory maintenance removed %d rows", removed)
                if self.max_rows:
                    pruned = self.manager.prune_old_entries(self.max_rows)
                    logger.debug("memory maintenance pruned %d rows", pruned)
            except Exception:
                logger.exception("memory maintenance failed during maintenance")

            if self.knowledge_service is not None:
                try:
                    self.knowledge_service.update_insights()
                except Exception:
                    logger.exception("knowledge service update failed")

            kg = getattr(self.manager, "graph", None)
            if kg is not None:
                try:
                    kg.ingest_gpt_memory(self.manager)
                except Exception:
                    logger.exception("knowledge graph update failed")
