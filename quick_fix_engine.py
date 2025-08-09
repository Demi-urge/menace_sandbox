from __future__ import annotations

"""Automatically propose fixes for recurring errors."""

from pathlib import Path
import logging
import subprocess
from typing import Tuple

from .error_bot import ErrorDB
from .self_coding_manager import SelfCodingManager
from .knowledge_graph import KnowledgeGraph


class QuickFixEngine:
    """Analyse frequent errors and trigger small patches."""

    def __init__(
        self,
        error_db: ErrorDB,
        manager: SelfCodingManager,
        *,
        threshold: int = 3,
        graph: KnowledgeGraph | None = None,
    ) -> None:
        self.db = error_db
        self.manager = manager
        self.threshold = threshold
        self.graph = graph or KnowledgeGraph()
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def _top_error(
        self, bot: str
    ) -> Tuple[str, str, dict[str, int], int] | None:
        try:
            info = self.db.top_error_module(bot)
        except Exception:
            return None
        if not info:
            return None
        etype, module, mods, count, _ = info
        return etype, module, mods, count

    def run(self, bot: str) -> None:
        """Attempt a quick patch for the most frequent error of ``bot``."""
        info = self._top_error(bot)
        if not info:
            return
        etype, module, mods, count = info
        if count < self.threshold:
            return
        path = Path(f"{module}.py")
        if not path.exists():
            return
        desc = f"quick fix {etype}"
        try:
            self.manager.run_patch(path, desc)
        except Exception as exc:  # pragma: no cover - runtime issues
            self.logger.error("quick fix failed for %s: %s", bot, exc)
        try:
            self.graph.add_telemetry_event(bot, etype, module, mods)
            self.graph.update_error_stats(self.db)
        except Exception as exc:
            self.logger.exception("telemetry update failed: %s", exc)
            raise

    # ------------------------------------------------------------------
    def run_and_validate(self, bot: str) -> None:
        """Run :meth:`run` then execute the test suite."""
        self.run(bot)
        try:
            subprocess.run(["pytest", "-q"], check=True)
        except Exception as exc:
            self.logger.error("quick fix validation failed: %s", exc)


__all__ = ["QuickFixEngine"]
