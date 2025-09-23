"""Safety monitor for validating bots and workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, TYPE_CHECKING
import sqlite3
import logging

from db_router import DBRouter, GLOBAL_ROUTER
from .unified_event_bus import UnifiedEventBus

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .bot_testing_bot import BotTestingBot


@dataclass
class SafetyConfig:
    """Configuration for validation thresholds."""

    fail_threshold: int = 3


class SafetyMonitor:
    """Run basic smoke tests and sanity checks before deployment."""

    def __init__(
        self,
        tester: "BotTestingBot" | None = None,
        event_bus: Optional[UnifiedEventBus] = None,
        *,
        config: SafetyConfig | None = None,
    ) -> None:
        if tester is None:
            from .bot_testing_bot import BotTestingBot as _BotTestingBot

            tester = _BotTestingBot()
        self.tester = tester
        self.event_bus = event_bus
        self.config = config or SafetyConfig()
        self.fail_counts: Dict[str, int] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def _flag(self, key: str) -> None:
        count = self.fail_counts.get(key, 0) + 1
        self.fail_counts[key] = count
        if count >= self.config.fail_threshold and self.event_bus:
            try:
                self.event_bus.publish("safety:flag", {"target": key, "count": count})
            except Exception as exc:
                self.logger.error("failed publishing safety flag: %s", exc)

    def validate_bot(self, bot_id: str) -> bool:
        """Run quick unit tests for *bot_id*."""
        results = self.tester.run_unit_tests([bot_id])
        passed = all(r.passed for r in results)
        if not passed:
            self._flag(bot_id)
        else:
            self.fail_counts.pop(bot_id, None)
        return passed

    def validate_workflow(
        self,
        workflow_id: int,
        db: Optional[object] = None,
        router: DBRouter | None = None,
    ) -> bool:
        """Check workflow integrity using ``router`` if provided."""
        tasks: Iterable[str] = []
        router = router or GLOBAL_ROUTER
        if db and router:
            try:
                conn = router.get_connection("workflows")
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT workflow FROM workflows WHERE id=?",
                    (workflow_id,),
                ).fetchone()
                if row and row["workflow"]:
                    tasks = row["workflow"].split(",")
            except Exception:
                tasks = []
        passed = bool(tasks)
        if tasks:
            results = self.tester.run_unit_tests(list(tasks))
            passed = all(r.passed for r in results)
        if not passed:
            self._flag(f"workflow:{workflow_id}")
        else:
            self.fail_counts.pop(f"workflow:{workflow_id}", None)
        return passed


__all__ = ["SafetyMonitor", "SafetyConfig"]
