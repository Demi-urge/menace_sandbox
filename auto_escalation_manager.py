from __future__ import annotations

"""Automatic handler for escalation events."""

import logging
import os
from typing import Iterable, Optional
from .retry_utils import retry

from .advanced_error_management import SelfHealingOrchestrator
from .knowledge_graph import KnowledgeGraph
from .automated_debugger import AutomatedDebugger
from .self_coding_engine import SelfCodingEngine
from .code_database import CodeDB
from .rollback_manager import RollbackManager
from .error_bot import ErrorDB
from .unified_event_bus import UnifiedEventBus
from .local_knowledge_module import init_local_knowledge
try:
    from vector_service.context_builder_utils import get_default_context_builder
except ImportError:  # pragma: no cover - fallback when helper missing
    from vector_service.context_builder import ContextBuilder  # type: ignore

    def get_default_context_builder(**kwargs):  # type: ignore
        return ContextBuilder(**kwargs)


class AutoEscalationManager:
    """Analyse issues and initiate remediation without human input."""

    def __init__(
        self,
        healer: SelfHealingOrchestrator | None = None,
        debugger: AutomatedDebugger | None = None,
        rollback_mgr: RollbackManager | None = None,
        event_bus: UnifiedEventBus | None = None,
        *,
        publish_attempts: int = 1,
    ) -> None:
        self.healer = healer or SelfHealingOrchestrator(KnowledgeGraph())
        if debugger is None:
            gpt_mem = init_local_knowledge(
                os.getenv("GPT_MEMORY_DB", "gpt_memory.db")
            ).memory
            builder = get_default_context_builder()
            builder.refresh_db_weights()
            engine = SelfCodingEngine(
                CodeDB(), gpt_mem, event_bus=event_bus, context_builder=builder
            )
            debugger = AutomatedDebugger(ErrorDB(), engine, context_builder=builder)
        self.debugger = debugger
        self.rollback_mgr = rollback_mgr
        self.event_bus = event_bus
        self.publish_attempts = max(1, publish_attempts)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def handle(self, message: str, attachments: Iterable[str] | None = None) -> None:
        """Attempt automated recovery actions."""
        try:
            self.debugger.analyse_and_fix()
        except Exception:
            self.logger.exception("debugger failed")
        try:
            self.healer.probe_and_heal("menace")
        except Exception:
            self.logger.exception("healing failed")
        if self.rollback_mgr:
            try:
                self.rollback_mgr.auto_rollback("latest", [])
            except Exception:
                self.logger.exception("rollback failed")
        if self.event_bus:
            @retry(Exception, attempts=self.publish_attempts, delay=0.1)
            def _publish() -> None:
                self.event_bus.publish("escalation:handled", {"message": message})

            try:
                _publish()
            except Exception:
                self.logger.exception("failed publishing escalation event")


__all__ = ["AutoEscalationManager"]
