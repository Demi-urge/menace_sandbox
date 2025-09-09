from __future__ import annotations

"""Automatic handler for escalation events.

This module requires :class:`vector_service.ContextBuilder`.  A clear
``ImportError`` is raised during import when the dependency is missing to
avoid silently running without context retrieval capabilities.
"""

import logging
import os
from typing import Iterable
from .retry_utils import retry

from .knowledge_graph import KnowledgeGraph
try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path  # type: ignore

try:
    from vector_service.context_builder import ContextBuilder
except Exception as exc:  # pragma: no cover - fail fast when dependency missing
    raise ImportError(
        "auto_escalation_manager requires vector_service.ContextBuilder; install the"
        " vector_service package to enable context retrieval"
    ) from exc
try:  # pragma: no cover - optional dependency
    from .automated_debugger import AutomatedDebugger
except Exception:  # pragma: no cover - gracefully degrade in tests
    AutomatedDebugger = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .self_coding_engine import SelfCodingEngine
except Exception:  # pragma: no cover - gracefully degrade in tests
    SelfCodingEngine = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .code_database import CodeDB
except Exception:  # pragma: no cover - gracefully degrade in tests
    CodeDB = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .error_bot import ErrorDB
except Exception:  # pragma: no cover - gracefully degrade in tests
    ErrorDB = None  # type: ignore
try:  # pragma: no cover - optional dependency
    from .unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - gracefully degrade in tests
    class UnifiedEventBus:  # type: ignore[override]
        def publish(self, *a, **k) -> None:
            pass
try:  # pragma: no cover - optional dependency
    from .local_knowledge_module import init_local_knowledge
except Exception:  # pragma: no cover - gracefully degrade in tests
    def init_local_knowledge(*a, **k):  # type: ignore
        class _Mem:
            memory = None

        return _Mem()
try:  # pragma: no cover - optional dependency
    from .advanced_error_management import SelfHealingOrchestrator
except Exception:  # pragma: no cover - gracefully degrade in tests
    class SelfHealingOrchestrator:  # type: ignore[override]
        def __init__(self, *a, **k) -> None:
            pass

        def probe_and_heal(self, *a, **k) -> None:
            pass

try:  # pragma: no cover - optional dependency
    from .rollback_manager import RollbackManager
except Exception:  # pragma: no cover - gracefully degrade in tests
    class RollbackManager:  # type: ignore[override]
        def auto_rollback(self, *a, **k) -> None:
            pass


class AutoEscalationManager:
    """Analyse issues and initiate remediation without human input."""

    def __init__(
        self,
        healer: SelfHealingOrchestrator | None = None,
        debugger: AutomatedDebugger | None = None,
        rollback_mgr: RollbackManager | None = None,
        event_bus: UnifiedEventBus | None = None,
        *,
        context_builder: ContextBuilder,
        publish_attempts: int = 1,
    ) -> None:
        self.healer = healer or SelfHealingOrchestrator(KnowledgeGraph())

        # Use the provided context builder for the debugger and expose it for reuse.
        self.context_builder = context_builder
        self.context_builder.refresh_db_weights()

        if debugger is None:
            if (
                SelfCodingEngine is not None
                and CodeDB is not None
                and ErrorDB is not None
                and AutomatedDebugger is not None
            ):
                gpt_mem = init_local_knowledge(
                    os.getenv(
                        "GPT_MEMORY_DB",
                        resolve_path("gpt_memory.db").as_posix(),
                    )
                ).memory
                engine = SelfCodingEngine(
                    CodeDB(),
                    gpt_mem,
                    event_bus=event_bus,
                    context_builder=self.context_builder,
                )
                debugger = AutomatedDebugger(
                    ErrorDB(), engine, context_builder=self.context_builder
                )
            else:  # pragma: no cover - fallback when components missing
                class _DummyDebugger:
                    def analyse_and_fix(self) -> None:
                        pass

                debugger = _DummyDebugger()
        self.debugger = debugger
        self.rollback_mgr = rollback_mgr
        self.event_bus = event_bus
        self.publish_attempts = max(1, publish_attempts)
        self.logger = logging.getLogger(self.__class__.__name__)

    # ------------------------------------------------------------------
    def handle(
        self,
        message: str,
        attachments: Iterable[str] | None = None,
        session_id: str | None = None,
        vector_metadata: list[tuple[str, str, float]] | None = None,
    ) -> None:
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
