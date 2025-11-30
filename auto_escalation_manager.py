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

from .bot_registry import BotRegistry
from .data_bot import DataBot
from .coding_bot_interface import (
    _GLOBAL_BOOTSTRAP_COORDINATOR,
    _bootstrap_dependency_broker,
    _current_bootstrap_context,
    _is_bootstrap_placeholder,
    _looks_like_pipeline_candidate,
    _using_bootstrap_sentinel,
    prepare_pipeline_for_bootstrap,
    self_coding_managed,
)

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


registry = BotRegistry()
data_bot = DataBot()


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
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
        bootstrap_context: object | None = None,
        bootstrap_dependency_broker: object | None = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
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
                try:
                    bootstrap_context = bootstrap_context or _current_bootstrap_context()
                    dependency_broker = (
                        bootstrap_dependency_broker
                        or getattr(bootstrap_context, "dependency_broker", None)
                        or _bootstrap_dependency_broker()
                    )

                    broker_pipeline, broker_manager = dependency_broker.resolve()

                    pipeline_candidate = None
                    manager_candidate = None
                    promote_pipeline = None

                    if bootstrap_context is not None:
                        pipeline_candidate = getattr(bootstrap_context, "pipeline", None)
                        manager_candidate = getattr(bootstrap_context, "manager", None)
                        if manager_candidate is None:
                            manager_candidate = getattr(bootstrap_context, "sentinel", None)

                    if pipeline_candidate is None and _looks_like_pipeline_candidate(
                        broker_pipeline
                    ):
                        pipeline_candidate = broker_pipeline
                    if manager_candidate is None:
                        manager_candidate = broker_manager

                    if pipeline_candidate is None:
                        active_promise = getattr(
                            _GLOBAL_BOOTSTRAP_COORDINATOR, "_active", None
                        )
                        if active_promise is not None and not active_promise.done:
                            try:
                                pipeline_candidate, promote_pipeline = active_promise.wait()
                                dependency_broker.advertise(
                                    pipeline=pipeline_candidate,
                                    sentinel=getattr(pipeline_candidate, "manager", None),
                                )
                            except Exception as exc:  # pragma: no cover - defensive
                                raise RuntimeError(
                                    "AutoEscalationManager bootstrap promise failed"
                                ) from exc

                    pipeline = None

                    if pipeline_candidate is not None:
                        if not _looks_like_pipeline_candidate(pipeline_candidate):
                            raise RuntimeError(
                                "AutoEscalationManager received invalid bootstrap pipeline"
                            )
                        pipeline = pipeline_candidate
                        if manager_candidate is None:
                            manager_candidate = getattr(pipeline_candidate, "manager", None)
                        if _is_bootstrap_placeholder(manager_candidate):
                            self.logger.info(
                                "auto_escalation.bootstrap.reuse_placeholder",
                                extra={
                                    "event": "auto-escalation-bootstrap-placeholder-reuse",
                                },
                            )
                        if promote_pipeline is None:
                            promote_pipeline = getattr(
                                pipeline_candidate, "_pipeline_promoter", lambda *_a: None
                            )
                    elif manager_candidate is not None and _using_bootstrap_sentinel(
                        manager_candidate
                    ):
                        raise RuntimeError(
                            "AutoEscalationManager cannot resolve bootstrap pipeline from sentinel"
                        )
                    else:
                        from .self_coding_manager import SelfCodingManager
                        from .model_automation_pipeline import ModelAutomationPipeline

                        pipeline, promote_pipeline = prepare_pipeline_for_bootstrap(
                            pipeline_cls=ModelAutomationPipeline,
                            context_builder=self.context_builder,
                            bot_registry=registry,
                            data_bot=data_bot,
                            event_bus=event_bus,
                            manager_override=manager_candidate,
                        )

                    if pipeline is None or promote_pipeline is None:
                        raise RuntimeError(
                            "AutoEscalationManager failed to resolve bootstrap pipeline"
                        )

                    from .self_coding_manager import SelfCodingManager
                    from .model_automation_pipeline import ModelAutomationPipeline

                    manager = SelfCodingManager(
                        engine,
                        pipeline,
                        bot_name=self.__class__.__name__,
                        bot_registry=registry,
                        data_bot=data_bot,
                        event_bus=event_bus,
                    )
                    promote_pipeline(manager)
                    manager.register_bot(self.__class__.__name__)
                    self.manager = manager
                except Exception as exc:
                    logger = logging.getLogger(self.__class__.__name__)
                    logger.exception("failed to initialize SelfCodingManager")
                    raise RuntimeError(
                        "SelfCodingManager initialization failed"
                    ) from exc
                debugger = AutomatedDebugger(
                    ErrorDB(), engine, self.context_builder, manager=manager
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
