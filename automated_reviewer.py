from __future__ import annotations

import logging
import json
import threading
import uuid
from typing import Optional

from typing import TYPE_CHECKING, Any, Callable

from .bot_registry import BotRegistry
from .data_bot import DataBot, persist_sc_thresholds
from .coding_bot_interface import (
    _BOOTSTRAP_STATE,
    _bootstrap_dependency_broker,
    _current_bootstrap_context,
    get_active_bootstrap_pipeline,
    normalise_manager_arg,
    prepare_pipeline_for_bootstrap,
    self_coding_managed,
)
from .self_coding_engine import SelfCodingEngine
from .model_automation_pipeline import ModelAutomationPipeline
from .code_database import CodeDB
from .menace_memory_manager import MenaceMemoryManager
from .threshold_service import ThresholdService
from .self_coding_manager import (
    SelfCodingManager,
    internalize_coding_bot,
    _manager_generate_helper_with_builder as manager_generate_helper,
)
from .self_coding_thresholds import get_thresholds
from .shared_evolution_orchestrator import get_orchestrator
from context_builder_util import create_context_builder, ensure_fresh_weights

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from .auto_escalation_manager import AutoEscalationManager
    from .bot_database import BotDB
    from .evolution_orchestrator import EvolutionOrchestrator

# ``vector_service`` is required.  Fail fast if the import is unavailable.
try:  # pragma: no cover - optional dependency used in runtime
    from vector_service import CognitionLayer, ContextBuilder
except Exception as exc:  # pragma: no cover - dependency missing or broken
    raise RuntimeError("vector_service import failed") from exc

# ``FallbackResult`` and ``ErrorResult`` may not always be present on the
# ``vector_service`` module.  Provide light-weight fallbacks so our type checks
# remain functional during tests when the real classes are absent.
try:  # pragma: no cover - optional dependency used in runtime
    from vector_service import FallbackResult  # type: ignore
except Exception:  # pragma: no cover - module missing the attribute
    class FallbackResult:  # type: ignore[override]
        pass

try:  # pragma: no cover - optional dependency used in runtime
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - module missing the attribute
    class ErrorResult(Exception):  # type: ignore[override]
        pass

from snippet_compressor import compress_snippets


registry = BotRegistry()
data_bot = DataBot(start_server=False)

_self_coding_lock = threading.RLock()
_context_builder: ContextBuilder | None = None
_engine: SelfCodingEngine | None = None
_pipeline: ModelAutomationPipeline | None = None
_pipeline_promoter: Callable[[SelfCodingManager], None] | None = None
_evolution_orchestrator = None
_thresholds = None
_manager_instance: SelfCodingManager | None = None


def _ensure_self_coding_manager() -> SelfCodingManager:
    """Initialise shared self-coding infrastructure lazily."""

    global _context_builder, _engine, _pipeline_promoter, _pipeline
    global _evolution_orchestrator, _thresholds, _manager_instance

    with _self_coding_lock:
        if _context_builder is None:
            _context_builder = create_context_builder()

        if _engine is None:
            _engine = SelfCodingEngine(
                CodeDB(), MenaceMemoryManager(), context_builder=_context_builder
            )

        dependency_broker = _bootstrap_dependency_broker()

        bootstrap_pipeline: ModelAutomationPipeline | None = None
        bootstrap_manager: SelfCodingManager | None = None
        promoter: Callable[[SelfCodingManager], None] | None = None

        try:
            bootstrap_pipeline, bootstrap_manager = dependency_broker.resolve()
        except Exception:
            bootstrap_pipeline, bootstrap_manager = None, None

        if _pipeline is None or _manager_instance is None:
            try:
                active_pipeline, active_manager = get_active_bootstrap_pipeline()
            except Exception:
                active_pipeline, active_manager = None, None

            if bootstrap_pipeline is None:
                bootstrap_pipeline = active_pipeline
            if bootstrap_manager is None:
                bootstrap_manager = active_manager

            bootstrap_context = _current_bootstrap_context()
            if bootstrap_context is not None:
                if bootstrap_pipeline is None:
                    bootstrap_pipeline = getattr(bootstrap_context, "pipeline", None)
                if bootstrap_manager is None:
                    bootstrap_manager = getattr(bootstrap_context, "manager", None)

            for candidate in (
                bootstrap_pipeline,
                bootstrap_manager,
                getattr(bootstrap_context, "sentinel", None)
                if "bootstrap_context" in locals()
                else None,
            ):
                if promoter is None and candidate is not None:
                    promoter = getattr(candidate, "_pipeline_promoter", None)

            if _pipeline is None and bootstrap_pipeline is not None:
                _pipeline = bootstrap_pipeline
            if _pipeline_promoter is None and promoter is not None:
                _pipeline_promoter = promoter
            if _manager_instance is None and bootstrap_manager is not None:
                _manager_instance = bootstrap_manager

        bootstrap_heartbeat = False
        try:
            from bootstrap_timeout_policy import read_bootstrap_heartbeat

            bootstrap_heartbeat = bool(read_bootstrap_heartbeat())
        except Exception:
            bootstrap_heartbeat = False

        bootstrap_inflight = bool(
            bootstrap_heartbeat
            or getattr(_BOOTSTRAP_STATE, "depth", 0)
            or _current_bootstrap_context()
        )

        skip_prepare = bootstrap_inflight and bool(_pipeline or _manager_instance)
        advertise_owner = not skip_prepare

        if _pipeline is None and not skip_prepare:
            pipeline, promoter = prepare_pipeline_for_bootstrap(
                pipeline_cls=ModelAutomationPipeline,
                context_builder=_context_builder,
                bot_registry=registry,
                data_bot=data_bot,
            )
            _pipeline = pipeline
            _pipeline_promoter = promoter

        sentinel_candidate: SelfCodingManager | None = (
            _manager_instance or bootstrap_manager
        )
        if _pipeline is not None:
            if sentinel_candidate is None:
                try:
                    sentinel_candidate = getattr(_pipeline, "manager", None)
                except Exception:
                    sentinel_candidate = None
            try:
                dependency_broker.advertise(
                    pipeline=_pipeline,
                    sentinel=sentinel_candidate,
                    owner=True if advertise_owner else None,
                )
            except Exception:
                pass

        if _evolution_orchestrator is None:
            assert _engine is not None
            _evolution_orchestrator = get_orchestrator(
                "AutomatedReviewer", data_bot, _engine
            )

        if _thresholds is None:
            _thresholds = get_thresholds("AutomatedReviewer")
            persist_sc_thresholds(
                "AutomatedReviewer",
                roi_drop=_thresholds.roi_drop,
                error_increase=_thresholds.error_increase,
                test_failure_increase=_thresholds.test_failure_increase,
            )

        if _manager_instance is None:
            manager = internalize_coding_bot(
                "AutomatedReviewer",
                _engine,
                _pipeline,
                data_bot=data_bot,
                bot_registry=registry,
                evolution_orchestrator=_evolution_orchestrator,
                roi_threshold=_thresholds.roi_drop,
                error_threshold=_thresholds.error_increase,
                test_failure_threshold=_thresholds.test_failure_increase,
                threshold_service=ThresholdService(),
            )
            _manager_instance = manager
            if _pipeline_promoter is not None:
                try:
                    _pipeline_promoter(manager)
                finally:
                    _pipeline_promoter = None

    assert _manager_instance is not None
    return _manager_instance


class _ManagerProxy:
    """Proxy resolving the shared self-coding manager lazily."""

    __self_coding_lazy__ = True  # type: ignore[attr-defined]

    def __call__(self) -> SelfCodingManager:
        return _ensure_self_coding_manager()

    def __getattr__(self, item: str) -> Any:
        return getattr(_ensure_self_coding_manager(), item)


manager: SelfCodingManager | _ManagerProxy = _ManagerProxy()


def get_manager() -> SelfCodingManager:
    """Return the shared :class:`SelfCodingManager` instance."""

    return _ensure_self_coding_manager()


def get_engine() -> SelfCodingEngine:
    """Return the lazily initialised :class:`SelfCodingEngine`."""

    _ensure_self_coding_manager()
    assert _engine is not None
    return _engine


def get_pipeline() -> ModelAutomationPipeline:
    """Return the shared :class:`ModelAutomationPipeline` instance."""

    _ensure_self_coding_manager()
    assert _pipeline is not None
    return _pipeline


def get_context_builder() -> ContextBuilder:
    """Return the shared :class:`ContextBuilder` instance."""

    _ensure_self_coding_manager()
    assert _context_builder is not None
    return _context_builder


def __getattr__(name: str) -> Any:
    if name == "engine":
        return get_engine()
    if name == "pipeline":
        return get_pipeline()
    if name == "context_builder":
        return get_context_builder()
    if name == "manager":
        return get_manager()
    raise AttributeError(name)


@self_coding_managed(bot_registry=registry, data_bot=data_bot, manager=manager)
class AutomatedReviewer:
    """Analyse review events and trigger remediation."""

    manager: SelfCodingManager

    def __init__(
        self,
        context_builder: ContextBuilder,
        bot_db: "BotDB" | None = None,
        escalation_manager: "AutoEscalationManager" | None = None,
        *,
        manager: "SelfCodingManager" | None = None,
    ) -> None:
        if bot_db is None:
            from .bot_database import BotDB

            bot_db = BotDB()
        self.bot_db = bot_db
        if escalation_manager is None:
            from .auto_escalation_manager import AutoEscalationManager

            escalation_manager = AutoEscalationManager(context_builder=context_builder)
        self.escalation_manager = escalation_manager
        self.logger = logging.getLogger(self.__class__.__name__)
        self.manager = normalise_manager_arg(manager, type(self))
        try:
            name = getattr(self, "name", getattr(self, "bot_name", self.__class__.__name__))
            self.manager.register_bot(name)
            orch = getattr(self.manager, "evolution_orchestrator", None)
            if orch:
                orch.register_bot(name)
        except Exception:  # pragma: no cover - best effort
            self.logger.exception("bot registration failed")
        evo = getattr(self.manager, "evolution_orchestrator", None)
        if evo:
            try:  # pragma: no cover - best effort registration
                evo.register_bot(self.manager.bot_name)
            except Exception:
                self.logger.exception("evolution orchestrator registration failed")
            try:  # pragma: no cover - best effort subscription
                evo._ensure_degradation_subscription()
            except Exception:
                self.logger.exception("failed to subscribe to degradation events")
        if context_builder is None:
            raise ValueError("context_builder is required")
        if not hasattr(context_builder, "build"):
            raise TypeError("context_builder must implement build()")
        self.context_builder = context_builder
        try:
            ensure_fresh_weights(self.context_builder)
        except Exception as exc:
            self.logger.error("context builder refresh failed: %s", exc)
            raise RuntimeError("context builder refresh failed") from exc
        try:
            self.cognition_layer = CognitionLayer(context_builder=context_builder)
        except Exception:  # pragma: no cover - optional dependency failed
            self.logger.error("failed to initialise CognitionLayer", exc_info=True)
            raise

    # ------------------------------------------------------------------
    def handle(self, event: object) -> None:
        """Process a review event."""
        bot_id: Optional[str] = None
        severity: Optional[str] = None
        if isinstance(event, dict):
            bot_id = event.get("bot_id")
            severity = event.get("severity")
        if not bot_id:
            self.logger.warning("invalid review event: %s", event)
            return
        if severity == "critical":
            try:
                self.bot_db.update_bot(int(bot_id), status="disabled")
            except Exception:
                self.logger.exception("failed disabling bot %s", bot_id)

            session_id = str(uuid.uuid4())
            ctx: str = ""
            vectors: list[tuple[str, str, float]] = []
            try:
                ctx_res = self.context_builder.build(
                    json.dumps({"bot_id": bot_id, "severity": severity}),
                    session_id=session_id,
                    include_vectors=True,
                )
                context_obj = ctx_res
                if isinstance(ctx_res, tuple):
                    context_obj, session_id, vectors = ctx_res
                if isinstance(context_obj, (FallbackResult, ErrorResult)):
                    ctx = ""
                elif isinstance(context_obj, dict):
                    ctx = json.dumps(compress_snippets(context_obj))
                else:
                    ctx = context_obj  # type: ignore[assignment]
            except Exception:
                ctx = ""
                vectors = []
            try:
                manager_generate_helper(
                    self.manager,
                    f"review for bot {bot_id}",
                    context_builder=self.context_builder,
                )
            except Exception:
                self.logger.exception("helper generation failed")
            try:
                self.escalation_manager.handle(
                    f"review for bot {bot_id}",
                    attachments=[ctx],
                    session_id=session_id,
                    vector_metadata=vectors,
                )
            except Exception:
                self.logger.exception("escalation failed")

            roi = 0.0
            errors = 0.0
            try:
                roi = data_bot.roi(str(bot_id))
            except Exception:
                self.logger.exception("failed to fetch ROI for %s", bot_id)
            try:
                errors = data_bot.average_errors(str(bot_id))
            except Exception:
                self.logger.exception("failed to fetch errors for %s", bot_id)
            try:
                data_bot.record_metrics(str(bot_id), float(roi), float(errors))
            except Exception:
                self.logger.exception("failed to record metrics for %s", bot_id)
            try:
                data_bot.check_degradation(str(bot_id), float(roi), float(errors))
            except Exception:
                self.logger.exception("degradation check failed for %s", bot_id)


__all__ = ["AutomatedReviewer"]
