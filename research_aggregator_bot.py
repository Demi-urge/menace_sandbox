"""Research Aggregator Bot for autonomous multi-layered research.

This module participates in the bootstrap-sensitive pipeline chain; it must
advertise or reuse the broker-published bootstrap placeholder before
constructing any pipelines so downstream helpers preserve the broker-first
pattern. See ``docs/bootstrap_troubleshooting.md`` for guidance on keeping the
placeholder visible during orchestration.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Type, Callable

if __package__ in (None, ""):
    from menace_sandbox.bootstrap_gate import resolve_bootstrap_placeholders
else:
    from .bootstrap_gate import resolve_bootstrap_placeholders

from .bot_registry import BotRegistry
from .data_bot import DataBot, persist_sc_thresholds

from .coding_bot_interface import (
    _BOOTSTRAP_STATE,
    _looks_like_pipeline_candidate,
    _bootstrap_dependency_broker,
    advertise_bootstrap_placeholder,
    read_bootstrap_heartbeat,
    get_active_bootstrap_pipeline,
    _current_bootstrap_context,
    _using_bootstrap_sentinel,
    _peek_owner_promise,
    _GLOBAL_BOOTSTRAP_COORDINATOR,
    _resolve_bootstrap_wait_timeout,
    claim_bootstrap_dependency_entry,
    prepare_pipeline_for_bootstrap,
    self_coding_managed,
)
from .bootstrap_helpers import bootstrap_state_snapshot, ensure_bootstrapped
from bootstrap_readiness import readiness_signal, probe_embedding_service
from bootstrap_timeout_policy import resolve_bootstrap_gate_timeout
from .self_coding_manager import SelfCodingManager, internalize_coding_bot
from .self_coding_engine import SelfCodingEngine
if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model_automation_pipeline import ModelAutomationPipeline
else:  # pragma: no cover - runtime fallback keeps runtime import lazy
    ModelAutomationPipeline = object  # type: ignore[misc, assignment]
from .threshold_service import ThresholdService
from .code_database import CodeDB
from .gpt_memory import GPTMemoryManager
from .self_coding_thresholds import get_thresholds
from vector_service.context_builder import ContextBuilder
from .shared_evolution_orchestrator import get_orchestrator
from context_builder_util import create_context_builder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .evolution_orchestrator import EvolutionOrchestrator
    from .capital_management_bot import CapitalManagementBot
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, cast

import os
import logging
from types import ModuleType, SimpleNamespace

from .chatgpt_enhancement_bot import (
    EnhancementDB,
    ChatGPTEnhancementBot,
    Enhancement,
)
from .chatgpt_prediction_bot import ChatGPTPredictionBot, IdeaFeatures
from .text_research_bot import TextResearchBot
try:
    from .video_research_bot import VideoResearchBot
except Exception:  # pragma: no cover - optional dependency
    VideoResearchBot = None  # type: ignore
from .chatgpt_research_bot import ChatGPTResearchBot, Exchange
from .database_manager import get_connection, DB_PATH
from .db_router import DBRouter
from vector_service import ContextBuilder
from snippet_compressor import compress_snippets
from .research_storage import InfoDB, ResearchItem

logger = logging.getLogger(__name__)
_VECTOR_BOOTSTRAP_SKIP_ENV = "SKIP_VECTOR_BOOTSTRAP"
_VECTOR_SEEDING_STRICT_ENV = "VECTOR_SEEDING_STRICT"


def _vector_bootstrap_disabled() -> bool:
    raw_skip = os.getenv(_VECTOR_BOOTSTRAP_SKIP_ENV, "").strip().lower()
    if raw_skip in {"1", "true", "yes", "on"}:
        return True
    raw_strict = os.getenv(_VECTOR_SEEDING_STRICT_ENV, "").strip().lower()
    if raw_strict in {"0", "false", "no", "off"}:
        return True
    return False
_BOOTSTRAP_READINESS = readiness_signal()

_BOOTSTRAP_PLACEHOLDER: object | None = None
_BOOTSTRAP_SENTINEL: object | None = None
_BOOTSTRAP_BROKER: object | None = None
def _bootstrap_gate_timeout(*, vector_heavy: bool = True, fallback: float | None = None) -> float:
    resolved_fallback = fallback if fallback is not None else 180.0
    resolved_wait = _resolve_bootstrap_wait_timeout(vector_heavy=vector_heavy)
    if resolved_wait is not None:
        resolved_fallback = max(resolved_fallback, resolved_wait)
    resolved_fallback = max(resolved_fallback, 180.0)
    return resolve_bootstrap_gate_timeout(vector_heavy=vector_heavy, fallback_timeout=resolved_fallback)


_BOOTSTRAP_GATE_TIMEOUT = _bootstrap_gate_timeout(vector_heavy=True)
_DEPENDENCY_RESOLUTION_WAIT_FLOOR = 0.05
_DEPENDENCY_RESOLUTION_MAX_WAIT_DEFAULT = 3.0


def _bootstrap_placeholders(allow_degraded: bool = False) -> tuple[object, object, object]:
    """Resolve bootstrap placeholders after the readiness gate clears."""

    global _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER
    if None not in (_BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER):
        return _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER

    active_pipeline, active_manager = get_active_bootstrap_pipeline()
    broker = _bootstrap_dependency_broker()
    broker_owner = bool(getattr(broker, "active_owner", False))
    if active_pipeline is not None or active_manager is not None:
        _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL = advertise_bootstrap_placeholder(
            dependency_broker=broker,
            pipeline=active_pipeline,
            manager=active_manager,
            owner=broker_owner,
        )
        _BOOTSTRAP_BROKER = broker
        if not broker_owner:
            logger.error(
                "Bootstrap dependency broker missing active owner; reusing active ResearchAggregatorBot placeholder",
                extra={
                    "event": "research-aggregator-broker-owner-missing",
                    "broker_owner": broker_owner,
                },
            )
        return _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER

    if not broker_owner:
        logger.warning(
            "Bootstrap dependency broker owner inactive; continuing with degraded ResearchAggregatorBot placeholders",
            extra={
                "event": "research-aggregator-broker-owner-missing",
                "broker_owner": broker_owner,
            },
        )
        if allow_degraded:
            logger.warning(
                "Bootstrap dependency broker owner inactive; entering degraded ResearchAggregatorBot bootstrap",
                extra={
                    "event": "research-aggregator-bootstrap-degraded",
                    "broker_owner": broker_owner,
                },
            )
        if any(
            item is not None
            for item in (_BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER)
        ):
            logger.warning(
                "Bootstrap dependency broker owner inactive; returning cached ResearchAggregatorBot placeholders",
                extra={
                    "event": "research-aggregator-bootstrap-degraded",
                    "broker_owner": broker_owner,
                },
            )
            if _BOOTSTRAP_BROKER is None:
                _BOOTSTRAP_BROKER = broker
            return _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER
        placeholder, sentinel = advertise_bootstrap_placeholder(
            dependency_broker=broker,
            owner=False,
        )
        _BOOTSTRAP_PLACEHOLDER = placeholder
        _BOOTSTRAP_SENTINEL = sentinel
        _BOOTSTRAP_BROKER = broker
        return _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER

    pipeline, manager, broker = resolve_bootstrap_placeholders(
        timeout=_BOOTSTRAP_GATE_TIMEOUT,
        description="ResearchAggregatorBot bootstrap gate",
    )
    if not getattr(broker, "active_owner", False):
        logger.error(
            "Bootstrap dependency broker missing active owner; reusing cached placeholder for ResearchAggregatorBot",
            extra={
                "event": "research-aggregator-broker-owner-missing",
                "broker_owner": bool(getattr(broker, "active_owner", False)),
            },
        )
        return (
            _BOOTSTRAP_PLACEHOLDER or pipeline,
            _BOOTSTRAP_SENTINEL or manager,
            _BOOTSTRAP_BROKER or broker,
        )
    _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL = advertise_bootstrap_placeholder(
        dependency_broker=broker,
        pipeline=pipeline,
        manager=manager,
    )
    if not getattr(broker, "active_owner", False):
        raise RuntimeError(
            "Failed to advertise bootstrap placeholder with active owner; refusing to construct ResearchAggregatorBot"
        )
    _BOOTSTRAP_BROKER = broker
    return _BOOTSTRAP_PLACEHOLDER, _BOOTSTRAP_SENTINEL, _BOOTSTRAP_BROKER


@dataclass(frozen=True)
class _RuntimeDependencies:
    """Container for lazily created runtime helpers."""

    registry: BotRegistry
    data_bot: DataBot
    context_builder: ContextBuilder
    engine: SelfCodingEngine
    pipeline: "ModelAutomationPipeline"
    evolution_orchestrator: "EvolutionOrchestrator | None"
    manager: SelfCodingManager | None
    dependency_broker: object | None
    pipeline_promoter: Callable[[SelfCodingManager | None], None] | None


registry: BotRegistry | None = None
data_bot: DataBot | None = None
_context_builder: ContextBuilder | None = None
engine: SelfCodingEngine | None = None
_PipelineCls: "Type[ModelAutomationPipeline] | None" = None
pipeline: "ModelAutomationPipeline | None" = None
evolution_orchestrator: "EvolutionOrchestrator | None" = None
manager: SelfCodingManager | None = None
_CapitalManagerCls: "Type[CapitalManagementBot] | None" = None

_runtime_state: _RuntimeDependencies | None = None
_runtime_placeholder: _RuntimeDependencies | None = None
_runtime_initializing = False
_self_coding_configured = False


def _resolve_dependency_resolution_timeout(
    requested: float | None = None,
) -> float:
    """Return the maximum wait time for dependency resolution loops."""

    raw_timeout = os.getenv("MENACE_RUNTIME_DEPENDENCY_WAIT_SECS")
    if raw_timeout:
        try:
            resolved = float(raw_timeout)
        except ValueError:
            logger.warning(
                "Invalid MENACE_RUNTIME_DEPENDENCY_WAIT_SECS=%r; falling back to default", raw_timeout
            )
        else:
            if resolved < _DEPENDENCY_RESOLUTION_WAIT_FLOOR:
                logger.warning(
                    "MENACE_RUNTIME_DEPENDENCY_WAIT_SECS below floor; clamping to %ss",
                    _DEPENDENCY_RESOLUTION_WAIT_FLOOR,
                    extra={
                        "requested_timeout": resolved,
                        "timeout_floor": _DEPENDENCY_RESOLUTION_WAIT_FLOOR,
                    },
                )
            return max(resolved, _DEPENDENCY_RESOLUTION_WAIT_FLOOR)

    fallback_timeout = requested if requested is not None else _resolve_bootstrap_wait_timeout()
    if fallback_timeout is None:
        fallback_timeout = 15.0
    return max(float(fallback_timeout), _DEPENDENCY_RESOLUTION_WAIT_FLOOR)


def _resolve_dependency_max_wait(default: float, *, cap: float | None = None) -> float:
    """Resolve the hard cap for dependency wait/backoff loops."""

    cap_timeout = _DEPENDENCY_RESOLUTION_MAX_WAIT_DEFAULT if cap is None else cap

    raw_timeout = os.getenv("MENACE_RUNTIME_DEPENDENCY_MAX_WAIT_SECS")
    if raw_timeout:
        try:
            parsed = float(raw_timeout)
        except ValueError:
            logger.warning(
                "Invalid MENACE_RUNTIME_DEPENDENCY_MAX_WAIT_SECS=%r; using dependency timeout default",
                raw_timeout,
            )
        else:
            if parsed < _DEPENDENCY_RESOLUTION_WAIT_FLOOR:
                logger.warning(
                    "MENACE_RUNTIME_DEPENDENCY_MAX_WAIT_SECS below floor; clamping to %ss",
                    _DEPENDENCY_RESOLUTION_WAIT_FLOOR,
                    extra={
                        "requested_timeout": parsed,
                        "timeout_floor": _DEPENDENCY_RESOLUTION_WAIT_FLOOR,
                    },
                )
            return min(default, max(parsed, _DEPENDENCY_RESOLUTION_WAIT_FLOOR))

    return min(default, max(cap_timeout, _DEPENDENCY_RESOLUTION_WAIT_FLOOR))


def _ensure_bootstrap_ready(
    component: str, *, timeout: float | None = None, allow_degraded: bool = False
) -> bool:
    """Block until bootstrap readiness clears unless degraded mode is allowed."""

    if _vector_bootstrap_disabled():
        logger.critical(
            "%s bootstrap readiness bypassed because vector seeding is disabled; "
            "continuing with embeddings stubbed/disabled",
            component,
            extra={
                "event": "bootstrap-vector-seeding-disabled",
                "skip_env": os.getenv(_VECTOR_BOOTSTRAP_SKIP_ENV),
                "strict_env": os.getenv(_VECTOR_SEEDING_STRICT_ENV),
            },
        )
        return False
    resolved_timeout = _bootstrap_gate_timeout(
        vector_heavy=True, fallback=timeout if timeout is not None else 180.0
    )
    overall_budget = min(max(resolved_timeout, 120.0), 180.0)
    initial_timeout = min(resolved_timeout, max(overall_budget - 30.0, 30.0))
    start = time.monotonic()
    try:
        _BOOTSTRAP_READINESS.await_ready(timeout=initial_timeout)
        return True
    except TimeoutError as exc:  # pragma: no cover - defensive path
        timeout_exc = exc
        logger.warning(
            "%s bootstrap readiness timed out after %.1fs; waiting for fallback recovery",
            component,
            initial_timeout,
            extra={
                "event": "bootstrap-readiness-timeout",
                "timeout": initial_timeout,
                "budget": overall_budget,
            },
        )

    poll_interval = getattr(_BOOTSTRAP_READINESS, "poll_interval", 0.5)
    deadline = start + overall_budget
    while time.monotonic() < deadline:
        probe = _BOOTSTRAP_READINESS.probe()
        embedder_ready, embedder_mode = probe_embedding_service(readiness_loop=True)
        if probe.ready and embedder_ready:
            elapsed = time.monotonic() - start
            if embedder_mode.startswith("local"):
                logger.info(
                    "%s readiness recovered via local embedding fallback after %.1fs",
                    component,
                    elapsed,
                    extra={
                        "event": "bootstrap-embedder-local-fallback-ready",
                        "mode": embedder_mode,
                        "elapsed": elapsed,
                    },
                )
            else:
                logger.info(
                    "%s readiness recovered after %.1fs",
                    component,
                    elapsed,
                    extra={
                        "event": "bootstrap-readiness-recovered",
                        "mode": embedder_mode,
                        "elapsed": elapsed,
                    },
                )
            return True
        time.sleep(poll_interval)

    message = (
        f"{component} unavailable until bootstrap readiness clears: "
        f"{_BOOTSTRAP_READINESS.describe()}"
    )
    if allow_degraded:
        logger.warning(
            message,
            extra={"event": "research-aggregator-bootstrap-degraded"},
        )
        return False
    raise RuntimeError(message) from timeout_exc


# Eagerly advertise the bootstrap placeholder as soon as the module loads so
# downstream imports reuse the shared sentinel before instantiating helpers.
_bootstrap_ready = _ensure_bootstrap_ready(
    "ResearchAggregatorBot bootstrap placeholder",
    timeout=_BOOTSTRAP_GATE_TIMEOUT,
    allow_degraded=True,
)
_active_pipeline, _active_manager = get_active_bootstrap_pipeline()
_broker_owner_active = bool(getattr(_bootstrap_dependency_broker(), "active_owner", False))
if _bootstrap_ready or _active_pipeline is not None or _active_manager is not None or _broker_owner_active:
    if not _broker_owner_active and _active_pipeline is None and _active_manager is None:
        logger.warning(
            "Eager ResearchAggregatorBot bootstrap running in degraded mode; broker owner inactive",
            extra={
                "event": "research-aggregator-bootstrap-degraded",
                "broker_owner": _broker_owner_active,
            },
        )
    _bootstrap_placeholders(allow_degraded=True)
else:
    logger.info(
        "Skipping ResearchAggregatorBot placeholder bootstrap; no active broker owner or pipeline detected",
        extra={"event": "research-aggregator-broker-owner-missing", "broker_owner": _broker_owner_active},
    )


def _resolve_pipeline_cls() -> "Type[ModelAutomationPipeline]":
    """Return the concrete :class:`ModelAutomationPipeline` implementation."""

    module_candidates: tuple[str, ...] = (
        "menace_sandbox.model_automation_pipeline",
        ".model_automation_pipeline",
        "model_automation_pipeline",
    )
    last_error: Exception | None = None
    invalid_class_error = "get_pipeline_class returned an invalid pipeline handle from "

    def _resolve_via_loader(module: ModuleType) -> type | None:
        nonlocal last_error
        loader = getattr(module, "get_pipeline_class", None)
        if not callable(loader):
            loader = getattr(module, "_load_pipeline_cls", None)
        if callable(loader):
            try:
                pipeline_candidate = loader()
            except Exception as exc:
                last_error = ImportError(
                    f"ModelAutomationPipeline loader failed in {module.__name__}"
                ) from exc
                return None
            if isinstance(pipeline_candidate, type):
                return pipeline_candidate
            last_error = ImportError(f"{invalid_class_error}{module.__name__}")
        return None

    def _resolve_from_module(module: ModuleType) -> type | None:
        nonlocal last_error
        pipeline_candidate = getattr(module, "ModelAutomationPipeline", None)
        if isinstance(pipeline_candidate, type):
            return pipeline_candidate
        if pipeline_candidate is not None and not isinstance(pipeline_candidate, type):
            pipeline_cls = _resolve_via_loader(module)
            if pipeline_cls is not None:
                return pipeline_cls
            proxy_resolver = getattr(pipeline_candidate, "_resolve", None)
            if callable(proxy_resolver):
                try:
                    pipeline_candidate = proxy_resolver()
                except Exception as exc:
                    last_error = exc
                else:
                    if isinstance(pipeline_candidate, type):
                        return pipeline_candidate
                    last_error = ImportError(f"{invalid_class_error}{module.__name__}")
        return None

    for dotted in module_candidates:
        try:
            module = (
                importlib.import_module(dotted, __package__)
                if dotted.startswith(".")
                else importlib.import_module(dotted)
            )
        except ModuleNotFoundError as exc:
            last_error = exc
            continue
        pipeline_cls = _resolve_via_loader(module)
        if pipeline_cls is not None:
            return pipeline_cls  # type: ignore[return-value]
        pipeline_cls = _resolve_from_module(module)
        if pipeline_cls is not None:
            return pipeline_cls  # type: ignore[return-value]
        try:
            module = importlib.reload(module)
        except Exception:
            continue
        pipeline_cls = _resolve_from_module(module)
        if pipeline_cls is not None:
            return pipeline_cls  # type: ignore[return-value]
    raise ImportError(
        "ModelAutomationPipeline is unavailable; ensure menace_sandbox is fully initialised"
    ) from last_error


def _get_capital_manager_class() -> "Type[CapitalManagementBot]":
    """Import and return :class:`CapitalManagementBot` lazily."""

    global _CapitalManagerCls
    if _CapitalManagerCls is not None:
        return _CapitalManagerCls

    module_candidates: tuple[str, ...] = (
        "menace_sandbox.capital_management_bot",
        ".capital_management_bot",
        "capital_management_bot",
    )
    last_error: Exception | None = None
    for dotted in module_candidates:
        try:
            module = (
                importlib.import_module(dotted, __package__)
                if dotted.startswith(".")
                else importlib.import_module(dotted)
            )
        except ModuleNotFoundError as exc:
            last_error = exc
            continue
        capital_cls = getattr(module, "CapitalManagementBot", None)
        if isinstance(capital_cls, type):
            typed_cls = cast("Type[CapitalManagementBot]", capital_cls)
            _CapitalManagerCls = typed_cls
            return typed_cls
    raise ImportError(
        "CapitalManagementBot is unavailable; ensure menace_sandbox is fully initialised"
    ) from last_error


def _ensure_self_coding_decorated(deps: _RuntimeDependencies) -> None:
    """Apply :func:`self_coding_managed` once runtime helpers exist."""

    global _self_coding_configured
    if _self_coding_configured:
        return
    bot_cls = globals().get("ResearchAggregatorBot")
    if bot_cls is None:
        return
    decorated = self_coding_managed(
        bot_registry=deps.registry,
        data_bot=deps.data_bot,
        manager=deps.manager,
    )(bot_cls)
    if decorated is not bot_cls:
        globals()["ResearchAggregatorBot"] = decorated
    _self_coding_configured = True


_bootstrap_pipeline_cache: dict[object, tuple["ModelAutomationPipeline", Callable[[SelfCodingManager | None], None], SelfCodingManager | None]] = {}


def _active_bootstrap_promoter() -> Callable[[Any], None] | None:
    callbacks = getattr(_BOOTSTRAP_STATE, "helper_promotion_callbacks", None)
    if callbacks:
        return callbacks[-1]
    return None


def _ensure_runtime_dependencies(
    *,
    bootstrap_owner: object | None = None,
    pipeline_override: "ModelAutomationPipeline | None" = None,
    manager_override: SelfCodingManager | None = None,
    promote_pipeline: Callable[[SelfCodingManager | None], None] | None = None,
    bootstrap_state: Mapping[str, object] | None = None,
) -> _RuntimeDependencies:
    """Instantiate heavy runtime helpers on demand.

    During bootstrap the pipeline and manager must be injected (directly or via
    the dependency broker).  The bootstrap path will refuse to spin up a fresh
    pipeline and instead reuses any advertised placeholder/promise, raising an
    explicit error when nothing has been provided.
    """

    # Guard: bootstrap orchestration must flow through the shared readiness
    # snapshot so module constructors never recurse into bootstrap again.
    state = bootstrap_state or bootstrap_state_snapshot()
    if not state.get("ready") and not state.get("in_progress"):
        ensure_bootstrapped()
    placeholder_pipeline, placeholder_manager, placeholder_broker = (
        _bootstrap_placeholders(allow_degraded=False)
    )
    placeholder_broker_owner = bool(getattr(placeholder_broker, "active_owner", False))
    if not placeholder_broker_owner:
        if _looks_like_pipeline_candidate(placeholder_pipeline) or placeholder_manager:
            logger.error(
                "Bootstrap dependency broker owner inactive; reusing advertised ResearchAggregatorBot placeholder",
                extra={
                    "event": "research-aggregator-placeholder-owner-missing",
                    "broker_owner": placeholder_broker_owner,
                },
            )
        else:
            raise RuntimeError(
                "Bootstrap dependency broker owner not active; aborting ResearchAggregatorBot initialisation"
            )
    global registry
    global data_bot
    global _context_builder
    global engine
    global _PipelineCls
    global pipeline
    global evolution_orchestrator
    global manager
    global _runtime_state
    global _runtime_placeholder
    global _runtime_initializing

    pipeline_hint = pipeline_override
    promote_explicit = promote_pipeline is not None
    guard_pipeline, guard_manager = get_active_bootstrap_pipeline()
    if pipeline_hint is None and _looks_like_pipeline_candidate(guard_pipeline):
        pipeline_hint = guard_pipeline
        if manager_override is None:
            manager_override = guard_manager

    bootstrap_context = _current_bootstrap_context()
    if pipeline_hint is None and bootstrap_context is not None:
        context_pipeline = getattr(bootstrap_context, "pipeline", None)
        if _looks_like_pipeline_candidate(context_pipeline):
            pipeline_hint = context_pipeline
        context_manager = getattr(bootstrap_context, "manager", None)
        if manager_override is None and context_manager is not None:
            manager_override = context_manager

    manager_pipeline = getattr(manager_override, "pipeline", None)
    if pipeline_hint is None and _looks_like_pipeline_candidate(manager_pipeline):
        pipeline_hint = manager_pipeline

    if pipeline_hint is None and _looks_like_pipeline_candidate(placeholder_pipeline):
        pipeline_hint = placeholder_pipeline
    if manager_override is None and placeholder_manager is not None:
        manager_override = placeholder_manager

    sentinel_active = False
    try:
        sentinel_active = _using_bootstrap_sentinel(manager_override)
    except Exception:  # pragma: no cover - sentinel detection best effort
        sentinel_active = False

    if pipeline_hint is None and sentinel_active and pipeline is not None:
        pipeline_hint = pipeline

    owner = bootstrap_owner
    if not sentinel_active and pipeline_hint is None:
        try:
            if owner is None:
                from .coding_bot_interface import get_structural_bootstrap_owner

                owner = get_structural_bootstrap_owner()
        except Exception:  # pragma: no cover - bootstrap owner best effort
            owner = bootstrap_owner

    if owner is not None and pipeline_hint is None:
        cached = _bootstrap_pipeline_cache.get(owner)
        if cached is not None:
            pipeline_hint, promote_pipeline, manager_override = cached
            try:
                dependency_broker.advertise(
                    pipeline=pipeline_hint,
                    sentinel=manager_override,
                    owner=owner is not False,
                )
            except Exception:  # pragma: no cover - broker propagation best effort
                logger.debug(
                    "Failed to propagate cached bootstrap broker state", exc_info=True
                )

    pipeline_override = pipeline_hint

    dependency_broker = placeholder_broker or _bootstrap_dependency_broker()
    if dependency_broker is None:
        raise RuntimeError(
            "Bootstrap dependency broker unavailable; cannot initialise ResearchAggregatorBot"
        )

    broker_active_pipeline = getattr(dependency_broker, "active_pipeline", None)
    broker_active_sentinel = getattr(dependency_broker, "active_sentinel", None)
    broker_active_owner = bool(getattr(dependency_broker, "active_owner", False))
    broker_placeholder_seeded = False
    if _runtime_state is None:
        try:
            placeholder_pipeline, placeholder_manager = advertise_bootstrap_placeholder(
                dependency_broker=dependency_broker,
                pipeline=pipeline_override,
                manager=manager_override
                if manager_override is not None
                else getattr(bootstrap_context, "manager", None),
                owner=owner is not False,
            )
            broker_placeholder_seeded = True
        except Exception:  # pragma: no cover - best effort broker seeding
            logger.warning(
                "Failed to advertise early bootstrap placeholder; bootstrap helpers may re-initialise",
                exc_info=True,
            )

    if not broker_placeholder_seeded and not _looks_like_pipeline_candidate(
        broker_active_pipeline
    ):
        warnings.warn(
            "No bootstrap dependency broker placeholder available; refusing implicit pipeline construction",
            RuntimeWarning,
        )

    if not broker_active_owner:
        if _looks_like_pipeline_candidate(placeholder_pipeline):
            pipeline = pipeline or placeholder_pipeline
            pipeline_hint = pipeline_hint or placeholder_pipeline
            if manager_override is None:
                manager_override = placeholder_manager
            logger.error(
                "Bootstrap dependency broker owner inactive; reusing placeholder pipeline for ResearchAggregatorBot",
                extra={
                    "event": "research-aggregator-broker-owner-missing",
                    "has_placeholder": True,
                },
            )
        else:
            logger.error(
                "Bootstrap dependency broker owner inactive with no placeholder pipeline; refusing to claim new pipeline",
                extra={
                    "event": "research-aggregator-broker-missing-owner",
                    "has_placeholder": False,
                },
            )
            raise RuntimeError(
                "Bootstrap dependency broker owner not active; refusing to construct ResearchAggregatorBot pipeline"
            )

    if _runtime_state is not None:
        _ensure_self_coding_decorated(_runtime_state)
        return _runtime_state

    if _runtime_initializing:
        if _runtime_placeholder is not None:
            return _runtime_placeholder

        sentinel_seed = (
            manager_override
            if manager_override is not None
            else placeholder_manager
            if placeholder_manager is not None
            else broker_active_sentinel
        )
        pipeline_seed = (
            pipeline_override
            or pipeline
            or pipeline_hint
            or placeholder_pipeline
            or broker_active_pipeline
        )

        reg = (
            registry
            if registry is not None
            else BotRegistry(bootstrap=bool(sentinel_seed))
        )
        dbot = (
            data_bot
            if data_bot is not None
            else DataBot(start_server=False, bootstrap=bool(sentinel_seed))
        )
        ctx_builder = _context_builder
        if ctx_builder is None:
            ctx_builder = create_context_builder(bootstrap_safe=bool(sentinel_seed))
        eng = engine if engine is not None else SelfCodingEngine(
            CodeDB(),
            GPTMemoryManager(),
            context_builder=ctx_builder,
            pipeline=pipeline_seed,
            data_bot=dbot,
        )

        _runtime_placeholder = _RuntimeDependencies(
            registry=reg,
            data_bot=dbot,
            context_builder=ctx_builder,
            engine=eng,
            pipeline=pipeline_override if pipeline_override is not None else pipeline,
            evolution_orchestrator=evolution_orchestrator,
            manager=manager_override if manager_override is not None else manager,
            dependency_broker=dependency_broker,
            pipeline_promoter=None,
        )
        return _runtime_placeholder

    sentinel_seed = (
        manager_override
        if manager_override is not None
        else placeholder_manager
        if placeholder_manager is not None
        else broker_active_sentinel
    )
    pipeline_seed = (
        pipeline_override
        or pipeline
        or pipeline_hint
        or placeholder_pipeline
        or broker_active_pipeline
    )

    reg = (
        registry if registry is not None else BotRegistry(bootstrap=bool(sentinel_seed))
    )
    dbot = (
        data_bot
        if data_bot is not None
        else DataBot(start_server=False, bootstrap=bool(sentinel_seed))
    )

    ctx_builder = _context_builder
    if ctx_builder is None:
        ctx_builder = create_context_builder(bootstrap_safe=bool(sentinel_seed))

    eng = engine if engine is not None else SelfCodingEngine(
        CodeDB(),
        GPTMemoryManager(),
        context_builder=ctx_builder,
        pipeline=pipeline_seed,
        data_bot=dbot,
    )

    _runtime_initializing = True
    _runtime_placeholder = _RuntimeDependencies(
        registry=reg,
        data_bot=dbot,
        context_builder=ctx_builder,
        engine=eng,
        pipeline=pipeline_override if pipeline_override is not None else pipeline,
        evolution_orchestrator=evolution_orchestrator,
        manager=manager_override if manager_override is not None else manager,
        dependency_broker=dependency_broker,
        pipeline_promoter=None,
    )

    orchestrator = evolution_orchestrator
    mgr = manager_override if manager_override is not None else manager

    success = False

    def _is_bootstrap_active() -> bool:
        heartbeat = read_bootstrap_heartbeat()
        return bool(
            getattr(_BOOTSTRAP_STATE, "depth", 0)
            or guard_promise is not None
            or _current_bootstrap_context() is not None
            or heartbeat
            or broker_active_owner
            or broker_active_pipeline is not None
            or broker_active_sentinel is not None
        )

    try:
        pipeline_cls = _PipelineCls if _PipelineCls is not None else _resolve_pipeline_cls()
        promote_pipeline = promote_pipeline or _active_bootstrap_promoter()
        owner_guard = getattr(_BOOTSTRAP_STATE, "active_bootstrap_guard", None)
        guard_promise = _peek_owner_promise(owner_guard) if owner_guard is not None else None
        broker_pipeline, broker_manager = dependency_broker.resolve()
        if pipeline_hint is None and _looks_like_pipeline_candidate(broker_pipeline):
            pipeline_hint = broker_pipeline
            if manager_override is None and manager is None:
                manager_override = broker_manager
        if manager_override is None and manager is None and broker_manager is not None:
            manager_override = broker_manager

        prepared_pipeline = False

        pipe = None
        if pipeline_override is not None:
            pipe = pipeline_override
        elif pipeline is not None:
            pipe = pipeline
        elif _looks_like_pipeline_candidate(pipeline_hint):
            pipe = pipeline_hint
        wait_timeout = _resolve_bootstrap_wait_timeout()
        resolution_timeout = _resolve_dependency_resolution_timeout(wait_timeout)
        max_resolution_wait = _resolve_dependency_max_wait(
            resolution_timeout, cap=_DEPENDENCY_RESOLUTION_MAX_WAIT_DEFAULT
        )
        resolution_deadline = time.perf_counter() + max_resolution_wait
        wait_start = time.perf_counter()
        backoff = 0.01

        def _broker_snapshot() -> dict[str, object]:
            snapshot = {
                "active_owner": broker_active_owner,
                "active_pipeline": bool(broker_active_pipeline),
                "active_sentinel": bool(broker_active_sentinel),
                "broker_placeholder_seeded": broker_placeholder_seeded,
                "pipeline_hint": _looks_like_pipeline_candidate(pipeline_hint),
                "placeholder_pipeline": _looks_like_pipeline_candidate(placeholder_pipeline),
                "manager_override": manager_override is not None,
            }
            try:
                snapshot["bootstrap_heartbeat"] = bool(read_bootstrap_heartbeat())
            except Exception:  # pragma: no cover - heartbeat best effort
                snapshot["bootstrap_heartbeat"] = None
            return snapshot

        def _enforce_resolution_deadline(reason: str) -> None:
            if time.perf_counter() >= resolution_deadline:
                _raise_resolution_deadline(reason)

        def _raise_resolution_deadline(reason: str) -> None:
            waited = time.perf_counter() - wait_start
            snapshot = _broker_snapshot()
            message = (
                "Active bootstrap requires an injected ModelAutomationPipeline "
                f"or manager; deadline reached after waiting {round(waited, 3)}s "
                f"(cap {round(max_resolution_wait, 3)}s) while {reason}. "
                f"Broker snapshot: {snapshot}"
            )
            logger.error(
                message,
                extra={
                    "event": "research-aggregator-runtime-dependency-deadline",
                    "bot": "ResearchAggregatorBot",
                    "broker_state": snapshot,
                },
            )
            raise RuntimeError(message)
        while pipe is None and _is_bootstrap_active():
            remaining_resolution = resolution_deadline - time.perf_counter()
            if remaining_resolution <= 0:
                _raise_resolution_deadline("waiting for broker advertisement")
            broker_pipeline, broker_manager = dependency_broker.resolve()
            if manager_override is None and manager is None and broker_manager is not None:
                manager_override = broker_manager
            if _looks_like_pipeline_candidate(broker_pipeline):
                pipe = broker_pipeline
                if (
                    manager_override is None
                    and manager is None
                    and broker_manager is not None
                ):
                    manager_override = broker_manager
                break

            bootstrap_context = _current_bootstrap_context()
            if bootstrap_context is not None:
                context_pipeline = getattr(bootstrap_context, "pipeline", None)
                if _looks_like_pipeline_candidate(context_pipeline):
                    pipe = context_pipeline
                    if manager_override is None:
                        context_manager = getattr(bootstrap_context, "manager", None)
                        if context_manager is not None:
                            manager_override = context_manager
                    break

            guard_pipeline, guard_manager = get_active_bootstrap_pipeline()
            if _looks_like_pipeline_candidate(guard_pipeline):
                pipe = guard_pipeline
                if manager_override is None:
                    manager_override = guard_manager
                guard_promoter = _active_bootstrap_promoter()
                if guard_promoter is not None:
                    promote_pipeline = guard_promoter
                break

            try:
                active_promise = _GLOBAL_BOOTSTRAP_COORDINATOR.peek_active()
            except Exception:
                active_promise = None
            if active_promise is not None:
                if getattr(active_promise, "done", False):
                    pipe_promised, promote_promised = active_promise.wait()
                    pipe = pipe_promised
                    if promote_pipeline is None:
                        promote_pipeline = promote_promised
                    if manager_override is None:
                        manager_override = getattr(pipe_promised, "manager", broker_manager)
                    break
                event = getattr(active_promise, "_event", None)
                if event is not None:
                    _enforce_resolution_deadline("waiting for broker advertisement")
                    event.wait(
                        timeout=max(
                            _DEPENDENCY_RESOLUTION_WAIT_FLOOR,
                            min(backoff, remaining_resolution),
                        )
                    )
                    _enforce_resolution_deadline("waiting for broker advertisement")

            remaining_resolution = resolution_deadline - time.perf_counter()
            if remaining_resolution <= 0:
                _raise_resolution_deadline("waiting for broker advertisement")
            time.sleep(min(backoff, max(remaining_resolution, _DEPENDENCY_RESOLUTION_WAIT_FLOOR)))
            backoff = min(backoff * 2, 0.25)

        if pipe is None and time.perf_counter() >= resolution_deadline:
            _raise_resolution_deadline("waiting for broker advertisement")

        if pipe is None:
            bootstrap_active = _is_bootstrap_active()
            if bootstrap_active:
                broker_expectation = bool(
                    broker_active_owner
                    or broker_active_pipeline is not None
                    or broker_active_sentinel is not None
                )
                broker_pipeline, broker_manager = dependency_broker.resolve()
                if (
                    pipe is None
                    and not _looks_like_pipeline_candidate(broker_pipeline)
                    and _looks_like_pipeline_candidate(broker_active_pipeline)
                ):
                    broker_pipeline = broker_active_pipeline
                    broker_manager = broker_manager or broker_active_sentinel
                if _looks_like_pipeline_candidate(broker_pipeline):
                    pipe = broker_pipeline
                    if manager_override is None and manager is None:
                        manager_override = broker_manager
                elif broker_expectation:
                    wait_deadline = min(
                        resolution_deadline,
                        time.perf_counter()
                        + (
                            resolution_timeout
                            if wait_timeout is None
                            else max(wait_timeout, _DEPENDENCY_RESOLUTION_WAIT_FLOOR)
                        ),
                    )
                    backoff = 0.01
                    while pipe is None and (
                        wait_deadline is None or time.perf_counter() < wait_deadline
                    ):
                        broker_pipeline, broker_manager = dependency_broker.resolve()
                        broker_active_pipeline = getattr(
                            dependency_broker, "active_pipeline", broker_active_pipeline
                        )
                        broker_active_sentinel = getattr(
                            dependency_broker, "active_sentinel", broker_active_sentinel
                        )
                        broker_active_owner = bool(
                            getattr(dependency_broker, "active_owner", broker_active_owner)
                        )
                        if _looks_like_pipeline_candidate(broker_pipeline):
                            pipe = broker_pipeline
                            if manager_override is None and manager is None:
                                manager_override = broker_manager
                            break
                        if _looks_like_pipeline_candidate(broker_active_pipeline):
                            pipe = broker_active_pipeline
                            if manager_override is None:
                                manager_override = (
                                    broker_active_sentinel or broker_manager
                                )
                            break
                        if not broker_active_owner and not broker_active_pipeline:
                            break
                        _enforce_resolution_deadline(
                            "waiting for broker placeholder reuse"
                        )
                        time.sleep(backoff)
                        backoff = min(backoff * 2, 0.25)
                if pipe is None and broker_active_pipeline is not None:
                    pipe = broker_active_pipeline
                    if manager_override is None:
                        manager_override = broker_active_sentinel
                if pipe is None and placeholder_pipeline is not None:
                    pipe = placeholder_pipeline
                    if manager_override is None and placeholder_manager is not None:
                        manager_override = placeholder_manager

                if pipe is None:
                    active_promise = None
                    try:
                        active_promise = _GLOBAL_BOOTSTRAP_COORDINATOR.peek_active()
                    except Exception:
                        active_promise = None

                    if active_promise is not None:
                        event = getattr(active_promise, "_event", None)
                        if event is not None:
                            remaining = resolution_deadline - time.perf_counter()
                            if remaining > 0:
                                event.wait(
                                    timeout=max(
                                        _DEPENDENCY_RESOLUTION_WAIT_FLOOR,
                                        min(resolution_timeout, remaining),
                                    )
                                )
                                if resolution_deadline is not None and time.perf_counter() >= resolution_deadline:
                                    pipe = None
                        if getattr(active_promise, "done", False):
                            pipe_promised, promote_promised = active_promise.wait()
                            pipe = pipe_promised
                            if promote_pipeline is None:
                                promote_pipeline = promote_promised
                            if manager_override is None:
                                manager_override = getattr(
                                    pipe_promised, "manager", broker_manager
                                )

                if pipe is None:
                    _raise_resolution_deadline("waiting for broker placeholder reuse")
            if pipe is None:
                broker_pipeline, broker_manager = dependency_broker.resolve()

                bootstrap_heartbeat = False
                try:
                    bootstrap_heartbeat = bool(read_bootstrap_heartbeat())
                except Exception:  # pragma: no cover - heartbeat best effort
                    bootstrap_heartbeat = False

                try:
                    active_promise = _GLOBAL_BOOTSTRAP_COORDINATOR.peek_active()
                except Exception:  # pragma: no cover - promise lookup best effort
                    active_promise = None

                promise_reused = False
                if active_promise is not None:
                    if getattr(active_promise, "done", False):
                        pipe_promised, promote_promised = active_promise.wait()
                        pipe = pipe_promised
                        promise_reused = True
                        if promote_pipeline is None:
                            promote_pipeline = promote_promised
                        if manager_override is None:
                            manager_override = getattr(
                                pipe_promised, "manager", broker_manager
                            )

                recursion_detected = bool(
                    bootstrap_heartbeat
                    or active_promise is not None
                    or broker_active_owner
                    or broker_active_pipeline is not None
                    or broker_active_sentinel is not None
                    or placeholder_pipeline is not None
                )

                if recursion_detected and pipe is None:
                    if _looks_like_pipeline_candidate(broker_active_pipeline):
                        pipe = broker_active_pipeline
                        if manager_override is None:
                            manager_override = broker_active_sentinel
                    elif _looks_like_pipeline_candidate(broker_pipeline):
                        pipe = broker_pipeline
                        if manager_override is None:
                            manager_override = broker_manager
                    elif _looks_like_pipeline_candidate(placeholder_pipeline):
                        pipe = placeholder_pipeline
                        if manager_override is None:
                            manager_override = placeholder_manager

                if recursion_detected and pipe is None and not promise_reused:
                    waited = time.perf_counter() - wait_start
                    message = (
                        "Recursive bootstrap detected while waiting for broker state; "
                        "refusing to start a new pipeline after "
                        f"{round(waited, 3)}s. "
                        f"Broker state: {{'active_owner': {broker_active_owner}, "
                        f"'active_pipeline': {bool(broker_active_pipeline)}, "
                        f"'active_sentinel': {bool(broker_active_sentinel)}, "
                        f"'bootstrap_heartbeat': {bootstrap_heartbeat}}}"
                    )
                    logger.error(message)
                    raise RuntimeError(message)

                if pipe is None:
                    if not broker_placeholder_seeded and not _looks_like_pipeline_candidate(
                        broker_active_pipeline
                    ):
                        message = (
                            "Bootstrap dependency broker missing placeholder; refusing to "
                            "construct a new pipeline for ResearchAggregatorBot"
                        )
                        logger.error(message)
                        raise RuntimeError(message)

                    pipeline_manager_hint = (
                        manager_override
                        if manager_override is not None
                        else broker_manager
                        if broker_manager is not None
                        else manager
                        if manager is not None
                        else placeholder_manager
                    )
                    pipe, promoted, resolved_manager, prepared = claim_bootstrap_dependency_entry(
                        dependency_broker=dependency_broker,
                        pipeline=pipeline_override or broker_pipeline,
                        manager=pipeline_manager_hint,
                        owner=owner is not False,
                        pipeline_cls=pipeline_cls,
                        context_builder=ctx_builder,
                        bot_registry=reg,
                        data_bot=dbot,
                        manager_override=manager_override,
                    )
                    manager_override = resolved_manager or manager_override
                    prepared_pipeline = prepared_pipeline or prepared
                    placeholder_pipeline = placeholder_pipeline or pipe
                    placeholder_manager = placeholder_manager or resolved_manager
                    if promoted is not None and not promote_explicit:
                        promote_pipeline = promoted

        if pipe is None:
            raise RuntimeError(
                "ModelAutomationPipeline must be provided during ResearchAggregatorBot initialisation"
            )

        if promote_pipeline is None:
            promote_pipeline = _active_bootstrap_promoter() or (lambda *_args: None)

        def _advertise_dependency_state(real_manager: SelfCodingManager | None = None) -> None:
            try:
                pipeline_candidate = pipe if pipe is not None else placeholder_pipeline
                sentinel_candidate = (
                    real_manager
                    if real_manager is not None
                    else manager_override if manager_override is not None else mgr
                )
                if sentinel_candidate is None:
                    sentinel_candidate = placeholder_manager
                dependency_broker.advertise(
                    pipeline=pipeline_candidate,
                    sentinel=sentinel_candidate,
                )
            except Exception:  # pragma: no cover - best effort broker update
                logger.debug(
                    "Failed to advertise bootstrap dependency broker state", exc_info=True
                )

        promote_target = promote_pipeline

        def _promote_with_broker(real_manager: SelfCodingManager | None) -> None:
            try:
                promote_target(real_manager)
            finally:
                _advertise_dependency_state(real_manager)

        promote_pipeline = _promote_with_broker

        _advertise_dependency_state()

        if owner is not None and owner not in _bootstrap_pipeline_cache:
            _bootstrap_pipeline_cache[owner] = (pipe, promote_pipeline, manager_override)

        try:
            orchestrator = (
                orchestrator
                if orchestrator is not None
                else get_orchestrator("ResearchAggregatorBot", dbot, eng)
            )
        except Exception as exc:  # pragma: no cover - fallback for degraded envs
            logger.warning(
                "Evolution orchestrator unavailable for ResearchAggregatorBot: %s", exc
            )
            orchestrator = None

        thresholds = get_thresholds("ResearchAggregatorBot")
        try:
            persist_sc_thresholds(
                "ResearchAggregatorBot",
                roi_drop=thresholds.roi_drop,
                error_increase=thresholds.error_increase,
                test_failure_increase=thresholds.test_failure_increase,
            )
        except Exception:  # pragma: no cover - best effort persistence
            logger.exception(
                "Failed to persist self-coding thresholds for ResearchAggregatorBot"
            )

        pipeline_placeholder = bool(getattr(pipe, "bootstrap_placeholder", False))
        try:
            if mgr is None and not pipeline_placeholder and _looks_like_pipeline_candidate(pipe):
                mgr = internalize_coding_bot(
                    "ResearchAggregatorBot",
                    eng,
                    pipe,
                    data_bot=dbot,
                    bot_registry=reg,
                    evolution_orchestrator=orchestrator,
                    threshold_service=ThresholdService(),
                    roi_threshold=thresholds.roi_drop,
                    error_threshold=thresholds.error_increase,
                    test_failure_threshold=thresholds.test_failure_increase,
                )
            elif mgr is None:
                mgr = manager_override or manager or getattr(pipe, "manager", None)
        except Exception as exc:  # pragma: no cover - fallback for degraded envs
            logger.warning(
                "Self-coding manager unavailable for ResearchAggregatorBot: %s", exc
            )
            mgr = None
        else:
            if promote_pipeline is None:
                promote_pipeline = _active_bootstrap_promoter() or (lambda *_args: None)
            promote_pipeline(mgr)

        if prepared_pipeline and callable(promote_pipeline):
            for target in (pipe, mgr):
                if target is None:
                    continue
                try:
                    setattr(target, "_pipeline_promoter", promote_pipeline)
                except Exception:  # pragma: no cover - best effort attachment
                    continue

        _advertise_dependency_state(mgr)

        _runtime_placeholder = _RuntimeDependencies(
            registry=reg,
            data_bot=dbot,
            context_builder=ctx_builder,
            engine=eng,
            pipeline=pipe,
            evolution_orchestrator=orchestrator,
            manager=mgr,
            dependency_broker=dependency_broker,
            pipeline_promoter=promote_pipeline,
        )

        registry = reg
        data_bot = dbot
        _context_builder = ctx_builder
        engine = eng
        _PipelineCls = pipeline_cls
        pipeline = pipe
        evolution_orchestrator = orchestrator
        manager = mgr

        _runtime_state = _RuntimeDependencies(
            registry=reg,
            data_bot=dbot,
            context_builder=ctx_builder,
            engine=eng,
            pipeline=pipe,
            evolution_orchestrator=orchestrator,
            manager=mgr,
            dependency_broker=dependency_broker,
            pipeline_promoter=promote_pipeline,
        )
        _runtime_placeholder = _runtime_state
        _ensure_self_coding_decorated(_runtime_state)
        success = True
        return _runtime_state

    finally:
        if not success:
            _runtime_placeholder = None
        _runtime_initializing = False


def _initialize_runtime(
    *,
    bootstrap_owner: object | None = None,
    pipeline_override: "ModelAutomationPipeline | None" = None,
    manager_override: SelfCodingManager | None = None,
    promote_pipeline: Callable[[SelfCodingManager | None], None] | None = None,
) -> _RuntimeDependencies:
    """Public wrapper for lazy runtime initialisation."""

    return _ensure_runtime_dependencies(
        bootstrap_owner=bootstrap_owner,
        pipeline_override=pipeline_override,
        manager_override=manager_override,
        promote_pipeline=promote_pipeline,
    )

class ResearchMemory:
    """Tiered memory with decay for storing research items."""

    def __init__(self) -> None:
        self.short: List[ResearchItem] = []
        self.medium: List[ResearchItem] = []
        self.long: List[ResearchItem] = []

    def add(self, item: ResearchItem, layer: str = "short") -> None:
        getattr(self, layer).append(item)

    def search(self, topic: str) -> List[ResearchItem]:
        topic = topic.lower()
        results: List[ResearchItem] = []
        for layer in [self.short, self.medium, self.long]:
            results.extend([i for i in layer if topic in i.topic.lower()])
        return results

    def decay(self, now: Optional[float] = None) -> None:
        now = now or time.time()
        self.short = [i for i in self.short if now - i.timestamp < 60]
        self.medium = [i for i in self.medium if now - i.timestamp < 3600]

    def archive(self, items: Iterable[ResearchItem]) -> None:
        for it in items:
            if it not in self.long:
                self.long.append(it)

class ResearchAggregatorBot:
    """Collects, refines and stores research with energy-based depth.

    When running under bootstrap, callers must provide the active
    ``ModelAutomationPipeline`` and/or ``SelfCodingManager`` (directly or via
    the dependency broker).  The bot will not construct a new pipeline while
    bootstrap coordination is in progress.
    """

    def __init__(
        self,
        requirements: Iterable[str],
        memory: Optional[ResearchMemory] = None,
        info_db: Optional[InfoDB] = None,
        enhancements_db: Optional[EnhancementDB] = None,
        enhancement_bot: Optional[ChatGPTEnhancementBot] = None,
        prediction_bot: Optional[ChatGPTPredictionBot] = None,
        text_bot: Optional[TextResearchBot] = None,
        video_bot: Optional[VideoResearchBot] = None,
        chatgpt_bot: Optional[ChatGPTResearchBot] = None,
        capital_manager: Optional[CapitalManagementBot] = None,
        db_router: Optional[DBRouter] = None,
        enhancement_interval: float = 300.0,
        cache_ttl: float = 3600.0,
        *,
        manager: SelfCodingManager | None = None,
        pipeline: "ModelAutomationPipeline | None" = None,
        pipeline_promoter: Callable[[SelfCodingManager | None], None] | None = None,
        bootstrap_owner: object | None = None,
        context_builder: ContextBuilder | None = None,
        bootstrap: bool = False,
        defer_migrations_until_ready: bool = False,
    ) -> None:
        _ensure_bootstrap_ready("ResearchAggregatorBot")
        init_start = time.perf_counter()
        deps = _ensure_runtime_dependencies(
            bootstrap_owner=bootstrap_owner,
            pipeline_override=pipeline,
            manager_override=manager,
            promote_pipeline=pipeline_promoter,
        )
        _ensure_self_coding_decorated(deps)
        builder = context_builder or deps.context_builder
        if builder is None:
            raise ValueError("ContextBuilder is required")
        mgr = manager or deps.manager
        self.manager = mgr
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = deps.data_bot
        self.requirements = list(requirements)
        self.memory = memory or ResearchMemory()
        bootstrap_active = bootstrap or bool(getattr(mgr, "bootstrap_mode", False))
        migration_skipped = bootstrap_active
        migration_deferred = defer_migrations_until_ready or bootstrap_active
        self.info_db = info_db or InfoDB(
            run_migrations=not migration_deferred,
            apply_nonessential_migrations=not migration_skipped,
            batch_migrations=True,
            bootstrap_mode=bootstrap_active,
            migration_timeout=0.0 if bootstrap_active else None,
            non_blocking_migrations=bootstrap_active,
        )
        self.db_router = db_router or DBRouter(info_db=self.info_db)
        self.enh_db = enhancements_db or EnhancementDB()
        self.enhancement_bot = enhancement_bot
        self.prediction_bot = prediction_bot
        self.text_bot = text_bot
        self.video_bot = video_bot
        self.chatgpt_bot = chatgpt_bot
        if self.chatgpt_bot and getattr(self.chatgpt_bot, "send_callback", None) is None:
            self.chatgpt_bot.send_callback = self.receive_chatgpt
        if capital_manager is None:
            capital_cls = _get_capital_manager_class()
            capital_manager = capital_cls()
        self.capital_manager = capital_manager
        self.enhancement_interval = enhancement_interval
        self._last_enhancement_time = 0.0
        self.sources_queried: List[str] = []
        self.cache_ttl = cache_ttl
        self.cache: dict[str, tuple[float, List[ResearchItem]]] = {}
        try:
            builder.refresh_db_weights()
        except Exception:
            logger.exception("Failed to initialise ContextBuilder")
            raise
        if migration_deferred:
            migration_start = time.perf_counter()
            try:
                self.info_db.apply_migrations(
                    apply_nonessential=not migration_skipped,
                    batch=True,
                )
            except Exception:
                logger.exception("Deferred migration pass failed during bootstrap")
                raise
            else:
                logger.info(
                    "ResearchAggregatorBot migrations applied after pipeline ready: %.3fs",
                    time.perf_counter() - migration_start,
                )
        elapsed = time.perf_counter() - init_start
        if bootstrap:
            logger.info(
                "ResearchAggregatorBot bootstrap initialisation elapsed=%.3fs (deferred_migrations=%s)",
                elapsed,
                migration_deferred,
            )
            if elapsed > 30:
                logger.warning(
                    "ResearchAggregatorBot bootstrap path exceeded 30s deadline: %.3fs",
                    elapsed,
                )
        self.context_builder = builder

    # ------------------------------------------------------------------
    def _increment_enh_count(self, model_id: int) -> None:
        """Increment enhancement counter in models.db if possible."""
        try:
            with get_connection(DB_PATH) as conn:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()]
                if "enhancement_count" not in cols:
                    conn.execute("ALTER TABLE models ADD COLUMN enhancement_count INTEGER DEFAULT 0")
                cur = conn.execute(
                    "SELECT enhancement_count FROM models WHERE id=?",
                    (model_id,),
                )
                row = cur.fetchone()
                count = int(row[0]) if row and row[0] is not None else 0
                conn.execute(
                    "UPDATE models SET enhancement_count=? WHERE id=?",
                    (count + 1, model_id),
                )
        except Exception as exc:
            logger.exception(
                "Failed to increment enhancement count for model %s: %s",
                model_id,
                exc,
            )

    def _info_ratio(self, energy: int) -> int:
        try:
            ratio = self.capital_manager.info_ratio(float(energy))
        except Exception:
            ratio = float(energy)
        return max(1, int(round(ratio)))

    def _compressed_context(self, query: str) -> str:
        try:
            ctx = str(self.context_builder.build(query))
        except Exception as exc:
            logger.exception("Context build failed for %s: %s", query, exc)
            ctx = ""
        return compress_snippets({"snippet": ctx}).get("snippet", ctx)

    def _query_local(self, topic: str) -> List[ResearchItem]:
        items = []
        lower = topic.lower()
        for it in self.info_db.search(topic):
            if it.title.lower() == lower or lower in [t.lower() for t in it.tags]:
                items.append(it)
        for enh in self.enh_db.fetch():
            text = f"{enh.idea} {enh.rationale}".lower()
            if lower in text:
                items.append(
                    ResearchItem(
                        topic=topic,
                        content=f"{enh.idea}: {enh.rationale}",
                        timestamp=time.time(),
                        categories=["enhancement"],
                    )
                )
        return items


if _runtime_state is not None:
    _ensure_self_coding_decorated(_runtime_state)

    def _missing_data_types(self, items: Iterable[ResearchItem], topic: str) -> List[str]:
        """Return data type names that are absent for the given topic."""
        have = {it.type_.lower() for it in items if it.topic == topic}
        required = {"text", "video", "chatgpt"}
        return [t for t in required if t not in have]

    def _gather_online(self, topic: str, energy: int, amount: int = 1) -> List[ResearchItem]:
        results: List[ResearchItem] = []
        for i in range(max(1, amount)):
            content = f"web data {i} for {topic} with energy {energy}"
            results.append(
                ResearchItem(
                    topic=topic,
                    content=content,
                    timestamp=time.time(),
                    type_="web",
                )
            )
            try:
                self.db_router.insert_info(results[-1])
            except Exception as exc:
                logger.exception("Failed to insert gathered online data: %s", exc)
        return results

    def _delegate_sub_bots(
        self,
        topic: str,
        energy: int,
        amount: int = 1,
        missing: Optional[Iterable[str]] = None,
    ) -> List[ResearchItem]:
        results: List[ResearchItem] = []
        collected_text = []
        queried: List[str] = []
        targets = set(missing or ["text", "video", "chatgpt"])
        ctx = self._compressed_context(topic)
        for _ in range(max(1, amount)):
            if "text" in targets and self.text_bot:
                try:
                    texts = self.text_bot.process([topic], [], ratio=0.2)
                    queried.append("text")
                    for t in texts:
                        item = ResearchItem(
                            topic=topic,
                            content=t.content,
                            timestamp=time.time(),
                            source_url=t.url,
                            type_="text",
                            associated_bots=[self.text_bot.__class__.__name__],
                        )
                        results.append(item)
                        try:
                            self.db_router.insert_info(item)
                        except Exception as exc:
                            logger.exception("Failed to insert text info: %s", exc)
                        collected_text.append(t.content)
                except Exception as exc:
                    logger.exception("Text bot failed for %s: %s", topic, exc)
            if "video" in targets and self.video_bot:
                try:
                    vids = self.video_bot.process(topic, ratio=0.2)
                    queried.append("video")
                    for v in vids:
                        item = ResearchItem(
                            topic=topic,
                            content=v.summary,
                            timestamp=time.time(),
                            source_url=v.url,
                            type_="video",
                            associated_bots=[self.video_bot.__class__.__name__],
                        )
                        results.append(item)
                        try:
                            self.db_router.insert_info(item)
                        except Exception as exc:
                            logger.exception("Failed to insert video info: %s", exc)
                except Exception as exc:
                    logger.exception("Video bot failed for %s: %s", topic, exc)
            if self.enhancement_bot and (not missing or "enhancement" in targets):
                try:
                    enhs = self.enhancement_bot.propose(
                        topic,
                        num_ideas=1,
                        context=ctx,
                        context_builder=self.context_builder,
                    )
                    for enh in enhs:
                        evaluation = None
                        if self.prediction_bot:
                            try:
                                evaluation = self.prediction_bot.evaluate_enhancement(
                                    enh.idea,
                                    enh.rationale,
                                    context_builder=self.context_builder,
                                )
                                enh.score = evaluation.value
                            except Exception:
                                evaluation = None
                                logger.exception("Enhancement prediction failed for %s", topic)
                        self.enh_db.add(enh)
                        item = ResearchItem(
                            topic=topic,
                            content=f"{enh.idea}: {enh.rationale}",
                            timestamp=time.time(),
                            categories=["enhancement"],
                            summary=evaluation.description if evaluation else "",
                            notes=evaluation.reason if evaluation else "",
                            quality=evaluation.value if evaluation else 0.0,
                            type_="enhancement",
                            associated_bots=[self.enhancement_bot.__class__.__name__],
                        )
                        results.append(item)
                        try:
                            self.db_router.insert_info(item)
                        except Exception as exc:
                            logger.exception("Failed to insert enhancement info: %s", exc)
                        self._increment_enh_count(self.info_db.current_model_id)
                except Exception as exc:
                    logger.exception("Enhancement bot failed for %s: %s", topic, exc)
        if "chatgpt" in targets and self.chatgpt_bot:
            try:
                instruction = topic
                if collected_text:
                    joined = " ".join(collected_text)
                    instruction = f"Summarise the following about {topic}: {joined}"
                if ctx:
                    instruction = f"{ctx}\n\n{instruction}"
                try:
                    res = self.chatgpt_bot.process(
                        instruction,
                        depth=1,
                        ratio=0.2,
                        context_builder=self.context_builder,
                    )
                except TypeError:
                    res = self.chatgpt_bot.process(instruction, depth=1, ratio=0.2)
                item = ResearchItem(
                    topic=topic,
                    content=res.summary,
                    timestamp=time.time(),
                    type_="chatgpt",
                    associated_bots=[self.chatgpt_bot.__class__.__name__],
                )
                results.append(item)
                queried.append("chatgpt")
                try:
                    self.db_router.insert_info(item)
                except Exception as exc:
                    logger.exception("Failed to insert chatgpt info: %s", exc)
            except Exception as exc:
                logger.exception("ChatGPT bot failed for %s: %s", topic, exc)
        if not results:
            content = f"sub bot research for {topic}"
            results.append(ResearchItem(topic=topic, content=content, timestamp=time.time()))
        for q in queried:
            if q not in self.sources_queried:
                self.sources_queried.append(q)
        return results

    @staticmethod
    def _refine(items: Iterable[ResearchItem]) -> List[ResearchItem]:
        seen: set[str] = set()
        refined: List[ResearchItem] = []
        for it in items:
            if it.content in seen:
                continue
            seen.add(it.content)
            refined.append(it)
        return refined

    def receive_chatgpt(self, convo: Iterable[Exchange], summary: str) -> None:
        """Store ChatGPT findings in memory."""
        item = ResearchItem(
            topic="chatgpt",
            content=summary,
            timestamp=time.time(),
            categories=["chatgpt"],
        )
        self.memory.add(item, layer="short")

    def _is_complete(self, items: Iterable[ResearchItem]) -> bool:
        topics = {it.topic for it in items}
        return all(req in topics for req in self.requirements)

    def _maybe_enhance(self, topic: str, reason: str) -> None:
        """Request an enhancement from the enhancement bot if available."""
        if self.manager and not self.manager.should_refactor():
            return
        if not self.enhancement_bot:
            return
        ctx = self._compressed_context(topic)
        instruction = f"{reason} about {topic}"
        if ctx:
            instruction = f"{ctx}\n\n{instruction}"
        try:
            enhancements = self.enhancement_bot.propose(
                instruction,
                num_ideas=1,
                context=ctx or topic,
                context_builder=self.context_builder,
            )
        except Exception:
            return
        for enh in enhancements:
            evaluation = None
            enh_id = 0
            try:
                if self.prediction_bot:
                    evaluation = self.prediction_bot.evaluate_enhancement(
                        enh.idea,
                        enh.rationale,
                        context_builder=self.context_builder,
                    )
                    enh.score = evaluation.value
                if getattr(self.info_db, "current_model_id", 0):
                    enh.model_ids = [self.info_db.current_model_id]
                bot_id = getattr(self, "bot_id", 0)
                if bot_id:
                    enh.bot_ids = [bot_id]
                workflow_id = getattr(self, "workflow_id", 0)
                if workflow_id:
                    enh.workflow_ids = [workflow_id]
                enh.triggered_by = self.__class__.__name__
                enh_id = self.enh_db.add(enh)
                for mid in enh.model_ids:
                    try:
                        self.enh_db.link_model(enh_id, mid)
                    except Exception as exc:
                        logger.exception("Failed to link model %s to enhancement %s: %s", mid, enh_id, exc)
                for bid in enh.bot_ids:
                    try:
                        self.enh_db.link_bot(enh_id, bid)
                    except Exception as exc:
                        logger.exception("Failed to link bot %s to enhancement %s: %s", bid, enh_id, exc)
                for wid in enh.workflow_ids:
                    try:
                        self.enh_db.link_workflow(enh_id, wid)
                    except Exception as exc:
                        logger.exception(
                            "Failed to link workflow %s to enhancement %s: %s",
                            wid,
                            enh_id,
                            exc,
                        )
                self._increment_enh_count(self.info_db.current_model_id)
            except Exception:
                evaluation = None
            item = ResearchItem(
                topic=topic,
                content=f"{enh.idea}: {enh.rationale}",
                timestamp=time.time(),
                categories=["enhancement"],
                summary=evaluation.description if evaluation else "",
                notes=evaluation.reason if evaluation else "",
                quality=evaluation.value if evaluation else 0.0,
                type_="enhancement",
            )
            self.memory.add(item, layer="medium")
            info_id = 0
            try:
                self.db_router.insert_info(item)
                info_id = item.item_id
            except Exception as exc:
                logger.exception("Failed to insert enhancement info: %s", exc)
            if info_id:
                try:
                    self.info_db.link_enhancement(info_id, enh_id)
                except Exception as exc:
                    logger.exception(
                        "Failed to link enhancement %s to info %s: %s",
                        enh_id,
                        info_id,
                        exc,
                    )

            # gather further research on the enhancement idea
            if self.text_bot:
                try:
                    texts = self.text_bot.process([enh.idea], [], ratio=0.2)
                    for t in texts:
                        r = ResearchItem(
                            topic=enh.idea,
                            content=t.content,
                            timestamp=time.time(),
                            source_url=t.url,
                            type_="text",
                        )
                        self.memory.add(r, layer="short")
                        try:
                            self.db_router.insert_info(r)
                            self.info_db.link_enhancement(r.item_id, enh_id)
                        except Exception as exc:
                            logger.exception(
                                "Failed to store text enhancement research: %s",
                                exc,
                            )
                except Exception:
                    logger.exception("Text bot failed while enhancing %s", enh.idea)
            if self.video_bot:
                try:
                    vids = self.video_bot.process(enh.idea, ratio=0.2)
                    for v in vids:
                        r = ResearchItem(
                            topic=enh.idea,
                            content=v.summary,
                            timestamp=time.time(),
                            source_url=v.url,
                            type_="video",
                        )
                        self.memory.add(r, layer="short")
                        try:
                            self.db_router.insert_info(r)
                            self.info_db.link_enhancement(r.item_id, enh_id)
                        except Exception as exc:
                            logger.exception(
                                "Failed to store video enhancement research: %s",
                                exc,
                            )
                except Exception:
                    logger.exception("Video bot failed while enhancing %s", enh.idea)
            if self.chatgpt_bot:
                try:
                    ctx_enh = self._compressed_context(enh.idea)
                    instruction = enh.idea
                    if ctx_enh:
                        instruction = f"{ctx_enh}\n\n{enh.idea}"
                    try:
                        res = self.chatgpt_bot.process(
                            instruction,
                            depth=1,
                            ratio=0.2,
                            context_builder=self.context_builder,
                        )
                    except TypeError:
                        res = self.chatgpt_bot.process(instruction, depth=1, ratio=0.2)
                    r = ResearchItem(
                        topic=enh.idea,
                        content=res.summary,
                        timestamp=time.time(),
                        type_="chatgpt",
                    )
                    self.memory.add(r, layer="short")
                    try:
                        self.db_router.insert_info(r)
                        self.info_db.link_enhancement(r.item_id, enh_id)
                    except Exception as exc:
                        logger.exception(
                            "Failed to store chatgpt enhancement research: %s",
                            exc,
                        )
                except Exception as exc:
                    logger.exception("ChatGPT bot failed while enhancing %s: %s", enh.idea, exc)

    def _collect_topic(self, topic: str, energy: int) -> List[ResearchItem]:
        if topic in self.sources_queried:
            return []
        self.sources_queried.append(topic)

        existing = self.memory.search(topic)
        local = self._query_local(topic)
        data = existing + local
        if not data:
            amount = self._info_ratio(energy)
            data = self._gather_online(topic, energy, amount)
            if energy > 2:
                data += self._delegate_sub_bots(topic, energy, amount)
        if not data:
            self._maybe_enhance(topic, "Gap detected")
        refined = self._refine(data)
        for it in refined:
            if it in local and ("workflow" in [t.lower() for t in it.tags] or it.category.lower() == "workflow"):
                tag_set = {t.lower() for t in it.tags}
                complete = set(r.lower() for r in self.requirements).issubset(tag_set)
                depth_ok = it.data_depth >= 0.5
                if not complete or not depth_ok:
                    if "partial_reusable" not in tag_set:
                        it.tags.append("partial_reusable")
                elif "reusable" not in tag_set:
                    it.tags.append("reusable")
            self.memory.add(it, layer="medium")
            try:
                self.db_router.insert_info(it)
            except Exception as exc:
                logger.exception("Failed to insert collected topic info: %s", exc)
        return refined

    def process(self, topic: str, energy: int = 1) -> List[ResearchItem]:
        now = time.time()
        if topic in self.cache:
            ts, data = self.cache[topic]
            if now - ts < self.cache_ttl:
                return list(data)

        self.memory.decay()
        now = time.time()
        if now - self._last_enhancement_time >= self.enhancement_interval:
            self._maybe_enhance(topic, "Periodic enhancement")
            self._last_enhancement_time = now
        results = self._collect_topic(topic, energy)
        missing_types = self._missing_data_types(results, topic)
        if energy > 2 and missing_types:
            results.extend(self._delegate_sub_bots(topic, energy, missing=missing_types))

        topics = {it.topic for it in results}
        missing = [req for req in self.requirements if req not in topics]
        attempts = 0
        # continually request missing topics until complete or attempts exceeded
        while missing and attempts < len(self.requirements) * 2:
            for req in list(missing):
                res = self._collect_topic(req, energy)
                results.extend(res)
                miss = self._missing_data_types(res, req)
                if energy > 2 and miss:
                    results.extend(self._delegate_sub_bots(req, energy, missing=miss))
            topics = {it.topic for it in results}
            missing = [req for req in self.requirements if req not in topics]
            attempts += 1
        refined = self._refine(results)
        if self._is_complete(refined):
            self.memory.archive(refined)
            send_to_stage3(refined)
        self.cache[topic] = (time.time(), list(refined))
        return refined


def send_to_stage3(items: Iterable[ResearchItem]) -> None:
    """Forward *items* to Stage 3 via HTTP if ``requests`` is available."""

    try:
        import requests  # type: ignore
        from dataclasses import asdict
    except Exception:  # pragma: no cover - optional dependency
        return

    url = os.getenv("STAGE3_URL")
    if not url:
        return

    payload = [asdict(it) for it in items]
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception:  # pragma: no cover - network issues
        logging.getLogger(__name__).warning("Failed to forward items to Stage 3")


__all__ = [
    "ResearchItem",
    "ResearchMemory",
    "InfoDB",
    "EnhancementDB",
    "ChatGPTEnhancementBot",
    "ChatGPTPredictionBot",
    "TextResearchBot",
    "VideoResearchBot",
    "ChatGPTResearchBot",
    "ResearchAggregatorBot",
    "send_to_stage3",
]
