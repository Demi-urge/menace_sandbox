"""Bootstrap helpers for seeding shared self-coding context.

The utilities here build the same pipeline/manager setup used by the runtime
bootstrapping helpers and then expose that state to lazy modules so they can
skip re-entrant ``prepare_pipeline_for_bootstrap`` calls.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from menace_sandbox.bot_registry import BotRegistry
from menace_sandbox.code_database import CodeDB
from menace_sandbox.context_builder_util import create_context_builder
from menace_sandbox.coding_bot_interface import (
    _push_bootstrap_context,
    fallback_helper_manager,
    prepare_pipeline_for_bootstrap,
)
from menace_sandbox.data_bot import DataBot, persist_sc_thresholds
from menace_sandbox.menace_memory_manager import MenaceMemoryManager
from menace_sandbox.model_automation_pipeline import ModelAutomationPipeline
from menace_sandbox.self_coding_engine import SelfCodingEngine
from menace_sandbox.self_coding_manager import SelfCodingManager, internalize_coding_bot
from menace_sandbox.self_coding_thresholds import get_thresholds
from menace_sandbox.threshold_service import ThresholdService

LOGGER = logging.getLogger(__name__)

_BOOTSTRAP_CACHE: Dict[str, Dict[str, Any]] = {}


def _seed_research_aggregator_context(
    *,
    registry: BotRegistry,
    data_bot: DataBot,
    context_builder: Any,
    engine: SelfCodingEngine,
    pipeline: ModelAutomationPipeline,
    manager: SelfCodingManager,
) -> None:
    try:
        from menace_sandbox import research_aggregator_bot as aggregator
    except Exception:  # pragma: no cover - optional optimisation
        LOGGER.debug("research_aggregator_bot unavailable during seeding", exc_info=True)
        return

    try:
        orchestrator = aggregator.get_orchestrator(
            "ResearchAggregatorBot", data_bot, engine
        )
    except Exception:  # pragma: no cover - orchestrator best effort
        LOGGER.debug("research aggregator orchestrator unavailable", exc_info=True)
        orchestrator = None

    aggregator.registry = registry
    aggregator.data_bot = data_bot
    aggregator._context_builder = context_builder
    aggregator.engine = engine
    aggregator.pipeline = pipeline
    aggregator.manager = manager
    aggregator.evolution_orchestrator = orchestrator
    aggregator._PipelineCls = type(pipeline)

    try:
        runtime_state = aggregator._RuntimeDependencies(
            registry=registry,
            data_bot=data_bot,
            context_builder=context_builder,
            engine=engine,
            pipeline=pipeline,
            evolution_orchestrator=orchestrator,
            manager=manager,
        )
        aggregator._runtime_state = runtime_state
        aggregator._ensure_self_coding_decorated(runtime_state)
    except Exception:  # pragma: no cover - optional runtime hints
        LOGGER.debug("unable to seed research aggregator runtime state", exc_info=True)


def initialize_bootstrap_context(
    bot_name: str = "ResearchAggregatorBot", *, use_cache: bool = True
) -> Dict[str, Any]:
    """Build and seed bootstrap helpers for reuse by entry points.

    The returned mapping contains the seeded ``registry``, ``data_bot``,
    ``context_builder``, ``engine``, ``pipeline`` and ``manager`` instances.
    Subsequent invocations return cached instances for the given ``bot_name`` when
    ``use_cache`` is ``True``. Pass ``use_cache=False`` to force a fresh bootstrap
    without populating or reading the shared cache.
    """

    global _BOOTSTRAP_CACHE
    if use_cache and bot_name in _BOOTSTRAP_CACHE:
        LOGGER.info(
            "reusing preseeded bootstrap context for %s; pipeline/manager already available",
            bot_name,
        )
        return _BOOTSTRAP_CACHE[bot_name]

    context_builder = create_context_builder()
    registry = BotRegistry()
    data_bot = DataBot(start_server=False)
    engine = SelfCodingEngine(
        CodeDB(),
        MenaceMemoryManager(),
        context_builder=context_builder,
    )

    with fallback_helper_manager(bot_registry=registry, data_bot=data_bot) as bootstrap_manager:
        pipeline, promote_pipeline = prepare_pipeline_for_bootstrap(
            pipeline_cls=ModelAutomationPipeline,
            context_builder=context_builder,
            bot_registry=registry,
            data_bot=data_bot,
            bootstrap_runtime_manager=bootstrap_manager,
            manager=bootstrap_manager,
        )

    thresholds = get_thresholds(bot_name)
    try:
        persist_sc_thresholds(
            bot_name,
            roi_drop=thresholds.roi_drop,
            error_increase=thresholds.error_increase,
            test_failure_increase=thresholds.test_failure_increase,
        )
    except Exception:  # pragma: no cover - best effort persistence
        LOGGER.debug("failed to persist thresholds for %s", bot_name, exc_info=True)

    manager = internalize_coding_bot(
        bot_name,
        engine,
        pipeline,
        data_bot=data_bot,
        bot_registry=registry,
        threshold_service=ThresholdService(),
        roi_threshold=thresholds.roi_drop,
        error_threshold=thresholds.error_increase,
        test_failure_threshold=thresholds.test_failure_increase,
    )
    promote_pipeline(manager)

    _push_bootstrap_context(
        registry=registry, data_bot=data_bot, manager=manager, pipeline=pipeline
    )
    _seed_research_aggregator_context(
        registry=registry,
        data_bot=data_bot,
        context_builder=context_builder,
        engine=engine,
        pipeline=pipeline,
        manager=manager,
    )

    if use_cache:
        _BOOTSTRAP_CACHE[bot_name] = {
            "registry": registry,
            "data_bot": data_bot,
            "context_builder": context_builder,
            "engine": engine,
            "pipeline": pipeline,
            "manager": manager,
        }
    return {
        "registry": registry,
        "data_bot": data_bot,
        "context_builder": context_builder,
        "engine": engine,
        "pipeline": pipeline,
        "manager": manager,
    }


__all__ = ["initialize_bootstrap_context"]
