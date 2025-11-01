"""Shared helpers for :mod:`model_automation_pipeline` and execution core."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Type, cast

from vector_service.context_builder import ContextBuilder

from .shared.cooperative_init import ensure_cooperative_init

from .db_router import GLOBAL_ROUTER, init_db_router

if TYPE_CHECKING:  # pragma: no cover - typing only imports
    from .hierarchy_assessment_bot import HierarchyAssessmentBot
    from .bot_planning_bot import BotPlanningBot, PlanningTask, BotPlan
    from .bot_registry import BotRegistry
    from .bot_creation_bot import BotCreationBot
    from .capital_management_bot import CapitalManagementBot
    from .research_aggregator_bot import ResearchAggregatorBot, ResearchItem
    from .information_synthesis_bot import InformationSynthesisBot
    from .synthesis_models import SynthesisTask
    from .implementation_optimiser_bot import ImplementationOptimiserBot
    from .pre_execution_roi_bot import PreExecutionROIBot, BuildTask, ROIResult
    from .task_validation_bot import TaskValidationBot
else:  # pragma: no cover - runtime fallback avoids circular imports
    HierarchyAssessmentBot = Any  # type: ignore
    BotPlanningBot = Any  # type: ignore
    PlanningTask = Any  # type: ignore
    BotPlan = Any  # type: ignore
    BotRegistry = Any  # type: ignore
    BotCreationBot = Any  # type: ignore
    CapitalManagementBot = Any  # type: ignore
    ResearchAggregatorBot = Any  # type: ignore
    ResearchItem = Any  # type: ignore
    InformationSynthesisBot = Any  # type: ignore
    SynthesisTask = Any  # type: ignore
    ImplementationOptimiserBot = Any  # type: ignore
    PreExecutionROIBot = Any  # type: ignore
    BuildTask = Any  # type: ignore
    ROIResult = Any  # type: ignore
    TaskValidationBot = Any  # type: ignore

MENACE_ID = "model_automation_pipeline"
DB_ROUTER = GLOBAL_ROUTER or init_db_router(MENACE_ID)


def _create_synthesis_task(**kwargs: Any) -> "SynthesisTask":
    """Construct :class:`SynthesisTask` without importing at module import time."""

    from .synthesis_models import SynthesisTask

    return SynthesisTask(**kwargs)


class _LazyAggregator:
    """Deferred loader for :class:`ResearchAggregatorBot` instances."""

    def __init__(self, factory: Callable[[], "ResearchAggregatorBot"]) -> None:
        self._factory = factory
        self._instance: "ResearchAggregatorBot | None" = None

    def _ensure(self) -> "ResearchAggregatorBot":
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensure(), name)


def _load_research_aggregator(
    context_builder: ContextBuilder,
) -> "ResearchAggregatorBot":
    from .research_aggregator_bot import ResearchAggregatorBot

    return ResearchAggregatorBot([], context_builder=context_builder)


def _make_research_item(**kwargs: Any) -> "ResearchItem":
    from .research_aggregator_bot import ResearchItem

    return ResearchItem(**kwargs)


def _implementation_optimiser_cls() -> Type["ImplementationOptimiserBot"]:
    """Return the optimiser class without importing during module initialisation."""

    from .implementation_optimiser_bot import (
        ImplementationOptimiserBot as _ImplementationOptimiserBot,
    )

    return _ImplementationOptimiserBot


@lru_cache(maxsize=1)
def _task_validation_cls() -> Type["TaskValidationBot"]:
    """Return the ``TaskValidationBot`` class without triggering circular imports."""

    from .task_validation_bot import TaskValidationBot as _TaskValidationBot

    return _TaskValidationBot


def _build_default_validator() -> "TaskValidationBot":
    """Instantiate a default task validator using the cached class reference."""

    validator_cls = _task_validation_cls()
    return cast("TaskValidationBot", validator_cls([]))


@lru_cache(maxsize=1)
def _hierarchy_bot_cls() -> Type["HierarchyAssessmentBot"]:
    """Return ``HierarchyAssessmentBot`` lazily to avoid circular imports."""

    from .hierarchy_assessment_bot import (
        HierarchyAssessmentBot as _HierarchyAssessmentBot,
    )

    return _HierarchyAssessmentBot


def _build_default_hierarchy() -> "HierarchyAssessmentBot":
    """Instantiate the default hierarchy bot using the cached class."""

    hierarchy_cls = _hierarchy_bot_cls()
    return cast("HierarchyAssessmentBot", hierarchy_cls())


@lru_cache(maxsize=1)
def _capital_manager_cls() -> Type["CapitalManagementBot"]:
    """Return the capital manager class lazily to avoid circular imports."""

    module = import_module(f"{__package__}.capital_management_bot")
    capital_cls = getattr(module, "CapitalManagementBot", None)
    if capital_cls is None:
        capital_cls = getattr(module, "_CapitalManagementBot", None)
    if capital_cls is None:
        raise ImportError("CapitalManagementBot class unavailable")

    cooperative_cls = ensure_cooperative_init(cast(type, capital_cls))
    return cast("Type[CapitalManagementBot]", cooperative_cls)


@lru_cache(maxsize=1)
def _planning_components() -> tuple[
    type["BotPlanningBot"],
    type["PlanningTask"],
    type["BotPlan"],
]:
    """Load planning helpers lazily to avoid circular imports."""

    from .bot_planning_bot import BotPlanningBot as _BotPlanningBot
    from .bot_planning_bot import BotPlan as _BotPlan
    from .bot_planning_bot import PlanningTask as _PlanningTask

    cooperative_cls = ensure_cooperative_init(cast(type, _BotPlanningBot))
    return cast(type["BotPlanningBot"], cooperative_cls), _PlanningTask, _BotPlan


__all__ = [
    "MENACE_ID",
    "DB_ROUTER",
    "_LazyAggregator",
    "_build_default_hierarchy",
    "_build_default_validator",
    "_capital_manager_cls",
    "_create_synthesis_task",
    "_implementation_optimiser_cls",
    "_load_research_aggregator",
    "_make_research_item",
    "_planning_components",
]
