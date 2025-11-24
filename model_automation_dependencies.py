"""Shared helpers for :mod:`model_automation_pipeline` and execution core."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from importlib import import_module
import logging
import queue
import threading
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Type, cast

from .shared.cooperative_init import ensure_cooperative_init, monkeypatch_class_references

from .db_router import GLOBAL_ROUTER, init_db_router

if TYPE_CHECKING:  # pragma: no cover - typing only imports
    from vector_service.context_builder import ContextBuilder
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
    ContextBuilder = Any  # type: ignore
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
_PLANNING_IMPORT_TIMEOUT = 5.0


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

    original_cls = cast(type, capital_cls)
    cooperative_cls = ensure_cooperative_init(cast(type, capital_cls))
    monkeypatch_class_references(original_cls, cooperative_cls)
    return cast("Type[CapitalManagementBot]", cooperative_cls)


def _import_planning_classes() -> tuple[type, type, type]:
    """Import the planning classes directly from :mod:`bot_planning_bot`."""

    from .bot_planning_bot import BotPlanningBot as _BotPlanningBot
    from .bot_planning_bot import BotPlan as _BotPlan
    from .bot_planning_bot import PlanningTask as _PlanningTask

    return _BotPlanningBot, _PlanningTask, _BotPlan


def _load_with_timeout(
    loader: Callable[[], tuple[type, type, type]], timeout: float
) -> tuple[tuple[type, type, type] | None, BaseException | None]:
    """Execute ``loader`` in a helper thread, returning on completion or timeout."""

    result_queue: "queue.Queue[tuple[tuple[type, type, type] | None, BaseException | None]]" = (
        queue.Queue(maxsize=1)
    )

    def _runner() -> None:
        try:
            result_queue.put((loader(), None))
        except BaseException as exc:  # pragma: no cover - passthrough
            result_queue.put((None, exc))

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()

    try:
        return result_queue.get(timeout=timeout)
    except queue.Empty:
        return None, TimeoutError(f"planning import exceeded {timeout} seconds")


@lru_cache(maxsize=1)
def _planning_components() -> tuple[
    type["BotPlanningBot"],
    type["PlanningTask"],
    type["BotPlan"],
]:
    """Load planning helpers lazily with a timeout to avoid bootstrap hangs."""

    loaded, error = _load_with_timeout(_import_planning_classes, _PLANNING_IMPORT_TIMEOUT)
    if error is None and loaded is not None:
        _BotPlanningBot, _PlanningTask, _BotPlan = loaded
        original_cls = cast(type, _BotPlanningBot)
        cooperative_cls = ensure_cooperative_init(cast(type, _BotPlanningBot))
        monkeypatch_class_references(original_cls, cooperative_cls)
        return cast(type["BotPlanningBot"], cooperative_cls), _PlanningTask, _BotPlan

    logger = logging.getLogger(__name__)
    logger.warning(
        "BotPlanningBot unavailable during model automation bootstrap; "
        "using inert planning stubs: %s",
        error,
    )

    @dataclass
    class _StubPlanningTask:  # type: ignore[local-name-defined]
        description: str = ""
        complexity: int = 0
        frequency: int = 0
        expected_time: float = 0
        actions: List[str] = field(default_factory=list)
        env: List[str] = field(default_factory=list)
        constraints: dict[str, int] = field(default_factory=dict)
        resources: dict[str, int] = field(default_factory=dict)

    @dataclass
    class _StubBotPlan:  # type: ignore[local-name-defined]
        name: str = ""
        template: str = ""
        scalability: float = 0.0
        level: str = ""

    class _StubPlanner:  # type: ignore[local-name-defined]
        """No-op planner used when real planning dependencies are unavailable."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            self.logger = logging.getLogger(f"{__name__}.StubPlanner")

        def evaluate_tasks(self, tasks: Iterable[_StubPlanningTask]) -> List[float]:
            return [0.0 for _ in tasks]

        def optimise_resources(
            self, tasks: Iterable[_StubPlanningTask], *, cpu_limit: float = 0.0
        ) -> List[float]:
            return [0.0 for _ in tasks]

        def plan_bots(
            self,
            tasks: Iterable[_StubPlanningTask],
            *,
            trust_weight: float = 1.0,
        ) -> List[_StubBotPlan]:
            return []

    return _StubPlanner, _StubPlanningTask, _StubBotPlan


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
