"""High level pipeline orchestrating model automation.

When :mod:`marshmallow` is not installed, a very small schema system is
provided to validate records.  The fallback checks that required fields
are present and match the expected type instead of blindly returning the
data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterable, Dict, Optional, Any, TYPE_CHECKING, Callable, Type, cast
from functools import lru_cache
import logging

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
try:
    from marshmallow import Schema, fields  # type: ignore
    from marshmallow import ValidationError as MMValidationError  # type: ignore
    ValidationError = MMValidationError
except Exception:  # pragma: no cover - optional dependency
    from .simple_validation import (  # noqa: F401
        SimpleSchema as Schema,
        fields,
        ValidationError,
    )

from .resource_prediction_bot import ResourcePredictionBot, ResourceMetrics
from .data_bot import DataBot
from .task_handoff_bot import TaskHandoffBot, TaskInfo, TaskPackage, WorkflowDB
from .efficiency_bot import EfficiencyBot
from .performance_assessment_bot import PerformanceAssessmentBot
from .communication_maintenance_bot import CommunicationMaintenanceBot
from .operational_monitor_bot import OperationalMonitoringBot
from .central_database_bot import CentralDatabaseBot, Proposal
from .sentiment_bot import SentimentBot
from .query_bot import QueryBot
from .memory_bot import MemoryBot
from .communication_testing_bot import CommunicationTestingBot
from .discrepancy_detection_bot import DiscrepancyDetectionBot
from .finance_router_bot import FinanceRouterBot
from .meta_genetic_algorithm_bot import MetaGeneticAlgorithmBot
from .offer_testing_bot import OfferTestingBot
from .research_fallback_bot import ResearchFallbackBot
from .resource_allocation_optimizer import ResourceAllocationOptimizer
from .resource_allocation_bot import ResourceAllocationBot, AllocationDB
from .database_manager import update_model
from .ai_counter_bot import AICounterBot
from .dynamic_resource_allocator_bot import DynamicResourceAllocator
from .diagnostic_manager import DiagnosticManager
from .idea_search_bot import KeywordBank
from .newsreader_bot import NewsDB
from .investment_engine import AutoReinvestmentBot
from .revenue_amplifier import (
    RevenueSpikeEvaluatorBot,
    CapitalAllocationBot,
    RevenueEventsDB,
)
from .bot_db_utils import wrap_bot_methods
from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from .unified_event_bus import UnifiedEventBus
from .neuroplasticity import Outcome, PathwayDB, PathwayRecord
from .unified_learning_engine import UnifiedLearningEngine
from .action_planner import ActionPlanner
from vector_service.context_builder import ContextBuilder
from .shared.model_pipeline_core import ModelAutomationPipeline

if TYPE_CHECKING:  # pragma: no cover - typing only
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
else:  # pragma: no cover - runtime fallback
    HierarchyAssessmentBot = Any  # type: ignore
    BotPlanningBot = Any  # type: ignore
    PlanningTask = Any  # type: ignore
    BotPlan = Any  # type: ignore
    BotCreationBot = Any  # type: ignore
    CapitalManagementBot = Any  # type: ignore
    ResearchAggregatorBot = Any  # type: ignore
    ResearchItem = Any  # type: ignore
    PreExecutionROIBot = Any  # type: ignore
    BuildTask = Any  # type: ignore
    ROIResult = Any  # type: ignore


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

    from .implementation_optimiser_bot import ImplementationOptimiserBot as _ImplementationOptimiserBot

    return _ImplementationOptimiserBot

MENACE_ID = "model_automation_pipeline"
DB_ROUTER = GLOBAL_ROUTER or init_db_router(MENACE_ID)


@dataclass
class AutomationResult:
    """Final pipeline output."""
    package: Optional[TaskPackage]
    roi: Optional[ROIResult]
    warnings: Dict[str, List[Dict[str, Any]]] | None = None
    workflow_evolution: List[Dict[str, Any]] | None = None


if TYPE_CHECKING:  # pragma: no cover - type checking only imports
    from .task_validation_bot import TaskValidationBot
else:  # pragma: no cover - runtime fallback avoids circular imports
    TaskValidationBot = Any  # type: ignore[misc, assignment]


@lru_cache(maxsize=1)
def _task_validation_cls() -> Type["TaskValidationBot"]:
    """Return the ``TaskValidationBot`` class without triggering circular imports.

    The import of :mod:`task_validation_bot` eagerly initialises the
    self-coding stack which, in turn, reaches back into this module to build the
    automation pipeline.  Importing the class lazily via this helper prevents
    Windows start-up sequences from oscillating between partially initialised
    modules and repeated internalisation retries.  The ``lru_cache`` guard keeps
    the import overhead negligible while remaining thread-safe.
    """

    from .task_validation_bot import TaskValidationBot as _TaskValidationBot

    return _TaskValidationBot


def _build_default_validator() -> "TaskValidationBot":
    """Instantiate a default task validator using the cached class reference."""

    validator_cls = _task_validation_cls()
    return cast("TaskValidationBot", validator_cls([]))


@lru_cache(maxsize=1)
def _hierarchy_bot_cls() -> Type["HierarchyAssessmentBot"]:
    """Return ``HierarchyAssessmentBot`` lazily to avoid circular imports."""

    from .hierarchy_assessment_bot import HierarchyAssessmentBot as _HierarchyAssessmentBot

    return _HierarchyAssessmentBot


def _build_default_hierarchy() -> "HierarchyAssessmentBot":
    """Instantiate the default hierarchy bot using the cached class."""

    hierarchy_cls = _hierarchy_bot_cls()
    return cast("HierarchyAssessmentBot", hierarchy_cls())


@lru_cache(maxsize=1)
def _capital_manager_cls() -> Type["CapitalManagementBot"]:
    """Return the capital manager class lazily to avoid circular imports."""

    from .capital_management_bot import CapitalManagementBot as _CapitalManagementBot

    return _CapitalManagementBot


@lru_cache(maxsize=1)
def _planning_components() -> tuple[type["BotPlanningBot"], type["PlanningTask"], type["BotPlan"]]:
    """Load planning helpers lazily to avoid circular imports."""

    from .bot_planning_bot import BotPlanningBot as _BotPlanningBot
    from .bot_planning_bot import BotPlan as _BotPlan
    from .bot_planning_bot import PlanningTask as _PlanningTask

    return _BotPlanningBot, _PlanningTask, _BotPlan


__all__ = ["AutomationResult", "ModelAutomationPipeline"]
