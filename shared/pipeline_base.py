"""Core implementation shared by automation pipeline components.

This module hosts the :class:`ModelAutomationPipeline` implementation that was
previously defined in :mod:`menace_sandbox.shared.execution_core`.  Extracting
the class into a neutral module allows both the self-coding engine and the
capital management bot to import the pipeline interface without triggering
the heavy bootstrap logic associated with the respective subsystems.  The
supporting helpers remain unchanged so the runtime behaviour of the pipeline
is preserved.
"""

from __future__ import annotations

import contextlib
import os
import traceback
import weakref


def _trace(message: str) -> None:
    """Emit verbose import traces when ``MENACE_TRACE_IMPORTS`` is set."""

    if os.environ.get("MENACE_TRACE_IMPORTS"):
        print(f">>> [trace] {message}")


_trace("Entered pipeline_base.py (BEGIN)")
_trace("Import stack:\n" + "".join(traceback.format_stack()[-10:]))

_trace("Entered pipeline_base.py")
_trace("Successfully imported annotations from __future__")

_trace("Importing logging...")
import logging
_trace("Successfully imported logging")
_trace("Importing dependency health helpers...")
from dependency_health import (
    dependency_registry,
    DependencyCategory,
    DependencySeverity,
)
_trace("Successfully imported dependency health helpers")

_LOGGER = logging.getLogger(__name__)
_trace("Importing ThreadPoolExecutor, as_completed from concurrent.futures...")
from concurrent.futures import ThreadPoolExecutor, as_completed
_trace("Successfully imported ThreadPoolExecutor, as_completed from concurrent.futures")
_trace("Importing SimpleNamespace from types...")
from types import SimpleNamespace
_trace("Successfully imported SimpleNamespace from types")
_trace("Importing Path from pathlib...")
from pathlib import Path
_trace("Successfully imported Path from pathlib")
_trace("Importing TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Tuple, Type, cast from typing...")
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Tuple,
    Type,
    cast,
)
_trace("Successfully imported TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Tuple, Type, cast from typing")

try:
    _trace("Importing pandas as pd...")
    import pandas as pd  # type: ignore
    _trace("Successfully imported pandas as pd")
    dependency_registry.mark_available(
        name="pandas",
        category=DependencyCategory.PYTHON,
        optional=True,
        description="Data analysis helper for pipeline automation",
        logger=_LOGGER,
    )
except Exception as exc:  # pragma: no cover - optional dependency
    message = f"Failed to import pandas, defaulting to None ({exc!r})"
    _trace(message)
    pd = None  # type: ignore
    dependency_registry.mark_missing(
        name="pandas",
        category=DependencyCategory.PYTHON,
        optional=True,
        severity=DependencySeverity.INFO,
        description="Data analysis helper for pipeline automation",
        reason=str(exc),
        remedy="pip install pandas",
        logger=_LOGGER,
    )
    _LOGGER.warning(
        "Pandas import failed; falling back to _TaskTableFallback", exc_info=True
    )


class _TaskTableFallback:
    """Light-weight stand-in for :class:`pandas.DataFrame` used in tests."""

    __slots__ = ("_rows",)

    def __init__(self, rows: Iterable[Dict[str, Any]]) -> None:
        self._rows = list(rows)

    def iterrows(self):
        """Yield ``(index, row)`` tuples compatible with ``DataFrame.iterrows``."""

        for idx, row in enumerate(self._rows):
            yield idx, row

    def __len__(self) -> int:  # pragma: no cover - convenience only
        return len(self._rows)

try:
    _trace("Importing Schema, fields, ValidationError from marshmallow...")
    from marshmallow import Schema, fields  # type: ignore
    from marshmallow import ValidationError as MMValidationError  # type: ignore

    ValidationError = MMValidationError
    _trace("Successfully imported Schema, fields, ValidationError from marshmallow")
except Exception:  # pragma: no cover - optional dependency
    _trace("Failed to import marshmallow, falling back to simple_validation")
    _trace("Importing SimpleSchema as Schema, fields, ValidationError from menace_sandbox.simple_validation...")
    from ..simple_validation import (  # type: ignore
        SimpleSchema as Schema,
        fields,
        ValidationError,
    )
    _trace(
        "Successfully imported SimpleSchema as Schema, fields, ValidationError from menace_sandbox.simple_validation"
    )

_trace("Importing ResourcePredictionBot, ResourceMetrics from menace_sandbox.resource_prediction_bot...")
from ..resource_prediction_bot import ResourcePredictionBot, ResourceMetrics
_trace("Successfully imported ResourcePredictionBot, ResourceMetrics from menace_sandbox.resource_prediction_bot")
_trace("Importing DataBotInterface from menace_sandbox.data_interfaces...")
from ..data_interfaces import DataBotInterface
_trace("Successfully imported DataBotInterface from menace_sandbox.data_interfaces")
_trace("Importing TaskHandoffBot, TaskInfo, TaskPackage, WorkflowDB from menace_sandbox.task_handoff_bot...")
from ..task_handoff_bot import TaskHandoffBot, TaskInfo, TaskPackage, WorkflowDB
_trace("Successfully imported TaskHandoffBot, TaskInfo, TaskPackage, WorkflowDB from menace_sandbox.task_handoff_bot")
_trace("Importing EfficiencyBot from menace_sandbox.efficiency_bot...")
from ..efficiency_bot import EfficiencyBot
_trace("Successfully imported EfficiencyBot from menace_sandbox.efficiency_bot")
_trace("Importing PerformanceAssessmentBot from menace_sandbox.performance_assessment_bot...")
from ..performance_assessment_bot import PerformanceAssessmentBot
_trace("Successfully imported PerformanceAssessmentBot from menace_sandbox.performance_assessment_bot")
_trace(
    "Preparing lazy import for CommunicationMaintenanceBot from "
    "menace_sandbox.communication_maintenance_bot..."
)
_trace("Importing OperationalMonitoringBot from menace_sandbox.operational_monitor_bot...")
from ..operational_monitor_bot import OperationalMonitoringBot
_trace("Successfully imported OperationalMonitoringBot from menace_sandbox.operational_monitor_bot")
_trace("Importing CentralDatabaseBot, Proposal from menace_sandbox.central_database_bot...")
from ..central_database_bot import CentralDatabaseBot, Proposal
_trace("Successfully imported CentralDatabaseBot, Proposal from menace_sandbox.central_database_bot")
_trace("Importing SentimentBot from menace_sandbox.sentiment_bot...")
from ..sentiment_bot import SentimentBot
_trace("Successfully imported SentimentBot from menace_sandbox.sentiment_bot")
_trace("Importing QueryBot from menace_sandbox.query_bot...")
from ..query_bot import QueryBot
_trace("Successfully imported QueryBot from menace_sandbox.query_bot")
_trace("Importing MemoryBot from menace_sandbox.memory_bot...")
from ..memory_bot import MemoryBot
_trace("Successfully imported MemoryBot from menace_sandbox.memory_bot")
_trace("Skipping eager import of DiscrepancyDetectionBot to avoid circular dependency...")
_trace("Importing OfferTestingBot from menace_sandbox.offer_testing_bot...")
from ..offer_testing_bot import OfferTestingBot
_trace("Successfully imported OfferTestingBot from menace_sandbox.offer_testing_bot")
_trace("Importing ResearchFallbackBot from menace_sandbox.research_fallback_bot...")
from ..research_fallback_bot import ResearchFallbackBot
from ..coding_bot_interface import (
    MANAGER_CONTEXT,
    normalise_manager_arg,
    pipeline_context_scope,
)
try:  # pragma: no cover - defensive import for stripped down tests
    from ..coding_bot_interface import (
        _BOOTSTRAP_STATE,
        _is_bootstrap_placeholder,
        _seed_existing_pipeline_placeholder,
    )
except Exception:  # pragma: no cover - fallback when self-coding engine absent
    _BOOTSTRAP_STATE = SimpleNamespace()  # type: ignore[assignment]

    def _is_bootstrap_placeholder(_candidate: object) -> bool:  # type: ignore[no-redef]
        return False

    def _seed_existing_pipeline_placeholder(  # type: ignore[no-redef]
        _pipeline: object, _placeholder: object
    ) -> None:
        return None

    @contextlib.contextmanager
    def pipeline_context_scope(_pipeline: object) -> Iterator[None]:  # type: ignore[no-redef]
        yield None
_trace("Successfully imported ResearchFallbackBot from menace_sandbox.research_fallback_bot")
_trace("Preparing lazy import for ResourceAllocationOptimizer...")


def get_resource_allocation_optimizer_cls() -> "type[ResourceAllocationOptimizer]":
    _trace("Lazily importing ResourceAllocationOptimizer from menace_sandbox.resource_allocation_optimizer...")
    from menace_sandbox.resource_allocation_optimizer import ResourceAllocationOptimizer

    _trace("Successfully lazy-imported ResourceAllocationOptimizer from menace_sandbox.resource_allocation_optimizer")
    return ResourceAllocationOptimizer


def get_communication_maintenance_bot_cls() -> "type[CommunicationMaintenanceBot]":
    _trace(
        "Lazily importing CommunicationMaintenanceBot from "
        "menace_sandbox.communication_maintenance_bot..."
    )
    from ..communication_maintenance_bot import CommunicationMaintenanceBot

    _trace(
        "Successfully lazy-imported CommunicationMaintenanceBot from "
        "menace_sandbox.communication_maintenance_bot"
    )
    return CommunicationMaintenanceBot


_trace("Preparing lazy import for MetaGeneticAlgorithmBot...")


def get_meta_genetic_algorithm_bot_cls() -> "type[MetaGeneticAlgorithmBot]":
    _trace("Lazily importing MetaGeneticAlgorithmBot from menace_sandbox.meta_genetic_algorithm_bot...")
    from ..meta_genetic_algorithm_bot import (
        MetaGeneticAlgorithmBot as _MetaGeneticAlgorithmBot,
    )

    _trace("Successfully lazy-imported MetaGeneticAlgorithmBot from menace_sandbox.meta_genetic_algorithm_bot")
    return _MetaGeneticAlgorithmBot


if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    from menace_sandbox.communication_maintenance_bot import CommunicationMaintenanceBot
    from menace_sandbox.resource_allocation_optimizer import ResourceAllocationOptimizer
    from menace_sandbox.dynamic_resource_allocator_bot import DynamicResourceAllocator
    from menace_sandbox.diagnostic_manager import DiagnosticManager
_trace("Importing update_model from menace_sandbox.database_manager...")
from ..database_manager import update_model
_trace("Successfully imported update_model from menace_sandbox.database_manager")
_trace("Importing AICounterBot from menace_sandbox.ai_counter_bot...")
from ..ai_counter_bot import AICounterBot
_trace("Successfully imported AICounterBot from menace_sandbox.ai_counter_bot")
_trace("Preparing lazy import for DynamicResourceAllocator...")


def get_dynamic_resource_allocator_cls() -> "type[DynamicResourceAllocator]":
    _trace("Lazily importing DynamicResourceAllocator from menace_sandbox.dynamic_resource_allocator_bot...")
    from menace_sandbox.dynamic_resource_allocator_bot import (
        DynamicResourceAllocator as _DynamicResourceAllocator,
    )

    _trace("Successfully lazy-imported DynamicResourceAllocator from menace_sandbox.dynamic_resource_allocator_bot")
    return _DynamicResourceAllocator
_trace("Preparing lazy import for DiagnosticManager...")


def get_diagnostic_manager_cls() -> "type[DiagnosticManager]":
    _trace("Lazily importing DiagnosticManager from menace_sandbox.diagnostic_manager...")
    from menace_sandbox.diagnostic_manager import (
        DiagnosticManager as _DiagnosticManager,
    )

    _trace("Successfully lazy-imported DiagnosticManager from menace_sandbox.diagnostic_manager")
    return _DiagnosticManager
_trace("Importing KeywordBank from menace_sandbox.idea_search_bot...")
from ..idea_search_bot import KeywordBank
_trace("Successfully imported KeywordBank from menace_sandbox.idea_search_bot")
_trace("Importing NewsDB from menace_sandbox.newsreader_bot...")
from ..newsreader_bot import NewsDB
_trace("Successfully imported NewsDB from menace_sandbox.newsreader_bot")
_trace("Importing AutoReinvestmentBot from menace_sandbox.investment_engine...")
from ..investment_engine import AutoReinvestmentBot
_trace("Successfully imported AutoReinvestmentBot from menace_sandbox.investment_engine")
_trace("Importing RevenueSpikeEvaluatorBot, CapitalAllocationBot, RevenueEventsDB from menace_sandbox.revenue_amplifier...")
from ..revenue_amplifier import (
    RevenueSpikeEvaluatorBot,
    CapitalAllocationBot,
    RevenueEventsDB,
)
_trace("Successfully imported RevenueSpikeEvaluatorBot, CapitalAllocationBot, RevenueEventsDB from menace_sandbox.revenue_amplifier")
_trace("Importing wrap_bot_methods from menace_sandbox.bot_db_utils...")
from ..bot_db_utils import wrap_bot_methods
_trace("Successfully imported wrap_bot_methods from menace_sandbox.bot_db_utils")
_trace("Importing DBRouter from menace_sandbox.db_router...")
from ..db_router import DBRouter
_trace("Successfully imported DBRouter from menace_sandbox.db_router")
_trace("Importing UnifiedEventBus from menace_sandbox.unified_event_bus...")
from ..unified_event_bus import UnifiedEventBus
_trace("Successfully imported UnifiedEventBus from menace_sandbox.unified_event_bus")
_trace("Importing Outcome, PathwayDB, PathwayRecord from menace_sandbox.neuroplasticity...")
from ..neuroplasticity import Outcome, PathwayDB, PathwayRecord
_trace("Successfully imported Outcome, PathwayDB, PathwayRecord from menace_sandbox.neuroplasticity")
_trace("Preparing lazy import for UnifiedLearningEngine...")


def get_unified_learning_engine_cls() -> "type[UnifiedLearningEngine]":
    _trace("Lazily importing UnifiedLearningEngine from menace_sandbox.unified_learning_engine...")
    from ..unified_learning_engine import UnifiedLearningEngine as _UnifiedLearningEngine

    _trace("Successfully lazy-imported UnifiedLearningEngine from menace_sandbox.unified_learning_engine")
    return _UnifiedLearningEngine
_trace("Importing ContextBuilder from vector_service.context_builder...")
from vector_service.context_builder import ContextBuilder
_trace("Successfully imported ContextBuilder from vector_service.context_builder")

_trace("Importing create_data_bot from menace_sandbox.shared.lazy_data_bot...")
from .lazy_data_bot import create_data_bot
_trace("Successfully imported create_data_bot from menace_sandbox.shared.lazy_data_bot")


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..action_planner import ActionPlanner
    from ..hierarchy_assessment_bot import HierarchyAssessmentBot
    from ..bot_planning_bot import BotPlanningBot, PlanningTask, BotPlan
    from ..bot_registry import BotRegistry
    from ..bot_creation_bot import BotCreationBot
    from ..communication_testing_bot import CommunicationTestingBot
    from ..discrepancy_detection_bot import DiscrepancyDetectionBot
    from .capital_management_bot import CapitalManagementBot
    from ..finance_router_bot import FinanceRouterBot
    from ..research_aggregator_bot import ResearchAggregatorBot, ResearchItem
    from ..information_synthesis_bot import InformationSynthesisBot
    from ..synthesis_models import SynthesisTask
    from ..implementation_optimiser_bot import ImplementationOptimiserBot
    from ..resource_allocation_bot import ResourceAllocationBot, AllocationDB
    from ..pre_execution_roi_bot import PreExecutionROIBot, BuildTask, ROIResult
    from ..task_validation_bot import TaskValidationBot
    from ..meta_genetic_algorithm_bot import MetaGeneticAlgorithmBot
    from ..unified_learning_engine import UnifiedLearningEngine
else:  # pragma: no cover - runtime fallback
    HierarchyAssessmentBot = Any  # type: ignore
    BotPlanningBot = Any  # type: ignore
    PlanningTask = Any  # type: ignore
    BotPlan = Any  # type: ignore
    BotRegistry = Any  # type: ignore
    BotCreationBot = Any  # type: ignore
    CommunicationTestingBot = Any  # type: ignore
    DiscrepancyDetectionBot = Any  # type: ignore
    CapitalManagementBot = Any  # type: ignore
    FinanceRouterBot = Any  # type: ignore
    ResearchAggregatorBot = Any  # type: ignore
    ResearchItem = Any  # type: ignore
    InformationSynthesisBot = Any  # type: ignore
    SynthesisTask = Any  # type: ignore
    ImplementationOptimiserBot = Any  # type: ignore
    ResourceAllocationBot = Any  # type: ignore
    AllocationDB = Any  # type: ignore
    PreExecutionROIBot = Any  # type: ignore
    BuildTask = Any  # type: ignore
    ROIResult = Any  # type: ignore
    TaskValidationBot = Any  # type: ignore
    MetaGeneticAlgorithmBot = Any  # type: ignore
    UnifiedLearningEngine = Any  # type: ignore


def _pre_execution_components() -> Tuple[type["PreExecutionROIBot"], type["BuildTask"], type["ROIResult"]]:
    """Return the ROI bot and associated dataclasses via a deferred import."""

    _trace("Importing PreExecutionROIBot, BuildTask, ROIResult from menace_sandbox.pre_execution_roi_bot...")
    from ..pre_execution_roi_bot import PreExecutionROIBot as _PreExecutionROIBot
    from ..pre_execution_roi_bot import BuildTask as _BuildTask
    from ..pre_execution_roi_bot import ROIResult as _ROIResult
    _trace("Successfully imported PreExecutionROIBot, BuildTask, ROIResult from menace_sandbox.pre_execution_roi_bot")

    return _PreExecutionROIBot, _BuildTask, _ROIResult


def _pipeline_helpers() -> Dict[str, Any]:
    """Return helper callables lazily imported from ``model_automation_pipeline``."""

    _trace("Importing pipeline helpers from menace_sandbox.model_automation_dependencies...")
    from ..model_automation_dependencies import (
        DB_ROUTER,
        _LazyAggregator,
        _build_default_hierarchy,
        _build_default_validator,
        _capital_manager_cls,
        _create_synthesis_task,
        _implementation_optimiser_cls,
        _load_research_aggregator,
        _make_research_item,
        _planning_components,
    )
    _trace("Successfully imported pipeline helpers from menace_sandbox.model_automation_dependencies")

    return {
        "db_router": DB_ROUTER,
        "LazyAggregator": _LazyAggregator,
        "build_default_hierarchy": _build_default_hierarchy,
        "build_default_validator": _build_default_validator,
        "capital_manager_cls": _capital_manager_cls,
        "create_synthesis_task": _create_synthesis_task,
        "implementation_optimiser_cls": _implementation_optimiser_cls,
        "load_research_aggregator": _load_research_aggregator,
        "make_research_item": _make_research_item,
        "planning_components": _planning_components,
    }


def _resource_allocation_components() -> Tuple[type["ResourceAllocationBot"], type["AllocationDB"]]:
    """Return resource allocation helpers via a deferred import."""

    from ..resource_allocation_bot import (
        ResourceAllocationBot as _ResourceAllocationBot,
        AllocationDB as _AllocationDB,
    )

    return _ResourceAllocationBot, _AllocationDB


def _finance_router_cls() -> type["FinanceRouterBot"]:
    """Return the finance router bot via a deferred import."""

    from ..finance_router_bot import FinanceRouterBot as _FinanceRouterBot

    return _FinanceRouterBot


def _communication_testing_bot_cls() -> type["CommunicationTestingBot"]:
    """Return the communication testing bot via a deferred import."""

    try:
        from ..communication_testing_bot import (
            CommunicationTestingBot as _CommunicationTestingBot,
        )
    except Exception as exc:  # pragma: no cover - degraded bootstrap
        _LOGGER.warning(
            "CommunicationTestingBot unavailable for ModelAutomationPipeline: %s",
            exc,
        )

        class _CommunicationTestingBotStub:
            """Minimal stub used when communication testing dependencies fail."""

            def __init__(self, *args, **kwargs) -> None:
                self.logger = logging.getLogger("CommTestingStub")

            def _result(self, name: str, passed: bool, details: str) -> SimpleNamespace:
                return SimpleNamespace(name=name, passed=passed, details=details)

            def functional_tests(self, modules: Iterable[str]) -> List[SimpleNamespace]:
                mods = list(modules)
                if mods:
                    self.logger.info(
                        "communication testing unavailable; skipping functional tests",
                        extra={"modules": mods},
                    )
                else:
                    self.logger.info(
                        "communication testing unavailable; no modules supplied",
                    )
                return []

            def integration_test(self, *args, **kwargs) -> SimpleNamespace:
                self.logger.info(
                    "communication testing unavailable; integration test skipped"
                )
                return self._result(
                    "integration", False, "communication testing unavailable"
                )

            def benchmark_mirror(self, *args, **kwargs):
                self.logger.info(
                    "communication testing unavailable; benchmark mirror skipped"
                )
                if pd is not None:
                    return pd.DataFrame()
                return _TaskTableFallback([])

            async def functional_tests_async(
                self, modules: Iterable[str]
            ) -> List[SimpleNamespace]:
                return []

            async def integration_test_async(self, *args, **kwargs) -> SimpleNamespace:
                return self.integration_test(*args, **kwargs)

            async def benchmark_mirror_async(self, *args, **kwargs):
                return self.benchmark_mirror(*args, **kwargs)

        return _CommunicationTestingBotStub

    return _CommunicationTestingBot


def _discrepancy_detection_bot_cls() -> type["DiscrepancyDetectionBot"]:
    """Return the discrepancy detection bot via a deferred import."""

    _trace("Lazily importing DiscrepancyDetectionBot from menace_sandbox.discrepancy_detection_bot...")
    try:
        from ..discrepancy_detection_bot import (
            DiscrepancyDetectionBot as _DiscrepancyDetectionBot,
        )
    except Exception as exc:  # pragma: no cover - degraded bootstrap
        _LOGGER.warning(
            "DiscrepancyDetectionBot unavailable for ModelAutomationPipeline: %s",
            exc,
        )

        class _DiscrepancyDetectionBotStub:
            """Fallback bot when discrepancy detection dependencies are missing."""

            def __init__(self, *args, **kwargs) -> None:
                self.logger = logging.getLogger("DiscrepancyDetectionStub")

            def scan(self) -> List[SimpleNamespace]:
                self.logger.info(
                    "discrepancy detection unavailable; returning empty findings"
                )
                return []

        return _DiscrepancyDetectionBotStub

    _trace("Successfully imported DiscrepancyDetectionBot from menace_sandbox.discrepancy_detection_bot")
    return _DiscrepancyDetectionBot


class _LazyHelperProxy:
    """Placeholder that defers helper construction until a manager is ready."""

    __slots__ = ("_pipeline_ref", "attr_name")

    def __init__(self, pipeline: "ModelAutomationPipeline", attr_name: str) -> None:
        self._pipeline_ref = weakref.ref(pipeline)
        self.attr_name = attr_name

    def resolve(self, *, force: bool = False) -> Any | None:
        pipeline = self._pipeline_ref()
        if pipeline is None:
            return None
        current = getattr(pipeline, self.attr_name, None)
        if current is not self:
            return current
        builder = pipeline._pending_manager_helpers.get(self.attr_name)
        if builder is None:
            return None
        manager = pipeline._effective_manager()
        if not force:
            if pipeline._should_defer_manager_helpers:
                return None
            if manager is None or pipeline._is_placeholder_manager(manager):
                return None
        with pipeline_context_scope(pipeline):
            return builder(manager=manager)

    def __getattr__(self, item: str) -> Any:
        helper = self.resolve()
        if helper is None:
            raise AttributeError(
                f"helper '{self.attr_name}' is not available until the manager attaches"
            )
        return getattr(helper, item)

    def __bool__(self) -> bool:  # pragma: no cover - simple delegation
        helper = self.resolve()
        return bool(helper)


class ModelAutomationPipeline:
    """Orchestrate bots to automate a model end-to-end."""

    _BOT_ATTRIBUTE_ORDER: tuple[str, ...] = (
        "aggregator",
        "synthesis_bot",
        "planner",
        "hierarchy",
        "predictor",
        "data_bot",
        "capital_manager",
        "roi_bot",
        "handoff",
        "optimiser",
        "efficiency_bot",
        "performance_bot",
        "comms_bot",
        "monitor_bot",
        "db_bot",
        "sentiment_bot",
        "query_bot",
        "memory_bot",
        "comms_test_bot",
        "discrepancy_bot",
        "finance_bot",
        "creation_bot",
        "meta_ga_bot",
        "offer_bot",
        "fallback_bot",
        "optimizer",
        "ai_counter_bot",
        "allocator",
        "diagnostic_manager",
        "idea_bank",
        "news_db",
        "reinvestment_bot",
        "spike_bot",
        "allocation_bot",
    )

    def __init__(
        self,
        aggregator: ResearchAggregatorBot | None = None,
        synthesis_bot: "InformationSynthesisBot" | None = None,
        validator: "TaskValidationBot" | None = None,
        planner: BotPlanningBot | None = None,
        hierarchy: HierarchyAssessmentBot | None = None,
        predictor: ResourcePredictionBot | None = None,
        data_bot: DataBotInterface | None = None,
        capital_manager: "CapitalManagementBot" | None = None,
        roi_bot: PreExecutionROIBot | None = None,
        handoff: TaskHandoffBot | None = None,
        optimiser: "ImplementationOptimiserBot" | None = None,
        workflow_db: WorkflowDB | None = None,
        funds: float = 100.0,
        roi_threshold: float = 0.0,
        efficiency_bot: EfficiencyBot | None = None,
        performance_bot: PerformanceAssessmentBot | None = None,
        comms_bot: "CommunicationMaintenanceBot | None" = None,
        monitor_bot: OperationalMonitoringBot | None = None,
        db_bot: CentralDatabaseBot | None = None,
        sentiment_bot: SentimentBot | None = None,
        query_bot: QueryBot | None = None,
        memory_bot: MemoryBot | None = None,
        comms_test_bot: CommunicationTestingBot | None = None,
        discrepancy_bot: DiscrepancyDetectionBot | None = None,
        finance_bot: FinanceRouterBot | None = None,
        creation_bot: "BotCreationBot" | None = None,
        meta_ga_bot: MetaGeneticAlgorithmBot | None = None,
        offer_bot: OfferTestingBot | None = None,
        fallback_bot: ResearchFallbackBot | None = None,
        optimizer: ResourceAllocationOptimizer | None = None,
        ai_counter_bot: AICounterBot | None = None,
        allocator: "DynamicResourceAllocator" | None = None,
        diagnostic_manager: DiagnosticManager | None = None,
        idea_bank: KeywordBank | None = None,
        news_db: NewsDB | None = None,
        reinvestment_bot: AutoReinvestmentBot | None = None,
        spike_bot: RevenueSpikeEvaluatorBot | None = None,
        allocation_bot: CapitalAllocationBot | None = None,
        db_router: DBRouter | None = None,
        pathway_db: PathwayDB | None = None,
        myelination_threshold: float = 1.0,
        learning_engine: UnifiedLearningEngine | None = None,
        action_planner: "ActionPlanner" | None = None,
        *,
        event_bus: UnifiedEventBus | None = None,
        bot_registry: "BotRegistry | None" = None,
        context_builder: ContextBuilder,
        validator_factory: Callable[[], "TaskValidationBot"] | None = None,
        manager: "SelfCodingManager | None" = None,
    ) -> None:
        helpers = _pipeline_helpers()
        LazyAggregator = helpers["LazyAggregator"]
        load_research_aggregator = helpers["load_research_aggregator"]
        build_default_hierarchy = helpers["build_default_hierarchy"]
        build_default_validator = helpers["build_default_validator"]
        capital_manager_cls_factory = helpers["capital_manager_cls"]
        implementation_optimiser_cls = helpers["implementation_optimiser_cls"]
        planning_components = helpers["planning_components"]

        self._create_synthesis_task = helpers["create_synthesis_task"]
        self._make_research_item = helpers["make_research_item"]
        self._build_default_validator = build_default_validator
        self.db_router = db_router or helpers["db_router"]

        self._pre_execution_cache: Tuple[
            type["PreExecutionROIBot"],
            type["BuildTask"],
            type["ROIResult"],
        ] | None = None
        self._build_task_cls: type["BuildTask"] | None = None
        self._roi_result_cls: type["ROIResult"] | None = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            "ModelAutomationPipeline MRO: %s",
            [cls.__name__ for cls in type(self).__mro__],
        )
        if context_builder is None:
            raise ValueError("context_builder is required")
        self.context_builder = context_builder
        initial_manager = normalise_manager_arg(manager, None, fallback=None)
        self._placeholder_context_token = None
        placeholder_candidate = (
            initial_manager if self._is_placeholder_manager(initial_manager) else None
        )
        if placeholder_candidate is None:
            placeholder_candidate = self._bootstrap_manager_candidate()
        if placeholder_candidate is not None and self._is_placeholder_manager(
            placeholder_candidate
        ):
            try:
                self._placeholder_context_token = _seed_existing_pipeline_placeholder(
                    self, placeholder_candidate
                )
            except Exception:  # pragma: no cover - placeholder seeding best effort
                self._placeholder_context_token = None
            if initial_manager is None:
                initial_manager = placeholder_candidate
        self._pending_manager_helpers: dict[str, Callable[..., Any]] = {}
        self._lazy_helper_attrs: set[str] = set()
        self._bot_attribute_order: tuple[str, ...] = self._BOT_ATTRIBUTE_ORDER
        self._registered_bot_attrs: set[str] = set()
        self._bots: list[Any] = []
        self._should_defer_manager_helpers = self._should_defer_for_manager(
            initial_manager
        )
        self._manager: "SelfCodingManager | None" = None
        self.manager = initial_manager

        try:
            self.context_builder.refresh_db_weights()
        except Exception as exc:
            self.logger.exception("context builder refresh failed: %s", exc)
        if aggregator is None:
            aggregator = LazyAggregator(
                lambda: load_research_aggregator(self.context_builder)
            )
        self.aggregator = aggregator
        self.workflow_db = workflow_db or WorkflowDB(event_bus=event_bus)
        if synthesis_bot is None:
            from ..information_synthesis_bot import InformationSynthesisBot

            synthesis_bot = self._call_with_manager(
                InformationSynthesisBot,
                aggregator=self.aggregator,
                workflow_db=self.workflow_db,
                event_bus=event_bus,
                context_builder=self.context_builder,
            )
        self.synthesis_bot = self._bind_manager(synthesis_bot)
        self._validator_factory: Callable[[], "TaskValidationBot"] | None = validator_factory
        self._validator: "TaskValidationBot | None" = validator
        self._validator_wrapped = False
        self._bots_primed = False
        planner_cls, planning_task_cls, _ = planning_components()
        self._planning_task_cls = planning_task_cls
        self.planner = planner or self._call_with_manager(planner_cls)
        self.planner = self._bind_manager(self.planner)
        hierarchy = hierarchy or build_default_hierarchy()
        self.hierarchy = self._bind_manager(hierarchy)
        data_bot = data_bot or create_data_bot(self.logger)
        self.data_bot = self._bind_manager(data_bot)
        if capital_manager is None:
            capital_manager_cls = capital_manager_cls_factory()
            capital_manager = cast(
                "CapitalManagementBot",
                self._call_with_manager(cast(Callable[..., Any], capital_manager_cls)),
            )
        self.capital_manager = self._bind_manager(capital_manager)

        if predictor:
            self.predictor = self._bind_manager(predictor)
            if getattr(self.predictor, "data_bot", None) is None:
                self.predictor.data_bot = self.data_bot
            if getattr(self.predictor, "capital_bot", None) is None:
                self.predictor.capital_bot = self.capital_manager
        else:
            self.predictor = self._build_or_defer_predictor()

        self.handoff = self._build_or_defer_handoff(handoff, event_bus=event_bus)
        pre_bot_cls, build_task_cls, roi_result_cls = self._ensure_pre_execution_components()
        self._build_task_cls = build_task_cls
        self._roi_result_cls = roi_result_cls
        if roi_bot:
            self.roi_bot = self._bind_manager(roi_bot)
            if getattr(self.roi_bot, "handoff", None) is None:
                self.roi_bot.handoff = self.handoff
        else:
            self.roi_bot = self._call_with_manager(pre_bot_cls, handoff=self.handoff)
            self.roi_bot = self._bind_manager(self.roi_bot)
        if optimiser is None:
            optimiser_cls = implementation_optimiser_cls()
            optimiser = self._call_with_manager(
                optimiser_cls, context_builder=self.context_builder
            )
        self.optimiser = self._bind_manager(optimiser)
        self.funds = funds
        self.roi_threshold = roi_threshold

        self.efficiency_bot = efficiency_bot or self._call_with_manager(EfficiencyBot)
        self.efficiency_bot = self._bind_manager(self.efficiency_bot)
        self.performance_bot = performance_bot or self._call_with_manager(
            PerformanceAssessmentBot
        )
        self.performance_bot = self._bind_manager(self.performance_bot)
        self.comms_bot = self._build_or_defer_comms_bot(comms_bot)
        self.monitor_bot = self._bind_manager(
            monitor_bot or OperationalMonitoringBot()
        )
        self.db_bot = self._build_or_defer_db_bot(db_bot)
        self.sentiment_bot = sentiment_bot or self._call_with_manager(SentimentBot)
        self.sentiment_bot = self._bind_manager(self.sentiment_bot)
        self.query_bot = query_bot or self._call_with_manager(
            QueryBot, context_builder=self.context_builder
        )
        self.query_bot = self._bind_manager(self.query_bot)
        self.memory_bot = memory_bot or self._call_with_manager(MemoryBot)
        self.memory_bot = self._bind_manager(self.memory_bot)
        if comms_test_bot is None:
            comms_test_bot_cls = _communication_testing_bot_cls()
            comms_test_bot = self._call_with_manager(comms_test_bot_cls)
        self.comms_test_bot = self._bind_manager(comms_test_bot)
        if discrepancy_bot is None:
            discrepancy_cls = _discrepancy_detection_bot_cls()
            discrepancy_bot = self._call_with_manager(discrepancy_cls)
        self.discrepancy_bot = self._bind_manager(discrepancy_bot)
        if finance_bot is None:
            try:
                finance_cls = _finance_router_cls()
            except Exception as exc:  # pragma: no cover - degraded bootstrap
                self.logger.warning(
                    "FinanceRouterBot unavailable for ModelAutomationPipeline: %s",
                    exc,
                )
                finance_bot = None
            else:
                try:
                    finance_bot = self._call_with_manager(finance_cls)
                except Exception as exc:  # pragma: no cover - degraded bootstrap
                    self.logger.warning(
                        "FinanceRouterBot initialisation failed for ModelAutomationPipeline: %s",
                        exc,
                    )
                    finance_bot = None
        self.finance_bot = self._bind_manager(finance_bot)
        if creation_bot is None:
            _bot_creation_cls: type["BotCreationBot"] | None
            try:
                from ..bot_creation_bot import (  # type: ignore
                    BotCreationBot as _BotCreationBot,
                )
            except Exception as exc:  # pragma: no cover - degraded bootstrap
                self.logger.warning(
                    "BotCreationBot unavailable for ModelAutomationPipeline: %s",
                    exc,
                )
                _bot_creation_cls = None
            else:
                _bot_creation_cls = _BotCreationBot

            if _bot_creation_cls is not None:
                try:
                    creation_bot = self._call_with_manager(
                        _bot_creation_cls, context_builder=self.context_builder
                    )
                except Exception as exc:  # pragma: no cover - degraded bootstrap
                    self.logger.warning(
                        "BotCreationBot initialisation failed for ModelAutomationPipeline: %s",
                        exc,
                    )
                    creation_bot = None

        self.creation_bot = self._bind_manager(creation_bot)
        if meta_ga_bot is None:
            meta_cls = get_meta_genetic_algorithm_bot_cls()
            meta_ga_bot = self._call_with_manager(meta_cls)
        self.meta_ga_bot = self._bind_manager(meta_ga_bot)
        self.offer_bot = offer_bot or self._call_with_manager(OfferTestingBot)
        self.offer_bot = self._bind_manager(self.offer_bot)
        self.fallback_bot = fallback_bot or self._call_with_manager(ResearchFallbackBot)
        self.fallback_bot = self._bind_manager(self.fallback_bot)
        if optimizer is None:
            ResourceAllocationOptimizerCls = get_resource_allocation_optimizer_cls()
            optimizer = self._call_with_manager(ResourceAllocationOptimizerCls)
        self.optimizer = self._bind_manager(optimizer)
        self.ai_counter_bot = ai_counter_bot or self._call_with_manager(AICounterBot)
        self.ai_counter_bot = self._bind_manager(self.ai_counter_bot)
        if allocator is None:
            alloc_bot_cls, alloc_db_cls = _resource_allocation_components()
            DynamicResourceAllocatorCls = get_dynamic_resource_allocator_cls()
            alloc_bot = self._call_with_manager(
                alloc_bot_cls, alloc_db_cls(), context_builder=self.context_builder
            )
            allocator = self._call_with_manager(
                DynamicResourceAllocatorCls,
                alloc_bot=alloc_bot,
                context_builder=self.context_builder,
            )
        self.allocator = self._bind_manager(allocator)
        def _diagnostic_builder(*, manager: Any) -> DiagnosticManager | None:
            DiagnosticManagerCls = get_diagnostic_manager_cls()
            return self._call_with_manager(
                DiagnosticManagerCls,
                context_builder=self.context_builder,
                manager=manager,
            )

        self.diagnostic_manager = self._lazy_helper(
            "diagnostic_manager",
            _diagnostic_builder,
            existing=diagnostic_manager,
        )
        self.idea_bank = idea_bank or KeywordBank()
        self.news_db = news_db or NewsDB()

        def _reinvest_builder(*, manager: Any) -> AutoReinvestmentBot | None:
            return self._call_with_manager(AutoReinvestmentBot, manager=manager)

        self.reinvestment_bot = self._lazy_helper(
            "reinvestment_bot",
            _reinvest_builder,
            existing=reinvestment_bot,
        )

        revenue_db = RevenueEventsDB()

        def _spike_builder(*, manager: Any) -> RevenueSpikeEvaluatorBot | None:
            return self._call_with_manager(
                RevenueSpikeEvaluatorBot,
                revenue_db,
                manager=manager,
            )

        self.spike_bot = self._lazy_helper(
            "spike_bot",
            _spike_builder,
            existing=spike_bot,
        )

        def _allocation_builder(*, manager: Any) -> CapitalAllocationBot | None:
            return self._call_with_manager(CapitalAllocationBot, manager=manager)

        self.allocation_bot = self._lazy_helper(
            "allocation_bot",
            _allocation_builder,
            existing=allocation_bot,
        )
        if bot_registry is None:
            from ..bot_registry import BotRegistry as _BotRegistry

            bot_registry = _BotRegistry()
        self.bot_registry = bot_registry
        self.event_bus = event_bus
        self.pathway_db = pathway_db
        self.myelination_threshold = myelination_threshold
        self.learning_engine = learning_engine
        self.action_planner = action_planner
        if self.event_bus:
            try:
                self.db_router.memory_mgr.subscribe(
                    lambda entry: self.event_bus.publish("memory:broadcast", entry)
                )
            except Exception as exc:
                self.logger.exception("memory broadcast hook failed: %s", exc)
        self._initialize_bot_registry()
        if self._validator is not None and not self._validator_wrapped:
            self._register_validator(self._validator)
        self._attach_information_synthesis_manager()

    # ------------------------------------------------------------------

    def _build_or_defer_predictor(self) -> ResourcePredictionBot | None:
        def _builder(*, manager: Any) -> ResourcePredictionBot | None:
            return self._call_with_manager(
                ResourcePredictionBot,
                data_bot=self.data_bot,
                capital_bot=self.capital_manager,
                manager=manager,
            )

        predictor = self._build_or_defer_helper("predictor", None, _builder)
        if predictor is not None:
            if getattr(predictor, "data_bot", None) is None:
                predictor.data_bot = self.data_bot
            if getattr(predictor, "capital_bot", None) is None:
                predictor.capital_bot = self.capital_manager
        return predictor

    def _build_or_defer_handoff(
        self, handoff: TaskHandoffBot | None, *, event_bus: UnifiedEventBus | None
    ) -> TaskHandoffBot | None:
        def _builder(*, manager: Any) -> TaskHandoffBot | None:
            return self._call_with_manager(
                TaskHandoffBot, event_bus=event_bus, manager=manager
            )

        return self._build_or_defer_helper("handoff", handoff, _builder)

    def _build_or_defer_comms_bot(
        self, comms_bot: "CommunicationMaintenanceBot | None"
    ) -> "CommunicationMaintenanceBot | None":
        def _builder(*, manager: Any) -> "CommunicationMaintenanceBot | None":
            comms_bot_cls = get_communication_maintenance_bot_cls()
            try:
                return comms_bot_cls(
                    context_builder=self.context_builder,
                    manager=manager,
                )
            except TypeError as exc:
                if "context_builder" not in str(exc):
                    raise
                _LOGGER.debug(
                    "CommunicationMaintenanceBot rejected context_builder argument; "
                    "falling back to default constructor",
                    exc_info=True,
                )
                return self._call_with_manager(comms_bot_cls, manager=manager)

        return self._lazy_helper("comms_bot", _builder, existing=comms_bot)

    def _build_or_defer_db_bot(self, db_bot: CentralDatabaseBot | None) -> CentralDatabaseBot | None:
        def _builder(*, manager: Any) -> CentralDatabaseBot | None:
            return self._call_with_manager(
                CentralDatabaseBot,
                db_router=self.db_router,
                manager=manager,
            )

        return self._build_or_defer_helper("db_bot", db_bot, _builder)

    def _build_or_defer_helper(
        self,
        attr_name: str,
        existing: Any | None,
        builder: Callable[..., Any],
    ) -> Any | None:
        if existing is not None:
            helper = self._bind_manager(existing)
            setattr(self, attr_name, helper)
            return helper
        manager = self._effective_manager()
        defer = self._should_defer_manager_helpers or manager is None
        if not defer and self._is_placeholder_manager(manager):
            defer = True
        if defer:
            self._pending_manager_helpers[attr_name] = builder
            setattr(self, attr_name, None)
            return None
        with pipeline_context_scope(self):
            helper = builder(manager=manager)
        helper = self._bind_manager(helper)
        setattr(self, attr_name, helper)
        return helper

    def _lazy_helper(
        self,
        attr_name: str,
        builder: Callable[..., Any],
        *,
        existing: Any | None = None,
    ) -> Any | None:
        if existing is not None:
            helper = self._bind_manager(existing)
            setattr(self, attr_name, helper)
            return helper
        manager = self._effective_manager()
        defer = self._should_defer_manager_helpers or manager is None
        if not defer and self._is_placeholder_manager(manager):
            defer = True
        if defer:
            self._pending_manager_helpers[attr_name] = builder
            proxy = _LazyHelperProxy(self, attr_name)
            self._lazy_helper_attrs.add(attr_name)
            setattr(self, attr_name, proxy)
            return proxy
        with pipeline_context_scope(self):
            helper = builder(manager=manager)
        helper = self._bind_manager(helper)
        setattr(self, attr_name, helper)
        return helper

    def _attach_helper(self, attr_name: str, helper: Any, *, register: bool) -> Any:
        setattr(self, attr_name, helper)
        self._lazy_helper_attrs.discard(attr_name)
        if register:
            self._register_bot_attr(attr_name)
        return helper

    def _resolve_lazy_helper_attr(
        self,
        attr_name: str,
        *,
        force: bool = False,
        register: bool = True,
    ) -> Any | None:
        candidate = getattr(self, attr_name, None)
        if not isinstance(candidate, _LazyHelperProxy):
            return candidate
        helper = candidate.resolve(force=force)
        if helper is None:
            return None
        helper = self._bind_manager(helper)
        return self._attach_helper(attr_name, helper, register=register)

    def _resolve_helper_candidate(self, candidate: Any) -> Any:
        if isinstance(candidate, _LazyHelperProxy):
            return self._resolve_lazy_helper_attr(candidate.attr_name)
        return candidate

    def _resolve_all_lazy_helpers(self, *, force: bool = False) -> None:
        for attr_name in list(self._lazy_helper_attrs):
            helper = self._resolve_lazy_helper_attr(attr_name, force=force)
            if helper is None and force:
                self._lazy_helper_attrs.discard(attr_name)

    def _effective_manager(self, manager: Any | None = None) -> Any | None:
        return normalise_manager_arg(manager, None, fallback=self._manager)

    def _is_placeholder_manager(self, manager: Any | None) -> bool:
        if manager is None:
            return False
        try:
            return _is_bootstrap_placeholder(manager)
        except Exception:  # pragma: no cover - defensive guard
            return False

    def _should_defer_for_manager(self, manager: Any | None) -> bool:
        return manager is None or self._is_placeholder_manager(manager)

    def _bootstrap_manager_candidate(self) -> Any | None:
        candidate: Any | None = None
        if MANAGER_CONTEXT is not None:
            try:
                candidate = MANAGER_CONTEXT.get()
            except LookupError:  # pragma: no cover - context var not initialised
                candidate = None
        if candidate is None:
            candidate = getattr(_BOOTSTRAP_STATE, "sentinel_manager", None)
        return candidate

    def _call_with_manager(
        self,
        factory: Callable[..., Any],
        *args: Any,
        manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        with pipeline_context_scope(self):
            if factory is None:
                raise ValueError("factory is required")
            effective_manager = self._effective_manager(manager)
            if effective_manager is None:
                effective_manager = self._bootstrap_manager_candidate()
            if effective_manager is None:
                return factory(*args, **kwargs)
            try:
                return factory(*args, **kwargs, manager=effective_manager)
            except TypeError as exc:
                if "manager" not in str(exc):
                    raise
                _LOGGER.debug(
                    "%s rejected manager argument; continuing without manager",
                    getattr(factory, "__name__", repr(factory)),
                    exc_info=True,
                )
                return factory(*args, **kwargs)

    def _bind_manager(self, candidate: Any) -> Any:
        if candidate is None:
            return candidate
        manager = self._effective_manager()
        if manager is None:
            return candidate
        try:
            current = getattr(candidate, "manager", None)
        except Exception:  # pragma: no cover - defensive attribute access
            return candidate
        if current is None:
            try:
                setattr(candidate, "manager", manager)
            except Exception:  # pragma: no cover - best effort binding
                _LOGGER.debug(
                    "manager binding failed for %s during pipeline bootstrap",
                    candidate,
                    exc_info=True,
                )
        return candidate

    def _activate_deferred_helpers(self) -> None:
        if not self._pending_manager_helpers:
            return
        manager = self._effective_manager()
        if (
            manager is None
            or self._should_defer_manager_helpers
            or self._is_placeholder_manager(manager)
        ):
            return
        pending = list(self._pending_manager_helpers.items())
        self._pending_manager_helpers.clear()
        for attr_name, builder in pending:
            with pipeline_context_scope(self):
                helper = builder(manager=manager)
            helper = self._bind_manager(helper)
            self._attach_helper(attr_name, helper, register=True)

    def _register_bot_attr(self, attr_name: str) -> None:
        if attr_name in self._registered_bot_attrs:
            return
        bot = getattr(self, attr_name, None)
        if bot is None:
            return
        if isinstance(bot, _LazyHelperProxy):
            return
        bot = self._bind_manager(bot)
        setattr(self, attr_name, bot)
        insert_at = 0
        for name in self._bot_attribute_order:
            if name == attr_name:
                break
            if name in self._registered_bot_attrs:
                insert_at += 1
        wrap_bot_methods(bot, self.db_router, self.bot_registry)
        self._bots.insert(insert_at, bot)
        self._registered_bot_attrs.add(attr_name)

    def _initialize_bot_registry(self) -> None:
        self._bots = []
        self._registered_bot_attrs = set()
        for attr in self._bot_attribute_order:
            self._register_bot_attr(attr)

    @property
    def manager(self) -> "SelfCodingManager | None":
        return self._manager

    @manager.setter
    def manager(self, value: "SelfCodingManager | None") -> None:
        self._manager = value
        self._should_defer_manager_helpers = self._should_defer_for_manager(value)
        if not self._should_defer_manager_helpers:
            self._activate_deferred_helpers()

    def finalize_helpers(self, manager: Any | None) -> None:
        effective = self._effective_manager(manager)
        if effective is None or self._is_placeholder_manager(effective):
            return
        if self._manager is not effective:
            try:
                self.manager = effective
            except Exception:  # pragma: no cover - fallback when setter fails
                self._manager = effective
        self._should_defer_manager_helpers = False
        self._activate_deferred_helpers()
        self._resolve_all_lazy_helpers(force=True)
        for attr_name in self._bot_attribute_order:
            bot = getattr(self, attr_name, None)
            if bot is None or isinstance(bot, _LazyHelperProxy):
                continue
            self._bind_manager(bot)

    def _ensure_pre_execution_components(
        self,
    ) -> Tuple[type["PreExecutionROIBot"], type["BuildTask"], type["ROIResult"]]:
        """Import ROI components lazily to avoid circular dependencies."""

        if self._pre_execution_cache is None:
            self._pre_execution_cache = _pre_execution_components()
        return self._pre_execution_cache

    def _build_task_type(self) -> type["BuildTask"]:
        if self._build_task_cls is None:
            _, build_task_cls, _ = self._ensure_pre_execution_components()
            self._build_task_cls = build_task_cls
        return self._build_task_cls

    def _roi_result_type(self) -> type["ROIResult"]:
        if self._roi_result_cls is None:
            _, _, roi_result_cls = self._ensure_pre_execution_components()
            self._roi_result_cls = roi_result_cls
        return self._roi_result_cls

    # ------------------------------------------------------------------

    def _attach_information_synthesis_manager(self) -> None:
        """Register the self-coding manager for ``InformationSynthesisBot`` if available."""

        registry = self.bot_registry
        if registry is None:
            return

        try:
            node = registry.graph.nodes.get("InformationSynthesisBot", {})
        except Exception:  # pragma: no cover - defensive lookup
            self.logger.debug(
                "unable to inspect registry nodes while attaching InformationSynthesisBot manager",
                exc_info=True,
            )
            return

        existing = node.get("selfcoding_manager") or node.get("manager")
        if existing is not None:
            # Manager already registered – nothing to do.
            return

        manager = getattr(self.synthesis_bot, "manager", None)
        if manager is None:
            self.logger.debug(
                "InformationSynthesisBot manager unavailable during pipeline bootstrap",
            )
            return

        try:  # pragma: no cover - optional dependency guard
            from ..self_coding_manager import SelfCodingManager as _RealSelfCodingManager
        except Exception:  # pragma: no cover - runtime environments without self-coding
            _RealSelfCodingManager = None  # type: ignore[assignment]

        if _RealSelfCodingManager is None or not isinstance(manager, _RealSelfCodingManager):
            self.logger.debug(
                "InformationSynthesisBot manager is not an active SelfCodingManager instance",
            )
            return

        data_bot = getattr(self.synthesis_bot, "data_bot", None) or getattr(self, "data_bot", None)
        if data_bot is None:
            self.logger.debug(
                "InformationSynthesisBot data bot helper missing; skipping manager attachment",
            )
            return

        try:
            registry.register_bot(
                "InformationSynthesisBot",
                manager=manager,
                data_bot=data_bot,
                is_coding_bot=True,
            )
        except Exception:  # pragma: no cover - registry failures are logged for diagnosis
            self.logger.exception(
                "failed to attach SelfCodingManager for InformationSynthesisBot",
            )
        else:
            # Ensure downstream helpers can discover the relationship immediately.
            setattr(self.synthesis_bot, "bot_registry", registry)
            setattr(self.synthesis_bot, "data_bot", data_bot)
            self.logger.info(
                "SelfCodingManager attached to InformationSynthesisBot",
            )

    @property
    def validator(self) -> "TaskValidationBot":
        """Return the task validator, instantiating it lazily when required."""

        return self._ensure_validator()

    @validator.setter
    def validator(self, value: "TaskValidationBot | None") -> None:
        """Inject a validator instance and register it with pipeline helpers."""

        if value is None:
            bots = getattr(self, "_bots", None)
            if self._validator_wrapped and bots and self._validator in bots:
                try:
                    bots.remove(self._validator)
                except ValueError:  # pragma: no cover - best effort cleanup
                    pass
            self._validator = None
            self._validator_wrapped = False
            return
        self._validator = value
        self._register_validator(value)

    # ------------------------------------------------------------------

    def _ensure_validator(self) -> "TaskValidationBot":
        """Create and register the validator if it has not been resolved yet."""

        if self._validator is None:
            factory = self._validator_factory or self._build_default_validator
            validator = factory()
            self._validator = validator
            self._register_validator(validator)
        return self._validator

    # ------------------------------------------------------------------

    def _register_validator(self, validator: "TaskValidationBot") -> None:
        """Wrap and register *validator* without triggering circular imports."""

        if self._validator_wrapped:
            return
        # Preserve historical ordering by inserting the validator after the
        # synthesis bot once it becomes available.
        self._bots.insert(2, validator)
        wrap_bot_methods(validator, self.db_router, self.bot_registry)
        self._validator_wrapped = True
        if self._bots_primed:
            prime = getattr(validator, "prime", None)
            if callable(prime):
                try:
                    prime()
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.exception("validator prime failed: %s", exc)

    # ------------------------------------------------------------------

    def _prime_bots(self) -> None:
        if self._bots_primed:
            return
        for bot in self._bots:
            prime = getattr(bot, "prime", None)
            if callable(prime):
                try:
                    prime()
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.exception("bot prime failed: %s", exc)
        self._bots_primed = True

    # ------------------------------------------------------------------

    def _gather_research(self, model: str, energy: int) -> Iterable[ResearchItem]:
        try:
            return self.aggregator.collect(model, energy=energy)
        except Exception as exc:
            self.logger.exception("aggregator collect failed: %s", exc)
            return []

    def _flagged_workflows(self, items: Iterable[ResearchItem]) -> List[str]:
        flagged = []
        try:
            flagged = [
                item.title
                for item in items
                if getattr(item, "category", "") == "workflow"
            ]
        except Exception as exc:
            self.logger.exception("workflow flag extraction failed: %s", exc)
        return flagged

    def _reuse_workflows(self, workflows: Iterable[str]) -> List[str]:
        reuse = []
        for wf in workflows:
            try:
                if self.db_router.workflow_mgr.exists(wf):
                    reuse.append(wf)
            except Exception as exc:
                self.logger.exception("workflow lookup failed: %s", exc)
        return reuse

    def _items_to_tasks(self, items: Iterable[ResearchItem]) -> List["SynthesisTask"]:
        tasks: List["SynthesisTask"] = []
        for item in items:
            try:
                tasks.append(
                    self._create_synthesis_task(
                        description=item.content,
                        urgency=getattr(item, "urgency", 1),
                        complexity=getattr(item, "complexity", 1),
                        category=getattr(item, "category", "general"),
                    )
                )
            except Exception as exc:
                self.logger.exception("task creation failed: %s", exc)
        return tasks

    def _build_task_table(self, tasks: Iterable["SynthesisTask"]):
        """Return a tabular structure accepted by ``synthesis_bot.create_tasks``."""

        task_list = list(tasks)
        rows = [
            {
                "id": idx,
                "name": getattr(task, "description", ""),
                "content": getattr(task, "description", ""),
            }
            for idx, task in enumerate(task_list)
        ]
        if pd is not None:
            try:
                return pd.DataFrame(rows)
            except Exception as exc:  # pragma: no cover - pandas runtime failure
                self.logger.exception("DataFrame creation failed: %s", exc)
        return _TaskTableFallback(rows)

    def _validate_tasks(self, tasks: Iterable["SynthesisTask"]):
        validator = self._ensure_validator()
        validated = []
        for task in tasks:
            if not hasattr(task, "description"):
                if isinstance(task, dict):
                    desc = (
                        task.get("description")
                        or task.get("content")
                        or task.get("name")
                        or ""
                    )
                    try:
                        urgency = int(task.get("urgency", 1))
                    except Exception:
                        urgency = 1
                    try:
                        complexity = int(task.get("complexity", 1))
                    except Exception:
                        complexity = 1
                    category = str(task.get("category", "general"))
                    task = self._create_synthesis_task(
                        description=str(desc),
                        urgency=urgency,
                        complexity=complexity,
                        category=category,
                    )
                else:
                    self.logger.debug(
                        "Skipping task without description attribute: %r", task
                    )
                    continue
            try:
                if validator.validate(task):
                    validated.append(task)
            except Exception as exc:
                self.logger.exception("task validation failed: %s", exc)
        return validated

    def _plan_bots(
        self, tasks: Iterable["SynthesisTask"], *, trust_weight: float
    ) -> List["BotPlan"]:
        planner_tasks = []
        for task in tasks:
            try:
                planner_tasks.append(self._planning_task_cls(task.description))
            except Exception as exc:
                self.logger.exception("planning task creation failed: %s", exc)
        try:
            return self.planner.plan(planner_tasks, weight=trust_weight)
        except Exception as exc:
            self.logger.exception("planner failed: %s", exc)
            return []

    def _validate_plan(self, plans: Iterable["BotPlan"]) -> List["BotPlan"]:
        validator = self._ensure_validator()
        validated = []
        for plan in plans:
            try:
                if validator.validate_plan(plan):
                    validated.append(plan)
            except Exception as exc:
                self.logger.exception("plan validation failed: %s", exc)
        return validated

    def _assess_hierarchy(self, plans: Iterable["BotPlan"]) -> None:
        for plan in plans:
            try:
                self.hierarchy.assess(plan)
            except Exception as exc:
                self.logger.exception("hierarchy assessment failed: %s", exc)

    def _predict_plan_success(self, name: str) -> float:
        try:
            return float(self.predictor.predict(name))
        except Exception as exc:
            self.logger.exception("prediction failed for %s: %s", name, exc)
            return 0.0

    def _predict_resources(self, plans: Iterable["BotPlan"]) -> Dict[str, ResourceMetrics]:
        predictions: Dict[str, ResourceMetrics] = {}
        for plan in plans:
            try:
                predictions[plan.name] = self.predictor.resources(plan.name)
            except Exception as exc:
                self.logger.exception("resource prediction failed: %s", exc)
        return predictions

    def _roi(self, model: str, tasks: Iterable["BuildTask"]) -> "ROIResult":
        try:
            return self.roi_bot.evaluate(model, list(tasks))
        except Exception as exc:
            self.logger.exception("ROI evaluation failed: %s", exc)
            roi_result_cls = self._roi_result_type()
            try:
                fallback = roi_result_cls(
                    income=0.0,
                    cost=float("inf"),
                    time=0.0,
                    roi=-1.0,
                    margin=0.0,
                )
            except Exception:
                try:
                    fallback = roi_result_cls(roi=-1.0, cost=float("inf"))  # type: ignore[arg-type]
                except Exception:
                    fallback = cast(
                        "ROIResult",
                        SimpleNamespace(
                            income=0.0,
                            cost=float("inf"),
                            time=0.0,
                            roi=-1.0,
                            margin=0.0,
                            benefit=0.0,
                        ),
                    )
            if hasattr(fallback, "benefit"):
                setattr(fallback, "benefit", getattr(fallback, "benefit", 0.0))
            return fallback
        return result

    def _run_support_bots(self, model: str, **kwargs: Any) -> None:
        bots = [
            (self.efficiency_bot, "evaluate", (model,), kwargs),
            (self.performance_bot, "assess", (model,), kwargs),
            (self.comms_bot, "update", (model,), kwargs),
            (self.monitor_bot, "monitor", (model,), kwargs),
            (self.db_bot, "log", (Proposal(model=model, details={})), {}),
            (self.sentiment_bot, "analyse", (model,), {}),
            (self.query_bot, "index", (model,), {}),
            (self.memory_bot, "store", (model,), {}),
            (self.comms_test_bot, "probe", (model,), {}),
            (self.discrepancy_bot, "detect", (model,), {}),
            (self.finance_bot, "route", (model,), {}),
            (self.creation_bot, "bootstrap", (model,), {}),
            (self.meta_ga_bot, "evolve", (1,), {}),
            (self.offer_bot, "test", (model,), {}),
            (self.fallback_bot, "fallback", (model,), {}),
            (self.optimizer, "optimise", (model,), {}),
            (self.ai_counter_bot, "increment", (model,), {}),
            (self.allocator, "allocate", (model,), {}),
            (self.diagnostic_manager, "diagnose", (model,), {}),
            (self.idea_bank, "tag", (model,), {}),
            (self.news_db, "record", (model,), {}),
            (self.reinvestment_bot, "reinvest", ()),
            (self.spike_bot, "detect_spike", (model,)),
            (self.allocation_bot, "rebalance", (model, 0.0)),
        ]

        def call(entry: tuple) -> None:
            bot = self._resolve_helper_candidate(entry[0])
            if bot is None:
                return
            method = entry[1]
            args = entry[2] if len(entry) > 2 else ()
            kwargs_local = entry[3] if len(entry) > 3 else {}
            try:
                getattr(bot, method)(*args, **kwargs_local)
            except Exception as exc:
                self.logger.exception(
                    "support bot %s.%s failed: %s",
                    bot.__class__.__name__,
                    method,
                    exc,
                )

        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = [ex.submit(call, e) for e in bots]
            for f in as_completed(futures):
                f.result()

    def _preload_memory(self, model: str, limit: int = 5) -> None:
        """Load recent memory entries tagged with the model."""
        entries = []
        try:
            entries = list(self.memory_bot.search(model, limit=limit))
        except Exception as exc:
            self.logger.exception("memory bot search failed: %s", exc)
        try:
            entries += list(self.db_router.memory_mgr.search_by_tag(model))
        except Exception as exc:
            self.logger.exception("db search failed: %s", exc)
        for ent in entries:
            text = getattr(ent, "text", getattr(ent, "data", ""))
            try:
                ts = float(getattr(ent, "ts", 0.0))
            except Exception as exc:
                self.logger.exception("invalid timestamp in entry: %s", exc)
                ts = 0.0
            try:
                self.aggregator.memory.add(
                    self._make_research_item(
                        topic=model,
                        content=text,
                        timestamp=ts,
                        title=model,
                        tags=[model],
                        category="memory",
                    ),
                    layer="short",
                )
            except Exception as exc:
                self.logger.exception("memory preload add failed: %s", exc)

    def run(self, model: str, energy: int = 1) -> "AutomationResult":
        from ..model_automation_pipeline import AutomationResult

        high = False
        trust_weight = 1.0
        if self.pathway_db:
            sim = self.pathway_db.similar_actions(f"pipeline:{model}", limit=1)
            if sim:
                score = sim[0][1]
                trust_weight = max(1.0, float(score))
                if score >= self.myelination_threshold:
                    high = True
                    self._preload_memory(model)
                    self._prime_bots()
        items = self._gather_research(model, energy)
        flagged = self._flagged_workflows(items)
        reuse_workflows = self._reuse_workflows(flagged)

        tasks = self._items_to_tasks(items)
        for wf in reuse_workflows:
            tasks.append(
                self._create_synthesis_task(
                    description=f"Reuse workflow {wf}",
                    urgency=1,
                    complexity=1,
                    category="workflow",
                )
            )
        tasks.append(
            self._create_synthesis_task(
                description=f"Complete {model}",
                urgency=1,
                complexity=1,
                category="completion",
            )
        )

        validated = list(tasks) if high else self._validate_tasks(tasks)
        if not validated:
            table = self._build_task_table(tasks)
            refined = self.synthesis_bot.create_tasks(table)
            validated = self._validate_tasks(refined)
            if not validated:
                items = self._gather_research(model, energy + 1)
                validated = self._validate_tasks(self._items_to_tasks(items))

        plans = self._plan_bots(validated, trust_weight=trust_weight)
        if not plans:
            validated = list(validated) if high else self._validate_tasks(validated)
            if not validated:
                table = self._build_task_table(tasks)
                refined = self.synthesis_bot.create_tasks(table)
                validated = list(refined) if high else self._validate_tasks(refined)
                if not validated:
                    items = self._gather_research(model, energy + 1)
                    cand = self._items_to_tasks(items)
                    validated = list(cand) if high else self._validate_tasks(cand)
            plans = self._plan_bots(validated, trust_weight=trust_weight)

        plan_valid = plans if high else self._validate_plan(plans)
        if not plan_valid:
            items = self._gather_research(model, energy + 1)
            cand = self._items_to_tasks(items)
            validated = list(cand) if high else self._validate_tasks(cand)
            plans = self._plan_bots(validated, trust_weight=trust_weight)
            if not high:
                self._validate_plan(plans)

        self._assess_hierarchy(plans)

        weight = 1.0
        if plans:
            preds = [self._predict_plan_success(p.name) for p in plans]
            if preds:
                weight = 1.0 + float(sum(preds) / len(preds))

        resources = self._predict_resources(plans)
        build_task_cls = self._build_task_type()
        build_tasks = [
            build_task_cls(
                name=name,
                complexity=1.0,
                frequency=1.0,
                expected_income=1.0,
                resources={
                    "compute": met.cpu,
                    "storage": met.memory,
                    "api": 0.0,
                    "supervision": 0.0,
                },
            )
            for name, met in resources.items()
        ]

        roi = self._roi(model, build_tasks)

        package = None
        if roi.roi >= self.roi_threshold and roi.cost <= self.funds:
            package = self.roi_bot.handoff_to_implementation(
                build_tasks, self.optimiser, title=model
            )
        elif roi.roi < self.roi_threshold:
            try:
                with self.db_router.get_connection("models") as conn:
                    row = conn.execute(
                        "SELECT id FROM models WHERE name LIKE ? ORDER BY id DESC LIMIT 1",
                        (f"%{model}%",),
                    ).fetchone()
                    if row:
                        update_model(row[0], exploration_status="killed")
            except Exception as exc:
                self.logger.exception("failed to update exploration status: %s", exc)

        if not package:
            infos = [
                TaskInfo(
                    name=plan.name,
                    dependencies=[],
                    resources={"cpu": resources[plan.name].cpu},
                    schedule="once",
                    code="# plan",
                    metadata={},
                )
                for plan in plans
            ]
            package = self.handoff.compile(infos)

        final_weight = weight * (2.0 if high else 1.0)
        self._run_support_bots(model, energy=float(energy), weight=final_weight)
        if self.pathway_db:
            outcome = Outcome.SUCCESS if package else Outcome.FAILURE
            self.pathway_db.log(
                PathwayRecord(
                    actions=("pipeline:" + model + (":fast" if high else "")),
                    inputs=model,
                    outputs=str(package),
                    exec_time=0.0,
                    resources="",
                    outcome=outcome,
                    roi=roi.roi,
                )
            )
        result = AutomationResult(package=package, roi=roi)
        try:
            target = self.pathway_db or Path("bot_graph.db")
            self.bot_registry.save(target)
        except Exception as exc:
            self.logger.exception("bot registry save failed: %s", exc)
        return result


__all__ = ["ModelAutomationPipeline"]

