"""Core implementation shared by automation pipeline components."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
    Type,
    cast,
)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:
    from marshmallow import Schema, fields  # type: ignore
    from marshmallow import ValidationError as MMValidationError  # type: ignore

    ValidationError = MMValidationError
except Exception:  # pragma: no cover - optional dependency
    from ..simple_validation import (  # type: ignore
        SimpleSchema as Schema,
        fields,
        ValidationError,
    )

from ..resource_prediction_bot import ResourcePredictionBot, ResourceMetrics
from ..data_interfaces import DataBotInterface, RawMetrics
from ..task_handoff_bot import TaskHandoffBot, TaskInfo, TaskPackage, WorkflowDB
from ..efficiency_bot import EfficiencyBot
from ..performance_assessment_bot import PerformanceAssessmentBot
from ..communication_maintenance_bot import CommunicationMaintenanceBot
from ..operational_monitor_bot import OperationalMonitoringBot
from ..central_database_bot import CentralDatabaseBot, Proposal
from ..sentiment_bot import SentimentBot
from ..query_bot import QueryBot
from ..memory_bot import MemoryBot
from ..communication_testing_bot import CommunicationTestingBot
from ..discrepancy_detection_bot import DiscrepancyDetectionBot
from ..finance_router_bot import FinanceRouterBot
from ..meta_genetic_algorithm_bot import MetaGeneticAlgorithmBot
from ..offer_testing_bot import OfferTestingBot
from ..research_fallback_bot import ResearchFallbackBot
from ..resource_allocation_optimizer import ResourceAllocationOptimizer
from ..resource_allocation_bot import ResourceAllocationBot, AllocationDB
from ..database_manager import update_model
from ..ai_counter_bot import AICounterBot
from ..dynamic_resource_allocator_bot import DynamicResourceAllocator
from ..diagnostic_manager import DiagnosticManager
from ..idea_search_bot import KeywordBank
from ..newsreader_bot import NewsDB
from ..investment_engine import AutoReinvestmentBot
from ..revenue_amplifier import (
    RevenueSpikeEvaluatorBot,
    CapitalAllocationBot,
    RevenueEventsDB,
)
from ..bot_db_utils import wrap_bot_methods
from ..db_router import DBRouter
from ..unified_event_bus import UnifiedEventBus
from ..neuroplasticity import Outcome, PathwayDB, PathwayRecord
from ..unified_learning_engine import UnifiedLearningEngine
from ..action_planner import ActionPlanner
from vector_service.context_builder import ContextBuilder

try:
    from ..data_bot import DataBot as _RuntimeDataBot
except Exception as exc:  # pragma: no cover - optional dependency during bootstrap
    _DATA_BOT_IMPORT_ERROR: Exception | None = exc
    _RuntimeDataBot: type[DataBotInterface] | None = None
else:  # pragma: no cover - import succeeded
    _DATA_BOT_IMPORT_ERROR = None
    _RuntimeDataBot = cast("type[DataBotInterface]", _RuntimeDataBot)

_data_bot_fallback_logged = False


if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..hierarchy_assessment_bot import HierarchyAssessmentBot
    from ..bot_planning_bot import BotPlanningBot, PlanningTask, BotPlan
    from ..bot_registry import BotRegistry
    from ..bot_creation_bot import BotCreationBot
    from ..capital_management_bot import CapitalManagementBot
    from ..research_aggregator_bot import ResearchAggregatorBot, ResearchItem
    from ..information_synthesis_bot import InformationSynthesisBot
    from ..synthesis_models import SynthesisTask
    from ..implementation_optimiser_bot import ImplementationOptimiserBot
    from ..pre_execution_roi_bot import PreExecutionROIBot, BuildTask, ROIResult
else:  # pragma: no cover - runtime fallback
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


def _pre_execution_components() -> Tuple[type["PreExecutionROIBot"], type["BuildTask"], type["ROIResult"]]:
    """Return the ROI bot and associated dataclasses via a deferred import."""

    from ..pre_execution_roi_bot import PreExecutionROIBot as _PreExecutionROIBot
    from ..pre_execution_roi_bot import BuildTask as _BuildTask
    from ..pre_execution_roi_bot import ROIResult as _ROIResult

    return _PreExecutionROIBot, _BuildTask, _ROIResult


def _pipeline_helpers() -> Dict[str, Any]:
    """Return helper callables lazily imported from ``model_automation_pipeline``."""

    from ..model_automation_pipeline import (  # local import to avoid circular deps
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


def _build_fallback_data_bot() -> DataBotInterface:
    """Return a minimal :class:`DataBotInterface` implementation."""

    class _FallbackDataBot:
        """Lightweight stand-in used when the real DataBot cannot load."""

        def __init__(self) -> None:
            self.db = SimpleNamespace(fetch=lambda *args, **kwargs: [])

        def collect(
            self,
            bot: str,
            response_time: float = 0.0,
            errors: int = 0,
            **metrics: float,
        ) -> RawMetrics:
            required = {"cpu", "memory", "disk_io", "net_io"}
            optional_fields = RawMetrics.__dataclass_fields__.keys()
            filtered_metrics: Dict[str, Any] = {
                key: metrics[key]
                for key in optional_fields
                if key in metrics and key not in required and key not in {"bot", "errors", "response_time"}
            }
            return RawMetrics(
                bot=bot,
                cpu=float(metrics.get("cpu", 0.0)),
                memory=float(metrics.get("memory", 0.0)),
                response_time=response_time,
                disk_io=float(metrics.get("disk_io", 0.0)),
                net_io=float(metrics.get("net_io", 0.0)),
                errors=int(errors),
                **filtered_metrics,
            )

        def detect_anomalies(
            self,
            data: Any,
            metric: str,
            *,
            threshold: float = 3.0,
            metrics_db: Any | None = None,
        ) -> Iterable[int]:
            return []

        def roi(self, bot: str) -> float:
            return 0.0

    return cast(DataBotInterface, _FallbackDataBot())


def _create_data_bot(logger: logging.Logger) -> DataBotInterface:
    """Instantiate the runtime data bot with graceful degradation."""

    global _data_bot_fallback_logged
    if _RuntimeDataBot is not None:
        try:
            return cast(DataBotInterface, _RuntimeDataBot())
        except Exception as exc:  # pragma: no cover - degraded bootstrap
            logger.warning(
                "DataBot initialisation failed for ModelAutomationPipeline: %s",
                exc,
            )
            _data_bot_fallback_logged = True
    else:
        if not _data_bot_fallback_logged:
            logger.warning(
                "DataBot unavailable for ModelAutomationPipeline: %s",
                _DATA_BOT_IMPORT_ERROR,
            )
            _data_bot_fallback_logged = True
    return _build_fallback_data_bot()


class ModelAutomationPipeline:
    """Orchestrate bots to automate a model end-to-end."""

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
        comms_bot: CommunicationMaintenanceBot | None = None,
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
        allocator: DynamicResourceAllocator | None = None,
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
        if context_builder is None:
            raise ValueError("context_builder is required")
        self.context_builder = context_builder
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

            synthesis_bot = InformationSynthesisBot(
                aggregator=self.aggregator,
                workflow_db=self.workflow_db,
                event_bus=event_bus,
                context_builder=self.context_builder,
            )
        self.synthesis_bot = synthesis_bot
        self._validator_factory: Callable[[], "TaskValidationBot"] | None = validator_factory
        self._validator: "TaskValidationBot | None" = validator
        self._validator_wrapped = False
        self._bots_primed = False
        planner_cls, planning_task_cls, _ = planning_components()
        self._planning_task_cls = planning_task_cls
        self.planner = planner or planner_cls()
        self.hierarchy = hierarchy or build_default_hierarchy()
        self.data_bot = data_bot or _create_data_bot(self.logger)
        if capital_manager is None:
            capital_manager_cls = capital_manager_cls_factory()
            capital_manager = cast("CapitalManagementBot", capital_manager_cls())
        self.capital_manager = capital_manager

        if predictor:
            self.predictor = predictor
            if getattr(self.predictor, "data_bot", None) is None:
                self.predictor.data_bot = self.data_bot
            if getattr(self.predictor, "capital_bot", None) is None:
                self.predictor.capital_bot = self.capital_manager
        else:
            self.predictor = ResourcePredictionBot(
                data_bot=self.data_bot, capital_bot=self.capital_manager
            )

        self.handoff = handoff or TaskHandoffBot(event_bus=event_bus)
        pre_bot_cls, build_task_cls, roi_result_cls = self._ensure_pre_execution_components()
        self._build_task_cls = build_task_cls
        self._roi_result_cls = roi_result_cls
        if roi_bot:
            self.roi_bot = roi_bot
            if getattr(self.roi_bot, "handoff", None) is None:
                self.roi_bot.handoff = self.handoff
        else:
            self.roi_bot = pre_bot_cls(handoff=self.handoff)
        if optimiser is None:
            optimiser_cls = implementation_optimiser_cls()
            optimiser = optimiser_cls(context_builder=self.context_builder)
        self.optimiser = optimiser
        self.funds = funds
        self.roi_threshold = roi_threshold

        self.efficiency_bot = efficiency_bot or EfficiencyBot()
        self.performance_bot = performance_bot or PerformanceAssessmentBot()
        self.comms_bot = comms_bot or CommunicationMaintenanceBot()
        self.monitor_bot = monitor_bot or OperationalMonitoringBot()
        self.db_bot = db_bot or CentralDatabaseBot(db_router=self.db_router)
        self.sentiment_bot = sentiment_bot or SentimentBot()
        self.query_bot = query_bot or QueryBot(context_builder=self.context_builder)
        self.memory_bot = memory_bot or MemoryBot()
        self.comms_test_bot = comms_test_bot or CommunicationTestingBot()
        self.discrepancy_bot = discrepancy_bot or DiscrepancyDetectionBot()
        self.finance_bot = finance_bot or FinanceRouterBot()
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
                    creation_bot = _bot_creation_cls(
                        context_builder=self.context_builder
                    )
                except Exception as exc:  # pragma: no cover - degraded bootstrap
                    self.logger.warning(
                        "BotCreationBot initialisation failed for ModelAutomationPipeline: %s",
                        exc,
                    )
                    creation_bot = None

        self.creation_bot = creation_bot
        self.meta_ga_bot = meta_ga_bot or MetaGeneticAlgorithmBot()
        self.offer_bot = offer_bot or OfferTestingBot()
        self.fallback_bot = fallback_bot or ResearchFallbackBot()
        self.optimizer = optimizer or ResourceAllocationOptimizer()
        self.ai_counter_bot = ai_counter_bot or AICounterBot()
        self.allocator = allocator or DynamicResourceAllocator(
            alloc_bot=ResourceAllocationBot(
                AllocationDB(), context_builder=self.context_builder
            ),
            context_builder=self.context_builder,
        )
        self.diagnostic_manager = diagnostic_manager or DiagnosticManager(
            context_builder=self.context_builder
        )
        self.idea_bank = idea_bank or KeywordBank()
        self.news_db = news_db or NewsDB()
        self.reinvestment_bot = reinvestment_bot or AutoReinvestmentBot()
        self.spike_bot = spike_bot or RevenueSpikeEvaluatorBot(RevenueEventsDB())
        self.allocation_bot = allocation_bot or CapitalAllocationBot()
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
        self._bots = [
            self.aggregator,
            self.synthesis_bot,
            self.planner,
            self.hierarchy,
            self.predictor,
            self.data_bot,
            self.capital_manager,
            self.roi_bot,
            self.handoff,
            self.optimiser,
            self.efficiency_bot,
            self.performance_bot,
            self.comms_bot,
            self.monitor_bot,
            self.db_bot,
            self.sentiment_bot,
            self.query_bot,
            self.memory_bot,
            self.comms_test_bot,
            self.discrepancy_bot,
            self.finance_bot,
            self.creation_bot,
            self.meta_ga_bot,
            self.offer_bot,
            self.fallback_bot,
            self.optimizer,
            self.ai_counter_bot,
            self.allocator,
            self.diagnostic_manager,
            self.idea_bank,
            self.news_db,
            self.reinvestment_bot,
            self.spike_bot,
            self.allocation_bot,
        ]
        for bot in self._bots:
            wrap_bot_methods(bot, self.db_router, self.bot_registry)
        if self._validator is not None and not self._validator_wrapped:
            self._register_validator(self._validator)

    # ------------------------------------------------------------------

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
                except Exception as exc:  # pragma: no cover - defensive logging
                    self.logger.exception("prime() failed for %s: %s", validator.__class__.__name__, exc)

    # ------------------------------------------------------------------

    def _prime_bots(self) -> None:
        """Call optional prime() on all registered bots."""
        for bot in self._bots:
            prime = getattr(bot, "prime", None)
            if callable(prime):
                try:
                    prime()
                except Exception as exc:
                    self.logger.exception("prime() failed for %s: %s", bot.__class__.__name__, exc)
        self._bots_primed = True

    # ------------------------------------------------------------------

    def _gather_research(self, model: str, energy: int = 1) -> List["ResearchItem"]:
        return self.aggregator.process(model, energy=energy)

    @staticmethod
    def _flagged_workflows(items: Iterable["ResearchItem"]) -> List[str]:
        return [
            it.title or it.topic
            for it in items
            if it.category.lower() == "workflow" or "workflow" in (t.lower() for t in it.tags)
        ]

    def _reuse_workflows(self, names: Iterable[str]) -> List[str]:
        """Return derivative workflow steps for reuse."""
        if not names:
            return []
        try:
            records = self.workflow_db.fetch()
        except Exception as exc:
            self.logger.exception("workflow fetch failed: %s", exc)
            return []
        steps: List[str] = []
        seen = set()
        for rec in records:
            if rec.title in names or rec.description in names:
                for step in rec.workflow[:2]:  # use first few steps as hints
                    if step and step not in seen:
                        seen.add(step)
                        steps.append(step)
        return steps

    def _items_to_tasks(self, items: Iterable["ResearchItem"]) -> List["SynthesisTask"]:
        data = [
            {"id": it.item_id or 0, "name": it.title or it.topic, "content": it.content}
            for it in items
        ]
        if not data:
            return []
        df = pd.DataFrame(data)

        class ResearchSchema(Schema):
            id = fields.Int(required=True)
            name = fields.Str(required=True)
            content = fields.Str(required=True)

        self.synthesis_bot.analyse(df, ResearchSchema(), "research")
        return self.synthesis_bot.create_tasks(df)

    # ------------------------------------------------------------------

    def _validate_tasks(
        self, tasks: Iterable["SynthesisTask"]
    ) -> List["SynthesisTask"]:
        validator = self._ensure_validator()
        return validator.validate_tasks(list(tasks))

    def _plan_bots(
        self,
        tasks: Iterable["SynthesisTask"],
        *,
        trust_weight: float = 1.0,
    ) -> List["BotPlan"]:
        planning = [
            self._planning_task_cls(
                description=t.description,
                complexity=t.complexity,
                frequency=1,
                expected_time=1.0,
                actions=["python"],
                resources={"cpu": 1},
            )
            for t in tasks
        ]
        if self.action_planner:
            seq = "->".join(t.description for t in tasks)
            try:
                self.action_planner.predict_next_action(seq)
            except Exception as exc:
                self.logger.exception("action planner failed: %s", exc)
        return self.planner.plan_bots(planning, trust_weight=trust_weight)

    def _validate_plan(self, plans: Iterable["BotPlan"]) -> List["SynthesisTask"]:
        tasks = [
            self._create_synthesis_task(
                description=p.name,
                urgency=1,
                complexity=1,
                category="plan",
            )
            for p in plans
        ]
        return self._validate_tasks(tasks)

    def _assess_hierarchy(self, plans: Iterable["BotPlan"]) -> None:
        for p in plans:
            try:
                self.hierarchy.register(p.name, p.template)
            except Exception as exc:
                self.logger.exception(
                    "hierarchy register failed for %s: %s", p.name, exc
                )
        try:
            self.hierarchy.redundancy_analysis()
            self.hierarchy.assess_risk()
            self.hierarchy.monitor_system()
        except Exception as exc:
            self.logger.exception("hierarchy assessment failed: %s", exc)

    # ------------------------------------------------------------------
    def _predict_plan_success(self, name: str) -> float:
        """Return success probability for a plan using the learning engine."""
        if not self.learning_engine or not self.pathway_db:
            return 0.0
        try:
            return float(self.learning_engine.predict_outcome(name, self.pathway_db))
        except Exception as exc:
            self.logger.exception("learning engine failed: %s", exc)
            return 0.0

    def _predict_resources(self, plans: Iterable["BotPlan"]) -> Dict[str, ResourceMetrics]:
        metrics: Dict[str, ResourceMetrics] = {}
        for plan in plans:
            try:
                metrics[plan.name] = self.predictor.predict(plan.template)
            except Exception as exc:
                self.logger.exception("predictor failed for %s: %s", plan.name, exc)
        return metrics

    def _roi(self, model: str, tasks: Iterable["BuildTask"]) -> "ROIResult":
        try:
            result = self.roi_bot.evaluate(model, list(tasks))
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
            bot = entry[0]
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
            df = pd.DataFrame({
                "id": range(len(tasks)),
                "name": [t.description for t in tasks],
                "content": [t.description for t in tasks],
            })
            refined = self.synthesis_bot.create_tasks(df)
            validated = self._validate_tasks(refined)
            if not validated:
                items = self._gather_research(model, energy + 1)
                validated = self._validate_tasks(self._items_to_tasks(items))

        plans = self._plan_bots(validated, trust_weight=trust_weight)
        if not plans:
            validated = list(validated) if high else self._validate_tasks(validated)
            if not validated:
                df = pd.DataFrame({
                    "id": range(len(tasks)),
                    "name": [t.description for t in tasks],
                    "content": [t.description for t in tasks],
                })
                refined = self.synthesis_bot.create_tasks(df)
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
