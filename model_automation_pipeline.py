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

from .hierarchy_assessment_bot import HierarchyAssessmentBot
from .resource_prediction_bot import ResourcePredictionBot, ResourceMetrics
from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from .pre_execution_roi_bot import PreExecutionROIBot, BuildTask, ROIResult
from .implementation_optimiser_bot import ImplementationOptimiserBot
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
from .bot_creation_bot import BotCreationBot
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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .bot_planning_bot import BotPlanningBot, PlanningTask, BotPlan
    from .bot_registry import BotRegistry
    from .research_aggregator_bot import ResearchAggregatorBot, ResearchItem
    from .information_synthesis_bot import InformationSynthesisBot
    from .synthesis_models import SynthesisTask
else:  # pragma: no cover - runtime fallback
    BotPlanningBot = Any  # type: ignore
    PlanningTask = Any  # type: ignore
    BotPlan = Any  # type: ignore
    ResearchAggregatorBot = Any  # type: ignore
    ResearchItem = Any  # type: ignore


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


class ModelAutomationPipeline:
    """Orchestrate bots to automate a model end-to-end."""

    def __init__(
        self,
        aggregator: ResearchAggregatorBot | None = None,
        synthesis_bot: "InformationSynthesisBot" | None = None,
        validator: TaskValidationBot | None = None,
        planner: BotPlanningBot | None = None,
        hierarchy: HierarchyAssessmentBot | None = None,
        predictor: ResourcePredictionBot | None = None,
        data_bot: DataBot | None = None,
        capital_manager: CapitalManagementBot | None = None,
        roi_bot: PreExecutionROIBot | None = None,
        handoff: TaskHandoffBot | None = None,
        optimiser: ImplementationOptimiserBot | None = None,
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
        creation_bot: BotCreationBot | None = None,
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
        validator_factory: Callable[[], TaskValidationBot] | None = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if context_builder is None:
            raise ValueError("context_builder is required")
        self.context_builder = context_builder
        try:
            self.context_builder.refresh_db_weights()
        except Exception as exc:
            self.logger.exception("context builder refresh failed: %s", exc)
        self.db_router = db_router or DB_ROUTER
        if aggregator is None:
            aggregator = _LazyAggregator(
                lambda: _load_research_aggregator(self.context_builder)
            )
        self.aggregator = aggregator
        self.workflow_db = workflow_db or WorkflowDB(event_bus=event_bus)
        if synthesis_bot is None:
            from .information_synthesis_bot import InformationSynthesisBot

            synthesis_bot = InformationSynthesisBot(
                aggregator=self.aggregator,
                workflow_db=self.workflow_db,
                event_bus=event_bus,
                context_builder=self.context_builder,
            )
        self.synthesis_bot = synthesis_bot
        self._validator_factory: Callable[[], TaskValidationBot] | None = validator_factory
        self._validator: TaskValidationBot | None = validator
        self._validator_wrapped = False
        self._bots_primed = False
        planner_cls, planning_task_cls, _ = _planning_components()
        self._planning_task_cls = planning_task_cls
        self.planner = planner or planner_cls()
        self.hierarchy = hierarchy or HierarchyAssessmentBot()
        self.data_bot = data_bot or DataBot()
        self.capital_manager = capital_manager or CapitalManagementBot()

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
        if roi_bot:
            self.roi_bot = roi_bot
            if getattr(self.roi_bot, "handoff", None) is None:
                self.roi_bot.handoff = self.handoff
        else:
            self.roi_bot = PreExecutionROIBot(handoff=self.handoff)
        self.optimiser = optimiser or ImplementationOptimiserBot(
            context_builder=self.context_builder
        )
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
        self.creation_bot = creation_bot or BotCreationBot(
            context_builder=self.context_builder
        )
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
            from .bot_registry import BotRegistry as _BotRegistry

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

    @property
    def validator(self) -> TaskValidationBot:
        """Return the task validator, instantiating it lazily when required."""

        return self._ensure_validator()

    @validator.setter
    def validator(self, value: TaskValidationBot | None) -> None:
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

    def _ensure_validator(self) -> TaskValidationBot:
        """Create and register the validator if it has not been resolved yet."""

        if self._validator is None:
            factory = self._validator_factory or _build_default_validator
            validator = factory()
            self._validator = validator
            self._register_validator(validator)
        return self._validator

    # ------------------------------------------------------------------

    def _register_validator(self, validator: TaskValidationBot) -> None:
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

    def _gather_research(self, model: str, energy: int = 1) -> List[ResearchItem]:
        return self.aggregator.process(model, energy=energy)

    @staticmethod
    def _flagged_workflows(items: Iterable[ResearchItem]) -> List[str]:
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

    def _items_to_tasks(self, items: Iterable[ResearchItem]) -> List["SynthesisTask"]:
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
    ) -> List[BotPlan]:
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

    def _validate_plan(self, plans: Iterable[BotPlan]) -> List["SynthesisTask"]:
        tasks = [
            _create_synthesis_task(
                description=p.name,
                urgency=1,
                complexity=1,
                category="plan",
            )
            for p in plans
        ]
        return self._validate_tasks(tasks)

    def _assess_hierarchy(self, plans: Iterable[BotPlan]) -> None:
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
        row = self.pathway_db.conn.execute(
            """
            SELECT m.frequency, m.avg_exec_time, m.avg_roi, m.myelination_score
            FROM metadata m JOIN pathways p ON p.id=m.pathway_id
            WHERE p.actions=? ORDER BY m.last_activation DESC LIMIT 1
            """,
            (name,),
        ).fetchone()
        if row:
            freq, exec_time, roi, score = row
        else:
            freq = exec_time = roi = score = 0.0
        try:
            return float(
                self.learning_engine.predict_success(
                    float(freq), float(exec_time), float(roi), float(score), name
                )
            )
        except Exception as exc:
            self.logger.exception("success prediction failed: %s", exc)
            return 0.0

    def _predict_resources(self, plans: Iterable[BotPlan]) -> Dict[str, ResourceMetrics]:
        results: Dict[str, ResourceMetrics] = {}
        for plan in plans:
            try:
                results[plan.name] = self.predictor.predict(plan.name)
            except Exception as exc:
                self.logger.exception("resource prediction failed for %s: %s", plan.name, exc)
                results[plan.name] = ResourceMetrics(cpu=1.0, memory=100.0, disk=10.0, time=1.0)
        return results

    def _roi(self, model: str, tasks: Iterable[BuildTask]) -> ROIResult:
        return self.roi_bot.predict_model_roi(model, list(tasks))

    # ------------------------------------------------------------------

    def _run_support_bots(
        self, model: str, *, energy: float = 1.0, weight: float = 1.0
    ) -> None:
        bots = [
            (self.monitor_bot, "collect_and_export", (model,)),
            (self.efficiency_bot, "assess_efficiency", ()),
            (self.performance_bot, "self_assess", (model,)),
            (self.comms_bot, "check_updates", ()),
            (
                self.db_bot,
                "enqueue",
                (
                    Proposal(
                        operation="insert",
                        target_table="logs",
                        payload={"model": model},
                        origin_bot_id="pipeline",
                    ),
                ),
            ),
            (self.sentiment_bot, "process", ([],)),
            (self.query_bot, "history", ("default",)),
            (self.memory_bot, "log", ("pipeline", model)),
            (self.comms_test_bot, "report", ()),
            (self.discrepancy_bot, "scan", ()),
            (self.finance_bot, "route_payment", (0.0, model)),
            (self.creation_bot, "needs_new_bot", ()),
            (self.meta_ga_bot, "evolve", (1,)),
            (self.offer_bot, "best_variants", ()),
            (self.fallback_bot, "process", (model,)),
            (self.optimizer, "update_priorities", ([model],)),
            (self.ai_counter_bot, "analyse", ([],)),
            (self.allocator, "allocate", ([model],), {"weight": weight}),
            (self.diagnostic_manager, "run", ()),
            (self.idea_bank, "generate_queries", (energy,)),
            (self.news_db, "fetch", ()),
            (self.reinvestment_bot, "reinvest", ()),
            (self.spike_bot, "detect_spike", (model,)),
            (self.allocation_bot, "rebalance", (model, 0.0)),
        ]
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def call(entry: tuple) -> None:
            bot = entry[0]
            method = entry[1]
            args = entry[2] if len(entry) > 2 else ()
            kwargs = entry[3] if len(entry) > 3 else {}
            try:
                getattr(bot, method)(*args, **kwargs)
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
                    _make_research_item(
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

    def run(self, model: str, energy: int = 1) -> AutomationResult:
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
                _create_synthesis_task(
                    description=f"Reuse workflow {wf}",
                    urgency=1,
                    complexity=1,
                    category="workflow",
                )
            )
        tasks.append(
            _create_synthesis_task(
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
        build_tasks = [
            BuildTask(
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
                with DB_ROUTER.get_connection("models") as conn:
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


@lru_cache(maxsize=1)
def _planning_components() -> tuple[type["BotPlanningBot"], type["PlanningTask"], type["BotPlan"]]:
    """Load planning helpers lazily to avoid circular imports."""

    from .bot_planning_bot import BotPlanningBot as _BotPlanningBot
    from .bot_planning_bot import BotPlan as _BotPlan
    from .bot_planning_bot import PlanningTask as _PlanningTask

    return _BotPlanningBot, _PlanningTask, _BotPlan


__all__ = ["AutomationResult", "ModelAutomationPipeline"]
