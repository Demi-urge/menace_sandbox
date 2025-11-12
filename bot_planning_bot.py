"""Bot Planning Bot for assessing and creating new bots."""

from __future__ import annotations

import logging

from .bot_registry import BotRegistry
from .data_bot import DataBot, persist_sc_thresholds
from .coding_bot_interface import (
    prepare_pipeline_for_bootstrap,
    self_coding_managed,
)
try:  # pragma: no cover - fail fast if self-coding manager missing
    from .self_coding_manager import SelfCodingManager, internalize_coding_bot
except Exception as exc:  # pragma: no cover - critical dependency
    raise RuntimeError(
        "BotPlanningBot requires SelfCodingManager; install self-coding dependencies."
    ) from exc
from .self_coding_engine import SelfCodingEngine
try:  # pragma: no cover - optional to avoid circular imports in tests
    from .model_automation_pipeline import ModelAutomationPipeline
except Exception as exc:  # pragma: no cover - provide stub when unavailable
    _pipeline_import_error = exc

    class ModelAutomationPipeline:  # type: ignore[misc]
        """Fallback stub that surfaces the original import failure."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "ModelAutomationPipeline is unavailable; ensure menace is installed "
                "as a package so relative imports resolve correctly."
            ) from _pipeline_import_error
from .threshold_service import ThresholdService
from .code_database import CodeDB
from .gpt_memory import GPTMemoryManager
from .self_coding_thresholds import get_thresholds
from vector_service.context_builder import ContextBuilder
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, cast
from .shared_evolution_orchestrator import get_orchestrator
from context_builder_util import create_context_builder
from .shared.cooperative_init import ensure_cooperative_init, monkeypatch_class_references

import networkx as nx
try:
    from sklearn.linear_model import LinearRegression
except Exception:  # pragma: no cover - optional dependency
    class LinearRegression:  # type: ignore
        def fit(self, X, y):
            self.coef_ = [0 for _ in X[0]]

        def predict(self, X):
            return [0.0 for _ in X]
import pulp

logger = logging.getLogger(__name__)

registry = BotRegistry()
data_bot = DataBot(start_server=False)

_context_builder = create_context_builder()
engine = SelfCodingEngine(CodeDB(), GPTMemoryManager(), context_builder=_context_builder)

pipeline: "ModelAutomationPipeline | None" = None
_pipeline_promoter: Callable[[SelfCodingManager], None] | None = None

evolution_orchestrator = get_orchestrator("BotPlanningBot", data_bot, engine)
_th = get_thresholds("BotPlanningBot")
persist_sc_thresholds(
    "BotPlanningBot",
    roi_drop=_th.roi_drop,
    error_increase=_th.error_increase,
    test_failure_increase=_th.test_failure_increase,
)

manager: SelfCodingManager | None = None


def _promote_pipeline_manager(manager: SelfCodingManager | None) -> None:
    global _pipeline_promoter
    promoter = _pipeline_promoter
    if promoter is None or manager is None:
        return
    promoter(manager)
    _pipeline_promoter = None


def _initialise_self_coding() -> None:
    """Initialise the ModelAutomationPipeline and self-coding manager lazily."""

    global pipeline, manager, _pipeline_promoter
    if pipeline is None:
        try:
            pipeline, _pipeline_promoter = prepare_pipeline_for_bootstrap(
                pipeline_cls=ModelAutomationPipeline,
                context_builder=_context_builder,
                bot_registry=registry,
                data_bot=data_bot,
            )
        except Exception as exc:  # pragma: no cover - degraded bootstrap path
            logger.warning(
                "ModelAutomationPipeline unavailable for BotPlanningBot: %s",
                exc,
            )
            pipeline = None
            _pipeline_promoter = None

    if pipeline is None:
        logger.warning(
            "BotPlanningBot self-coding manager unavailable; running without ModelAutomationPipeline",
        )
        return

    if manager is None:
        try:
            manager = internalize_coding_bot(
                "BotPlanningBot",
                engine,
                pipeline,
                data_bot=data_bot,
                bot_registry=registry,
                evolution_orchestrator=evolution_orchestrator,
                threshold_service=ThresholdService(),
                roi_threshold=_th.roi_drop,
                error_threshold=_th.error_increase,
                test_failure_threshold=_th.test_failure_increase,
            )
        except Exception as exc:  # pragma: no cover - degraded bootstrap path
            logger.warning(
                "BotPlanningBot self-coding manager initialisation failed: %s",
                exc,
            )
            manager = None

    if manager is not None and not isinstance(manager, SelfCodingManager):  # pragma: no cover - safety
        raise RuntimeError("internalize_coding_bot failed to return a SelfCodingManager")

    _promote_pipeline_manager(manager)


class TemplateManager:
    """Minimal interface to a Template Management Bot."""

    def __init__(self) -> None:
        self.templates: Dict[str, Dict[str, object]] = {"default": {"languages": ["python"]}}

    def has_template(self, languages: Iterable[str]) -> bool:
        lang_set = set(languages)
        tmpl = self.templates.get("default", {})
        return lang_set.issubset(set(tmpl.get("languages", [])))

    def create_template(self, name: str, languages: Iterable[str]) -> None:
        self.templates[name] = {"languages": list(languages)}

    def store_bot(self, name: str, code: str) -> None:
        self.templates[name] = {"code": code}


@dataclass
class PlanningTask:
    """Definition of a task requiring a bot."""

    description: str
    complexity: int
    frequency: int
    expected_time: float
    actions: List[str]
    env: List[str] = field(default_factory=list)
    constraints: Dict[str, int] = field(default_factory=dict)
    resources: Dict[str, int] = field(default_factory=dict)  # cpu, memory


@dataclass
class BotPlan:
    """Finalised bot planning result."""

    name: str
    template: str
    scalability: float
    level: str


class BotPlanningBot:
    """Analyse tasks and plan bots with hierarchy mapping."""

    def __init__(
        self,
        template_manager: Optional[TemplateManager] = None,
        *,
        manager: SelfCodingManager | None = None,
    ) -> None:
        self.tm = template_manager or TemplateManager()
        self.graph = nx.DiGraph()
        self.regressor = LinearRegression()

    def evaluate_tasks(self, tasks: Iterable[PlanningTask]) -> List[float]:
        """Predict creation time from task attributes."""
        data = list(tasks)
        if not data:
            return []
        X = [[t.complexity, t.frequency, t.expected_time] for t in data]
        y = [t.expected_time for t in data]
        self.regressor.fit(X, y)
        return list(self.regressor.predict(X))

    def optimise_resources(
        self, tasks: Iterable[PlanningTask], cpu_limit: float = 4
    ) -> List[float]:
        """Use linear programming to fit tasks within a CPU budget."""
        data = list(tasks)
        if not data:
            return []
        problem = pulp.LpProblem("res", pulp.LpMaximize)
        vars = [
            pulp.LpVariable(f"x{i}", lowBound=0, upBound=1)
            for i in range(len(data))
        ]
        problem += pulp.lpSum(vars)
        problem += pulp.lpSum(
            v * data[i].resources.get("cpu", 1) for i, v in enumerate(vars)
        ) <= cpu_limit
        problem.solve(pulp.PULP_CBC_CMD(msg=False))
        return [float(v.value()) for v in vars]

    def plan_bots(
        self, tasks: Iterable[PlanningTask], *, trust_weight: float = 1.0
    ) -> List[BotPlan]:
        """Create bot plans and build the hierarchy graph.

        Parameters
        ----------
        tasks:
            Planning tasks describing the required bots.
        trust_weight:
            Factor biasing resource allocation; higher values allow more bots to
            be planned.
        """

        data = list(tasks)
        cpu_limit = 4 * float(trust_weight)
        allocations = self.optimise_resources(data, cpu_limit=cpu_limit)
        plans: List[BotPlan] = []
        for idx, (task, alloc) in enumerate(zip(data, allocations)):
            if alloc <= 0:
                continue
            template = "default"
            if not self.tm.has_template(task.actions):
                template = f"tmpl-{idx}"
                self.tm.create_template(template, task.actions)
            name = f"bot{idx}"
            self.tm.store_bot(name, f"# {task.description}")
            scal = sum(task.resources.values() or [1]) / (task.frequency + 1)
            level = self._assign_level(task.complexity)
            plans.append(
                BotPlan(name=name, template=template, scalability=scal, level=level)
            )
            self.graph.add_node(name, task=task.description)
            if idx:
                self.graph.add_edge(f"bot{idx-1}", name)
        return plans

    @staticmethod
    def _assign_level(complexity: int) -> str:
        if complexity < 3:
            return "L1"
        if complexity < 5:
            return "L2"
        if complexity < 7:
            return "L3"
        if complexity < 9:
            return "M1"
        return "M2"


_initialise_self_coding()

_decorator_kwargs = {"bot_registry": registry, "data_bot": data_bot}
if manager is not None:
    _decorator_kwargs["manager"] = manager

BotPlanningBot = self_coding_managed(**_decorator_kwargs)(BotPlanningBot)

_UnwrappedBotPlanningBot = BotPlanningBot
BotPlanningBot = cast(
    "type[BotPlanningBot]",
    ensure_cooperative_init(cast(type, BotPlanningBot), logger=logger),
)
monkeypatch_class_references(_UnwrappedBotPlanningBot, BotPlanningBot)


__all__ = [
    "PlanningTask",
    "BotPlan",
    "TemplateManager",
    "BotPlanningBot",
]
