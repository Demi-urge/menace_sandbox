"""Bot Planning Bot for assessing and creating new bots."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
from dataclasses import dataclass, field
from typing import Iterable, List, Dict, Optional

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


@self_coding_managed
class BotPlanningBot:
    """Analyse tasks and plan bots with hierarchy mapping."""

    def __init__(self, template_manager: Optional[TemplateManager] = None) -> None:
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

    def optimise_resources(self, tasks: Iterable[PlanningTask], cpu_limit: float = 4) -> List[float]:
        """Use linear programming to fit tasks within a CPU budget."""
        data = list(tasks)
        if not data:
            return []
        problem = pulp.LpProblem("res", pulp.LpMaximize)
        vars = [pulp.LpVariable(f"x{i}", lowBound=0, upBound=1) for i in range(len(data))]
        problem += pulp.lpSum(vars)
        problem += pulp.lpSum(v * data[i].resources.get("cpu", 1) for i, v in enumerate(vars)) <= cpu_limit
        problem.solve(pulp.PULP_CBC_CMD(msg=False))
        return [float(v.value()) for v in vars]

    def plan_bots(self, tasks: Iterable[PlanningTask], *, trust_weight: float = 1.0) -> List[BotPlan]:
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


__all__ = [
    "PlanningTask",
    "BotPlan",
    "TemplateManager",
    "BotPlanningBot",
]
