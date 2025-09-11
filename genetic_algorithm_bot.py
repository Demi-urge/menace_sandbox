"""Genetic Algorithm Bot for evolutionary strategy generation."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import random
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import List

from typing import TYPE_CHECKING

from .data_bot import DataBot

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .capital_management_bot import CapitalManagementBot

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
from prometheus_client import CollectorRegistry, Counter, Gauge

try:
    from deap import base, creator, tools  # type: ignore
except Exception:  # pragma: no cover - optional
    base = creator = tools = None  # type: ignore


@dataclass
class GARecord:
    """Record of GA run."""

    params: List[float]
    roi: float


class GAStore:
    """Simple CSV-backed store for GA history."""

    def __init__(self, path: Path = Path("ga_history.csv")) -> None:
        self.path = path
        if self.path.exists():
            try:
                self.df = pd.read_csv(self.path)
            except Exception:
                self.df = pd.DataFrame(columns=["p0", "p1", "p2", "roi"])
        else:
            self.df = pd.DataFrame(columns=["p0", "p1", "p2", "roi"])

    def add(self, rec: GARecord) -> None:
        row = {"p0": rec.params[0], "p1": rec.params[1], "p2": rec.params[2], "roi": rec.roi}
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

    def save(self) -> None:
        self.df.to_csv(self.path, index=False)


@self_coding_managed
class GeneticAlgorithmBot:
    """Run a simple DEAP-based genetic algorithm."""

    def __init__(
        self,
        pop_size: int = 10,
        registry: CollectorRegistry | None = None,
        store: GAStore | None = None,
        *,
        data_bot: "DataBot" | None = None,
        capital_bot: "CapitalManagementBot" | None = None,
        name: str = "genetic_algorithm",
    ) -> None:
        self.pop_size = pop_size
        self.registry = registry or CollectorRegistry()
        self.eval_counter = Counter("ga_evaluations", "Total evaluations", registry=self.registry)
        self.best_gauge = Gauge("ga_best_roi", "Best ROI", registry=self.registry)
        self.store = store or GAStore()
        self.toolbox = self._init_toolbox()
        self.name = name
        self.population = self.toolbox.population(n=self.pop_size)
        self.history: List[GARecord] = []
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.logger = logging.getLogger(self.__class__.__name__)

    def _init_toolbox(self):
        if base is None or creator is None or tools is None:  # pragma: no cover - optional
            raise RuntimeError("DEAP library required")
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 3)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self._evaluate)
        toolbox.register("mate", tools.cxBlend, alpha=0.5)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        return toolbox

    def _evaluate(self, individual: List[float]):
        """Return ROI for an individual using a non-linear scoring function."""
        contrarian, compliant, base_roi = individual
        noise = random.gauss(0, 0.05)
        # Handle potential negative values to avoid complex results when using
        # fractional exponents. Negative values for ``base_roi`` or
        # ``compliant`` can appear after mutation and would otherwise produce
        # complex numbers. ``abs`` ensures the calculation stays in the real
        # domain so the fitness comparison works correctly.
        roi = (
            abs(base_roi) ** 1.5
            + contrarian ** 2 * 0.3
            - abs(compliant) ** 1.5 * 0.2
            + noise
        )
        if self.capital_bot:
            try:
                roi += float(
                    self.capital_bot.energy_score(
                        load=contrarian,
                        success_rate=max(0.0, 1.0 - compliant),
                        deploy_eff=1.0,
                        failure_rate=compliant * 0.1,
                    )
                )
            except Exception as exc:
                self.logger.error("energy_score failed: %s", exc)
        self.eval_counter.inc()
        if self.data_bot:
            try:
                self.data_bot.collect(bot=self.name, revenue=roi, expense=0.0)
            except Exception as exc:
                self.logger.error("data collection failed: %s", exc)
        return (roi,)

    def evolve(self, generations: int = 5) -> GARecord:
        pop = self.population
        for _ in range(generations):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.5:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values
            for mut in offspring:
                if random.random() < 0.2:
                    self.toolbox.mutate(mut)
                    del mut.fitness.values
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                ind.fitness.values = self.toolbox.evaluate(ind)
            pop[:] = offspring
            best = max(pop, key=lambda ind: ind.fitness.values[0])
            self.best_gauge.set(best.fitness.values[0])
            rec = GARecord(params=list(best), roi=best.fitness.values[0])
            self.history.append(rec)
            self.store.add(rec)
            if self.data_bot:
                self.data_bot.collect(
                    bot=self.name,
                    revenue=rec.roi,
                    expense=0.0,
                )
            if self.capital_bot:
                self.capital_bot.update_rois()
        self.store.save()
        self.population = pop
        return self.history[-1]

    def evaluation_count(self) -> int:
        return int(self.eval_counter._value.get())

    def inject(self, params_list: List[List[float]]) -> None:
        """Inject external parameter suggestions into the population."""
        for params in params_list:
            if len(params) != 3:
                continue
            ind = self.toolbox.individual()
            for i, val in enumerate(params):
                ind[i] = float(val)
            del ind.fitness.values
            self.population.append(ind)
        # keep population size consistent
        self.population = self.population[-self.pop_size :]


__all__ = ["GARecord", "GAStore", "GeneticAlgorithmBot"]
