"""GA Prediction Bot using genetic algorithms to evolve forecasting models."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .coding_bot_interface import self_coding_managed
from .data_bot import DataBot
import random
import logging

registry = BotRegistry()
data_bot = DataBot(start_server=False)

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Iterable as TIterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .capital_management_bot import CapitalManagementBot
    from .prediction_manager_bot import PredictionManager, PredictionBotEntry

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
from prometheus_client import CollectorRegistry, Counter, Gauge

try:
    from deap import base, creator, tools  # type: ignore
except Exception:  # pragma: no cover - optional
    base = creator = tools = None  # type: ignore

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass
class TemplateEntry:
    """Stored model configuration and score."""

    params: List[float]
    score: float


class TemplateDB:
    """CSV-backed template storage."""

    def __init__(self, path: Path | str = "ga_prediction_templates.csv") -> None:
        self.path = Path(path)
        if self.path.exists():
            self.df = pd.read_csv(self.path)
        else:
            self.df = pd.DataFrame(columns=["p0", "p1", "p2", "score"])

    def add(self, entry: TemplateEntry) -> None:
        row = {"p0": entry.params[0], "p1": entry.params[1], "p2": entry.params[2], "score": entry.score}
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)

    def save(self) -> None:
        self.df.to_csv(self.path, index=False)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class GAPredictionBot:
    """Evolve ML models to maximise prediction accuracy."""

    prediction_profile = {"scope": ["prediction"], "risk": ["medium"]}

    def __init__(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[int],
        pop_size: int = 6,
        registry: CollectorRegistry | None = None,
        db: TemplateDB | None = None,
        *,
        data_bot: "DataBot" | None = None,
        capital_bot: "CapitalManagementBot" | None = None,
        prediction_manager: "PredictionManager" | None = None,
        name: str = "ga_prediction",
    ) -> None:
        self.X = np.array(list(X))
        self.y = np.array(list(y))
        self.pop_size = pop_size
        self.registry = registry or CollectorRegistry()
        self.eval_counter = Counter("ga_pred_evals", "GA prediction evaluations", registry=self.registry)
        self.best_gauge = Gauge("ga_pred_best", "Best prediction score", registry=self.registry)
        self.db = db or TemplateDB()
        self.toolbox = self._init_toolbox()
        self.population = self.toolbox.population(n=self.pop_size)
        self.history: List[TemplateEntry] = []
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.prediction_manager = prediction_manager
        self.name = name
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)

    def _apply_prediction_bots(self) -> float:
        """Return mean prediction across assigned bots."""
        if not self.prediction_manager or not self.assigned_prediction_bots:
            return 0.0
        preds: List[float] = []
        for bid in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bid)
            bot = entry.bot if entry else None
            if not bot:
                continue
            try:
                if hasattr(bot, "batch_predict"):
                    res = bot.batch_predict(self.X)
                    vals = [r[1] if isinstance(r, tuple) else r for r in res]
                    preds.extend(float(v) for v in vals)
                elif hasattr(bot, "predict"):
                    res = bot.predict(self.X)
                    if isinstance(res, TIterable):
                        vals = [r[1] if isinstance(r, tuple) else r for r in res]
                        preds.extend(float(v) for v in vals)
                    else:
                        preds.append(float(res))
            except Exception:
                continue
        return float(np.mean(preds)) if preds else 0.0

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
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        return toolbox

    def _model_from_params(self, params: List[float]):
        algo = int(round(params[0])) % 2
        p1 = min(max(params[1], 0.0), 1.0)
        p2 = min(max(params[2], 0.0), 1.0)
        if algo == 0:
            C = 0.1 + p1 * 10
            return LogisticRegression(C=C, solver="liblinear")
        n_estimators = int(10 + p1 * 90)
        max_depth = int(1 + p2 * 10)
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

    # ------------------------------------------------------------------
    def _extra_features(self) -> List[float]:
        """Gather additional context from data/capital/prediction bots."""
        feats: List[float] = []
        if self.data_bot:
            try:
                df = self.data_bot.db.fetch(20)
            except Exception:
                df = pd.DataFrame()
            load = float(df["cpu"].mean() or 0.0) / 100.0 if not df.empty else 0.0
            success = 1.0 - float(df["errors"].mean() or 0.0) / (
                float(df["errors"].mean() or 0.0) + 1.0
            ) if not df.empty else 1.0
            feats.extend([load, success])
        if self.capital_bot:
            energy = self.capital_bot.energy_score(
                load=feats[0] if feats else 0.0,
                success_rate=feats[1] if len(feats) > 1 else 1.0,
                deploy_eff=1.0,
                failure_rate=1.0 - (feats[1] if len(feats) > 1 else 1.0),
            )
            feats.append(float(energy))
        pred = self._apply_prediction_bots()
        if pred:
            feats.append(pred)
        return feats

    def _evaluate(self, individual: List[float]):
        model = self._model_from_params(individual)
        X = self.X
        extra = self._extra_features()
        if extra:
            extra_arr = np.tile(extra, (len(X), 1))
            X = np.hstack([X, extra_arr])
        X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        self.eval_counter.inc()
        if self.data_bot:
            try:
                self.data_bot.collect(bot=self.name, revenue=score, expense=0.0)
            except Exception:
                logging.getLogger(__name__).exception("data bot collect failed")
        return (score,)

    def evolve(self, generations: int = 3) -> TemplateEntry:
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
            entry = TemplateEntry(params=list(best), score=best.fitness.values[0])
            self.history.append(entry)
            self.db.add(entry)
            if self.data_bot:
                self.data_bot.collect(
                    bot=self.name,
                    revenue=entry.score,
                    expense=0.0,
                )
            if self.capital_bot:
                self.capital_bot.update_rois()
        self.db.save()
        self.population = pop
        return self.history[-1]

    def evaluation_count(self) -> int:
        return int(self.eval_counter._value.get())


__all__ = ["TemplateEntry", "TemplateDB", "GAPredictionBot"]
