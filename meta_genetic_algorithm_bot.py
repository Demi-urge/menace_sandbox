"""Meta genetic algorithm for tuning GeneticAlgorithmBot parameters."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import random
from dataclasses import dataclass
from typing import Iterable, List

from .genetic_algorithm_bot import GeneticAlgorithmBot, GARecord

registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:
    from deap import tools  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tools = None  # type: ignore


@dataclass
class GAConfig:
    pop_size: int
    generations: int
    mutate_sigma: float


@dataclass
class MetaGARecord:
    config: GAConfig
    roi: float


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class MetaGeneticAlgorithmBot:
    """Evolve configurations for GeneticAlgorithmBot instances."""

    def __init__(self, population: int = 5) -> None:
        self.population = population
        self.history: List[MetaGARecord] = []

    @staticmethod
    def _evaluate(config: GAConfig) -> float:
        """Run a GA with the given configuration and return best ROI."""
        bot = GeneticAlgorithmBot(pop_size=config.pop_size)
        if bot.toolbox is None or tools is None:
            return 0.0
        bot.toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0,
            sigma=config.mutate_sigma,
            indpb=0.1,
        )
        record = bot.evolve(config.generations)
        return record.roi

    def evolve(self, generations: int = 3) -> MetaGARecord:
        configs = [
            GAConfig(
                pop_size=random.randint(5, 20),
                generations=random.randint(1, 5),
                mutate_sigma=random.uniform(0.1, 0.5),
            )
            for _ in range(self.population)
        ]
        best_cfg = None
        best_roi = -1.0
        for cfg in configs:
            roi = self._evaluate(cfg)
            if roi > best_roi:
                best_roi = roi
                best_cfg = cfg
        rec = MetaGARecord(config=best_cfg or configs[0], roi=best_roi)
        self.history.append(rec)
        return rec


__all__ = ["GAConfig", "MetaGARecord", "MetaGeneticAlgorithmBot"]