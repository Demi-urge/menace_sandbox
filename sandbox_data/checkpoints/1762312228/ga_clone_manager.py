"""Module providing GA clones for administrative bots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from typing import TYPE_CHECKING

from .genetic_algorithm_bot import GeneticAlgorithmBot, GARecord, GAStore
from .ga_prediction_bot import GAPredictionBot, TemplateDB, TemplateEntry
from .menace_memory_manager import MenaceMemoryManager, MemoryEntry
from .data_bot import DataBot

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .capital_management_bot import CapitalManagementBot


@dataclass
class GALineage:
    ga_bot: GeneticAlgorithmBot
    pred_bot: GAPredictionBot
    version: int = 0


class GALearningManager:
    """Instantiate GA bots for each admin bot and track evolution history."""

    def __init__(
        self,
        bots: Iterable[str],
        memory: MenaceMemoryManager | None = None,
        *,
        data_bot: DataBot | None = None,
        capital_bot: "CapitalManagementBot" | None = None,
    ) -> None:
        self.memory = memory or MenaceMemoryManager()
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.lineage: Dict[str, GALineage] = {}
        for b in bots:
            self.lineage[b] = GALineage(
                ga_bot=GeneticAlgorithmBot(
                    data_bot=self.data_bot,
                    capital_bot=self.capital_bot,
                    name=f"ga_{b}",
                ),
                pred_bot=GAPredictionBot(
                    [[0, 0, 0]],
                    [0],
                    data_bot=self.data_bot,
                    capital_bot=self.capital_bot,
                    name=f"gap_{b}",
                ),
            )

    def run_evolution(self, bot: str, generations: int = 1) -> GARecord:
        line = self.lineage[bot]
        record = line.ga_bot.evolve(generations)
        line.version += 1
        self.memory.log(
            MemoryEntry(
                key=f"{bot}_ga", data=str(record.params), version=line.version, tags="ga"
            )
        )
        return record

    def run_prediction_evolution(self, bot: str, generations: int = 1) -> TemplateEntry:
        line = self.lineage[bot]
        entry = line.pred_bot.evolve(generations)
        line.version += 1
        self.memory.log(
            MemoryEntry(
                key=f"{bot}_gapred", data=str(entry.params), version=line.version, tags="ga_pred"
            )
        )
        return entry

    def run_cycle(self, bot: str, generations: int = 1) -> GARecord:
        """Run evolution and prediction, feeding results back into the GA."""
        ga_rec = self.run_evolution(bot, generations)
        pred_entry = self.run_prediction_evolution(bot, generations)
        self.lineage[bot].ga_bot.inject([pred_entry.params])
        return ga_rec

    def lineage_version(self, bot: str) -> int:
        return self.lineage[bot].version

__all__ = ["GALearningManager"]
