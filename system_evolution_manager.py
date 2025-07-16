from __future__ import annotations

"""System-wide GA-driven evolution manager."""

from dataclasses import dataclass
from typing import Iterable, Dict, Any
import logging

from .ga_clone_manager import GALearningManager
from .structural_evolution_bot import (
    StructuralEvolutionBot,
    EvolutionRecord,
    SystemSnapshot,
)
from .data_bot import MetricsDB


@dataclass
class EvolutionCycleResult:
    ga_results: Dict[str, float]
    predictions: Iterable[EvolutionRecord]


class SystemEvolutionManager:
    """Coordinate GA evolution and structural predictions."""

    def __init__(self, bots: Iterable[str], metrics_db: MetricsDB | None = None) -> None:
        self.bots = list(bots)
        self.ga_manager = GALearningManager(self.bots)
        self.struct_bot = StructuralEvolutionBot()
        self.metrics_db = metrics_db or MetricsDB()
        self.last_error_rate = 0.0
        self.last_energy = 1.0

    def run_if_signals(
        self,
        *,
        error_rate: float,
        energy: float,
        error_thresh: float = 0.2,
        energy_thresh: float = 0.3,
    ) -> EvolutionCycleResult | None:
        if (
            error_rate > error_thresh
            or energy < energy_thresh
            or error_rate > self.last_error_rate
            or energy < self.last_energy
        ):
            self.last_error_rate = error_rate
            self.last_energy = energy
            return self.run_cycle()
        self.last_error_rate = error_rate
        self.last_energy = energy
        return None

    def run_cycle(self) -> EvolutionCycleResult:
        snap: SystemSnapshot = self.struct_bot.take_snapshot()
        preds = self.struct_bot.predict_changes(snap)
        ga_results = {}
        for b in self.bots:
            rec = self.ga_manager.run_cycle(b)
            ga_results[b] = rec.roi
        if ga_results:
            avg_roi = float(sum(ga_results.values()) / len(ga_results))
            try:
                self.metrics_db.log_eval("evolution_cycle", "avg_roi", avg_roi)
            except Exception:
                logging.getLogger(__name__).exception("metrics logging failed")
        return EvolutionCycleResult(ga_results=ga_results, predictions=preds)


__all__ = ["SystemEvolutionManager", "EvolutionCycleResult"]
