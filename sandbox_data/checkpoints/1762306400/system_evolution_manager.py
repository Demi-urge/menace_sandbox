from __future__ import annotations

"""System-wide GA-driven evolution manager."""

from dataclasses import dataclass
from typing import Iterable, Dict, List, TYPE_CHECKING
import logging

from .ga_clone_manager import GALearningManager
from .data_bot import MetricsDB
from .relevancy_radar import flagged_modules
from .relevancy_radar_service import RelevancyRadarService

if TYPE_CHECKING:  # pragma: no cover - import for type hints only
    from .structural_evolution_bot import (
        EvolutionRecord,
        StructuralEvolutionBot,
        SystemSnapshot,
    )


@dataclass
class EvolutionCycleResult:
    ga_results: Dict[str, float]
    predictions: Iterable["EvolutionRecord"]


@dataclass
class RadarRefactor:
    """Recommendation produced from relevancy radar data."""

    module: str
    action: str


class SystemEvolutionManager:
    """Coordinate GA evolution and structural predictions."""

    def __init__(
        self,
        bots: Iterable[str],
        metrics_db: MetricsDB | None = None,
        radar_service: RelevancyRadarService | None = None,
    ) -> None:
        self.bots = list(bots)
        self.ga_manager = GALearningManager(self.bots)
        from .structural_evolution_bot import StructuralEvolutionBot

        self.struct_bot = StructuralEvolutionBot()
        self.metrics_db = metrics_db or MetricsDB()
        self.last_error_rate = 0.0
        self.last_energy = 1.0
        self.radar_service = radar_service or RelevancyRadarService()
        try:
            self.radar_service.start()
        except Exception:  # pragma: no cover - best effort
            logging.getLogger(__name__).exception("radar service start failed")

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

    def radar_refactors(self) -> List[RadarRefactor]:
        """Return refactor proposals based on relevancy radar flags."""

        proposals: List[RadarRefactor] = []
        try:
            for mod, action in flagged_modules().items():
                proposals.append(RadarRefactor(module=mod, action=action))
        except Exception:  # pragma: no cover - best effort logging
            logging.getLogger(__name__).exception("radar query failed")
        return proposals

    def run_cycle(self) -> EvolutionCycleResult:
        snap: "SystemSnapshot" = self.struct_bot.take_snapshot()
        preds = self.struct_bot.predict_changes(snap)
        ga_results = {}
        for b in self.bots:
            try:
                rec = self.ga_manager.run_cycle(b)
                roi_val = rec.roi
            except Exception as exc:  # pragma: no cover - best effort logging
                logging.getLogger(__name__).warning("GA cycle failed for %s: %s", b, exc)
                roi_val = 0.0
            ga_results[b] = roi_val
            logging.getLogger(__name__).info(
                "bot_variant=%s change=%.4f reason=%s trigger=%s parent=%s",
                b,
                roi_val,
                "evolution",
                "ga_cycle",
                None,
            )
        if ga_results:
            avg_roi = float(sum(ga_results.values()) / len(ga_results))
            try:
                self.metrics_db.log_eval("evolution_cycle", "avg_roi", avg_roi)
            except Exception:
                logging.getLogger(__name__).exception("metrics logging failed")
        return EvolutionCycleResult(ga_results=ga_results, predictions=preds)


__all__ = ["SystemEvolutionManager", "EvolutionCycleResult", "RadarRefactor"]
