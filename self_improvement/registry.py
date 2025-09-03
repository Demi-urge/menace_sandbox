"""Registry for coordinating multiple self‑improvement engines."""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

from ..model_automation_pipeline import AutomationResult
from ..capital_management_bot import CapitalManagementBot
from ..data_bot import DataBot

try:  # pragma: no cover - simplified environments
    from ..logging_utils import get_logger
except Exception:  # pragma: no cover - fallback
    import logging

    def get_logger(name: str) -> logging.Logger:  # type: ignore
        return logging.getLogger(name)

from .engine import SelfImprovementEngine
from .baseline_tracker import TRACKER as BASELINE_TRACKER
from .init import settings


class ImprovementEngineRegistry:
    """Register and run multiple :class:`SelfImprovementEngine` instances."""

    def __init__(self) -> None:
        self.engines: dict[str, SelfImprovementEngine] = {}
        self.logger = get_logger(self.__class__.__name__)
        self.baseline_tracker = BASELINE_TRACKER
        self._lag_count = 0

    def register_engine(self, name: str, engine: SelfImprovementEngine) -> None:
        """Add *engine* under *name*."""
        self.engines[name] = engine

    def unregister_engine(self, name: str) -> None:
        """Remove the engine referenced by *name* if present."""
        self.engines.pop(name, None)

    def run_all_cycles(self, energy: int = 1) -> dict[str, AutomationResult]:
        """Execute ``run_cycle`` on all registered engines."""
        results: dict[str, AutomationResult] = {}
        for name, eng in self.engines.items():
            if eng._should_trigger():
                res = eng.run_cycle(energy=energy)
                results[name] = res
        return results

    async def run_all_cycles_async(self, energy: int = 1) -> dict[str, AutomationResult]:
        """Asynchronously execute ``run_cycle`` on all registered engines."""

        async def _run(name: str, eng: SelfImprovementEngine):
            if eng._should_trigger():
                res = await asyncio.to_thread(eng.run_cycle, energy=energy)
                return name, res
            return None

        tasks = [asyncio.create_task(_run(n, e)) for n, e in self.engines.items()]
        results: dict[str, AutomationResult] = {}
        for t in tasks:
            out = await t
            if out:
                name, res = out
                results[name] = res
        return results

    def schedule_all(
        self, energy: int = 1, *, loop: asyncio.AbstractEventLoop | None = None
    ) -> list[asyncio.Task]:
        """Start schedules for all engines and return the created tasks."""
        tasks: list[asyncio.Task] = []
        for eng in self.engines.values():
            tasks.append(eng.schedule(energy=energy, loop=loop))
        return tasks

    async def shutdown_all(self) -> None:
        """Gracefully stop all running schedules."""
        for eng in self.engines.values():
            await eng.shutdown_schedule()

    def autoscale(
        self,
        *,
        capital_bot: CapitalManagementBot,
        data_bot: DataBot,
        factory: Callable[[str], SelfImprovementEngine],
        max_engines: int = 5,
        min_engines: int = 1,
        create_energy: float | None = None,
        remove_energy: float | None = None,
        roi_threshold: float | None = None,
        cost_per_engine: float = 0.0,
        approval_callback: Optional[Callable[[], bool]] = None,
        max_instances: Optional[int] = None,
    ) -> None:
        """Dynamically create or remove engines based on ROI and resources."""
        try:
            energy = capital_bot.energy_score(
                load=0.0,
                success_rate=1.0,
                deploy_eff=1.0,
                failure_rate=0.0,
            )
        except Exception as exc:
            self.logger.exception("autoscale energy check failed: %s", exc)
            energy = 0.0
        try:
            trend = data_bot.long_term_roi_trend(limit=200)
        except Exception as exc:
            self.logger.exception("autoscale trend fetch failed: %s", exc)
            trend = 0.0

        energy_avg = self.baseline_tracker.get("energy")
        roi_avg = self.baseline_tracker.get("roi")
        energy_std = self.baseline_tracker.std("energy")
        roi_std = self.baseline_tracker.std("roi")
        energy_dev = energy - energy_avg
        roi_dev = trend - roi_avg
        self.baseline_tracker.update(energy=energy, roi=trend)
        roi_tol = getattr(getattr(settings, "roi", None), "deviation_tolerance", 0.0)
        syn_tol = getattr(getattr(settings, "synergy", None), "deviation_tolerance", 0.0)

        if energy_dev < -syn_tol or roi_dev < -roi_tol:
            self._lag_count += 1
        else:
            self._lag_count = 0
        if self._lag_count >= 3:
            if len(self.engines) > min_engines:
                name = next(iter(self.engines))
                self.unregister_engine(name)
            else:
                self.logger.warning(
                    "ROI or energy below baseline for three cycles; escalating"
                )
            self._lag_count = 0
            return

        if not capital_bot.check_budget():
            return
        if max_instances is not None and len(self.engines) >= max_instances:
            return

        projected_roi = trend - roi_avg - cost_per_engine
        create_mult = (
            create_energy
            if create_energy is not None
            else getattr(settings, "autoscale_create_dev_multiplier", 0.8)
        )
        remove_mult = (
            remove_energy
            if remove_energy is not None
            else getattr(settings, "autoscale_remove_dev_multiplier", 0.3)
        )
        roi_mult = (
            roi_threshold
            if roi_threshold is not None
            else getattr(settings, "autoscale_roi_dev_multiplier", 0.0)
        )
        create_thresh = energy_avg + create_mult * energy_std
        remove_thresh = energy_avg - remove_mult * energy_std
        roi_high = roi_avg + roi_mult * roi_std
        roi_low = roi_avg - roi_mult * roi_std
        if (
            energy >= create_thresh
            and trend >= roi_high
            and projected_roi > roi_tol
            and len(self.engines) < max_engines
        ):
            if approval_callback and not approval_callback():
                return
            name = f"engine{len(self.engines)}"
            self.register_engine(name, factory(name))
            return
        if (
            energy <= remove_thresh
            or trend <= roi_low
            or projected_roi <= -roi_tol
        ) and len(self.engines) > min_engines:
            name = next(iter(self.engines))
            self.unregister_engine(name)


def auto_x(
    engines: list[SelfImprovementEngine] | None = None,
    *,
    energy: int = 1,
) -> dict[str, AutomationResult]:
    """Convenience helper to run a self‑improvement cycle."""
    registry = ImprovementEngineRegistry()
    if engines:
        for idx, eng in enumerate(engines):
            registry.register_engine(f"engine{idx}", eng)
    else:
        registry.register_engine("default", SelfImprovementEngine())
    results = registry.run_all_cycles(energy=energy)
    return results


__all__ = ["ImprovementEngineRegistry", "auto_x"]
