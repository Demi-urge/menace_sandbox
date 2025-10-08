"""Registry for coordinating multiple self‑improvement engines."""

from __future__ import annotations

import asyncio
from typing import Callable, Optional

from menace_sandbox.model_automation_pipeline import AutomationResult
from menace_sandbox.capital_management_bot import CapitalManagementBot
from menace_sandbox.data_bot import DataBot

try:  # pragma: no cover - simplified environments
    from menace_sandbox.logging_utils import get_logger
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

        # Snapshot baseline statistics before recording current metrics
        energy_avg = self.baseline_tracker.get("energy")
        roi_avg = self.baseline_tracker.get("roi")
        pass_rate_avg = self.baseline_tracker.get("pass_rate")
        entropy_avg = self.baseline_tracker.get("entropy")
        momentum_avg = self.baseline_tracker.get("momentum")
        energy_std = max(self.baseline_tracker.std("energy"), 1e-6)
        roi_std = max(self.baseline_tracker.std("roi"), 1e-6)
        pass_rate_std = max(self.baseline_tracker.std("pass_rate"), 1e-6)
        entropy_std = max(self.baseline_tracker.std("entropy"), 1e-6)
        momentum_std = max(self.baseline_tracker.std("momentum"), 1e-6)

        pass_rate = self.baseline_tracker.current("pass_rate")
        entropy = self.baseline_tracker.current("entropy")

        # Record latest metrics including pass rate and entropy
        self.baseline_tracker.update(
            energy=energy, roi=trend, pass_rate=pass_rate, entropy=entropy
        )

        energy_dev = energy - energy_avg
        roi_dev = trend - roi_avg
        pass_rate_dev = pass_rate - pass_rate_avg
        entropy_dev = entropy - entropy_avg
        momentum = self.baseline_tracker.current("momentum")
        momentum_dev = momentum - momentum_avg

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
                    "ROI or energy below baseline for three cycles; escalating",
                )
            self._lag_count = 0
            return

        if not capital_bot.check_budget():
            return
        if max_instances is not None and len(self.engines) >= max_instances:
            return

        projected_roi = trend - roi_avg - cost_per_engine

        # Normalised deviations (z-scores)
        energy_z = energy_dev / energy_std
        roi_z = roi_dev / roi_std
        pass_rate_z = pass_rate_dev / pass_rate_std
        entropy_z = entropy_dev / entropy_std
        momentum_z = momentum_dev / momentum_std

        # Combined delta score emphasising ROI, pass rate and momentum while
        # penalising entropy spikes
        score = roi_z + pass_rate_z + momentum_z - entropy_z
        score_avg = self.baseline_tracker.get("delta_score")
        score_std = self.baseline_tracker.std("delta_score")
        self.baseline_tracker.update(delta_score=score, record_momentum=False)

        # Adapt thresholds using recent volatility and momentum
        create_weight = (
            create_energy
            if create_energy is not None
            else 1.0 + max(momentum_z, 0.0)
        )
        remove_weight = (
            remove_energy
            if remove_energy is not None
            else 1.0 + max(-momentum_z, 0.0)
        )
        roi_weight = roi_threshold if roi_threshold is not None else 1.0
        create_thresh = energy_avg + create_weight * energy_std
        remove_thresh = energy_avg - remove_weight * energy_std
        roi_high = roi_avg + roi_weight * roi_std
        roi_low = roi_avg - roi_weight * roi_std

        if (
            energy >= create_thresh
            and trend >= roi_high
            and projected_roi > roi_tol
            and score > score_avg + score_std
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
            or score < score_avg - score_std
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
        registry.register_engine(
            "default",
            SelfImprovementEngine(context_builder=_auto_context_builder()),
        )
    results = registry.run_all_cycles(energy=energy)
    return results


__all__ = ["ImprovementEngineRegistry", "auto_x"]


def _auto_context_builder():
    from context_builder_util import create_context_builder

    return create_context_builder()
