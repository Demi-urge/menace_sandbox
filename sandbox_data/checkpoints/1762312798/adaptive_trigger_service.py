from __future__ import annotations

"""Service that publishes evolution events based on metrics."""

import threading
import time
from typing import Callable, Dict, List, Optional, Tuple
import logging

from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from .unified_event_bus import UnifiedEventBus


class AdaptiveTriggerService:
    """Monitor metrics and emit events when thresholds are crossed."""

    def __init__(
        self,
        data_bot: DataBot,
        capital_bot: CapitalManagementBot,
        event_bus: UnifiedEventBus,
        *,
        interval: int = 60,
        error_threshold: float = 0.2,
        energy_threshold: float = 0.3,
        load_getter: Optional[Callable[[], float]] = None,
        success_getter: Optional[Callable[[], float]] = None,
        deploy_getter: Optional[Callable[[], float]] = None,
        threshold_strategy: Optional[
            Callable[[str, float, List[float]], float]
        ] = None,
        extra_metrics: Optional[
            Dict[str, Tuple[Callable[[], float], float, str]]
        ] = None,
        history_len: int = 20,
    ) -> None:
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.event_bus = event_bus
        self.interval = interval
        self.error_threshold = error_threshold
        self.energy_threshold = energy_threshold
        self.load_getter = load_getter
        self.success_getter = success_getter
        self.deploy_getter = deploy_getter
        self.threshold_strategy = threshold_strategy
        self.extra_metrics = extra_metrics or {}
        self.history_len = history_len
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.failure_count = 0
        self.error_history: List[float] = []
        self.energy_history: List[float] = []
        self.metric_histories: Dict[str, List[float]] = {
            name: [] for name in self.extra_metrics
        }

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            error_rate = 0.0
            try:
                df = self.data_bot.db.fetch(limit=30)
                if hasattr(df, "empty"):
                    if not getattr(df, "empty", True):
                        error_rate = float(df["errors"].mean())
                elif isinstance(df, list) and df:
                    error_rate = float(
                        sum(r.get("errors", 0.0) for r in df) / len(df)
                    )
            except Exception as exc:
                error_rate = 0.0
                self.logger.exception("fetch errors failed: %s", exc)
                self.failure_count += 1
            try:
                load = (
                    self.load_getter() if self.load_getter else self._default_load()
                )
                success = (
                    self.success_getter()
                    if self.success_getter
                    else self._default_success_rate()
                )
                deploy = (
                    self.deploy_getter()
                    if self.deploy_getter
                    else self._default_deploy_eff()
                )
                energy = self.capital_bot.energy_score(
                    load=load,
                    success_rate=success,
                    deploy_eff=deploy,
                    failure_rate=error_rate,
                )
            except Exception as exc:
                energy = 1.0
                self.logger.exception("energy score failed: %s", exc)
                self.failure_count += 1
            self._update_history(self.error_history, error_rate)
            self._update_history(self.energy_history, energy)
            err_thr = (
                self.threshold_strategy(
                    "error_rate", self.error_threshold, self.error_history
                )
                if self.threshold_strategy
                else self.error_threshold
            )
            energy_thr = (
                self.threshold_strategy(
                    "energy_score", self.energy_threshold, self.energy_history
                )
                if self.threshold_strategy
                else self.energy_threshold
            )
            if error_rate > err_thr:
                self.event_bus.publish(
                    "evolve:self_improve",
                    self._build_payload({"errors": error_rate}),
                )
            if energy < energy_thr:
                self.event_bus.publish(
                    "evolve:system", self._build_payload({"energy": energy})
                )
            for name, (getter, thr, topic) in self.extra_metrics.items():
                try:
                    value = getter()
                except Exception as exc:
                    value = 0.0
                    self.logger.exception("%s metric failed: %s", name, exc)
                    self.failure_count += 1
                hist = self.metric_histories.setdefault(name, [])
                self._update_history(hist, value)
                limit = (
                    self.threshold_strategy(name, thr, hist)
                    if self.threshold_strategy
                    else thr
                )
                if value > limit:
                    self.event_bus.publish(
                        topic, self._build_payload({name: value})
                    )
            time.sleep(self.interval)
        self.cleanup()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=0)
            self._thread = None

    # ------------------------------------------------------------------
    def _default_load(self) -> float:
        try:
            df = self.data_bot.db.fetch(limit=30)
            if hasattr(df, "empty"):
                if not getattr(df, "empty", True) and "cpu" in df:
                    return float(df["cpu"].mean()) / 100.0
            elif isinstance(df, list) and df:
                vals = [r.get("cpu", 0.0) for r in df if "cpu" in r]
                if vals:
                    return float(sum(vals) / len(vals)) / 100.0
        except Exception as exc:
            self.logger.exception("load metric failed: %s", exc)
            self.failure_count += 1
        return 0.0

    def _default_success_rate(self) -> float:
        try:
            patch_db = getattr(self.data_bot, "patch_db", None)
            if patch_db:
                return float(patch_db.success_rate())
        except Exception as exc:
            self.logger.exception("success rate metric failed: %s", exc)
            self.failure_count += 1
        return 1.0

    def _default_deploy_eff(self) -> float:
        try:
            patch_db = getattr(self.data_bot, "patch_db", None)
            if patch_db:
                return float(patch_db.success_rate())
        except Exception as exc:
            self.logger.exception("deploy efficiency metric failed: %s", exc)
            self.failure_count += 1
        return 1.0

    def _update_history(self, hist: List[float], value: float) -> None:
        hist.append(value)
        if len(hist) > self.history_len:
            hist.pop(0)

    def _build_payload(self, data: Dict[str, float]) -> Dict[str, float]:
        payload = {
            **data,
            "timestamp": time.time(),
            "interval": self.interval,
            "failure_count": self.failure_count,
        }
        return payload

    def cleanup(self) -> None:
        try:
            close = getattr(self.event_bus, "close", None)
            if close:
                close()
        except Exception as exc:
            self.logger.exception("cleanup failed: %s", exc)


__all__ = ["AdaptiveTriggerService"]
