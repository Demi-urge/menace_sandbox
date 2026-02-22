from __future__ import annotations

"""Automatic scheduler triggering evolution cycles."""

import threading
import time
from typing import Optional, Sequence, Any
import logging

from analytics.retrain_retrieval_ranker import (
    retrain_and_reload as retrain_ranker,
)

from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from .evolution_orchestrator import EvolutionOrchestrator
from .operational_monitor_bot import OperationalMonitoringBot


class EvolutionScheduler:
    """Monitor metrics and run :class:`EvolutionOrchestrator` when needed."""

    def __init__(
        self,
        orchestrator: EvolutionOrchestrator,
        data_bot: DataBot,
        capital_bot: CapitalManagementBot,
        *,
        monitor: Optional[OperationalMonitoringBot] = None,
        interval: int = 60,
        engagement_threshold: float = -0.2,
        anomaly_threshold: int = 1,
        engagement_window: int = 3,
        retrain_ranker_every: int = 0,
        ranker_services: Sequence[Any] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.monitor = monitor
        self.interval = interval
        self.engagement_threshold = engagement_threshold
        self.anomaly_threshold = anomaly_threshold
        self.engagement_window = engagement_window
        self.retrain_ranker_every = retrain_ranker_every
        self._ranker_services = list(ranker_services or [])
        self._engagement_history: list[float] = []
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.failure_count = 0
        self._cycle_count = 0

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            try:
                df = self.data_bot.db.fetch(limit=30)
                if getattr(df, "empty", True):
                    error_rate = 0.0
                elif hasattr(df, "mean"):
                    error_rate = float(df["errors"].mean())
                else:  # pragma: no cover - fallback
                    error_rate = float(
                        sum(r.get("errors", 0.0) for r in df) / max(len(df), 1)
                    )
                error_spikes = []
                try:
                    error_spikes = DataBot.detect_anomalies(df, "errors")
                except Exception as exc:
                    error_spikes = []
                    self.logger.exception("detect_anomalies failed: %s", exc)
                    self.failure_count += 1
                energy = self.capital_bot.energy_score(
                    load=0.0,
                    success_rate=1.0,
                    deploy_eff=1.0,
                    failure_rate=error_rate,
                )
                anomalies = []
                if self.monitor:
                    try:
                        anomalies = self.monitor.detect_anomalies("menace")
                    except Exception as exc:
                        anomalies = []
                        self.logger.exception("monitor anomalies failed: %s", exc)
                        self.failure_count += 1
                anomaly_count = len(anomalies)
                engagement_delta = 0.0
                try:
                    engagement_delta = self.data_bot.engagement_delta(limit=50)
                except Exception as exc:
                    engagement_delta = 0.0
                    self.logger.exception("engagement delta failed: %s", exc)
                    self.failure_count += 1
                self._engagement_history.append(engagement_delta)
                if len(self._engagement_history) > self.engagement_window:
                    self._engagement_history.pop(0)
                engagement_trend = sum(self._engagement_history) / len(self._engagement_history)
                if error_spikes:
                    try:
                        self.orchestrator.improvement_engine.run_cycle()
                    except Exception as exc:
                        self.logger.exception("improvement cycle failed: %s", exc)
                        self.failure_count += 1
                self.orchestrator.run_cycle()
                self._cycle_count += 1
                if (
                    self.retrain_ranker_every > 0
                    and self._cycle_count >= self.retrain_ranker_every
                ):
                    try:
                        retrain_ranker(self._ranker_services)
                    except Exception as exc:
                        self.logger.exception("ranker retrain failed: %s", exc)
                        self.failure_count += 1
                    self._cycle_count = 0
            except Exception as exc:
                self.logger.exception("evolution cycle failed: %s", exc)
                self.failure_count += 1
            time.sleep(self.interval)

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
