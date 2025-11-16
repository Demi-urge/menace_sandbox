from __future__ import annotations

"""Utilities for monitoring the SynergyExporter."""

import json
import os
import threading
import time
import urllib.request

if os.getenv("SANDBOX_CENTRAL_LOGGING") == "1":
    from logging_utils import setup_logging

    setup_logging()

from menace.audit_trail import AuditTrail
from menace.synergy_exporter import SynergyExporter
try:
    from .metrics_exporter import Gauge
except Exception:  # pragma: no cover - module may not be a package
    from menace.metrics_exporter import Gauge  # type: ignore
from logging_utils import get_logger
from alert_dispatcher import dispatch_alert

EXPORTER_CHECK_INTERVAL = float(os.getenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "10"))
ALERT_THRESHOLD = int(os.getenv("SYNERGY_ALERT_THRESHOLD", "5"))

logger = get_logger(__name__)

synergy_exporter_restarts_total = Gauge(
    "synergy_exporter_restarts_total",
    "Total number of SynergyExporter restarts",
)
synergy_trainer_restarts_total = Gauge(
    "synergy_trainer_restarts_total",
    "Total number of SynergyAutoTrainer restarts",
)


def _exporter_health_ok(exp: SynergyExporter, *, max_age: float = 30.0) -> bool:
    """Return ``True`` if the exporter health endpoint responds and is fresh."""
    if exp._thread is None or not exp._thread.is_alive():
        return False
    if exp.health_port is None:
        return False
    try:
        with urllib.request.urlopen(
            f"http://localhost:{exp.health_port}/health", timeout=3
        ) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode())
            updated = data.get("updated")
            if updated is None:
                return False
            if time.time() - float(updated) > max_age:
                return False
    except Exception:
        return False
    return True


class ExporterMonitor:
    """Background monitor to keep the exporter running."""

    def __init__(
        self,
        exporter: SynergyExporter,
        log: AuditTrail,
        *,
        interval: float = EXPORTER_CHECK_INTERVAL,
    ) -> None:
        self.exporter = exporter
        self.log = log
        self.interval = float(interval)
        self.restart_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self.exporter.stop()
        except Exception:
            logger.exception("failed to stop synergy exporter")

    def _restart(self) -> None:
        try:
            self.exporter.stop()
        except Exception:
            logger.exception("failed to stop synergy exporter")
        try:
            self.exporter = SynergyExporter(
                history_file=str(self.exporter.history_file),
                interval=self.exporter.interval,
                port=self.exporter.port,
            )
            self.exporter.start()
            self.restart_count += 1
            try:
                synergy_exporter_restarts_total.inc()
            except Exception:
                pass
            self.log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "exporter_restarted",
                    "restart_count": self.restart_count,
                }
            )
            if self.restart_count > ALERT_THRESHOLD:
                try:
                    dispatch_alert(
                        "synergy_exporter_restart_threshold",
                        2,
                        "SynergyExporter restarted repeatedly",
                        {
                            "component": "exporter",
                            "restarts": self.restart_count,
                        },
                    )
                except Exception:
                    logger.exception("failed to dispatch exporter alert")
        except Exception as exc:
            logger.warning("failed to restart synergy exporter: %s", exc)
            self.log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "exporter_restart_failed",
                    "error": str(exc),
                }
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not _exporter_health_ok(self.exporter):
                self._restart()
            self._stop.wait(self.interval)


TRAINER_CHECK_INTERVAL = float(os.getenv("SYNERGY_TRAINER_CHECK_INTERVAL", "10"))


class AutoTrainerMonitor:
    """Background monitor to keep the auto trainer running."""

    def __init__(
        self,
        trainer: "SynergyAutoTrainer",
        log: AuditTrail,
        *,
        interval: float = TRAINER_CHECK_INTERVAL,
    ) -> None:
        from menace.synergy_auto_trainer import SynergyAutoTrainer

        self.trainer: SynergyAutoTrainer = trainer
        self.log = log
        self.interval = float(interval)
        self.restart_count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self.trainer.stop()
        except Exception:
            logger.exception("failed to stop synergy auto trainer")

    def _restart(self) -> None:
        from menace.synergy_auto_trainer import SynergyAutoTrainer

        try:
            self.trainer.stop()
        except Exception:
            logger.exception("failed to stop synergy auto trainer")
        try:
            self.trainer = SynergyAutoTrainer(
                history_file=str(self.trainer.history_file),
                weights_file=str(self.trainer.weights_file),
                interval=self.trainer.interval,
                progress_file=str(self.trainer.progress_file),
            )
            self.trainer.start()
            self.restart_count += 1
            try:
                synergy_trainer_restarts_total.inc()
            except Exception:
                pass
            self.log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "auto_trainer_restarted",
                    "restart_count": self.restart_count,
                }
            )
            if self.restart_count > ALERT_THRESHOLD:
                try:
                    dispatch_alert(
                        "synergy_trainer_restart_threshold",
                        2,
                        "SynergyAutoTrainer restarted repeatedly",
                        {
                            "component": "auto_trainer",
                            "restarts": self.restart_count,
                        },
                    )
                except Exception:
                    logger.exception("failed to dispatch trainer alert")
        except Exception as exc:
            logger.warning("failed to restart synergy auto trainer: %s", exc)
            self.log.record(
                {
                    "timestamp": int(time.time()),
                    "event": "auto_trainer_restart_failed",
                    "error": str(exc),
                }
            )

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.trainer._thread is None or not self.trainer._thread.is_alive():
                self._restart()
            self._stop.wait(self.interval)
