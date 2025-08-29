from __future__ import annotations

"""Previously toggled safe mode based on ROI and errors."""

import logging
import os
import subprocess
import threading
from typing import Optional

from .resource_allocation_optimizer import ROIDB
from .data_bot import MetricsDB
from .error_bot import ErrorDB


_ROI_ENV = "SAFE_MODE_ROI_DROP"
_ERR_ENV = "SAFE_MODE_ERROR_THRESHOLD"


class SelfServiceOverride:
    """Toggle system behavior without human input."""

    def __init__(
        self,
        roi_db: ROIDB,
        metrics_db: MetricsDB,
        error_db: Optional[ErrorDB] = None,
        roi_drop: float | None = None,
        error_threshold: float | None = None,
    ) -> None:
        self.roi_db = roi_db
        self.metrics_db = metrics_db
        self.error_db = error_db
        if roi_drop is None:
            roi_drop = float(os.getenv(_ROI_ENV, "0.1"))
        if error_threshold is None:
            error_threshold = float(os.getenv(_ERR_ENV, "0.25"))
        self.roi_drop = roi_drop
        self.error_threshold = error_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self._safe_mode = os.environ.get("MENACE_SAFE") == "1"

    # ------------------------------------------------------------------
    def _calc_roi_drop(self) -> float:
        df = self.roi_db.history(limit=2)
        if len(df) < 2:
            return 0.0
        prev = float(df.iloc[1]["revenue"] - df.iloc[1]["api_cost"])
        curr = float(df.iloc[0]["revenue"] - df.iloc[0]["api_cost"])
        if prev == 0:
            return 0.0
        return (prev - curr) / prev

    def _error_rate(self) -> float:
        try:
            df = self.metrics_db.fetch(limit=10)
            if hasattr(df, "empty"):
                return float(df["errors"].mean()) if not df.empty else 0.0
            if isinstance(df, list) and df:
                return sum(float(r.get("errors", 0.0)) for r in df) / len(df)
        except Exception:
            self.logger.exception("failed to fetch metrics")
        return 0.0

    # ------------------------------------------------------------------
    def _enable_safe_mode(self) -> None:
        if not self._safe_mode:
            os.environ["MENACE_SAFE"] = "1"
            self._safe_mode = True

    def _disable_safe_mode(self) -> None:
        if self._safe_mode:
            os.environ["MENACE_SAFE"] = "0"
            self._safe_mode = False

    def adjust(self) -> None:
        drop = self._calc_roi_drop()
        err = self._error_rate()
        if drop >= self.roi_drop or err >= self.error_threshold:
            self.logger.warning("performance drop detected; enabling safe mode")
            self._enable_safe_mode()
        else:
            self._disable_safe_mode()

    # ------------------------------------------------------------------
    def run_continuous(
        self, interval: float = 60.0, *, stop_event: threading.Event | None = None
    ) -> threading.Thread:
        """Invoke :meth:`adjust` periodically in a background thread."""
        if hasattr(self, "_thread") and self._thread.is_alive():
            return self._thread
        self._stop = stop_event or threading.Event()

        def _loop() -> None:
            while not self._stop.is_set():
                try:
                    self.adjust()
                except Exception:
                    self.logger.exception("adjust failed")
                if self._stop.wait(interval):
                    break

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        if hasattr(self, "_stop"):
            self._stop.set()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1.0)


class AutoRollbackService(SelfServiceOverride):
    """Revert the last patch when thresholds are crossed."""

    def __init__(
        self,
        roi_db: ROIDB,
        metrics_db: MetricsDB,
        error_db: Optional[ErrorDB] = None,
        roi_drop: float = 0.1,
        error_threshold: float = 0.25,
        energy_threshold: float = 0.3,
        revert_cmd: Optional[list[str]] = None,
    ) -> None:
        super().__init__(roi_db, metrics_db, error_db, roi_drop, error_threshold)
        self.energy_threshold = energy_threshold
        self.revert_cmd = revert_cmd or ["git", "revert", "--no-edit", "HEAD"]

    def _energy_score(self) -> float:
        try:
            rows = self.metrics_db.fetch_eval("system")
            for cycle, metric, value, _ in reversed(rows):
                if metric == "avg_energy_score":
                    return float(value)
        except Exception:
            self.logger.exception("failed to fetch energy score")
        return 1.0

    def _revert_last_patch(self) -> None:
        try:
            subprocess.run(
                self.revert_cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.logger.warning("last patch reverted")
        except Exception:
            self.logger.exception("git revert failed")

    def adjust(self) -> None:  # type: ignore[override]
        drop = self._calc_roi_drop()
        err = self._error_rate()
        energy = self._energy_score()
        if drop >= self.roi_drop or err >= self.error_threshold or energy < self.energy_threshold:
            self.logger.warning(
                "rollback triggered; enabling safe mode"
            )
            self._enable_safe_mode()
            self._revert_last_patch()
        else:
            self._disable_safe_mode()


__all__ = ["SelfServiceOverride", "AutoRollbackService"]
