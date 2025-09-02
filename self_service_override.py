from __future__ import annotations

"""Previously toggled safe mode based on ROI and errors."""

import logging
import math
import os
import subprocess
import threading
from collections import defaultdict, deque
from typing import Dict, Optional

from .resource_allocation_optimizer import ROIDB
from .data_bot import MetricsDB
from .error_bot import ErrorDB


class BaselineTracker:
    """Track moving averages and standard deviations for metrics.

    The tracker keeps a sliding window of recent values for each metric and
    determines whether the latest value deviates from the moving average by more
    than ``multiplier * std`` for a number of consecutive cycles.
    """

    def __init__(
        self,
        window: int = 20,
        deviation_multipliers: Optional[Dict[str, float]] = None,
    ) -> None:
        self.window = window
        self.values: Dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.window)
        )
        self.multipliers = deviation_multipliers or {}
        self.counts: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    def update(
        self, name: str, value: float, *, high_is_bad: bool = True
    ) -> tuple[float, float, float, int]:
        """Update tracked metric and return baseline stats.

        Parameters
        ----------
        name:
            Metric name to update.
        value:
            Latest observed value.
        high_is_bad:
            Whether higher values represent worse performance. If ``False`` the
            deviation will be calculated as ``mean - value``.

        Returns
        -------
        mean, std, deviation, consecutive_count
        """

        dq = self.values[name]
        dq.append(value)
        mean = sum(dq) / len(dq)
        if len(dq) > 1:
            std = math.sqrt(sum((x - mean) ** 2 for x in dq) / len(dq))
        else:
            std = 0.0

        deviation = (value - mean) if high_is_bad else (mean - value)
        mult = self.multipliers.get(name, 2.0)
        bad = std > 0 and deviation > mult * std
        if bad:
            self.counts[name] += 1
        else:
            self.counts[name] = 0
        return mean, std, deviation, self.counts[name]


class SelfServiceOverride:
    """Toggle system behavior without human input."""

    _baseline_tracker: BaselineTracker | None = None

    def __init__(
        self,
        roi_db: ROIDB,
        metrics_db: MetricsDB,
        error_db: Optional[ErrorDB] = None,
        *,
        tracker: BaselineTracker | None = None,
        tracker_window: int = 20,
        deviation_multipliers: Optional[Dict[str, float]] = None,
    ) -> None:
        self.roi_db = roi_db
        self.metrics_db = metrics_db
        self.error_db = error_db
        if tracker is not None:
            self.tracker = tracker
        else:
            if SelfServiceOverride._baseline_tracker is None:
                SelfServiceOverride._baseline_tracker = BaselineTracker(
                    tracker_window, deviation_multipliers
                )
            self.tracker = SelfServiceOverride._baseline_tracker
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
        roi_mean, roi_std, roi_dev, roi_bad = self.tracker.update("roi", drop)
        err_mean, err_std, err_dev, err_bad = self.tracker.update("error", err)
        self.logger.debug(
            "roi_drop=%.4f baseline=%.4f±%.4f dev=%.4f", drop, roi_mean, roi_std, roi_dev
        )
        self.logger.debug(
            "error_rate=%.4f baseline=%.4f±%.4f dev=%.4f", err, err_mean, err_std, err_dev
        )
        if roi_bad >= 3 or err_bad >= 3:
            self.logger.warning("performance deviation detected; enabling safe mode")
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
        *,
        tracker: BaselineTracker | None = None,
        tracker_window: int = 20,
        deviation_multipliers: Optional[Dict[str, float]] = None,
        revert_cmd: Optional[list[str]] = None,
    ) -> None:
        super().__init__(
            roi_db,
            metrics_db,
            error_db,
            tracker=tracker,
            tracker_window=tracker_window,
            deviation_multipliers=deviation_multipliers,
        )
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
        roi_mean, roi_std, roi_dev, roi_bad = self.tracker.update("roi", drop)
        err_mean, err_std, err_dev, err_bad = self.tracker.update("error", err)
        energy_mean, energy_std, energy_dev, energy_bad = self.tracker.update(
            "energy", energy, high_is_bad=False
        )
        self.logger.debug(
            "roi_drop=%.4f baseline=%.4f±%.4f dev=%.4f",
            drop,
            roi_mean,
            roi_std,
            roi_dev,
        )
        self.logger.debug(
            "error_rate=%.4f baseline=%.4f±%.4f dev=%.4f",
            err,
            err_mean,
            err_std,
            err_dev,
        )
        self.logger.debug(
            "energy=%.4f baseline=%.4f±%.4f dev=%.4f",
            energy,
            energy_mean,
            energy_std,
            energy_dev,
        )
        if roi_bad >= 3 or err_bad >= 3 or energy_bad >= 3:
            self.logger.warning("rollback triggered; enabling safe mode")
            self._enable_safe_mode()
            self._revert_last_patch()
        else:
            self._disable_safe_mode()


__all__ = ["BaselineTracker", "SelfServiceOverride", "AutoRollbackService"]
