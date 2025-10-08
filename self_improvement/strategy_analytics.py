from __future__ import annotations

"""Utilities for computing prompt strategy analytics.

This module aggregates success and failure logs to derive per-strategy
return-on-investment (ROI) metrics.  The resulting statistics are applied to
:class:`~self_improvement.prompt_strategy_manager.PromptStrategyManager`
which is then able to select the most promising strategy via its
:meth:`best_strategy` method.
"""

from pathlib import Path
from typing import Any, Dict
import json
import time

try:  # pragma: no cover - allow flat imports
    from menace_sandbox.dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path  # type: ignore

from sandbox_settings import SandboxSettings
from .prompt_strategy_manager import PromptStrategyManager


class StrategyAnalytics:
    """Aggregate per-strategy ROI statistics from prompt logs."""

    def __init__(
        self,
        *,
        manager: PromptStrategyManager | None = None,
        success_log: str | Path | None = None,
        failure_log: str | Path | None = None,
        refresh_interval: float = 300.0,
    ) -> None:
        settings = SandboxSettings()
        self.success_log = Path(resolve_path(success_log or settings.prompt_success_log_path))
        self.failure_log = Path(resolve_path(failure_log or settings.prompt_failure_log_path))
        self.manager = manager or PromptStrategyManager()
        self.refresh_interval = float(refresh_interval)
        self._last_refresh = 0.0

    # ------------------------------------------------------------------
    def refresh_if_stale(self) -> None:
        """Refresh analytics if the configured interval has elapsed."""

        if time.time() - self._last_refresh >= self.refresh_interval:
            self.refresh()

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Read logs, compute ROI stats and update ``manager``."""

        stats: Dict[str, Dict[str, Any]] = {}
        for path, success in ((self.success_log, True), (self.failure_log, False)):
            if not path.exists():
                continue
            try:
                fh = path.open("r", encoding="utf-8")
            except Exception:  # pragma: no cover - best effort
                continue
            with fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except Exception:
                        continue
                    strat = data.get("strategy")
                    if not strat:
                        strat = data.get("metadata", {}).get("strategy")
                    if not strat:
                        continue
                    roi = 0.0
                    meta = data.get("roi_meta")
                    if isinstance(meta, dict):
                        roi = float(meta.get("roi_delta") or meta.get("roi", 0.0))
                    elif data.get("roi") is not None:
                        try:
                            roi = float(data.get("roi", 0.0))
                        except Exception:
                            roi = 0.0
                    ts = data.get("timestamp") or data.get("time") or time.time()
                    try:
                        ts = float(ts)
                    except Exception:
                        ts = time.time()
                    rec = stats.setdefault(
                        str(strat),
                        {
                            "total": 0,
                            "success": 0,
                            "roi_sum": 0.0,
                            "weighted_roi_sum": 0.0,
                            "weight_sum": 0.0,
                            "records": [],
                        },
                    )
                    rec["total"] += 1
                    if success:
                        rec["success"] += 1
                    rec["roi_sum"] += roi
                    rec["weighted_roi_sum"] += roi
                    rec["weight_sum"] += 1.0
                    rec.setdefault("records", []).append(
                        {"ts": ts, "roi": roi, "success": success}
                    )
        self.manager.stats = stats
        try:
            self.manager._save_stats()
        except Exception:  # pragma: no cover - best effort
            pass
        self._last_refresh = time.time()


__all__ = ["StrategyAnalytics"]
