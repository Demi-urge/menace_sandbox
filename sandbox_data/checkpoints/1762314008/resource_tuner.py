from __future__ import annotations

"""Resource tuning utilities for sandbox cycles."""

import logging

from logging_utils import get_logger, setup_logging
from typing import Any, Dict, List

# The sandbox_runner package may be executed as a top-level module. In that
# situation there is no parent package, so relative imports referring to the
# parent will fail.  Use an absolute import instead to ensure the constants are
# available regardless of how the module is executed.
# Importing from the ``menace`` package ensures that relative imports inside
# :mod:`environment_generator` resolve correctly regardless of whether this
# package is executed as ``menace`` or ``menace_sandbox``.
from menace.environment_generator import _CPU_LIMITS, _MEMORY_LIMITS

logger = get_logger(__name__)


class ResourceTuner:
    """Suggest CPU and memory limits based on ROI history."""

    def __init__(self, window: int = 3) -> None:
        self.window = max(1, int(window))

    # --------------------------------------------------------------
    def _avg(self, seq: List[float]) -> float:
        return sum(seq) / len(seq) if seq else 0.0

    def adjust(
        self, tracker: Any, presets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return ``presets`` updated according to ``tracker`` history."""

        if not getattr(tracker, "roi_history", None):
            return presets

        roi_vals = tracker.roi_history[-self.window :]
        deltas = [roi_vals[i] - roi_vals[i - 1] for i in range(1, len(roi_vals))]
        avg_delta = self._avg(deltas)

        res_hist = getattr(tracker, "resource_metrics", [])[-self.window :]
        avg_cpu = self._avg([r[0] for r in res_hist])
        avg_mem = self._avg([r[1] for r in res_hist])

        def _next(seq: List[str], cur: str, up: bool) -> str:
            cur = str(cur)
            try:
                idx = [str(x) for x in seq].index(cur)
            except ValueError:
                idx = 0
            idx = min(idx + 1, len(seq) - 1) if up else max(idx - 1, 0)
            return seq[idx]

        for p in presets:
            cpu = str(p.get("CPU_LIMIT", "1"))
            mem = str(p.get("MEMORY_LIMIT", _MEMORY_LIMITS[0]))
            if avg_delta > 0:
                if avg_cpu > 0.8:
                    cpu = _next(_CPU_LIMITS, cpu, True)
                if avg_mem > 0.8:
                    mem = _next(_MEMORY_LIMITS, mem, True)
            elif avg_delta < 0:
                if avg_cpu < 0.2:
                    cpu = _next(_CPU_LIMITS, cpu, False)
                if avg_mem < 0.2:
                    mem = _next(_MEMORY_LIMITS, mem, False)
            p["CPU_LIMIT"] = cpu
            p["MEMORY_LIMIT"] = mem
        return presets


__all__ = ["ResourceTuner"]


if __name__ == "__main__":  # pragma: no cover - manual invocation
    setup_logging()
