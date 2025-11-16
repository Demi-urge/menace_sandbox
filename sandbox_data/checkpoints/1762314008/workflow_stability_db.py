from __future__ import annotations

"""Persist workflow stability status across runs."""

from pathlib import Path
from typing import Dict, Tuple
import json
try:  # pragma: no cover - allow package and flat imports
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path  # type: ignore

try:  # pragma: no cover - compute default path via resolve_path
    _DEFAULT_PATH = resolve_path("sandbox_data/stable_workflows.json")
except FileNotFoundError:  # pragma: no cover - directory exists but file may not
    _DEFAULT_PATH = resolve_path("sandbox_data") / "stable_workflows.json"


class WorkflowStabilityDB:
    """Store workflow IDs deemed stable with their last observed ROI.

    The data is backed by a simple JSON file so state survives process restarts.
    When ``is_stable`` is invoked with a ``current_roi`` and the change exceeds
    ``threshold`` the workflow is automatically cleared from the stable set.
    """

    def __init__(self, path: str | Path = _DEFAULT_PATH) -> None:
        try:
            self.path = Path(resolve_path(str(path)))
        except FileNotFoundError:
            self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except Exception:
            raw = {}
        self.data: Dict[str, dict[str, float | int]] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                self.data[str(k)] = {
                    "roi": float(v.get("roi", 0.0)),
                    "ema": float(v.get("ema", 0.0)),
                    "count": int(v.get("count", 0)),
                }
            else:  # backward compatibility with float-only entries
                self.data[str(k)] = {"roi": float(v), "ema": 0.0, "count": 0}

    def _save(self) -> None:
        try:
            with self.path.open("w", encoding="utf-8") as fh:
                json.dump(self.data, fh)
        except Exception:
            pass  # pragma: no cover - best effort

    # ------------------------------------------------------------------
    def is_stable(
        self,
        workflow_id: str,
        current_roi: float | None = None,
        threshold: float | None = None,
    ) -> bool:
        wf = str(workflow_id)
        entry = self.data.get(wf)
        if not entry or "roi" not in entry:
            return False
        if current_roi is not None and threshold is not None:
            prev = float(entry.get("roi", 0.0))
            if abs(current_roi - prev) > float(threshold):
                # metrics changed; remove from stable set
                del self.data[wf]
                self._save()
                return False
        return True

    # ------------------------------------------------------------------
    def mark_stable(self, workflow_id: str, roi: float) -> None:
        wf = str(workflow_id)
        entry = self.data.get(wf, {})
        entry.update({"roi": float(roi)})
        self.data[wf] = entry
        self._save()

    def record_metrics(
        self,
        workflow_id: str,
        roi: float,
        failures: float,
        entropy: float,
        *,
        roi_delta: float | None = None,
        roi_var: float = 0.0,
        failures_var: float = 0.0,
        entropy_var: float = 0.0,
    ) -> None:
        """Persist metrics for ``workflow_id``.

        Parameters
        ----------
        workflow_id:
            Identifier of the evaluated workflow chain.
        roi:
            Observed ROI for the latest run.
        failures:
            Number of failed modules during execution.
        entropy:
            Synergy entropy of the workflow chain.
        roi_delta:
            Optional ROI delta.  When omitted it is derived from the previous
            stored ROI.
        roi_var:
            Population variance of observed ROI across runs.
        failures_var:
            Population variance of failure counts across runs.
        entropy_var:
            Population variance of entropy across runs.
        """

        wf = str(workflow_id)
        entry = self.data.get(wf, {})
        if roi_delta is None:
            prev = float(entry.get("roi", 0.0))
            roi_delta = float(roi) - prev
        entry.update(
            {
                "roi": float(roi),
                "roi_delta": float(roi_delta),
                "roi_var": float(roi_var),
                "failures": float(failures),
                "failures_var": float(failures_var),
                "entropy": float(entropy),
                "entropy_var": float(entropy_var),
            }
        )
        self.data[wf] = entry
        self._save()

    def clear(self, workflow_id: str) -> None:
        if str(workflow_id) in self.data:
            del self.data[str(workflow_id)]
            self._save()

    def clear_all(self) -> None:
        self.data.clear()
        self._save()

    # ------------------------------------------------------------------
    def get_ema(self, workflow_id: str) -> Tuple[float, int]:
        entry = self.data.get(str(workflow_id), {})
        return float(entry.get("ema", 0.0)), int(entry.get("count", 0))

    def set_ema(self, workflow_id: str, ema: float, count: int) -> None:
        wf = str(workflow_id)
        entry = self.data.get(wf, {})
        entry.update({"ema": float(ema), "count": int(count)})
        self.data[wf] = entry
        self._save()
