from __future__ import annotations

"""Persist workflow stability status across runs."""

from pathlib import Path
from typing import Dict
import json


class WorkflowStabilityDB:
    """Store workflow IDs deemed stable with their last observed ROI.

    The data is backed by a simple JSON file so state survives process restarts.
    When ``is_stable`` is invoked with a ``current_roi`` and the change exceeds
    ``threshold`` the workflow is automatically cleared from the stable set.
    """

    def __init__(self, path: str | Path = "sandbox_data/stable_workflows.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            data = {}
        self.data: Dict[str, float] = {str(k): float(v) for k, v in data.items()}

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
        if wf not in self.data:
            return False
        if current_roi is not None and threshold is not None:
            prev = float(self.data.get(wf, 0.0))
            if abs(current_roi - prev) > float(threshold):
                # metrics changed; remove from stable set
                del self.data[wf]
                self._save()
                return False
        return True

    # ------------------------------------------------------------------
    def mark_stable(self, workflow_id: str, roi: float) -> None:
        self.data[str(workflow_id)] = float(roi)
        self._save()

    def clear(self, workflow_id: str) -> None:
        if str(workflow_id) in self.data:
            del self.data[str(workflow_id)]
            self._save()

    def clear_all(self) -> None:
        self.data.clear()
        self._save()
