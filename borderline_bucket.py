"""Simple JSONL-backed borderline workflow bucket."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


class BorderlineBucket:
    """Persist borderline workflow information to a JSONL file.

    Each record stores ``raroi`` history, confidence and current status. The
    bucket is deliberately lightweight to keep tests fast and avoids external
    dependencies.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path or "borderline_bucket.jsonl"
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        obj = json.loads(line)
                        wf = obj.pop("workflow_id")
                        self.data[wf] = obj
        except FileNotFoundError:
            pass

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            for wf, info in self.data.items():
                rec = {"workflow_id": wf, **info}
                fh.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------
    def add_candidate(self, workflow_id: str, raroi: float, confidence: float) -> None:
        info = self.data.setdefault(
            workflow_id,
            {"raroi": [], "confidence": float(confidence), "status": "candidate"},
        )
        info.setdefault("raroi", []).append(float(raroi))
        info["confidence"] = float(confidence)
        info.setdefault("status", "candidate")
        self._save()

    def get_candidate(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(workflow_id)

    def all_candidates(self, status: str | None = None) -> Dict[str, Dict[str, Any]]:
        if status is None:
            return dict(self.data)
        return {wf: info for wf, info in self.data.items() if info.get("status") == status}

    def record_result(self, workflow_id: str, raroi: float) -> None:
        info = self.data.setdefault(
            workflow_id,
            {"raroi": [], "confidence": 0.0, "status": "candidate"},
        )
        info.setdefault("raroi", []).append(float(raroi))
        self._save()

    def promote(self, workflow_id: str) -> None:
        if workflow_id in self.data:
            self.data[workflow_id]["status"] = "promoted"
            self._save()

    def terminate(self, workflow_id: str) -> None:
        if workflow_id in self.data:
            self.data[workflow_id]["status"] = "terminated"
            self._save()


__all__ = ["BorderlineBucket"]

