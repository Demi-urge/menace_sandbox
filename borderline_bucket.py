"""Simple JSONL-backed borderline workflow bucket."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


class BorderlineBucket:
    """Persist borderline workflow information to a JSONL file.

    Each record stores ``raroi`` history, confidence, context and current
    status.  The bucket is deliberately lightweight to keep tests fast and
    avoids external dependencies.
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
    def enqueue(
        self,
        workflow_id: str,
        raroi: float,
        confidence: float,
        context: Any | None = None,
    ) -> None:
        """Persist *workflow_id* as a borderline candidate.

        Parameters
        ----------
        workflow_id:
            Unique identifier for the workflow.
        raroi:
            Latest risk-adjusted ROI measurement.
        confidence:
            Confidence score associated with the workflow.
        context:
            Optional JSON-serialisable payload providing additional information
            about the workflow.
        """

        info = self.data.setdefault(
            workflow_id,
            {
                "raroi": [],
                "confidence": float(confidence),
                "status": "candidate",
                "context": context,
            },
        )
        info.setdefault("raroi", []).append(float(raroi))
        info["confidence"] = float(confidence)
        if context is not None:
            info["context"] = context
        info.setdefault("status", "candidate")
        self._save()

    # Backwards compatible alias ------------------------------------------------
    def add_candidate(
        self,
        workflow_id: str,
        raroi: float,
        confidence: float,
        context: Any | None = None,
    ) -> None:
        self.enqueue(workflow_id, raroi, confidence, context)

    def get_candidate(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(workflow_id)

    def all_candidates(self, status: str | None = None) -> Dict[str, Dict[str, Any]]:
        if status is None:
            return dict(self.data)
        return {
            wf: info
            for wf, info in self.data.items()
            if info.get("status") == status
        }

    def pending(self) -> Dict[str, Dict[str, Any]]:
        """Return all candidates still awaiting a decision."""

        return self.all_candidates(status="candidate")

    def status(self, workflow_id: str) -> Optional[str]:
        """Return the current status for *workflow_id* or ``None``."""

        info = self.data.get(workflow_id)
        return info.get("status") if info else None

    def record_result(
        self, workflow_id: str, raroi: float, confidence: float | None = None
    ) -> None:
        info = self.data.setdefault(
            workflow_id,
            {
                "raroi": [],
                "confidence": float(confidence) if confidence is not None else 0.0,
                "status": "candidate",
            },
        )
        info.setdefault("raroi", []).append(float(raroi))
        if confidence is not None:
            info["confidence"] = float(confidence)
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

