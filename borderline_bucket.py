import json
import os
from typing import Dict, List, Any


class BorderlineBucket:
    """Manage borderline workflow candidates with persistent JSONL storage."""

    def __init__(self, path: str = "borderline_bucket.jsonl") -> None:
        self.path = path
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self._load()

    # internal helper to load state from JSONL
    def _load(self) -> None:
        if not os.path.exists(self.path):
            open(self.path, "a").close()
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                event = json.loads(line)
                wid = event["workflow_id"]
                action = event["action"]
                if action == "add":
                    self.candidates[wid] = {
                        "raroi": [event["raroi"]],
                        "confidence": event["confidence"],
                        "status": "candidate",
                    }
                elif action == "result" and wid in self.candidates:
                    self.candidates[wid]["raroi"].append(event["raroi"])
                elif action == "promote" and wid in self.candidates:
                    self.candidates[wid]["status"] = "promoted"
                elif action == "terminate" and wid in self.candidates:
                    self.candidates[wid]["status"] = "terminated"

    def _append_event(self, data: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            json.dump(data, f)
            f.write("\n")

    def add_candidate(self, workflow_id: str, raroi: float, confidence: float) -> None:
        """Add a new borderline workflow candidate."""
        self.candidates[workflow_id] = {
            "raroi": [raroi],
            "confidence": confidence,
            "status": "candidate",
        }
        self._append_event(
            {
                "action": "add",
                "workflow_id": workflow_id,
                "raroi": raroi,
                "confidence": confidence,
            }
        )

    def record_result(self, workflow_id: str, raroi: float) -> None:
        """Record a test result for a candidate."""
        if workflow_id not in self.candidates:
            raise KeyError(f"Unknown workflow_id {workflow_id}")
        self.candidates[workflow_id]["raroi"].append(raroi)
        self._append_event(
            {"action": "result", "workflow_id": workflow_id, "raroi": raroi}
        )

    def promote(self, workflow_id: str) -> None:
        """Promote a candidate after successful testing."""
        if workflow_id not in self.candidates:
            raise KeyError(f"Unknown workflow_id {workflow_id}")
        self.candidates[workflow_id]["status"] = "promoted"
        self._append_event({"action": "promote", "workflow_id": workflow_id})

    def terminate(self, workflow_id: str) -> None:
        """Terminate a candidate after failed testing."""
        if workflow_id not in self.candidates:
            raise KeyError(f"Unknown workflow_id {workflow_id}")
        self.candidates[workflow_id]["status"] = "terminated"
        self._append_event({"action": "terminate", "workflow_id": workflow_id})
