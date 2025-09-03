"""JSONL-backed storage for borderline workflow candidates.

The :class:`BorderlineBucket` keeps track of workflows whose risk-adjusted
return on investment (RAROI) or confidence scores fall near the configured
thresholds.  Each update appends a JSON record to
``sandbox_data/borderline_bucket.jsonl`` so external tooling can inspect the
state without needing the process that created it.

Records contain the ``workflow_id``, a history of ``raroi`` measurements, the
latest ``confidence`` score, current ``status`` and creation/update timestamps.
The file is append-only and protected with a simple lock file to reduce the
chance of corruption under concurrent writes.
"""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from filelock import FileLock
from dynamic_path_router import resolve_path

logger = logging.getLogger(__name__)


class BorderlineBucket:
    """Persist borderline workflow information to a JSONL file."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = (
            Path(path)
            if path
            else Path(resolve_path("sandbox_data")) / "borderline_bucket.jsonl"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path) + ".lock")
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        self.data: Dict[str, Dict[str, Any]] = {}
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:  # pragma: no cover - best effort
                    continue
                wf = str(record.get("workflow_id"))
                self.data[wf] = record

    def _append(self, record: Dict[str, Any]) -> None:
        """Atomically append *record* to the JSONL file."""
        with self.lock:
            with self.path.open("a", encoding="utf-8") as fh:
                json.dump(record, fh)
                fh.write("\n")

    # ------------------------------------------------------------------
    def add_candidate(
        self,
        workflow_id: str,
        raroi: float,
        confidence: float,
        context: Any | None = None,
    ) -> None:
        """Persist *workflow_id* as a pending candidate."""

        now = datetime.utcnow().isoformat()
        record = {
            "workflow_id": workflow_id,
            "raroi": [float(raroi)],
            "confidence": float(confidence),
            "status": "pending",
            "created_at": now,
            "updated_at": now,
        }
        if context is not None:
            record["context"] = context
        self.data[workflow_id] = record
        self._append(record)

    # Backwards-compatible alias --------------------------------------
    def enqueue(
        self,
        workflow_id: str,
        raroi: float,
        confidence: float,
        context: Any | None = None,
    ) -> None:
        self.add_candidate(workflow_id, raroi, confidence, context)

    # ------------------------------------------------------------------
    def record_result(
        self, workflow_id: str, raroi: float, confidence: float | None = None
    ) -> None:
        """Record a new ``raroi`` value for ``workflow_id``."""

        now = datetime.utcnow().isoformat()
        info = self.data.setdefault(
            workflow_id,
            {
                "workflow_id": workflow_id,
                "raroi": [],
                "confidence": float(confidence) if confidence is not None else 0.0,
                "status": "pending",
                "created_at": now,
            },
        )
        info.setdefault("raroi", []).append(float(raroi))
        if confidence is not None:
            info["confidence"] = float(confidence)
        info["updated_at"] = now
        self.data[workflow_id] = info
        self._append(info)

    def promote(self, workflow_id: str) -> None:
        info = self.data.get(workflow_id)
        if not info:
            return
        info["status"] = "promoted"
        info["updated_at"] = datetime.utcnow().isoformat()
        self._append(info)

    def terminate(self, workflow_id: str) -> None:
        info = self.data.get(workflow_id)
        if not info:
            return
        info["status"] = "terminated"
        info["updated_at"] = datetime.utcnow().isoformat()
        self._append(info)

    # ------------------------------------------------------------------
    def get_candidate(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get(workflow_id)

    def get_candidates(self, status: str | None = "pending") -> Dict[str, Dict[str, Any]]:
        if status is None:
            return dict(self.data)
        return {
            wf: info
            for wf, info in self.data.items()
            if info.get("status") == status
        }

    # Compatibility helpers -------------------------------------------
    def all_candidates(self, status: str | None = None) -> Dict[str, Dict[str, Any]]:
        return self.get_candidates(status)

    def pending(self) -> Dict[str, Dict[str, Any]]:
        return self.get_candidates("pending")

    def status(self, workflow_id: str) -> Optional[str]:
        info = self.data.get(workflow_id)
        return info.get("status") if info else None

    # ------------------------------------------------------------------
    def process(
        self,
        evaluator: Callable[[str, Dict[str, Any]], Tuple[float, float] | float] | None = None,
        raroi_threshold: float = 0.0,
        confidence_threshold: float = 0.0,
    ) -> None:
        """Run micro-pilot evaluations for queued candidates."""

        evaluate = evaluator or (
            lambda wf, info: (info["raroi"][-1], info.get("confidence", 0.0))
        )
        for wf, info in list(self.get_candidates("pending").items()):
            try:
                result = evaluate(wf, info)
                if isinstance(result, (tuple, list)):
                    raroi, conf = float(result[0]), float(result[1])
                else:
                    raroi, conf = float(result), info.get("confidence", 0.0)
                self.record_result(wf, raroi, conf)
                if raroi > raroi_threshold and conf >= confidence_threshold:
                    self.promote(wf)
                else:
                    self.terminate(wf)
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed processing borderline candidate %s", wf)


__all__ = ["BorderlineBucket"]

