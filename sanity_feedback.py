from __future__ import annotations

"""Feedback loop based on Sanity Layer detections."""

import json
import threading
import time
from typing import Any, Optional
import logging

from .dynamic_path_router import resolve_path
from .log_tags import FEEDBACK
from .coding_bot_interface import self_coding_managed
from .bot_registry import BotRegistry
from .data_bot import DataBot

try:  # pragma: no cover - optional dependency
    from failure_learning_system import DiscrepancyDB as DetectionDB
except Exception:  # pragma: no cover - best effort
    DetectionDB = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from discrepancy_db import DiscrepancyDB as OutcomeDB, DiscrepancyRecord
except Exception:  # pragma: no cover - best effort
    OutcomeDB = None  # type: ignore
    DiscrepancyRecord = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from shared_gpt_memory import GPT_MEMORY_MANAGER
except Exception:  # pragma: no cover - best effort
    GPT_MEMORY_MANAGER = None  # type: ignore

logger = logging.getLogger(__name__)

registry = BotRegistry()
data_bot = DataBot(start_server=False)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class SanityFeedback:
    """Poll Sanity Layer detections and trigger self-coding patches."""

    def __init__(
        self,
        manager,
        *,
        threshold: float = 1.0,
        interval: int = 60,
        detection_db: DetectionDB | None = None,
        outcome_db: OutcomeDB | None = None,
    ) -> None:
        self.manager = manager
        self.engine = getattr(manager, "engine", None)
        self.threshold = threshold
        self.interval = interval
        self.detection_db = detection_db or (DetectionDB() if DetectionDB else None)
        self.outcome_db = outcome_db or (OutcomeDB() if OutcomeDB else None)
        self.last_ts: str | None = None
        self.running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=0)

    # ------------------------------------------------------------------
    def check(self) -> None:
        """Run a single feedback cycle."""
        self._run_cycle()

    def _loop(self) -> None:
        while self.running:
            self._run_cycle()
            time.sleep(self.interval)

    def _run_cycle(self) -> None:
        if not self.detection_db:
            return
        try:
            df = self.detection_db.fetch_detections(min_severity=self.threshold)
        except Exception:
            logger.exception("failed to fetch detections")
            return
        if df.empty:
            return
        if self.last_ts:
            df = df[df["ts"] > self.last_ts]
            if df.empty:
                return
        max_ts: str | None = None
        for _idx, row in df.iterrows():
            max_ts = row["ts"] if max_ts is None else max(max_ts, row["ts"])
            self._handle_row(row)
        if max_ts:
            self.last_ts = max_ts

    def _handle_row(self, row) -> None:  # pragma: no cover - integration
        rule = row.get("rule", "")
        message = row.get("message", "")
        severity = float(row.get("severity", 0.0) or 0.0)
        workflow = row.get("workflow", "")
        try:
            meta = json.loads(message)
        except Exception:
            meta = {}
        path = meta.get("module") or meta.get("path")
        desc = f"address {rule} detection"
        patch_id = None
        success = False
        summary: dict[str, Any] | None = None
        if path:
            try:
                path_obj = resolve_path(path if path.endswith(".py") else f"{path}.py")
                summary = self.manager.auto_run_patch(
                    path_obj,
                    desc,
                    context_meta={"reason": desc, "trigger": "sanity_feedback"},
                )
                patch_id = getattr(self.manager, "_last_patch_id", None)
                failed_tests = int(summary.get("self_tests", {}).get("failed", 0)) if summary else 0
                success = bool(patch_id) and summary is not None and failed_tests == 0
                if summary is None and patch_id is not None:
                    engine = getattr(self.manager, "engine", None)
                    if hasattr(engine, "rollback_patch"):
                        try:
                            engine.rollback_patch(str(patch_id))
                        except Exception:
                            logger.exception("sanity rollback failed")
                    patch_id = None
                elif failed_tests and patch_id is not None:
                    engine = getattr(self.manager, "engine", None)
                    if hasattr(engine, "rollback_patch"):
                        try:
                            engine.rollback_patch(str(patch_id))
                        except Exception:
                            logger.exception("sanity rollback failed")
                    patch_id = None
            except Exception:
                logger.exception("patch application failed", extra={"path": path})
        # log memory interaction
        if GPT_MEMORY_MANAGER is not None:
            try:
                GPT_MEMORY_MANAGER.log_interaction(
                    desc,
                    json.dumps(
                        {
                            "meta": meta,
                            "patch_id": patch_id,
                            "success": success,
                            "rule": rule,
                            "severity": severity,
                            "workflow": workflow,
                        },
                        sort_keys=True,
                    ),
                    tags=[FEEDBACK, "sanity_feedback"],
                )
            except Exception:
                logger.exception("memory logging failed")
        # record outcome
        if self.outcome_db is not None and DiscrepancyRecord is not None:
            metadata = {
                "rule": rule,
                "severity": severity,
                "workflow": workflow,
                "patch_id": patch_id,
                "confidence": severity,
                "outcome_score": 1.0 if success else 0.0,
            }
            try:
                rec = DiscrepancyRecord(message=message, metadata=metadata)
                self.outcome_db.add(rec)
            except Exception:
                logger.exception("discrepancy logging failed")


__all__ = ["SanityFeedback"]
