from __future__ import annotations

import logging
import time
import uuid
from statistics import fmean
from typing import Any, Callable, Mapping

from composite_workflow_scorer import CompositeWorkflowScorer
from logging_utils import log_record


class SandboxOrchestrator:
    """Coordinate end-to-end workflow execution with ROI monitoring."""

    def __init__(
        self,
        workflows: Mapping[str, Callable[[], Any]],
        *,
        logger: logging.Logger | None = None,
        loop_interval: float = 15.0,
        diminishing_threshold: float = 0.0,
        patience: int = 3,
    ) -> None:
        self.workflows: dict[str, Callable[[], Any]] = dict(workflows)
        self.logger = logger or logging.getLogger(__name__)
        self.scorer = CompositeWorkflowScorer(failure_logger=self.logger)
        self.loop_interval = loop_interval
        self.diminishing_threshold = diminishing_threshold
        self.patience = max(1, int(patience))
        self._stagnation_streak = 0
        self._running = False

    def run_once(self) -> None:
        """Execute the configured workflow callables once."""

        batch_id = uuid.uuid4().hex
        for workflow_id, workflow_callable in self.workflows.items():
            try:
                self.scorer.run(
                    workflow_callable,
                    workflow_id=workflow_id,
                    run_id=f"{batch_id}-{workflow_id}",
                )
            except Exception:
                self.logger.exception(
                    "workflow execution failed",
                    extra=log_record(workflow_id=workflow_id, batch_id=batch_id),
                )

    def _aggregate_roi_delta(self) -> float:
        deltas: list[float] = []
        for workflow_id in self.workflows:
            stats = self.scorer.results_db.fetch_chain_stats(workflow_id)
            deltas.append(float(stats.get("delta_roi", 0.0)))
        return fmean(deltas) if deltas else 0.0

    def _should_continue(self) -> bool:
        aggregate_delta = self._aggregate_roi_delta()
        if aggregate_delta <= self.diminishing_threshold:
            self._stagnation_streak += 1
        else:
            self._stagnation_streak = 0

        self.logger.info(
            "evaluated aggregate ROI delta",
            extra=log_record(
                aggregate_delta=aggregate_delta,
                stagnation_streak=self._stagnation_streak,
                diminishing_threshold=self.diminishing_threshold,
            ),
        )
        return self._stagnation_streak < self.patience

    def transition_to_maintenance_mode(self) -> None:
        """Log and transition the orchestrator into maintenance mode."""

        self.logger.info(
            "diminishing returns detected; entering maintenance mode",
            extra=log_record(event="maintenance-mode"),
        )
        self._running = False

    def run(self) -> None:
        """Run chained workflows until ROI gains diminish."""

        self._running = True
        while self._running:
            self.run_once()
            if not self._should_continue():
                self.transition_to_maintenance_mode()
                break
            time.sleep(self.loop_interval)

    def stop(self) -> None:
        """Request an orderly stop."""

        self._running = False
