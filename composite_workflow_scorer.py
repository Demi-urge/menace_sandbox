"""Lightweight workflow scoring utilities for self-improvement modules.

This module provides a trimmed down :class:`CompositeWorkflowScorer` that
executes a workflow callable, records ROI related metrics via
``ROITracker``/``ROICalculator`` and persists aggregated results in
``ROIResultsDB``.  The implementation mirrors the behaviour of the heavier
``roi_scorer`` module but keeps the dependency surface small so that unit
tests can exercise the scoring pipeline in isolation.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Mapping, Tuple
import time
import uuid
import concurrent.futures
import logging
import traceback
from pathlib import Path

import json
import yaml

from .roi_tracker import ROITracker
from .roi_calculator import ROICalculator
from .roi_results_db import ROIResultsDB
from . import sandbox_runner
from .dynamic_path_router import resolve_path
from .workflow_scorer_core import (
    ROIScorer as BaseROIScorer,
    EvaluationResult,
    compute_workflow_synergy,
    compute_bottleneck_index,
    compute_patchability,
)
from vector_service.context_builder import ContextBuilder

try:  # pragma: no cover - optional dependency
    from .code_database import PatchHistoryDB
except Exception:  # pragma: no cover - fallback when DB unavailable
    PatchHistoryDB = None  # type: ignore

try:  # pragma: no cover - runtime failures should not break scoring
    PATCH_SUCCESS_RATE = (
        PatchHistoryDB().success_rate() if PatchHistoryDB is not None else 1.0
    )
except Exception:  # pragma: no cover - defensive fallback
    PATCH_SUCCESS_RATE = 1.0

WINNING_SEQUENCES_PATH = Path(resolve_path("sandbox_data")) / "winning_sequences.json"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


class ROIScorer(BaseROIScorer):
    """Thin wrapper around :class:`ROITracker` and ``ROICalculator``."""

    def __init__(
        self,
        tracker: ROITracker | None = None,
        calculator_factory: Callable[[], ROICalculator] | None = None,
        profile_type: str | None = None,
    ) -> None:
        try:
            super().__init__(
                tracker=tracker,
                calculator_factory=calculator_factory,
                profile_type=profile_type,
            )
        except Exception as exc:  # pragma: no cover - configuration errors
            logging.critical("Failed to initialise ROICalculator: %s", exc)
            try:
                with resolve_path("configs/roi_profiles.yaml").open(
                    "r", encoding="utf-8"
                ) as fh:
                    profiles: Dict[str, Dict[str, Any]] = yaml.safe_load(fh) or {}
                default_type, default_profile = next(iter(profiles.items()))
            except Exception as profile_exc:  # pragma: no cover - missing config
                raise RuntimeError(
                    "ROICalculator unavailable and no default ROI profile could be loaded"
                ) from profile_exc

            calc = ROICalculator.__new__(ROICalculator)
            calc.profiles = {default_type: default_profile}
            calc.logger = logging.getLogger(__name__)
            try:
                calc._validate_profiles()
            except Exception as validation_exc:  # pragma: no cover - invalid profile
                raise RuntimeError(
                    "Default ROI profile is invalid; check configs/roi_profiles.yaml"
                ) from validation_exc

            self.tracker = tracker or ROITracker()
            self.calculator = calc
            self.profile_type = profile_type or default_type


class CompositeWorkflowScorer(ROIScorer):
    """Execute workflows and compute aggregate ROI metrics."""

    def __init__(
        self,
        tracker: ROITracker | None = None,
        calculator_factory: Callable[[], ROICalculator] | None = None,
        results_db: ROIResultsDB | None = None,
        profile_type: str | None = None,
        failure_logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(tracker, calculator_factory, profile_type)
        self.results_db = results_db or ROIResultsDB()
        self.failure_logger = failure_logger or logging.getLogger(
            f"{__name__}.failures"
        )
        self.failure_details: list[Dict[str, Any]] = []
        # offsets used to compute deltas for individual runs
        self._roi_start: int = 0
        self._module_start: Dict[str, int] = {}
        self._module_successes: Dict[str, bool] = {}
        self._module_failures: Dict[str, str] = {}
        self.history_window = 20
        self.module_attribution: Dict[str, Dict[str, float]] = {}

    def record_winning_sequence(self, sequence: list[str]) -> None:
        """Persist a successful workflow sequence for later reinforcement."""
        try:
            data: list[list[str]] = []
            if WINNING_SEQUENCES_PATH.exists():
                data = json.loads(WINNING_SEQUENCES_PATH.read_text())
            data.append(sequence)
            WINNING_SEQUENCES_PATH.parent.mkdir(parents=True, exist_ok=True)
            WINNING_SEQUENCES_PATH.write_text(json.dumps(data, indent=2))
        except Exception:  # pragma: no cover - best effort logging
            self.failure_logger.exception("failed to record winning sequence")

    # ------------------------------------------------------------------
    def run(
        self,
        workflow_callable: Callable[[], bool],
        workflow_id: str,
        run_id: str,
        *,
        patch_success: float | None = None,
        sequence: list[str] | None = None,
    ) -> EvaluationResult:
        """Execute ``workflow_callable`` within a sandbox and persist metrics."""

        start = time.perf_counter()
        runner = sandbox_runner.WorkflowSandboxRunner()
        try:
            result = runner.run(workflow_callable, safe_mode=True)
            success = False if isinstance(result, Exception) else bool(result)
        except Exception as exc:  # pragma: no cover - runner failures
            tb = traceback.format_exc()
            self.failure_logger.error(
                "Workflow %s run %s raised an exception:\n%s",
                workflow_id,
                run_id,
                tb,
            )
            self.failure_details.append(
                {
                    "scope": "workflow",
                    "workflow_id": workflow_id,
                    "run_id": run_id,
                    "exception": exc,
                    "traceback": tb,
                }
            )
            success = False
        runtime = time.perf_counter() - start

        roi_gain = sum(float(x) for x in self.tracker.roi_history[self._roi_start:])

        per_module: Dict[str, Dict[str, float]] = {}
        timings = getattr(self.tracker, "timings", {})
        overheads = getattr(self.tracker, "scheduling_overhead", {})
        total_runtime = sum(float(t) for t in timings.values())
        for mod, deltas in self.tracker.module_deltas.items():
            start_idx = self._module_start.get(mod, 0)
            roi_delta = sum(float(x) for x in deltas[start_idx:])
            runtime = float(timings.get(mod, 0.0))
            success_rate = 1.0 if self._module_successes.get(mod) else 0.0
            bottleneck = runtime / total_runtime if total_runtime > 0 else 0.0
            per_module[mod] = {
                "runtime": runtime,
                "roi_delta": roi_delta,
                "success_rate": success_rate,
                "bottleneck_contribution": bottleneck,
                "scheduling_overhead": float(overheads.get(mod, 0.0)),
            }
            failure_reason = self._module_failures.get(mod)
            if failure_reason is not None:
                per_module[mod]["failure_reason"] = failure_reason
            self.results_db.log_module_attribution(mod, roi_delta, bottleneck)
        self.module_attribution = {
            mod: {
                "roi_delta": data["roi_delta"],
                "bottleneck_contribution": data["bottleneck_contribution"],
            }
            for mod, data in per_module.items()
        }

        workflow_synergy_score = compute_workflow_synergy(
            self.tracker, self.history_window
        )
        bottleneck_index = compute_bottleneck_index(timings)
        rate = (
            patch_success
            if patch_success is not None
            else getattr(
                self.tracker,
                "patch_success",
                getattr(self.tracker, "patch_success_rate", PATCH_SUCCESS_RATE),
            )
        )
        patchability_score = compute_patchability(
            self.tracker.roi_history,
            window=self.history_window,
            patch_success=rate,
        )

        failure_reason = (
            "; ".join(f"{m}: {r}" for m, r in self._module_failures.items()) or None
        )

        self.results_db.log_result(
            workflow_id=workflow_id,
            run_id=run_id,
            runtime=runtime,
            success_rate=1.0 if success else 0.0,
            roi_gain=roi_gain,
            workflow_synergy_score=workflow_synergy_score,
            bottleneck_index=bottleneck_index,
            patchability_score=patchability_score,
            module_deltas=per_module,
            failure_reason=failure_reason,
        )

        if success and sequence and roi_gain > 0:
            self.record_winning_sequence(sequence)

        return EvaluationResult(
            runtime=runtime,
            success_rate=1.0 if success else 0.0,
            roi_gain=roi_gain,
            workflow_synergy_score=workflow_synergy_score,
            bottleneck_index=bottleneck_index,
            patchability_score=patchability_score,
            per_module=per_module,
        )

    # ------------------------------------------------------------------
    def score_workflow(
        self,
        workflow_id: str,
        modules: Mapping[str, Callable[[], bool]],
        run_id: str | None = None,
        *,
        concurrency_hints: Mapping[str, Any] | None = None,
        patch_success: float | None = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Execute ``modules`` possibly in parallel and return metrics."""

        run_id = run_id or uuid.uuid4().hex
        self._roi_start = len(self.tracker.roi_history)
        self._module_start = {m: len(d) for m, d in self.tracker.module_deltas.items()}
        self._module_successes = {}
        self._module_failures = {}
        self.failure_details = []
        self.tracker.timings = {}
        self.tracker.scheduling_overhead = {}

        hints = dict(concurrency_hints or {})
        max_workers = int(hints.get("max_workers", 1)) or 1

        def _run_module(name: str, func: Callable[[], bool], scheduled: float) -> bool:
            start = time.perf_counter()
            self.tracker.scheduling_overhead[name] = start - scheduled
            try:
                ok = bool(func())
            except Exception as exc:
                tb = traceback.format_exc()
                self.failure_logger.error(
                    "Module %s execution failed in workflow %s run %s:\n%s",
                    name,
                    workflow_id,
                    run_id,
                    tb,
                )
                ok = False
                self._module_failures[name] = f"{type(exc).__name__}: {exc}"
                self.failure_details.append(
                    {
                        "scope": "module",
                        "module": name,
                        "workflow_id": workflow_id,
                        "run_id": run_id,
                        "exception": exc,
                        "traceback": tb,
                    }
                )
            duration = time.perf_counter() - start
            self.tracker.timings[name] = duration
            self._module_successes[name] = ok
            metrics = {
                "reliability": 1.0 if ok else 0.0,
                "efficiency": 1.0 / duration if duration > 0 else 0.0,
            }
            result = self.calculator.calculate(metrics, self.profile_type)
            self.tracker.update(
                0.0,
                result.score,
                modules=[name],
                metrics=metrics,
                profile_type=self.profile_type,
            )
            return ok

        def workflow_callable() -> bool:
            overall = True
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures: Dict[concurrent.futures.Future[bool], str] = {}
                for name, func in modules.items():
                    scheduled = time.perf_counter() if max_workers > 1 else 0.0
                    futures[pool.submit(_run_module, name, func, scheduled)] = name
                for fut in concurrent.futures.as_completed(futures):
                    ok = bool(fut.result())
                    overall = overall and ok
            return overall

        result = self.run(
            workflow_callable, workflow_id, run_id, patch_success=patch_success
        )
        return run_id, asdict(result)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        workflow_id: str,
        env_presets: (
            Mapping[str, list[Dict[str, Any]]] | list[Dict[str, Any]] | None
        ) = None,
        *,
        patch_success: float | None = None,
        context_builder: ContextBuilder,
    ) -> EvaluationResult:
        """Backwards compatible wrapper executing sandbox simulations.

        Parameters
        ----------
        workflow_id:
            Identifier of the workflow to evaluate.
        env_presets:
            Optional execution environment presets passed to the sandbox runner.
        patch_success:
            Override for the global patch success rate when computing
            ``patchability_score``.
        context_builder:
            Builder used to construct context for workflow runs. The builder's
            database weights are refreshed prior to invoking the sandbox
            simulations.
        """

        start = time.perf_counter()
        builder = context_builder
        try:
            builder.refresh_db_weights()
        except Exception:  # pragma: no cover - best effort refresh
            pass
        tracker, details = sandbox_runner.environment.run_workflow_simulations(
            workflows_db=workflow_id,
            env_presets=env_presets,
            return_details=True,
            tracker=self.tracker,
            context_builder=builder,
        )
        runtime = time.perf_counter() - start

        total = success = 0
        per_module_counts: Dict[str, Dict[str, int]] = {}
        for runs in details.values():
            for entry in runs:
                result = entry.get("result", {})
                module = entry.get("module")
                total += 1
                if result.get("exit_code", 1) == 0:
                    success += 1
                    if module:
                        per_module_counts.setdefault(module, {"total": 0, "success": 0})
                        per_module_counts[module]["success"] += 1
                if module:
                    per_module_counts.setdefault(module, {"total": 0, "success": 0})
                    per_module_counts[module]["total"] += 1

        success_rate = success / total if total else 0.0
        roi_gain = sum(float(r) for r in getattr(tracker, "roi_history", []))
        run_id = uuid.uuid4().hex
        per_module_metrics: Dict[str, Dict[str, float]] = {}
        timings = getattr(tracker, "timings", {})
        overheads = getattr(tracker, "scheduling_overhead", {})
        total_runtime = sum(float(t) for t in timings.values())
        for mod, counts in per_module_counts.items():
            roi_delta = sum(float(x) for x in tracker.module_deltas.get(mod, []))
            sr = counts["success"] / counts["total"] if counts["total"] else 0.0
            runtime = float(timings.get(mod, 0.0))
            bottleneck = runtime / total_runtime if total_runtime > 0 else 0.0
            per_module_metrics[mod] = {
                "success_rate": sr,
                "roi_delta": roi_delta,
                "runtime": runtime,
                "bottleneck_contribution": bottleneck,
                "scheduling_overhead": float(overheads.get(mod, 0.0)),
            }
            self.results_db.log_module_attribution(mod, roi_delta, bottleneck)
        self.module_attribution = {
            mod: {
                "roi_delta": data["roi_delta"],
                "bottleneck_contribution": data["bottleneck_contribution"],
            }
            for mod, data in per_module_metrics.items()
        }

        workflow_synergy_score = compute_workflow_synergy(tracker, self.history_window)
        bottleneck_index = compute_bottleneck_index(getattr(tracker, "timings", {}))
        rate = (
            patch_success
            if patch_success is not None
            else getattr(
                tracker,
                "patch_success",
                getattr(tracker, "patch_success_rate", PATCH_SUCCESS_RATE),
            )
        )
        patchability_score = compute_patchability(
            getattr(tracker, "roi_history", []),
            window=self.history_window,
            patch_success=rate,
        )
        self.results_db.log_result(
            workflow_id=workflow_id,
            run_id=run_id,
            runtime=runtime,
            success_rate=success_rate,
            roi_gain=roi_gain,
            workflow_synergy_score=workflow_synergy_score,
            bottleneck_index=bottleneck_index,
            patchability_score=patchability_score,
            module_deltas=per_module_metrics,
            failure_reason=None,
        )

        return EvaluationResult(
            runtime=runtime,
            success_rate=success_rate,
            roi_gain=roi_gain,
            workflow_synergy_score=workflow_synergy_score,
            bottleneck_index=bottleneck_index,
            patchability_score=patchability_score,
            per_module=per_module_metrics,
        )


__all__ = [
    "ROIScorer",
    "CompositeWorkflowScorer",
    "EvaluationResult",
    "compute_workflow_synergy",
    "compute_bottleneck_index",
    "compute_patchability",
]
