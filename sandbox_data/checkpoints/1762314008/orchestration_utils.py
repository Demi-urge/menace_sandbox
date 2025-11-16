"""Utility helpers for coordinating self-improvement workflows."""

from __future__ import annotations

import logging
from typing import Callable, Mapping

from menace_sandbox.composite_workflow_scorer import CompositeWorkflowScorer
from menace_sandbox.roi_results_db import ROIResultsDB
from menace_sandbox.workflow_scorer_core import EvaluationResult
from menace_sandbox import mutation_logger as MutationLogger


def benchmark_workflow_variants(
    workflow_id: int | str, variants: Mapping[str, Callable[[], bool]]
) -> dict[str, tuple[EvaluationResult, float]]:
    """Benchmark *variants* against the baseline workflow.

    Parameters
    ----------
    workflow_id:
        Identifier used for ROI tracking.
    variants:
        Mapping of variant identifiers to callables.  The mapping **must**
        contain a ``"baseline"`` entry representing the current workflow.

    Returns
    -------
    dict
        Mapping of variant names to tuples of ``(EvaluationResult, roi_delta)``.
    """

    wf_id_str = str(workflow_id)
    if "baseline" not in variants:
        raise ValueError("variants must include a 'baseline' entry")

    results_db = ROIResultsDB()

    baseline_callable = variants["baseline"]
    baseline_scorer = CompositeWorkflowScorer(results_db=results_db)
    baseline_result = baseline_scorer.run(
        baseline_callable, wf_id_str, run_id="baseline"
    )
    baseline_roi = baseline_result.roi_gain

    results: dict[str, tuple[EvaluationResult, float]] = {
        "baseline": (baseline_result, 0.0)
    }

    try:
        wf_id_int = int(workflow_id)
    except Exception:
        wf_id_int = 0

    for name, func in variants.items():
        if name == "baseline":
            continue
        run_id = f"variant-{hash(name) & 0xffffffff:x}"
        scorer = CompositeWorkflowScorer(results_db=results_db)
        variant_result = scorer.run(func, wf_id_str, run_id=run_id)
        roi_delta = variant_result.roi_gain - baseline_roi
        results[name] = (variant_result, roi_delta)

        results_db.log_module_delta(
            wf_id_str,
            run_id,
            module=f"variant:{name}",
            runtime=variant_result.runtime,
            success_rate=variant_result.success_rate,
            roi_delta=roi_delta,
        )

        try:
            event_id = MutationLogger.log_mutation(
                change=name,
                reason="variant_evaluation",
                trigger="benchmark_workflow_variants",
                performance=0.0,
                workflow_id=wf_id_int,
                before_metric=baseline_roi,
                after_metric=baseline_roi,
            )
            MutationLogger.record_mutation_outcome(
                event_id,
                after_metric=variant_result.roi_gain,
                roi=roi_delta,
                performance=roi_delta,
            )
        except Exception:  # pragma: no cover - logging shouldn't break flow
            logging.getLogger(__name__).exception(
                "mutation logging failed for variant %s", name
            )

    return results


__all__ = ["benchmark_workflow_variants"]

