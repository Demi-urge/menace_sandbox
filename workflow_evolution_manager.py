from __future__ import annotations

"""Manage workflow evolution by benchmarking generated variants."""

from typing import Callable, Optional
import importlib
import logging

from .composite_workflow_scorer import CompositeWorkflowScorer
from .workflow_evolution_bot import WorkflowEvolutionBot
from .roi_results_db import ROIResultsDB
from . import mutation_logger as MutationLogger

logger = logging.getLogger(__name__)


def _build_callable(sequence: str) -> Callable[[], bool]:
    """Best-effort construction of a workflow callable from *sequence*.

    Each element in ``sequence`` is treated as an importable module name.  The
    manager attempts to execute a ``main`` or ``run`` attribute from each module
    sequentially.  Missing modules or callables fall back to a no-op returning
    ``True`` so evaluation can continue in minimal environments.
    """

    steps = [s for s in sequence.split("-") if s]
    funcs: list[Callable[[], bool]] = []
    for step in steps:
        try:
            mod = importlib.import_module(step)
            func = getattr(mod, "main", None)
            if not callable(func):
                func = getattr(mod, "run", None)
            if callable(func):
                funcs.append(func)  # type: ignore[arg-type]
                continue
        except Exception as exc:  # pragma: no cover - import failures
            logger.debug("failed importing %s: %s", step, exc)
        funcs.append(lambda: True)

    def _workflow() -> bool:
        ok = True
        for fn in funcs:
            try:
                ok = bool(fn()) and ok
            except Exception:  # pragma: no cover - execution errors
                ok = False
        return ok

    return _workflow


def evolve(
    workflow_callable: Callable[[], bool],
    workflow_id: int | str,
    variants: int = 5,
) -> Callable[[], bool]:
    """Evolve ``workflow_callable`` by benchmarking generated variants.

    The baseline workflow is scored using :class:`CompositeWorkflowScorer` to
    establish ROI and per-module attribution.  Variants suggested by
    :class:`WorkflowEvolutionBot.generate_variants` are converted into
    callables, evaluated and compared against the baseline.  Results are stored
    in :class:`~roi_results_db.ROIResultsDB` and :mod:`mutation_logger`.

    Parameters
    ----------
    workflow_callable:
        Baseline workflow implementation.
    workflow_id:
        Identifier for ROI tracking.
    variants:
        Number of variants to evaluate.

    Returns
    -------
    Callable[[], bool]
        The promoted workflow implementation.  If no variant improves on the
        baseline, the original ``workflow_callable`` is returned.
    """

    wf_id_str = str(workflow_id)
    results_db = ROIResultsDB()

    # Baseline evaluation
    baseline_scorer = CompositeWorkflowScorer(results_db=results_db)
    baseline_result = baseline_scorer.run(workflow_callable, wf_id_str, run_id="baseline")
    baseline_roi = baseline_result.roi_gain

    bot = WorkflowEvolutionBot()

    best_callable = workflow_callable
    best_roi = baseline_roi
    best_variant_seq: Optional[str] = None

    for seq in bot.generate_variants(limit=variants, workflow_id=int(workflow_id)):
        variant_callable = _build_callable(seq)
        run_id = f"variant-{hash(seq) & 0xffffffff:x}"

        scorer = CompositeWorkflowScorer(results_db=results_db)
        variant_result = scorer.run(variant_callable, wf_id_str, run_id=run_id)
        roi_delta = variant_result.roi_gain - baseline_roi

        # Persist ROI delta with variant identifier
        results_db.log_module_delta(
            wf_id_str,
            run_id,
            module=f"variant:{seq}",
            runtime=variant_result.runtime,
            success_rate=variant_result.success_rate,
            roi_delta=roi_delta,
        )

        event_id = bot._rearranged_events.get(seq)
        if event_id is not None:
            MutationLogger.record_mutation_outcome(
                event_id,
                after_metric=variant_result.roi_gain,
                roi=roi_delta,
                performance=roi_delta,
            )

        if roi_delta > (best_roi - baseline_roi):
            best_roi = variant_result.roi_gain
            best_callable = variant_callable
            best_variant_seq = seq

    if best_variant_seq is not None and best_roi > baseline_roi:
        MutationLogger.log_mutation(
            change=best_variant_seq,
            reason="promoted",
            trigger="workflow_evolution_manager",
            performance=best_roi - baseline_roi,
            workflow_id=int(workflow_id),
            before_metric=baseline_roi,
            after_metric=best_roi,
        )
        return best_callable

    MutationLogger.log_mutation(
        change="stable",
        reason="stable",
        trigger="workflow_evolution_manager",
        performance=0.0,
        workflow_id=int(workflow_id),
        before_metric=baseline_roi,
        after_metric=baseline_roi,
    )
    return workflow_callable


__all__ = ["evolve"]
