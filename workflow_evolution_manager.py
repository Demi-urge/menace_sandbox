from __future__ import annotations

"""Manage workflow evolution by benchmarking generated variants."""

from typing import Callable, Optional
import importlib
import logging
import os

from .composite_workflow_scorer import CompositeWorkflowScorer
from .workflow_evolution_bot import WorkflowEvolutionBot
from .roi_results_db import ROIResultsDB
from . import mutation_logger as MutationLogger

try:  # pragma: no cover - settings optional
    from .sandbox_settings import SandboxSettings

    ROI_EMA_ALPHA = SandboxSettings().roi_ema_alpha
except Exception:  # pragma: no cover - fallback when settings missing
    ROI_EMA_ALPHA = 0.1

GATING_THRESHOLD = float(os.getenv("ROI_GATING_THRESHOLD", "0.0"))
GATING_CONSECUTIVE = int(os.getenv("ROI_GATING_CONSECUTIVE", "3"))

_roi_delta_ema: dict[str, float] = {}
_gating_counts: dict[str, int] = {}

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


def _update_ema(workflow_id: str, delta: float) -> None:
    prev = _roi_delta_ema.get(workflow_id, 0.0)
    ema = (1 - ROI_EMA_ALPHA) * prev + ROI_EMA_ALPHA * delta
    _roi_delta_ema[workflow_id] = ema
    if ema < GATING_THRESHOLD:
        _gating_counts[workflow_id] = _gating_counts.get(workflow_id, 0) + 1
    else:
        _gating_counts[workflow_id] = 0


def is_stable(workflow_id: int | str) -> bool:
    """Return True when *workflow_id* is gated as stable."""
    return _gating_counts.get(str(workflow_id), 0) >= GATING_CONSECUTIVE


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
    if is_stable(wf_id_str):
        logger.info("workflow %s gated by ROI EMA", wf_id_str)
        return workflow_callable

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

    delta = best_roi - baseline_roi
    _update_ema(wf_id_str, delta)

    if best_variant_seq is not None and best_roi > baseline_roi:
        MutationLogger.log_mutation(
            change=best_variant_seq,
            reason="promoted",
            trigger="workflow_evolution_manager",
            performance=delta,
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


__all__ = ["evolve", "is_stable"]
