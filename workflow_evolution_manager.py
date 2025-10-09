from __future__ import annotations

"""Manage workflow evolution by benchmarking generated variants."""

from typing import Callable, Optional, Iterable, Sequence, Mapping, Dict, Any
import importlib
import json
import logging
import os
from pathlib import Path

try:  # pragma: no cover - prefer package relative import
    from .composite_workflow_scorer import CompositeWorkflowScorer
except ImportError:  # pragma: no cover - allow execution as script
    from composite_workflow_scorer import CompositeWorkflowScorer  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from .workflow_evolution_bot import WorkflowEvolutionBot
except ImportError:  # pragma: no cover - allow execution as script
    from workflow_evolution_bot import WorkflowEvolutionBot  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from .roi_results_db import ROIResultsDB
except ImportError:  # pragma: no cover - allow execution as script
    from roi_results_db import ROIResultsDB  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from .roi_tracker import ROITracker
except ImportError:  # pragma: no cover - allow execution as script
    from roi_tracker import ROITracker  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from .workflow_stability_db import WorkflowStabilityDB
except ImportError:  # pragma: no cover - allow execution as script
    from workflow_stability_db import WorkflowStabilityDB  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from .workflow_summary_db import WorkflowSummaryDB
except ImportError:  # pragma: no cover - allow execution as script
    from workflow_summary_db import WorkflowSummaryDB  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from .sandbox_settings import SandboxSettings
except ImportError:  # pragma: no cover - allow execution as script
    from sandbox_settings import SandboxSettings  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from .workflow_synthesizer import save_workflow
except ImportError:  # pragma: no cover - allow execution as script
    from workflow_synthesizer import save_workflow  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from . import workflow_run_summary
except ImportError:  # pragma: no cover - allow execution as script
    import workflow_run_summary  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from . import sandbox_runner
except ImportError:  # pragma: no cover - allow execution as script
    import sandbox_runner  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from .workflow_synergy_comparator import WorkflowSynergyComparator
except ImportError:  # pragma: no cover - allow execution as script
    from workflow_synergy_comparator import WorkflowSynergyComparator  # type: ignore

try:  # pragma: no cover - prefer package relative import
    from .meta_workflow_planner import MetaWorkflowPlanner
except ImportError:  # pragma: no cover - allow execution as script
    from meta_workflow_planner import MetaWorkflowPlanner  # type: ignore
from context_builder_util import create_context_builder
try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder  # type: ignore
except Exception:  # pragma: no cover - allow running without builder
    ContextBuilder = None  # type: ignore
try:  # pragma: no cover - allow running as script
    from .dynamic_path_router import resolve_path, get_project_root  # type: ignore
except Exception:  # pragma: no cover - fallback when executed directly
    from dynamic_path_router import resolve_path, get_project_root  # type: ignore
try:  # pragma: no cover - optional dependency
    from vector_service.retriever import Retriever  # type: ignore
except Exception:  # pragma: no cover - allow running without retriever
    Retriever = None  # type: ignore
try:  # pragma: no cover - optional at runtime
    from .workflow_graph import WorkflowGraph
except Exception:  # pragma: no cover - best effort
    try:  # pragma: no cover - allow execution as script
        from workflow_graph import WorkflowGraph  # type: ignore
    except Exception:  # pragma: no cover - best effort
        WorkflowGraph = None  # type: ignore
try:  # pragma: no cover - optional at runtime
    from .workflow_lineage import load_specs as _load_specs
except Exception:  # pragma: no cover - best effort
    try:  # pragma: no cover - allow execution as script
        from workflow_lineage import load_specs as _load_specs  # type: ignore
    except Exception:  # pragma: no cover - best effort
        _load_specs = None  # type: ignore
try:
    from .evolution_history_db import EvolutionHistoryDB, EvolutionEvent
except Exception:  # pragma: no cover - optional dependency
    try:  # pragma: no cover - allow execution as script
        from evolution_history_db import EvolutionHistoryDB, EvolutionEvent  # type: ignore
    except Exception:  # pragma: no cover - best effort
        EvolutionHistoryDB = None  # type: ignore
        EvolutionEvent = None  # type: ignore
try:  # pragma: no cover - prefer package relative import
    from . import mutation_logger as MutationLogger
except ImportError:  # pragma: no cover - allow execution as script
    import mutation_logger as MutationLogger  # type: ignore

try:  # pragma: no cover - optional dependency
    from .workflow_branch_manager import merge_sibling_branches as _merge_sibling_branches
except Exception:  # pragma: no cover - best effort
    try:  # pragma: no cover - allow execution as script
        from workflow_branch_manager import (  # type: ignore
            merge_sibling_branches as _merge_sibling_branches,
        )
    except Exception:  # pragma: no cover - best effort
        _merge_sibling_branches = None  # type: ignore

logger = logging.getLogger(__name__)


def consume_planner_suggestions(chains: Iterable[Sequence[str]]) -> None:
    """Persist planner-suggested chains for later evaluation."""
    for idx, chain in enumerate(chains, start=1):
        try:
            path = resolve_path(f"sandbox_data/planner_chain_{idx}.workflow.json")
            save_workflow([{"module": m} for m in chain], path)
        except Exception:
            logger.exception("failed to save planner suggestion", extra={"chain": chain})

STABLE_WORKFLOWS = WorkflowStabilityDB()
EVOLUTION_DB = EvolutionHistoryDB() if EvolutionHistoryDB is not None else None


def merge_variant_records(
    records: Sequence[Dict[str, Any]],
    workflows: Mapping[str, Callable[[], Any]],
    *,
    roi_threshold: float = 0.0,
    planner: MetaWorkflowPlanner | None = None,
    runner: sandbox_runner.workflow_sandbox_runner.WorkflowSandboxRunner | None = None,
    failure_threshold: int = 0,
    entropy_threshold: float = 2.0,
    retriever: Retriever | None = None,
    context_builder: ContextBuilder,
) -> list[Dict[str, Any]]:
    """Merge high-performing variants using :class:`MetaWorkflowPlanner`.

    This helper delegates to
    :meth:`meta_workflow_planner.MetaWorkflowPlanner.merge_high_performing_variants`
    to combine variant pipelines that individually exceed ``roi_threshold``.
    """

    planner = planner or MetaWorkflowPlanner(context_builder=context_builder)
    if retriever is None:
        raise ValueError("merge_variant_records requires a Retriever")
    return planner.merge_high_performing_variants(
        records,
        workflows,
        roi_threshold=roi_threshold,
        runner=runner,
        failure_threshold=failure_threshold,
        entropy_threshold=entropy_threshold,
        context_builder=context_builder,
        retriever=retriever,
    )


def _merge_branches_for_parent(parent_id: str) -> None:
    """Attempt automatic merging of sibling branches for ``parent_id``."""

    if _merge_sibling_branches is None:
        return
    try:  # pragma: no cover - best effort
        _merge_sibling_branches(parent_id=parent_id)
    except Exception:  # pragma: no cover - best effort
        logger.exception("failed merging sibling branches for %s", parent_id)


def _update_ema(workflow_id: str, delta: float) -> bool:
    """Update ROI improvement EMA and return ``True`` when gating triggers."""

    alpha = SandboxSettings().roi_ema_alpha
    ema, count = STABLE_WORKFLOWS.get_ema(workflow_id)
    ema = alpha * float(delta) + (1 - alpha) * ema

    threshold = float(os.getenv("ROI_GATING_THRESHOLD", "0") or 0.0)
    if ema < threshold:
        count += 1
    else:
        count = 0

    STABLE_WORKFLOWS.set_ema(workflow_id, ema, count)

    limit = int(os.getenv("ROI_GATING_CONSECUTIVE", "0") or 0)
    return limit > 0 and count >= limit


def workflow_roi_ema(workflow_id: str | int) -> float:
    ema, _ = STABLE_WORKFLOWS.get_ema(str(workflow_id))
    return ema


def _build_callable(sequence: str) -> Callable[[], bool]:
    """Best-effort construction of a workflow callable from *sequence*.

    Each element in ``sequence`` is treated as an importable module name.  The
    manager attempts to execute a ``main`` or ``run`` attribute from each module
    sequentially.  Missing modules or callables log an error and propagate
    failure so ROI metrics accurately reflect the absence of steps.
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
            logger.error("module %s lacks main/run callable", step)
        except Exception as exc:  # pragma: no cover - import failures
            logger.error("failed importing %s: %s", step, exc)

        def _missing() -> bool:
            raise ModuleNotFoundError(f"module {step} missing or invalid")

        _missing.__name__ = step
        funcs.append(_missing)

    def _workflow() -> bool:
        ok = True
        for fn in funcs:
            ok = bool(fn()) and ok
        return ok

    return _workflow


def _scores_overfit(scores) -> bool:
    """Return ``True`` if either workflow shows overfitting signals."""

    def _flag(report):
        return bool(report) and getattr(report, "is_overfitting", lambda: False)()

    return _flag(getattr(scores, "overfit_a", None)) or _flag(
        getattr(scores, "overfit_b", None)
    )


def _evaluate_merge(
    base_id: str,
    dup_id: str,
    out_dir: Path,
    scorer: CompositeWorkflowScorer,
    wf_id: str,
    run_id: str,
    baseline_roi: float,
):
    """Merge ``dup_id`` into ``base_id`` and re-evaluate the result."""

    merged_file = WorkflowSynergyComparator.merge_duplicate(base_id, dup_id, out_dir)
    if not merged_file:
        return None
    try:
        merged_data = json.loads(merged_file.read_text())
    except Exception:
        return None

    merged_steps = merged_data.get("steps", [])
    seq = "-".join(s.get("module") for s in merged_steps if s.get("module"))
    merged_callable = _build_callable(seq)
    merge_run_id = f"merge-{run_id}"
    merged_result = scorer.run(merged_callable, wf_id, run_id=merge_run_id)

    merged_id = merged_data.get("metadata", {}).get("workflow_id")
    if merged_id:
        workflow_run_summary.record_run(str(merged_id), merged_result.roi_gain)
        setattr(merged_callable, "workflow_id", merged_id)
        if EVOLUTION_DB is not None and EvolutionEvent is not None:
            try:  # pragma: no cover - best effort
                EVOLUTION_DB.add(
                    EvolutionEvent(
                        action="merge",
                        before_metric=baseline_roi,
                        after_metric=merged_result.roi_gain,
                        roi=merged_result.roi_gain - baseline_roi,
                        workflow_id=int(merged_id),
                        reason="merge",
                        trigger="workflow_evolution_manager",
                        performance=merged_result.roi_gain - baseline_roi,
                    )
                )
            except Exception:
                logger.exception("failed logging merged lineage event")

    return merged_callable, merged_result, merged_id, merged_file


def is_stable(workflow_id: int | str) -> bool:
    """Return ``True`` when *workflow_id* is marked stable."""
    return STABLE_WORKFLOWS.is_stable(str(workflow_id))


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
    tracker = ROITracker()

    # Baseline evaluation
    baseline_scorer = CompositeWorkflowScorer(
        results_db=results_db, tracker=tracker
    )
    baseline_result = baseline_scorer.run(
        workflow_callable, wf_id_str, run_id="baseline"
    )
    baseline_roi = baseline_result.roi_gain
    baseline_synergy = getattr(baseline_result, "workflow_synergy_score", 0.0)
    workflow_run_summary.record_run(wf_id_str, baseline_roi)

    # Attach lineage metadata to the baseline callable so it can be returned
    # unchanged while still exposing parent information.
    setattr(workflow_callable, "parent_id", workflow_id)
    setattr(workflow_callable, "mutation_description", None)

    # Skip variant generation when workflow marked stable and ROI unchanged
    if STABLE_WORKFLOWS.is_stable(wf_id_str, baseline_roi, tracker.diminishing()):
        logger.info("workflow %s marked stable", wf_id_str)
        _, raroi, _ = tracker.calculate_raroi(baseline_roi)
        tracker.score_workflow(wf_id_str, raroi)
        workflow_run_summary.save_all_summaries("workflows")
        _merge_branches_for_parent(wf_id_str)
        return workflow_callable

    bot = WorkflowEvolutionBot()
    settings = SandboxSettings()

    best_callable = workflow_callable
    best_roi = baseline_roi
    best_variant_seq: Optional[str] = None
    best_variant_result: Optional[object] = None
    best_delta = 0.0

    for seq in bot.generate_variants(limit=variants, workflow_id=int(workflow_id)):
        variant_callable = _build_callable(seq)
        # propagate lineage metadata to variant implementations
        setattr(variant_callable, "parent_id", workflow_id)
        setattr(variant_callable, "mutation_description", seq)
        run_id = f"variant-{hash(seq) & 0xffffffff:x}"

        scorer = CompositeWorkflowScorer(results_db=results_db, tracker=tracker)
        variant_result = scorer.run(variant_callable, wf_id_str, run_id=run_id)

        baseline_spec = getattr(baseline_result, "workflow_spec", None)
        variant_spec = getattr(variant_result, "workflow_spec", None)
        merged_parent: str | None = None
        if baseline_spec and variant_spec:
            try:
                scores = WorkflowSynergyComparator.compare(
                    {"steps": baseline_spec}, {"steps": variant_spec}
                )
                bp_thresh = getattr(settings, "best_practice_match_threshold", 1.0)
                bp_match = getattr(scores, "best_practice_match_b", (0.0, []))
                bp_score = bp_match[0]
                if bp_score >= bp_thresh:
                    try:
                        mods = WorkflowSynergyComparator._extract_modules(
                            {"steps": variant_spec}
                        )
                        WorkflowSynergyComparator._update_best_practices(mods)
                    except Exception:  # pragma: no cover - best effort
                        logger.exception("failed updating best practices")
                    if getattr(scores, "entropy_b", 1.0) <= settings.duplicate_entropy:
                        merged_parent = "best_practice"
                setattr(variant_result, "best_practice_match", bp_score)

                ent_delta = abs(scores.entropy_a - scores.entropy_b)
                if (
                    merged_parent is None
                    and scores.aggregate >= settings.workflow_merge_similarity
                    and ent_delta <= settings.workflow_merge_entropy_delta
                    and not _scores_overfit(scores)
                ):
                    tmp_dir = get_project_root() / f"{wf_id_str}.merge"
                    merged_file: Path | None = None
                    try:
                        tmp_dir.mkdir(exist_ok=True)
                        base_id = "base"
                        dup_id = "variant"
                        (tmp_dir / f"{base_id}.workflow.json").write_text(
                            json.dumps({"steps": baseline_spec})
                        )
                        (tmp_dir / f"{dup_id}.workflow.json").write_text(
                            json.dumps({"steps": variant_spec})
                        )
                        merged = _evaluate_merge(
                            base_id,
                            dup_id,
                            tmp_dir,
                            scorer,
                            wf_id_str,
                            run_id,
                            baseline_roi,
                        )
                        if merged:
                            variant_callable, variant_result, merged_id, merged_file = merged
                            setattr(variant_callable, "parent_id", workflow_id)
                    finally:
                        for p in [
                            tmp_dir / f"{base_id}.workflow.json",
                            tmp_dir / f"{dup_id}.workflow.json",
                            merged_file,
                        ]:
                            try:
                                if p:
                                    p.unlink()
                            except Exception:
                                pass
                        try:
                            tmp_dir.rmdir()
                        except Exception:
                            pass
            except Exception:
                logger.exception("workflow merge check failed")

        # Deduplicate against existing stable workflows
        if variant_spec:
            comparator = WorkflowSynergyComparator()
            candidates: list[tuple[str, Path]] = []
            seen_paths: set[Path] = set()
            try:
                if _load_specs is not None:
                    for spec in _load_specs("workflows"):
                        wid = str(spec.get("workflow_id"))
                        if wid and STABLE_WORKFLOWS.is_stable(wid):
                            path = resolve_path(f"workflows/{wid}.workflow.json")
                            if path.exists() and path not in seen_paths:
                                candidates.append((wid, path))
                                seen_paths.add(path)
                elif WorkflowGraph is not None:
                    graph = WorkflowGraph()
                    node_ids = getattr(getattr(graph, "graph", graph), "nodes", lambda: [])
                    for wid in node_ids() if callable(node_ids) else node_ids:
                        wid_str = str(wid)
                        if STABLE_WORKFLOWS.is_stable(wid_str):
                            path = resolve_path(f"workflows/{wid_str}.workflow.json")
                            if path.exists() and path not in seen_paths:
                                candidates.append((wid_str, path))
                                seen_paths.add(path)
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed loading candidate workflows")

            for cand_id, cand_path in candidates:
                try:
                    cand_spec = json.loads(cand_path.read_text())
                    scores = comparator.compare({"steps": variant_spec}, cand_spec)
                    bp_thresh = getattr(settings, "best_practice_match_threshold", 1.0)
                    bp_match = getattr(scores, "best_practice_match_a", (0.0, []))
                    if bp_match[0] >= bp_thresh:
                        try:
                            mods = type(comparator)._extract_modules({"steps": variant_spec})
                            type(comparator)._update_best_practices(mods)
                        except Exception:  # pragma: no cover - best effort
                            logger.exception("failed updating best practices")
                        if getattr(scores, "entropy_a", 1.0) <= settings.duplicate_entropy:
                            merged_parent = "best_practice"
                            break
                    overfit = _scores_overfit(scores)
                    if (
                        merged_parent is None
                        and comparator.is_duplicate(
                            scores,
                            similarity_threshold=settings.duplicate_similarity,
                            entropy_threshold=settings.duplicate_entropy,
                        )
                        and scores.aggregate >= settings.duplicate_similarity
                        and not overfit
                    ):
                        dup_id = f"{cand_id}_dup"
                        dup_path = comparator.workflow_dir / f"{dup_id}.workflow.json"
                        merged = None
                        try:
                            dup_path.write_text(json.dumps({"steps": variant_spec}))
                            merged = _evaluate_merge(
                                cand_id,
                                dup_id,
                                comparator.workflow_dir,
                                CompositeWorkflowScorer(
                                    results_db=results_db, tracker=tracker
                                ),
                                str(cand_id),
                                f"{cand_id}-{run_id}",
                                baseline_roi,
                            )
                            if merged:
                                (
                                    variant_callable,
                                    variant_result,
                                    merged_id,
                                    merged_file,
                                ) = merged
                                setattr(variant_callable, "parent_id", cand_id)
                                merged_parent = str(cand_id)
                        finally:
                            try:
                                dup_path.unlink()
                            except Exception:
                                pass
                            try:
                                if merged and merged_file:
                                    merged_file.unlink()
                            except Exception:
                                pass
                        if merged:
                            break
                except Exception:  # pragma: no cover - best effort
                    logger.exception("failed comparing workflow %s", cand_id)

        if merged_parent is not None:
            setattr(variant_callable, "parent_id", merged_parent)

        roi_delta = variant_result.roi_gain - baseline_roi
        bp_thresh = getattr(settings, "best_practice_match_threshold", 1.0)
        bp_score = getattr(variant_result, "best_practice_match", 0.0)
        if bp_score >= bp_thresh:
            roi_delta += bp_score * 0.001

        # Persist ROI delta with variant identifier
        results_db.log_module_delta(
            wf_id_str,
            run_id,
            module=f"variant:{seq}",
            runtime=variant_result.runtime,
            success_rate=variant_result.success_rate,
            roi_delta=roi_delta,
        )

        parent_id = merged_parent or bot._rearranged_events.get(seq)
        event_id = MutationLogger.log_mutation(
            change=seq,
            reason="benchmark",
            trigger="workflow_evolution_manager",
            performance=roi_delta,
            workflow_id=int(workflow_id),
            before_metric=baseline_roi,
            after_metric=variant_result.roi_gain,
            parent_id=parent_id,
        )

        if EVOLUTION_DB is not None and EvolutionEvent is not None:
            try:
                EVOLUTION_DB.add(
                    EvolutionEvent(
                        action=seq,
                        before_metric=baseline_roi,
                        after_metric=variant_result.roi_gain,
                        roi=roi_delta,
                        workflow_id=int(workflow_id),
                        reason="benchmark",
                        trigger="workflow_evolution_manager",
                        performance=roi_delta,
                        parent_event_id=parent_id,
                    )
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed logging variant benchmark")

        log_evo = getattr(MutationLogger, "log_workflow_evolution", None)
        if callable(log_evo):
            try:
                log_evo(
                    workflow_id=int(workflow_id),
                    variant=seq,
                    baseline_roi=baseline_roi,
                    variant_roi=variant_result.roi_gain,
                    baseline_synergy=baseline_synergy,
                    variant_synergy=getattr(
                        variant_result, "workflow_synergy_score", 0.0
                    ),
                    mutation_id=event_id,
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed logging workflow evolution")

        if _update_ema(wf_id_str, roi_delta):
            STABLE_WORKFLOWS.mark_stable(wf_id_str, baseline_roi)
            logger.info("workflow %s stable (ema gating)", wf_id_str)
            try:
                WorkflowSummaryDB().set_summary(int(workflow_id), "stable")
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed to flag workflow %s as stable", workflow_id)
            MutationLogger.log_mutation(
                change="stable",
                reason="stable",
                trigger="workflow_evolution_manager",
                performance=0.0,
                workflow_id=int(workflow_id),
                before_metric=baseline_roi,
                after_metric=baseline_roi,
            )
            workflow_run_summary.save_all_summaries("workflows")
            _merge_branches_for_parent(wf_id_str)
            return workflow_callable

        if roi_delta > best_delta:
            best_delta = roi_delta
            best_roi = variant_result.roi_gain
            best_callable = variant_callable
            best_variant_seq = seq
            best_variant_result = variant_result

    delta = best_delta

    # Record final RAROI for history
    _, final_raroi, _ = tracker.calculate_raroi(best_roi)
    tracker.score_workflow(wf_id_str, final_raroi)

    if delta <= 0:
        STABLE_WORKFLOWS.mark_stable(wf_id_str, baseline_roi)
        logger.info("workflow %s stable (delta=%.4f)", wf_id_str, delta)
        try:
            WorkflowSummaryDB().set_summary(int(workflow_id), "stable")
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed to flag workflow %s as stable", workflow_id)
        MutationLogger.log_mutation(
            change="stable",
            reason="stable",
            trigger="workflow_evolution_manager",
            performance=0.0,
            workflow_id=int(workflow_id),
            before_metric=baseline_roi,
            after_metric=baseline_roi,
        )
        # lineage already attached to workflow_callable
        workflow_run_summary.save_all_summaries("workflows")
        _merge_branches_for_parent(wf_id_str)
        return workflow_callable

    if best_variant_seq is not None and delta > 0:
        STABLE_WORKFLOWS.clear(wf_id_str)
        event_id = MutationLogger.log_mutation(
            change=best_variant_seq,
            reason="promoted",
            trigger="workflow_evolution_manager",
            performance=delta,
            workflow_id=int(workflow_id),
            before_metric=baseline_roi,
            after_metric=best_roi,
        )
        log_evo = getattr(MutationLogger, "log_workflow_evolution", None)
        if callable(log_evo):
            try:
                log_evo(
                    workflow_id=int(workflow_id),
                    variant=best_variant_seq,
                    baseline_roi=baseline_roi,
                    variant_roi=best_roi,
                    baseline_synergy=baseline_synergy,
                    variant_synergy=getattr(
                        best_variant_result, "workflow_synergy_score", 0.0
                    ),
                    mutation_id=event_id,
                )
            except Exception:
                logger.exception("record promotion failed")
        new_id = None
        created_at = None
        try:
            spec_steps = getattr(best_variant_result, "workflow_spec", None)
            if isinstance(spec_steps, list):
                steps = spec_steps
            else:
                steps = [
                    {"module": s, "inputs": [], "outputs": []}
                    for s in best_variant_seq.split("-")
                    if s
                ]
            path = resolve_path(f"{workflow_id}.workflow.json")
            saved_path, metadata = save_workflow(
                steps,
                path,
                parent_id=str(workflow_id),
                mutation_description=best_variant_seq,
            )
            new_id = metadata.get("workflow_id")
            created_at = metadata.get("created_at")
            if new_id is not None:
                workflow_run_summary.record_run(str(new_id), best_roi)
                tracker.score_workflow(str(new_id), final_raroi)
                setattr(best_callable, "workflow_id", new_id)
                try:
                    if saved_path.name != f"{new_id}.workflow.json":
                        new_path = saved_path.with_name(
                            f"{new_id}.workflow.json"
                        )
                        saved_path.rename(new_path)
                        saved_path = new_path
                except OSError:
                    logger.exception(
                        "failed renaming workflow spec to %s", new_id
                    )
            if created_at is not None:
                setattr(best_callable, "created_at", created_at)
            logger.info("saved promoted workflow to %s", saved_path)
            if getattr(best_variant_result, "best_practice_match", 0.0) >= getattr(settings, "best_practice_match_threshold", 1.0):
                try:
                    WorkflowSynergyComparator._update_best_practices(
                        [s.get("module") for s in steps if s.get("module")]
                    )
                except Exception:  # pragma: no cover - best effort
                    logger.exception("failed updating best practices")
        except Exception:  # pragma: no cover - best effort
            logger.exception("failed persisting promoted workflow")
        if WorkflowGraph is not None:
            graph = WorkflowGraph()
            target_id = new_id if new_id is not None else str(workflow_id)
            if hasattr(graph, "update_workflow"):
                graph.update_workflow(target_id, roi=best_roi)
            else:
                graph.add_workflow(target_id, roi=best_roi)
            if new_id is not None:
                graph.add_dependency(
                    str(workflow_id), new_id, dependency_type="evolution"
                )

        try:
            from .workflow_benchmark import benchmark_workflow
            from .data_bot import MetricsDB
            from .neuroplasticity import PathwayDB

            runner = sandbox_runner.WorkflowSandboxRunner()

            def _sandboxed() -> bool:
                metrics = runner.run(best_callable, safe_mode=True)
                return all(m.success for m in getattr(metrics, "modules", []))

            benchmark_workflow(_sandboxed, MetricsDB(), PathwayDB(), name=wf_id_str)
        except Exception:
            logger.exception("benchmark promoted workflow failed")

        # Post-promotion orphan discovery and integration
        try:
            from db_router import GLOBAL_ROUTER
            from sandbox_runner.post_update import integrate_orphans
        except Exception:
            integrate_orphans = None  # type: ignore
        if integrate_orphans is not None:
            repo = get_project_root()
            builder = create_context_builder()
            integrate_orphans(repo, router=GLOBAL_ROUTER, context_builder=builder)

        # Deduplicate against existing stable workflows
        comparator = WorkflowSynergyComparator()
        merged_callable = best_callable
        try:
            promoted_spec = json.loads(saved_path.read_text())
        except Exception:
            promoted_spec = None
        candidates: list[tuple[str, Path]] = []
        if new_id is not None and promoted_spec is not None:
            seen_paths: set[Path] = set()
            try:
                if _load_specs is not None:
                    for spec in _load_specs("workflows"):
                        wid = str(spec.get("workflow_id"))
                        if wid and wid != str(new_id) and STABLE_WORKFLOWS.is_stable(wid):
                            path = resolve_path(f"workflows/{wid}.workflow.json")
                            if path.exists() and path not in seen_paths:
                                candidates.append((wid, path))
                                seen_paths.add(path)
                elif WorkflowGraph is not None:
                    graph = WorkflowGraph()
                    node_ids = getattr(getattr(graph, "graph", graph), "nodes", lambda: [])
                    for wid in node_ids() if callable(node_ids) else node_ids:
                        wid_str = str(wid)
                        if wid_str != str(new_id) and STABLE_WORKFLOWS.is_stable(wid_str):
                            path = resolve_path(f"workflows/{wid_str}.workflow.json")
                            if path.exists() and path not in seen_paths:
                                candidates.append((wid_str, path))
                                seen_paths.add(path)
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed loading candidate workflows")

        for cand_id, cand_path in candidates:
            try:
                cand_spec = json.loads(cand_path.read_text())
                scores = comparator.compare(promoted_spec, cand_spec)
                bp_thresh = getattr(settings, "best_practice_match_threshold", 1.0)
                bp_match = getattr(scores, "best_practice_match_a", (0.0, []))
                if bp_match[0] >= bp_thresh:
                    try:
                        mods = type(comparator)._extract_modules(promoted_spec)
                        type(comparator)._update_best_practices(mods)
                    except Exception:  # pragma: no cover - best effort
                        logger.exception("failed updating best practices")
                    if getattr(scores, "entropy_a", 1.0) <= settings.duplicate_entropy:
                        setattr(best_callable, "parent_id", "best_practice")
                        break
                ent_gap = abs(scores.entropy_a - scores.entropy_b)
                logger.debug(
                    "duplicate check %s -> %s: aggregate=%.3f expandability=%.3f ent_gap=%.3f",
                    new_id,
                    cand_id,
                    scores.aggregate,
                    scores.expandability,
                    ent_gap,
                )
                if (
                    scores.aggregate >= settings.duplicate_similarity
                    and scores.expandability <= settings.duplicate_entropy
                    and ent_gap <= settings.duplicate_entropy
                    and not _scores_overfit(scores)
                ):
                    merged = _evaluate_merge(
                        cand_id,
                        str(new_id),
                        comparator.workflow_dir,
                        CompositeWorkflowScorer(
                            results_db=results_db, tracker=tracker
                        ),
                        str(cand_id),
                        f"{cand_id}-{new_id}",
                        best_roi,
                    )
                    if merged:
                        merged_callable, merged_result, merged_id, merged_file = merged
                        if merged_id:
                            tracker.score_workflow(str(merged_id), final_raroi)
                        best_callable = merged_callable
                        saved_path = merged_file
                        new_id = merged_id or new_id
                        best_roi = merged_result.roi_gain
                        break
            except Exception:  # pragma: no cover - best effort
                logger.exception("failed comparing workflow %s", cand_id)

        # ensure promoted callable exposes lineage metadata
        if not getattr(best_callable, "parent_id", None):
            setattr(best_callable, "parent_id", workflow_id)
        setattr(best_callable, "mutation_description", best_variant_seq)

        workflow_run_summary.save_all_summaries("workflows")
        _merge_branches_for_parent(wf_id_str)
        return best_callable

    workflow_run_summary.save_all_summaries("workflows")
    _merge_branches_for_parent(wf_id_str)
    return workflow_callable


class WorkflowEvolutionManager:
    """Lightweight wrapper exposing workflow evolution helpers.

    This class allows embedding the evolution utilities as a dependency
    without relying on module level functions.  It simply delegates to the
    functional implementations defined in this module.
    """

    @staticmethod
    def build_callable(sequence: str) -> Callable[[], bool]:
        """Return a callable constructed from *sequence* steps."""

        return _build_callable(sequence)

    def evolve(
        self,
        workflow_callable: Callable[[], bool],
        workflow_id: int | str,
        variants: int = 5,
    ) -> Callable[[], bool]:
        """Proxy to :func:`evolve`."""

        return evolve(workflow_callable, workflow_id, variants)

    def is_stable(self, workflow_id: int | str) -> bool:
        """Proxy to :func:`is_stable`."""

        return is_stable(workflow_id)


__all__ = ["evolve", "is_stable", "workflow_roi_ema", "WorkflowEvolutionManager"]
