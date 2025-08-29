from __future__ import annotations

"""Simulate suggested workflow chains and persist outcomes.

This utility composes workflow callables from suggested chains of module names,
executes them within the :class:`sandbox_runner.WorkflowSandboxRunner` in safe
mode via :class:`CompositeWorkflowScorer` and records ROI, failure rates and
entropy/stability metrics.  Results are appended to
``sandbox_data/chain_simulations.json`` for later retrieval and reinforcement.
"""

from typing import Sequence, Iterable, List, Dict, Any, Mapping, Callable
import json
from pathlib import Path

try:  # pragma: no cover - allow import when used as package or module
    from .workflow_chain_suggester import WorkflowChainSuggester
    from .workflow_evolution_manager import _build_callable
    from .composite_workflow_scorer import CompositeWorkflowScorer
    from .workflow_synergy_comparator import WorkflowSynergyComparator
    from .workflow_stability_db import WorkflowStabilityDB
    from .meta_workflow_planner import MetaWorkflowPlanner
    from . import workflow_run_summary
except Exception:  # pragma: no cover - fallback to absolute imports
    from workflow_chain_suggester import WorkflowChainSuggester  # type: ignore
    from workflow_evolution_manager import _build_callable  # type: ignore
    from composite_workflow_scorer import CompositeWorkflowScorer  # type: ignore
    from workflow_synergy_comparator import WorkflowSynergyComparator  # type: ignore
    from workflow_stability_db import WorkflowStabilityDB  # type: ignore
    from meta_workflow_planner import MetaWorkflowPlanner  # type: ignore
    import workflow_run_summary  # type: ignore

RESULTS_PATH = Path("sandbox_data/chain_simulations.json")


def _persist_outcomes(outcomes: List[Dict[str, Any]], path: Path = RESULTS_PATH) -> None:
    """Append ``outcomes`` to ``path`` best effort."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, Any]] = []
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                existing = data
        except Exception:
            pass
    existing.extend(outcomes)
    try:
        path.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass  # pragma: no cover - best effort


def simulate_chains(chains: Iterable[Sequence[str]]) -> List[Dict[str, Any]]:
    """Execute ``chains`` of module names and persist evaluation metrics."""

    scorer = CompositeWorkflowScorer()
    stability_db = WorkflowStabilityDB()
    outcomes: List[Dict[str, Any]] = []
    for idx, chain in enumerate(chains):
        seq = "-".join(chain)
        workflow_fn = _build_callable(seq)
        wf_id = f"chain-{idx}"
        result = scorer.run(workflow_fn, wf_id, run_id="simulation")

        spec = {"steps": [{"module": m} for m in chain]}
        entropy = WorkflowSynergyComparator._entropy(spec)

        stable = stability_db.is_stable(wf_id, current_roi=result.roi_gain, threshold=0.0)
        if not stable:
            stability_db.mark_stable(wf_id, result.roi_gain)

        workflow_run_summary.record_run(wf_id, result.roi_gain)

        outcomes.append(
            {
                "workflow_id": wf_id,
                "chain": list(chain),
                "roi_gain": result.roi_gain,
                "failure_rate": 1.0 - result.success_rate,
                "entropy": entropy,
                "stable": stable,
            }
        )

    _persist_outcomes(outcomes)
    return outcomes


def simulate_suggested_chains(
    target_embedding: Sequence[float], top_k: int = 3
) -> List[Dict[str, Any]]:
    """Suggest chains via :class:`WorkflowChainSuggester` and evaluate them."""

    suggester = WorkflowChainSuggester()
    chains = suggester.suggest_chains(target_embedding, top_k)
    return simulate_chains(chains)


def run_scheduler(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    roi_delta_threshold: float = 0.01,
    entropy_delta_threshold: float = 0.01,
    runs: int = 3,
) -> List[Dict[str, Any]]:
    """Execute :class:`MetaWorkflowPlanner` scheduler and persist results."""

    planner = MetaWorkflowPlanner()
    records = planner.schedule(
        workflows,
        roi_delta_threshold=roi_delta_threshold,
        entropy_delta_threshold=entropy_delta_threshold,
        runs=runs,
    )
    _persist_outcomes(records)
    return records


__all__ = ["simulate_chains", "simulate_suggested_chains", "run_scheduler"]
