"""Meta-planning routines for the self-improvement package.

The :func:`self_improvement_cycle` coroutine drives background workflow
optimization using the optional :class:`MetaWorkflowPlanner` component.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

import asyncio

from ..logging_utils import get_logger, log_record
from ..sandbox_settings import SandboxSettings, load_sandbox_settings
from ..workflow_stability_db import WorkflowStabilityDB

try:  # pragma: no cover - optional dependency
    from ..meta_workflow_planner import MetaWorkflowPlanner
except Exception:  # pragma: no cover - gracefully degrade
    try:
        from meta_workflow_planner import MetaWorkflowPlanner  # type: ignore
    except Exception:  # pragma: no cover - best effort fallback
        MetaWorkflowPlanner = None  # type: ignore

settings = load_sandbox_settings()
PLANNER_INTERVAL = getattr(settings, "meta_planning_interval", 0)
STABLE_WORKFLOWS = WorkflowStabilityDB()


def _get_entropy_threshold(
    cfg: SandboxSettings, db: WorkflowStabilityDB
) -> float:
    """Determine entropy threshold from settings or stored metrics."""
    threshold = cfg.meta_entropy_threshold
    if threshold is None:
        try:
            entropies = [abs(float(v.get("entropy", 0.0))) for v in db.data.values()]
            threshold = max(entropies) if entropies else 0.2
        except Exception:  # pragma: no cover - best effort
            threshold = 0.2
    return float(threshold)


def _should_encode(record: Mapping[str, Any], *, entropy_threshold: float) -> bool:
    """Return True if ``record`` indicates improvement and stability."""
    return (
        float(record.get("roi_gain", 0.0)) > 0.0
        and abs(float(record.get("entropy", 0.0))) <= float(entropy_threshold)
    )


async def self_improvement_cycle(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    interval: float = PLANNER_INTERVAL,
    event_bus: "UnifiedEventBus" | None = None,
) -> None:
    """Background loop evolving ``workflows`` using the meta planner."""
    logger = get_logger("SelfImprovementCycle")
    if MetaWorkflowPlanner is None:
        logger.warning("MetaWorkflowPlanner unavailable; skipping cycle")
        return

    planner = MetaWorkflowPlanner()

    cfg = SandboxSettings()
    mutation_rate = cfg.meta_mutation_rate
    roi_weight = cfg.meta_roi_weight
    domain_penalty = cfg.meta_domain_penalty

    for name, value in {
        "mutation_rate": mutation_rate,
        "roi_weight": roi_weight,
        "domain_transition_penalty": domain_penalty,
    }.items():
        if hasattr(planner, name):
            setattr(planner, name, value)

    entropy_threshold = _get_entropy_threshold(cfg, STABLE_WORKFLOWS)

    async def _log(record: Mapping[str, Any]) -> None:
        chain = record.get("chain", [])
        cid = "->".join(chain)
        roi = float(record.get("roi_gain", 0.0))
        failures = int(record.get("failures", 0))
        entropy = float(record.get("entropy", 0.0))
        if planner.roi_db is not None:
            try:
                planner.roi_db.log_result(
                    workflow_id=cid,
                    run_id="bg",
                    runtime=0.0,
                    success_rate=1.0,
                    roi_gain=roi,
                    workflow_synergy_score=max(0.0, 1.0 - entropy),
                    bottleneck_index=0.0,
                    patchability_score=0.0,
                    module_deltas={},
                )
            except Exception:  # pragma: no cover - logging best effort
                logger.exception("ROI logging failed", extra=log_record(workflow_id=cid))
        if planner.stability_db is not None:
            try:
                planner.stability_db.record_metrics(
                    cid, roi, failures, entropy, roi_delta=roi
                )
            except Exception:  # pragma: no cover - logging best effort
                logger.exception(
                    "stability logging failed", extra=log_record(workflow_id=cid)
                )
        if event_bus is not None:
            try:
                event_bus.publish(
                    "metrics:new",
                    {
                        "bot": cid,
                        "errors": failures,
                        "entropy": entropy,
                        "expense": 1.0,
                        "revenue": 1.0 + roi,
                    },
                )
            except Exception:  # pragma: no cover - best effort
                logger.exception(
                    "failed to publish metrics", extra=log_record(workflow_id=cid)
                )

    while True:
        try:
            records = planner.discover_and_persist(workflows)
            active: list[list[str]] = []
            for rec in records:
                await _log(rec)
                if not _should_encode(rec, entropy_threshold=entropy_threshold):
                    continue
                chain = rec.get("chain", [])
                roi = float(rec.get("roi_gain", 0.0))
                failures = int(rec.get("failures", 0))
                entropy = float(rec.get("entropy", 0.0))
                if chain and roi > 0:
                    active.append(chain)
                    chain_id = "->".join(chain)
                    try:
                        STABLE_WORKFLOWS.record_metrics(
                            chain_id, roi, failures, entropy, roi_delta=roi
                        )
                    except Exception:  # pragma: no cover - best effort
                        logger.exception(
                            "global stability logging failed",
                            extra=log_record(workflow_id=chain_id),
                        )
        except Exception as exc:  # pragma: no cover - planner is best effort
            logger.exception("meta planner execution failed", exc_info=exc)

        for chain in list(active):
            planner.cluster_map.pop(tuple(chain), None)

        await asyncio.sleep(interval)


__all__ = ["self_improvement_cycle", "PLANNER_INTERVAL"]
