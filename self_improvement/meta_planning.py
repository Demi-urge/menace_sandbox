"""Meta-planning routines for the self-improvement package.

The :func:`self_improvement_cycle` coroutine drives background workflow
optimization using the optional :class:`MetaWorkflowPlanner` component.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from statistics import fmean
import asyncio

from ..logging_utils import get_logger, log_record
from ..sandbox_settings import SandboxSettings, load_sandbox_settings
from ..workflow_stability_db import WorkflowStabilityDB
from ..roi_results_db import ROIResultsDB

try:  # pragma: no cover - optional dependency
    from ..unified_event_bus import UnifiedEventBus
except Exception:  # pragma: no cover - fallback when event bus missing
    UnifiedEventBus = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from ..meta_workflow_planner import MetaWorkflowPlanner
except Exception:  # pragma: no cover - gracefully degrade
    try:
        from meta_workflow_planner import MetaWorkflowPlanner  # type: ignore
    except Exception:  # pragma: no cover - best effort fallback
        MetaWorkflowPlanner = None  # type: ignore


class _FallbackPlanner:
    """Lightweight planner leveraging ROI and stability metrics.

    The implementation intentionally mirrors the public interface of
    :class:`MetaWorkflowPlanner` so the rest of the system can continue to
    operate when the full planner is unavailable.  It performs a very small
    amount of optimisation by selecting individual workflows that have shown a
    positive average ROI and have been marked stable.
    """

    def __init__(self) -> None:
        try:
            self.roi_db: ROIResultsDB | None = ROIResultsDB()
        except Exception:  # pragma: no cover - best effort
            self.roi_db = None
        try:
            self.stability_db: WorkflowStabilityDB | None = WorkflowStabilityDB()
        except Exception:  # pragma: no cover - best effort
            self.stability_db = None
        self.cluster_map: dict[tuple[str, ...], dict[str, Any]] = {}
        self.mutation_rate = 0.0
        self.roi_weight = 0.0
        self.domain_transition_penalty = 0.0
        self.roi_window = 5

    # ------------------------------------------------------------------
    def begin_run(self, workflow_id: str, run_id: str) -> None:  # pragma: no cover - no-op
        """Compatibility stub for :class:`MetaWorkflowPlanner`."""

    # ------------------------------------------------------------------
    def discover_and_persist(
        self, workflows: Mapping[str, Callable[[], Any]]
    ) -> list[Mapping[str, Any]]:
        """Return stable workflows ordered by average ROI."""

        records: list[dict[str, Any]] = []
        for wid in workflows:
            roi = 0.0
            failures = 0
            entropy = 0.0
            if self.roi_db is not None:
                try:
                    results = self.roi_db.fetch_results(wid)
                    recent = [r.roi_gain for r in results[-self.roi_window :]]
                    roi = fmean(recent) if recent else 0.0
                except Exception:  # pragma: no cover - best effort
                    roi = 0.0
            stable = True
            if self.stability_db is not None:
                try:
                    entry = self.stability_db.data.get(wid, {})
                    failures = int(entry.get("failures", 0))
                    entropy = float(entry.get("entropy", 0.0))
                    stable = self.stability_db.is_stable(
                        wid, current_roi=roi, threshold=1.0
                    )
                except Exception:  # pragma: no cover - best effort
                    stable = True
            if roi > 0.0 and stable:
                self.cluster_map[(wid,)] = {"last_roi": roi}
                records.append(
                    {
                        "chain": [wid],
                        "roi_gain": roi,
                        "failures": failures,
                        "entropy": entropy,
                    }
                )

        records.sort(key=lambda r: r["roi_gain"], reverse=True)
        return records

    # ------------------------------------------------------------------
    def mutate_pipeline(
        self,
        chain: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        **_: Any,
    ) -> list[Mapping[str, Any]]:  # pragma: no cover - interface stub
        return []

    # ------------------------------------------------------------------
    def split_pipeline(
        self,
        chain: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        **_: Any,
    ) -> list[Mapping[str, Any]]:  # pragma: no cover - interface stub
        return []

    # ------------------------------------------------------------------
    def remerge_pipelines(
        self,
        pipelines: Sequence[Sequence[str]],
        workflows: Mapping[str, Callable[[], Any]],
        **_: Any,
    ) -> list[Mapping[str, Any]]:  # pragma: no cover - interface stub
        return []


settings = load_sandbox_settings()


def _validate_config(cfg: SandboxSettings) -> None:
    """Validate meta planning configuration on import."""
    if cfg.meta_entropy_threshold is not None and not 0 <= cfg.meta_entropy_threshold <= 1:
        raise ValueError("meta_entropy_threshold must be between 0 and 1")
    for attr in ("meta_mutation_rate", "meta_roi_weight", "meta_domain_penalty"):
        if getattr(cfg, attr) < 0:
            raise ValueError(f"{attr} must be non-negative")


_validate_config(settings)

PLANNER_INTERVAL = getattr(settings, "meta_planning_interval", 0)
MUTATION_RATE = settings.meta_mutation_rate
ROI_WEIGHT = settings.meta_roi_weight
DOMAIN_PENALTY = settings.meta_domain_penalty
DEFAULT_ENTROPY_THRESHOLD = 0.2
ENTROPY_THRESHOLD = (
    settings.meta_entropy_threshold
    if settings.meta_entropy_threshold is not None
    else DEFAULT_ENTROPY_THRESHOLD
)
STABLE_WORKFLOWS = WorkflowStabilityDB()


def _get_entropy_threshold(
    cfg: SandboxSettings, db: WorkflowStabilityDB
) -> float:
    """Determine entropy threshold from settings or stored metrics."""
    threshold = cfg.meta_entropy_threshold
    if threshold is None:
        try:
            entropies = [abs(float(v.get("entropy", 0.0))) for v in db.data.values()]
            threshold = max(entropies) if entropies else DEFAULT_ENTROPY_THRESHOLD
        except Exception:  # pragma: no cover - best effort
            threshold = DEFAULT_ENTROPY_THRESHOLD
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
    event_bus: UnifiedEventBus | None = None,
) -> None:
    """Background loop evolving ``workflows`` using the meta planner."""
    logger = get_logger("SelfImprovementCycle")
    cfg = load_sandbox_settings()
    if MetaWorkflowPlanner is None:
        if getattr(cfg, "enable_meta_planner", False):
            raise RuntimeError("MetaWorkflowPlanner required but not installed")
        logger.warning("MetaWorkflowPlanner unavailable; using fallback planner")
        planner = _FallbackPlanner()
    else:
        planner = MetaWorkflowPlanner()

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


__all__ = [
    "self_improvement_cycle",
    "PLANNER_INTERVAL",
    "MUTATION_RATE",
    "ROI_WEIGHT",
    "DOMAIN_PENALTY",
    "ENTROPY_THRESHOLD",
]
