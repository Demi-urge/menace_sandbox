"""Meta-planning routines for the self-improvement package.

The :func:`self_improvement_cycle` coroutine drives background workflow
optimization using the optional :class:`MetaWorkflowPlanner` component.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from statistics import fmean
import asyncio
import json
import os
import time
import threading
from pathlib import Path

from ..logging_utils import get_logger, log_record
from ..sandbox_settings import SandboxSettings, load_sandbox_settings
from ..workflow_stability_db import WorkflowStabilityDB
from ..roi_results_db import ROIResultsDB
from ..lock_utils import SandboxLock, Timeout, LOCK_TIMEOUT

try:  # pragma: no cover - optional dependency
    from ..unified_event_bus import UnifiedEventBus
except ImportError as exc:  # pragma: no cover - fallback when event bus missing
    get_logger(__name__).warning(
        "unified event bus unavailable",  # noqa: TRY300
        extra=log_record(module=__name__, dependency="unified_event_bus"),
        exc_info=exc,
    )
    UnifiedEventBus = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from ..meta_workflow_planner import MetaWorkflowPlanner
except ImportError as exc:  # pragma: no cover - gracefully degrade
    get_logger(__name__).warning(
        "meta_workflow_planner import failed",  # noqa: TRY300
        extra=log_record(module=__name__, dependency="meta_workflow_planner"),
        exc_info=exc,
    )
    try:
        from meta_workflow_planner import MetaWorkflowPlanner  # type: ignore
    except ImportError as exc2:  # pragma: no cover - best effort fallback
        get_logger(__name__).warning(
            "local meta_workflow_planner import failed",  # noqa: TRY300
            extra=log_record(module=__name__, dependency="meta_workflow_planner"),
            exc_info=exc2,
        )
        MetaWorkflowPlanner = None  # type: ignore


class _FallbackPlanner:
    """Lightweight planner with basic mutation and persistence.

    Concurrency:
        Access to the persistent state file is guarded by a
        :class:`SandboxLock`, ensuring that multiple planner instances do not
        corrupt state.  Writes are performed atomically by first writing to a
        temporary file and then replacing the target.
    """

    def __init__(self) -> None:
        cfg = load_sandbox_settings()
        roi_window = int(getattr(cfg, "roi_window", getattr(cfg, "roi_cycles", 5) or 5))
        try:
            self.roi_db: ROIResultsDB | None = ROIResultsDB(window=roi_window)
        except (OSError, RuntimeError) as exc:  # pragma: no cover - best effort
            get_logger(__name__).warning(
                "ROIResultsDB unavailable",
                extra=log_record(module=__name__),
                exc_info=exc,
            )
            self.roi_db = None
        try:
            self.stability_db: WorkflowStabilityDB | None = WorkflowStabilityDB()
        except (OSError, RuntimeError) as exc:  # pragma: no cover - best effort
            get_logger(__name__).warning(
                "WorkflowStabilityDB unavailable",
                extra=log_record(module=__name__),
                exc_info=exc,
            )
            self.stability_db = None

        self.logger = get_logger("FallbackPlanner")
        self.state_path = Path("sandbox_data/fallback_planner.json")
        self.state_lock = SandboxLock(str(self.state_path.with_suffix(self.state_path.suffix + ".lock")))
        self.cluster_map: dict[tuple[str, ...], dict[str, Any]] = {}
        self.mutation_rate = getattr(cfg, "mutation_rate", getattr(cfg, "meta_mutation_rate", 1.0))
        self.roi_weight = getattr(cfg, "roi_weight", getattr(cfg, "meta_roi_weight", 1.0))
        self.domain_transition_penalty = getattr(
            cfg, "domain_transition_penalty", getattr(cfg, "meta_domain_penalty", 1.0)
        )
        self.roi_window = roi_window
        self._load_state()

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        """Load planner state from disk with inter-process locking."""

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with self.state_lock.acquire(timeout=LOCK_TIMEOUT):
                if self.state_path.exists():
                    data = json.loads(self.state_path.read_text())
                else:
                    data = {}
            self.cluster_map = {tuple(k.split("|")): v for k, v in data.items()}
        except Timeout as exc:  # pragma: no cover - lock contention
            self.logger.warning(
                "state lock acquisition timed out",
                extra=log_record(path=str(self.state_path)),
                exc_info=exc,
            )
            self.cluster_map = {}
        except (OSError, json.JSONDecodeError) as exc:
            get_logger(__name__).debug(
                "failed to load fallback planner state",
                extra=log_record(path=str(self.state_path)),
                exc_info=exc,
            )
            self.cluster_map = {}

    def _save_state(self) -> None:
        """Persist planner state to disk atomically.

        A file lock guards against concurrent writers.  Data is first written to
        a temporary file and then moved into place to avoid partial saves.
        """

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"|".join(k): v for k, v in self.cluster_map.items()}
            tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            with self.state_lock.acquire(timeout=LOCK_TIMEOUT):
                tmp_path.write_text(json.dumps(data, indent=2))
                os.replace(tmp_path, self.state_path)
        except Timeout as exc:  # pragma: no cover - lock contention
            self.logger.warning(
                "state lock acquisition timed out",
                extra=log_record(path=str(self.state_path)),
                exc_info=exc,
            )
        except OSError as exc:  # pragma: no cover - best effort
            self.logger.debug(
                "failed to persist fallback planner state",
                extra=log_record(path=str(self.state_path)),
                exc_info=exc,
            )

    # ------------------------------------------------------------------
    def begin_run(self, workflow_id: str, run_id: str) -> None:
        """Record the start of a workflow run.

        The fallback planner lacks advanced tracking, but we still persist a
        minimal record so ROI and stability metrics capture the run context.
        """

        self.logger.info(
            "begin run %s/%s", workflow_id, run_id, extra=log_record(workflow_id=workflow_id, run_id=run_id)
        )
        if self.roi_db is not None:
            try:
                self.roi_db.log_result(
                    workflow_id=workflow_id,
                    run_id=run_id,
                    runtime=0.0,
                    success_rate=0.0,
                    roi_gain=0.0,
                    workflow_synergy_score=0.0,
                    bottleneck_index=0.0,
                    patchability_score=0.0,
                    module_deltas={},
                )
            except Exception:  # pragma: no cover - best effort
                self.logger.exception(
                    "ROI logging failed", extra=log_record(workflow_id=workflow_id, run_id=run_id)
                )
        if self.stability_db is not None:
            try:
                self.stability_db.record_metrics(workflow_id, 0.0, 0.0, 0.0, roi_delta=0.0)
            except Exception:  # pragma: no cover - best effort
                self.logger.exception(
                    "stability logging failed", extra=log_record(workflow_id=workflow_id, run_id=run_id)
                )

    # ------------------------------------------------------------------
    def _domain(self, wid: str) -> str:
        return wid.split(".", 1)[0]

    def _score(self, chain: Sequence[str], roi: float) -> float:
        transitions = sum(
            1
            for i in range(1, len(chain))
            if self._domain(chain[i]) != self._domain(chain[i - 1])
        )
        return self.roi_weight * roi - self.domain_transition_penalty * transitions

    # ------------------------------------------------------------------
    def discover_and_persist(
        self, workflows: Mapping[str, Callable[[], Any]]
    ) -> list[Mapping[str, Any]]:
        """Explore pipelines via mutation, splitting and merging."""

        evaluated: set[tuple[str, ...]] = set()
        records: list[dict[str, Any]] = []

        for chain_key in list(self.cluster_map):
            rec = self._evaluate_chain(list(chain_key))
            if rec:
                records.append(rec)
                evaluated.add(tuple(rec["chain"]))

        for wid in workflows:
            chain = (wid,)
            if chain in evaluated:
                continue
            rec = self._evaluate_chain(list(chain))
            if rec:
                records.append(rec)
                evaluated.add(chain)

        candidates = list(records)
        for rec in list(records):
            candidates.extend(self.mutate_pipeline(rec["chain"], workflows))
            if len(rec["chain"]) > 1:
                candidates.extend(self.split_pipeline(rec["chain"], workflows))

        pipelines = [c["chain"] for c in candidates][:5]
        candidates.extend(self.remerge_pipelines(pipelines, workflows))

        dedup: dict[tuple[str, ...], dict[str, Any]] = {}
        for rec in candidates:
            key = tuple(rec["chain"])
            if key not in dedup or rec["score"] > dedup[key]["score"]:
                dedup[key] = rec

        results = sorted(dedup.values(), key=lambda r: r["score"], reverse=True)
        self._save_state()
        return results

    # ------------------------------------------------------------------
    def _evaluate_chain(self, chain: Sequence[str]) -> dict[str, Any] | None:
        roi_values: list[float] = []
        entropies: list[float] = []
        failures = 0

        for wid in chain:
            roi = 0.0
            if self.roi_db is not None:
                try:
                    results = self.roi_db.fetch_results(wid)
                    recent = [r.roi_gain for r in results[-self.roi_window :]]
                    roi = fmean(recent) if recent else 0.0
                except Exception:  # pragma: no cover - best effort
                    self.logger.warning(
                        "roi fetch failed", extra=log_record(workflow_id=wid)
                    )
                    roi = 0.0

            stable = True
            entropy = 0.0
            if self.stability_db is not None:
                try:
                    entry = self.stability_db.data.get(wid, {})
                    failures += int(entry.get("failures", 0))
                    entropy = float(entry.get("entropy", 0.0))
                    stable = self.stability_db.is_stable(
                        wid, current_roi=roi, threshold=1.0
                    )
                except Exception:  # pragma: no cover - best effort
                    self.logger.warning(
                        "stability check failed", extra=log_record(workflow_id=wid)
                    )
                    stable = True

            if roi <= 0.0 or not stable:
                self.logger.debug(
                    "rejecting chain %s", "->".join(chain), extra=log_record(workflow_id=wid)
                )
                return None

            roi_values.append(roi)
            entropies.append(entropy)

        if not roi_values:
            return None

        chain_roi = fmean(roi_values)
        chain_entropy = fmean(entropies) if entropies else 0.0
        score = self._score(chain, chain_roi)
        record = {
            "chain": list(chain),
            "roi_gain": chain_roi,
            "failures": failures,
            "entropy": chain_entropy,
            "score": score,
        }
        self.cluster_map[tuple(chain)] = {
            "last_roi": chain_roi,
            "last_entropy": chain_entropy,
            "score": score,
            "ts": time.time(),
        }
        self.logger.debug(
            "evaluated chain %s score %.3f",
            "->".join(chain),
            score,
            extra=log_record(workflow_id="->".join(chain)),
        )

        if self.roi_db is not None:
            try:
                self.roi_db.log_result(
                    workflow_id="->".join(chain),
                    run_id="evaluation",
                    runtime=0.0,
                    success_rate=1.0,
                    roi_gain=chain_roi,
                    workflow_synergy_score=max(0.0, 1.0 - chain_entropy),
                    bottleneck_index=0.0,
                    patchability_score=0.0,
                    module_deltas={},
                )
            except Exception:  # pragma: no cover - logging best effort
                self.logger.exception(
                    "ROI logging failed",
                    extra=log_record(workflow_id="->".join(chain)),
                )
        if self.stability_db is not None:
            try:
                self.stability_db.record_metrics(
                    "->".join(chain), chain_roi, failures, chain_entropy, roi_delta=chain_roi
                )
            except Exception:  # pragma: no cover - logging best effort
                self.logger.exception(
                    "stability logging failed",
                    extra=log_record(workflow_id="->".join(chain)),
                )

        return record

    # ------------------------------------------------------------------
    def _generate_mutations(
        self, chain: list[str], pool: list[str], depth: int
    ) -> list[list[str]]:
        if depth == 0:
            return []
        mutations: list[list[str]] = []
        for wid in pool:
            if wid in chain:
                continue
            mutated = chain + [wid]
            mutations.append(mutated)
            next_pool = [w for w in pool if w not in mutated]
            mutations.extend(self._generate_mutations(mutated, next_pool, depth - 1))
        return mutations

    def mutate_pipeline(
        self,
        chain: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        **_: Any,
    ) -> list[Mapping[str, Any]]:
        """Create mutations by appending up to ``mutation_rate`` steps."""

        depth = max(1, int(self.mutation_rate))
        pool = [wid for wid in workflows if wid not in chain]
        candidate_chains = self._generate_mutations(list(chain), pool, depth)
        results: list[Mapping[str, Any]] = []
        for cand in candidate_chains:
            rec = self._evaluate_chain(cand)
            if rec:
                results.append(rec)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    def split_pipeline(
        self,
        chain: Sequence[str],
        workflows: Mapping[str, Callable[[], Any]],
        **_: Any,
    ) -> list[Mapping[str, Any]]:
        """Split ``chain`` into two halves and score each half."""

        if len(chain) <= 1:
            return []

        mid = len(chain) // 2
        segments = [chain[:mid], chain[mid:]]
        results: list[Mapping[str, Any]] = []
        for seg in segments:
            rec = self._evaluate_chain(seg)
            if rec:
                results.append(rec)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    def remerge_pipelines(
        self,
        pipelines: Sequence[Sequence[str]],
        workflows: Mapping[str, Callable[[], Any]],
        **_: Any,
    ) -> list[Mapping[str, Any]]:
        """Combine pipelines pairwise and score merged candidates."""

        results: list[Mapping[str, Any]] = []
        for i in range(len(pipelines)):
            for j in range(i + 1, len(pipelines)):
                merged = list(pipelines[i]) + [
                    w for w in pipelines[j] if w not in pipelines[i]
                ]
                rec = self._evaluate_chain(merged)
                if rec:
                    results.append(rec)

        results.sort(key=lambda r: r["score"], reverse=True)
        return results


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




def start_self_improvement_cycle(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    event_bus: UnifiedEventBus | None = None,
    interval: float = PLANNER_INTERVAL,
) -> threading.Thread:
    """Launch `self_improvement_cycle` in a background thread.

    Sandbox settings are loaded to validate configuration and ensure required
    databases are initialised. The returned thread runs indefinitely as a
    daemon.
    """
    load_sandbox_settings()
    try:
        ROIResultsDB()
        WorkflowStabilityDB()
    except Exception:
        pass

    def _runner() -> None:
        asyncio.run(self_improvement_cycle(workflows, interval=interval, event_bus=event_bus))

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    return thread


__all__ = [
    "self_improvement_cycle",
    "start_self_improvement_cycle",
    "PLANNER_INTERVAL",
    "MUTATION_RATE",
    "ROI_WEIGHT",
    "DOMAIN_PENALTY",
    "ENTROPY_THRESHOLD",
]
