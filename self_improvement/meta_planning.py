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
import queue
from pathlib import Path
from contextlib import contextmanager, nullcontext
from datetime import datetime

from ..logging_utils import get_logger, log_record
from ..sandbox_settings import SandboxSettings
from . import init as _init
from ..workflow_stability_db import WorkflowStabilityDB
from ..roi_results_db import ROIResultsDB
from ..lock_utils import SandboxLock, Timeout, LOCK_TIMEOUT
from .baseline_tracker import BaselineTracker, TRACKER as BASELINE_TRACKER
from ..error_logger import TelemetryEvent


_cycle_thread: Any | None = None
_stop_event: threading.Event | None = None

try:  # pragma: no cover - optional dependency
    from ..unified_event_bus import UnifiedEventBus
except ImportError as exc:  # pragma: no cover - fallback when event bus missing
    get_logger(__name__).warning(
        "unified event bus unavailable",  # noqa: TRY300
        extra=log_record(component=__name__, dependency="unified_event_bus"),
        exc_info=exc,
    )
    UnifiedEventBus = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from ..meta_workflow_planner import MetaWorkflowPlanner
except ImportError as exc:  # pragma: no cover - gracefully degrade
    get_logger(__name__).warning(
        "meta_workflow_planner import failed",  # noqa: TRY300
        extra=log_record(component=__name__, dependency="meta_workflow_planner"),
        exc_info=exc,
    )
    try:
        from meta_workflow_planner import MetaWorkflowPlanner  # type: ignore
    except ImportError as exc2:  # pragma: no cover - best effort fallback
        get_logger(__name__).warning(
            "local meta_workflow_planner import failed",  # noqa: TRY300
            extra=log_record(component=__name__, dependency="meta_workflow_planner"),
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
        cfg = _init.settings
        roi_window = int(getattr(cfg, "roi_window", getattr(cfg, "roi_cycles", 5) or 5))
        try:
            self.roi_db: ROIResultsDB | None = ROIResultsDB(window=roi_window)
        except (OSError, RuntimeError) as exc:  # pragma: no cover - best effort
            get_logger(__name__).warning(
                "ROIResultsDB unavailable",
                extra=log_record(component=__name__),
                exc_info=exc,
            )
            self.roi_db = None
        try:
            self.stability_db: WorkflowStabilityDB | None = WorkflowStabilityDB()
        except (OSError, RuntimeError) as exc:  # pragma: no cover - best effort
            get_logger(__name__).warning(
                "WorkflowStabilityDB unavailable",
                extra=log_record(component=__name__),
                exc_info=exc,
            )
            self.stability_db = None

        self.logger = get_logger("FallbackPlanner")
        data_dir = Path(SandboxSettings().sandbox_data_dir)
        self.state_path = data_dir / "fallback_planner.json"
        self.state_lock = SandboxLock(
            str(self.state_path.with_suffix(self.state_path.suffix + ".lock"))
        )
        self.cluster_map: dict[tuple[str, ...], dict[str, Any]] = {}
        self.state_capacity = getattr(cfg, "meta_state_capacity", 1000)
        self.mutation_rate = getattr(cfg, "mutation_rate", getattr(cfg, "meta_mutation_rate", 1.0))
        self.roi_weight = getattr(cfg, "roi_weight", getattr(cfg, "meta_roi_weight", 1.0))
        self.domain_transition_penalty = getattr(
            cfg, "domain_transition_penalty", getattr(cfg, "meta_domain_penalty", 1.0)
        )
        self.entropy_weight = getattr(cfg, "meta_entropy_weight", 0.0)
        self.stability_weight = getattr(
            cfg, "stability_weight", getattr(cfg, "meta_stability_weight", 1.0)
        )
        self.entropy_threshold = 0.0
        self.state_prune_strategy = getattr(
            cfg, "state_prune_strategy", getattr(cfg, "meta_state_prune_strategy", "recent")
        )
        self.roi_window = roi_window
        self._load_state()

    # ------------------------------------------------------------------
    def _load_state(self, *, lock: bool = True) -> None:
        """Load planner state from disk with optional locking."""

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            ctx = (
                self.state_lock.acquire(timeout=LOCK_TIMEOUT)
                if lock
                else nullcontext()
            )
            with ctx:
                if self.state_path.exists():
                    data = json.loads(self.state_path.read_text())
                else:
                    data = {}
            clusters = data.get("clusters", data)
            self.cluster_map = {tuple(k.split("|")): v for k, v in clusters.items()}
            baseline_state = data.get("baseline", {})
            if isinstance(baseline_state, Mapping):
                BASELINE_TRACKER.load_state(baseline_state)
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

    def _save_state(self, *, lock: bool = True) -> None:
        """Persist planner state to disk atomically.

        A file lock guards against concurrent writers.  Data is first written to
        a temporary file and then moved into place to avoid partial saves.
        """

        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "clusters": {"|".join(k): v for k, v in self.cluster_map.items()},
                "baseline": BASELINE_TRACKER.to_state(),
            }
            tmp_path = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
            ctx = (
                self.state_lock.acquire(timeout=LOCK_TIMEOUT)
                if lock
                else nullcontext()
            )
            with ctx:
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
    @contextmanager
    def _state_update(self) -> Any:
        """Context manager providing atomic load/modify/save of state."""

        try:
            with self.state_lock.acquire(timeout=LOCK_TIMEOUT):
                self._load_state(lock=False)
                yield
                self._save_state(lock=False)
        except Timeout as exc:  # pragma: no cover - lock contention
            self.logger.warning(
                "state lock acquisition timed out",
                extra=log_record(path=str(self.state_path)),
                exc_info=exc,
            )
            self.cluster_map = {}
            yield

    # ------------------------------------------------------------------
    def _prune_state(self) -> None:
        """Trim ``cluster_map`` according to pruning strategy."""

        if self.state_capacity <= 0:
            return
        if len(self.cluster_map) <= self.state_capacity:
            return
        if self.state_prune_strategy == "score":
            items = sorted(
                self.cluster_map.items(),
                key=lambda kv: float(kv[1].get("score", 0.0)),
                reverse=True,
            )
        else:  # default "recent"
            items = sorted(
                self.cluster_map.items(),
                key=lambda kv: float(kv[1].get("ts", 0.0)),
                reverse=True,
            )
        self.cluster_map = dict(items[: self.state_capacity])

    # ------------------------------------------------------------------
    def begin_run(self, workflow_id: str, run_id: str) -> None:
        """Record the start of a workflow run.

        The fallback planner lacks advanced tracking, but we still persist a
        minimal record so ROI and stability metrics capture the run context.
        """

        self.logger.info(
            "begin run %s/%s",
            workflow_id,
            run_id,
            extra=log_record(workflow_id=workflow_id, run_id=run_id),
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
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception(
                    "ROI logging failed",
                    extra=log_record(workflow_id=workflow_id, run_id=run_id),
                    exc_info=exc,
                )
        if self.stability_db is not None:
            try:
                self.stability_db.record_metrics(workflow_id, 0.0, 0.0, 0.0, roi_delta=0.0)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception(
                    "stability logging failed",
                    extra=log_record(workflow_id=workflow_id, run_id=run_id),
                    exc_info=exc,
                )

    # ------------------------------------------------------------------
    def _domain(self, wid: str) -> str:
        return wid.split(".", 1)[0]

    def _score(
        self, chain: Sequence[str], roi_delta: float, entropy_delta: float, failures: int
    ) -> float:
        transitions = sum(
            1
            for i in range(1, len(chain))
            if self._domain(chain[i]) != self._domain(chain[i - 1])
        )
        return (
            self.roi_weight * roi_delta
            - self.domain_transition_penalty * transitions
            - self.entropy_weight * abs(entropy_delta)
            - self.stability_weight * failures
        )

    def discover_and_persist(
        self, workflows: Mapping[str, Callable[[], Any]]
    ) -> list[Mapping[str, Any]]:
        """Explore pipelines via heuristic mutations and persist state."""

        results: list[Mapping[str, Any]] = []
        with self._state_update():
            evaluated: set[tuple[str, ...]] = set()

            existing = [list(k) for k in self.cluster_map.keys()]

            for chain in existing:
                rec = self._evaluate_chain(chain, allow_existing=True)
                if rec:
                    key = tuple(rec["chain"])
                    evaluated.add(key)
                    results.append(rec)

            for wid in workflows:
                key = (wid,)
                if key in evaluated:
                    continue
                rec = self._evaluate_chain([wid])
                if rec:
                    evaluated.add(key)
                    results.append(rec)
                    existing.append([wid])

            for chain in existing:
                for rec in self.mutate_pipeline(chain, workflows):
                    key = tuple(rec["chain"])
                    if key not in evaluated:
                        evaluated.add(key)
                        results.append(rec)

                for rec in self.split_pipeline(chain, workflows):
                    key = tuple(rec["chain"])
                    if key not in evaluated:
                        evaluated.add(key)
                        results.append(rec)

            if len(existing) >= 2:
                for rec in self.remerge_pipelines(existing, workflows):
                    key = tuple(rec["chain"])
                    if key not in evaluated:
                        evaluated.add(key)
                        results.append(rec)

            self._prune_state()

        return sorted(results, key=lambda r: r["score"], reverse=True)

    # ------------------------------------------------------------------
    def _evaluate_chain(
        self, chain: Sequence[str], *, allow_existing: bool = False
    ) -> dict[str, Any] | None:
        if len(chain) != len(set(chain)):
            self.logger.debug(
                "rejecting chain %s due to cycle", "->".join(chain),
                extra=log_record(workflow_id="->".join(chain)),
            )
            return None
        if not allow_existing and tuple(chain) in self.cluster_map:
            return None

        roi_values: list[float] = []
        delta_rois: list[float] = []
        entropies: list[float] = []
        failures = 0

        for wid in chain:
            current_roi = 0.0
            moving_avg = 0.0
            delta_roi = 0.0
            if self.roi_db is not None:
                try:
                    stats = self.roi_db.fetch_chain_stats(wid)
                    current_roi = float(stats.get("last_roi", 0.0))
                    moving_avg = float(stats.get("moving_avg_roi", 0.0))
                    delta_roi = current_roi - moving_avg
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.warning(
                        "roi fetch failed", extra=log_record(workflow_id=wid), exc_info=exc
                    )
                    current_roi = moving_avg = delta_roi = 0.0

            stable = True
            entropy = 0.0
            if self.stability_db is not None:
                try:
                    entry = self.stability_db.data.get(wid, {})
                    failures += int(entry.get("failures", 0))
                    entropy = float(entry.get("entropy", 0.0))
                    stable = self.stability_db.is_stable(
                        wid, current_roi=current_roi, threshold=moving_avg
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.warning(
                        "stability check failed", extra=log_record(workflow_id=wid), exc_info=exc
                    )
                    stable = True

            roi_base = BASELINE_TRACKER.get("roi_delta")
            roi_tol = getattr(getattr(_init.settings, "roi", None), "deviation_tolerance", 0.0)
            if delta_roi < roi_base - roi_tol or not stable:
                self.logger.debug(
                    "rejecting chain %s", "->".join(chain), extra=log_record(workflow_id=wid)
                )
                return None

            roi_values.append(current_roi)
            delta_rois.append(delta_roi)
            entropies.append(entropy)

        if not roi_values:
            return None

        chain_roi = fmean(roi_values)
        chain_roi_delta = fmean(delta_rois) if delta_rois else chain_roi
        chain_entropy = fmean(entropies) if entropies else 0.0
        entropy_delta = chain_entropy - BASELINE_TRACKER.get("entropy")
        score = self._score(chain, chain_roi_delta, entropy_delta, failures)
        record = {
            "chain": list(chain),
            "roi_gain": chain_roi,
            "roi_delta": chain_roi_delta,
            "failures": failures,
            "entropy": chain_entropy,
            "entropy_delta": entropy_delta,
            "score": score,
        }
        self.cluster_map[tuple(chain)] = {
            "last_roi": chain_roi,
            "last_entropy": chain_entropy,
            "score": score,
            "failures": failures,
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
                    code_entropy=chain_entropy,
                    entropy_delta=entropy_delta,
                    module_deltas={},
                )
            except Exception as exc:  # pragma: no cover - logging best effort
                self.logger.exception(
                    "ROI logging failed",
                    extra=log_record(workflow_id="->".join(chain)),
                    exc_info=exc,
                )
        if self.stability_db is not None:
            try:
                self.stability_db.record_metrics(
                    "->".join(chain), chain_roi, failures, chain_entropy, roi_delta=chain_roi
                )
            except Exception as exc:  # pragma: no cover - logging best effort
                self.logger.exception(
                    "stability logging failed",
                    extra=log_record(workflow_id="->".join(chain)),
                    exc_info=exc,
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


settings = _init.settings


def reload_settings(cfg: SandboxSettings) -> None:
    """Update module-level settings and derived constants."""
    global settings, PLANNER_INTERVAL, MUTATION_RATE, ROI_WEIGHT, DOMAIN_PENALTY, ENTROPY_THRESHOLD
    global SEARCH_DEPTH, BEAM_WIDTH, ENTROPY_WEIGHT
    settings = cfg
    _validate_config(settings)
    PLANNER_INTERVAL = getattr(settings, "meta_planning_interval", 0)
    MUTATION_RATE = settings.meta_mutation_rate
    ROI_WEIGHT = settings.meta_roi_weight
    DOMAIN_PENALTY = settings.meta_domain_penalty
    ENTROPY_WEIGHT = settings.meta_entropy_weight
    SEARCH_DEPTH = settings.meta_search_depth
    BEAM_WIDTH = settings.meta_beam_width
    if settings.meta_entropy_threshold is not None:
        ENTROPY_THRESHOLD = float(settings.meta_entropy_threshold)
    else:
        dev = getattr(settings, "entropy_deviation", 1.0)
        base = BASELINE_TRACKER.get("entropy")
        std = BASELINE_TRACKER.std("entropy")
        ENTROPY_THRESHOLD = base + dev * std


def _validate_config(cfg: SandboxSettings) -> None:
    """Validate meta planning configuration on import."""
    if cfg.meta_entropy_threshold is not None and not 0 <= cfg.meta_entropy_threshold <= 1:
        raise ValueError("meta_entropy_threshold must be between 0 and 1")
    for attr in (
        "meta_mutation_rate",
        "meta_roi_weight",
        "meta_domain_penalty",
        "meta_entropy_weight",
    ):
        if getattr(cfg, attr) < 0:
            raise ValueError(f"{attr} must be non-negative")
    for attr in ("meta_search_depth", "meta_beam_width"):
        if getattr(cfg, attr) <= 0:
            raise ValueError(f"{attr} must be positive")


reload_settings(settings)
_stable_workflows: WorkflowStabilityDB | None = None


def get_stable_workflows() -> WorkflowStabilityDB:
    """Return the cached :class:`WorkflowStabilityDB` instance.

    The database connection is created lazily on first access and wrapped in a
    ``try``/``except`` block so callers receive a descriptive ``RuntimeError``
    if initialisation fails.
    """

    global _stable_workflows
    if _stable_workflows is None:
        try:
            _stable_workflows = WorkflowStabilityDB()
        except Exception as exc:  # pragma: no cover - best effort
            raise RuntimeError("WorkflowStabilityDB initialisation failed") from exc
    return _stable_workflows


def _get_entropy_threshold(cfg: SandboxSettings, tracker: BaselineTracker) -> float:
    """Determine entropy threshold from baseline statistics."""
    threshold = cfg.meta_entropy_threshold
    if threshold is not None:
        return float(threshold)

    base = tracker.get("entropy")
    std = tracker.std("entropy")
    dev = getattr(cfg, "entropy_deviation", 1.0)
    return base + dev * std


def _should_encode(
    record: Mapping[str, Any],
    tracker: BaselineTracker,
    *,
    entropy_threshold: float,
) -> bool:
    """Return True if ``record`` indicates improvement and stability."""
    entropy = float(record.get("entropy", 0.0))
    delta_entropy = abs(entropy - float(entropy_threshold))

    roi_gain = float(record.get("roi_gain", 0.0))
    roi_delta = (roi_gain - tracker.get("roi")) * (getattr(tracker, "momentum", 1.0) or 1.0)

    return roi_delta > 0.0 and delta_entropy <= float(entropy_threshold)


def evaluate_cycle(
    record: Mapping[str, Any],
    tracker: BaselineTracker,
    errors: Sequence[TelemetryEvent] | Sequence[Any],
) -> tuple[bool, str]:
    """Evaluate cycle metrics and errors.

    Parameters
    ----------
    record:
        Metrics record for the current cycle. Only used for timestamp context.
    tracker:
        Baseline metric tracker containing historical values.
    errors:
        Sequence of :class:`TelemetryEvent` instances observed during the cycle.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` when any metric delta is non-positive or a recent
        critical error is present. Otherwise ``(False, "all_deltas_positive")``.
    """

    # Compute current deltas for all tracked metrics
    for metric in list(tracker._history):
        if tracker.delta(metric) <= 0:
            return True, ""

    def _to_ts(val: Any) -> float:
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val).timestamp()
            except Exception:
                try:
                    return float(val)
                except Exception:
                    return 0.0
        return 0.0

    cycle_ts = _to_ts(record.get("timestamp"))

    for err in errors:
        sev = getattr(getattr(err, "error_type", None), "severity", None)
        if sev == "critical":
            if _to_ts(getattr(err, "timestamp", 0)) >= cycle_ts:
                return True, ""

    return False, "all_deltas_positive"


async def self_improvement_cycle(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    interval: float = PLANNER_INTERVAL,
    event_bus: UnifiedEventBus | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    """Background loop evolving ``workflows`` using the meta planner."""
    logger = get_logger("SelfImprovementCycle")
    cfg = _init.settings
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

    stability_db = get_stable_workflows()
    setattr(planner, "entropy_threshold", _get_entropy_threshold(cfg, BASELINE_TRACKER))

    async def _log(record: Mapping[str, Any]) -> None:
        chain = record.get("chain", [])
        cid = "->".join(chain)
        roi = float(record.get("roi_gain", 0.0))
        failures = int(record.get("failures", 0))
        entropy = float(record.get("entropy", 0.0))
        entropy_delta = float(
            record.get("entropy_delta", entropy - BASELINE_TRACKER.get("entropy"))
        )
        try:
            from .metrics import record_entropy as _record_entropy
            _record_entropy(
                float(record.get("code_diversity", entropy)),
                float(record.get("token_complexity", entropy)),
                roi=roi,
            )
        except Exception:
            pass
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
                    code_entropy=entropy,
                    entropy_delta=entropy_delta,
                    module_deltas={},
                )
            except Exception as exc:  # pragma: no cover - logging best effort
                logger.exception(
                    "ROI logging failed", extra=log_record(workflow_id=cid), exc_info=exc
                )
        if planner.stability_db is not None:
            try:
                planner.stability_db.record_metrics(
                    cid, roi, failures, entropy, roi_delta=roi
                )
            except Exception as exc:  # pragma: no cover - logging best effort
                logger.exception(
                    "stability logging failed", extra=log_record(workflow_id=cid), exc_info=exc
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
            except Exception as exc:  # pragma: no cover - best effort
                logger.exception(
                    "failed to publish metrics",
                    extra=log_record(workflow_id=cid),
                    exc_info=exc,
                )

    while True:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            records = planner.discover_and_persist(workflows)
            active: list[list[str]] = []
            for rec in records:
                await _log(rec)
                roi = float(rec.get("roi_gain", 0.0))
                failures = int(rec.get("failures", 0))
                entropy = float(rec.get("entropy", 0.0))
                pass_rate = float(rec.get("pass_rate", 1.0 if failures == 0 else 0.0))
                BASELINE_TRACKER.update(roi=roi, pass_rate=pass_rate, entropy=entropy)
                tracker = BASELINE_TRACKER
                should_skip, reason = evaluate_cycle(
                    rec,
                    tracker,
                    rec.get("errors", []),
                )
                entropy_shift = abs(tracker.delta("entropy"))
                has_critical = any(
                    getattr(getattr(err, "error_type", None), "severity", None)
                    == "critical"
                    for err in rec.get("errors", [])
                )
                if should_skip and reason == "all_deltas_positive" and (
                    entropy_shift > cfg.overfitting_entropy_threshold or has_critical
                ):
                    logger.debug(
                        "run",
                        extra=log_record(
                            reason="overfitting_fallback",
                            metrics=tracker.to_dict(),
                        ),
                    )
                elif should_skip:
                    logger.debug(
                        "skip",
                        extra=log_record(reason=reason, metrics=tracker.to_dict()),
                    )
                    continue
                else:
                    logger.debug(
                        "run",
                        extra=log_record(metrics=tracker.to_dict()),
                    )
                chain = rec.get("chain", [])
                if chain and roi > 0:
                    active.append(chain)
                    chain_id = "->".join(chain)
                    try:
                        stability_db.record_metrics(
                            chain_id, roi, failures, entropy, roi_delta=roi
                        )
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.exception(
                            "global stability logging failed",
                            extra=log_record(workflow_id=chain_id),
                            exc_info=exc,
                        )
        except Exception as exc:  # pragma: no cover - planner is best effort
            logger.debug("error", extra=log_record(err=str(exc)))
            logger.exception("meta planner execution failed", exc_info=exc)

        for chain in list(active):
            planner.cluster_map.pop(tuple(chain), None)

        await asyncio.sleep(interval)


def start_self_improvement_cycle(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    event_bus: UnifiedEventBus | None = None,
    interval: float = PLANNER_INTERVAL,
):
    """Prepare a background thread running :func:`self_improvement_cycle`.

    The returned object exposes ``start()``, ``join()`` and ``stop()`` methods.
    It captures exceptions raised inside the background task and re-raises them
    when ``join()`` is invoked.  ``stop()`` gracefully cancels the running
    coroutine.
    """

    stop_event = threading.Event()

    class _CycleThread:
        def __init__(self) -> None:
            self._loop = asyncio.new_event_loop()
            self._task: asyncio.Task[None] | None = None
            self._exc: queue.Queue[BaseException] = queue.Queue()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._stop_event = stop_event

        # --------------------------------------------------
        def _run(self) -> None:
            from inspect import signature

            asyncio.set_event_loop(self._loop)
            kwargs: dict[str, Any] = {
                "interval": interval,
                "event_bus": event_bus,
            }
            if "stop_event" in signature(self_improvement_cycle).parameters:
                kwargs["stop_event"] = self._stop_event
            self._task = self._loop.create_task(
                self_improvement_cycle(workflows, **kwargs)
            )
            self._task.add_done_callback(self._finished)
            try:
                self._loop.run_forever()
            finally:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.close()

        def _finished(self, task: asyncio.Task[None]) -> None:
            if task.cancelled():
                msg = getattr(task, "_cancel_message", None)
                reason = str(msg) if msg else "cancelled"
                logger = get_logger(__name__)
                logger.info(
                    "self improvement cycle cancelled",
                    extra=log_record(reason=reason),
                )
                try:  # pragma: no cover - metrics are best effort
                    from ..metrics_exporter import self_improvement_failure_total

                    self_improvement_failure_total.labels(reason=reason).inc()
                except Exception as exc:
                    logger.debug(
                        "cancellation metric update failed",
                        extra=log_record(reason=reason),
                        exc_info=exc,
                    )
            else:
                try:
                    task.result()
                except BaseException as exc:  # pragma: no cover - best effort
                    self._exc.put(exc)
            self._loop.call_soon_threadsafe(self._loop.stop)

        # --------------------------------------------------
        def start(self) -> None:
            self._thread.start()

        def join(self, timeout: float | None = None) -> None:
            self._thread.join(timeout)
            if not self._exc.empty():
                raise self._exc.get()

        def stop(self) -> None:
            self._stop_event.set()
            self._loop.call_soon_threadsafe(
                lambda: self._task.cancel() if self._task is not None else None
            )
            self.join()

    _ = _init.settings
    logger_fn = globals().get("get_logger")
    log_record_fn = globals().get("log_record")
    logger = logger_fn(__name__) if logger_fn else None
    try:
        ROIResultsDB()
    except (OSError, RuntimeError) as exc:
        if logger is not None:
            logger.error(
                "ROIResultsDB initialisation failed",
                extra=(log_record_fn(module=__name__) if log_record_fn else None),
                exc_info=exc,
            )
        raise RuntimeError("ROIResultsDB initialisation failed") from exc
    try:
        WorkflowStabilityDB()
    except (OSError, RuntimeError) as exc:
        if logger is not None:
            logger.error(
                "WorkflowStabilityDB initialisation failed",
                extra=(log_record_fn(module=__name__) if log_record_fn else None),
                exc_info=exc,
            )
        raise RuntimeError("WorkflowStabilityDB initialisation failed") from exc

    thread = _CycleThread()
    global _cycle_thread, _stop_event
    _cycle_thread = thread
    _stop_event = stop_event
    return thread


def stop_self_improvement_cycle() -> None:
    """Signal the background self improvement cycle to stop and wait for it."""
    global _cycle_thread, _stop_event
    if _cycle_thread is None:
        return
    if _stop_event is not None:
        _stop_event.set()
    _cycle_thread.stop()
    _cycle_thread = None
    _stop_event = None


__all__ = [
    "self_improvement_cycle",
    "start_self_improvement_cycle",
    "stop_self_improvement_cycle",
    "reload_settings",
    "PLANNER_INTERVAL",
    "MUTATION_RATE",
    "ROI_WEIGHT",
    "DOMAIN_PENALTY",
    "ENTROPY_THRESHOLD",
    "SEARCH_DEPTH",
    "BEAM_WIDTH",
    "ENTROPY_WEIGHT",
]
