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
import sys
import time
import threading
import queue
from pathlib import Path
from contextlib import contextmanager, nullcontext

try:  # pragma: no cover - optional dependency location
    from .. import dynamic_path_router as _dynamic_path_router
except Exception:  # pragma: no cover
    import dynamic_path_router as _dynamic_path_router  # type: ignore
resolve_path = _dynamic_path_router.resolve_path  # type: ignore
sys.modules.setdefault("menace.dynamic_path_router", _dynamic_path_router)

if "menace.logging_utils" in sys.modules and not hasattr(
    sys.modules["menace.logging_utils"], "setup_logging"
):  # pragma: no cover - test stub fallback
    sys.modules["menace.logging_utils"].setup_logging = lambda *a, **k: None

try:  # pragma: no cover - allow flat package layout
    from ..logging_utils import get_logger, log_record
except ImportError:  # pragma: no cover - fallback for manual_bootstrap environments
    from logging_utils import get_logger, log_record  # type: ignore

try:  # pragma: no cover - allow flat package layout
    from ..sandbox_settings import SandboxSettings, DEFAULT_SEVERITY_SCORE_MAP
except ImportError:  # pragma: no cover - fallback for manual_bootstrap environments
    from sandbox_settings import (  # type: ignore
        SandboxSettings,
        DEFAULT_SEVERITY_SCORE_MAP,
    )

from . import init as _init

try:  # pragma: no cover - allow flat package layout
    from ..workflow_stability_db import WorkflowStabilityDB
except ImportError:  # pragma: no cover - fallback for manual_bootstrap environments
    from workflow_stability_db import WorkflowStabilityDB  # type: ignore

try:  # pragma: no cover - allow flat package layout
    from ..roi_results_db import ROIResultsDB
except ImportError:  # pragma: no cover - fallback for manual_bootstrap environments
    from roi_results_db import ROIResultsDB  # type: ignore

try:  # pragma: no cover - allow flat package layout
    from ..lock_utils import SandboxLock, Timeout, LOCK_TIMEOUT
except ImportError:  # pragma: no cover - fallback for manual_bootstrap environments
    from lock_utils import SandboxLock, Timeout, LOCK_TIMEOUT  # type: ignore

from .baseline_tracker import BaselineTracker, TRACKER as BASELINE_TRACKER

try:  # pragma: no cover - allow flat package layout
    from ..error_logger import TelemetryEvent
except ImportError:  # pragma: no cover - fallback for manual_bootstrap environments
    from error_logger import TelemetryEvent  # type: ignore

from .sandbox_score import get_latest_sandbox_score
from context_builder_util import create_context_builder


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
except Exception as exc:  # pragma: no cover - gracefully degrade
    get_logger(__name__).warning(
        "meta_workflow_planner import failed",  # noqa: TRY300
        extra=log_record(component=__name__, dependency="meta_workflow_planner"),
        exc_info=exc,
    )
    try:
        from meta_workflow_planner import MetaWorkflowPlanner  # type: ignore
    except Exception as exc2:  # pragma: no cover - best effort fallback
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
        data_dir = Path(resolve_path(SandboxSettings().sandbox_data_dir))
        self.state_path = Path(resolve_path(data_dir / "fallback_planner.json"))
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
            self.logger.debug(
                "state capacity disabled; skipping prune",
                extra=log_record(component=__name__),
            )
            return
        if len(self.cluster_map) <= self.state_capacity:
            self.logger.debug(
                "state within capacity; skipping prune",
                extra=log_record(component=__name__),
            )
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
            self.logger.debug(
                "rejecting existing chain %s",
                "->".join(chain),
                extra=log_record(workflow_id="->".join(chain)),
            )
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
            self.logger.debug(
                "no ROI values for chain %s", "->".join(chain),
                extra=log_record(workflow_id="->".join(chain)),
            )
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


def _percentile(data: Sequence[float], pct: float) -> float:
    """Return the *pct* percentile for *data* where *pct* is 0-1."""
    if not data:
        return 0.0
    if not 0 <= pct <= 1:
        raise ValueError("percentile must be between 0 and 1")
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * pct
    f = int(k)
    c = min(f + 1, len(data_sorted) - 1)
    if f == c:
        return data_sorted[f]
    return data_sorted[f] + (data_sorted[c] - data_sorted[f]) * (k - f)


def _get_overfit_thresholds(
    cfg: SandboxSettings, tracker: BaselineTracker
) -> tuple[int, float]:
    """Return dynamic error and entropy thresholds.

    ``max_allowed_errors`` and ``entropy_overfit_threshold`` may be configured
    explicitly on *cfg*.  When they are ``None`` the thresholds are derived from
    tracker histories using a percentile-based heuristic (default 95th
    percentile).  This keeps the fallback logic responsive to recent behaviour
    without requiring static configuration.
    """

    max_errors = getattr(cfg, "max_allowed_errors", None)
    if max_errors is None:
        pct = getattr(cfg, "error_overfit_percentile", 0.95)
        errors = tracker.to_dict().get("error_count", [])
        max_errors = _percentile(errors, pct)

    entropy_thresh = getattr(cfg, "entropy_overfit_threshold", None)
    if entropy_thresh is None:
        pct = getattr(cfg, "entropy_overfit_percentile", 0.95)
        deltas = [abs(x) for x in tracker.to_dict().get("entropy_delta", [])]
        entropy_thresh = _percentile(deltas, pct)

    return int(max_errors), float(entropy_thresh)


def _should_encode(
    record: Mapping[str, Any],
    tracker: BaselineTracker,
    *,
    entropy_threshold: float,
) -> tuple[bool, str]:
    """Determine if the latest cycle warrants encoding.

    The tracker is expected to be updated with the metrics contained in
    ``record`` *before* this function is invoked.  Deltas for all tracked
    metrics are therefore derived from :class:`BaselineTracker` rather than
    recomputed from raw ``record`` values.  A tuple ``(should_encode, reason)``
    is returned where ``reason`` provides additional context when encoding is
    skipped.
    """

    # Momentum-weighted ROI delta must be positive.
    momentum = getattr(tracker, "momentum", 1.0) or 1.0
    roi_delta = tracker.delta("roi") * momentum
    if roi_delta <= 0:
        return False, "no_delta"

    # All other tracked metrics (excluding momentum and entropy) require
    # positive deltas.
    for metric in tracker._history:
        if metric.endswith("_delta") or metric in {
            "roi",
            "momentum",
            "entropy",
        }:
            continue
        if tracker.delta(metric) < 0:
            return False, "no_delta"

    # Treat entropy spikes as potential overfitting even when ROI is high.
    history = tracker._history.get("entropy", [])
    prev_entropy = history[-2] if len(history) > 1 else (history[-1] if history else 0.0)
    current_entropy = float(record.get("entropy", prev_entropy))
    if abs(current_entropy - prev_entropy) >= float(entropy_threshold):
        return False, "entropy_spike"

    # Any failures or critical errors invalidate the improvement signal.
    if int(record.get("failures", 0)) > 0:
        return False, "errors_present"
    for err in record.get("errors", []) or []:
        sev = getattr(getattr(err, "error_type", None), "severity", None)
        if sev == "critical":
            return False, "errors_present"

    return True, "improved"


def _recent_error_entropy(
    error_log: Any | None,
    tracker: BaselineTracker,
    limit: int | None = None,
) -> tuple[Sequence[Any], float, int, float, float]:
    """Return recent error traces and entropy statistics.

    Parameters
    ----------
    error_log:
        Error log or telemetry object exposing ``recent_errors`` or
        ``recent_events``.  ``None`` is treated as no errors.
    tracker:
        Baseline tracker providing access to entropy history.

    Returns
    -------
    tuple[Sequence[Any], float, int, float, float]
        ``(events, entropy_delta, error_count, delta_mean, delta_std)`` where
        ``entropy_delta`` is the change in entropy since the previous update
        while ``delta_mean`` and ``delta_std`` are the rolling average and
        standard deviation of recent entropy deltas as tracked by
        :class:`BaselineTracker`.
    """

    events: Sequence[Any] | None = None
    # Allow callers or configuration to control the error history window.
    window = int(limit or getattr(_init.settings, "error_window", 5))
    try:
        if error_log is None:
            events = []
        elif isinstance(error_log, Sequence):
            events = error_log
        elif hasattr(error_log, "recent_errors"):
            events = error_log.recent_errors(limit=window)
        elif hasattr(error_log, "recent_events"):
            events = error_log.recent_events(limit=window)
        elif hasattr(getattr(error_log, "db", None), "recent_errors"):
            events = error_log.db.recent_errors(limit=window)  # type: ignore[attr-defined]
        else:
            events = []
    except Exception:
        events = []

    entropy_delta = getattr(tracker, "entropy_delta", 0.0)
    delta_mean = getattr(tracker, "get", lambda _m: 0.0)("entropy_delta")
    delta_std = getattr(tracker, "std", lambda _m: 0.0)("entropy_delta")
    error_count = len(events or [])
    return (
        list(events or []),
        float(entropy_delta),
        error_count,
        float(delta_mean),
        float(delta_std),
    )


# Canonical metrics that must be tracked for cycle evaluation.
REQUIRED_METRICS: tuple[str, ...] = ("roi", "pass_rate", "entropy")


def _evaluate_cycle(
    tracker: BaselineTracker, error_state: Any
) -> tuple[str, Mapping[str, Any]]:
    """Evaluate whether the improvement cycle should run.

    Parameters
    ----------
    tracker:
        Baseline metric tracker providing recent deltas.
    error_state:
        Error log or sequence of :class:`TelemetryEvent` instances used to
        detect critical failures.

    Returns
    -------
    tuple[str, Mapping[str, Any]]
        A decision of ``"run"`` or ``"skip"`` and a mapping containing the
        quantitative metric deltas along with a ``reason`` string.
    """

    metrics: dict[str, float] = {}
    history = getattr(tracker, "_history", {})
    for name in history.keys():
        if name.endswith("_delta"):
            continue
        metrics[name] = float(tracker.delta(name))

    missing = [m for m in REQUIRED_METRICS if m not in metrics]

    # Examine recent errors for critical severity
    critical = False
    events: Sequence[TelemetryEvent] | Sequence[Any] | None = None
    try:
        if isinstance(error_state, Sequence):
            events = error_state
        elif hasattr(error_state, "recent_errors"):
            events = error_state.recent_errors(limit=5)
        elif hasattr(error_state, "recent_events"):
            events = error_state.recent_events(limit=5)
        elif hasattr(getattr(error_state, "db", None), "recent_errors"):
            events = error_state.db.recent_errors(limit=5)  # type: ignore[attr-defined]
    except Exception:
        events = None

    threshold = getattr(_init.settings, "critical_severity_threshold", 75.0)
    severity_map = getattr(
        _init.settings, "severity_score_map", DEFAULT_SEVERITY_SCORE_MAP
    )

    def _severity_to_score(sev: Any) -> float | None:
        if isinstance(sev, str):
            s = sev.lower()
            if s in severity_map:
                return severity_map[s]
            try:
                sev = float(sev)
            except ValueError:
                return None
        if isinstance(sev, (int, float)):
            val = float(sev)
            if 0 <= val <= 1:
                return val * 100
            if 0 <= val <= 5:
                return val * 20
            if 0 <= val <= 10:
                return val * 10
            return val
        return None

    for ev in events or []:
        sev = getattr(getattr(ev, "error_type", None), "severity", None)
        score = _severity_to_score(sev)
        if score is not None and score >= threshold:
            critical = True
            break
    logger = get_logger(__name__)

    if not missing and not critical and metrics and all(v > 0 for v in metrics.values()):
        logger.debug(
            "cycle evaluation skipped; all deltas positive",
            extra=log_record(reason="all_deltas_positive", metrics=metrics),
        )
        return "skip", {"reason": "all_deltas_positive", "metrics": metrics}
    if missing:
        logger.debug(
            "cycle evaluation running; missing metrics",
            extra=log_record(reason="missing_metrics", metrics=metrics, missing=missing),
        )
        return "run", {"reason": "missing_metrics", "metrics": metrics, "missing": missing}
    if critical:
        logger.debug(
            "cycle evaluation running; critical error detected",
            extra=log_record(reason="critical_error", metrics=metrics),
        )
        reason = "critical_error"
    else:
        reason = "non_positive_delta"
        logger.debug(
            "cycle evaluation running; non-positive delta",
            extra=log_record(reason=reason, metrics=metrics),
        )
    return "run", {"reason": reason, "metrics": metrics}


async def self_improvement_cycle(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    interval: float = PLANNER_INTERVAL,
    event_bus: UnifiedEventBus | None = None,
    error_log: Any | None = None,
    stop_event: threading.Event | None = None,
    should_encode: Callable[[Mapping[str, Any], "BaselineTracker", float], tuple[bool, str]]
    | None = None,
    evaluate_cycle: Callable[["BaselineTracker", Any | None], tuple[str, Mapping[str, Any]]]
    = _evaluate_cycle,
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
        planner = MetaWorkflowPlanner(context_builder=create_context_builder())

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
    repo_root_fn = getattr(_init, "_repo_path", lambda: Path("."))
    repo_path = Path(resolve_path(repo_root_fn()))
    try:  # pragma: no cover - optional snapshot dependencies
        from .snapshot_tracker import SnapshotTracker  # local import to avoid heavy deps
        snapshot_tracker = SnapshotTracker()
    except Exception:  # pragma: no cover - best effort fallback
        snapshot_tracker = None

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
        except Exception as exc:
            logger.warning(
                "entropy metric recording failed",
                extra=log_record(workflow_id=cid),
                exc_info=exc,
            )
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

    def _debug_cycle(
        outcome: str,
        *,
        reason: str | None = None,
        **extra: Any,
    ) -> None:
        tracker = BASELINE_TRACKER
        metrics: dict[str, float] = {}
        for name in getattr(tracker, "_history", {}):
            if name.endswith("_delta"):
                continue
            if name == "entropy":
                metrics["entropy_delta"] = float(
                    getattr(tracker, "entropy_delta", tracker.delta("entropy"))
                )
            else:
                metrics[f"{name}_delta"] = float(tracker.delta(name))
        extra_fields: dict[str, Any] = {}
        for key, value in extra.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            else:
                extra_fields[key] = value
        logger.debug(
            "cycle",
            extra=log_record(
                outcome=outcome,
                reason=reason,
                **metrics,
                **extra_fields,
            ),
        )

    if evaluate_cycle is None:
        raise ValueError("evaluate_cycle callable required")

    while True:
        if stop_event is not None and stop_event.is_set():
            _debug_cycle("skipped", reason="stop_event")
            break
        try:
            decision, info = evaluate_cycle(BASELINE_TRACKER, error_log)
            if info.get("reason") == "missing_metrics":
                _debug_cycle(
                    "run",
                    reason="missing_metrics",
                    missing_metrics=",".join(info.get("missing", [])),
                )
            if decision == "skip":
                traces, ent_delta, err_count, delta_mean, delta_std = _recent_error_entropy(
                    error_log,
                    BASELINE_TRACKER,
                    getattr(cfg, "error_window", 5),
                )
                max_errors, z_threshold = _get_overfit_thresholds(cfg, BASELINE_TRACKER)
                z_score = (
                    abs(ent_delta - delta_mean) / delta_std if delta_std > 0 else 0.0
                )
                if err_count > max_errors or z_score > z_threshold:
                    logger.debug(
                        "fallback_overfitting",
                        extra=log_record(
                            decision="run",
                            reason="fallback_overfitting",
                            entropy_delta=ent_delta,
                            entropy_z=z_score,
                            errors=err_count,
                            max_allowed_errors=max_errors,
                            entropy_overfit_threshold=z_threshold,
                        ),
                    )
                    decision = "run"
                    _debug_cycle(
                        "fallback",
                        reason="overfitting",
                        errors=err_count,
                        entropy_delta=ent_delta,
                        entropy_z=z_score,
                        entropy_overfit_threshold=z_threshold,
                        error_traces=traces,
                    )
                else:
                    _debug_cycle("skipped", reason=info.get("reason"))
                    continue

            if snapshot_tracker is not None:
                snapshot_tracker.capture(
                    "before",
                    {
                        "files": list(repo_path.rglob("*.py")),
                        "roi": BASELINE_TRACKER.current("roi"),
                        "sandbox_score": get_latest_sandbox_score(
                            SandboxSettings().sandbox_score_db
                        ),
                    },
                    repo_path=repo_path,
                )
            records = planner.discover_and_persist(workflows)
            active: list[list[str]] = []
            outcome_logged = False
            for rec in records:
                await _log(rec)
                roi = float(rec.get("roi_gain", 0.0))
                failures = int(rec.get("failures", 0))
                entropy = float(rec.get("entropy", 0.0))
                pass_rate = float(rec.get("pass_rate", 1.0 if failures == 0 else 0.0))
                repo = _init._repo_path()
                try:
                    from .metrics import compute_call_graph_complexity
                    call_complexity = compute_call_graph_complexity(repo)
                except Exception:
                    call_complexity = 0.0
                try:
                    from .metrics import compute_entropy_metrics
                    _, _, token_div = compute_entropy_metrics(
                        list(repo.rglob("*.py")), settings=cfg
                    )
                except Exception:
                    token_div = 0.0
                BASELINE_TRACKER.update(
                    roi=roi,
                    pass_rate=pass_rate,
                    entropy=entropy,
                    call_graph_complexity=call_complexity,
                    token_diversity=token_div,
                )
                tracker = BASELINE_TRACKER
                encode_fn = should_encode
                if encode_fn is None:
                    momentum = getattr(tracker, "momentum", 1.0) or 1.0
                    roi_delta = tracker.delta("roi") * momentum
                    if roi_delta <= 0:
                        should_encode, reason = False, "no_delta"
                    else:
                        skip_reason = None
                        for metric in tracker._history:
                            if metric.endswith("_delta") or metric in {
                                "roi",
                                "momentum",
                                "entropy",
                            }:
                                continue
                            if tracker.delta(metric) <= 0:
                                skip_reason = "no_delta"
                                break
                        if skip_reason is not None:
                            should_encode, reason = False, skip_reason
                        elif abs(tracker.delta("entropy")) > float(
                            cfg.overfitting_entropy_threshold
                        ):
                            should_encode, reason = False, "entropy_spike"
                        elif failures > 0:
                            should_encode, reason = False, "errors_present"
                        else:
                            should_encode, reason = True, "improved"
                else:
                    should_encode, reason = encode_fn(
                        rec,
                        tracker,
                        cfg.overfitting_entropy_threshold,
                    )
                if not should_encode:
                    outcome = (
                        "fallback" if reason in {"entropy_spike", "errors_present"} else "skipped"
                    )
                    _debug_cycle(outcome, reason=reason)
                    outcome_logged = True
                    continue
                else:
                    _debug_cycle("improved")
                    outcome_logged = True
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
            if not outcome_logged:
                _debug_cycle("skipped", reason="no_records")

            for chain in list(active):
                planner.cluster_map.pop(tuple(chain), None)

            if snapshot_tracker is not None:
                snapshot_tracker.capture(
                    "after",
                    {
                        "files": list(repo_path.rglob("*.py")),
                        "roi": BASELINE_TRACKER.current("roi"),
                        "sandbox_score": get_latest_sandbox_score(
                            SandboxSettings().sandbox_score_db
                        ),
                    },
                    repo_path=repo_path,
                )
                delta = snapshot_tracker.delta()
                _debug_cycle(
                    "snapshot",
                    roi_delta=delta.get("roi", 0.0),
                    entropy_delta=delta.get("entropy", 0.0),
                )

        except Exception as exc:  # pragma: no cover - planner is best effort
            _debug_cycle("error", reason=str(exc))
            logger.debug("error", extra=log_record(err=str(exc)))
            logger.exception("meta planner execution failed", exc_info=exc)
        finally:
            await asyncio.sleep(interval)


def start_self_improvement_cycle(
    workflows: Mapping[str, Callable[[], Any]],
    *,
    event_bus: UnifiedEventBus | None = None,
    interval: float = PLANNER_INTERVAL,
    error_log: Any | None = None,
    should_encode: Callable[
        [Mapping[str, Any], "BaselineTracker", float], tuple[bool, str]
    ]
    | None = None,
    evaluate_cycle: Callable[["BaselineTracker", Any | None], tuple[str, Mapping[str, Any]]]
    = _evaluate_cycle,
):
    """Prepare a background thread running :func:`self_improvement_cycle`.

    The returned object exposes ``start()``, ``join()`` and ``stop()`` methods.
    It captures exceptions raised inside the background task and re-raises them
    when ``join()`` is invoked.  ``stop()`` gracefully cancels the running
    coroutine.
    """

    if evaluate_cycle is None:
        raise ValueError("evaluate_cycle callable required")

    stop_event = threading.Event()

    class _CycleThread:
        def __init__(self) -> None:
            self._loop = asyncio.new_event_loop()
            self._task: asyncio.Task[None] | None = None
            self._exc: queue.Queue[BaseException] = queue.Queue()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._stop_event = stop_event
            self._error_log = error_log
            self._should_encode = should_encode
            self._evaluate_cycle = evaluate_cycle

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
            if "error_log" in signature(self_improvement_cycle).parameters:
                kwargs["error_log"] = self._error_log
            if "should_encode" in signature(self_improvement_cycle).parameters:
                kwargs["should_encode"] = self._should_encode
            if "evaluate_cycle" in signature(self_improvement_cycle).parameters:
                kwargs["evaluate_cycle"] = self._evaluate_cycle
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
