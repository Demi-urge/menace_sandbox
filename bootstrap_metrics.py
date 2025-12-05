"""Helpers for persisting bootstrap timing metrics and calibrating budgets."""
from __future__ import annotations

import json
import logging
import math
import os
import statistics
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import metrics_exporter as _metrics


def _normalize_label(label: object | None) -> str:
    if label is None:
        return "unknown"
    try:
        normalized = str(label)
    except Exception:
        return "unknown"
    return normalized or "unknown"

BOOTSTRAP_ATTEMPTS_TOTAL = getattr(
    _metrics,
    "bootstrap_attempts_total",
    _metrics.Gauge("bootstrap_attempts_total", "Total bootstrap attempts observed"),
)
_metrics.bootstrap_attempts_total = BOOTSTRAP_ATTEMPTS_TOTAL

BOOTSTRAP_CONTENTION_TOTAL = getattr(
    _metrics,
    "bootstrap_concurrent_contention_total",
    _metrics.Gauge(
        "bootstrap_concurrent_contention_total",
        "Bootstrap attempts that started while another attempt was active",
    ),
)
_metrics.bootstrap_concurrent_contention_total = BOOTSTRAP_CONTENTION_TOTAL

BOOTSTRAP_SUCCESS_TOTAL = getattr(
    _metrics,
    "bootstrap_success_total",
    _metrics.Gauge("bootstrap_success_total", "Successful bootstrap attempts"),
)
_metrics.bootstrap_success_total = BOOTSTRAP_SUCCESS_TOTAL

BOOTSTRAP_FAILURE_TOTAL = getattr(
    _metrics,
    "bootstrap_failure_total",
    _metrics.Gauge("bootstrap_failure_total", "Failed bootstrap attempts"),
)
_metrics.bootstrap_failure_total = BOOTSTRAP_FAILURE_TOTAL

BOOTSTRAP_TIMEOUT_TOTAL = getattr(
    _metrics,
    "bootstrap_timeout_total",
    _metrics.Gauge(
        "bootstrap_timeout_total", "Bootstrap attempts that exhausted their budget"
    ),
)
_metrics.bootstrap_timeout_total = BOOTSTRAP_TIMEOUT_TOTAL

BOOTSTRAP_INFLIGHT = getattr(
    _metrics,
    "bootstrap_inflight",
    _metrics.Gauge("bootstrap_inflight", "Concurrent in-flight bootstrap attempts"),
)
_metrics.bootstrap_inflight = BOOTSTRAP_INFLIGHT

BOOTSTRAP_PREPARE_REPEAT_TOTAL = getattr(
    _metrics,
    "bootstrap_prepare_repeat_total",
    _metrics.Gauge(
        "bootstrap_prepare_repeat_total",
        "Prepare pipeline invocations suppressed because a prior call is still active",
        ["caller"],
    ),
)
_metrics.bootstrap_prepare_repeat_total = BOOTSTRAP_PREPARE_REPEAT_TOTAL

BOOTSTRAP_ENTRY_TOTAL = getattr(
    _metrics,
    "bootstrap_entry_total",
    _metrics.Gauge(
        "bootstrap_entry_total",
        "Bootstrap entrypoint invocations by status",
        ["status", "module", "step"],
    ),
)
_metrics.bootstrap_entry_total = BOOTSTRAP_ENTRY_TOTAL

BOOTSTRAP_ATTEMPT_DURATION = getattr(
    _metrics,
    "bootstrap_attempt_duration_seconds",
    _metrics.Gauge(
        "bootstrap_attempt_duration_seconds",
        "Duration of bootstrap attempts by outcome",
        ["outcome"],
    ),
)
_metrics.bootstrap_attempt_duration_seconds = BOOTSTRAP_ATTEMPT_DURATION

BOOTSTRAP_SKIP_TOTAL = getattr(
    _metrics,
    "bootstrap_skip_total",
    _metrics.Gauge(
        "bootstrap_skip_total",
        "Bootstrap invocations skipped before executing",
        ["reason", "module", "step"],
    ),
)
_metrics.bootstrap_skip_total = BOOTSTRAP_SKIP_TOTAL

BOOTSTRAP_GUARD_TOTAL = getattr(
    _metrics,
    "bootstrap_guard_total",
    _metrics.Gauge(
        "bootstrap_guard_total",
        "Bootstrap guard interceptions by reason",
        ["reason", "module", "step"],
    ),
)
_metrics.bootstrap_guard_total = BOOTSTRAP_GUARD_TOTAL

BOOTSTRAP_LIFECYCLE_TOTAL = getattr(
    _metrics,
    "bootstrap_lifecycle_total",
    _metrics.Gauge(
        "bootstrap_lifecycle_total",
        "Bootstrap lifecycle transitions by event",
        ["event", "module", "step"],
    ),
)
_metrics.bootstrap_lifecycle_total = BOOTSTRAP_LIFECYCLE_TOTAL

BOOTSTRAP_RECURSION_TOTAL = getattr(
    _metrics,
    "bootstrap_recursion_total",
    _metrics.Gauge(
        "bootstrap_recursion_total",
        "Recursive bootstrap attempts detected",
        ["module", "step"],
    ),
)
_metrics.bootstrap_recursion_total = BOOTSTRAP_RECURSION_TOTAL

BOOTSTRAP_DENSITY_TOTAL = getattr(
    _metrics,
    "bootstrap_attempt_density_total",
    _metrics.Gauge(
        "bootstrap_attempt_density_total",
        "Bootstrap attempts observed within the active density window",
        ["module", "step"],
    ),
)
_metrics.bootstrap_attempt_density_total = BOOTSTRAP_DENSITY_TOTAL

_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT_COUNT = 0


class BootstrapAttempt(AbstractContextManager["BootstrapAttempt"]):
    """Context manager that records bootstrap attempt outcomes and contention."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self.logger = logger
        self._start: float | None = None
        self._outcome: str | None = None
        self._contended = False

    def __enter__(self) -> "BootstrapAttempt":  # type: ignore[override]
        global _INFLIGHT_COUNT
        with _INFLIGHT_LOCK:
            _INFLIGHT_COUNT += 1
            BOOTSTRAP_INFLIGHT.set(float(_INFLIGHT_COUNT))
            self._contended = _INFLIGHT_COUNT > 1
            if self._contended:
                BOOTSTRAP_CONTENTION_TOTAL.inc()
        BOOTSTRAP_ATTEMPTS_TOTAL.inc()
        self._start = time.monotonic()
        return self

    def _record_outcome(self, outcome: str) -> None:
        duration = None
        if self._start is not None:
            duration = max(time.monotonic() - self._start, 0.0)
            BOOTSTRAP_ATTEMPT_DURATION.labels(outcome).set(duration)

        if outcome == "success":
            BOOTSTRAP_SUCCESS_TOTAL.inc()
        elif outcome == "timeout":
            BOOTSTRAP_TIMEOUT_TOTAL.inc()
        else:
            BOOTSTRAP_FAILURE_TOTAL.inc()

        if self.logger:
            extra = {
                "event": "bootstrap-attempt",
                "outcome": outcome,
                "contended": self._contended,
            }
            if duration is not None:
                extra["duration"] = round(duration, 2)
            self.logger.info("bootstrap attempt recorded", extra=extra)

    def success(self) -> None:
        """Mark the attempt as successful."""

        self._outcome = "success"

    def failure(self) -> None:
        """Mark the attempt as failed."""

        self._outcome = "failure"

    def timeout(self) -> None:
        """Mark the attempt as a timeout."""

        self._outcome = "timeout"

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[override]
        global _INFLIGHT_COUNT
        outcome = self._outcome
        if outcome is None:
            if exc_type and issubclass(exc_type, TimeoutError):
                outcome = "timeout"
            elif exc_type is not None:
                outcome = "failure"
            else:
                outcome = "success"

        try:
            self._record_outcome(outcome)
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT_COUNT = max(_INFLIGHT_COUNT - 1, 0)
                BOOTSTRAP_INFLIGHT.set(float(_INFLIGHT_COUNT))
        return False


def bootstrap_attempt(*, logger: logging.Logger | None = None) -> BootstrapAttempt:
    """Return a :class:`BootstrapAttempt` for measuring bootstrap health."""

    return BootstrapAttempt(logger=logger)


def record_bootstrap_entry(
    status: str,
    *,
    module: str | None = None,
    step: str | None = None,
    logger: logging.Logger | None = None,
    **context: object,
) -> None:
    """Record a bootstrap entrypoint attempt with the provided status."""

    label = status.replace(" ", "-")
    BOOTSTRAP_ENTRY_TOTAL.labels(
        status=label, module=_normalize_label(module), step=_normalize_label(step)
    ).inc()
    if logger:
        extra = {
            "event": "bootstrap-entry",
            "status": status,
            "bootstrap_module": module,
            "step": step,
        }
        if context:
            extra.update({f"context.{k}": v for k, v in context.items()})
        logger.info("bootstrap entry recorded", extra=extra)


def record_bootstrap_skip(
    reason: str,
    *,
    module: str | None = None,
    step: str | None = None,
    logger: logging.Logger | None = None,
    **context: object,
) -> None:
    """Record that a bootstrap invocation was skipped.

    Parameters
    ----------
    reason:
        Short descriptor for why the call was skipped, e.g. ``"cached"`` or
        ``"in-flight"``.
    logger:
        Optional logger for structured auditing.
    context:
        Additional fields to enrich the structured log output.
    """

    label = reason.replace(" ", "-")
    BOOTSTRAP_SKIP_TOTAL.labels(
        reason=label, module=_normalize_label(module), step=_normalize_label(step)
    ).inc()
    if logger:
        extra = {
            "event": "bootstrap-skip",
            "reason": reason,
            "bootstrap_module": module,
            "step": step,
        }
        if context:
            extra.update({f"context.{k}": v for k, v in context.items()})
        logger.info("bootstrap invocation skipped", extra=extra)


def record_bootstrap_guard(
    reason: str,
    *,
    module: str | None = None,
    step: str | None = None,
    logger: logging.Logger | None = None,
    **context: object,
) -> None:
    """Record a guard interception (skip/queued) with structured metadata."""

    label = reason.replace(" ", "-")
    BOOTSTRAP_GUARD_TOTAL.labels(
        reason=label, module=_normalize_label(module), step=_normalize_label(step)
    ).inc()
    if logger:
        extra = {
            "event": "bootstrap-guard",
            "reason": reason,
            "bootstrap_module": module,
            "step": step,
        }
        if context:
            extra.update({f"context.{k}": v for k, v in context.items()})
        logger.info("bootstrap guard interception", extra=extra)


def record_bootstrap_lifecycle(
    event: str,
    *,
    module: str | None = None,
    step: str | None = None,
    logger: logging.Logger | None = None,
    **context: object,
) -> None:
    """Record structured lifecycle transitions for bootstrap helpers."""

    label = event.replace(" ", "-")
    BOOTSTRAP_LIFECYCLE_TOTAL.labels(
        event=label, module=_normalize_label(module), step=_normalize_label(step)
    ).inc()
    if logger:
        extra = {
            "event": "bootstrap-lifecycle",
            "lifecycle": event,
            "bootstrap_module": module,
            "step": step,
        }
        if context:
            extra.update({f"context.{k}": v for k, v in context.items()})
        logger.info("bootstrap lifecycle event", extra=extra)


def record_bootstrap_recursion(
    *,
    module: str | None = None,
    step: str | None = None,
    logger: logging.Logger | None = None,
    **context: object,
) -> None:
    """Record a recursive bootstrap invocation that was intercepted."""

    BOOTSTRAP_RECURSION_TOTAL.labels(
        module=_normalize_label(module), step=_normalize_label(step)
    ).inc()
    if logger:
        extra = {
            "event": "bootstrap-recursion",
            "bootstrap_module": module,
            "step": step,
        }
        if context:
            extra.update({f"context.{k}": v for k, v in context.items()})
        logger.error("recursive bootstrap attempt detected", extra=extra)


def record_attempt_density(
    *,
    module: str | None = None,
    step: str | None = None,
    attempts_in_window: int,
    window_seconds: float,
    logger: logging.Logger | None = None,
    **context: object,
) -> None:
    """Record the number of attempts observed within the recent window."""

    BOOTSTRAP_DENSITY_TOTAL.labels(
        module=_normalize_label(module), step=_normalize_label(step)
    ).set(float(attempts_in_window))
    if logger:
        extra = {
            "event": "bootstrap-density",
            "bootstrap_module": module,
            "step": step,
            "attempts_in_window": attempts_in_window,
            "window_seconds": window_seconds,
        }
        if context:
            extra.update({f"context.{k}": v for k, v in context.items()})
        logger.warning("bootstrap attempt density updated", extra=extra)

BOOTSTRAP_DURATION_STORE = Path(__file__).resolve().parent / "sandbox_data" / "bootstrap_durations.json"
DURATION_HISTORY_LIMIT = int(os.getenv("BOOTSTRAP_DURATION_HISTORY_LIMIT", "40"))
BUDGET_BUFFER_MULTIPLIER = float(os.getenv("BOOTSTRAP_BUDGET_BUFFER", "1.15"))
BUDGET_MAX_SCALE = float(os.getenv("BOOTSTRAP_BUDGET_MAX_SCALE", "3.0"))


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _safe_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        # Best-effort persistence; failures should not block bootstrap.
        return


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("values required for percentile")
    if len(values) == 1:
        return float(values[0])
    clipped = max(0.0, min(1.0, percentile))
    ordered = sorted(values)
    index = clipped * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[int(index)])
    fraction = index - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def load_duration_store() -> dict[str, Any]:
    return _safe_read_json(BOOTSTRAP_DURATION_STORE)


def record_durations(
    *,
    durations: Mapping[str, float],
    category: str = "bootstrap_steps",
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Persist observed durations under *category* and return the updated store."""

    store = load_duration_store()
    bucket: MutableMapping[str, list[float]] = store.setdefault(category, {})  # type: ignore[assignment]
    changed = False

    for step, duration in durations.items():
        if duration <= 0:
            continue
        history = list(bucket.get(step, []))
        history.append(float(duration))
        if len(history) > DURATION_HISTORY_LIMIT:
            history = history[-DURATION_HISTORY_LIMIT:]
        bucket[step] = history  # type: ignore[index]
        changed = True

    if changed:
        _safe_write_json(BOOTSTRAP_DURATION_STORE, store)
        if logger:
            logger.info(
                "recorded bootstrap durations",
                extra={
                    "event": "bootstrap-duration-recorded",
                    "category": category,
                    "steps": sorted(durations),
                },
            )
    return store


def compute_stats(history: Mapping[str, Sequence[float]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for step, samples in history.items():
        sample_list = [float(value) for value in samples if value is not None]
        if not sample_list:
            continue
        try:
            stats[step] = {
                "median": statistics.median(sample_list),
                "p95": _percentile(sample_list, 0.95),
            }
        except Exception:
            continue
    return stats


def calibrate_step_budgets(
    *,
    base_budgets: Mapping[str, float],
    stats: Mapping[str, Mapping[str, float]],
    budget_buffer: float | None = None,
    max_scale: float | None = None,
    floors: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Calibrate step budgets using observed medians/p95s.

    Returns a tuple of ``(calibrated_budgets, debug_metadata)``.
    """

    buffer = budget_buffer if budget_buffer is not None else BUDGET_BUFFER_MULTIPLIER
    scale_cap = max_scale if max_scale is not None else BUDGET_MAX_SCALE

    calibrated: dict[str, float] = {}
    debug: dict[str, Any] = {"adjusted": {}, "buffer": buffer, "scale_cap": scale_cap}

    for step, base_budget in base_budgets.items():
        floor = floors.get(step, 0.0) if floors else 0.0
        baseline = max(base_budget, floor)
        step_stats = stats.get(step, {})
        candidates = [value for key, value in step_stats.items() if key in {"median", "p95"} and value is not None]
        calibrated_budget = baseline

        if candidates:
            target = max(candidates) * buffer
            calibrated_budget = max(baseline, target)
        cap = baseline * scale_cap if scale_cap > 0 else None
        if cap is not None:
            calibrated_budget = min(calibrated_budget, cap)

        calibrated[step] = calibrated_budget
        if calibrated_budget != base_budget:
            debug["adjusted"][step] = {
                "from": base_budget,
                "to": calibrated_budget,
                "stats": dict(step_stats),
                "baseline": baseline,
            }

    return calibrated, debug


def calibrate_overall_timeout(
    *, base_timeout: float, calibrated_budgets: Mapping[str, float], max_scale: float | None = None
) -> tuple[float, Mapping[str, Any]]:
    """Scale the overall timeout to account for calibrated per-step budgets."""

    scale_cap = max_scale if max_scale is not None else BUDGET_MAX_SCALE
    suggested_budget = max(calibrated_budgets.values(), default=base_timeout)
    adjusted_timeout = max(base_timeout, suggested_budget)
    adjusted_timeout = min(adjusted_timeout, base_timeout * scale_cap)

    return adjusted_timeout, {"suggested_budget": suggested_budget, "scale_cap": scale_cap}
