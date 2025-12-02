from __future__ import annotations

"""Lazy bootstrap helpers for vector service assets.

This module centralises deferred initialisation for heavyweight resources such
as the bundled embedding model and the embedding scheduler.  Callers can either
rely on the on-demand helpers (which cache results) or invoke the warmup
routine to pre-populate caches before the first real request.
"""

import importlib.util
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import metrics_exporter as _metrics

try:  # pragma: no cover - lightweight import wrapper
    from dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback when run standalone
    resolve_path = Path  # type: ignore[assignment]


_MODEL_LOCK = threading.Lock()
_MODEL_READY = False
_SCHEDULER_LOCK = threading.Lock()
_SCHEDULER: Any | None | bool = None  # False means attempted and unavailable
_WARMUP_STAGE_MEMO: dict[str, str] = {}

_CONSERVATIVE_STAGE_TIMEOUTS = {
    "model": 12.0,
    "handlers": 12.0,
    "vectorise": 6.0,
}

VECTOR_WARMUP_STAGE_TOTAL = getattr(
    _metrics,
    "vector_warmup_stage_total",
    _metrics.Gauge(
        "vector_warmup_stage_total",
        "Vector warmup stage results by status",
        ["stage", "status"],
    ),
)


def _model_bundle_path() -> Path:
    return resolve_path("vector_service/minilm/tiny-distilroberta-base.tar.xz")


def ensure_embedding_model(
    *,
    logger: logging.Logger | None = None,
    warmup: bool = False,
    stop_event: threading.Event | None = None,
    budget_check: Callable[[threading.Event | None], None] | None = None,
) -> Path | None:
    """Ensure the bundled embedding model archive exists.

    The download is performed at most once per process and only when the model
    is missing.  When ``warmup`` is False the function favours fast failure so
    first-use callers can fall back gracefully; during warmup we log and swallow
    errors to avoid breaking bootstrap flows.
    """

    global _MODEL_READY
    log = logger or logging.getLogger(__name__)

    def _coerce_timeout(value: object) -> float | None:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
    def _check_cancelled(context: str) -> None:
        if stop_event is not None and stop_event.is_set():
            raise TimeoutError(f"embedding model download cancelled during {context}")
        if budget_check is not None:
            budget_check(stop_event)

    _check_cancelled("init")

    if _MODEL_READY:
        return _model_bundle_path()

    with _MODEL_LOCK:
        if _MODEL_READY:
            return _model_bundle_path()
        dest = _model_bundle_path()
        if dest.exists():
            _MODEL_READY = True
            return dest

        _check_cancelled("init")

        if importlib.util.find_spec("huggingface_hub") is None:
            log.info(
                "embedding model download skipped (huggingface-hub unavailable); will retry on demand"
            )
            return None

        try:
            from . import download_model as _dm

            _check_cancelled("fetch")
            _dm.bundle(dest, stop_event=stop_event, budget_check=budget_check)
            _MODEL_READY = True
            return dest
        except Exception as exc:  # pragma: no cover - best effort during warmup
            log.warning("embedding model bootstrap failed: %s", exc)
            if warmup:
                return None
            raise


def ensure_scheduler_started(*, logger: logging.Logger | None = None) -> Any | None:
    """Start the embedding scheduler once and cache the result."""

    global _SCHEDULER
    log = logger or logging.getLogger(__name__)
    with _SCHEDULER_LOCK:
        if _SCHEDULER is not None:
            return None if _SCHEDULER is False else _SCHEDULER
        try:
            from .embedding_scheduler import start_scheduler_from_env

            _SCHEDULER = start_scheduler_from_env()
            return _SCHEDULER
        except Exception as exc:  # pragma: no cover - defensive logging
            log.warning("embedding scheduler warmup failed: %s", exc)
            _SCHEDULER = False
            return None


def warmup_vector_service(
    *,
    download_model: bool = False,
    probe_model: bool = False,
    hydrate_handlers: bool = False,
    start_scheduler: bool = False,
    run_vectorise: bool | None = None,
    check_budget: Callable[[], None] | None = None,
    budget_remaining: Callable[[], float | None] | None = None,
    logger: logging.Logger | None = None,
    force_heavy: bool = False,
    bootstrap_fast: bool | None = None,
    warmup_lite: bool = True,
    warmup_model: bool | None = None,
    warmup_handlers: bool | None = None,
    warmup_probe: bool | None = None,
    stage_timeouts: dict[str, float] | float | None = None,
    deferred_stages: set[str] | None = None,
    background_hook: Callable[[set[str]], None] | None = None,
) -> Mapping[str, str]:
    """Eagerly initialise vector assets and caches.

    The default behaviour favours a "light" warmup that validates scheduler
    configuration and optional model presence without instantiating
    ``SharedVectorService``.  Callers may opt-in to handler hydration and
    vectorisation by setting ``hydrate_handlers=True`` (and optionally
    ``run_vectorise=True``) when a heavier warmup is desired.  ``warmup_lite``
    defaults to True so bootstrap flows skip handler hydration and vectorise
    steps unless explicitly requested.

    When ``bootstrap_fast`` is True the vector service keeps the patch handler
    stubbed during warmup and avoids loading heavy indexes.  Stage timeouts can
    be provided via ``stage_timeouts`` to cap how long heavyweight tasks are
    allowed to block before they are deferred to background execution.
    ``stage_timeouts`` may be a mapping of per-stage ceilings or a numeric
    budget that is split across the stages, allowing callers without
    ``check_budget`` hooks to bound the work.  A ``budget_remaining`` callback
    can be supplied to shorten those per-stage limits (or skip stages entirely)
    when the caller's remaining bootstrap time falls below the configured
    thresholds.  Callers that intentionally defer stages can supply
    ``deferred_stages`` so the warmup summary reflects the deferral rather than
    a silent skip.  ``background_hook`` is invoked with any stages proactively
    deferred for background execution so callers can enqueue follow-up tasks.
    """

    log = logger or logging.getLogger(__name__)
    stage_timeouts_supplied = stage_timeouts is not None
    env_budget = _coerce_timeout(os.getenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"))
    if env_budget is None:
        env_budget = _coerce_timeout(os.getenv("BOOTSTRAP_VECTOR_STEP_TIMEOUT"))
    if env_budget is None:
        env_budget = _coerce_timeout(os.getenv("MENACE_BOOTSTRAP_TIMEOUT"))
    env_budget = env_budget if env_budget is not None and env_budget > 0 else None
    budget_start = time.monotonic()

    def _default_budget_remaining() -> float | None:
        if env_budget is None:
            return None
        remaining = env_budget - (time.monotonic() - budget_start)
        return max(0.0, remaining)

    def _default_check_budget(_evt: threading.Event | None = None) -> None:
        remaining = _default_budget_remaining()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("bootstrap vector warmup budget exhausted")

    if budget_remaining is None:
        budget_remaining = _default_budget_remaining
    if check_budget is None:
        check_budget = _default_check_budget if env_budget is not None else None
    if stage_timeouts is None and env_budget is not None:
        stage_timeouts = env_budget

    bootstrap_context = any(
        os.getenv(flag, "").strip().lower() in {"1", "true", "yes", "on"}
        for flag in ("MENACE_BOOTSTRAP", "MENACE_BOOTSTRAP_FAST", "MENACE_BOOTSTRAP_MODE")
    )

    bootstrap_fast = bool(bootstrap_fast)

    fast_vector_env = os.getenv("MENACE_VECTOR_WARMUP_FAST", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if warmup_model is not None:
        download_model = warmup_model
    if warmup_handlers is not None:
        hydrate_handlers = warmup_handlers
    if warmup_probe is not None:
        probe_model = warmup_probe

    if fast_vector_env and not force_heavy:
        if download_model:
            log.info("Fast vector warmup requested; skipping embedding model download")
        if probe_model:
            log.info("Fast vector warmup requested; skipping embedding model probe")
        if hydrate_handlers:
            log.info("Fast vector warmup requested; skipping handler hydration")
        download_model = False
        probe_model = False
        hydrate_handlers = False
        run_vectorise = False

    bootstrap_lite = bootstrap_context and not force_heavy
    heavy_requested = any(
        flag
        for flag in (
            download_model,
            hydrate_handlers,
            start_scheduler,
            bool(run_vectorise),
        )
    )
    deferred_bootstrap: set[str] = set()

    if bootstrap_lite:
        heavy_requested = download_model or hydrate_handlers or start_scheduler or run_vectorise
        if heavy_requested:
            log.info(
                "Bootstrap context detected; deferring heavy vector warmup (force_heavy to override)",
                extra={
                    "download_model": download_model,
                    "hydrate_handlers": hydrate_handlers,
                    "start_scheduler": start_scheduler,
                    "run_vectorise": run_vectorise,
                },
            )
        if not warmup_lite:
            log.info("Bootstrap context detected; forcing warmup_lite")
        warmup_lite = True
        run_vectorise = False
        if download_model:
            log.info("Bootstrap context detected; skipping embedding model download")
            download_model = False
            deferred_bootstrap.add("model")
        if hydrate_handlers:
            log.info("Bootstrap context detected; deferring handler hydration")
            hydrate_handlers = False
            deferred_bootstrap.add("handlers")
        if start_scheduler:
            log.info("Bootstrap context detected; scheduler start deferred")
            start_scheduler = False
            deferred_bootstrap.add("scheduler")
        if run_vectorise:
            deferred_bootstrap.add("vectorise")
        summary_flag = "deferred-bootstrap"
    else:
        summary_flag = "normal"

    warmup_lite = bool(warmup_lite)
    lite_deferrals: set[str] = set()
    if warmup_lite and not force_heavy:
        lite_deferrals.update({"handlers", "scheduler", "vectorise"})
        if hydrate_handlers or start_scheduler or run_vectorise:
            log.info(
                "Warmup-lite enabled; deferring heavy vector warmup stages (force_heavy to override)",
                extra={
                    "hydrate_handlers": hydrate_handlers,
                    "start_scheduler": start_scheduler,
                    "run_vectorise": run_vectorise,
                },
            )
        hydrate_handlers = False
        start_scheduler = False
        run_vectorise = False

    no_timeout_deferrals: set[str] = set()
    if not stage_timeouts_supplied and not force_heavy:
        no_timeout_deferrals.update({"handlers", "scheduler", "vectorise"})
        if hydrate_handlers or start_scheduler or run_vectorise:
            log.info(
                "Stage timeouts not provided; deferring heavy vector warmup (force_heavy to override)",
                extra={
                    "hydrate_handlers": hydrate_handlers,
                    "start_scheduler": start_scheduler,
                    "run_vectorise": run_vectorise,
                },
            )
        hydrate_handlers = False
        start_scheduler = False
        run_vectorise = False
    lite_deferrals.update(no_timeout_deferrals)

    summary: dict[str, str] = {"bootstrap": summary_flag, "warmup_lite": str(warmup_lite)}
    deferred = set(deferred_stages or ()) | deferred_bootstrap | lite_deferrals
    memoised_results = dict(_WARMUP_STAGE_MEMO)
    prior_deferred = set(deferred_stages or ()) | {
        stage for stage, status in memoised_results.items() if status.startswith("deferred")
    }

    recorded_deferred: set[str] = set()
    background_candidates: set[str] = set()
    effective_timeouts: dict[str, float | None] = {}

    background_warmup: set[str] = set()

    def _record(stage: str, status: str) -> None:
        summary[stage] = status
        if status.startswith("deferred"):
            recorded_deferred.add(stage)
        _WARMUP_STAGE_MEMO[stage] = status
        try:
            VECTOR_WARMUP_STAGE_TOTAL.labels(stage, status).inc()
        except Exception:  # pragma: no cover - metrics best effort
            log.debug("failed emitting vector warmup metric", exc_info=True)

    def _record_deferred(stage: str, status: str) -> None:
        _record(stage, status)
        if stage in prior_deferred:
            return
        background_candidates.add(stage)

    def _record_background(stage: str, status: str) -> None:
        _record(stage, status)
        if stage in prior_deferred:
            return
        background_warmup.add(stage)
        background_candidates.add(stage)

    def _reuse(stage: str) -> bool:
        status = memoised_results.get(stage)
        if status is None:
            return False
        if force_heavy and status.startswith("deferred"):
            return False
        summary[stage] = status
        if status.startswith("deferred"):
            recorded_deferred.add(stage)
        return True

    budget_exhausted = False

    def _remaining_budget() -> float | None:
        if budget_remaining is None:
            return None
        try:
            return budget_remaining()
        except Exception:  # pragma: no cover - budget hint is advisory
            log.debug("budget_remaining callback failed", exc_info=True)
            return None

    def _guard(stage: str) -> bool:
        nonlocal budget_exhausted
        if budget_exhausted:
            _record_deferred(stage, "skipped-budget")
            log.info("Vector warmup budget already exhausted; deferring %s", stage)
            return False
        remaining = _remaining_budget()
        if remaining is not None and remaining <= 0:
            budget_exhausted = True
            _record_deferred(stage, "skipped-budget")
            log.info(
                "Vector warmup budget exhausted before %s; skipping heavy stages", stage
            )
            return False
        if check_budget is None:
            return True
        try:
            check_budget()
            log.debug("vector warmup budget check after %s", stage)
            return True
        except TimeoutError as exc:
            budget_exhausted = True
            _record_deferred(stage, "deferred-budget")
            log.warning("Vector warmup deadline reached before %s: %s", stage, exc)
            return False

    def _should_abort(stage: str) -> bool:
        if budget_exhausted:
            log.info(
                "Vector warmup budget exhausted; skipping remaining heavy stages after %s",
                stage,
            )
            return True
        return False

    def _record_timeout(stage: str) -> None:
        try:
            VECTOR_WARMUP_STAGE_TOTAL.labels(stage, "timeout").inc()
        except Exception:  # pragma: no cover - metrics best effort
            log.debug("failed emitting vector warmup timeout metric", exc_info=True)

    def _cooperative_budget_check(stage: str, stop_event: threading.Event | None) -> None:
        if stop_event is not None and stop_event.is_set():
            raise TimeoutError(f"vector warmup {stage} cancelled")
        if check_budget is None:
            return
        try:
            check_budget()
        except TimeoutError:
            if stop_event is not None:
                stop_event.set()
            raise

    def _finalise() -> Mapping[str, str]:
        deferred_record = deferred | recorded_deferred
        if deferred_record:
            summary["deferred"] = ",".join(sorted(deferred_record))
        if background_candidates:
            summary["background"] = ",".join(sorted(background_candidates))
        for stage, timeout in effective_timeouts.items():
            summary[f"budget_{stage}"] = (
                f"{timeout:.3f}" if timeout is not None else "none"
            )
        hook_dispatched = False
        if background_candidates and background_hook is not None:
            try:
                background_hook(set(background_candidates))
                hook_dispatched = True
            except Exception:  # pragma: no cover - advisory hook
                log.debug("background hook failed", exc_info=True)
        if background_warmup and not hook_dispatched:
            _launch_background_warmup(set(background_warmup))

        log.info(
            "vector warmup stages recorded", extra={"event": "vector-warmup", "warmup": summary}
        )
        log.debug("vector warmup summary: %s", summary)

        return summary

    def _run_with_budget(
        stage: str,
        func: Callable[[threading.Event], Any],
        *,
        timeout: float | None = None,
    ) -> tuple[bool, Any | None]:
        nonlocal budget_exhausted
        stop_event = threading.Event()
        if check_budget is None and timeout is None:
            return True, func(stop_event)

        result: list[Any | None] = []
        error: list[BaseException] = []
        done = threading.Event()

        def _detach_runner(reason: str) -> None:
            def _cleanup() -> None:
                if done.wait(timeout=0.5):
                    return
                if thread.is_alive():
                    log.debug(
                        "vector warmup %s thread still active after %s; leaving detached",
                        stage,
                        reason,
                    )

            threading.Thread(target=_cleanup, daemon=True).start()

        def _runner() -> None:
            try:
                result.append(func(stop_event))
            except BaseException as exc:  # pragma: no cover - propagated to caller
                error.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()

        start = time.monotonic()
        while not done.wait(timeout=0.05):
            if timeout is not None and time.monotonic() - start >= timeout:
                stop_event.set()
                _record_timeout(stage)
                budget_exhausted = True
                _record_deferred(stage, "deferred-timeout")
                log.warning(
                    "Vector warmup %s timed out after %.2fs; deferring", stage, timeout
                )
                _detach_runner("timeout")
                return False, None

            if check_budget is not None:
                try:
                    check_budget()
                except TimeoutError as exc:
                    stop_event.set()
                    budget_exhausted = True
                    _record_timeout(stage)
                    _record_deferred(stage, "deferred-budget")
                    log.warning("Vector warmup deadline reached during %s: %s", stage, exc)
                    done.wait(timeout=0.25)
                    return False, None

        if error:
            err = error[0]
            if isinstance(err, TimeoutError):
                stop_event.set()
                budget_exhausted = True
                log.info("Vector warmup %s cancelled: %s", stage, err)
                _record_deferred(stage, "deferred-budget")
                return False, None
            raise err

        return True, result[0] if result else None

    base_timeouts = {
        "model": 10.0,
        "handlers": 10.0,
        "vectorise": 5.0,
    }
    base_stage_cost = {"model": 20.0, "handlers": 25.0, "vectorise": 8.0}
    if bootstrap_context or bootstrap_fast or not stage_timeouts_supplied:
        base_timeouts = dict(_CONSERVATIVE_STAGE_TIMEOUTS)

    provided_budget = _coerce_timeout(stage_timeouts) if not isinstance(stage_timeouts, Mapping) else None
    resolved_timeouts: dict[str, float | None] = dict(base_timeouts)
    explicit_timeouts: set[str] = set()

    if isinstance(stage_timeouts, Mapping):
        for name, timeout in stage_timeouts.items():
            if name == "budget":
                provided_budget = _coerce_timeout(timeout)
                continue
            coerced = _coerce_timeout(timeout)
            if coerced is None:
                continue
            resolved_timeouts[name] = coerced
            explicit_timeouts.add(name)

    def _apply_budget_caps(timeouts: dict[str, float | None], budget: float | None) -> dict[str, float | None]:
        if budget is None:
            return timeouts

        explicit_total = sum(
            value for key, value in timeouts.items() if key in explicit_timeouts and value is not None
        )
        remaining_budget = budget - explicit_total
        flexible = [key for key in timeouts if key not in explicit_timeouts and timeouts.get(key) is not None]

        if remaining_budget <= 0:
            for stage in flexible:
                timeouts[stage] = 0.0
            return timeouts

        flexible_default = sum(base_timeouts.get(stage, timeouts.get(stage, 0.0)) or 0.0 for stage in flexible)
        if flexible_default <= 0:
            return timeouts

        scale = min(1.0, remaining_budget / flexible_default)
        for stage in flexible:
            base_default = base_timeouts.get(stage, timeouts.get(stage) or 0.0) or 0.0
            timeouts[stage] = max(0.0, min(timeouts[stage] or base_default, base_default * scale))
        return timeouts

    bootstrap_budget_cap = _remaining_budget() if bootstrap_lite else None
    if bootstrap_budget_cap is not None:
        provided_budget = (
            bootstrap_budget_cap
            if provided_budget is None
            else min(provided_budget, bootstrap_budget_cap)
        )

    resolved_timeouts = _apply_budget_caps(resolved_timeouts, provided_budget)

    def _launch_background_warmup(stages: set[str]) -> None:
        if not stages:
            return

        background_timeouts: dict[str, float | None] = dict(resolved_timeouts)
        if provided_budget is not None:
            background_timeouts["budget"] = provided_budget

        def _run_background() -> None:
            try:
                warmup_vector_service(
                    download_model=download_model,
                    probe_model=probe_model,
                    hydrate_handlers="handlers" in stages,
                    start_scheduler="scheduler" in stages,
                    run_vectorise="vectorise" in stages,
                    check_budget=None,
                    budget_remaining=None,
                    logger=log,
                    force_heavy=True,
                    bootstrap_fast=bootstrap_fast,
                    warmup_lite=False,
                    warmup_model=warmup_model,
                    warmup_handlers=True,
                    warmup_probe=warmup_probe,
                    stage_timeouts=background_timeouts,
                    deferred_stages=set(),
                    background_hook=None,
                )
            except Exception:  # pragma: no cover - background best effort
                log.debug("background vector warmup failed", exc_info=True)

        threading.Thread(target=_run_background, daemon=True).start()

    def _stage_budget_cap() -> float | None:
        if isinstance(stage_timeouts, Mapping):
            return _coerce_timeout(stage_timeouts.get("budget"))
        return _coerce_timeout(stage_timeouts)

    stage_budget_cap = _stage_budget_cap()
    heavy_budget_needed = 0.0
    if download_model:
        heavy_budget_needed += base_stage_cost["model"]
    if hydrate_handlers:
        heavy_budget_needed += base_stage_cost["handlers"]
    if run_vectorise:
        heavy_budget_needed += base_stage_cost["vectorise"]

    if stage_budget_cap is not None and heavy_budget_needed > stage_budget_cap and not force_heavy:
        deferred.update(
            {
                stage
                for stage, enabled in (
                    ("model", download_model),
                    ("handlers", hydrate_handlers),
                    ("vectorise", run_vectorise),
                )
                if enabled
            }
        )
        warmup_lite = True
        download_model = False
        hydrate_handlers = False
        run_vectorise = False
        log.info(
            "Vector warmup budget capped at %.2fs; deferring heavy stages requiring %.2fs",
            stage_budget_cap,
            heavy_budget_needed,
        )
        if heavy_requested:
            log.info(
                "Heavy vector warmup requested but skipped due to budget cap",
                extra={
                    "event": "vector-warmup-heavy",
                    "requested": True,
                    "budget_cap": stage_budget_cap,
                    "needed": heavy_budget_needed,
                },
            )
        probe_model = False
    elif heavy_requested:
        log.info(
            "Heavy vector warmup explicitly requested",
            extra={
                "event": "vector-warmup-heavy",
                "requested": True,
                "budget_cap": stage_budget_cap,
                "needed": heavy_budget_needed,
            },
        )

    def _has_stage_budget(stage: str) -> bool:
        timeout = resolved_timeouts.get(stage, base_timeouts.get(stage))
        return timeout is not None and timeout > 0

    fast_heavy_allowed = force_heavy and all(
        not flag or _has_stage_budget("handlers")
        for flag in (hydrate_handlers, start_scheduler)
    )
    fast_heavy_allowed = fast_heavy_allowed and (
        not run_vectorise or _has_stage_budget("vectorise")
    )

    if bootstrap_fast and not fast_heavy_allowed:
        if hydrate_handlers:
            log.info(
                "Bootstrap-fast mode deferring handler hydration until heavy warmup with budgets available"
            )
            hydrate_handlers = False
            deferred_bootstrap.add("handlers")
        if start_scheduler:
            log.info(
                "Bootstrap-fast mode deferring scheduler start until heavy warmup with budgets available"
            )
            start_scheduler = False
            deferred_bootstrap.add("scheduler")
        if run_vectorise:
            log.info(
                "Bootstrap-fast mode deferring vectorise warmup until heavy warmup with budgets available"
            )
            run_vectorise = False
            deferred_bootstrap.add("vectorise")
        warmup_lite = True

    def _effective_timeout(stage: str) -> float | None:
        remaining = _remaining_budget()
        stage_timeout = resolved_timeouts.get(stage, base_timeouts.get(stage))
        fallback_budget = provided_budget if provided_budget is not None else None
        if remaining is None:
            timeout = stage_timeout if stage_timeout is not None else fallback_budget
            effective_timeouts[stage] = timeout
            return timeout
        if stage_timeout is None:
            if fallback_budget is None:
                timeout = remaining
            else:
                timeout = max(0.0, min(remaining, fallback_budget))
            effective_timeouts[stage] = timeout
            return timeout
        timeout = max(0.0, min(stage_timeout, remaining))
        effective_timeouts[stage] = timeout
        return timeout

    def _has_estimated_budget(stage: str, *, budget_cap: float | None = None) -> bool:
        remaining = _remaining_budget()
        estimate = base_stage_cost.get(stage)
        if remaining is not None:
            budget_cap = remaining if budget_cap is None else min(remaining, budget_cap)
        if estimate is None:
            return True
        if budget_cap is None:
            return True
        if budget_cap >= estimate:
            return True
        reason = "deferred-estimate" if remaining is not None else "deferred-ceiling"
        _record_deferred(stage, reason)
        log.info(
            "Vector warmup deferring %s; available budget %.2fs below estimated cost %.2fs",
            stage,
            budget_cap,
            estimate,
        )
        return False

    if not _guard("init"):
        log.info("Vector warmup aborted before start: insufficient bootstrap budget")
        return _finalise()
    if _reuse("model"):
        pass
    elif not _guard("model"):
        if _should_abort("model"):
            return _finalise()
    else:
        model_timeout = _effective_timeout("model")
        if download_model:
            if model_timeout is not None and model_timeout <= 0:
                budget_exhausted = True
                _record_deferred("model", "skipped-budget")
                log.info("Vector warmup model download skipped: no remaining budget")
                return _finalise()
            if not _has_estimated_budget("model", budget_cap=model_timeout):
                return _finalise()
            completed, path = _run_with_budget(
                "model",
                lambda stop_event: ensure_embedding_model(
                    logger=log,
                    warmup=True,
                    stop_event=stop_event,
                    budget_check=lambda evt: _cooperative_budget_check(
                        "model", evt
                    ),
                ),
                timeout=model_timeout,
            )
            if completed:
                if path:
                    _record("model", f"ready:{path}" if path.exists() else "ready")
                else:
                    _record("model", "missing")
            elif budget_exhausted:
                if "model" not in summary:
                    _record_deferred("model", "deferred-budget")
                log.info("Vector warmup model download deferred after budget exhaustion")
                return _finalise()
        elif probe_model:
            completed, dest = _run_with_budget(
                "model", lambda _stop: _model_bundle_path()
            )
            if completed and dest:
                if dest.exists():
                    log.info("embedding model already present at %s (probe only)", dest)
                    _record("model", "present")
                else:
                    log.info("embedding model probe: archive missing; will fetch on demand")
                    _record("model", "absent-probe")
        else:
            if "model" in deferred_bootstrap:
                status = "deferred-bootstrap"
                log.info("Skipping embedding model download in bootstrap-lite mode")
            else:
                status = "deferred" if ("model" in deferred or warmup_model) else "skipped"
                log.info("Skipping embedding model download (disabled)")
            _record("model", status)

    svc = None
    if _reuse("handlers"):
        pass
    elif not _guard("handlers"):
        if _should_abort("handlers"):
            return _finalise()
    else:
        handler_timeout = _effective_timeout("handlers")
        if hydrate_handlers:
            if handler_timeout is not None and handler_timeout <= 0:
                budget_exhausted = True
                _record_deferred("handlers", "skipped-budget")
                log.info("Vector warmup handler hydration skipped: no remaining budget")
                return _finalise()
            if not _has_estimated_budget("handlers", budget_cap=handler_timeout):
                return _finalise()
            try:
                from .vectorizer import SharedVectorService

                completed, svc = _run_with_budget(
                    "handlers",
                    lambda stop_event: SharedVectorService(
                        bootstrap_fast=bootstrap_fast,
                        warmup_lite=warmup_lite,
                        stop_event=stop_event,
                        budget_check=lambda evt: _cooperative_budget_check(
                            "handlers", evt
                        ),
                    ),
                    timeout=handler_timeout,
                )
                if completed:
                    _record("handlers", "hydrated")
                elif budget_exhausted:
                    if "handlers" not in summary:
                        _record_deferred("handlers", "deferred-budget")
                    log.info(
                        "Vector warmup handler hydration deferred after budget exhaustion",
                    )
                    return _finalise()
            except Exception as exc:  # pragma: no cover - best effort logging
                log.warning("SharedVectorService warmup failed: %s", exc)
                _record("handlers", "failed")
        else:
            if "handlers" in deferred_bootstrap:
                status = "deferred-bootstrap"
                log.info("Vector handler hydration deferred for bootstrap-lite")
                _record("handlers", status)
            elif "handlers" in lite_deferrals:
                status = "deferred-lite"
                log.info("Vector handler hydration deferred for warmup-lite")
                _record_background("handlers", status)
            else:
                status = "deferred" if ("handlers" in deferred or warmup_handlers) else "skipped"
                log.info("Vector handler hydration skipped")
                _record("handlers", status)

    if _reuse("scheduler"):
        pass
    elif _guard("scheduler"):
        if start_scheduler:
            ensure_scheduler_started(logger=log)
            _record("scheduler", "started")
        else:
            if "scheduler" in deferred_bootstrap:
                status = "deferred-bootstrap"
                _record("scheduler", status)
            elif "scheduler" in lite_deferrals:
                status = "deferred-lite"
                _record_background("scheduler", status)
            else:
                status = "skipped"
                _record("scheduler", status)
            log.info(
                "Scheduler warmup %s",
                "deferred for bootstrap-lite"
                if status == "deferred-bootstrap"
                else ("deferred for warmup-lite" if status == "deferred-lite" else "skipped"),
            )

    should_vectorise = run_vectorise if run_vectorise is not None else hydrate_handlers
    if _reuse("vectorise"):
        pass
    elif _guard("vectorise"):
        vectorise_timeout = _effective_timeout("vectorise")
        if should_vectorise and svc is not None:
            if vectorise_timeout is not None and vectorise_timeout <= 0:
                budget_exhausted = True
                _record_deferred("vectorise", "skipped-budget")
                log.info("Vectorise warmup skipped: no remaining budget")
            elif _has_estimated_budget("vectorise", budget_cap=vectorise_timeout):
                try:
                    completed, _ = _run_with_budget(
                        "vectorise",
                        lambda stop_event: svc.vectorise(
                            "text", {"text": "warmup"}, stop_event=stop_event
                        ),
                        timeout=vectorise_timeout,
                    )
                    if completed:
                        _record("vectorise", "ok")
                except Exception:  # pragma: no cover - allow partial warmup
                    log.debug("vector warmup transform failed; continuing", exc_info=True)
                    _record("vectorise", "failed")
        else:
            if should_vectorise:
                if "vectorise" in deferred_bootstrap:
                    status = "deferred-bootstrap"
                    log.info("Vectorise warmup deferred for bootstrap-lite")
                    _record("vectorise", status)
                elif "vectorise" in lite_deferrals:
                    status = "deferred-lite"
                    log.info("Vectorise warmup deferred for warmup-lite")
                    _record_background("vectorise", status)
                else:
                    status = "deferred" if ("vectorise" in deferred or "handlers" in deferred) else "skipped-no-service"
                    log.info("Vectorise warmup skipped: service unavailable")
                    _record("vectorise", status)
            else:
                if "vectorise" in deferred_bootstrap:
                    status = "deferred-bootstrap"
                elif "vectorise" in lite_deferrals:
                    status = "deferred-lite"
                    _record_background("vectorise", status)
                    log.info("Vectorise warmup deferred for warmup-lite")
                else:
                    status = "deferred" if "vectorise" in deferred else "skipped"
                    _record("vectorise", status)
                if status == "deferred-bootstrap":
                    log.info("Vectorise warmup deferred for bootstrap-lite")
                log.info("Vectorise warmup skipped")

    return _finalise()


__all__ = ["ensure_embedding_model", "ensure_scheduler_started", "warmup_vector_service"]
