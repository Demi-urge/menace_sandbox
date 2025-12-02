from __future__ import annotations

"""Lazy bootstrap helpers for vector service assets.

This module centralises deferred initialisation for heavyweight resources such
as the bundled embedding model and the embedding scheduler.  Callers can either
rely on the on-demand helpers (which cache results) or invoke the warmup
routine to pre-populate caches before the first real request.
"""

import ctypes
import importlib.util
import json
import logging
import os
import tempfile
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
_WARMUP_CACHE_LOADED = False
_PROCESS_START = int(time.time())

_CONSERVATIVE_STAGE_TIMEOUTS = {
    "model": 9.0,
    "handlers": 9.0,
    "vectorise": 4.5,
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


def _coerce_timeout(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _warmup_cache_path() -> Path:
    base_dir = os.getenv("VECTOR_WARMUP_CACHE_DIR", "").strip()
    base = Path(base_dir) if base_dir else Path(tempfile.gettempdir())
    return base / f"vector_warmup_{os.getpid()}_{_PROCESS_START}.json"


def _load_warmup_cache(logger: logging.Logger) -> None:
    global _WARMUP_CACHE_LOADED
    if _WARMUP_CACHE_LOADED:
        return
    _WARMUP_CACHE_LOADED = True
    cache_path = _warmup_cache_path()
    try:
        content = cache_path.read_text()
    except FileNotFoundError:
        return
    except Exception:  # pragma: no cover - advisory cache
        logger.debug("Failed reading warmup cache", exc_info=True)
        return
    try:
        cached = json.loads(content)
    except Exception:  # pragma: no cover - advisory cache
        logger.debug("Invalid warmup cache content", exc_info=True)
        return
    if not isinstance(cached, dict):
        return
    for stage, status in cached.items():
        if isinstance(stage, str) and isinstance(status, str):
            _WARMUP_STAGE_MEMO.setdefault(stage, status)


def _persist_warmup_cache(logger: logging.Logger) -> None:
    cache_path = _warmup_cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(_WARMUP_STAGE_MEMO))
    except Exception:  # pragma: no cover - advisory cache
        logger.debug("Failed persisting warmup cache", exc_info=True)


def _clear_warmup_cache() -> None:
    global _WARMUP_CACHE_LOADED
    _WARMUP_CACHE_LOADED = False
    _WARMUP_STAGE_MEMO.clear()
    cache_path = _warmup_cache_path()
    try:
        cache_path.unlink()
    except FileNotFoundError:
        pass
    except Exception:  # pragma: no cover - advisory cache
        logging.getLogger(__name__).debug("Failed clearing warmup cache", exc_info=True)


def _model_bundle_path() -> Path:
    return resolve_path("vector_service/minilm/tiny-distilroberta-base.tar.xz")


def ensure_embedding_model(
    *,
    logger: logging.Logger | None = None,
    warmup: bool = False,
    warmup_lite: bool = False,
    stop_event: threading.Event | None = None,
    budget_check: Callable[[threading.Event | None], None] | None = None,
    download_timeout: float | None = None,
) -> Path | tuple[Path | None, str | None] | None:
    """Ensure the bundled embedding model archive exists.

    The download is performed at most once per process and only when the model
    is missing.  When ``warmup`` is False the function favours fast failure so
    first-use callers can fall back gracefully; during warmup we log and swallow
    errors to avoid breaking bootstrap flows.  When ``warmup_lite`` is True the
    function performs a presence probe only and defers the download if the
    bundle is absent, returning a ``(path, status)`` tuple so callers can
    propagate the deferral state.
    """

    global _MODEL_READY
    log = logger or logging.getLogger(__name__)

    def _stage_budget_deadline() -> float | None:
        deadline = getattr(stop_event, "_stage_deadline", None)
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0.0
        return remaining

    def _timebox_error(reason: str, timeout_hint: float | None) -> TimeoutError:
        err = TimeoutError(reason)
        setattr(err, "_warmup_timebox", True)
        if timeout_hint is not None:
            setattr(err, "_timebox_timeout", timeout_hint)
        return err

    def _result(path: Path | None, status: str | None = None):
        if warmup_lite:
            return path, status
        return path

    stage_budget_remaining = _stage_budget_deadline()
    effective_timeout = _coerce_timeout(download_timeout)
    if stage_budget_remaining is not None:
        if stage_budget_remaining <= 0:
            raise _timebox_error("embedding model stage budget exhausted", 0.0)
        if effective_timeout is None:
            effective_timeout = stage_budget_remaining
        else:
            effective_timeout = max(0.0, min(effective_timeout, stage_budget_remaining))

    def _check_cancelled(context: str) -> None:
        if stop_event is not None and stop_event.is_set():
            raise TimeoutError(f"embedding model download cancelled during {context}")
        remaining_budget = _stage_budget_deadline()
        if remaining_budget is not None and remaining_budget <= 0:
            raise _timebox_error(
                f"embedding model download timed out during {context}",
                effective_timeout,
            )
        if budget_check is not None:
            budget_check(stop_event)

    def _handle_timeout(error: TimeoutError) -> tuple[Path | None, str | None] | None:
        if not warmup:
            raise error
        status = "deferred-timebox" if getattr(error, "_warmup_timebox", False) else "deferred-budget"
        timeout_hint = getattr(error, "_timebox_timeout", None)
        log.info(
            "embedding model warmup deferred after cancellation",
            extra={
                "event": "vector-warmup",
                "stage": "model",
                "status": status,
                "timeout": timeout_hint,
            },
        )
        return _result(None, status)

    _check_cancelled("init")

    if _MODEL_READY:
        return _result(_model_bundle_path(), "ready")

    with _MODEL_LOCK:
        if _MODEL_READY:
            return _result(_model_bundle_path(), "ready")
        dest = _model_bundle_path()
        if dest.exists():
            _MODEL_READY = True
            return _result(dest, "ready")

        if warmup_lite:
            status = "deferred-absent-probe"
            log.info(
                "embedding model warmup-lite probe: archive missing; deferring download",
                extra={"event": "vector-warmup", "model_status": status},
            )
            return _result(None, status)

        try:
            _check_cancelled("init")
        except TimeoutError as exc:
            timed_out = _handle_timeout(exc)
            if timed_out is not None:
                return timed_out
            raise

        if importlib.util.find_spec("huggingface_hub") is None:
            log.info(
                "embedding model download skipped (huggingface-hub unavailable); will retry on demand"
            )
            return _result(None, "missing")

        try:
            from . import download_model as _dm

            _check_cancelled("fetch")
            _dm.bundle(
                dest,
                stop_event=stop_event,
                budget_check=budget_check,
                timeout=effective_timeout,
            )
            _MODEL_READY = True
            return _result(dest, "ready")
        except TimeoutError as exc:
            deferred = _handle_timeout(_timebox_error(str(exc), effective_timeout))
            if deferred is not None:
                return deferred
            raise
        except Exception as exc:  # pragma: no cover - best effort during warmup
            log.warning("embedding model bootstrap failed: %s", exc)
            if warmup:
                return _result(None, "failed")
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
    _load_warmup_cache(log)
    stage_timeouts_supplied = stage_timeouts is not None
    env_budget = _coerce_timeout(os.getenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"))
    if env_budget is None:
        env_budget = _coerce_timeout(os.getenv("BOOTSTRAP_VECTOR_STEP_TIMEOUT"))
    if env_budget is None:
        env_budget = _coerce_timeout(os.getenv("MENACE_BOOTSTRAP_TIMEOUT"))
    env_budget = env_budget if env_budget is not None and env_budget > 0 else None

    if stage_timeouts is None and env_budget is None:
        stage_timeouts = dict(_CONSERVATIVE_STAGE_TIMEOUTS)

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
    model_probe_only = False
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
            model_probe_only = True
            download_model = False
            probe_model = True
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
        if download_model:
            model_probe_only = True
            probe_model = True
            lite_deferrals.add("model")
        if download_model or hydrate_handlers or start_scheduler or run_vectorise:
            log.info(
                "Warmup-lite enabled; deferring heavy vector warmup stages (force_heavy to override)",
                extra={
                    "download_model": download_model,
                    "hydrate_handlers": hydrate_handlers,
                    "start_scheduler": start_scheduler,
                    "run_vectorise": run_vectorise,
                },
            )
            if model_probe_only:
                log.info(
                    "Warmup-lite model probe enabled; skipping download thread",
                    extra={"event": "vector-warmup", "model_status": "probe-only"},
                )
        hydrate_handlers = False
        start_scheduler = False
        run_vectorise = False

    summary: dict[str, str] = {"bootstrap": summary_flag, "warmup_lite": str(warmup_lite)}
    explicit_deferred: set[str] = set(deferred_stages or ())
    deferred = explicit_deferred | deferred_bootstrap | lite_deferrals
    memoised_results = dict(_WARMUP_STAGE_MEMO)
    prior_deferred = explicit_deferred | {
        stage for stage, status in memoised_results.items() if status.startswith("deferred")
    }

    recorded_deferred: set[str] = set()
    background_candidates: set[str] = set()
    effective_timeouts: dict[str, float | None] = {}

    background_warmup: set[str] = set()
    background_stage_timeouts: dict[str, float | None] | None = None
    budget_gate_reason: str | None = None
    heavy_admission: str | None = None

    if deferred:
        background_warmup.update(deferred)
        background_candidates.update(deferred)

    def _record(stage: str, status: str) -> None:
        summary[stage] = status
        if status.startswith("deferred"):
            recorded_deferred.add(stage)
        _WARMUP_STAGE_MEMO[stage] = status
        _persist_warmup_cache(log)
        try:
            VECTOR_WARMUP_STAGE_TOTAL.labels(stage, status).inc()
        except Exception:  # pragma: no cover - metrics best effort
            log.debug("failed emitting vector warmup metric", exc_info=True)

    def _record_deferred(stage: str, status: str) -> None:
        _record(stage, status)
        if stage in prior_deferred:
            return
        background_candidates.add(stage)

    def _record_deferred_background(stage: str, status: str) -> None:
        _record_background(stage, status)
        if stage in prior_deferred:
            return

    def _record_background(stage: str, status: str) -> None:
        _record(stage, status)
        if stage in prior_deferred and stage not in explicit_deferred:
            return
        background_warmup.add(stage)
        background_candidates.add(stage)

    def _reuse(stage: str) -> bool:
        if stage in explicit_deferred and not force_heavy:
            status = memoised_results.get(stage, "deferred-explicit")
            if not isinstance(status, str) or not status.startswith("deferred"):
                status = "deferred-explicit"
            _record_background(stage, status)
            memoised_results[stage] = status
            return True
        status = memoised_results.get(stage)
        if status is None:
            return False
        if force_heavy and (
            status.startswith("deferred")
            or status in {"failed", "absent-probe", "skipped-budget"}
        ):
            return False
        summary[stage] = status
        if status.startswith("deferred"):
            recorded_deferred.add(stage)
            background_candidates.add(stage)
            background_warmup.add(stage)
        elif status in {"failed", "absent-probe", "skipped-budget", "skipped-cap"}:
            background_candidates.add(stage)
            background_warmup.add(stage)
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
        nonlocal budget_exhausted, budget_gate_reason
        shared_remaining = _remaining_shared_budget()
        if shared_remaining is not None and shared_remaining <= 0:
            budget_exhausted = True
            status = "deferred-shared-budget"
            _record_deferred_background(stage, status)
            budget_gate_reason = budget_gate_reason or status
            log.info(
                "Vector warmup shared budget exhausted; deferring %s", stage
            )
            return False
        if budget_exhausted:
            status = "deferred-budget"
            _record_deferred_background(stage, status)
            budget_gate_reason = budget_gate_reason or status
            log.info("Vector warmup budget already exhausted; deferring %s", stage)
            return False
        remaining = _remaining_budget()
        if remaining is not None and remaining <= 0:
            budget_exhausted = True
            status = "deferred-budget"
            _record_deferred_background(stage, status)
            budget_gate_reason = budget_gate_reason or status
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
            status = "deferred-budget"
            _record_deferred_background(stage, status)
            budget_gate_reason = budget_gate_reason or status
            log.warning("Vector warmup deadline reached before %s: %s", stage, exc)
            return False

    def _hint_background_budget(stage: str, stage_timeout: float | None = None) -> None:
        nonlocal background_stage_timeouts
        remaining = _remaining_budget()
        if background_stage_timeouts is None and (
            remaining is not None or stage_timeout is not None
        ):
            background_stage_timeouts = {}
        if remaining is not None:
            background_stage_timeouts.setdefault("budget", max(0.0, remaining))
        if stage_timeout is None:
            stage_timeout = resolved_timeouts.get(stage)
        if stage_timeout is not None:
            background_stage_timeouts.setdefault(stage, max(0.0, stage_timeout))

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

    cumulative_elapsed = 0.0

    def _remaining_shared_budget() -> float | None:
        if stage_budget_cap is None:
            return None
        return max(0.0, stage_budget_cap - cumulative_elapsed)

    def _record_elapsed(stage: str, elapsed: float) -> None:
        nonlocal cumulative_elapsed, budget_exhausted
        cumulative_elapsed += max(0.0, elapsed)
        remaining_shared = _remaining_shared_budget()
        if remaining_shared is not None:
            summary[f"shared_budget_remaining_after_{stage}"] = f"{remaining_shared:.3f}"
        if stage_budget_cap is not None and cumulative_elapsed >= stage_budget_cap:
            budget_exhausted = True

    def _defer_handler_chain(status: str, *, stage_timeout: float | None = None) -> None:
        nonlocal hydrate_handlers, run_vectorise, budget_gate_reason
        _record_deferred_background("handlers", status)
        _hint_background_budget("handlers", stage_timeout)
        hydrate_handlers = False
        budget_gate_reason = budget_gate_reason or status
        if run_vectorise:
            _record_deferred_background("vectorise", status)
            _hint_background_budget("vectorise", _effective_timeout("vectorise"))
            run_vectorise = False

    def _finalise() -> Mapping[str, str]:
        deferred_record = deferred | recorded_deferred
        if deferred_record:
            summary["deferred"] = ",".join(sorted(deferred_record))
        if background_candidates:
            summary["background"] = ",".join(sorted(background_candidates))
        if heavy_admission is not None:
            summary["heavy_admission"] = heavy_admission
        for stage, ceiling in stage_budget_ceiling.items():
            summary[f"budget_ceiling_{stage}"] = (
                f"{ceiling:.3f}" if ceiling is not None else "none"
            )
        for stage, timeout in effective_timeouts.items():
            summary[f"budget_{stage}"] = (
                f"{timeout:.3f}" if timeout is not None else "none"
            )
        shared_remaining = _remaining_shared_budget()
        if shared_remaining is not None:
            summary["shared_budget_remaining"] = f"{shared_remaining:.3f}"
        remaining = _remaining_budget()
        if remaining is not None:
            summary["remaining_budget"] = f"{remaining:.3f}"
        hook_dispatched = False
        if budget_gate_reason is not None:
            summary["budget_gate"] = budget_gate_reason
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
    ) -> tuple[bool, Any | None, float]:
        nonlocal budget_exhausted
        stop_event = threading.Event()
        start = time.monotonic()
        stage_deadline = start + timeout if timeout is not None else None
        if stage_deadline is not None:
            setattr(stop_event, "_stage_deadline", stage_deadline)
        if check_budget is None and timeout is None:
            result = func(stop_event)
            return True, result, time.monotonic() - start

        result: list[Any | None] = []
        error: list[BaseException] = []
        done = threading.Event()

        def _force_terminate_thread(thread: threading.Thread) -> bool:
            ident = thread.ident
            if ident is None:
                return False
            try:
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(  # type: ignore[attr-defined]
                    ctypes.c_long(ident), ctypes.py_object(SystemExit)
                )
            except Exception:
                return False
            if res > 1:
                try:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(  # type: ignore[attr-defined]
                        ctypes.c_long(ident), None
                    )
                except Exception:
                    pass
                return False
            return res == 1

        def _stop_thread(reason: str) -> None:
            stop_event.set()
            join_deadline = time.monotonic() + 1.0
            while thread.is_alive() and time.monotonic() < join_deadline:
                thread.join(timeout=0.05)
            if thread.is_alive():
                forced = _force_terminate_thread(thread)
                if forced:
                    log.warning(
                        "Vector warmup %s thread forcibly terminated after %s", stage, reason
                    )
                else:
                    log.warning(
                        "Vector warmup %s thread still active after %s despite stop signal",
                        stage,
                        reason,
                    )
            done.set()

        def _runner() -> None:
            try:
                result.append(func(stop_event))
            except BaseException as exc:  # pragma: no cover - propagated to caller
                error.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        while not done.wait(timeout=0.05):
            if timeout is not None and time.monotonic() - start >= timeout:
                _stop_thread("timeout")
                _record_timeout(stage)
                budget_exhausted = True
                _record_deferred_background(stage, "deferred-timeout")
                log.warning(
                    "Vector warmup %s timed out after %.2fs; deferring", stage, timeout
                )
                return False, None, time.monotonic() - start

            if check_budget is not None:
                try:
                    check_budget()
                except TimeoutError as exc:
                    _stop_thread("budget deadline")
                    budget_exhausted = True
                    _record_timeout(stage)
                    _record_deferred_background(stage, "deferred-budget")
                    log.warning("Vector warmup deadline reached during %s: %s", stage, exc)
                    return False, None, time.monotonic() - start

        if error:
            err = error[0]
            if isinstance(err, TimeoutError):
                _stop_thread("cancelled")
                budget_exhausted = True
                timeboxed = getattr(err, "_warmup_timebox", False)
                status = "deferred-timebox" if timeboxed else "deferred-budget"
                timeout_hint = getattr(err, "_timebox_timeout", None)
                if timeboxed:
                    _record_timeout(stage)
                log.info(
                    "Vector warmup %s cancelled%s: %s",
                    stage,
                    " after timebox" if timeboxed else "",
                    err,
                    extra={
                        "event": "vector-warmup",
                        "stage": stage,
                        "status": status,
                        "timeout": timeout_hint,
                    },
                )
                _record_deferred_background(stage, status)
                return False, None, time.monotonic() - start
            raise err

        return True, result[0] if result else None, time.monotonic() - start

    base_timeouts = {
        "model": 8.0,
        "handlers": 9.0,
        "vectorise": 4.0,
    }
    base_stage_cost = {"model": 20.0, "handlers": 25.0, "vectorise": 8.0, "scheduler": 5.0}
    if bootstrap_context or bootstrap_fast or not stage_timeouts_supplied:
        base_timeouts = dict(_CONSERVATIVE_STAGE_TIMEOUTS)

    provided_budget = _coerce_timeout(stage_timeouts) if not isinstance(stage_timeouts, Mapping) else None
    initial_budget_remaining = _remaining_budget()
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

    def _distribute_budget(timeouts: dict[str, float | None], budget: float | None) -> dict[str, float | None]:
        if budget is None:
            return timeouts

        explicit_total = sum(
            value for key, value in timeouts.items() if key in explicit_timeouts and value is not None
        )
        remaining_budget = budget - explicit_total

        if remaining_budget <= 0:
            for stage in timeouts:
                if stage not in explicit_timeouts:
                    timeouts[stage] = 0.0
            return timeouts

        weights = {
            stage: base_stage_cost.get(stage, 1.0)
            for stage in timeouts
            if stage not in explicit_timeouts
        }
        weight_total = sum(weights.values())
        if weight_total <= 0:
            return timeouts

        for stage, weight in weights.items():
            share = max(0.0, remaining_budget * (weight / weight_total))
            timeouts[stage] = share
        return timeouts

    bootstrap_budget_cap = _remaining_budget() if bootstrap_lite else None
    if bootstrap_budget_cap is not None:
        provided_budget = (
            bootstrap_budget_cap
            if provided_budget is None
            else min(provided_budget, bootstrap_budget_cap)
        )

    if (
        provided_budget is None
        and not stage_timeouts_supplied
        and initial_budget_remaining is not None
    ):
        provided_budget = initial_budget_remaining

    resolved_timeouts = _distribute_budget(resolved_timeouts, provided_budget)
    stage_budget_ceiling = {stage: resolved_timeouts.get(stage) for stage in base_timeouts}

    def _below_conservative_budget(stage: str) -> bool:
        threshold = _CONSERVATIVE_STAGE_TIMEOUTS.get(stage)
        ceiling = stage_budget_ceiling.get(stage)
        return threshold is not None and ceiling is not None and ceiling < threshold

    if hydrate_handlers and _below_conservative_budget("handlers"):
        _record_deferred_background("handlers", "deferred-ceiling")
        hydrate_handlers = False
        if run_vectorise:
            _record_deferred_background("vectorise", "deferred-ceiling")
            run_vectorise = False
    elif run_vectorise and _below_conservative_budget("vectorise"):
        _record_deferred_background("vectorise", "deferred-ceiling")
        run_vectorise = False

    def _launch_background_warmup(stages: set[str]) -> None:
        if not stages:
            return

        if background_stage_timeouts is not None:
            background_timeouts: dict[str, float | None] = dict(background_stage_timeouts)
        else:
            background_timeouts = {
                stage: stage_budget_ceiling.get(stage, resolved_timeouts.get(stage))
                for stage in stages
            }
            if provided_budget is not None and (
                stage_budget_cap is None or stage_budget_cap > 0
            ):
                background_timeouts["budget"] = provided_budget
            elif stage_budget_cap is not None and stage_budget_cap <= 0:
                log.info(
                    "Skipping background warmup launch; shared budget cap exhausted",
                    extra={"event": "vector-warmup", "stage": ",".join(sorted(stages))},
                )
                return

        def _run_background() -> None:
            try:
                for stage in stages:
                    _WARMUP_STAGE_MEMO.pop(stage, None)
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
    if stage_budget_cap is None and provided_budget is not None:
        stage_budget_cap = provided_budget
    heavy_budget_needed = 0.0
    if download_model:
        heavy_budget_needed += base_stage_cost["model"]
    if hydrate_handlers:
        heavy_budget_needed += base_stage_cost["handlers"]
    if start_scheduler:
        heavy_budget_needed += base_stage_cost["scheduler"]
    if run_vectorise:
        heavy_budget_needed += base_stage_cost["vectorise"]

    cap_exceeded = stage_budget_cap is not None and heavy_budget_needed > stage_budget_cap
    if cap_exceeded:
        capped_stages = {
            stage
            for stage, enabled in (
                ("model", download_model),
                ("handlers", hydrate_handlers),
                ("vectorise", run_vectorise),
            )
            if enabled
        }
        deferred.update(capped_stages)
        warmup_lite = True
        download_model = False
        hydrate_handlers = False
        run_vectorise = False
        budget_gate_reason = "skipped-cap"
        background_stage_timeouts = {
            stage: stage_budget_ceiling.get(stage, resolved_timeouts.get(stage))
            for stage in base_timeouts
        }
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
        for stage in capped_stages:
            _record_background(stage, "skipped-cap")
            memoised_results[stage] = "skipped-cap"
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

    budget_missing_gate = (
        heavy_requested
        and not force_heavy
        and not stage_timeouts_supplied
        and provided_budget is None
        and initial_budget_remaining is None
    )

    if budget_missing_gate:
        budget_gate_reason = "deferred-no-budget"
        if download_model or model_probe_only:
            _record_background("model", budget_gate_reason)
            background_stage_timeouts = background_stage_timeouts or {
                stage: stage_budget_ceiling.get(stage, resolved_timeouts.get(stage))
                for stage in base_timeouts
            }
            download_model = False
            probe_model = False
        if hydrate_handlers:
            _defer_handler_chain(budget_gate_reason, stage_timeout=_effective_timeout("handlers"))
        if run_vectorise:
            _record_deferred_background("vectorise", budget_gate_reason)
            _hint_background_budget("vectorise", _effective_timeout("vectorise"))
            run_vectorise = False
        warmup_lite = True

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

    def _stage_budget_window(stage_timeout: float | None) -> float | None:
        shared_remaining = _remaining_shared_budget()
        if shared_remaining is None:
            return stage_timeout
        if stage_timeout is None:
            return shared_remaining
        return min(shared_remaining, stage_timeout)

    def _should_defer_upfront(
        stage: str, *, stage_timeout: float | None, stage_enabled: bool
    ) -> bool:
        if not stage_enabled:
            return False
        if force_heavy:
            return False

        available_budget = _stage_budget_window(stage_timeout)
        estimate = base_stage_cost.get(stage)

        if warmup_lite:
            _record_background(stage, "deferred-budget")
            log.info("Warmup-lite deferring %s prior to budget guard", stage)
            return True

        if estimate is None or available_budget is None:
            return False

        if available_budget >= estimate:
            return False

        _record_background(stage, "deferred-budget")
        log.info(
            "Vector warmup %s deferred before guard; budget %.2fs below estimate %.2fs",
            stage,
            available_budget,
            estimate,
        )
        return True

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
        _record_deferred_background(stage, reason)
        log.info(
            "Vector warmup deferring %s; available budget %.2fs below estimated cost %.2fs",
            stage,
            budget_cap,
            estimate,
        )
        return False

    def _needs_stage_estimate(stage: str, enabled: bool) -> bool:
        if not enabled:
            return False
        status = memoised_results.get(stage)
        if status is None:
            return True
        if status.startswith("deferred") or status in {
            "failed",
            "absent-probe",
            "skipped-budget",
            "skipped-cap",
        }:
            return True
        return False

    def _shared_budget_preflight() -> None:
        nonlocal hydrate_handlers, start_scheduler, run_vectorise, budget_gate_reason, heavy_admission

        remaining_shared = _remaining_shared_budget()
        if remaining_shared is None:
            heavy_admission = "admitted"
            return

        planned_stages = [
            stage
            for stage, enabled in (
                ("handlers", hydrate_handlers),
                ("scheduler", start_scheduler),
                ("vectorise", run_vectorise if run_vectorise is not None else hydrate_handlers),
            )
            if _needs_stage_estimate(stage, enabled)
        ]

        if not planned_stages:
            heavy_admission = "admitted"
            return

        estimate_total = sum(base_stage_cost.get(stage, 0.0) for stage in planned_stages)
        if remaining_shared >= estimate_total:
            heavy_admission = "admitted"
            return

        heavy_admission = "deferred-shared-budget"
        budget_gate_reason = budget_gate_reason or heavy_admission
        log.info(
            "Vector warmup heavy stages deferred up front; shared budget %.2fs below estimated %.2fs",
            remaining_shared,
            estimate_total,
        )

        if "handlers" in planned_stages:
            _defer_handler_chain(
                heavy_admission,
                stage_timeout=_effective_timeout("handlers"),
            )
            memoised_results["handlers"] = heavy_admission

        if "scheduler" in planned_stages:
            _record_background("scheduler", heavy_admission)
            _hint_background_budget("scheduler", _effective_timeout("scheduler"))
            start_scheduler = False
            memoised_results["scheduler"] = heavy_admission

        if "vectorise" in planned_stages and "handlers" not in planned_stages:
            _record_background("vectorise", heavy_admission)
            _hint_background_budget("vectorise", _effective_timeout("vectorise"))
            run_vectorise = False
            memoised_results["vectorise"] = heavy_admission

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
                _record_deferred_background("model", "skipped-budget")
                log.info("Vector warmup model download skipped: no remaining budget")
                return _finalise()
            if not _has_estimated_budget("model", budget_cap=model_timeout):
                return _finalise()
            completed, path, elapsed = _run_with_budget(
                "model",
                lambda stop_event: ensure_embedding_model(
                    logger=log,
                    warmup=True,
                    warmup_lite=False,
                    stop_event=stop_event,
                    budget_check=lambda evt: _cooperative_budget_check(
                        "model", evt
                    ),
                    download_timeout=model_timeout,
                ),
                timeout=model_timeout,
            )
            _record_elapsed("model", elapsed)
            if completed:
                resolved_path: Path | None
                status: str | None
                if isinstance(path, tuple):
                    resolved_path, status = path
                else:
                    resolved_path, status = path, None

                if status:
                    _record("model", status)
                elif resolved_path:
                    _record(
                        "model",
                        f"ready:{resolved_path}" if resolved_path.exists() else "ready",
                    )
                else:
                    _record("model", "missing")
            elif budget_exhausted:
                if "model" not in summary:
                    _record_deferred("model", "deferred-budget")
                log.info("Vector warmup model download deferred after budget exhaustion")
                return _finalise()
        elif probe_model or model_probe_only:
            def _probe(stop_event: threading.Event) -> tuple[Path | None, str | None] | Path:
                if warmup_lite and model_probe_only and not force_heavy:
                    return ensure_embedding_model(
                        logger=log,
                        warmup=True,
                        warmup_lite=True,
                        stop_event=stop_event,
                    )
                return _model_bundle_path()

            completed, dest, elapsed = _run_with_budget("model", _probe)
            _record_elapsed("model", elapsed)
            if completed and dest:
                probe_path, status = (
                    dest if isinstance(dest, tuple) else (dest, None)
                )
                if probe_path and isinstance(probe_path, Path) and probe_path.exists():
                    log.info("embedding model already present at %s (probe only)", probe_path)
                    _record("model", "present")
                else:
                    status = status or (
                        "deferred-absent-probe" if model_probe_only else "absent-probe"
                    )
                    log.info(
                        "embedding model probe detected absent archive; deferring download",
                        extra={"event": "vector-warmup", "model_status": status},
                    )
                    if status.startswith("deferred"):
                        _record_background("model", status)
                    else:
                        _record("model", status)
        else:
            if "model" in deferred_bootstrap:
                status = "deferred-bootstrap"
                log.info("Skipping embedding model download in bootstrap-lite mode")
            else:
                status = "deferred" if ("model" in deferred or warmup_model) else "skipped"
                log.info("Skipping embedding model download (disabled)")
            if status.startswith("deferred"):
                _record_background("model", status)
            else:
                _record("model", status)

    _shared_budget_preflight()

    svc = None
    if _reuse("handlers"):
        pass
    else:
        handler_timeout = _effective_timeout("handlers")
        if _should_defer_upfront(
            "handlers", stage_timeout=handler_timeout, stage_enabled=hydrate_handlers
        ):
            pass
        elif not _guard("handlers"):
            handler_status = summary.get("handlers", "deferred-budget")
            if handler_status.startswith("skipped"):
                handler_status = handler_status.replace("skipped", "deferred", 1)
            _defer_handler_chain(handler_status, stage_timeout=handler_timeout)
            if _should_abort("handlers"):
                return _finalise()
        else:
            if hydrate_handlers:
                if handler_timeout is not None and handler_timeout <= 0:
                    budget_exhausted = True
                    _defer_handler_chain("deferred-budget", stage_timeout=handler_timeout)
                    log.info(
                        "Vector warmup handler hydration skipped: no remaining budget"
                    )
                    return _finalise()
                if not _has_estimated_budget("handlers", budget_cap=handler_timeout):
                    _defer_handler_chain(
                        summary.get("handlers", "deferred-budget"),
                        stage_timeout=handler_timeout,
                    )
                    return _finalise()
                try:
                    from .vectorizer import SharedVectorService

                    completed, svc, elapsed = _run_with_budget(
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
                    _record_elapsed("handlers", elapsed)
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
                    _record_background("handlers", status)
                elif "handlers" in lite_deferrals:
                    status = "deferred-lite"
                    log.info("Vector handler hydration deferred for warmup-lite")
                    _record_background("handlers", status)
                else:
                    status = "deferred" if ("handlers" in deferred or warmup_handlers) else "skipped"
                    log.info("Vector handler hydration skipped")
                    if status.startswith("deferred"):
                        _record_background("handlers", status)
                    else:
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
                _record_background("scheduler", status)
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
    else:
        vectorise_timeout = _effective_timeout("vectorise")
        if _should_defer_upfront(
            "vectorise", stage_timeout=vectorise_timeout, stage_enabled=should_vectorise
        ):
            pass
        elif _guard("vectorise"):
            if should_vectorise and svc is not None:
                if vectorise_timeout is not None and vectorise_timeout <= 0:
                    budget_exhausted = True
                    _record_deferred_background("vectorise", "deferred-budget")
                    _hint_background_budget("vectorise", vectorise_timeout)
                    log.info("Vectorise warmup skipped: no remaining budget")
                elif _has_estimated_budget("vectorise", budget_cap=vectorise_timeout):
                    try:
                        from governed_embeddings import (
                            apply_bootstrap_timeout_caps,
                            get_embedder,
                        )

                        embedder_timeout = (
                            vectorise_timeout
                            if vectorise_timeout is not None
                            else apply_bootstrap_timeout_caps()
                        )

                        def _vectorise(stop_event: threading.Event) -> Any:
                            embedder = get_embedder(
                                timeout=embedder_timeout,
                                bootstrap_timeout=embedder_timeout,
                                bootstrap_mode=True,
                                stop_event=stop_event,
                            )
                            placeholder_reason = getattr(
                                embedder, "_placeholder_reason", None
                            )
                            if embedder is None or placeholder_reason in {
                                "timeout",
                                "stop_requested",
                                "bootstrap_cancelled",
                            }:
                                if stop_event is not None:
                                    stop_event.set()
                                raise TimeoutError("embedder warmup deferred")
                            return svc.vectorise(
                                "text", {"text": "warmup"}, stop_event=stop_event
                            )

                        completed, _, elapsed = _run_with_budget(
                            "vectorise",
                            _vectorise,
                            timeout=vectorise_timeout,
                        )
                        _record_elapsed("vectorise", elapsed)
                        if completed:
                            _record("vectorise", "ok")
                    except TimeoutError:
                        _record_deferred_background("vectorise", "deferred-embedder")
                        log.info(
                            "Vector warmup vectorise stage deferred after embedder budget exhaustion",
                        )
                    except Exception:  # pragma: no cover - allow partial warmup
                        log.debug("vector warmup transform failed; continuing", exc_info=True)
                        _record("vectorise", "failed")
                else:
                    _hint_background_budget("vectorise", vectorise_timeout)
        else:
            _hint_background_budget("vectorise", vectorise_timeout)
            if should_vectorise:
                if "vectorise" in deferred_bootstrap:
                    status = "deferred-bootstrap"
                    log.info("Vectorise warmup deferred for bootstrap-lite")
                    _record_background("vectorise", status)
                elif "vectorise" in lite_deferrals:
                    status = "deferred-lite"
                    log.info("Vectorise warmup deferred for warmup-lite")
                    _record_background("vectorise", status)
                else:
                    status = (
                        "deferred"
                        if ("vectorise" in deferred or "handlers" in deferred)
                        else "skipped-no-service"
                    )
                    log.info("Vectorise warmup skipped: service unavailable")
                    if status.startswith("deferred"):
                        _record_background("vectorise", status)
                    else:
                        _record("vectorise", status)
            else:
                if "vectorise" in deferred_bootstrap:
                    status = "deferred-bootstrap"
                elif "vectorise" in lite_deferrals:
                    status = "deferred-lite"
                else:
                    status = "deferred" if "vectorise" in deferred else "skipped"
                if status.startswith("deferred"):
                    _record_background("vectorise", status)
                else:
                    _record("vectorise", status)
                if status == "deferred-bootstrap":
                    log.info("Vectorise warmup deferred for bootstrap-lite")
            log.info("Vectorise warmup skipped")

    return _finalise()


__all__ = ["ensure_embedding_model", "ensure_scheduler_started", "warmup_vector_service"]
