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
_MODEL_BACKGROUND_LOCK = threading.Lock()
_MODEL_BACKGROUND_THREAD: threading.Thread | None = None
_SCHEDULER_LOCK = threading.Lock()
_SCHEDULER: Any | None | bool = None  # False means attempted and unavailable
_WARMUP_STAGE_MEMO: dict[str, str] = {}
_WARMUP_STAGE_META: dict[str, dict[str, object]] = {}
_WARMUP_CACHE_LOADED = False
_PROCESS_START = int(time.time())

_CONSERVATIVE_STAGE_TIMEOUTS = {
    "model": 7.5,
    "handlers": 7.5,
    "scheduler": 4.5,
    "vectorise": 5.0,
}

_BOOTSTRAP_STAGE_TIMEOUT = 3.0
_HEAVY_STAGE_CEILING = 30.0
_BACKGROUND_QUEUE_FLAG = "queued"

VECTOR_WARMUP_STAGE_TOTAL = getattr(
    _metrics,
    "vector_warmup_stage_total",
    _metrics.Gauge(
        "vector_warmup_stage_total",
        "Vector warmup stage results by status",
        ["stage", "status"],
    ),
)


def _update_warmup_stage_cache(
    stage: str,
    status: str,
    logger: logging.Logger,
    *,
    meta: Mapping[str, object] | None = None,
    emit_metric: bool = True,
) -> None:
    meta_payload = _WARMUP_STAGE_META.setdefault(stage, {})
    if meta:
        meta_payload.update(meta)
    _WARMUP_STAGE_MEMO[stage] = status
    try:
        _persist_warmup_cache(logger)
    except Exception:  # pragma: no cover - advisory cache
        logger.debug("Failed persisting warmup cache for %s", stage, exc_info=True)

    if not emit_metric:
        return

    try:
        VECTOR_WARMUP_STAGE_TOTAL.labels(stage, status).inc()
    except Exception:  # pragma: no cover - metrics best effort
        logger.debug("failed emitting vector warmup metric", exc_info=True)


def _coerce_timeout(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _warmup_cache_path() -> Path:
    base_dir = os.getenv("VECTOR_WARMUP_CACHE_DIR", "").strip()
    base = Path(base_dir) if base_dir else Path(tempfile.gettempdir()) / "menace"
    return base / "vector_warmup_cache.json"


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
    if isinstance(cached, dict) and "stages" in cached:
        cached = cached.get("stages")
    if not isinstance(cached, dict):
        return
    for stage, payload in cached.items():
        if isinstance(payload, str):
            status = payload
            meta: dict[str, object] = {"status": status}
        elif isinstance(payload, dict):
            status = payload.get("status") if isinstance(payload.get("status"), str) else None
            meta = dict(payload)
        else:
            continue
        if isinstance(stage, str) and isinstance(status, str):
            _WARMUP_STAGE_MEMO.setdefault(stage, status)
            _WARMUP_STAGE_META.setdefault(stage, meta)


def _persist_warmup_cache(logger: logging.Logger) -> None:
    cache_path = _warmup_cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot: dict[str, dict[str, object]] = {}
        now = time.time()
        for stage, status in _WARMUP_STAGE_MEMO.items():
            meta = dict(_WARMUP_STAGE_META.get(stage, {}))
            meta.setdefault("recorded_at", now)
            meta["updated_at"] = now
            meta["status"] = status
            meta["source_pid"] = os.getpid()
            snapshot[stage] = meta
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps({"version": 1, "stages": snapshot, "persisted_at": now})
        )
        os.replace(tmp_path, cache_path)
    except Exception:  # pragma: no cover - advisory cache
        logger.debug("Failed persisting warmup cache", exc_info=True)


def _clear_warmup_cache() -> None:
    global _WARMUP_CACHE_LOADED
    _WARMUP_CACHE_LOADED = False
    _WARMUP_STAGE_MEMO.clear()
    _WARMUP_STAGE_META.clear()
    cache_path = _warmup_cache_path()
    try:
        cache_path.unlink()
    except FileNotFoundError:
        pass
    except Exception:  # pragma: no cover - advisory cache
        logging.getLogger(__name__).debug("Failed clearing warmup cache", exc_info=True)


def _model_bundle_path() -> Path:
    return resolve_path("vector_service/minilm/tiny-distilroberta-base.tar.xz")


def _note_model_background(
    state: str,
    logger: logging.Logger,
    *,
    emit_metric: bool = False,
    timeout: float | None = None,
) -> None:
    _update_warmup_stage_cache(
        "model",
        _WARMUP_STAGE_MEMO.get("model", "deferred"),
        logger,
        meta={
            "background_state": state,
            "background_updated_at": time.time(),
            "background_timeout": timeout,
        },
        emit_metric=emit_metric,
    )


def _queue_background_model_download(
    logger: logging.Logger,
    *,
    download_timeout: float | None = None,
    force_heavy: bool = False,
) -> None:
    global _MODEL_BACKGROUND_THREAD, _MODEL_READY

    with _MODEL_BACKGROUND_LOCK:
        if _MODEL_READY:
            return
        if _model_bundle_path().exists():
            _MODEL_READY = True
            _update_warmup_stage_cache(
                "model",
                "ready",
                logger,
                meta={"background_state": "complete"},
            )
            return

        if _MODEL_BACKGROUND_THREAD is not None and _MODEL_BACKGROUND_THREAD.is_alive():
            _note_model_background("running", logger, timeout=_coerce_timeout(download_timeout))
            return

        effective_timeout = _coerce_timeout(download_timeout)
        if effective_timeout is not None and effective_timeout <= 0:
            _update_warmup_stage_cache(
                "model",
                "deferred-timebox",
                logger,
                meta={
                    "background_state": "skipped",
                    "background_timeout": effective_timeout,
                },
            )
            return

        _note_model_background("queued", logger, timeout=effective_timeout)

        def _background_download() -> None:
            global _MODEL_BACKGROUND_THREAD
            _note_model_background("running", logger, timeout=effective_timeout)
            try:
                stop_event = threading.Event()
                if effective_timeout is not None:
                    setattr(stop_event, "_stage_deadline", time.monotonic() + effective_timeout)
                path = ensure_embedding_model(
                    logger=logger,
                    warmup=True,
                    warmup_lite=not force_heavy,
                    warmup_heavy=force_heavy,
                    stop_event=stop_event,
                    budget_check=None,
                    download_timeout=download_timeout,
                )
                if path:
                    _MODEL_READY = True
                    _update_warmup_stage_cache(
                        "model",
                        "ready",
                        logger,
                        meta={"background_state": "complete", "background_timeout": effective_timeout},
                    )
                    return
                _update_warmup_stage_cache(
                    "model",
                    _WARMUP_STAGE_MEMO.get("model", "deferred"),
                    logger,
                    meta={
                        "background_state": "deferred",
                        "background_timeout": effective_timeout,
                    },
                )
            except Exception:  # pragma: no cover - background best effort
                logger.debug("background embedding model download failed", exc_info=True)
                _note_model_background("failed", logger, timeout=effective_timeout)
            finally:
                with _MODEL_BACKGROUND_LOCK:
                    _MODEL_BACKGROUND_THREAD = None

        _MODEL_BACKGROUND_THREAD = threading.Thread(
            target=_background_download, name="vector-model-warmup", daemon=True
        )
        _MODEL_BACKGROUND_THREAD.start()


def ensure_embedding_model(
    *,
    logger: logging.Logger | None = None,
    warmup: bool = False,
    warmup_lite: bool | None = None,
    warmup_heavy: bool = False,
    stop_event: threading.Event | None = None,
    budget_check: Callable[[threading.Event | None], None] | None = None,
    download_timeout: float | None = None,
) -> Path | tuple[Path | None, str | None] | None:
    """Ensure the bundled embedding model archive exists.

    The download is performed at most once per process and only when the model
    is missing.  When ``warmup`` is False the function favours fast failure so
    first-use callers can fall back gracefully; during warmup we log and swallow
    errors to avoid breaking bootstrap flows.  Warmup callers default to the
    "lite" behaviour unless ``warmup_heavy`` is True.  When ``warmup_lite`` is
    True the function performs a presence probe only and defers the download if
    the bundle is absent, returning a ``(path, status)`` tuple so callers can
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

    warmup_lite_enabled = warmup_lite
    if warmup and not warmup_heavy:
        warmup_lite_enabled = True

    def _result(path: Path | None, status: str | None = None):
        if warmup_lite_enabled:
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
        _queue_background_model_download(
            log, download_timeout=effective_timeout, force_heavy=False
        )
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
            _update_warmup_stage_cache(
                "model", "ready", log, meta={"background_state": "complete"}
            )
            return _result(dest, "ready")

        if warmup_lite_enabled:
            status = "deferred-absent-probe"
            log.info(
                "embedding model warmup-lite probe: archive missing; deferring download",
                extra={"event": "vector-warmup", "model_status": status},
            )
            _update_warmup_stage_cache("model", status, log, meta={"probe_only": True})
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
            _update_warmup_stage_cache(
                "model", "ready", log, meta={"background_state": "complete"}
            )
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
    run_vectorise: bool | None = False,
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
    background_hook: Callable[[set[str]], None] | Callable[[set[str], Mapping[str, float | None] | None], None] | None = None,
    bootstrap_lite: bool | None = None,
) -> Mapping[str, str]:
    """Eagerly initialise vector assets and caches.

    The default behaviour favours a "light" warmup that validates scheduler
    configuration and optional model presence without instantiating
    ``SharedVectorService``.  Callers may opt-in to handler hydration and
    vectorisation by setting ``hydrate_handlers=True`` and opting into
    ``run_vectorise=True`` when a heavier warmup is desired.  ``warmup_lite``
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
    deferred for background execution (and optional ``budget_hints`` per stage)
    so callers can enqueue follow-up tasks with the same ceilings.  A
    ``bootstrap_lite`` flag allows bootstrap callers to explicitly defer handler
    hydration and vectorisation while still probing model presence, even when
    generous budgets are available.  When neither environment nor caller
    timeouts are supplied, the conservative per-stage defaults are applied
    automatically and enforced before heavy work begins to avoid uncapped
    warmups.
    """

    log = logger or logging.getLogger(__name__)
    _load_warmup_cache(log)
    budget_remaining_supplied = budget_remaining is not None
    check_budget_supplied = check_budget is not None
    stage_timeouts_supplied = stage_timeouts is not None

    def _normalise_stage_timeouts(
        value: dict[str, float] | float | bool | None,
    ) -> dict[str, float] | float | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, bool):
            return dict(_CONSERVATIVE_STAGE_TIMEOUTS) if value else None
        numeric_timeout = _coerce_timeout(value)
        if numeric_timeout is None:
            return dict(_CONSERVATIVE_STAGE_TIMEOUTS)
        return numeric_timeout
    env_budget = _coerce_timeout(os.getenv("MENACE_BOOTSTRAP_VECTOR_WAIT_SECS"))
    if env_budget is None:
        env_budget = _coerce_timeout(os.getenv("BOOTSTRAP_VECTOR_STEP_TIMEOUT"))
    if env_budget is None:
        env_budget = _coerce_timeout(os.getenv("MENACE_BOOTSTRAP_TIMEOUT"))
    env_budget = env_budget if env_budget is not None and env_budget > 0 else None

    bootstrap_context = any(
        os.getenv(flag, "").strip().lower() in {"1", "true", "yes", "on"}
        for flag in ("MENACE_BOOTSTRAP", "MENACE_BOOTSTRAP_FAST", "MENACE_BOOTSTRAP_MODE")
    )

    warmup_requested = any(
        flag
        for flag in (
            download_model,
            probe_model,
            hydrate_handlers,
            start_scheduler,
            bool(run_vectorise),
            warmup_lite,
        )
    )

    stage_timeouts = _normalise_stage_timeouts(stage_timeouts)

    if stage_timeouts is None:
        if bootstrap_context or warmup_requested:
            stage_timeouts = dict(_CONSERVATIVE_STAGE_TIMEOUTS)
            if env_budget is not None:
                stage_timeouts = dict(stage_timeouts)
                stage_timeouts["budget"] = env_budget
        elif env_budget is None:
            stage_timeouts = dict(_CONSERVATIVE_STAGE_TIMEOUTS)

    budget_start = time.monotonic()
    timebox_deadline: float | None = None

    def _default_budget_remaining() -> float | None:
        if env_budget is None:
            return None
        remaining = env_budget - (time.monotonic() - budget_start)
        return max(0.0, remaining)

    def _timebox_remaining() -> float | None:
        if timebox_deadline is None:
            return None
        return max(0.0, timebox_deadline - time.monotonic())

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
    if stage_timeouts is None:
        stage_timeouts = dict(_CONSERVATIVE_STAGE_TIMEOUTS)

    bootstrap_guard_ceiling: float | None = None
    if bootstrap_context and env_budget is None and not stage_timeouts_supplied:
        bootstrap_guard_ceiling = _BOOTSTRAP_STAGE_TIMEOUT

    if bootstrap_fast is None:
        bootstrap_fast = bootstrap_context
    bootstrap_fast = bool(bootstrap_fast)
    bootstrap_lite = bool(bootstrap_context if bootstrap_lite is None else bootstrap_lite)

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

    bootstrap_hard_timebox: float | None = None
    bootstrap_deferred_records: set[str] = set()
    warmup_lite_source = "caller"

    bootstrap_force_lite = bootstrap_context and not force_heavy and bootstrap_lite
    if bootstrap_force_lite and not warmup_lite:
        log.info("Bootstrap context detected; forcing warmup_lite")
        warmup_lite = True
        warmup_lite_source = "bootstrap"

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

    model_probe_only = False
    requested_handlers = bool(hydrate_handlers)
    requested_scheduler = bool(start_scheduler)
    requested_model = bool(download_model or probe_model)
    requested_vectorise = bool(run_vectorise)
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

    if bootstrap_force_lite:
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
            bootstrap_deferred_records.add("model")
            deferred_bootstrap.add("model")
        elif probe_model:
            deferred_bootstrap.add("model")
        if requested_handlers:
            log.info("Bootstrap context detected; deferring handler hydration")
            hydrate_handlers = False
            deferred_bootstrap.add("handlers")
            bootstrap_deferred_records.add("handlers")
        if start_scheduler:
            log.info("Bootstrap context detected; scheduler start deferred")
            start_scheduler = False
            deferred_bootstrap.add("scheduler")
            bootstrap_deferred_records.add("scheduler")
        if requested_vectorise:
            deferred_bootstrap.add("vectorise")
            bootstrap_deferred_records.add("vectorise")
        summary_flag = "deferred-bootstrap"
    else:
        summary_flag = "normal"

    if bootstrap_context and not force_heavy:
        bootstrap_hard_timebox = _BOOTSTRAP_STAGE_TIMEOUT

    if bootstrap_lite and not force_heavy and run_vectorise:
        log.info("Bootstrap-lite enabled; deferring vectorise warmup")
        run_vectorise = False
        bootstrap_deferred_records.add("vectorise")
        deferred_bootstrap.add("vectorise")

    warmup_lite = bool(warmup_lite)
    budget_hooks_missing = not (budget_remaining_supplied or check_budget_supplied)
    if budget_hooks_missing and not force_heavy:
        if warmup_lite_source == "caller":
            warmup_lite_source = "missing-budget-hooks"
        warmup_lite = True
        if heavy_requested or requested_handlers or requested_vectorise:
            log.info(
                "No budget callbacks supplied; enabling warmup-lite and deferring heavy vector warmup",
                extra={
                    "event": "vector-warmup",
                    "warmup_lite": True,
                    "budget_callbacks": "missing",
                },
            )
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
    if warmup_lite_source != "caller":
        summary["warmup_lite_source"] = warmup_lite_source
    explicit_deferred: set[str] = set(deferred_stages or ())
    deferred = explicit_deferred | deferred_bootstrap | lite_deferrals
    memoised_results = dict(_WARMUP_STAGE_MEMO)
    model_background_state = _WARMUP_STAGE_META.get("model", {}).get("background_state")
    prior_deferred = explicit_deferred | {
        stage for stage, status in memoised_results.items() if status.startswith("deferred")
    }

    recorded_deferred: set[str] = set()
    proactive_deferred: set[str] = set()
    background_candidates: set[str] = set()
    effective_timeouts: dict[str, float | None] = {}

    background_warmup: set[str] = set()
    background_stage_timeouts: dict[str, float | None] | None = None
    background_budget_ceiling: dict[str, float | None] = {}
    budget_gate_reason: str | None = None
    heavy_admission: str | None = None
    lazy_sentinel_active = bootstrap_context and not force_heavy

    if deferred:
        background_warmup.update(deferred)
        background_candidates.update(deferred)

    if (
        not force_heavy
        and (
            model_background_state in {"queued", "running"}
            or memoised_results.get("model") in {"deferred-budget", "deferred-timebox"}
        )
    ):
        if download_model:
            log.info(
                "Embedding model download already queued; falling back to probe-only warmup",
                extra={"event": "vector-warmup", "model_status": model_background_state},
            )
        model_probe_only = True
        probe_model = True
        download_model = False

    def _record(stage: str, status: str) -> None:
        current_status = summary.get(stage)
        if current_status and current_status.startswith("deferred") and status.startswith("deferred"):
            priority = {
                "deferred-heavy": 6,
                "deferred-bootstrap": 5,
                "deferred-ceiling": 4,
                "deferred-timebox": 3,
                "deferred-budget": 2,
                "deferred-embedder": 2,
                "deferred-lite": 1,
                "deferred-no-budget": 1,
            }
            if priority.get(current_status, 0) >= priority.get(status, 0):
                return
        summary[stage] = status
        if stage in {"handlers", "vectorise"}:
            if status.startswith("deferred"):
                summary[f"{stage}_queued"] = _BACKGROUND_QUEUE_FLAG
            elif status.startswith("skipped"):
                summary[f"{stage}_skipped"] = status
        if status.startswith("deferred"):
            recorded_deferred.add(stage)
        memoised_results[stage] = status
        _update_warmup_stage_cache(stage, status, log)

    def _record_deferred(stage: str, status: str) -> None:
        _record_background(stage, status)
        if stage in prior_deferred:
            return

    def _record_deferred_background(stage: str, status: str) -> None:
        _record_background(stage, status)
        if stage in prior_deferred:
            return

    def _record_bootstrap_deferrals() -> None:
        for stage in bootstrap_deferred_records:
            _record_background(stage, "deferred-bootstrap")
        for stage in lite_deferrals:
            if stage in bootstrap_deferred_records:
                continue
            _record_background(stage, "deferred-lite")

    def _record_background(stage: str, status: str) -> None:
        _record(stage, status)
        summary[f"{stage}_background"] = status
        if stage == "model" and status.startswith("deferred"):
            _queue_background_model_download(
                log, download_timeout=_effective_timeout(stage)
            )
        background_warmup.add(stage)
        background_candidates.add(stage)
        _hint_background_budget(stage, _effective_timeout(stage))
        if stage in prior_deferred and stage not in explicit_deferred:
            return

    def _record_proactive_deferral(
        stage: str, status: str, *, stage_timeout: float | None
    ) -> None:
        existing_status = memoised_results.get(stage)
        if existing_status and not str(existing_status).startswith("deferred"):
            return
        if summary.get(stage):
            return
        _record_background(stage, status)
        _hint_background_budget(stage, stage_timeout)
        memoised_results[stage] = status
        proactive_deferred.add(stage)

    def _record_lazy_sentinel(
        stage: str,
        *,
        reason: str = "deferred-lazy",
        stage_timeout: float | None = None,
        chain_vectorise: bool = False,
        vectorise_timeout: float | None = None,
    ) -> None:
        nonlocal hydrate_handlers, run_vectorise, download_model, probe_model, model_probe_only

        _record_background(stage, reason)
        _update_warmup_stage_cache(
            stage,
            reason,
            log,
            meta={
                "lazy_sentinel": True,
                "stage_timeout": stage_timeout,
                "recorded_at": time.time(),
            },
            emit_metric=False,
        )
        if chain_vectorise and run_vectorise and stage != "vectorise":
            _record_background("vectorise", reason)
            _hint_background_budget("vectorise", vectorise_timeout)
            run_vectorise = False
            _update_warmup_stage_cache(
                "vectorise",
                reason,
                log,
                meta={"lazy_sentinel": True, "stage_timeout": vectorise_timeout},
                emit_metric=False,
            )
        if stage == "handlers":
            hydrate_handlers = False
        elif stage == "model":
            download_model = False
            probe_model = False
            model_probe_only = False
        elif stage == "vectorise":
            run_vectorise = False

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
    timebox_skipped: set[str] = set()

    stage_budget_cap: float | None = None

    def _shared_budget_remaining() -> float | None:
        if stage_budget_cap is None:
            return None
        elapsed = time.monotonic() - budget_start
        remaining = stage_budget_cap - max(elapsed, cumulative_elapsed)
        return max(0.0, remaining)

    def _remaining_budget() -> float | None:
        shared_remaining = _shared_budget_remaining()
        if budget_remaining is None:
            if shared_remaining is not None:
                return shared_remaining
            return _timebox_remaining()
        try:
            remaining = budget_remaining()
            timebox_remaining = _timebox_remaining()
            candidates = [remaining, shared_remaining, timebox_remaining]
            candidates = [value for value in candidates if value is not None]
            if not candidates:
                return None
            return min(candidates)
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
        timebox_remaining = _timebox_remaining()
        if timebox_remaining is not None and timebox_remaining <= 0:
            budget_exhausted = True
            status = "deferred-timebox"
            timebox_skipped.add(stage)
            _record_deferred_background(stage, status)
            budget_gate_reason = budget_gate_reason or status
            log.info(
                "Vector warmup timebox exhausted before %s; deferring remaining stages", stage
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
        budget_hint = _available_budget_hint(stage, stage_timeout)
        budget_window = _stage_budget_window(stage_timeout)
        shared_remaining = _remaining_shared_budget()
        timebox_remaining = _timebox_remaining()
        shared_budget = (
            shared_remaining
            if shared_remaining is not None
            else (
                max(0.0, stage_budget_cap - cumulative_elapsed)
                if stage_budget_cap is not None
                else None
            )
        )

        if background_stage_timeouts is None:
            background_stage_timeouts = {}

        if shared_budget is not None:
            current_budget = background_stage_timeouts.get("budget")
            shared_budget = max(0.0, shared_budget)
            if current_budget is None:
                background_stage_timeouts["budget"] = shared_budget
            else:
                background_stage_timeouts["budget"] = min(current_budget, shared_budget)

        if timebox_remaining is not None:
            current_budget = background_stage_timeouts.get("budget")
            if current_budget is None:
                background_stage_timeouts["budget"] = max(0.0, timebox_remaining)
            else:
                background_stage_timeouts["budget"] = min(
                    current_budget, max(0.0, timebox_remaining)
                )

        if stage_timeout is None:
            stage_timeout = resolved_timeouts.get(stage)

        effective_ceiling = budget_hint
        if effective_ceiling is None:
            effective_ceiling = budget_window
        if effective_ceiling is None:
            effective_ceiling = stage_timeout

        if effective_ceiling is not None:
            ceiling_value = max(0.0, effective_ceiling)
            background_stage_timeouts[stage] = ceiling_value
            background_budget_ceiling[stage] = ceiling_value

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

    def _stage_gate_timeout(stage: str, timeout_hint: float | None) -> TimeoutError:
        err = TimeoutError(f"{stage} warmup exceeded stage budget")
        setattr(err, "_warmup_timebox", True)
        if timeout_hint is not None:
            setattr(err, "_timebox_timeout", timeout_hint)
        return err

    def _start_stage_gate(stage: str, stage_timeout: float | None) -> tuple[float | None, float | None]:
        gate_budget = _available_budget_hint(stage, stage_timeout)
        if gate_budget is None:
            return None, None
        return gate_budget, time.monotonic()

    def _apply_stage_gate(
        stage: str,
        gate_budget: float | None,
        gate_start: float | None,
        stage_timeout: float | None,
        *,
        elapsed_hint: float | None = None,
    ) -> None:
        nonlocal budget_exhausted

        if gate_budget is None or gate_start is None:
            return

        elapsed = elapsed_hint if elapsed_hint is not None else time.monotonic() - gate_start
        if elapsed <= gate_budget:
            return

        budget_exhausted = True
        status = "deferred-timebox" if stage_timeout is not None else "deferred-budget"
        _record_deferred_background(stage, status)
        _hint_background_budget(stage, gate_budget)
        _record_timeout(stage)
        raise _stage_gate_timeout(stage, stage_timeout if stage_timeout is not None else gate_budget)

    def _record_cancelled(stage: str, reason: str) -> None:
        summary[f"{stage}_cancelled"] = reason

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

    def _timebox_or_budget_remaining() -> float | None:
        budget_remaining = _remaining_budget()
        if budget_remaining is None:
            return _timebox_remaining()
        return budget_remaining

    def _available_budget_hint(
        stage: str, stage_timeout: float | None = None
    ) -> float | None:
        hints: list[float] = []
        remaining = _remaining_budget()
        shared_remaining = _remaining_shared_budget()
        timebox_remaining = _timebox_remaining()
        if remaining is not None:
            hints.append(remaining)
        if shared_remaining is not None:
            hints.append(shared_remaining)
        if timebox_remaining is not None:
            hints.append(timebox_remaining)
        if stage_timeout is None:
            stage_timeout = resolved_timeouts.get(stage)
        if stage_timeout is not None:
            hints.append(stage_timeout)
        if not hints:
            return None
        return min(hints)

    def _gate_conservative_budget(
        stage: str, stage_enabled: bool, stage_timeout: float | None = None
    ) -> bool:
        nonlocal hydrate_handlers, start_scheduler, run_vectorise, budget_gate_reason
        if not stage_enabled:
            return True
        threshold = _CONSERVATIVE_STAGE_TIMEOUTS.get(stage)
        if threshold is None:
            return True
        available = _available_budget_hint(stage, stage_timeout)
        budget_window = _stage_budget_window(stage_timeout)
        shared_remaining = _remaining_shared_budget()

        shared_conservative = (
            stage in {"handlers", "vectorise"}
            and budget_window is not None
            and threshold is not None
            and budget_window < threshold
            and (shared_remaining is not None or bootstrap_context)
        )
        status = "deferred-ceiling"
        if shared_conservative:
            status = "deferred-shared-budget" if shared_remaining is not None else "deferred-bootstrap-budget"
        elif available is None or available >= threshold:
            return True

        _record_deferred_background(stage, status)
        _hint_background_budget(stage, stage_timeout)
        budget_gate_reason = budget_gate_reason or status
        if stage == "handlers":
            hydrate_handlers = False
            if run_vectorise:
                _record_deferred_background("vectorise", status)
                _hint_background_budget("vectorise", _effective_timeout("vectorise"))
                run_vectorise = False
        elif stage == "scheduler":
            start_scheduler = False
        elif stage == "vectorise":
            run_vectorise = False
        log.info(
            "Remaining budget %.2fs below conservative ceiling for %s; deferring",
            available if available is not None else budget_window,
            stage,
        )
        return False

    def _record_elapsed(stage: str, elapsed: float) -> None:
        nonlocal cumulative_elapsed, budget_exhausted
        cumulative_elapsed += max(0.0, elapsed)
        summary[f"elapsed_{stage}"] = f"{elapsed:.3f}"
        remaining_shared = _remaining_shared_budget()
        if remaining_shared is not None:
            summary[f"shared_budget_remaining_after_{stage}"] = f"{remaining_shared:.3f}"
        if stage_budget_cap is not None and cumulative_elapsed >= stage_budget_cap:
            budget_exhausted = True
        remaining_timebox = _timebox_remaining()
        if remaining_timebox is not None and remaining_timebox <= 0:
            budget_exhausted = True
            timebox_skipped.add(stage)

    def _defer_handler_chain(
        status: str,
        *,
        stage_timeout: float | None = None,
        vectorise_timeout: float | None = None,
    ) -> None:
        nonlocal hydrate_handlers, run_vectorise, budget_gate_reason
        _record_deferred_background("handlers", status)
        _hint_background_budget("handlers", stage_timeout)
        hydrate_handlers = False
        budget_gate_reason = budget_gate_reason or status
        if run_vectorise:
            _record_deferred_background("vectorise", status)
            _hint_background_budget("vectorise", vectorise_timeout)
            run_vectorise = False

    def _finalise() -> Mapping[str, str]:
        nonlocal background_stage_timeouts
        deferred_record = deferred | recorded_deferred
        if deferred_record:
            summary["deferred"] = ",".join(sorted(deferred_record))
        if background_candidates:
            summary["background"] = ",".join(sorted(background_candidates))
        summary["deferred_stages"] = (
            ",".join(sorted(background_candidates | deferred_record))
            if (background_candidates or deferred_record)
            else ""
        )
        if proactive_deferred:
            summary["proactive_deferred"] = ",".join(sorted(proactive_deferred))
        summary["capped_stages"] = ",".join(sorted(capped_stages)) if capped_stages else ""
        if heavy_admission is not None:
            summary["heavy_admission"] = heavy_admission
        for stage, ceiling in stage_budget_ceiling.items():
            summary[f"budget_ceiling_{stage}"] = (
                f"{ceiling:.3f}" if ceiling is not None else "none"
            )
        for stage, ceiling in background_budget_ceiling.items():
            summary[f"background_budget_ceiling_{stage}"] = (
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
        if timebox_deadline is not None:
            summary["warmup_timebox"] = f"{stage_budget_cap:.3f}"
        if timebox_skipped:
            summary["timebox_skipped"] = ",".join(sorted(timebox_skipped))
        hook_dispatched = False
        if budget_gate_reason is not None:
            summary["budget_gate"] = budget_gate_reason
        if background_candidates and (bootstrap_context or bootstrap_fast or warmup_lite):
            log.info(
                "Vector warmup deferrals queued for background completion",
                extra={
                    "event": "vector-warmup",
                    "deferred": ",".join(sorted(background_candidates)),
                    "budget_hints": background_stage_timeouts or {},
                },
            )
        effective_background_hook = background_hook
        if (
            effective_background_hook is None
            and bootstrap_context
            and background_candidates
        ):
            def _queue_background(
                stages: set[str], *, budget_hints: Mapping[str, float | None] | None = None
            ) -> None:
                if budget_hints and background_stage_timeouts is None:
                    background_stage_timeouts = dict(budget_hints)
                _launch_background_warmup(set(stages))

                effective_background_hook = _queue_background

        if background_candidates and effective_background_hook is not None:
            try:
                hook_kwargs = {"budget_hints": background_stage_timeouts or {}}
                hook_code = getattr(effective_background_hook, "__code__", None)
                if hook_code is not None and "budget_hints" in hook_code.co_varnames:
                    effective_background_hook(set(background_candidates), **hook_kwargs)
                else:
                    effective_background_hook(set(background_candidates))
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
    ) -> tuple[bool, Any | None, float, str | None]:
        nonlocal budget_exhausted
        stop_event = threading.Event()
        start = time.monotonic()
        stage_deadline = start + timeout if timeout is not None else None
        if stage_deadline is not None:
            setattr(stop_event, "_stage_deadline", stage_deadline)
        if check_budget is None and timeout is None:
            result = func(stop_event)
            return True, result, time.monotonic() - start, None

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
                _record_deferred_background(stage, "deferred-timebox")
                log.warning(
                    "Vector warmup %s timed out after %.2fs; deferring", stage, timeout
                )
                return False, None, time.monotonic() - start, "timebox"

            if check_budget is not None:
                try:
                    check_budget()
                except TimeoutError as exc:
                    _stop_thread("budget deadline")
                    budget_exhausted = True
                    _record_timeout(stage)
                    _record_deferred_background(stage, "deferred-budget")
                    log.warning("Vector warmup deadline reached during %s: %s", stage, exc)
                    return False, None, time.monotonic() - start, "budget"

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
                return False, None, time.monotonic() - start, "timebox" if timeboxed else "budget"
            raise err

        return True, result[0] if result else None, time.monotonic() - start, None

    base_timeouts = dict(_CONSERVATIVE_STAGE_TIMEOUTS)
    min_vectorise_budget = 0.5
    base_stage_cost = {
        "model": 20.0,
        "handlers": 25.0,
        "vectorise": 8.0,
        "scheduler": 5.0,
    }
    if bootstrap_context or bootstrap_fast or not stage_timeouts_supplied:
        base_timeouts = dict(_CONSERVATIVE_STAGE_TIMEOUTS)

    stage_hard_cap: float | None = None
    if bootstrap_hard_timebox is not None:
        stage_hard_cap = bootstrap_hard_timebox
    elif (bootstrap_fast or warmup_lite) and not force_heavy:
        stage_hard_cap = _BOOTSTRAP_STAGE_TIMEOUT
    elif not force_heavy and not (budget_remaining_supplied or check_budget_supplied):
        stage_hard_cap = _BOOTSTRAP_STAGE_TIMEOUT
    if bootstrap_guard_ceiling is not None:
        stage_hard_cap = (
            bootstrap_guard_ceiling
            if stage_hard_cap is None
            else min(stage_hard_cap, bootstrap_guard_ceiling)
        )

    provided_budget = _coerce_timeout(stage_timeouts) if not isinstance(stage_timeouts, Mapping) else None
    initial_budget_remaining = _remaining_budget()
    resolved_timeouts: dict[str, float | None] = dict(base_timeouts)
    explicit_timeouts: set[str] = set()

    heavy_stage_cap_hits: set[str] = set()

    background_first = warmup_lite or stage_hard_cap is not None

    def _apply_stage_cap(timeouts: dict[str, float | None]) -> None:
        if stage_hard_cap is None:
            return
        for stage, timeout in list(timeouts.items()):
            if timeout is None:
                timeouts[stage] = stage_hard_cap
            else:
                timeouts[stage] = min(timeout, stage_hard_cap)

    def _apply_heavy_stage_cap(timeouts: dict[str, float | None]) -> None:
        for stage in ("handlers", "model", "vectorise"):
            timeout = timeouts.get(stage)
            if timeout is None:
                timeouts[stage] = _HEAVY_STAGE_CEILING
            else:
                capped_timeout = min(timeout, _HEAVY_STAGE_CEILING)
                if capped_timeout != timeout:
                    heavy_stage_cap_hits.add(stage)
                timeouts[stage] = capped_timeout

    def _apply_bootstrap_guard(timeouts: dict[str, float | None]) -> None:
        if bootstrap_guard_ceiling is None:
            return
        for stage in base_timeouts:
            timeout = timeouts.get(stage)
            if timeout is None or timeout > bootstrap_guard_ceiling:
                timeouts[stage] = bootstrap_guard_ceiling

    if isinstance(stage_timeouts, Mapping):
        for name, timeout in stage_timeouts.items():
            if name == "budget":
                provided_budget = _coerce_timeout(timeout)
                continue
            coerced = _coerce_timeout(timeout)
            if coerced is None:
                continue
            target_name = "handlers" if name == "legacy-handlers" else name
            resolved_timeouts[target_name] = coerced
            explicit_timeouts.add(target_name)

    _apply_bootstrap_guard(base_timeouts)
    _apply_bootstrap_guard(resolved_timeouts)
    _apply_stage_cap(base_timeouts)
    _apply_stage_cap(resolved_timeouts)
    _apply_heavy_stage_cap(base_timeouts)
    _apply_heavy_stage_cap(resolved_timeouts)

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
    _apply_stage_cap(resolved_timeouts)
    _apply_heavy_stage_cap(resolved_timeouts)

    if bootstrap_hard_timebox is not None:
        for stage in ("handlers", "vectorise"):
            timeout = resolved_timeouts.get(stage)
            if timeout is None or timeout > bootstrap_hard_timebox:
                resolved_timeouts[stage] = bootstrap_hard_timebox
    stage_budget_ceiling = {stage: resolved_timeouts.get(stage) for stage in base_timeouts}
    capped_stages: set[str] = {
        stage for stage, timeout in stage_budget_ceiling.items() if timeout is not None
    }

    def _below_conservative_budget(stage: str) -> bool:
        threshold = _CONSERVATIVE_STAGE_TIMEOUTS.get(stage)
        ceiling = stage_budget_ceiling.get(stage)
        return threshold is not None and ceiling is not None and ceiling < threshold

    def _insufficient_stage_budget(stage: str) -> bool:
        ceiling = stage_budget_ceiling.get(stage)
        estimate = base_stage_cost.get(stage)
        if ceiling is None or estimate is None:
            return False
        return ceiling < estimate

    if hydrate_handlers and _insufficient_stage_budget("handlers"):
        _defer_handler_chain(
            "deferred-ceiling",
            stage_timeout=stage_budget_ceiling.get("handlers"),
            vectorise_timeout=stage_budget_ceiling.get("vectorise"),
        )
    pending_vectorise = bool(run_vectorise)
    if pending_vectorise and not hydrate_handlers and _insufficient_stage_budget("vectorise"):
        status = "deferred-ceiling"
        _record_deferred_background("vectorise", status)
        _hint_background_budget("vectorise", stage_budget_ceiling.get("vectorise"))
        budget_gate_reason = budget_gate_reason or status
        run_vectorise = False

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
                for stage in stages:
                    _update_warmup_stage_cache(
                        stage,
                        _WARMUP_STAGE_MEMO.get(stage, "deferred-budget"),
                        log,
                        meta={"background_state": "skipped", "background_timeout": stage_budget_cap},
                        emit_metric=False,
                    )
                return

        for stage in stages:
            _update_warmup_stage_cache(
                stage,
                _WARMUP_STAGE_MEMO.get(stage, "deferred"),
                log,
                meta={"background_state": "queued", "background_timeout": background_timeouts.get(stage)},
                emit_metric=False,
            )

        def _run_background() -> None:
            try:
                warmup_vector_service(
                    download_model=download_model,
                    probe_model=probe_model,
                    hydrate_handlers="handlers" in stages,
                    start_scheduler="scheduler" in stages,
                    run_vectorise="vectorise" in stages,
                    check_budget=check_budget,
                    budget_remaining=_remaining_budget,
                    logger=log,
                    force_heavy=False,
                    bootstrap_fast=bootstrap_fast,
                    warmup_lite=warmup_lite,
                    warmup_model=warmup_model,
                    warmup_handlers=True,
                    warmup_probe=warmup_probe,
                    stage_timeouts=background_timeouts,
                    deferred_stages=recorded_deferred or set(),
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
    if stage_budget_cap is None:
        conservative_ceiling = sum(
            timeout for timeout in _CONSERVATIVE_STAGE_TIMEOUTS.values() if timeout is not None
        )
        stage_budget_cap = conservative_ceiling if conservative_ceiling > 0 else None
    if stage_budget_cap is not None:
        initial_remaining = _remaining_budget()
        if initial_remaining is not None:
            stage_budget_cap = min(stage_budget_cap, initial_remaining)
        timebox_deadline = budget_start + stage_budget_cap

    if stage_budget_cap is not None and stage_budget_cap <= 0:
        status = "deferred-budget"
        background_stage_timeouts = background_stage_timeouts or {
            stage: stage_budget_ceiling.get(stage, resolved_timeouts.get(stage))
            for stage in base_timeouts
        }
        for stage, enabled in (
            (
                "model",
                download_model or probe_model or model_probe_only,
            ),
            ("handlers", hydrate_handlers),
            ("scheduler", start_scheduler),
            ("vectorise", bool(run_vectorise)),
        ):
            if not enabled:
                continue
            _record_background(stage, status)
            memoised_results[stage] = status
            if stage == "model":
                download_model = False
                model_probe_only = False
                probe_model = False
            elif stage == "handlers":
                hydrate_handlers = False
                if run_vectorise:
                    _record_background("vectorise", status)
                    memoised_results["vectorise"] = status
                    run_vectorise = False
            elif stage == "scheduler":
                start_scheduler = False
            elif stage == "vectorise":
                run_vectorise = False
        warmup_lite = True
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
        remaining = _timebox_or_budget_remaining()
        stage_timeout = resolved_timeouts.get(stage, base_timeouts.get(stage))
        fallback_budget = provided_budget if provided_budget is not None else None
        timebox_remaining = _timebox_remaining()
        if remaining is None:
            timeout_candidates = [timebox_remaining]
            if stage_timeout is not None:
                timeout_candidates.append(stage_timeout)
            elif fallback_budget is not None:
                timeout_candidates.append(fallback_budget)
            elif stage_hard_cap is not None:
                timeout_candidates.append(stage_hard_cap)
            else:
                timeout_candidates.append(_HEAVY_STAGE_CEILING)
            timeout_candidates = [t for t in timeout_candidates if t is not None]
            timeout = min(timeout_candidates) if timeout_candidates else None
            effective_timeouts[stage] = timeout
            return timeout
        if stage_timeout is None:
            if fallback_budget is None:
                timeout = remaining
            else:
                timeout = max(0.0, min(remaining, fallback_budget))
            if timebox_remaining is not None:
                timeout = min(timeout, timebox_remaining)
            effective_timeouts[stage] = timeout
            return timeout
        timeout = max(0.0, min(stage_timeout, remaining))
        if timebox_remaining is not None:
            timeout = min(timeout, timebox_remaining)
        effective_timeouts[stage] = timeout
        return timeout

    def _stage_budget_window(stage_timeout: float | None) -> float | None:
        shared_remaining = _remaining_shared_budget()
        if shared_remaining is None:
            return stage_timeout
        if stage_timeout is None:
            return shared_remaining
        return min(shared_remaining, stage_timeout)

    _record_bootstrap_deferrals()

    def _enforce_heavy_inline_flag() -> None:
        nonlocal download_model, probe_model, model_probe_only
        nonlocal hydrate_handlers, run_vectorise, heavy_admission

        if force_heavy:
            return

        heavy_gate_status = "deferred-heavy"
        heavy_admission = heavy_admission or heavy_gate_status

        def _defer(stage: str, timeout: float | None) -> None:
            _record_background(stage, heavy_gate_status)
            _hint_background_budget(stage, timeout)

        model_timeout = _effective_timeout("model")
        handler_timeout = _effective_timeout("handlers")
        vectorise_timeout = _effective_timeout("vectorise")

        if download_model:
            _defer("model", model_timeout)
            download_model = False
            probe_model = False
            model_probe_only = False

        if hydrate_handlers:
            _defer("handlers", handler_timeout)
            hydrate_handlers = False
            if run_vectorise:
                _defer("vectorise", vectorise_timeout)
                run_vectorise = False

        if run_vectorise:
            _defer("vectorise", vectorise_timeout)
            run_vectorise = False

    _enforce_heavy_inline_flag()

    def _admit_stage_budget(
        stage: str, planned_timeout: float | None, *, stage_cap: float | None = None
    ) -> tuple[bool, float | None]:
        nonlocal budget_exhausted, budget_gate_reason
        remaining = _remaining_budget()
        if remaining is not None:
            if remaining <= 0:
                budget_exhausted = True
                status = "deferred-budget"
                _record_deferred_background(stage, status)
                budget_gate_reason = budget_gate_reason or status
                log.info("Remaining bootstrap budget exhausted before %s stage", stage)
                return False, None
            if planned_timeout is None:
                planned_timeout = remaining
            elif remaining < planned_timeout:
                budget_exhausted = True
                status = "deferred-budget"
                _record_deferred_background(stage, status)
                budget_gate_reason = budget_gate_reason or status
                log.info(
                    "Vector warmup deferring %s; remaining budget %.2fs below stage ceiling %.2fs",
                    stage,
                    remaining,
                    planned_timeout,
                )
                return False, None
        if check_budget is not None:
            try:
                check_budget()
            except TimeoutError:
                budget_exhausted = True
                status = "deferred-budget"
                _record_deferred_background(stage, status)
                budget_gate_reason = budget_gate_reason or status
                log.info("Vector warmup budget check failed before %s stage; deferring", stage)
                return False, None
        estimate = base_stage_cost.get(stage)
        if estimate is not None and planned_timeout is not None:
            cap_limited = stage_cap is not None and stage_cap < estimate
            if planned_timeout < min(estimate, base_stage_cost.get(stage, estimate)):
                status = "deferred-ceiling" if cap_limited else "deferred-budget"
                _record_deferred_background(stage, status)
                budget_gate_reason = budget_gate_reason or status
                log.info(
                    "Deferring %s warmup; %.2fs remaining below estimated cost %.2fs",
                    stage,
                    planned_timeout,
                    estimate,
                )
                return False, None
        return True, planned_timeout

    budget_callback_missing = budget_remaining is None or budget_remaining is _default_budget_remaining
    check_budget_missing = check_budget is None or check_budget is _default_check_budget
    has_budget_signal = any(
        value is not None for value in (provided_budget, initial_budget_remaining, env_budget)
    ) or not budget_callback_missing or not check_budget_missing

    legacy_budget_missing = (
        not force_heavy
        and not stage_timeouts_supplied
        and not has_budget_signal
    )

    runtime_budget_missing = (
        not force_heavy
        and budget_callback_missing
        and check_budget_missing
        and not has_budget_signal
    )

    heavy_without_budget = (legacy_budget_missing or runtime_budget_missing) and (
        download_model
        or hydrate_handlers
        or start_scheduler
        or run_vectorise
        or not warmup_lite
    )

    if heavy_without_budget:
        status = "deferred-no-budget"
        budget_gate_reason = status
        background_stage_timeouts = background_stage_timeouts or {
            stage: stage_budget_ceiling.get(stage, resolved_timeouts.get(stage))
            for stage in base_timeouts
        }
        for stage in ("handlers", "scheduler", "vectorise"):
            memoised_results[stage] = status
            if stage == "handlers":
                handler_timeout = _effective_timeout("handlers")
                _record_background("handlers", status)
                _hint_background_budget("handlers", handler_timeout)
                hydrate_handlers = False
                if run_vectorise:
                    _record_background("vectorise", status)
                    _hint_background_budget("vectorise", _effective_timeout("vectorise"))
                    run_vectorise = False
            elif stage == "scheduler":
                _record_background("scheduler", status)
                _hint_background_budget("scheduler", _effective_timeout("scheduler"))
                start_scheduler = False
            else:
                _record_background("vectorise", status)
                _hint_background_budget("vectorise", _effective_timeout("vectorise"))
                run_vectorise = False
        if download_model or model_probe_only or probe_model:
            _record_background("model", status)
            download_model = False
            probe_model = False
            warmup_lite = True
            model_probe_only = False

    proactive_background_block = (
        not force_heavy
        and not stage_timeouts_supplied
        and (warmup_lite or bootstrap_fast)
    )

    if proactive_background_block:
        conservative_timeouts = dict(_CONSERVATIVE_STAGE_TIMEOUTS)

        def _planned_timeout(stage: str) -> float | None:
            timeout = stage_budget_ceiling.get(stage, resolved_timeouts.get(stage))
            if timeout is not None:
                return timeout
            return conservative_timeouts.get(stage)

        for stage, enabled in (
            (
                "model",
                requested_model or download_model or probe_model or model_probe_only,
            ),
            ("handlers", requested_handlers or hydrate_handlers),
            ("vectorise", requested_vectorise or bool(run_vectorise)),
        ):
            if not enabled:
                continue
            _record_proactive_deferral(
                stage, "deferred-ceiling", stage_timeout=_planned_timeout(stage)
            )

    def _apply_bootstrap_deferrals() -> None:
        for stage in bootstrap_deferred_records:
            if summary.get(stage):
                continue
            base_status = "deferred-bootstrap"
            if stage in {"handlers", "vectorise"}:
                requested = requested_handlers if stage == "handlers" else requested_vectorise
                if not requested:
                    base_status = "deferred-bootstrap-noop"
            _record_background(stage, base_status)
            _hint_background_budget(stage, _effective_timeout(stage))
            memoised_results[stage] = base_status
            if stage == "handlers" and run_vectorise:
                _record_background("vectorise", "deferred-bootstrap")
                _hint_background_budget("vectorise", _effective_timeout("vectorise"))
                memoised_results["vectorise"] = "deferred-bootstrap"
                background_warmup.add("vectorise")
        if bootstrap_deferred_records:
            background_warmup.update(bootstrap_deferred_records)
            background_candidates.update(bootstrap_deferred_records)

        if warmup_lite and not force_heavy:
            if "handlers" in lite_deferrals - bootstrap_deferred_records:
                if not summary.get("handlers"):
                    status = (
                        "deferred-lite"
                        if requested_handlers
                        else "deferred-lite-noop"
                    )
                    _record_background("handlers", status)
                    _hint_background_budget("handlers", _effective_timeout("handlers"))
            if "model" in lite_deferrals - bootstrap_deferred_records:
                if not summary.get("model"):
                    status = "deferred-lite" if requested_model else "deferred-lite-noop"
                    _record_background("model", status)
                    _hint_background_budget("model", _effective_timeout("model"))
            if "vectorise" in lite_deferrals - bootstrap_deferred_records:
                if not summary.get("vectorise"):
                    status = (
                        "deferred-lite"
                        if requested_vectorise
                        else "deferred-lite-noop"
                    )
                    _record_background("vectorise", status)
                    _hint_background_budget("vectorise", _effective_timeout("vectorise"))
            if "scheduler" in lite_deferrals - bootstrap_deferred_records:
                if not summary.get("scheduler"):
                    status = (
                        "deferred-lite"
                        if requested_scheduler
                        else "deferred-lite-noop"
                    )
                    _record_background("scheduler", status)
                    _hint_background_budget("scheduler", _effective_timeout("scheduler"))

    _apply_bootstrap_deferrals()

    def _should_defer_upfront(
        stage: str, *, stage_timeout: float | None, stage_enabled: bool
    ) -> bool:
        if not stage_enabled:
            return False
        if force_heavy:
            return False

        available_budget = _stage_budget_window(stage_timeout)
        estimate = base_stage_cost.get(stage)
        budget_hint = _available_budget_hint(stage, stage_timeout)

        if warmup_lite:
            _record_background(stage, "deferred-budget")
            log.info("Warmup-lite deferring %s prior to budget guard", stage)
            return True

        if estimate is None:
            return False

        if available_budget is None:
            _record_background(stage, "deferred-no-budget")
            _hint_background_budget(stage, stage_timeout)
            budget_gate_reason = budget_gate_reason or "deferred-no-budget"
            log.info("No budget hints provided; deferring %s to background warmup", stage)
            return True

        remaining_window = budget_hint if budget_hint is not None else available_budget

        if remaining_window is None or remaining_window >= estimate:
            return False

        status = "deferred-budget"
        if stage_timeout is not None and stage_timeout < estimate:
            status = "deferred-ceiling"
        _record_background(stage, status)
        _hint_background_budget(stage, stage_timeout)
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

    def _background_first_gate(
        stage: str,
        stage_timeout: float | None,
        *,
        stage_enabled: bool,
        chain_vectorise: bool = False,
        vectorise_timeout: float | None = None,
    ) -> bool:
        nonlocal download_model, probe_model, model_probe_only
        nonlocal hydrate_handlers, run_vectorise, budget_gate_reason

        if not background_first or not stage_enabled:
            return False

        budget_window = _stage_budget_window(stage_timeout)

        if stage_timeout is None:
            status = "deferred-no-budget"
        elif budget_window is not None and budget_window <= 0:
            status = "deferred-timebox" if stage_hard_cap is not None else "deferred-ceiling"
        elif stage_timeout >= _HEAVY_STAGE_CEILING:
            status = "deferred-ceiling"
        else:
            return False

        budget_gate_reason = budget_gate_reason or status
        _record_deferred_background(stage, status)
        _hint_background_budget(stage, stage_timeout)

        if chain_vectorise and run_vectorise and stage != "vectorise":
            _record_deferred_background("vectorise", status)
            _hint_background_budget("vectorise", vectorise_timeout)
            run_vectorise = False

        if stage == "model":
            download_model = False
            probe_model = False
            model_probe_only = False
        elif stage == "handlers":
            hydrate_handlers = False
        elif stage == "vectorise":
            run_vectorise = False

        log.info(
            "Background-first deferral triggered for %s with timeout %s", stage, stage_timeout
        )
        return True

    def _abort_missing_timeout(
        stage: str,
        stage_timeout: float | None,
        *,
        stage_enabled: bool,
        chain_vectorise: bool = False,
    ) -> bool:
        nonlocal hydrate_handlers, start_scheduler, run_vectorise, budget_gate_reason
        if not stage_enabled or stage_timeout is not None:
            return False

        status = "deferred-no-budget"
        budget_gate_reason = budget_gate_reason or status
        _record_deferred_background(stage, status)
        _hint_background_budget(stage, stage_timeout)
        if stage == "handlers":
            hydrate_handlers = False
        elif stage == "scheduler":
            start_scheduler = False
        if chain_vectorise and run_vectorise:
            _record_deferred_background("vectorise", status)
            _hint_background_budget("vectorise", _effective_timeout("vectorise"))
            run_vectorise = False

        log.info("No stage budget available for %s; deferring to background", stage)
        return True

    def _shared_budget_probe(
        stage: str,
        *,
        stage_timeout: float | None,
        stage_enabled: bool,
        chain_vectorise: bool = False,
        vectorise_timeout: float | None = None,
    ) -> bool:
        nonlocal hydrate_handlers, run_vectorise, budget_gate_reason, budget_exhausted

        if not stage_enabled:
            return False

        shared_remaining = _shared_budget_remaining()
        if shared_remaining is None:
            return False

        stage_window = _stage_budget_window(stage_timeout)
        if stage_window is not None:
            shared_remaining = min(shared_remaining, stage_window)

        estimate = base_stage_cost.get(stage)
        if shared_remaining > 0 and (estimate is None or shared_remaining >= estimate):
            return False

        status = "deferred-shared-budget" if stage_budget_cap is not None else "deferred-budget"
        budget_gate_reason = budget_gate_reason or status
        budget_exhausted = True
        _record_deferred_background(stage, status)
        _hint_background_budget(stage, stage_timeout)

        if chain_vectorise and run_vectorise:
            _record_deferred_background("vectorise", status)
            _hint_background_budget("vectorise", vectorise_timeout)
            run_vectorise = False

        if stage == "handlers":
            hydrate_handlers = False
        elif stage == "vectorise":
            run_vectorise = False

        log.info("Shared warmup budget depleted before %s; deferring", stage)
        return True

    def _shared_budget_preflight() -> None:
        nonlocal hydrate_handlers, start_scheduler, run_vectorise, budget_gate_reason, heavy_admission

        if heavy_admission is not None and heavy_admission != "admitted":
            return

        remaining_shared = _remaining_shared_budget()
        if remaining_shared is None:
            heavy_admission = "admitted"
            return

        planned_stages = [
            stage
            for stage, enabled in (
                ("handlers", hydrate_handlers),
                ("scheduler", start_scheduler),
                ("vectorise", bool(run_vectorise)),
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
                vectorise_timeout=_effective_timeout("vectorise"),
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
    else:
        model_timeout = _effective_timeout("model")
        model_enabled = download_model or probe_model or model_probe_only
        model_cap_deferral = (
            download_model
            and "model" in heavy_stage_cap_hits
            and not _MODEL_READY
            and model_timeout is not None
            and model_timeout >= _HEAVY_STAGE_CEILING
        )

        if model_cap_deferral:
            _record_background("model", "deferred-ceiling")
            _record_cancelled("model", "ceiling")
            download_model = False
            probe_model = False
            model_probe_only = False
            model_enabled = False

        if _background_first_gate("model", model_timeout, stage_enabled=model_enabled):
            return _finalise()

        model_budget_window = _stage_budget_window(model_timeout)
        if model_enabled:
            if model_budget_window is not None and model_budget_window <= 0:
                status = "deferred-timebox" if model_timeout is not None else "deferred-budget"
                _record_lazy_sentinel("model", reason=status, stage_timeout=model_timeout)
                _record_cancelled(
                    "model", "ceiling" if status == "deferred-timebox" else "budget"
                )
                return _finalise()
            if lazy_sentinel_active and model_budget_window is None:
                _record_lazy_sentinel("model", stage_timeout=model_timeout)
                _record_cancelled("model", "budget")
                return _finalise()

        admitted, model_timeout = _admit_stage_budget(
            "model", model_timeout, stage_cap=stage_budget_ceiling.get("model")
        )
        if not admitted:
            _record_cancelled("model", "budget")
            return _finalise()

        if _abort_missing_timeout("model", model_timeout, stage_enabled=model_enabled):
            _record_cancelled("model", "budget")
            return _finalise()
        if _gate_conservative_budget("model", model_enabled, model_timeout):
            if _should_defer_upfront(
                "model", stage_timeout=model_timeout, stage_enabled=model_enabled
            ):
                _record_cancelled("model", "budget")
                return _finalise()
            elif not _guard("model"):
                if _should_abort("model"):
                    return _finalise()
        elif _should_abort("model"):
            return _finalise()
        else:
            _record_cancelled("model", "ceiling")
            return _finalise()

        if download_model:
            if model_timeout is not None and model_timeout <= 0:
                budget_exhausted = True
                _record_deferred_background("model", "skipped-budget")
                _record_cancelled("model", "budget")
                log.info("Vector warmup model download skipped: no remaining budget")
                return _finalise()
            if not _has_estimated_budget("model", budget_cap=model_timeout):
                _record_cancelled("model", "budget")
                return _finalise()
            model_gate_budget, model_gate_start = _start_stage_gate("model", model_timeout)
            completed, path, elapsed, cancelled = _run_with_budget(
                "model",
                lambda stop_event: ensure_embedding_model(
                    logger=log,
                    warmup=True,
                    warmup_lite=False,
                    warmup_heavy=not warmup_lite or force_heavy,
                    stop_event=stop_event,
                    budget_check=lambda evt: _cooperative_budget_check(
                        "model", evt
                    ),
                    download_timeout=model_timeout,
                ),
                timeout=model_timeout,
            )
            _record_elapsed("model", elapsed)
            if cancelled:
                _record_cancelled("model", cancelled)
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
                try:
                    _apply_stage_gate(
                        "model",
                        model_gate_budget,
                        model_gate_start,
                        model_timeout,
                        elapsed_hint=elapsed,
                    )
                except TimeoutError as exc:
                    _record_cancelled(
                        "model", "ceiling" if model_timeout is not None else "budget"
                    )
                    log.info(
                        "Vector warmup model gate exceeded; deferring", extra={"error": str(exc)}
                    )
                    return _finalise()
            elif budget_exhausted:
                if "model" not in summary:
                    _record_deferred("model", "deferred-budget")
                log.info("Vector warmup model download deferred after budget exhaustion")
                return _finalise()
            else:
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

            completed, dest, elapsed, cancelled = _run_with_budget("model", _probe)
            _record_elapsed("model", elapsed)
            if cancelled:
                _record_cancelled("model", cancelled)
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
                try:
                    _apply_stage_gate(
                        "model",
                        *_start_stage_gate("model", model_timeout),
                        model_timeout,
                        elapsed_hint=elapsed,
                    )
                except TimeoutError as exc:
                    _record_cancelled(
                        "model", "ceiling" if model_timeout is not None else "budget"
                    )
                    log.info(
                        "Vector warmup model probe gate exceeded; deferring", extra={"error": str(exc)}
                    )
                    return _finalise()
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
        vectorise_timeout = _effective_timeout("vectorise")
        handler_budget_window = _stage_budget_window(handler_timeout)
        if hydrate_handlers:
            if handler_budget_window is not None and handler_budget_window <= 0:
                status = (
                    "deferred-timebox" if handler_timeout is not None else "deferred-budget"
                )
                _record_lazy_sentinel(
                    "handlers",
                    reason=status,
                    stage_timeout=handler_timeout,
                    chain_vectorise=bool(run_vectorise),
                    vectorise_timeout=vectorise_timeout,
                )
                _record_cancelled(
                    "handlers", "ceiling" if status == "deferred-timebox" else "budget"
                )
                return _finalise()
            if lazy_sentinel_active and handler_budget_window is None:
                _record_lazy_sentinel(
                    "handlers",
                    stage_timeout=handler_timeout,
                    chain_vectorise=bool(run_vectorise),
                    vectorise_timeout=vectorise_timeout,
                )
                _record_cancelled("handlers", "budget")
                return _finalise()
        if _background_first_gate(
            "handlers",
            handler_timeout,
            stage_enabled=hydrate_handlers,
            chain_vectorise=bool(run_vectorise),
            vectorise_timeout=vectorise_timeout,
        ):
            return _finalise()
        if _shared_budget_probe(
            "handlers",
            stage_timeout=handler_timeout,
            stage_enabled=hydrate_handlers,
            chain_vectorise=bool(run_vectorise),
            vectorise_timeout=vectorise_timeout,
        ):
            _record_cancelled("handlers", "budget")
            return _finalise()
        admitted, handler_timeout = _admit_stage_budget(
            "handlers", handler_timeout, stage_cap=stage_budget_ceiling.get("handlers")
        )
        if not admitted:
            _record_cancelled("handlers", "budget")
            if run_vectorise:
                _record_deferred_background("vectorise", "deferred-budget")
            return _finalise()
        if _abort_missing_timeout(
            "handlers",
            handler_timeout,
            stage_enabled=hydrate_handlers,
            chain_vectorise=bool(run_vectorise),
        ):
            _record_cancelled("handlers", "budget")
            return _finalise()
        if _gate_conservative_budget("handlers", hydrate_handlers, handler_timeout):
            if _should_defer_upfront(
                "handlers", stage_timeout=handler_timeout, stage_enabled=hydrate_handlers
            ):
                _record_cancelled("handlers", "budget")
                return _finalise()
            elif (
                hydrate_handlers
                and handler_budget_window is not None
                and handler_budget_window <= 0
            ):
                status = "deferred-budget"
                cancelled_reason = "budget"
                if handler_timeout is not None and handler_timeout <= 0:
                    status = "deferred-ceiling"
                    cancelled_reason = "ceiling"
                _defer_handler_chain(
                    status,
                    stage_timeout=handler_timeout,
                    vectorise_timeout=vectorise_timeout,
                )
                _record_cancelled("handlers", cancelled_reason)
                return _finalise()
            elif not _guard("handlers"):
                handler_status = summary.get("handlers", "deferred-budget")
                if handler_status.startswith("skipped"):
                    handler_status = handler_status.replace("skipped", "deferred", 1)
                _defer_handler_chain(
                    handler_status,
                    stage_timeout=handler_timeout,
                    vectorise_timeout=vectorise_timeout,
                )
                if _should_abort("handlers"):
                    return _finalise()
            else:
                if hydrate_handlers:
                    if handler_timeout is not None and handler_timeout <= 0:
                        budget_exhausted = True
                        _defer_handler_chain(
                            "deferred-budget",
                            stage_timeout=handler_timeout,
                            vectorise_timeout=vectorise_timeout,
                        )
                        _record_cancelled("handlers", "budget")
                        log.info(
                            "Vector warmup handler hydration skipped: no remaining budget"
                        )
                        return _finalise()
                    if not _has_estimated_budget("handlers", budget_cap=handler_timeout):
                        _defer_handler_chain(
                            summary.get("handlers", "deferred-budget"),
                            stage_timeout=handler_timeout,
                            vectorise_timeout=vectorise_timeout,
                        )
                        _record_cancelled("handlers", "budget")
                        return _finalise()
                    handler_gate_budget, handler_gate_start = _start_stage_gate(
                        "handlers", handler_timeout
                    )
                    try:
                        from .vectorizer import SharedVectorService

                        completed, svc, elapsed, cancelled = _run_with_budget(
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
                        if cancelled:
                            _record_cancelled("handlers", cancelled)
                        if completed:
                            _record("handlers", "hydrated")
                            handler_deferrals = getattr(
                                svc, "handler_deferrals", None
                            ) or {}
                            if handler_deferrals:
                                summary["handler_deferrals"] = json.dumps(
                                    handler_deferrals, sort_keys=True
                                )
                                _update_warmup_stage_cache(
                                    "handlers",
                                    summary.get("handlers", "hydrated"),
                                    log,
                                    meta={"handler_deferrals": handler_deferrals},
                                    emit_metric=False,
                                )
                        try:
                            _apply_stage_gate(
                                "handlers",
                                handler_gate_budget,
                                handler_gate_start,
                                handler_timeout,
                                elapsed_hint=elapsed,
                            )
                        except TimeoutError:
                            status = (
                                "deferred-timebox"
                                if handler_timeout is not None
                                else "deferred-budget"
                            )
                            _record_cancelled(
                                "handlers",
                                "ceiling" if handler_timeout is not None else "budget",
                            )
                            _defer_handler_chain(
                                status,
                                stage_timeout=handler_timeout,
                                vectorise_timeout=vectorise_timeout,
                            )
                            log.info("Handler warmup gate exceeded; deferring stage")
                            return _finalise()
                        if budget_exhausted:
                            if "handlers" not in summary:
                                _record_deferred("handlers", "deferred-budget")
                            log.info(
                                "Vector warmup handler hydration deferred after budget exhaustion",
                            )
                            return _finalise()
                        else:
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
        elif _should_abort("handlers"):
            return _finalise()
        else:
            _record_cancelled("handlers", "ceiling")
            return _finalise()

    if _reuse("scheduler"):
        pass
    else:
        scheduler_timeout = _effective_timeout("scheduler")
        admitted, scheduler_timeout = _admit_stage_budget(
            "scheduler", scheduler_timeout, stage_cap=stage_budget_ceiling.get("scheduler")
        )
        if not admitted:
            _record_cancelled("scheduler", "budget")
            return _finalise()
        if _abort_missing_timeout(
            "scheduler", scheduler_timeout, stage_enabled=start_scheduler
        ):
            _record_cancelled("scheduler", "budget")
            return _finalise()
        if _gate_conservative_budget("scheduler", start_scheduler, scheduler_timeout):
            if _guard("scheduler"):
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
                        else (
                            "deferred for warmup-lite"
                            if status == "deferred-lite"
                            else "skipped"
                        ),
                    )
        elif _should_abort("scheduler"):
            return _finalise()
        else:
            _record_cancelled("scheduler", "ceiling")
            return _finalise()

    should_vectorise = bool(run_vectorise)
    if _reuse("vectorise"):
        pass
    else:
        vectorise_timeout = _effective_timeout("vectorise")
        vectorise_budget_window = _stage_budget_window(vectorise_timeout)
        if should_vectorise:
            if vectorise_budget_window is not None and vectorise_budget_window <= 0:
                status = (
                    "deferred-timebox" if vectorise_timeout is not None else "deferred-budget"
                )
                _record_lazy_sentinel(
                    "vectorise", reason=status, stage_timeout=vectorise_timeout
                )
                _record_cancelled(
                    "vectorise", "ceiling" if status == "deferred-timebox" else "budget"
                )
                return _finalise()
            if lazy_sentinel_active and vectorise_budget_window is None:
                _record_lazy_sentinel(
                    "vectorise", stage_timeout=vectorise_timeout
                )
                _record_cancelled("vectorise", "budget")
                return _finalise()
        if _background_first_gate(
            "vectorise", vectorise_timeout, stage_enabled=should_vectorise
        ):
            return _finalise()
        if _shared_budget_probe(
            "vectorise",
            stage_timeout=vectorise_timeout,
            stage_enabled=should_vectorise,
        ):
            _record_cancelled("vectorise", "budget")
            return _finalise()
        admitted, vectorise_timeout = _admit_stage_budget(
            "vectorise", vectorise_timeout, stage_cap=stage_budget_ceiling.get("vectorise")
        )
        if not admitted:
            _record_cancelled("vectorise", "budget")
            return _finalise()
        if _abort_missing_timeout(
            "vectorise", vectorise_timeout, stage_enabled=should_vectorise
        ):
            _record_cancelled("vectorise", "budget")
            return _finalise()
        if _gate_conservative_budget(
            "vectorise", should_vectorise, vectorise_timeout
        ):
            if _should_defer_upfront(
                "vectorise", stage_timeout=vectorise_timeout, stage_enabled=should_vectorise
            ):
                _record_cancelled("vectorise", "budget")
                return _finalise()
            elif (
                should_vectorise
                and vectorise_budget_window is not None
                and vectorise_budget_window <= 0
            ):
                status = "deferred-budget"
                cancelled_reason = "budget"
                if vectorise_timeout is not None and vectorise_timeout <= 0:
                    status = "deferred-ceiling"
                    cancelled_reason = "ceiling"
                _record_deferred_background("vectorise", status)
                _hint_background_budget("vectorise", vectorise_timeout)
                _record_cancelled("vectorise", cancelled_reason)
                return _finalise()
            elif _guard("vectorise"):
                if should_vectorise and svc is not None:
                    budget_hint = _available_budget_hint("vectorise", vectorise_timeout)
                    if budget_hint is not None and budget_hint < min_vectorise_budget:
                        status = "deferred-ceiling" if budget_hint <= 0 else "deferred-budget"
                        _record_deferred_background("vectorise", status)
                        _hint_background_budget("vectorise", vectorise_timeout)
                        _record_cancelled("vectorise", "budget")
                        log.info(
                            "Vectorise warmup deferred before embedder check; budget %.2fs below minimum %.2fs",
                            budget_hint,
                            min_vectorise_budget,
                        )
                        return _finalise()
                    embedder_available = getattr(svc, "text_embedder", None) is not None
                    placeholder_present = False
                    if not embedder_available:
                        try:
                            probe = getattr(svc, "probe_text_embedder", None)
                            if callable(probe):
                                embedder_available, placeholder_present = probe()
                        except Exception:  # pragma: no cover - defensive logging
                            log.debug("Embedder preflight probe failed", exc_info=True)
                    if not embedder_available:
                        status = "deferred-embedder"
                        if warmup_lite and not placeholder_present:
                            status = "deferred-lite"
                        _record_deferred_background("vectorise", status)
                        _hint_background_budget("vectorise", vectorise_timeout)
                        log.info("Vectorise warmup deferred: embedder unavailable")
                        return _finalise()
                    if vectorise_timeout is not None and vectorise_timeout <= 0:
                        budget_exhausted = True
                        _record_deferred_background("vectorise", "deferred-budget")
                        _hint_background_budget("vectorise", vectorise_timeout)
                        _record_cancelled("vectorise", "budget")
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
                                if warmup_lite:
                                    return {"vectorise": "probe"}
                                return svc.vectorise(
                                    "text", {"text": "warmup"}, stop_event=stop_event
                                )

                            vectorise_gate_budget, vectorise_gate_start = _start_stage_gate(
                                "vectorise", vectorise_timeout
                            )
                            completed, _, elapsed, cancelled = _run_with_budget(
                                "vectorise",
                                _vectorise,
                                timeout=vectorise_timeout,
                            )
                            _record_elapsed("vectorise", elapsed)
                            if cancelled:
                                _record_cancelled("vectorise", cancelled)
                            if completed:
                                _record("vectorise", "ok")
                                try:
                                    _apply_stage_gate(
                                        "vectorise",
                                        vectorise_gate_budget,
                                        vectorise_gate_start,
                                        vectorise_timeout,
                                        elapsed_hint=elapsed,
                                    )
                                except TimeoutError:
                                    status = (
                                        "deferred-timebox"
                                        if vectorise_timeout is not None
                                        else "deferred-budget"
                                    )
                                    _record_cancelled(
                                        "vectorise",
                                        "ceiling"
                                        if vectorise_timeout is not None
                                        else "budget",
                                    )
                                    _record_deferred_background("vectorise", status)
                                    log.info(
                                        "Vectorise warmup gate exceeded; deferring stage"
                                    )
                                    return _finalise()
                            else:
                                return _finalise()
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
        elif _should_abort("vectorise"):
            return _finalise()
        else:
            _record_cancelled("vectorise", "ceiling")
            return _finalise()

    return _finalise()


__all__ = ["ensure_embedding_model", "ensure_scheduler_started", "warmup_vector_service"]
