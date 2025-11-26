"""Bootstrap helpers for seeding shared self-coding context.

The utilities here build the same pipeline/manager setup used by the runtime
bootstrapping helpers and then expose that state to lazy modules so they can
skip re-entrant ``prepare_pipeline_for_bootstrap`` calls.

Timeouts are sourced from ``coding_bot_interface._resolve_bootstrap_wait_timeout``
when available so they respect ``MENACE_BOOTSTRAP_WAIT_SECS`` and
``MENACE_BOOTSTRAP_VECTOR_WAIT_SECS`` while retaining a minimum fallback to the
legacy 30s defaults for compatibility.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
import traceback
import faulthandler
from time import perf_counter
from typing import Any, Dict

from menace_sandbox import coding_bot_interface as _coding_bot_interface
from menace_sandbox.bot_registry import BotRegistry
from menace_sandbox.code_database import CodeDB
from menace_sandbox.context_builder_util import create_context_builder
from menace_sandbox.coding_bot_interface import (
    _pop_bootstrap_context,
    _push_bootstrap_context,
    fallback_helper_manager,
    prepare_pipeline_for_bootstrap,
)
from menace_sandbox.db_router import set_audit_bootstrap_safe_default
from menace_sandbox.data_bot import DataBot, persist_sc_thresholds
from menace_sandbox.menace_memory_manager import MenaceMemoryManager
from menace_sandbox.model_automation_pipeline import ModelAutomationPipeline
from menace_sandbox.self_coding_engine import SelfCodingEngine
from menace_sandbox.self_coding_manager import SelfCodingManager, internalize_coding_bot
from menace_sandbox.self_coding_thresholds import get_thresholds
from menace_sandbox.threshold_service import ThresholdService
from safe_repr import summarise_value
from security.secret_redactor import redact_dict

LOGGER = logging.getLogger(__name__)

_BOOTSTRAP_CACHE: Dict[str, Dict[str, Any]] = {}
_BOOTSTRAP_CACHE_LOCK = threading.Lock()
BOOTSTRAP_PROGRESS: Dict[str, str] = {"last_step": "not-started"}
BOOTSTRAP_STEP_TIMELINE: list[tuple[str, float]] = []
_BOOTSTRAP_TIMELINE_START: float | None = None
_BOOTSTRAP_TIMELINE_LOCK = threading.Lock()
_DEFAULT_BOOTSTRAP_STEP_TIMEOUT = float(os.getenv("BOOTSTRAP_STEP_TIMEOUT", "30.0"))
BOOTSTRAP_STEP_TIMEOUT = _DEFAULT_BOOTSTRAP_STEP_TIMEOUT
BOOTSTRAP_EMBEDDER_TIMEOUT = float(os.getenv("BOOTSTRAP_EMBEDDER_TIMEOUT", "20.0"))
SELF_CODING_MIN_REMAINING_BUDGET = float(
    os.getenv("SELF_CODING_MIN_REMAINING_BUDGET", "35.0")
)
BOOTSTRAP_DEADLINE_BUFFER = 5.0
_BOOTSTRAP_EMBEDDER_DISABLED = False
_BOOTSTRAP_EMBEDDER_STARTED = False


def _is_vector_bootstrap_heavy(candidate: Any) -> bool:
    """Identify vector-heavy helpers by their module/qualname."""

    module_name = getattr(candidate, "__module__", "") or ""
    qualname = getattr(candidate, "__qualname__", "") or ""
    text = f"{module_name}:{qualname}".lower()
    heavy_tokens = (
        "vector_service",
        "vectorservice",
        "vector_metrics",
        "vectormetrics",
        "patch_history",
        "patchhistory",
    )
    return any(token in text for token in heavy_tokens)


def _resolve_step_timeout(vector_heavy: bool = False) -> float:
    """Resolve a bootstrap step timeout with backwards-compatible defaults."""

    resolved_timeout: float | None = None
    resolver = getattr(_coding_bot_interface, "_resolve_bootstrap_wait_timeout", None)
    if resolver:
        try:
            resolved_timeout = resolver(vector_heavy)
        except Exception:  # pragma: no cover - helper availability best effort
            LOGGER.debug("failed to resolve bootstrap wait timeout", exc_info=True)

    if resolved_timeout is None:
        resolved_timeout = _DEFAULT_BOOTSTRAP_STEP_TIMEOUT

    return max(resolved_timeout, _DEFAULT_BOOTSTRAP_STEP_TIMEOUT)


# Resolve the default timeout eagerly so legacy users retain a stable baseline.
BOOTSTRAP_STEP_TIMEOUT = _resolve_step_timeout()


def _mark_bootstrap_step(step_name: str) -> None:
    """Record the latest bootstrap step for external visibility."""

    global _BOOTSTRAP_TIMELINE_START

    now = time.monotonic()
    with _BOOTSTRAP_TIMELINE_LOCK:
        if _BOOTSTRAP_TIMELINE_START is None:
            _BOOTSTRAP_TIMELINE_START = now

        BOOTSTRAP_STEP_TIMELINE.append((step_name, now))

    BOOTSTRAP_PROGRESS["last_step"] = step_name


def _format_timestamp(epoch_seconds: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch_seconds))


def _safe_kwargs_summary(kwargs: dict[str, Any]) -> dict[str, str]:
    try:
        redacted = redact_dict(kwargs)
    except Exception:  # pragma: no cover - defensive fallback
        redacted = kwargs

    summary: dict[str, str] = {}
    for key, value in redacted.items():
        try:
            summary[key] = summarise_value(value)
        except Exception:  # pragma: no cover - defensive fallback
            summary[key] = "<unrepresentable>"
    return summary


def _dump_thread_traces(target_thread: threading.Thread) -> None:
    frames = sys._current_frames()
    for thread in threading.enumerate():
        if not thread.ident:
            continue

        marker = f"[bootstrap-timeout][thread={thread.name} id={thread.ident}" + (
            " target]" if thread is target_thread else "]"
        )
        print(f"{marker} stack trace:", flush=True)
        buffer = io.StringIO()
        try:
            faulthandler.dump_traceback(
                file=buffer, all_threads=False, thread_id=thread.ident
            )
            trace = buffer.getvalue()
        except Exception:  # pragma: no cover - fallback for unsupported platforms
            trace = ""

        if not trace:
            frame = frames.get(thread.ident)
            if frame:
                trace = "".join(traceback.format_stack(frame))
        if trace:
            print(trace.rstrip(), flush=True)
        else:
            print(f"{marker} unable to capture stack trace", flush=True)


def _render_bootstrap_timeline(now: float) -> list[str]:
    with _BOOTSTRAP_TIMELINE_LOCK:
        timeline = list(BOOTSTRAP_STEP_TIMELINE)
        start = _BOOTSTRAP_TIMELINE_START

    if not timeline:
        return ["[bootstrap-timeout][timeline] no bootstrap steps recorded"]

    baseline = start if start is not None else timeline[0][1]

    lines = []
    for step, timestamp in timeline:
        elapsed_ms = int((timestamp - baseline) * 1000)
        lines.append(f"[bootstrap-timeout][timeline] {step} → {elapsed_ms}ms")

    return lines


def _resolve_timeout(
    base_timeout: float,
    *,
    bootstrap_deadline: float | None,
    heavy_bootstrap: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Resolve an effective timeout with deadline- and heavy-aware scaling."""

    effective_timeout = base_timeout
    now = time.monotonic()
    deadline_remaining = bootstrap_deadline - now if bootstrap_deadline else None
    metadata: dict[str, Any] = {
        "base_timeout": base_timeout,
        "heavy_bootstrap": heavy_bootstrap,
        "deadline_remaining": deadline_remaining,
        "deadline_buffer": BOOTSTRAP_DEADLINE_BUFFER,
    }

    if heavy_bootstrap:
        heavy_scale = float(os.getenv("BOOTSTRAP_HEAVY_TIMEOUT_SCALE", "1.5"))
        effective_timeout = max(effective_timeout, base_timeout * heavy_scale)
        metadata["heavy_scale"] = heavy_scale

    if deadline_remaining is not None:
        buffered_remaining = max(deadline_remaining - BOOTSTRAP_DEADLINE_BUFFER, 0.0)
        metadata["deadline_buffered_remaining"] = buffered_remaining
        if buffered_remaining > effective_timeout:
            effective_timeout = buffered_remaining
        effective_timeout = min(effective_timeout, max(deadline_remaining, 0.0))

    metadata["effective_timeout"] = effective_timeout
    return effective_timeout, metadata


def _run_with_timeout(
    fn,
    *,
    timeout: float,
    bootstrap_deadline: float | None = None,
    description: str,
    abort_on_timeout: bool = True,
    heavy_bootstrap: bool = False,
    resolved_timeout: tuple[float, dict[str, Any]] | None = None,
    **kwargs: Any,
):
    """Execute ``fn`` with a timeout to avoid indefinite hangs."""

    start_monotonic = time.monotonic()
    start_wall = time.time()
    if resolved_timeout is None:
        effective_timeout, timeout_context = _resolve_timeout(
            timeout, bootstrap_deadline=bootstrap_deadline, heavy_bootstrap=heavy_bootstrap
        )
    else:
        effective_timeout, timeout_context = resolved_timeout

    LOGGER.info(
        "%s starting with timeout (requested=%.1fs effective=%.1fs heavy=%s deadline=%s)",
        description,
        timeout,
        effective_timeout,
        heavy_bootstrap,
        bootstrap_deadline,
        extra={"timeout_context": timeout_context},
    )

    result: Dict[str, Any] = {}

    def _target() -> None:
        try:
            result["value"] = fn(**kwargs)
        except Exception as exc:  # pragma: no cover - error propagation
            result["exc"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(effective_timeout)

    last_step = BOOTSTRAP_PROGRESS.get("last_step", "unknown")

    if thread.is_alive():
        now = time.monotonic()
        with _BOOTSTRAP_TIMELINE_LOCK:
            timeline = list(BOOTSTRAP_STEP_TIMELINE)
        active_step = timeline[-1][0] if timeline else last_step
        active_started_at = timeline[-1][1] if timeline else None
        active_elapsed_ms = (
            int((now - active_started_at) * 1000)
            if active_started_at is not None
            else None
        )
        end_wall = time.time()
        deadline_remaining_now = (
            bootstrap_deadline - time.monotonic() if bootstrap_deadline else None
        )
        metadata = {
            "start_time": _format_timestamp(start_wall),
            "end_time": _format_timestamp(end_wall),
            "elapsed": round(time.monotonic() - start_monotonic, 3),
            "timeout_requested": timeout,
            "timeout_effective": effective_timeout,
            "bootstrap_deadline": bootstrap_deadline,
            "bootstrap_remaining_start": timeout_context.get("deadline_remaining"),
            "bootstrap_remaining_now": deadline_remaining_now,
            "function": getattr(fn, "__name__", fn.__class__.__name__),
            "kwargs": _safe_kwargs_summary(kwargs),
            "thread_name": thread.name,
            "thread_ident": thread.ident,
            "last_step": last_step,
            "active_step": active_step,
            "active_elapsed_ms": active_elapsed_ms,
            "timeout_context": timeout_context,
        }

        LOGGER.error(
            "%s timed out after %.1fs (last_step=%s) metadata=%s",
            description,
            effective_timeout,
            last_step,
            metadata,
        )
        active_fragment = (
            f"active_step={active_step} (+{active_elapsed_ms}ms)"
            if active_step is not None and active_elapsed_ms is not None
            else "active_step=unknown"
        )
        print(
            (
                "[bootstrap-timeout][metadata] %s timed out after %.1fs (%s): %s"
            )
            % (description, effective_timeout, active_fragment, metadata),
            flush=True,
        )

        for line in _render_bootstrap_timeline(now):
            print(line, flush=True)

        _dump_thread_traces(thread)

        if abort_on_timeout:
            raise TimeoutError(
                f"{description} timed out after {effective_timeout:.1f}s"
            )

        LOGGER.warning("skipping %s due to timeout", description)
        return None

    if "exc" in result:
        LOGGER.exception("%s failed", description, exc_info=result["exc"])
        print(
            f"[bootstrap-error] {description} failed after {effective_timeout:.1f}s (last_step={last_step})",
            flush=True,
        )
        raise result["exc"]

    return result.get("value")


def _seed_research_aggregator_context(
    *,
    registry: BotRegistry,
    data_bot: DataBot,
    context_builder: Any,
    engine: SelfCodingEngine,
    pipeline: ModelAutomationPipeline,
    manager: SelfCodingManager,
) -> None:
    try:
        from menace_sandbox import research_aggregator_bot as aggregator
    except Exception:  # pragma: no cover - optional optimisation
        LOGGER.debug("research_aggregator_bot unavailable during seeding", exc_info=True)
        return

    try:
        orchestrator = aggregator.get_orchestrator(
            "ResearchAggregatorBot", data_bot, engine
        )
    except Exception:  # pragma: no cover - orchestrator best effort
        LOGGER.debug("research aggregator orchestrator unavailable", exc_info=True)
        orchestrator = None

    aggregator.registry = registry
    aggregator.data_bot = data_bot
    aggregator._context_builder = context_builder
    aggregator.engine = engine
    aggregator.pipeline = pipeline
    aggregator.manager = manager
    aggregator.evolution_orchestrator = orchestrator
    aggregator._PipelineCls = type(pipeline)

    try:
        runtime_state = aggregator._RuntimeDependencies(
            registry=registry,
            data_bot=data_bot,
            context_builder=context_builder,
            engine=engine,
            pipeline=pipeline,
            evolution_orchestrator=orchestrator,
            manager=manager,
        )
        aggregator._runtime_state = runtime_state
        aggregator._ensure_self_coding_decorated(runtime_state)
    except Exception:  # pragma: no cover - optional runtime hints
        LOGGER.debug("unable to seed research aggregator runtime state", exc_info=True)


def _ensure_not_stopped(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise TimeoutError("initialize_bootstrap_context cancelled via stop event")


def _bootstrap_embedder(timeout: float, *, stop_event: threading.Event | None = None) -> None:
    """Attempt to initialise the shared embedder without blocking bootstrap."""

    global _BOOTSTRAP_EMBEDDER_DISABLED, _BOOTSTRAP_EMBEDDER_STARTED

    if timeout <= 0:
        LOGGER.info("bootstrap embedder timeout disabled; skipping embedder preload")
        return

    if _BOOTSTRAP_EMBEDDER_DISABLED:
        LOGGER.info("embedder preload disabled for this bootstrap run; skipping")
        return

    if _BOOTSTRAP_EMBEDDER_STARTED:
        LOGGER.debug("embedder preload already started; refusing to create another thread")
        return

    try:
        from menace_sandbox.governed_embeddings import (
            _activate_bundled_fallback,
            cancel_embedder_initialisation,
            get_embedder,
        )
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.debug("governed_embeddings unavailable; skipping embedder bootstrap", exc_info=True)
        return

    result: Dict[str, Any] = {}
    embedder_stop_event = threading.Event()
    _BOOTSTRAP_EMBEDDER_STARTED = True

    def _worker() -> None:
        try:
            result["embedder"] = get_embedder(
                timeout=timeout, stop_event=embedder_stop_event
            )
        except Exception as exc:  # pragma: no cover - diagnostics only
            result["error"] = exc

    thread = threading.Thread(target=_worker, name="bootstrap-embedder", daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        LOGGER.warning(
            "embedding model load exceeded %.1fs during bootstrap; attempting fallback", timeout
        )
        embedder_stop_event.set()
        cancel_embedder_initialisation(
            embedder_stop_event, reason="bootstrap_timeout", join_timeout=2.0
        )

        timed_out_embedder = None
        for join_timeout in (2.0, 5.0):
            thread.join(join_timeout)
            if not thread.is_alive():
                timed_out_embedder = result.get("embedder")
                break

        if thread.is_alive():
            LOGGER.warning(
                "bootstrap embedder preload worker still running after timeout cancellation",
                extra={"timeout": timeout},
            )
        elif timed_out_embedder is not None:
            LOGGER.info(
                "bootstrap embedder became ready after timeout: %s",
                type(timed_out_embedder).__name__,
            )
            return timed_out_embedder
        else:
            LOGGER.info("bootstrap embedder preload worker cancelled after timeout")

        fallback_used = False
        try:
            fallback_used = _activate_bundled_fallback("bootstrap_timeout")
        except Exception:  # pragma: no cover - diagnostics only
            LOGGER.warning(
                "failed to activate bundled fallback embedder after timeout", exc_info=True
            )

        _BOOTSTRAP_EMBEDDER_DISABLED = False
        _BOOTSTRAP_EMBEDDER_STARTED = False

        if fallback_used:
            fallback_embedder = get_embedder(timeout=0, stop_event=stop_event)
            if fallback_embedder is not None:
                LOGGER.info(
                    "bootstrap embedder fallback ready: %s",
                    type(fallback_embedder).__name__,
                )
                return fallback_embedder
            LOGGER.info(
                "bundled embedder fallback requested after timeout but none available yet"
            )
        return

    if "error" in result:
        LOGGER.info("embedder preload failed; bootstrap will proceed without embeddings")
        LOGGER.debug("embedder preload error", exc_info=result["error"])
        _BOOTSTRAP_EMBEDDER_DISABLED = True
        return

    embedder = result.get("embedder")
    if embedder is None:
        LOGGER.info(
            "embedder unavailable after %.1fs wait; proceeding without embeddings",
            timeout,
        )
        _BOOTSTRAP_EMBEDDER_DISABLED = True
    else:
        LOGGER.info("bootstrap embedder ready: %s", type(embedder).__name__)


def initialize_bootstrap_context(
    bot_name: str = "ResearchAggregatorBot",
    *,
    use_cache: bool = True,
    heavy_bootstrap: bool = False,
    stop_event: threading.Event | None = None,
    bootstrap_deadline: float | None = None,
) -> Dict[str, Any]:
    """Build and seed bootstrap helpers for reuse by entry points.

    The returned mapping contains the seeded ``registry``, ``data_bot``,
    ``context_builder``, ``engine``, ``pipeline`` and ``manager`` instances.
    Subsequent invocations return cached instances for the given ``bot_name`` when
    ``use_cache`` is ``True``. Pass ``use_cache=False`` to force a fresh bootstrap
    without populating or reading the shared cache. Enable ``heavy_bootstrap`` (or
    set ``BOOTSTRAP_HEAVY_BOOTSTRAP``) to allow timeouts to scale up when more time
    is available or heavy vector work is expected.
    """

    global _BOOTSTRAP_CACHE

    def _log_step(step_name: str, start_time: float) -> None:
        LOGGER.info("%s completed (elapsed=%.3fs)", step_name, perf_counter() - start_time)

    def _timed_callable(func: Any, *, label: str, **func_kwargs: Any) -> Any:
        start = perf_counter()
        LOGGER.debug("starting %s", label)
        try:
            return func(**func_kwargs)
        finally:
            LOGGER.debug("%s completed (elapsed=%.3fs)", label, perf_counter() - start)

    env_heavy = os.getenv("BOOTSTRAP_HEAVY_BOOTSTRAP", "")
    heavy_bootstrap = heavy_bootstrap or env_heavy.lower() in {"1", "true", "yes"}
    LOGGER.info(
        "initialize_bootstrap_context heavy mode=%s", heavy_bootstrap,
        extra={"heavy_env": env_heavy},
    )

    set_audit_bootstrap_safe_default(True)
    _ensure_not_stopped(stop_event)

    _mark_bootstrap_step("embedder_preload")
    _bootstrap_embedder(BOOTSTRAP_EMBEDDER_TIMEOUT, stop_event=stop_event)

    if use_cache:
        cached_context = _BOOTSTRAP_CACHE.get(bot_name)
        if cached_context:
            LOGGER.info(
                "reusing preseeded bootstrap context for %s; pipeline/manager already available",
                bot_name,
            )
            return cached_context

    with _BOOTSTRAP_CACHE_LOCK:
        _ensure_not_stopped(stop_event)
        if use_cache:
            cached_context = _BOOTSTRAP_CACHE.get(bot_name)
            if cached_context:
                LOGGER.info(
                    "reusing preseeded bootstrap context for %s; pipeline/manager already available",
                    bot_name,
                )
                return cached_context

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("context_builder")
        ctx_builder_start = perf_counter()
        try:
            context_builder = create_context_builder(bootstrap_safe=True)
        except Exception:
            LOGGER.exception("context_builder creation failed (step=context_builder)")
            raise
        _log_step("context_builder", ctx_builder_start)

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("bot_registry")
        registry_start = perf_counter()
        try:
            registry = BotRegistry()
        except Exception:
            LOGGER.exception("BotRegistry initialization failed (step=bot_registry)")
            raise
        _log_step("bot_registry", registry_start)

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("data_bot")
        data_bot_start = perf_counter()
        try:
            data_bot = DataBot(start_server=False)
        except Exception:
            LOGGER.exception("DataBot setup failed (step=data_bot)")
            raise
        _log_step("data_bot", data_bot_start)

        remaining_budget = (
            bootstrap_deadline - time.monotonic() if bootstrap_deadline else None
        )
        if remaining_budget is not None and remaining_budget < SELF_CODING_MIN_REMAINING_BUDGET:
            LOGGER.warning(
                "remaining bootstrap budget is low (%.1fs < %.1fs); skipping self-coding bootstrap",
                max(remaining_budget, 0.0),
                SELF_CODING_MIN_REMAINING_BUDGET,
            )
            _ensure_not_stopped(stop_event)
            with fallback_helper_manager(
                bot_registry=registry, data_bot=data_bot
            ) as bootstrap_manager:
                _mark_bootstrap_step("push_final_context")
                _run_with_timeout(
                    _push_bootstrap_context,
                    timeout=BOOTSTRAP_STEP_TIMEOUT,
                    bootstrap_deadline=bootstrap_deadline,
                    description="_push_bootstrap_context final",
                    abort_on_timeout=True,
                    heavy_bootstrap=heavy_bootstrap,
                    registry=registry,
                    data_bot=data_bot,
                    manager=bootstrap_manager,
                    pipeline=bootstrap_manager,
                )
                _mark_bootstrap_step("seed_final_context")
                _run_with_timeout(
                    _seed_research_aggregator_context,
                    timeout=BOOTSTRAP_STEP_TIMEOUT,
                    bootstrap_deadline=bootstrap_deadline,
                    description="_seed_research_aggregator_context final",
                    abort_on_timeout=False,
                    heavy_bootstrap=heavy_bootstrap,
                    registry=registry,
                    data_bot=data_bot,
                    context_builder=context_builder,
                    engine=getattr(bootstrap_manager, "engine", None),
                    pipeline=bootstrap_manager,
                    manager=bootstrap_manager,
                )

            bootstrap_context = {
                "registry": registry,
                "data_bot": data_bot,
                "context_builder": context_builder,
                "engine": getattr(bootstrap_manager, "engine", None),
                "pipeline": bootstrap_manager,
                "manager": bootstrap_manager,
            }
            if use_cache:
                _BOOTSTRAP_CACHE[bot_name] = bootstrap_context
            LOGGER.info(
                "initialize_bootstrap_context completed with disabled self-coding due to budget constraints",
                extra={"remaining_budget": remaining_budget},
            )
            return bootstrap_context

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("self_coding_engine")
        engine_start = perf_counter()
        try:
            engine = SelfCodingEngine(
                CodeDB(),
                MenaceMemoryManager(),
                context_builder=context_builder,
            )
        except Exception:
            LOGGER.exception("SelfCodingEngine instantiation failed (step=self_coding_engine)")
            raise
        _log_step("self_coding_engine", engine_start)

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("prepare_pipeline")
        with fallback_helper_manager(
            bot_registry=registry, data_bot=data_bot
        ) as bootstrap_manager:
            LOGGER.info(
                "seeding research aggregator with bootstrap manager before pipeline preparation"
            )
            LOGGER.info(
                "before _push_bootstrap_context (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            placeholder_context = _run_with_timeout(
                _timed_callable,
                timeout=BOOTSTRAP_STEP_TIMEOUT,
                bootstrap_deadline=bootstrap_deadline,
                description="_push_bootstrap_context placeholder",
                abort_on_timeout=True,
                heavy_bootstrap=heavy_bootstrap,
                func=_push_bootstrap_context,
                label="_push_bootstrap_context placeholder",
                registry=registry,
                data_bot=data_bot,
                manager=bootstrap_manager,
                pipeline=bootstrap_manager,
                bootstrap_safe=True,
                bootstrap_fast=True,
            )
            LOGGER.info(
                "after _push_bootstrap_context (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            LOGGER.info("_push_bootstrap_context completed (step=push_placeholder)")
            LOGGER.info(
                "before _seed_research_aggregator_context (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            _run_with_timeout(
                _timed_callable,
                timeout=BOOTSTRAP_STEP_TIMEOUT,
                bootstrap_deadline=bootstrap_deadline,
                description="_seed_research_aggregator_context placeholder",
                abort_on_timeout=False,
                heavy_bootstrap=heavy_bootstrap,
                func=_seed_research_aggregator_context,
                label="_seed_research_aggregator_context placeholder",
                registry=registry,
                data_bot=data_bot,
                context_builder=context_builder,
                engine=engine,
                pipeline=bootstrap_manager,
                manager=bootstrap_manager,
            )
            LOGGER.info(
                "after _seed_research_aggregator_context (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            LOGGER.info(
                "starting prepare_pipeline_for_bootstrap (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            vector_heavy = False
            vector_timeout = _resolve_step_timeout(vector_heavy=True)
            standard_timeout = _resolve_step_timeout(vector_heavy=False)
            try:
                vector_state = getattr(_coding_bot_interface, "_BOOTSTRAP_STATE", None)
                if vector_state is not None:
                    vector_heavy = bool(getattr(vector_state, "vector_heavy", False))
                if not vector_heavy and _is_vector_bootstrap_heavy(context_builder):
                    vector_heavy = True
                if not vector_heavy and _is_vector_bootstrap_heavy(ModelAutomationPipeline):
                    vector_heavy = True
            except Exception:  # pragma: no cover - diagnostics only
                LOGGER.debug("unable to inspect vector_heavy flag", exc_info=True)

            prepare_timeout = vector_timeout if vector_heavy else standard_timeout
            heavy_prepare = heavy_bootstrap or vector_heavy
            resolved_prepare_timeout = _resolve_timeout(
                prepare_timeout,
                bootstrap_deadline=bootstrap_deadline,
                heavy_bootstrap=heavy_prepare,
            )
            effective_prepare_timeout = resolved_prepare_timeout[0]
            LOGGER.info(
                "prepare_pipeline timeout selected",
                extra={
                    "vector_heavy": vector_heavy,
                    "timeout": effective_prepare_timeout,
                    "timeout_requested": prepare_timeout,
                    "vector_timeout": vector_timeout,
                    "standard_timeout": standard_timeout,
                    "heavy_bootstrap": heavy_prepare,
                    "timeout_context": resolved_prepare_timeout[1],
                },
            )
            print(
                (
                    "starting prepare_pipeline_for_bootstrap "
                    "(last_step=%s, timeout=%.1fs, elapsed=0.0s)"
                )
                % (BOOTSTRAP_PROGRESS["last_step"], effective_prepare_timeout),
                flush=True,
            )
            prepare_start = perf_counter()
            try:
                pipeline, promote_pipeline = _run_with_timeout(
                    _timed_callable,
                    timeout=prepare_timeout,
                    bootstrap_deadline=bootstrap_deadline,
                    description="prepare_pipeline_for_bootstrap",
                    abort_on_timeout=True,
                    heavy_bootstrap=heavy_prepare,
                    resolved_timeout=resolved_prepare_timeout,
                    func=prepare_pipeline_for_bootstrap,
                    label="prepare_pipeline_for_bootstrap",
                    stop_event=stop_event,
                    pipeline_cls=ModelAutomationPipeline,
                    context_builder=context_builder,
                    bot_registry=registry,
                    data_bot=data_bot,
                    bootstrap_runtime_manager=bootstrap_manager,
                    manager=bootstrap_manager,
                    bootstrap_safe=True,
                    bootstrap_fast=True,
                )
            except Exception:
                LOGGER.exception("prepare_pipeline_for_bootstrap failed (step=prepare_pipeline)")
                print(
                    ( 
                        "prepare_pipeline_for_bootstrap failed "
                        "(last_step=%s, timeout=%.1fs, elapsed=%.2fs)"
                    )
                    % (
                        BOOTSTRAP_PROGRESS["last_step"],
                        effective_prepare_timeout,
                        perf_counter() - prepare_start,
                    ),
                    flush=True,
                )
                raise
            finally:
                _pop_bootstrap_context(placeholder_context)
            LOGGER.info(
                "prepare_pipeline_for_bootstrap finished (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            print(
                (
                    "prepare_pipeline_for_bootstrap finished "
                    "(last_step=%s, timeout=%.1fs, elapsed=%.2fs)"
                )
                % (
                    BOOTSTRAP_PROGRESS["last_step"],
                    effective_prepare_timeout,
                    perf_counter() - prepare_start,
                ),
                flush=True,
            )
            _log_step("prepare_pipeline_for_bootstrap", prepare_start)

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("threshold_persistence")
        thresholds = get_thresholds(bot_name)
        try:
            persist_sc_thresholds(
                bot_name,
                roi_drop=thresholds.roi_drop,
                error_increase=thresholds.error_increase,
                test_failure_increase=thresholds.test_failure_increase,
            )
            LOGGER.info("persist_sc_thresholds completed (step=threshold_persistence)")
        except Exception:  # pragma: no cover - best effort persistence
            LOGGER.debug("failed to persist thresholds for %s", bot_name, exc_info=True)

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("internalize_coding_bot")
        internalize_start = perf_counter()
        try:
            LOGGER.info(
                "before internalize_coding_bot (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            manager = internalize_coding_bot(
                bot_name,
                engine,
                pipeline,
                data_bot=data_bot,
                bot_registry=registry,
                threshold_service=ThresholdService(),
                roi_threshold=thresholds.roi_drop,
                error_threshold=thresholds.error_increase,
                test_failure_threshold=thresholds.test_failure_increase,
            )
        except Exception:
            LOGGER.exception("internalize_coding_bot failed (step=internalize_coding_bot)")
            raise
        LOGGER.info(
            "after internalize_coding_bot (last_step=%s)", BOOTSTRAP_PROGRESS["last_step"]
        )
        _log_step("internalize_coding_bot", internalize_start)
        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("promote_pipeline")
        promote_start = perf_counter()
        try:
            LOGGER.info(
                "starting promote_pipeline (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
            )
            _run_with_timeout(
                _timed_callable,
                timeout=BOOTSTRAP_STEP_TIMEOUT,
                bootstrap_deadline=bootstrap_deadline,
                description="promote_pipeline",
                abort_on_timeout=True,
                heavy_bootstrap=heavy_bootstrap,
                func=promote_pipeline,
                label="promote_pipeline",
                manager=manager,
            )
        except Exception:
            LOGGER.exception("promote_pipeline failed (step=promote_pipeline)")
            raise
        LOGGER.info(
            "promote_pipeline finished (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        _log_step("promote_pipeline", promote_start)

        _ensure_not_stopped(stop_event)
        _mark_bootstrap_step("seed_final_context")
        LOGGER.info(
            "starting _push_bootstrap_context (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        _run_with_timeout(
            _push_bootstrap_context,
            timeout=BOOTSTRAP_STEP_TIMEOUT,
            bootstrap_deadline=bootstrap_deadline,
            description="_push_bootstrap_context final",
            abort_on_timeout=True,
            heavy_bootstrap=heavy_bootstrap,
            registry=registry,
            data_bot=data_bot,
            manager=manager,
            pipeline=pipeline,
            bootstrap_safe=True,
            bootstrap_fast=True,
        )
        LOGGER.info(
            "_push_bootstrap_context finished (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        LOGGER.info("_push_bootstrap_context completed (step=push_final_context)")
        LOGGER.info(
            "starting _seed_research_aggregator_context (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        _run_with_timeout(
            _seed_research_aggregator_context,
            timeout=BOOTSTRAP_STEP_TIMEOUT,
            bootstrap_deadline=bootstrap_deadline,
            description="_seed_research_aggregator_context final",
            abort_on_timeout=False,
            heavy_bootstrap=heavy_bootstrap,
            registry=registry,
            data_bot=data_bot,
            context_builder=context_builder,
            engine=engine,
            pipeline=pipeline,
            manager=manager,
        )
        LOGGER.info(
            "_seed_research_aggregator_context finished (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        LOGGER.info("_seed_research_aggregator_context completed (step=seed_final)")

        bootstrap_context = {
            "registry": registry,
            "data_bot": data_bot,
            "context_builder": context_builder,
            "engine": engine,
            "pipeline": pipeline,
            "manager": manager,
        }
        if use_cache:
            _BOOTSTRAP_CACHE[bot_name] = bootstrap_context
        _mark_bootstrap_step("bootstrap_complete")
        LOGGER.info(
            "initialize_bootstrap_context completed successfully for %s (step=bootstrap_complete)",
            bot_name,
        )
        LOGGER.info(
            "bootstrap marked complete; returning context to caller (last_step=%s)",
            BOOTSTRAP_PROGRESS["last_step"],
        )
        LOGGER.info("bootstrap complete; returning to caller")
        LOGGER.info(
            "bootstrap return (last_step=%s)", BOOTSTRAP_PROGRESS["last_step"]
        )
        return bootstrap_context


__all__ = ["initialize_bootstrap_context"]
