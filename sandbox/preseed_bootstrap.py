"""Bootstrap helpers for seeding shared self-coding context.

The utilities here build the same pipeline/manager setup used by the runtime
bootstrapping helpers and then expose that state to lazy modules so they can
skip re-entrant ``prepare_pipeline_for_bootstrap`` calls.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import traceback
from time import perf_counter
from typing import Any, Callable, Dict

from menace_sandbox.bot_registry import BotRegistry
from menace_sandbox.code_database import CodeDB
from menace_sandbox.context_builder_util import create_context_builder
from menace_sandbox.coding_bot_interface import (
    _pop_bootstrap_context,
    _push_bootstrap_context,
    fallback_helper_manager,
    prepare_pipeline_for_bootstrap,
)
from menace_sandbox.data_bot import DataBot, persist_sc_thresholds
from menace_sandbox.menace_memory_manager import MenaceMemoryManager
from menace_sandbox.model_automation_pipeline import ModelAutomationPipeline
from menace_sandbox.self_coding_engine import SelfCodingEngine
from menace_sandbox.self_coding_manager import SelfCodingManager, internalize_coding_bot
from menace_sandbox.self_coding_thresholds import get_thresholds
from menace_sandbox.threshold_service import ThresholdService

LOGGER = logging.getLogger(__name__)

_BOOTSTRAP_CACHE: Dict[str, Dict[str, Any]] = {}
_BOOTSTRAP_CACHE_LOCK = threading.Lock()
BOOTSTRAP_PROGRESS: Dict[str, str] = {"last_step": "not-started"}
BOOTSTRAP_STEP_TIMEOUT = 30.0
BOOTSTRAP_EMBEDDER_TIMEOUT = float(os.getenv("BOOTSTRAP_EMBEDDER_TIMEOUT", "20.0"))
SELF_CODING_MIN_REMAINING_BUDGET = float(
    os.getenv("SELF_CODING_MIN_REMAINING_BUDGET", "35.0")
)
PREPARE_PIPELINE_PROGRESS_HEARTBEAT = float(
    os.getenv("PREPARE_PIPELINE_PROGRESS_HEARTBEAT", "5.0")
)
BOOTSTRAP_DEADLINE_BUFFER = 5.0
_BOOTSTRAP_EMBEDDER_DISABLED = False
_BOOTSTRAP_EMBEDDER_STARTED = False


def _mark_bootstrap_step(step_name: str) -> None:
    """Record the latest bootstrap step for external visibility."""

    BOOTSTRAP_PROGRESS["last_step"] = step_name


def _run_with_timeout(
    fn,
    *,
    timeout: float,
    bootstrap_deadline: float | None = None,
    description: str,
    abort_on_timeout: bool = True,
    cancel: Callable[[], None] | None = None,
    progress_timeout: float | None = None,
    progress_heartbeat: float | None = None,
    **kwargs: Any,
):
    """Execute ``fn`` with a timeout to avoid indefinite hangs."""

    start_time = time.monotonic()
    hard_deadline = (
        bootstrap_deadline + BOOTSTRAP_DEADLINE_BUFFER if bootstrap_deadline else None
    )

    effective_timeout = timeout
    if bootstrap_deadline:
        time_remaining = bootstrap_deadline - start_time
        if time_remaining > 0:
            buffered_remaining = max(time_remaining - BOOTSTRAP_DEADLINE_BUFFER, 0.0)
            effective_timeout = min(timeout, buffered_remaining, time_remaining)

    initial_deadline = start_time + effective_timeout
    if hard_deadline is not None:
        initial_deadline = min(initial_deadline, hard_deadline)

    progress_window = progress_timeout or effective_timeout
    max_deadline = hard_deadline or (start_time + effective_timeout + progress_window)
    current_deadline = min(initial_deadline, max_deadline)

    result: Dict[str, Any] = {}

    def _refresh_deadline(sub_step: str | None = None) -> None:
        nonlocal current_deadline
        if progress_window <= 0:
            return
        proposed_deadline = min(time.monotonic() + progress_window, max_deadline)
        if proposed_deadline > current_deadline:
            LOGGER.info(
                "extending timeout for %s due to progress",
                description,
                extra={
                    "previous_deadline": current_deadline,
                    "new_deadline": proposed_deadline,
                    "sub_step": sub_step,
                },
            )
            current_deadline = min(proposed_deadline, max_deadline)

    progress_callback = kwargs.get("progress_callback")
    wrapped_progress_callback = None
    if progress_callback is not None:
        def _wrapped_progress(sub_step: str) -> None:
            _refresh_deadline(sub_step)
            return progress_callback(sub_step)

        wrapped_progress_callback = _wrapped_progress
        kwargs["progress_callback"] = _wrapped_progress

    def _target() -> None:
        try:
            result["value"] = fn(**kwargs)
        except Exception as exc:  # pragma: no cover - error propagation
            result["exc"] = exc

    heartbeat_stop = threading.Event()
    heartbeat_thread: threading.Thread | None = None

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    if (
        progress_heartbeat
        and progress_heartbeat > 0
        and wrapped_progress_callback is not None
    ):
        LOGGER.info(
            "starting progress heartbeat for %s", description, extra={"interval": progress_heartbeat}
        )

        def _heartbeat() -> None:
            while not heartbeat_stop.wait(progress_heartbeat):
                try:
                    wrapped_progress_callback("heartbeat")
                except Exception:  # pragma: no cover - defensive heartbeat logging
                    LOGGER.debug(
                        "progress heartbeat callback failed for %s", description, exc_info=True
                    )

        heartbeat_thread = threading.Thread(
            target=_heartbeat, name=f"{description}-heartbeat", daemon=True
        )
        heartbeat_thread.start()

    try:
        while thread.is_alive():
            remaining = current_deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(min(remaining, 0.1))
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(0.5)

    if thread.is_alive():
        if thread.ident is not None:
            frame = sys._current_frames().get(thread.ident)
            if frame is not None:
                LOGGER.error(
                    "stack trace for %s thread prior to cancellation:\n%s",
                    description,
                    "".join(traceback.format_stack(frame)),
                )
        elapsed = time.monotonic() - start_time
        LOGGER.error(
            "%s timed out after %.1fs (last_step=%s)",
            description,
            elapsed,
            BOOTSTRAP_PROGRESS.get("last_step", "unknown"),
        )
        if cancel is not None:
            try:
                cancel()
            except Exception:  # pragma: no cover - defensive cancel logging
                LOGGER.exception("failed to invoke cancellation for %s", description)
        thread.join(0.1)
        if abort_on_timeout:
            raise TimeoutError(
                f"{description} timed out after {elapsed:.1f}s "
                f"(last_step={BOOTSTRAP_PROGRESS.get('last_step', 'unknown')})"
            )

        LOGGER.warning("skipping %s due to timeout", description)
        return None

    if "exc" in result:
        LOGGER.exception("%s failed", description, exc_info=result["exc"])
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
    stop_event: threading.Event | None = None,
    bootstrap_deadline: float | None = None,
    prepare_pipeline_timeout: float | None = None,
) -> Dict[str, Any]:
    """Build and seed bootstrap helpers for reuse by entry points.

    The returned mapping contains the seeded ``registry``, ``data_bot``,
    ``context_builder``, ``engine``, ``pipeline`` and ``manager`` instances.
    Subsequent invocations return cached instances for the given ``bot_name`` when
    ``use_cache`` is ``True``. Pass ``use_cache=False`` to force a fresh bootstrap
    without populating or reading the shared cache.
    """

    global _BOOTSTRAP_CACHE

    def _log_step(step_name: str, start_time: float) -> None:
        LOGGER.info("%s completed (elapsed=%.3fs)", step_name, perf_counter() - start_time)

    _ensure_not_stopped(stop_event)

    _mark_bootstrap_step("embedder_preload")
    _bootstrap_embedder(BOOTSTRAP_EMBEDDER_TIMEOUT, stop_event=stop_event)

    cancel = stop_event.set if stop_event else None

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
            context_builder = create_context_builder()
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
                    cancel=cancel,
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
                    cancel=cancel,
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
                _push_bootstrap_context,
                timeout=BOOTSTRAP_STEP_TIMEOUT,
                bootstrap_deadline=bootstrap_deadline,
                description="_push_bootstrap_context placeholder",
                abort_on_timeout=True,
                cancel=cancel,
                registry=registry,
                data_bot=data_bot,
                manager=bootstrap_manager,
                pipeline=bootstrap_manager,
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
                _seed_research_aggregator_context,
                timeout=BOOTSTRAP_STEP_TIMEOUT,
                bootstrap_deadline=bootstrap_deadline,
                description="_seed_research_aggregator_context placeholder",
                abort_on_timeout=False,
                cancel=cancel,
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
            deadline_remaining = (
                bootstrap_deadline - time.monotonic() if bootstrap_deadline else None
            )
            if prepare_pipeline_timeout is not None:
                timeout_source = "configured_override"
                base_prepare_timeout = prepare_pipeline_timeout
            elif deadline_remaining is not None:
                timeout_source = "bootstrap_deadline"
                base_prepare_timeout = max(deadline_remaining, 0.0)
            else:
                timeout_source = "default_step_timeout"
                base_prepare_timeout = BOOTSTRAP_STEP_TIMEOUT

            if deadline_remaining is not None:
                prepare_timeout = max(
                    0.0, min(base_prepare_timeout, deadline_remaining)
                )
            else:
                prepare_timeout = base_prepare_timeout

            def _record_prepare_sub_step(sub_step: str) -> None:
                step_name = f"prepare_pipeline.{sub_step}"
                _mark_bootstrap_step(step_name)
                LOGGER.info(
                    "prepare_pipeline_for_bootstrap progress",
                    extra={"bootstrap_step": step_name, "sub_step": sub_step},
                )

            LOGGER.info(
                "prepare_pipeline_for_bootstrap timeout selected",
                extra={
                    "timeout": prepare_timeout,
                    "deadline_remaining": deadline_remaining,
                    "configured_timeout": prepare_pipeline_timeout,
                    "timeout_source": timeout_source,
                    "progress_heartbeat": PREPARE_PIPELINE_PROGRESS_HEARTBEAT,
                },
            )
            prepare_start = perf_counter()
            heartbeat_cushion = max(PREPARE_PIPELINE_PROGRESS_HEARTBEAT, 1.0)
            prepare_progress_window = prepare_timeout + heartbeat_cushion
            try:
                pipeline, promote_pipeline = _run_with_timeout(
                    prepare_pipeline_for_bootstrap,
                    timeout=prepare_timeout,
                    bootstrap_deadline=bootstrap_deadline,
                    description="prepare_pipeline_for_bootstrap",
                    abort_on_timeout=True,
                    cancel=cancel,
                    progress_timeout=prepare_progress_window,
                    progress_heartbeat=PREPARE_PIPELINE_PROGRESS_HEARTBEAT,
                    pipeline_cls=ModelAutomationPipeline,
                    context_builder=context_builder,
                    bot_registry=registry,
                    data_bot=data_bot,
                    bootstrap_runtime_manager=bootstrap_manager,
                    manager=bootstrap_manager,
                    progress_callback=_record_prepare_sub_step,
                )
            except Exception:
                LOGGER.exception("prepare_pipeline_for_bootstrap failed (step=prepare_pipeline)")
                raise
            finally:
                _pop_bootstrap_context(placeholder_context)
            LOGGER.info(
                "prepare_pipeline_for_bootstrap finished (last_step=%s)",
                BOOTSTRAP_PROGRESS["last_step"],
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
                promote_pipeline,
                timeout=BOOTSTRAP_STEP_TIMEOUT,
                bootstrap_deadline=bootstrap_deadline,
                description="promote_pipeline",
                abort_on_timeout=True,
                cancel=cancel,
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
            cancel=cancel,
            registry=registry,
            data_bot=data_bot,
            manager=manager,
            pipeline=pipeline,
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
            cancel=cancel,
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
