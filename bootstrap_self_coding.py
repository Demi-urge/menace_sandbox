"""Utility script to unblock self-coding bootstrap loops.

Running this script performs two actions:

1. Purge stale state and lock files that can leave the autonomous sandbox in a
   perpetual retry loop after a crashed or aborted bootstrap attempt.
2. Invoke :func:`internalize_coding_bot` for the requested bot so the
   :class:`~menace_sandbox.self_coding_manager.SelfCodingManager` is registered
   immediately instead of waiting for the registry back-off cycle.

It is intentionally lightweight so that it can be run from PowerShell or bash
without requiring any additional dependencies beyond the project itself.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from textwrap import shorten
from typing import Iterable, Iterator, Mapping, Any, Callable

from bootstrap_readiness import build_stage_deadlines
from bootstrap_timeout_policy import (
    compute_prepare_pipeline_component_budgets,
    load_component_timeout_floors,
    wait_for_bootstrap_quiet_period,
)
from bootstrap_metrics import BOOTSTRAP_DURATION_STORE, compute_stats, record_durations


LOGGER = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parent


DEFAULT_BOOTSTRAP_TIMEOUTS: Mapping[str, str] = {
    "MENACE_BOOTSTRAP_WAIT_SECS": "240",
    "MENACE_BOOTSTRAP_VECTOR_WAIT_SECS": "240",
    "BOOTSTRAP_STEP_TIMEOUT": "240",
    "BOOTSTRAP_VECTOR_STEP_TIMEOUT": "240",
}

for _timeout_env, _timeout_default in DEFAULT_BOOTSTRAP_TIMEOUTS.items():
    os.environ.setdefault(_timeout_env, _timeout_default)


# ``bootstrap_self_coding.py`` is often executed directly via ``python`` or
# ``py`` from the repository root on Windows.  In that scenario the process
# working directory is ``menace_sandbox`` itself, so ``sys.path`` only contains
# the package directory rather than its parent.  Absolute imports such as
# ``menace_sandbox.self_coding_manager`` therefore fail because Python expects
# ``sys.path`` entries to point to the *parent* of the package.  Insert the
# parent directory explicitly so that the package-style import works regardless
# of how the script is invoked.
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))


STALE_STATE_FILES: tuple[Path, ...] = (
    REPO_ROOT / "sandbox_data" / "self_coding_engine_state.json",
    REPO_ROOT / "vector_service" / "failed_tags.json",
)

JOURNAL_FILES: tuple[Path, ...] = (
    REPO_ROOT / "logs" / "audit" / "audit_log.db-journal",
    REPO_ROOT / "logs" / "audit_log.db-shm",
    REPO_ROOT / "audit" / "audit.db-journal",
)


def _apply_self_coding_guard() -> float:
    """Delay bootstrap when peers are active or host load is high."""

    guard_delay, guard_scale = wait_for_bootstrap_quiet_period(LOGGER)
    if guard_scale > 1.0:
        current = float(os.getenv("BOOTSTRAP_STEP_TIMEOUT", "240") or 240)
        adjusted = current * guard_scale
        os.environ["BOOTSTRAP_STEP_TIMEOUT"] = str(adjusted)
        os.environ["MENACE_BOOTSTRAP_WAIT_SECS"] = str(adjusted)
    return guard_delay


def _iter_cleanup_targets() -> Iterable[Path]:
    for path in STALE_STATE_FILES:
        yield path
        yield Path(f"{path}.lock")
    yield from JOURNAL_FILES


def _purge_path(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
            LOGGER.info("Removed stale file: %s", path)
    except IsADirectoryError:
        return
    except OSError:
        LOGGER.warning("Failed to remove stale file: %s", path, exc_info=True)


def purge_stale_files() -> None:
    """Delete well-known stale lock and journal files if they exist."""

    for candidate in _iter_cleanup_targets():
        _purge_path(candidate)


@contextlib.contextmanager
def _monitor_disabled_manager_stream() -> Iterator[list[Mapping[str, Any]]]:
    """Capture ``self_coding.disabled_manager`` telemetry for diagnostics."""

    events: list[Mapping[str, Any]] = []
    try:  # Deferred import keeps script start-up lightweight.
        from menace_sandbox import coding_bot_interface as _cbi
    except Exception:  # pragma: no cover - import guard for exotic layouts
        yield events
        return

    original_emit = getattr(_cbi, "_emit_disabled_manager_metric", None)
    if not callable(original_emit):
        yield events
        return

    def _tracking_emit(payload: Mapping[str, Any]) -> None:
        try:
            snapshot: Mapping[str, Any] = dict(payload)
        except Exception:  # pragma: no cover - payload already concrete
            snapshot = {"payload": payload}
        events.append(snapshot)
        original_emit(payload)

    _cbi._emit_disabled_manager_metric = _tracking_emit  # type: ignore[attr-defined]
    try:
        yield events
    finally:
        _cbi._emit_disabled_manager_metric = original_emit  # type: ignore[attr-defined]


def _persist_pipeline_durations(durations: Mapping[str, float]) -> None:
    if not durations:
        return

    store = record_durations(
        durations=durations, category="self_coding_pipeline", logger=LOGGER
    )
    stats = compute_stats(store.get("self_coding_pipeline", {}))
    LOGGER.info(
        "self-coding bootstrap timings persisted",
        extra={
            "event": "self-coding-bootstrap-durations",
            "steps": {key: round(value, 2) for key, value in durations.items()},
            "stats": {k: {m: round(v, 2) for m, v in vstats.items()} for k, v in stats.items()},
            "store": str(BOOTSTRAP_DURATION_STORE),
        },
    )


def bootstrap_self_coding(bot_name: str) -> None:
    """Trigger :func:`internalize_coding_bot` for *bot_name*."""

    guard_delay = _apply_self_coding_guard()
    baseline_timeout = float(os.getenv("BOOTSTRAP_STEP_TIMEOUT", "240") or 240)
    component_floors = load_component_timeout_floors()
    component_budgets = compute_prepare_pipeline_component_budgets(
        component_floors=component_floors
    )
    stage_policy = build_stage_deadlines(
        baseline_timeout,
        component_budgets=component_budgets,
        component_floors=component_floors,
    )
    LOGGER.info(
        "bootstrap readiness policy applied",
        extra={
            "stage_policy": stage_policy,
            "baseline_timeout": baseline_timeout,
            "component_budgets": component_budgets,
            "component_floors": component_floors,
        },
    )
    if guard_delay > 0:
        LOGGER.info(
            "self-coding bootstrap delayed for active peer",
            extra={"event": "bootstrap-guard", "delay": round(guard_delay, 2)},
        )
    policy_summary = [
        f"{stage}: {details.get('deadline')}s"
        + (" (optional)" if details.get("optional") else "")
        for stage, details in sorted(stage_policy.items())
    ]
    print("bootstrap readiness policy: %s" % ", ".join(policy_summary), flush=True)

    from menace_sandbox.bot_registry import BotRegistry
    from menace_sandbox.code_database import CodeDB
    from menace_sandbox.context_builder_util import create_context_builder
    from menace_sandbox.data_bot import DataBot, persist_sc_thresholds
    from menace_sandbox.menace_memory_manager import MenaceMemoryManager
    from menace_sandbox.model_automation_pipeline import ModelAutomationPipeline
    from menace_sandbox.coding_bot_interface import (
        fallback_helper_manager,
        prepare_pipeline_for_bootstrap,
    )
    from menace_sandbox.self_coding_engine import SelfCodingEngine
    from menace_sandbox.self_coding_manager import internalize_coding_bot
    from menace_sandbox.self_coding_thresholds import get_thresholds

    LOGGER.info("Bootstrapping self-coding manager for bot %s", bot_name)
    step_durations: dict[str, float] = {}

    def _record_step(name: str, action: Callable[[], Any]) -> Any:
        start = time.monotonic()
        try:
            return action()
        finally:
            step_durations[name] = time.monotonic() - start

    builder = _record_step("context_builder", create_context_builder)
    registry = _record_step("bot_registry", BotRegistry)
    data_bot = _record_step("data_bot", lambda: DataBot(start_server=False))
    engine = _record_step(
        "self_coding_engine",
        lambda: SelfCodingEngine(
            CodeDB(),
            MenaceMemoryManager(),
            context_builder=builder,
        ),
    )

    def _build_pipeline() -> tuple[Any, Any]:
        with fallback_helper_manager(
            bot_registry=registry,
            data_bot=data_bot,
        ) as fallback_manager:
            return prepare_pipeline_for_bootstrap(
                pipeline_cls=ModelAutomationPipeline,
                context_builder=builder,
                bot_registry=registry,
                data_bot=data_bot,
                bootstrap_runtime_manager=fallback_manager,
                manager=fallback_manager,
                component_timeouts=component_budgets,
            )

    pipeline, promote_manager = _record_step("prepare_pipeline", _build_pipeline)

    roi_threshold = error_threshold = test_failure_threshold = None
    try:
        thresholds = get_thresholds(bot_name)
    except Exception as exc:  # pragma: no cover - diagnostics only
        LOGGER.warning("Failed to load thresholds for %s: %s", bot_name, exc)
    else:
        roi_threshold = thresholds.roi_drop
        error_threshold = thresholds.error_increase
        test_failure_threshold = thresholds.test_failure_increase
        try:
            persist_sc_thresholds(
                bot_name,
                roi_drop=roi_threshold,
                error_increase=error_threshold,
                test_failure_increase=test_failure_threshold,
            )
        except Exception:  # pragma: no cover - best effort persistence
            LOGGER.exception("Failed to persist thresholds for %s", bot_name)

    class _Tee(io.StringIO):
        def __init__(self, target: io.TextIOBase) -> None:
            super().__init__()
            self._target = target

        def write(self, data: str) -> int:  # pragma: no cover - trivial wrapper
            self._target.write(data)
            return super().write(data)

        def flush(self) -> None:  # pragma: no cover - trivial wrapper
            self._target.flush()
            super().flush()

    stdout_buffer = _Tee(sys.stdout)
    stderr_buffer = _Tee(sys.stderr)
    manager = None
    disabled_manager_events: list[Mapping[str, Any]]
    with _monitor_disabled_manager_stream() as disabled_manager_events:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            manager = _record_step(
                "internalize_coding_bot",
                lambda: internalize_coding_bot(
                    bot_name=bot_name,
                    engine=engine,
                    pipeline=pipeline,
                    data_bot=data_bot,
                    bot_registry=registry,
                    roi_threshold=roi_threshold,
                    error_threshold=error_threshold,
                    test_failure_threshold=test_failure_threshold,
                ),
            )

    captured_stdout = stdout_buffer.getvalue().strip()
    captured_stderr = stderr_buffer.getvalue().strip()

    def _emit_diagnostics(message: str) -> None:
        LOGGER.error("%s", message)
        if captured_stdout:
            LOGGER.error("Captured bootstrap stdout:\n%s", captured_stdout)
        if captured_stderr:
            LOGGER.error("Captured bootstrap stderr:\n%s", captured_stderr)

    def _summarise_capture() -> str:
        parts: list[str] = []
        newline = "\n"
        space = " "
        if captured_stdout:
            normalised_stdout = captured_stdout.replace(newline, space)
            parts.append(
                f"stdout={shorten(normalised_stdout, width=200, placeholder='…')}"
            )
        if captured_stderr:
            normalised_stderr = captured_stderr.replace(newline, space)
            parts.append(
                f"stderr={shorten(normalised_stderr, width=200, placeholder='…')}"
            )
        if not parts:
            stack_excerpt = "".join(traceback.format_stack(limit=5)).strip()
            if stack_excerpt:
                normalised_stack = stack_excerpt.replace(newline, space)
                parts.append(
                    f"stack={shorten(normalised_stack, width=200, placeholder='…')}"
                )
        return ", ".join(parts) if parts else "no diagnostics captured"

    if manager is None:
        failure_summary = (
            "internalize_coding_bot returned None; self-coding manager registration failed"
        )
        diagnostic_summary = _summarise_capture()
        message = f"{failure_summary}. Diagnostics: {diagnostic_summary}"
        _emit_diagnostics(message)
        raise RuntimeError(message)

    manager_truthy = bool(manager)
    if not manager_truthy:
        disabled_summary = (
            "internalize_coding_bot returned a disabled manager; "
            "captured warnings/stack traces indicate re-entrant bootstrap prevented activation"
        )
        try:  # Optional clarity for readers familiar with the interface module
            from menace_sandbox import coding_bot_interface
        except Exception:  # pragma: no cover - defensive import guard
            pass
        else:
            if isinstance(manager, coding_bot_interface._DisabledSelfCodingManager):
                disabled_summary = (
                    "internalize_coding_bot returned coding_bot_interface._DisabledSelfCodingManager; "
                    "captured warnings/stack traces indicate re-entrant bootstrap prevented activation"
                )
        diagnostic_summary = _summarise_capture()
        _emit_diagnostics(
            f"{disabled_summary}. Diagnostics: {diagnostic_summary}"
        )
        raise RuntimeError(
            f"{disabled_summary}. Diagnostics: {diagnostic_summary}"
        )
    sentinel_reused_events = [
        payload
        for payload in disabled_manager_events
        if payload.get("sentinel_reused") or payload.get("reentrant")
    ]
    if sentinel_reused_events:
        summary_bits = [
            f"owner={payload.get('owner')}, reason={payload.get('reason')}, depth={payload.get('bootstrap_depth')}"
            for payload in sentinel_reused_events
        ]
        summary = "; ".join(summary_bits) if summary_bits else "unknown sentinel reuse"
        warning = (
            "self_coding.disabled_manager telemetry reported sentinel reuse during bootstrap; "
            f"investigate nested _bootstrap_manager invocations ({summary})"
        )
        _emit_diagnostics(
            f"{warning}. Diagnostics: {_summarise_capture()}"
        )
        raise RuntimeError(
            f"{warning}. Diagnostics: {_summarise_capture()}"
        )

    promote_manager(manager)
    LOGGER.info(
        "Promoted bootstrap pipeline manager for %s (%s)",
        bot_name,
        type(manager).__name__,
        extra={"stage_policy": stage_policy},
    )
    _persist_pipeline_durations(step_durations)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bot-name",
        default="InformationSynthesisBot",
        help="Name of the bot to internalize (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip removal of stale state and lock files",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if not args.skip_cleanup:
        purge_stale_files()
    else:
        LOGGER.info("Skipping stale file cleanup as requested")

    bootstrap_self_coding(args.bot_name)


if __name__ == "__main__":
    main()

