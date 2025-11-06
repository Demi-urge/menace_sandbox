"""Tkinter GUI for launching and monitoring the Windows sandbox."""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Sequence

from dependency_health import DependencyMode


logger = logging.getLogger(__name__)


class SandboxLauncherGUI(tk.Tk):
    """User interface for running the sandbox preflight and launch steps."""

    WINDOW_TITLE = "Windows Sandbox Launcher"
    WINDOW_GEOMETRY = "900x600"

    def __init__(self) -> None:
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.geometry(self.WINDOW_GEOMETRY)

        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.preflight_thread: threading.Thread | None = None
        self.preflight_abort = threading.Event()
        self.pause_event = threading.Event()
        self.retry_event = threading.Event()
        self.decision_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.debug_queue: "queue.Queue[str]" = queue.Queue()

        self._configure_icon()
        self._configure_style()

        self._notebook = ttk.Notebook(self)
        self._status_frame = ttk.Frame(self._notebook, padding=(12, 12, 12, 12))
        self._notebook.add(self._status_frame, text="Status")
        self._notebook.grid(row=0, column=0, sticky=tk.NSEW, padx=12, pady=(12, 6))

        self._create_status_view(self._status_frame)
        self._control_frame = ttk.Frame(self, padding=(12, 6, 12, 12))
        self._create_controls(self._control_frame)
        self._control_frame.grid(row=1, column=0, sticky=tk.EW)

        self._configure_weights()
        self._install_logging()
        self.after(100, self._drain_log_queue)

    # region setup helpers
    def _configure_icon(self) -> None:
        """Configure the window icon if an ``.ico`` file is available."""

        icon_path = Path(__file__).with_suffix(".ico")
        if icon_path.exists():
            try:
                self.iconbitmap(icon_path)
            except Exception:  # pragma: no cover - platform specific
                pass

    def _configure_style(self) -> None:
        """Apply consistent styling to the widgets."""

        style = ttk.Style(self)
        style.configure("TFrame", padding=0)
        style.configure("TButton", padding=(8, 4))
        style.configure("Sandbox.TButton", padding=(12, 6))

    def _install_logging(self) -> None:
        """Attach a queue-backed handler for cross-thread logging."""

        logger.setLevel(logging.INFO)
        logger.propagate = False

        self._queue_handler = _QueueLogHandler(self.log_queue)
        self._queue_handler.setFormatter(logging.Formatter("%(levelname)s — %(message)s"))

        if self._queue_handler not in logger.handlers:
            logger.addHandler(self._queue_handler)

    def _configure_weights(self) -> None:
        """Apply grid weights to keep the layout responsive."""

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self._status_frame.columnconfigure(0, weight=1)
        self._status_frame.rowconfigure(0, weight=1)

        self._control_frame.columnconfigure(0, weight=1)
        self._control_frame.columnconfigure(1, weight=1)
        self._control_frame.columnconfigure(2, weight=0)
    # endregion setup helpers

    # region widget builders
    def _create_status_view(self, parent: ttk.Frame) -> None:
        """Create the status log view with scrollbars."""

        self._status_text = tk.Text(
            parent,
            state=tk.DISABLED,
            wrap=tk.WORD,
            height=20,
            relief=tk.FLAT,
        )
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self._status_text.yview)
        self._status_text.configure(yscrollcommand=scrollbar.set)

        self._status_text.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)

        for tag, colour in {
            "info": "#0b5394",
            "success": "#38761d",
            "warning": "#b45f06",
            "error": "#990000",
        }.items():
            self._status_text.tag_configure(tag, foreground=colour)

    def _create_controls(self, parent: ttk.Frame) -> None:
        """Create the control buttons for preflight and sandbox launch."""

        run_preflight = ttk.Button(
            parent,
            text="Run Preflight",
            style="Sandbox.TButton",
            command=self._run_preflight_clicked,
        )
        run_preflight.grid(row=0, column=0, padx=(0, 6), sticky=tk.EW)

        start_sandbox = ttk.Button(
            parent,
            text="Start Sandbox",
            style="Sandbox.TButton",
            command=self._on_start_sandbox,
            state=tk.DISABLED,
        )
        start_sandbox.grid(row=0, column=1, padx=(6, 0), sticky=tk.EW)

        retry_step = ttk.Button(
            parent,
            text="Retry Step",
            style="Sandbox.TButton",
            command=self._on_retry_step,
            state=tk.DISABLED,
        )
        retry_step.grid(row=0, column=2, padx=(6, 0), sticky=tk.EW)
        retry_step.grid_remove()

        self._run_preflight_btn = run_preflight
        self._start_sandbox_btn = start_sandbox
        self._retry_step_btn = retry_step
    # endregion widget builders

    # region public API
    def log_message(self, message: str, tag: str = "info") -> None:
        """Append ``message`` to the status log using the provided tag."""

        self._status_text.configure(state=tk.NORMAL)
        self._status_text.insert(tk.END, f"{message}\n", (tag,))
        self._status_text.see(tk.END)
        self._status_text.configure(state=tk.DISABLED)
    # endregion public API

    # region callbacks
    def _run_preflight_clicked(self) -> None:  # pragma: no cover - GUI callback
        if self.preflight_thread and self.preflight_thread.is_alive():
            return

        self._run_preflight_btn.configure(state=tk.DISABLED)
        self._start_sandbox_btn.configure(state=tk.DISABLED)
        self.preflight_abort.clear()
        self.retry_event.clear()
        self.pause_event.clear()

        self.log_message("Running preflight checks...", "info")

        self.preflight_thread = threading.Thread(
            target=self._execute_preflight,
            name="PreflightThread",
            daemon=True,
        )
        self.preflight_thread.start()

    def _on_start_sandbox(self) -> None:  # pragma: no cover - GUI callback
        self.log_message("Starting sandbox...", "info")

    def _execute_preflight(self) -> None:
        """Run the Phase 5 preflight orchestration on a worker thread."""

        success = False
        completion_message = "Preflight aborted."
        completion_tag = "warning"

        try:
            logger.info("Starting sandbox preflight checks (Phase 5)...")

            runner = globals().get("run_full_preflight")
            if runner is None:
                raise RuntimeError("Preflight runner is unavailable")

            runner(
                logger=logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.preflight_abort,
                retry_event=self.retry_event,
                debug_queue=self.debug_queue,
            )

            if self.preflight_abort.is_set():
                completion_message = "Preflight aborted."
                completion_tag = "warning"
            else:
                success = True
                completion_message = "Preflight completed successfully."
                completion_tag = "success"
        except Exception:  # pragma: no cover - surfaced via logging
            logger.exception("Sandbox preflight failed.")
            completion_message = "Preflight failed. Check logs for details."
            completion_tag = "error"
        finally:
            self.log_queue.put((completion_message, completion_tag))
            self.after(0, self._on_preflight_done, success)

    def _on_preflight_done(self, success: bool) -> None:
        """Re-enable controls after the preflight thread finishes."""

        self.preflight_thread = None
        self._run_preflight_btn.configure(state=tk.NORMAL)
        if success:
            self._start_sandbox_btn.configure(state=tk.NORMAL)
        else:
            self._start_sandbox_btn.configure(state=tk.DISABLED)

    def _drain_log_queue(self) -> None:
        """Drain queued log records and append them to the status view."""

        drained = False
        while True:
            try:
                message, tag = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_message(message, tag)
            drained = True

        if self.pause_event.is_set():
            self._retry_step_btn.configure(state=tk.NORMAL)
            self._retry_step_btn.grid()
            try:
                title, message = self.decision_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                should_retry = messagebox.askyesno(title, message)
                if should_retry:
                    self.pause_event.clear()
                    self.log_queue.put(("Continuing…", "info"))
                    self.retry_event.set()
                    self._retry_step_btn.configure(state=tk.DISABLED)
                    self._retry_step_btn.grid_remove()
                else:
                    self.preflight_abort.set()
                    self.log_queue.put(("Preflight aborted by user.", "warning"))
                    self._retry_step_btn.configure(state=tk.DISABLED)
                    self._retry_step_btn.grid_remove()
                    self.after(0, self._on_preflight_done, False)
        else:
            self._retry_step_btn.configure(state=tk.DISABLED)
            self._retry_step_btn.grid_remove()

        delay = 100 if drained else 250
        self.after(delay, self._drain_log_queue)

    def _on_retry_step(self) -> None:  # pragma: no cover - GUI callback
        if not self.pause_event.is_set():
            return

        self.pause_event.clear()
        self.retry_event.set()
        self.log_queue.put(("Continuing…", "info"))
        self._retry_step_btn.configure(state=tk.DISABLED)
        self._retry_step_btn.grid_remove()
    # endregion callbacks


@dataclass(frozen=True)
class _PreflightStep:
    """Container describing a preflight step and its failure metadata."""

    label: str
    failure_title: str
    failure_message: str
    runner_name: str


class _PreflightWorker:
    """Coordinate execution of the sandbox preflight sequence."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        pause_event: threading.Event,
        decision_queue: "queue.Queue[tuple[str, str]]",
        abort_event: threading.Event,
        retry_event: threading.Event,
        debug_queue: "queue.Queue[str]",
        steps: Sequence[_PreflightStep],
    ) -> None:
        self._logger = logger
        self._pause_event = pause_event
        self._decision_queue = decision_queue
        self._abort_event = abort_event
        self._retry_event = retry_event
        self._debug_queue = debug_queue
        self._steps = steps

    def run(self) -> None:
        """Execute the configured preflight steps in sequence."""

        for step in self._steps:
            if self._abort_event.is_set():
                break
            should_continue = self._run_step(step)
            if not should_continue:
                break

    def _run_step(self, step: _PreflightStep) -> bool:
        """Execute ``step`` and handle pause/retry/abort semantics."""

        while True:
            if self._abort_event.is_set():
                self._logger.info("Preflight aborted before %s", step.label)
                return False

            self._logger.info("Starting %s", step.label)
            try:
                runner = getattr(sys.modules[__name__], step.runner_name)
                if not callable(runner):  # pragma: no cover - defensive guard
                    raise TypeError(f"Step '{step.runner_name}' is not callable")
                runner(self._logger)
            except Exception as exc:  # pragma: no cover - logged via GUI
                self._logger.exception("%s failed", step.label)
                self._debug_queue.put(str(exc))
                self._pause_event.set()
                self._decision_queue.put(
                    (
                        step.failure_title,
                        f"{step.failure_message}\n\n{exc}",
                    )
                )
                decision = self._await_decision()
                if decision == "retry":
                    continue
                if decision == "abort":
                    return False

                # ``skip`` – continue to the next step after logging.
                self._logger.warning(
                    "Continuing after failure in %s", step.label
                )
                return True

            self._logger.info("Finished %s", step.label)
            return True

    def _await_decision(self) -> str:
        """Wait for the GUI to signal retry, skip, or abort."""

        while True:
            if self._abort_event.is_set():
                return "abort"
            if self._retry_event.is_set():
                self._retry_event.clear()
                self._pause_event.clear()
                return "retry"
            if not self._pause_event.is_set():
                if self._retry_event.is_set():
                    self._retry_event.clear()
                    self._pause_event.clear()
                    return "retry"
                time.sleep(0.05)
                if self._retry_event.is_set():
                    self._retry_event.clear()
                    self._pause_event.clear()
                    return "retry"
                return "skip"
            time.sleep(0.05)


def run_full_preflight(
    *,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[tuple[str, str]]",
    abort_event: threading.Event,
    retry_event: threading.Event,
    debug_queue: "queue.Queue[str]",
) -> None:
    """Run the Phase 5 sandbox preflight orchestration."""

    worker = _PreflightWorker(
        logger=logger,
        pause_event=pause_event,
        decision_queue=decision_queue,
        abort_event=abort_event,
        retry_event=retry_event,
        debug_queue=debug_queue,
        steps=_DEFAULT_STEPS,
    )
    worker.run()


def _git_fetch_and_reset(logger: logging.Logger) -> None:
    """Ensure the local repository mirrors the remote state."""

    logger.info("Fetching latest repository state…")


def _purge_stale_state(logger: logging.Logger) -> None:
    """Remove temporary files that could affect subsequent runs."""

    logger.info("Purging stale state…")


def _remove_lock_artifacts(logger: logging.Logger) -> None:
    """Remove stale lock files that block dependency installs."""

    logger.info("Removing stale lock files…")


def _prefetch_heavy_dependencies(logger: logging.Logger) -> None:
    """Download heavy dependencies ahead of time."""

    logger.info("Prefetching heavy dependencies…")


def _warm_shared_vector_service(logger: logging.Logger) -> None:
    """Warm up the shared vector service cache."""

    logger.info("Warming shared vector service…")


def _ensure_environment(logger: logging.Logger) -> None:
    """Ensure environment prerequisites are satisfied."""

    logger.info("Ensuring environment configuration…")


def _prime_self_coding_registry(logger: logging.Logger) -> None:
    """Prime the self-coding registry to avoid runtime delays."""

    logger.info("Priming self-coding registry…")


def _run_pip_commands(logger: logging.Logger) -> None:
    """Run required pip commands for the sandbox."""

    logger.info("Running pip commands…")


def _bootstrap_ai_counter_bot(logger: logging.Logger) -> None:
    """Bootstrap the AI counter bot dependencies."""

    logger.info("Bootstrapping AI counter bot…")


def _evaluate_health_snapshot(
    snapshot: dict,
    *,
    dependency_mode: DependencyMode,
) -> tuple[bool, list[str]]:
    """Evaluate the health snapshot collected after preflight."""

    failures: list[str] = []

    if not snapshot.get("databases_accessible", True):
        db_errors = snapshot.get("database_errors", {})
        if db_errors:
            detail = ", ".join(
                f"{name}: {error}" for name, error in sorted(db_errors.items())
            )
        else:
            detail = "databases inaccessible"
        failures.append(f"databases inaccessible: {detail}")

    dependency_health = snapshot.get("dependency_health", {})
    missing = dependency_health.get("missing", [])
    for dep in missing:
        name = dep.get("name", "unknown dependency")
        optional = dep.get("optional", False)
        if optional and dependency_mode is DependencyMode.MINIMAL:
            continue
        failures.append(f"missing dependency: {name}")

    return (not failures, failures)


_DEFAULT_STEPS: Sequence[_PreflightStep] = (
    _PreflightStep(
        label="Fetching latest Git snapshot",
        failure_title="Git fetch failed",
        failure_message="Updating the repository from origin failed.",
        runner_name="_git_fetch_and_reset",
    ),
    _PreflightStep(
        label="Purging stale state",
        failure_title="State purge failed",
        failure_message="Purging stale artefacts encountered an error.",
        runner_name="_purge_stale_state",
    ),
    _PreflightStep(
        label="Removing lock artefacts",
        failure_title="Lock artefact removal failed",
        failure_message="Removing stale lock files was unsuccessful.",
        runner_name="_remove_lock_artifacts",
    ),
    _PreflightStep(
        label="Prefetching heavy dependencies",
        failure_title="Dependency prefetch failed",
        failure_message="Prefetching heavy dependencies encountered an error.",
        runner_name="_prefetch_heavy_dependencies",
    ),
    _PreflightStep(
        label="Warming shared vector service",
        failure_title="Vector service warm-up failed",
        failure_message="Warming shared vector service failed.",
        runner_name="_warm_shared_vector_service",
    ),
    _PreflightStep(
        label="Ensuring environment",
        failure_title="Environment validation failed",
        failure_message="Ensuring environment prerequisites failed.",
        runner_name="_ensure_environment",
    ),
    _PreflightStep(
        label="Priming self-coding registry",
        failure_title="Self-coding registry priming failed",
        failure_message="Priming the self-coding registry failed.",
        runner_name="_prime_self_coding_registry",
    ),
    _PreflightStep(
        label="Running pip commands",
        failure_title="Pip commands failed",
        failure_message="Executing pip commands failed.",
        runner_name="_run_pip_commands",
    ),
    _PreflightStep(
        label="Bootstrapping AI counter bot",
        failure_title="AI counter bot bootstrap failed",
        failure_message="Bootstrapping the AI counter bot failed.",
        runner_name="_bootstrap_ai_counter_bot",
    ),
)


class _QueueLogHandler(logging.Handler):
    """Simple handler that forwards log records into a queue."""

    def __init__(self, log_queue: "queue.Queue[tuple[str, str]]") -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return

        level = record.levelno
        if level >= logging.ERROR:
            tag = "error"
        elif level >= logging.WARNING:
            tag = "warning"
        else:
            tag = "info"
        self._queue.put((message, tag))


__all__ = ["SandboxLauncherGUI", "run_full_preflight"]


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    gui = SandboxLauncherGUI()
    gui.mainloop()
