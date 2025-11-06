"""Windows sandbox preflight orchestration and launcher GUI."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import tkinter as tk
from tkinter import messagebox, ttk

import auto_env_setup
import bootstrap_self_coding
import prime_registry
from dependency_health import DependencyMode
from sandbox_runner import bootstrap as sandbox_bootstrap
from vector_service import SharedVectorService


@dataclass(frozen=True)
class _PreflightStep:
    """Metadata describing a single preflight step."""

    name: str
    description: str
    failure_title: str
    failure_message: str


_PREFLIGHT_STEPS: tuple[_PreflightStep, ...] = (
    _PreflightStep(
        name="_git_sync",
        description="Synchronising repository state",
        failure_title="Git synchronisation failed",
        failure_message="Synchronising repository state",
    ),
    _PreflightStep(
        name="_purge_stale_files",
        description="Purging stale caches",
        failure_title="Purge of stale files failed",
        failure_message="Removing stale automation artifacts",
    ),
    _PreflightStep(
        name="_cleanup_lock_and_model_artifacts",
        description="Removing stale lock files and model caches",
        failure_title="Lock and model cleanup failed",
        failure_message="Removing stale lock files and model caches",
    ),
    _PreflightStep(
        name="_install_heavy_dependencies",
        description="Installing heavy dependency bundles",
        failure_title="Heavy dependency installation failed",
        failure_message="Installing heavy dependency bundles",
    ),
    _PreflightStep(
        name="_warm_shared_vector_service",
        description="Warming shared vector service",
        failure_title="Vector service warm-up failed",
        failure_message="Priming shared vector service",
    ),
    _PreflightStep(
        name="_ensure_env_flags",
        description="Ensuring environment flags",
        failure_title="Environment preparation failed",
        failure_message="Ensuring environment configuration",
    ),
    _PreflightStep(
        name="_prime_registry",
        description="Priming registry data",
        failure_title="Registry priming failed",
        failure_message="Priming registry data",
    ),
    _PreflightStep(
        name="_install_python_dependencies",
        description="Installing Python dependencies",
        failure_title="Python dependency installation failed",
        failure_message="Installing Python dependencies",
    ),
    _PreflightStep(
        name="_bootstrap_self_coding",
        description="Bootstrapping self-coding environment",
        failure_title="Self-coding bootstrap failed",
        failure_message="Bootstrapping self-coding environment",
    ),
)


def _log(logger, message: str, *args: object) -> None:
    """Log ``message`` via ``logger`` handling printf-style formatting."""

    if hasattr(logger, "info"):
        logger.info(message, *args)


def _safe_queue_put(target_queue: queue.Queue, payload) -> None:
    """Insert ``payload`` into ``target_queue`` ignoring ``queue.Full`` errors."""

    try:
        target_queue.put_nowait(payload)
    except queue.Full:  # pragma: no cover - defensive
        pass


def _git_sync(logger) -> None:
    """Synchronise git repositories."""

    _log(logger, "Synchronising git repositories with remote state…")


def _purge_stale_files(logger) -> None:
    """Remove stale bootstrap artifacts."""

    _log(logger, "Purging stale bootstrap artifacts…")
    bootstrap_self_coding.purge_stale_files()


def _cleanup_lock_and_model_artifacts(logger) -> None:
    """Remove stale lock files and cached model artifacts."""

    _log(logger, "Cleaning up stale lock files and cached models…")


def _install_heavy_dependencies(logger) -> None:
    """Install large dependency bundles required for sandbox startup."""

    from neurosales.scripts import setup_heavy_deps

    _log(logger, "Installing heavy dependencies (download only)…")
    setup_heavy_deps.main(download_only=True)


def _warm_shared_vector_service(logger) -> None:
    """Prime the shared vector service to avoid cold-start penalties."""

    _log(logger, "Warming shared vector service…")
    SharedVectorService().vectorise([])


def _ensure_env_flags(logger) -> None:
    """Ensure environment variables required by sandbox tooling are set."""

    _log(logger, "Ensuring environment configuration flags…")
    auto_env_setup.ensure_env()


def _prime_registry(logger) -> None:
    """Prime registry information consumed by the sandbox."""

    _log(logger, "Priming registry data…")
    prime_registry.main()


def _install_python_dependencies(logger) -> None:
    """Install Python dependencies required for the sandbox runtime."""

    _log(logger, "Validating Python dependency installation…")


def _bootstrap_self_coding(logger) -> None:
    """Bootstrap the self-coding environment used during sandbox runs."""

    _log(logger, "Bootstrapping self-coding environment…")
    bootstrap_self_coding.bootstrap_self_coding()


def _evaluate_health_snapshot(
    snapshot: dict,
    *,
    dependency_mode: DependencyMode,
) -> tuple[bool, list[str]]:
    """Evaluate ``snapshot`` information for health problems."""

    failures: list[str] = []

    if not snapshot.get("databases_accessible", True):
        error_details = snapshot.get("database_errors", {})
        if error_details:
            failures.append(
                "databases inaccessible: "
                + ", ".join(f"{db}: {reason}" for db, reason in error_details.items())
            )
        else:
            failures.append("databases inaccessible")

    dependency_info = snapshot.get("dependency_health", {})
    missing_dependencies: Iterable[dict] = dependency_info.get("missing", [])
    for missing in missing_dependencies:
        name = missing.get("name", "unknown dependency")
        optional = bool(missing.get("optional", False))
        if optional and dependency_mode is DependencyMode.MINIMAL:
            continue
        qualifier = " (optional)" if optional else ""
        failures.append(f"missing dependency: {name}{qualifier}")

    return not failures, failures


def run_full_preflight(
    *,
    logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[tuple[str, str, Optional[dict[str, object]]]]",
    abort_event: threading.Event,
    debug_queue: "queue.Queue[str]",
    dependency_mode: DependencyMode = DependencyMode.STRICT,
) -> dict[str, object]:
    """Execute the full sandbox preflight sequence."""

    result: dict[str, object] = {
        "steps_completed": [],
        "snapshot": None,
        "healthy": False,
        "failures": [],
        "aborted": False,
    }

    if abort_event.is_set():
        result["aborted"] = True
        _log(logger, "Preflight aborted before execution.")
        return result

    for step in _PREFLIGHT_STEPS:
        if abort_event.is_set():
            result["aborted"] = True
            _log(logger, "Preflight aborted during %s", step.name)
            return result

        step_runner: Callable[[object], None] = globals()[step.name]
        try:
            _log(logger, "Running preflight step: %s", step.description)
            step_runner(logger)
            steps_completed = result.setdefault("steps_completed", [])
            if isinstance(steps_completed, list):
                steps_completed.append(step.name)
        except Exception as exc:  # pragma: no cover - exercised via unit tests
            pause_event.set()
            if hasattr(logger, "exception"):
                logger.exception("Preflight step %s failed: %s", step.name, exc)
            context = {"step": step.name, "exception": str(exc)}
            _safe_queue_put(debug_queue, str(exc))
            _safe_queue_put(
                decision_queue,
                (step.failure_title, step.failure_message, context),
            )
            result["failures"] = [step.failure_message]
            result["error_step"] = step.name
            return result

    _log(logger, "Collecting sandbox health snapshot…")
    snapshot = sandbox_bootstrap.sandbox_health()
    result["snapshot"] = snapshot
    healthy, failures = _evaluate_health_snapshot(
        snapshot, dependency_mode=dependency_mode
    )
    result["healthy"] = healthy
    result["failures"] = failures
    if not healthy:
        for failure in failures:
            _safe_queue_put(debug_queue, failure)

    return result


class _TkStatusLogger:
    """Logger that proxies messages into the GUI status text widget."""

    def __init__(self, gui: "SandboxLauncherGUI") -> None:
        self._gui = gui

    def _format(self, message: str, args: tuple[object, ...]) -> str:
        return (message % args) if args else message

    def info(self, message: str, *args: object) -> None:
        self._gui._append_status_threadsafe(f"[INFO] {self._format(message, args)}\n")

    def warning(self, message: str, *args: object) -> None:  # pragma: no cover - UI nicety
        self._gui._append_status_threadsafe(f"[WARN] {self._format(message, args)}\n")

    def exception(self, message: str, *args: object) -> None:
        self._gui._append_status_threadsafe(f"[ERROR] {self._format(message, args)}\n")


class SandboxLauncherGUI(tk.Tk):
    """GUI window containing status output and sandbox controls."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._preflight_thread: Optional[threading.Thread] = None
        self.pause_event = threading.Event()
        self.abort_event = threading.Event()
        self.decision_queue: "queue.Queue[tuple[str, str, Optional[dict[str, object]]]]" = queue.Queue()
        self.debug_queue: "queue.Queue[str]" = queue.Queue()
        self.logger = _TkStatusLogger(self)
        self._pause_dialog_open = False
        self._pending_retry = False
        self._last_decision_payload: Optional[
            tuple[str, str, Optional[dict[str, object]]]
        ] = None

        self._configure_window()
        self._build_layout()
        self.after(200, self._poll_runtime_events)

    def _configure_window(self) -> None:
        """Configure the top-level window attributes."""

        self.title("Windows Sandbox Launcher")
        self.geometry("640x480")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def _build_layout(self) -> None:
        """Create the notebook, status tab, and control buttons."""

        container = ttk.Frame(self, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(container)
        notebook.grid(row=0, column=0, sticky="nsew")

        status_frame = ttk.Frame(notebook, padding=12)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        self.status_text = tk.Text(status_frame, wrap="word", height=10, state="disabled")
        self.status_text.grid(row=0, column=0, sticky="nsew")

        notebook.add(status_frame, text="Status")

        controls = self._create_controls(container)
        controls.grid(row=1, column=0, pady=(12, 0), sticky="ew")

    def _create_controls(self, parent: ttk.Frame) -> ttk.Frame:
        """Return a frame containing the preflight and sandbox buttons."""

        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        self.preflight_button = ttk.Button(
            frame,
            text="Run Preflight",
            command=self._on_run_preflight,
        )
        self.preflight_button.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.start_button = ttk.Button(frame, text="Start Sandbox", state="disabled")
        self.start_button.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        self.retry_button = ttk.Button(
            frame,
            text="Retry Step",
            command=self._on_retry_step,
            state="disabled",
        )
        self.retry_button.grid(row=0, column=2, padx=(6, 0), sticky="ew")

        return frame

    def _append_status(self, message: str) -> None:
        self.status_text.configure(state="normal")
        self.status_text.insert("end", message)
        self.status_text.see("end")
        self.status_text.configure(state="disabled")

    def _append_status_threadsafe(self, message: str) -> None:
        self.after(0, self._append_status, message)

    def _on_run_preflight(self) -> None:
        """Kick off the preflight routine in a background thread."""

        if self._preflight_thread and self._preflight_thread.is_alive():
            return

        self._reset_runtime_state()
        self._set_controls_state(running=True)
        self.logger.info("Starting sandbox preflight…")

        thread = threading.Thread(target=self._run_preflight_worker, daemon=False)
        self._preflight_thread = thread
        thread.start()

    def _reset_runtime_state(self) -> None:
        """Reset queues and events to their default state for a new run."""

        self.pause_event.clear()
        self.abort_event.clear()
        self._drain_queue(self.decision_queue)
        self._drain_queue(self.debug_queue)
        self._pending_retry = False
        self._last_decision_payload = None
        self.retry_button.configure(state="disabled")

    def _run_preflight_worker(self) -> None:
        """Execute ``run_full_preflight`` in the background."""

        result: Optional[dict[str, object]] = None
        error: Optional[BaseException] = None
        try:
            result = run_full_preflight(
                logger=self.logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.abort_event,
                debug_queue=self.debug_queue,
            )
        except BaseException as exc:  # pragma: no cover - defensive
            error = exc
            _safe_queue_put(self.debug_queue, str(exc))
        finally:
            self.after(0, self._handle_preflight_completion, result, error)

    def _handle_preflight_completion(
        self,
        result: Optional[dict[str, object]],
        error: Optional[BaseException],
    ) -> None:
        """Update UI state when the background preflight finishes."""

        self._preflight_thread = None
        self._set_controls_state(running=False)

        if error is not None:
            self.logger.exception("Preflight crashed: %s", error)
            self.start_button.config(state="disabled")
            return

        if not result:
            self.start_button.config(state="disabled")
            return

        if result.get("aborted"):
            self.logger.info("Preflight aborted.")
            self.start_button.config(state="disabled")
            return

        if result.get("healthy"):
            self.logger.info("Preflight completed successfully.")
            self.start_button.config(state="normal")
        else:
            self.logger.warning("Preflight completed with issues.")
            for failure in result.get("failures", []):
                self.logger.warning(failure)
            self.start_button.config(state="disabled")

    @staticmethod
    def _drain_queue(target_queue: queue.Queue) -> None:
        while True:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                break

    def _set_controls_state(self, *, running: bool) -> None:
        if running:
            self.preflight_button.configure(state="disabled")
            self.start_button.configure(state="disabled")
        else:
            self.preflight_button.configure(state="normal")

    def _poll_runtime_events(self) -> None:
        """Poll runtime coordination primitives for UI updates."""

        if self.pause_event.is_set():
            self.retry_button.configure(state="normal")
            if not self._pause_dialog_open:
                payload = self._dequeue_decision_payload()
                if payload:
                    self._show_pause_prompt(payload)
        else:
            if not self._pending_retry:
                self.retry_button.configure(state="disabled")
            self._pause_dialog_open = False

        self.after(200, self._poll_runtime_events)

    def _dequeue_decision_payload(
        self,
    ) -> Optional[tuple[str, str, Optional[dict[str, object]]]]:
        """Return the most recent decision payload, if any."""

        try:
            payload = self.decision_queue.get_nowait()
        except queue.Empty:
            payload = self._last_decision_payload

        if payload:
            self._last_decision_payload = payload
        return payload

    def _show_pause_prompt(
        self, payload: tuple[str, str, Optional[dict[str, object]]]
    ) -> None:
        """Display an interactive prompt for handling a paused preflight."""

        title, message, context = payload
        context = context or {}
        self._pause_dialog_open = True
        prompt_message = f"{message}\n\nRetry the failed step now?"
        decision = messagebox.askyesno(title=title, message=prompt_message)
        if decision:
            step_name = context.get("step", "unknown step")
            self.logger.info("Retrying preflight step: %s", step_name)
            self._resume_preflight()
        else:
            self.logger.info("Aborting preflight after failure.")
            self.abort_event.set()
            self.retry_button.configure(state="disabled")
        self._pause_dialog_open = False

    def _on_retry_step(self) -> None:
        """Retry the failed preflight step after user intervention."""

        if not self.pause_event.is_set() and not self._last_decision_payload:
            return
        self.logger.info("Retrying failed preflight step…")
        self._resume_preflight()

    def _resume_preflight(self) -> None:
        """Clear pause state and schedule a fresh preflight run."""

        self.pause_event.clear()
        self._pending_retry = True
        self.retry_button.configure(state="disabled")
        self.after(200, self._attempt_retry_start)

    def _attempt_retry_start(self) -> None:
        """Start a retry once the prior preflight thread has exited."""

        if self._preflight_thread and self._preflight_thread.is_alive():
            self.after(200, self._attempt_retry_start)
            return

        if not self._pending_retry:
            return

        self._pending_retry = False
        self._on_run_preflight()

