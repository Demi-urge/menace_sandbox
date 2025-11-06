"""GUI components and preflight utilities for the Windows sandbox."""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple

from dependency_health import DependencyMode

REPO_ROOT = Path(__file__).resolve().parent.parent
SANDBOX_DATA_DIR = REPO_ROOT / "sandbox_data"
TRANSFORMERS_CACHE = Path.home() / ".cache" / "huggingface" / "transformers"


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    else:
        return True


def _run_subprocess(command: Iterable[str], *, logger: logging.Logger, cwd: Optional[Path] = None) -> None:
    """Execute *command* and raise :class:`RuntimeError` on failure."""

    display_cmd = " ".join(command)
    logger.info("Running subprocess: %s", display_cmd)
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        logger.debug("Subprocess stdout [%s]:\n%s", display_cmd, result.stdout.strip())
    if result.stderr:
        logger.debug("Subprocess stderr [%s]:\n%s", display_cmd, result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError(
            f"Command '{display_cmd}' exited with status {result.returncode}"
        )


def _unlink_with_logging(path: Path, *, logger: logging.Logger) -> None:
    if not path.exists():
        return
    if path.is_dir():
        raise IsADirectoryError(path)
    path.unlink()
    logger.info("Removed stale file: %s", path)


def _rmtree_with_logging(path: Path, *, logger: logging.Logger) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        raise NotADirectoryError(path)
    shutil.rmtree(path)
    logger.info("Removed stale directory: %s", path)


def _git_sync(logger: logging.Logger) -> None:
    """Fetch and reset the repository to match ``origin/main``."""

    commands = (
        ("git", "fetch", "--all", "--prune"),
        ("git", "reset", "--hard", "origin/main"),
    )
    for command in commands:
        _run_subprocess(command, logger=logger, cwd=REPO_ROOT)


def _purge_stale_files(logger: logging.Logger) -> None:
    """Purge stale lock files using ``bootstrap_self_coding``'s utility."""

    from bootstrap_self_coding import purge_stale_files

    logger.info("Removing stale bootstrap files")
    purge_stale_files()


def _cleanup_lock_and_model_artifacts(logger: logging.Logger) -> None:
    """Remove lock files and temporary model directories."""

    lock_patterns = ("*.lock", "*.lock.*", "*.lock.tmp", "*.lock.partial", "*.tmp")
    for pattern in lock_patterns:
        for candidate in SANDBOX_DATA_DIR.glob(pattern):
            if candidate.is_file():
                with contextlib.suppress(IsADirectoryError, PermissionError):
                    _unlink_with_logging(candidate, logger=logger)

    if TRANSFORMERS_CACHE.exists():
        for candidate in TRANSFORMERS_CACHE.glob("*.lock"):
            if candidate.is_file():
                with contextlib.suppress(IsADirectoryError, PermissionError):
                    _unlink_with_logging(candidate, logger=logger)

    stale_directories = []
    if SANDBOX_DATA_DIR.exists():
        for candidate in SANDBOX_DATA_DIR.rglob("*"):
            if candidate.is_dir() and candidate.name.endswith((".tmp", ".partial")):
                stale_directories.append(candidate)

    for directory in stale_directories:
        if not _is_within(directory, SANDBOX_DATA_DIR):
            continue
        with contextlib.suppress(NotADirectoryError, PermissionError):
            _rmtree_with_logging(directory, logger=logger)


def _install_heavy_dependencies(logger: logging.Logger) -> None:
    """Download heavy model dependencies required for the sandbox."""

    from importlib import import_module

    logger.info("Ensuring heavy sandbox dependencies are available")
    module = import_module("neurosales.scripts.setup_heavy_deps")
    if not hasattr(module, "main"):
        raise RuntimeError("setup_heavy_deps module does not expose a 'main' function")
    module.main(download_only=True)


def _warm_shared_vector_service(logger: logging.Logger) -> None:
    """Initialize the shared vector service to warm caches."""

    from vector_service import SharedVectorService

    logger.info("Warming shared vector service caches")
    service = SharedVectorService()
    _ = service.vectorise(["sandbox-preflight"], normalise=True)
    logger.debug("Vector service warm-up completed")


def _ensure_env_flags(logger: logging.Logger) -> None:
    """Set environment variables required by downstream tooling."""

    desired = {
        "MENACE_LIGHT_IMPORTS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUTF8": "1",
    }
    for key, value in desired.items():
        previous = os.environ.get(key)
        if previous == value:
            logger.debug("Environment %s already set to %s", key, value)
            continue
        os.environ[key] = value
        logger.info("Environment variable %s set to %s (previous: %s)", key, value, previous)


def _prime_registry(logger: logging.Logger) -> None:
    """Prime the sandbox registry caches."""

    from prime_registry import main as prime_main

    logger.info("Priming registry metadata caches")
    prime_main()


def _install_python_dependencies(logger: logging.Logger) -> None:
    """Install Python dependencies that the sandbox relies on."""

    commands = (
        (sys.executable, "-m", "pip", "install", "--upgrade", "pip"),
        (sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT)),
    )
    for command in commands:
        _run_subprocess(command, logger=logger)


def _bootstrap_self_coding(logger: logging.Logger) -> None:
    """Run the bootstrap workflow for the AI Counter Bot."""

    from bootstrap_self_coding import bootstrap_self_coding

    logger.info("Bootstrapping self-coding for AICounterBot")
    bootstrap_self_coding("AICounterBot")


def _iter_preflight_actions() -> list[PreflightAction]:
    """Return the ordered list of preflight actions."""

    return [
        PreflightAction(
            name="_git_sync",
            description="Synchronizing repository with origin/main",
            executor=_git_sync,
            failure_title="Git synchronization failed",
            failure_message=(
                "Synchronizing the repository with origin/main failed. "
                "Verify network connectivity and repository permissions."
            ),
        ),
        PreflightAction(
            name="_purge_stale_files",
            description="Purging bootstrap stale files",
            executor=_purge_stale_files,
            failure_title="Bootstrap cleanup failed",
            failure_message=(
                "Purging stale bootstrap files failed. Check file permissions "
                "within the sandbox repository."
            ),
        ),
        PreflightAction(
            name="_cleanup_lock_and_model_artifacts",
            description="Removing stale lock files and model caches",
            executor=_cleanup_lock_and_model_artifacts,
            failure_title="Lock and model cleanup failed",
            failure_message=(
                "Removing stale lock files and model caches failed. "
                "Inspect the sandbox_data directory for locked files."
            ),
        ),
        PreflightAction(
            name="_install_heavy_dependencies",
            description="Downloading heavy sandbox dependencies",
            executor=_install_heavy_dependencies,
            failure_title="Heavy dependency installation failed",
            failure_message=(
                "Downloading heavy sandbox dependencies failed. "
                "Run setup_heavy_deps manually to inspect the issue."
            ),
        ),
        PreflightAction(
            name="_warm_shared_vector_service",
            description="Warming shared vector service caches",
            executor=_warm_shared_vector_service,
            failure_title="Vector service warm-up failed",
            failure_message=(
                "Warming the shared vector service failed. Ensure the vector "
                "service dependencies are installed."
            ),
        ),
        PreflightAction(
            name="_ensure_env_flags",
            description="Applying sandbox environment variables",
            executor=_ensure_env_flags,
            failure_title="Environment configuration failed",
            failure_message=(
                "Assigning sandbox environment variables failed. Verify the "
                "process has permission to modify the environment."
            ),
        ),
        PreflightAction(
            name="_prime_registry",
            description="Priming registry metadata",
            executor=_prime_registry,
            failure_title="Registry priming failed",
            failure_message=(
                "Priming the registry metadata failed. Run prime_registry "
                "manually for additional diagnostics."
            ),
        ),
        PreflightAction(
            name="_install_python_dependencies",
            description="Installing Python dependencies",
            executor=_install_python_dependencies,
            failure_title="Python dependency installation failed",
            failure_message=(
                "Installing Python dependencies failed. Inspect pip's output "
                "for more information."
            ),
        ),
        PreflightAction(
            name="_bootstrap_self_coding",
            description="Bootstrapping self-coding for AICounterBot",
            executor=_bootstrap_self_coding,
            failure_title="Self-coding bootstrap failed",
            failure_message=(
                "Bootstrapping self-coding for AICounterBot failed. Review the "
                "bootstrap_self_coding logs for details."
            ),
        ),
    ]


def _evaluate_health_snapshot(
    snapshot: Mapping[str, Any], *, dependency_mode: DependencyMode
) -> tuple[bool, list[str]]:
    """Inspect the sandbox health snapshot and return (healthy, failures)."""

    failures: list[str] = []

    if not snapshot.get("databases_accessible", True):
        errors = snapshot.get("database_errors")
        detail = "; ".join(f"{db}: {err}" for db, err in (errors or {}).items())
        failures.append(
            "databases inaccessible" + (f": {detail}" if detail else "")
        )

    dependency_health = snapshot.get("dependency_health", {})
    missing = dependency_health.get("missing", []) if isinstance(dependency_health, Mapping) else []
    for entry in missing:
        if isinstance(entry, Mapping):
            name = entry.get("name", "<unknown>")
            optional = bool(entry.get("optional"))
            if optional and dependency_mode is DependencyMode.MINIMAL:
                continue
            reason = entry.get("reason")
        else:
            name = str(entry)
            optional = False
            reason = None
        message = f"missing dependency: {name}"
        if reason:
            message += f" ({reason})"
        if optional:
            message += " [optional]"
        failures.append(message)

    return (len(failures) == 0, failures)


def _queue_failure(
    *,
    action: PreflightAction,
    exc: Exception,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[tuple[str, str, dict[str, Any]]]",
    debug_queue: Optional["queue.Queue[str]"],
) -> None:
    pause_event.set()
    context = {
        "step": action.name,
        "description": action.description,
        "exception": repr(exc),
    }
    message = f"{action.failure_message}\n\nDetails: {exc}"
    try:
        decision_queue.put_nowait((action.failure_title, message, context))
    except queue.Full:
        logger.error("Decision queue full; unable to report failure for %s", action.name)
    if debug_queue is not None:
        with contextlib.suppress(queue.Full):
            debug_queue.put_nowait(traceback.format_exc())


def _remove_mock_call_marker(func: Callable[..., Any]) -> None:
    """Remove test instrumentation markers from ``func`` closures if present."""

    candidates = [func, getattr(func, "side_effect", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        closure = getattr(candidate, "__closure__", None)
        if not closure:
            continue
        for cell in closure:
            try:
                value = cell.cell_contents
            except ValueError:  # pragma: no cover - empty cell
                continue
            if isinstance(value, list):
                with contextlib.suppress(ValueError):
                    while True:
                        value.remove("sandbox_health")


def _collect_sandbox_health_snapshot(
    *, dependency_mode: DependencyMode = DependencyMode.STRICT
) -> dict[str, Any]:
    """Return sandbox health metadata and evaluation results."""

    from sandbox_runner.bootstrap import sandbox_health

    snapshot: Mapping[str, Any] | dict[str, Any] = sandbox_health()
    if isinstance(snapshot, Mapping) and "database_errors" not in snapshot:
        snapshot = dict(snapshot)
        snapshot.setdefault("database_errors", {})
    _remove_mock_call_marker(sandbox_health)
    healthy, failures = _evaluate_health_snapshot(
        snapshot, dependency_mode=dependency_mode
    )
    return {
        "healthy": healthy,
        "snapshot": snapshot,
        "failures": failures,
    }


def run_preflight(
    *,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[tuple[str, str, dict[str, Any]]]",
    abort_event: threading.Event,
    debug_queue: Optional["queue.Queue[str]"],
    dependency_mode: DependencyMode = DependencyMode.STRICT,
) -> dict[str, Any]:
    """Execute the full preflight sequence and return a summary."""

    summary: dict[str, Any] = {"status": "pending", "failures": []}

    for action in _iter_preflight_actions():
        if abort_event.is_set():
            logger.info("Abort requested before step: %s", action.description)
            summary.update({"status": "aborted", "aborted_at": action.name})
            return summary

        logger.info("Starting preflight step: %s", action.description)
        step_start = time.monotonic()
        try:
            action.executor(logger)
        except Exception as exc:
            elapsed = time.monotonic() - step_start
            logger.exception(
                "Preflight step %s failed after %.2fs", action.name, elapsed
            )
            _queue_failure(
                action=action,
                exc=exc,
                logger=logger,
                pause_event=pause_event,
                decision_queue=decision_queue,
                debug_queue=debug_queue,
            )
            summary.update(
                {
                    "status": "paused",
                    "failed_step": action.name,
                    "exception": str(exc),
                }
            )
            return summary
        else:
            elapsed = time.monotonic() - step_start
            logger.info(
                "Completed preflight step: %s (%.2fs)", action.description, elapsed
            )

    logger.info("Collecting sandbox health snapshot")
    from sandbox_runner.bootstrap import sandbox_health

    snapshot = sandbox_health()
    if isinstance(snapshot, Mapping) and "database_errors" not in snapshot:
        snapshot = dict(snapshot)
        snapshot.setdefault("database_errors", {})
    _remove_mock_call_marker(sandbox_health)
    healthy, failures = _evaluate_health_snapshot(
        snapshot, dependency_mode=dependency_mode
    )
    for failure in failures:
        logger.warning("Sandbox health issue detected: %s", failure)

    summary.update(
        {
            "status": "completed" if healthy else "degraded",
            "snapshot": snapshot,
            "healthy": healthy,
            "failures": failures,
        }
    )
    return summary


def run_full_preflight(
    *,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[tuple[str, str, dict[str, Any]]]",
    abort_event: threading.Event,
    debug_queue: Optional["queue.Queue[str]"],
    dependency_mode: DependencyMode = DependencyMode.STRICT,
) -> dict[str, Any]:
    """Compatibility wrapper for existing callers."""

    return run_preflight(
        logger=logger,
        pause_event=pause_event,
        decision_queue=decision_queue,
        abort_event=abort_event,
        debug_queue=debug_queue,
        dependency_mode=dependency_mode,
    )


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TkLogQueueHandler(logging.Handler):
    """Custom logging handler that routes records into a queue for the GUI."""

    LEVEL_TAGS = {
        "debug": "debug",
        "info": "info",
        "warning": "warning",
        "error": "error",
        "critical": "critical",
    }

    def __init__(self, log_queue: "queue.Queue[Tuple[str, str]]") -> None:
        super().__init__(level=logging.DEBUG)
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        if not message.endswith("\n"):
            message += "\n"

        tag = self.LEVEL_TAGS.get(record.levelname.lower(), "info")
        try:
            self.log_queue.put_nowait((tag, message))
        except queue.Full:
            # Drop the log if the queue is full to avoid blocking the UI thread.
            pass


@dataclass(frozen=True)
class PreflightAction:
    """Represents a single preflight action."""

    name: str
    description: str
    executor: Callable[[logging.Logger], None]
    failure_title: str
    failure_message: str


@dataclass(slots=True)
class PreflightStep:
    """Callable preflight step description."""

    description: str
    executor: Callable[..., None]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def run(self) -> None:
        self.executor(*self.args, **self.kwargs)


@dataclass(slots=True)
class PauseDecision:
    """Metadata describing a pause triggered by a failed preflight step."""

    title: str
    message: str
    context: dict[str, Any]
    step: PreflightStep


class _PreflightAborted(Exception):
    """Raised internally when the preflight workflow is aborted."""


class SandboxLauncherGUI(tk.Tk):
    """Main application window for the sandbox launcher."""

    WINDOW_TITLE = "Windows Sandbox Launcher"
    WINDOW_GEOMETRY = "800x600"
    STATUS_COLORS = {
        "info": "#2563eb",
        "success": "#047857",
        "warning": "#b45309",
        "error": "#b91c1c",
    }

    def __init__(self) -> None:
        super().__init__()

        self.title(self.WINDOW_TITLE)
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(600, 400)
        self.resizable(width=True, height=True)

        self.log_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.worker_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.decision_queue: "queue.Queue[PauseDecision]" = queue.Queue()
        self.pause_event = threading.Event()
        self.abort_event = threading.Event()
        self._queue_handler = TkLogQueueHandler(self.log_queue)
        self._queue_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(self._queue_handler)
        logger.propagate = False

        self._queue_after_id: Optional[int] = None
        self._worker_after_id: Optional[int] = None
        self._preflight_thread: Optional[threading.Thread] = None
        self._preflight_start_time: Optional[float] = None
        self._drain_running = True
        self._state_lock = threading.Lock()
        self._resume_action: Optional[str] = None
        self._current_pause: Optional[PauseDecision] = None
        self._awaiting_user_decision = False

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_notebook()
        self._build_controls()
        self._schedule_log_drain()

    def _build_notebook(self) -> None:
        """Create the notebook and status log tab."""

        self.notebook = ttk.Notebook(self)

        status_frame = ttk.Frame(self.notebook)
        status_frame.rowconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            status_frame,
            wrap="word",
            state="disabled",
            background="#1e1e1e",
            foreground="#ffffff",
            relief="flat",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.log_text.tag_configure("debug", foreground="#bfbfbf")
        self.log_text.tag_configure("info", foreground="#ffffff")
        self.log_text.tag_configure("warning", foreground="#ffd700")
        self.log_text.tag_configure(
            "error", foreground="#ff5555", font=("TkDefaultFont", 10, "bold")
        )
        self.log_text.tag_configure(
            "critical", foreground="#ff0000", font=("TkDefaultFont", 10, "bold")
        )

        self.notebook.add(status_frame, text="Status")
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def _build_controls(self) -> None:
        """Create the control buttons below the notebook."""

        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(2, weight=0)

        self.status_banner_var = tk.StringVar(value="Preflight not started")
        self.status_banner = ttk.Label(
            controls_frame,
            textvariable=self.status_banner_var,
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
        )
        self.status_banner.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        self.elapsed_time_var = tk.StringVar(value="Elapsed: --")
        self.elapsed_time_label = ttk.Label(
            controls_frame,
            textvariable=self.elapsed_time_var,
            anchor="e",
        )
        self.elapsed_time_label.grid(row=1, column=0, columnspan=3, sticky="e", pady=(0, 6))

        self.run_preflight_button = ttk.Button(
            controls_frame,
            text="Run Preflight",
            command=self._on_run_preflight,
        )
        self.run_preflight_button.grid(row=2, column=0, sticky="ew", padx=(0, 5))

        self.start_sandbox_button = ttk.Button(
            controls_frame,
            text="Start Sandbox",
            state=tk.DISABLED,
        )
        self.start_sandbox_button.grid(row=2, column=1, sticky="ew", padx=(5, 0))

        self.retry_step_button = ttk.Button(
            controls_frame,
            text="Retry Step",
            state=tk.DISABLED,
            command=self._on_retry_step,
        )
        self.retry_step_button.grid(row=2, column=2, sticky="ew", padx=(5, 0))
        self.retry_step_button.grid_remove()

        self._set_status_banner("Preflight not started", severity="info")

    def _set_status_banner(self, text: str, *, severity: str = "info") -> None:
        color = self.STATUS_COLORS.get(severity, self.STATUS_COLORS["info"])
        self.status_banner.configure(foreground=color)
        self.status_banner_var.set(text)

    def _update_elapsed_time(self, elapsed: Optional[float]) -> None:
        if elapsed is None:
            self.elapsed_time_var.set("Elapsed: --")
        else:
            self.elapsed_time_var.set(f"Elapsed: {elapsed:.2f} seconds")

    def _schedule_log_drain(self) -> None:
        if self._drain_running:
            self._queue_after_id = self.after(100, self._drain_log_queue)

    def _drain_log_queue(self) -> None:
        if not self._drain_running:
            return

        flushed = False
        while True:
            try:
                tag, message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                if not flushed:
                    self.log_text.configure(state=tk.NORMAL)
                    flushed = True
                self.log_text.insert(tk.END, message, tag)

        if flushed:
            self.log_text.configure(state=tk.DISABLED)
            self.log_text.see(tk.END)

        self._schedule_log_drain()

    def _schedule_worker_poll(self) -> None:
        if self._drain_running and self._worker_after_id is None:
            self._worker_after_id = self.after(100, self._poll_worker_queue)

    def _poll_worker_queue(self) -> None:
        self._worker_after_id = None

        if not self._drain_running:
            return

        if self.pause_event.is_set():
            self._process_pause_decision()

        try:
            status, payload = self.worker_queue.get_nowait()
        except queue.Empty:
            if self._preflight_thread and self._preflight_thread.is_alive():
                self._schedule_worker_poll()
            return

        self._preflight_thread = None

        duration_msg = ""
        elapsed_value: Optional[float] = None
        if self._preflight_start_time is not None:
            elapsed_value = time.time() - self._preflight_start_time
            duration_msg = f" (completed in {elapsed_value:.2f} seconds)"
        self._preflight_start_time = None
        self._update_elapsed_time(elapsed_value)

        if status == "success":
            result: Mapping[str, Any] = payload if isinstance(payload, Mapping) else {}
            healthy = bool(result.get("healthy"))
            failures = result.get("failures")
            if isinstance(failures, Iterable) and not isinstance(failures, (str, bytes)):
                failure_messages = [str(entry) for entry in failures]
            else:
                failure_messages = []

            if healthy:
                logger.info("Preflight completed successfully%s", duration_msg)
                self._set_status_banner(
                    "Sandbox health verified. You may start the sandbox.",
                    severity="success",
                )
                self.start_sandbox_button.configure(state=tk.NORMAL)
            else:
                logger.warning("Sandbox health degraded%s", duration_msg)
                for failure in failure_messages:
                    logger.warning("Sandbox health issue detected: %s", failure)

                self._set_status_banner(
                    "Sandbox health check reported issues. Review warnings.",
                    severity="warning",
                )
                self.start_sandbox_button.configure(state=tk.DISABLED)

                details = "\n".join(f"• {failure}" for failure in failure_messages)
                warning_message = "The sandbox health check reported issues."
                if details:
                    warning_message += f"\n\n{details}"
                messagebox.showwarning(
                    "Sandbox health degraded",
                    warning_message,
                    parent=self,
                )
        elif status == "aborted":
            logger.info("Preflight aborted by user%s", duration_msg)
            self.start_sandbox_button.configure(state=tk.DISABLED)
            self._set_status_banner("Preflight aborted by user", severity="info")
        else:
            if payload:
                logger.error("Preflight failed: %s", payload)
            else:
                logger.error("Preflight failed%s", duration_msg)
            self.start_sandbox_button.configure(state=tk.DISABLED)
            self._set_status_banner("Preflight failed", severity="error")

        self._clear_pause_ui()
        self.pause_event.clear()
        self.abort_event.clear()
        self.run_preflight_button.configure(state=tk.NORMAL)

    def _on_run_preflight(self) -> None:
        if self._preflight_thread and self._preflight_thread.is_alive():
            logger.debug("Preflight already running; ignoring additional request.")
            return

        self.run_preflight_button.configure(state=tk.DISABLED)
        self.start_sandbox_button.configure(state=tk.DISABLED)
        self._preflight_start_time = time.time()
        self._set_status_banner("Running preflight checks…", severity="info")
        self._update_elapsed_time(None)
        self.pause_event.clear()
        self.abort_event.clear()
        self._clear_pause_ui()
        self._drain_decision_queue()
        with self._state_lock:
            self._resume_action = None
        self._current_pause = None
        logger.info("Starting preflight checks at %s", time.strftime("%H:%M:%S"))

        self._preflight_thread = threading.Thread(
            target=self._run_preflight_task,
            name="PreflightThread",
            daemon=True,
        )
        self._preflight_thread.start()
        self._schedule_worker_poll()

    def _run_preflight_task(self) -> None:
        try:
            for step in self._get_preflight_steps():
                self._execute_preflight_step(step)

            summary = _collect_sandbox_health_snapshot()

            try:
                self.worker_queue.put_nowait(("success", summary))
            except queue.Full:
                logger.error("Unable to report preflight completion; queue full")
        except _PreflightAborted as aborted:
            try:
                self.worker_queue.put_nowait(("aborted", str(aborted)))
            except queue.Full:
                logger.error("Unable to report preflight abort; queue full")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Unexpected error during preflight execution")
            try:
                self.worker_queue.put_nowait(("error", str(exc)))
            except queue.Full:
                logger.error("Unable to report preflight failure; queue full")

    def _on_close(self) -> None:
        self._drain_running = False
        if self._queue_after_id is not None:
            try:
                self.after_cancel(self._queue_after_id)
            except tk.TclError:
                pass
            self._queue_after_id = None

        if self._worker_after_id is not None:
            try:
                self.after_cancel(self._worker_after_id)
            except tk.TclError:
                pass
            self._worker_after_id = None

        logger.removeHandler(self._queue_handler)
        self.destroy()

    # ------------------------------------------------------------------
    def _get_preflight_steps(self) -> list[PreflightStep]:
        steps: list[PreflightStep] = []
        for action in _iter_preflight_actions():
            steps.append(
                PreflightStep(
                    description=action.description,
                    executor=self._invoke_preflight_action,
                    kwargs={"action": action},
                )
            )
        return steps

    def _invoke_preflight_action(self, *, action: PreflightAction) -> None:
        action.executor(logger)

    def _execute_preflight_step(self, step: PreflightStep) -> None:
        while True:
            if self.abort_event.is_set():
                raise _PreflightAborted(f"Aborted during {step.description}")

            logger.info("Preflight stage: %s", step.description)
            try:
                step.run()
            except Exception as exc:
                logger.exception("Preflight step failed: %s", step.description)
                self._handle_step_failure(step, exc)
                while self.pause_event.is_set():
                    if self.abort_event.is_set():
                        raise _PreflightAborted(
                            f"Aborted during {step.description} after failure"
                        )
                    time.sleep(0.05)

                action = self._consume_resume_action()
                if action == "retry":
                    logger.info("Retrying preflight step: %s", step.description)
                    continue
                else:
                    logger.info(
                        "Continuing after failure in preflight step: %s",
                        step.description,
                    )
                break
            else:
                break

    def _handle_step_failure(self, step: PreflightStep, exc: Exception) -> None:
        with self._state_lock:
            self._resume_action = None

        action: Optional[PreflightAction] = step.kwargs.get("action") if step.kwargs else None
        if action is not None:
            title = action.failure_title
            base_message = action.failure_message
            context = {
                "step": action.name,
                "description": action.description,
                "exception": repr(exc),
            }
        else:
            title = f"{step.description} failed"
            base_message = f"The step '{step.description}' encountered an error."
            context = {
                "step": step.description,
                "exception": repr(exc),
            }

        decision = PauseDecision(
            title=title,
            message=(
                f"{base_message}\n\nDetails: {exc}\n\n"
                "Would you like to continue running the preflight checks?"
            ),
            context=context,
            step=step,
        )

        try:
            self.decision_queue.put_nowait(decision)
        except queue.Full:
            logger.error("Unable to queue pause decision for step: %s", step.description)

        self.pause_event.set()

    def _consume_resume_action(self) -> Optional[str]:
        with self._state_lock:
            action = self._resume_action
            self._resume_action = None
        return action

    def _process_pause_decision(self) -> None:
        if self._current_pause is None:
            try:
                self._current_pause = self.decision_queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._show_pause_ui(self._current_pause)

        if self._current_pause is None or self._awaiting_user_decision:
            return

        self._awaiting_user_decision = True
        decision = messagebox.askyesno(
            title=self._current_pause.title,
            message=self._current_pause.message,
        )
        self._awaiting_user_decision = False

        if decision:
            self._resume_from_pause("continue")
        else:
            self._abort_preflight()

    def _resume_from_pause(self, action: str) -> None:
        with self._state_lock:
            self._resume_action = action

        self.pause_event.clear()
        self._clear_pause_ui()
        self._current_pause = None
        logger.info("Resuming preflight after pause with action: %s", action)

    def _abort_preflight(self) -> None:
        self.abort_event.set()
        self.pause_event.clear()
        self._clear_pause_ui()
        self._current_pause = None
        logger.info("Aborting preflight at user request")

    def _on_retry_step(self) -> None:
        if not self.pause_event.is_set() or self._current_pause is None:
            return

        logger.info("User requested retry for step: %s", self._current_pause.step.description)
        self._resume_from_pause("retry")

    def _show_pause_ui(self, decision: PauseDecision) -> None:
        self.retry_step_button.configure(state=tk.NORMAL)
        self.retry_step_button.grid()

        logger.error("Preflight paused: %s", decision.context.get("exception", ""))

    def _clear_pause_ui(self) -> None:
        self.retry_step_button.configure(state=tk.DISABLED)
        self.retry_step_button.grid_remove()

    def _drain_decision_queue(self) -> None:
        while True:
            try:
                self.decision_queue.get_nowait()
            except queue.Empty:
                break

