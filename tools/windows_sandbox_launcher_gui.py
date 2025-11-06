"""Windows sandbox preflight orchestration and launcher GUI."""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, TextIO

import tkinter as tk
from tkinter import messagebox, ttk

import auto_env_setup
import bootstrap_self_coding
import prime_registry
from dependency_health import DependencyMode
from sandbox_runner import bootstrap as sandbox_bootstrap
from vector_service import SharedVectorService


REPO_ROOT = Path(__file__).resolve().parents[1]

_TARGETED_CLEANUP_DIRECTORIES: tuple[Path, ...] = (
    REPO_ROOT / "sandbox_data",
    REPO_ROOT / "vector_service",
    REPO_ROOT / "logs",
)

_TARGETED_CLEANUP_PATTERNS: tuple[str, ...] = ("*.lock", "*.tmp", "*.wal", "*.shm")

_SUBPROCESS_LOGGER = logging.getLogger("windows_sandbox.preflight.subprocess")

_MAX_STEP_ATTEMPTS = 2


@dataclass(frozen=True)
class _PreflightStep:
    """Metadata describing a single preflight step."""

    name: str
    description: str
    failure_title: str
    failure_message: str
    max_attempts: int = 1


_PREFLIGHT_STEPS: tuple[_PreflightStep, ...] = (
    _PreflightStep(
        name="_git_sync",
        description="Synchronising repository state",
        failure_title="Git synchronisation failed",
        failure_message="Synchronising repository state",
        max_attempts=2,
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
        max_attempts=2,
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
        max_attempts=2,
    ),
    _PreflightStep(
        name="_install_python_dependencies",
        description="Installing Python dependencies",
        failure_title="Python dependency installation failed",
        failure_message="Installing Python dependencies",
        max_attempts=2,
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


def _warn(logger, message: str, *args: object) -> None:
    """Log ``message`` at warning level if possible."""

    if hasattr(logger, "warning"):
        logger.warning(message, *args)
    else:  # pragma: no cover - defensive: fall back to info logging
        _log(logger, message, *args)


def _log_subprocess_stream(prefix: str, payload: str | None, logger) -> None:
    """Emit subprocess ``payload`` lines with ``prefix`` to both loggers."""

    if not payload:
        return

    for line in payload.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        message = f"{prefix} {stripped}"
        _SUBPROCESS_LOGGER.info(message)
        _log(logger, "%s", message)


def _run_command(
    args: Sequence[str],
    logger,
    *,
    description: str | None = None,
    cwd: Path | str | None = None,
    retries: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Execute ``args`` capturing stdout/stderr and log all output."""

    attempt = 0
    last_error: subprocess.CalledProcessError | None = None
    while attempt <= retries:
        attempt += 1
        if description:
            if attempt == 1:
                _log(logger, "%s", description)
            else:
                _log(
                    logger,
                    "%s (retry %d/%d)",
                    description,
                    attempt,
                    retries + 1,
                )
        try:
            completed = subprocess.run(
                list(args),
                cwd=cwd,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            last_error = exc
            _log_subprocess_stream("[stdout]", exc.stdout, logger)
            _log_subprocess_stream("[stderr]", exc.stderr, logger)
            if attempt <= retries:
                _warn(
                    logger,
                    "Command %s failed with exit code %s; retrying…",
                    args[0],
                    exc.returncode,
                )
                continue
            raise RuntimeError(
                f"Command {' '.join(args)} failed with exit code {exc.returncode}"
            ) from exc
        else:
            _log_subprocess_stream("[stdout]", completed.stdout, logger)
            _log_subprocess_stream("[stderr]", completed.stderr, logger)
            return completed

    if last_error is not None:  # pragma: no cover - defensive
        raise RuntimeError(str(last_error))

    raise RuntimeError("Subprocess execution aborted before start")  # pragma: no cover


def _safe_remove_path(path: Path, logger) -> None:
    """Remove ``path`` if it exists, logging successes and failures."""

    try:
        if path.is_dir():
            shutil.rmtree(path)
            _log(logger, "Removed directory: %s", path)
        elif path.exists():
            path.unlink()
            _log(logger, "Removed file: %s", path)
    except FileNotFoundError:  # pragma: no cover - defensive
        return
    except OSError as exc:
        _warn(logger, "Failed to remove %s: %s", path, exc)


def _safe_queue_put(target_queue: queue.Queue, payload) -> None:
    """Insert ``payload`` into ``target_queue`` ignoring ``queue.Full`` errors."""

    try:
        target_queue.put_nowait(payload)
    except queue.Full:  # pragma: no cover - defensive
        pass


def _git_sync(logger) -> None:
    """Synchronise git repositories."""

    repo_cwd = str(REPO_ROOT)
    _log(logger, "Synchronising git repositories with remote state…")
    _run_command(
        ["git", "fetch", "--all", "--prune"],
        logger,
        description="Fetching remote references",
        cwd=repo_cwd,
        retries=1,
    )
    upstream: str | None = None
    try:
        result = _run_command(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            logger,
            description="Determining upstream branch",
            cwd=repo_cwd,
        )
        upstream = result.stdout.strip()
    except RuntimeError:
        _warn(
            logger,
            "Unable to determine upstream branch; defaulting to origin/HEAD",
        )
        upstream = "origin/HEAD"

    if not upstream:
        upstream = "origin/HEAD"

    _run_command(
        ["git", "reset", "--hard", upstream],
        logger,
        description=f"Resetting local branch to {upstream}",
        cwd=repo_cwd,
        retries=1,
    )


def _purge_stale_files(logger) -> None:
    """Remove stale bootstrap artifacts."""

    _log(logger, "Purging stale bootstrap artifacts…")
    bootstrap_self_coding.purge_stale_files()


def _cleanup_lock_and_model_artifacts(logger) -> None:
    """Remove stale lock files and cached model artifacts."""

    _log(logger, "Cleaning up stale lock files and cached models…")
    for directory in _TARGETED_CLEANUP_DIRECTORIES:
        if not directory.exists():
            continue
        for pattern in _TARGETED_CLEANUP_PATTERNS:
            for path in directory.glob(pattern):
                _safe_remove_path(path, logger)


def _install_heavy_dependencies(logger) -> None:
    """Install large dependency bundles required for sandbox startup."""

    try:
        from neurosales.scripts import setup_heavy_deps
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError("Heavy dependency installer not available") from exc

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
    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
    os.environ.setdefault("MENACE_SANDBOX_PREPARED", "1")
    _log(logger, "Environment flags set: MENACE_LIGHT_IMPORTS, MENACE_SANDBOX_PREPARED")


def _prime_registry(logger) -> None:
    """Prime registry information consumed by the sandbox."""

    _log(logger, "Priming registry data…")
    prime_registry.main()


def _install_python_dependencies(logger) -> None:
    """Install Python dependencies required for the sandbox runtime."""

    _log(logger, "Validating Python dependency installation…")
    requirements = REPO_ROOT / "requirements.txt"
    if not requirements.exists():
        _warn(logger, "No requirements.txt found at %s; skipping pip install", requirements)
        return

    _run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "-r",
            str(requirements),
        ],
        logger,
        description="Installing Python requirements",
        cwd=str(REPO_ROOT),
        retries=1,
    )


def _bootstrap_self_coding(logger) -> None:
    """Bootstrap the self-coding environment used during sandbox runs."""

    _log(logger, "Bootstrapping self-coding environment…")
    bootstrap_self_coding.bootstrap_self_coding("AICounterBot")


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
        max_attempts = max(1, min(step.max_attempts, _MAX_STEP_ATTEMPTS))
        attempt = 0
        while attempt < max_attempts:
            attempt += 1
            try:
                _log(
                    logger,
                    "Running preflight step: %s (attempt %d/%d)",
                    step.description,
                    attempt,
                    max_attempts,
                )
                step_runner(logger)
            except Exception as exc:  # pragma: no cover - exercised via unit tests
                if attempt < max_attempts:
                    _warn(
                        logger,
                        "Preflight step %s failed on attempt %d/%d: %s",
                        step.name,
                        attempt,
                        max_attempts,
                        exc,
                    )
                    _safe_queue_put(debug_queue, str(exc))
                    continue

                pause_event.set()
                if hasattr(logger, "exception"):
                    logger.exception("Preflight step %s failed: %s", step.name, exc)
                context = {
                    "step": step.name,
                    "exception": str(exc),
                    "attempts": attempt,
                }
                _safe_queue_put(debug_queue, str(exc))
                _safe_queue_put(
                    decision_queue,
                    (step.failure_title, step.failure_message, context),
                )
                result["failures"] = [step.failure_message]
                result["error_step"] = step.name
                return result
            else:
                _log(logger, "Completed preflight step: %s", step.description)
                steps_completed = result.setdefault("steps_completed", [])
                if isinstance(steps_completed, list):
                    steps_completed.append(step.name)
                break

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
        self._sandbox_thread: Optional[threading.Thread] = None
        self._sandbox_process: Optional[subprocess.Popen[str]] = None
        self._button_states_before_launch: Optional[dict[str, str]] = None

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

        self.start_button = ttk.Button(
            frame,
            text="Start Sandbox",
            state="disabled",
            command=self._on_start_sandbox,
        )
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

        preflight_healthy = bool(result.get("healthy"))
        if preflight_healthy:
            self.logger.info("Preflight completed successfully.")
        else:
            self.logger.warning("Preflight completed with issues.")
            for failure in result.get("failures", []):
                self.logger.warning(failure)

        try:
            snapshot = sandbox_bootstrap.sandbox_health()
        except Exception as exc:  # pragma: no cover - defensive UI handling
            self.logger.exception("Sandbox health verification failed: %s", exc)
            messagebox.showwarning(
                title="Sandbox Health Check Failed",
                message=(
                    "Unable to verify sandbox health.\n\n"
                    f"Details: {exc}"
                ),
            )
            self.start_button.config(state="disabled")
            return

        healthy, failures = _evaluate_health_snapshot(
            snapshot, dependency_mode=DependencyMode.STRICT
        )

        if healthy and preflight_healthy:
            self.logger.info("Sandbox health verification succeeded.")
            self.start_button.config(state="normal")
            return

        self.start_button.config(state="disabled")
        if healthy:
            self.logger.info(
                "Sandbox health verification succeeded, but preflight reported issues."
            )
            return

        self.logger.warning("Sandbox health check reported issues.")
        for failure in failures:
            self.logger.warning(failure)
        issues = "\n".join(f"- {failure}" for failure in failures) or "Unknown issues"
        messagebox.showwarning(
            title="Sandbox Health Issues Detected",
            message=(
                "The sandbox environment reported health issues:\n\n"
                f"{issues}"
            ),
        )

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

    def _on_start_sandbox(self) -> None:
        """Launch the autonomous sandbox in a background process."""

        if self._sandbox_thread and self._sandbox_thread.is_alive():
            self.logger.info("Sandbox launch already in progress.")
            return

        if self._sandbox_process and self._sandbox_process.poll() is None:
            self.logger.info("Sandbox process already running.")
            return

        self._button_states_before_launch = {
            "preflight": self.preflight_button.cget("state"),
            "start": self.start_button.cget("state"),
            "retry": self.retry_button.cget("state"),
        }
        self.preflight_button.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.retry_button.configure(state="disabled")
        self.logger.info("Launching sandbox process…")

        thread = threading.Thread(target=self._launch_sandbox_worker, daemon=True)
        self._sandbox_thread = thread
        thread.start()

    def _launch_sandbox_worker(self) -> None:
        """Spawn the sandbox process and stream its output."""

        command = [
            sys.executable,
            "-c",
            "import start_autonomous_sandbox as _sas; _sas.main([])",
        ]

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # pragma: no cover - defensive UI handling
            self.after(0, self._handle_sandbox_launch_failure, exc)
            return

        self._sandbox_process = process

        stream_threads: list[threading.Thread] = []
        if process.stdout is not None:
            stdout_thread = threading.Thread(
                target=self._read_process_stream,
                args=(process.stdout, "stdout"),
                daemon=True,
            )
            stream_threads.append(stdout_thread)
            stdout_thread.start()

        if process.stderr is not None:
            stderr_thread = threading.Thread(
                target=self._read_process_stream,
                args=(process.stderr, "stderr"),
                daemon=True,
            )
            stream_threads.append(stderr_thread)
            stderr_thread.start()

        return_code = process.wait()

        for thread in stream_threads:
            thread.join(timeout=1)

        self.after(0, self._handle_sandbox_exit, return_code)

    def _read_process_stream(self, stream: TextIO, stream_name: str) -> None:
        """Forward sandbox output streams into the GUI logger and debug queue."""

        try:
            for line in iter(stream.readline, ""):
                message = line.rstrip()
                if not message:
                    continue
                formatted = f"[sandbox {stream_name}] {message}"
                self.logger.info(formatted)
                _safe_queue_put(self.debug_queue, formatted)
        finally:
            try:
                stream.close()
            except Exception:  # pragma: no cover - defensive close
                pass

    def _handle_sandbox_launch_failure(self, error: BaseException) -> None:
        """Handle failures that occur while starting the sandbox process."""

        self._sandbox_thread = None
        self._sandbox_process = None
        self.logger.error("Failed to start sandbox: %s", error)
        _safe_queue_put(self.debug_queue, str(error))
        self._restore_controls_after_launch()

    def _handle_sandbox_exit(self, return_code: int | None) -> None:
        """Restore UI state when the sandbox process terminates."""

        self._sandbox_thread = None
        self._sandbox_process = None

        if return_code == 0:
            self.logger.info("Sandbox exited successfully.")
        else:
            self.logger.warning("Sandbox exited with code %s.", return_code)

        self._restore_controls_after_launch()

    def _restore_controls_after_launch(self) -> None:
        """Restore button states captured before sandbox launch."""

        previous_states = self._button_states_before_launch or {}
        self.preflight_button.configure(state=previous_states.get("preflight", "normal"))
        self.start_button.configure(state=previous_states.get("start", "disabled"))
        self.retry_button.configure(state=previous_states.get("retry", "disabled"))
        self._button_states_before_launch = None


def main() -> None:
    """Launch the sandbox GUI."""

    SandboxLauncherGUI().mainloop()


if __name__ == "__main__":
    SandboxLauncherGUI().mainloop()

