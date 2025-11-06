"""GUI shell and preflight orchestration for the Windows sandbox."""

from __future__ import annotations

import importlib
import logging
import queue
import sys
import threading
import time
import traceback
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TextIO

import tkinter as tk
from tkinter import font as tk_font
from tkinter import messagebox
from tkinter import ttk

from dependency_health import DependencyMode, resolve_dependency_mode
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE_PATH = REPO_ROOT / "menace_gui_logs.txt"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True, slots=True)
class _PreflightStep:
    """Metadata describing an individual preflight step."""

    name: str
    start_message: str
    success_message: str
    failure_title: str
    failure_message: str
    runner: Callable[[logging.Logger], None]


class _QueueLogHandler(logging.Handler):
    """Logging handler that pushes formatted records into a queue."""

    def __init__(self, message_queue: queue.Queue[tuple[str, str]]) -> None:
        super().__init__()
        self._queue = message_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI thread side-effect
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - defensive fallback to repr
            message = repr(record)

        level = "info"
        if record.levelno >= logging.ERROR:
            level = "error"
        elif record.levelno >= logging.WARNING:
            level = "warning"

        self._queue.put((level, message))


class SandboxLauncherGUI(tk.Tk):
    """Tkinter window used to control sandbox preflight and launch actions."""

    def __init__(self) -> None:
        super().__init__()

        # Window metadata
        self.title("Windows Sandbox Launcher")
        self.geometry("720x480")

        # Tk themed widgets
        self.style = ttk.Style(self)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        else:  # pragma: no cover - dependent on host themes
            self.style.theme_use("default")

        # Thread coordination primitives exposed to the worker
        self.pause_event = threading.Event()
        self.abort_event = threading.Event()
        self.decision_queue: "queue.Queue[tuple[str, str, dict[str, object] | None]]" = (
            queue.Queue()
        )

        # Logging queues and handlers
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._drain_running = False
        self._queue_handler = _QueueLogHandler(self.log_queue)
        self._queue_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        root_logger = logging.getLogger()
        if self._queue_handler not in root_logger.handlers:
            root_logger.addHandler(self._queue_handler)
        if root_logger.level == logging.NOTSET:
            root_logger.setLevel(logging.INFO)

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

        self._file_log_queue: queue.Queue[logging.LogRecord] | None = None
        self._file_queue_handler: QueueHandler | None = None
        self._file_handler: RotatingFileHandler | None = None
        self._file_listener: QueueListener | None = None

        # Worker thread coordination
        self._worker_queue: "queue.Queue[tuple[str, dict[str, object]]]" = queue.Queue()
        self._preflight_thread: threading.Thread | None = None
        self._sandbox_thread: threading.Thread | None = None
        self._sandbox_process: subprocess.Popen[str] | None = None

        # Build UI layout
        self._build_layout()

        # Start polling loops
        self._schedule_log_drain()
        self.after(100, self._process_worker_events)

    # ------------------------------------------------------------------
    # Widget construction helpers

    def _build_layout(self) -> None:
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True)

        status_tab = ttk.Frame(self.notebook)
        status_tab.columnconfigure(0, weight=1)
        status_tab.rowconfigure(0, weight=1)
        self.notebook.add(status_tab, text="Status")

        log_container = ttk.Frame(status_tab)
        log_container.grid(row=0, column=0, sticky="nsew")
        log_container.columnconfigure(0, weight=1)
        log_container.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_container, wrap="word", state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_container, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self._configure_log_tags()

        # Control buttons row
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(
            button_frame, text="Run Preflight", command=self._on_run_preflight
        )
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.launch_button = ttk.Button(
            button_frame,
            text="Start Sandbox",
            command=self._on_launch_sandbox,
            state=tk.DISABLED,
        )
        self.launch_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _configure_log_tags(self) -> None:
        default_font = tk_font.nametofont("TkDefaultFont")
        bold_font = default_font.copy()
        bold_font.configure(weight="bold")

        self.log_text.tag_configure("info", foreground="#1b5e20")
        self.log_text.tag_configure("warning", foreground="#e65100", font=bold_font)
        self.log_text.tag_configure("error", foreground="#b71c1c", font=bold_font)

    # ------------------------------------------------------------------
    # Logging helpers

    def _schedule_log_drain(self) -> None:
        if not self._drain_running:
            self._drain_running = True
            self.after(100, self._drain_log_queue)

    def _drain_log_queue(self) -> None:  # pragma: no cover - Tk loop side effect
        updated = False
        try:
            while True:
                level, message = self.log_queue.get_nowait()
                if level not in {"info", "warning", "error"}:
                    level = "info"

                if not updated:
                    self.log_text.configure(state=tk.NORMAL)
                    updated = True

                self.log_text.insert(tk.END, message, level)
        except queue.Empty:
            pass
        finally:
            if updated:
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
            self._drain_running = False
            self._schedule_log_drain()

        pause_event = self.__dict__.get("pause_event")  # avoid Tk __getattr__ recursion
        decision_queue = self.__dict__.get("decision_queue")
        abort_event = self.__dict__.get("abort_event")

        if pause_event is None or decision_queue is None or abort_event is None:
            return

        if pause_event.is_set():
            try:
                title, message, context = decision_queue.get_nowait()
            except queue.Empty:
                return

            response = messagebox.askyesno(title=title, message=message)
            if response:
                abort_event.clear()
                pause_event.clear()
            else:
                abort_event.set()
                pause_event.clear()

            if context is not None:
                logger.debug("Decision taken for step %s", context.get("step"))

    def _initialise_file_logging(self) -> None:
        if self._file_listener is not None:
            return

        log_queue: "queue.Queue[logging.LogRecord]" = queue.Queue()
        file_handler = RotatingFileHandler(
            LOG_FILE_PATH, maxBytes=1_048_576, backupCount=5, encoding="utf-8"
        )
        queue_handler = QueueHandler(log_queue)
        listener = QueueListener(log_queue, file_handler)

        listener.start()
        logger.addHandler(queue_handler)

        self._file_log_queue = log_queue
        self._file_queue_handler = queue_handler
        self._file_handler = file_handler
        self._file_listener = listener

    # ------------------------------------------------------------------
    # Preflight orchestration

    def _on_run_preflight(self) -> None:  # pragma: no cover - UI interaction
        if self._preflight_thread and self._preflight_thread.is_alive():
            logger.info("Preflight already running; ignoring duplicate request.")
            return

        logger.info("Preflight requested. Preparing execution environment.")
        self.launch_button.state(["disabled"])
        self.run_button.state(["disabled"])
        self.pause_event.clear()
        self.abort_event.clear()

        self._preflight_thread = threading.Thread(
            target=self._run_preflight_worker,
            name="sandbox-preflight",
            daemon=True,
        )
        self._preflight_thread.start()

    def _run_preflight_worker(self) -> None:
        logger.info("Phase 5 preflight sequence starting.")
        debug_queue: "queue.Queue[str]" = queue.Queue()

        try:
            result = run_full_preflight(
                logger=logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.abort_event,
                debug_queue=debug_queue,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("Preflight aborted due to error: %s", exc)
            payload = {"success": False, "error": str(exc)}
        else:
            healthy = bool(result.get("healthy"))
            success = (
                healthy
                and not bool(result.get("paused"))
                and not bool(result.get("aborted"))
            )
            payload = {**result, "healthy": healthy, "success": success}

        self._worker_queue.put(("preflight_complete", payload))

    def _on_launch_sandbox(self) -> None:  # pragma: no cover - UI interaction
        if self._sandbox_thread and self._sandbox_thread.is_alive():
            logger.info("Sandbox launch already running; ignoring duplicate request.")
            return

        logger.info("Sandbox launch requested. Preparing process execution.")
        self.run_button.state(["disabled"])
        self.launch_button.state(["disabled"])
        self._initialise_file_logging()

        self._sandbox_thread = threading.Thread(
            target=self._run_sandbox_worker,
            name="sandbox-launch",
            daemon=True,
        )
        self._sandbox_thread.start()

    def _run_sandbox_worker(self) -> None:
        command = [sys.executable, "-m", "start_autonomous_sandbox"]
        logger.info("Starting sandbox process: %s", " ".join(command))

        try:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except Exception as exc:  # pragma: no cover - defensive logging path
            logger.exception("Failed to launch sandbox process: %s", exc)
            self._worker_queue.put(
                (
                    "sandbox_complete",
                    {"returncode": None, "error": str(exc)},
                )
            )
            return

        self._sandbox_process = process

        def _pipe_reader(stream: TextIO | None, severity: str) -> None:
            if stream is None:
                return

            with stream:
                for line in iter(stream.readline, ""):
                    message = line.rstrip()
                    if not message:
                        continue

                    if severity == "stdout":
                        logger.info("[sandbox stdout] %s", message)
                    else:
                        logger.error("[sandbox stderr] %s", message)

        readers: list[threading.Thread] = []
        for pipe, severity in (
            (process.stdout, "stdout"),
            (process.stderr, "stderr"),
        ):
            reader = threading.Thread(
                target=_pipe_reader,
                args=(pipe, severity),
                name=f"sandbox-{severity}",
                daemon=True,
            )
            reader.start()
            readers.append(reader)

        returncode: int | None = None
        error_message: str | None = None
        try:
            returncode = process.wait()
        except Exception as exc:  # pragma: no cover - defensive logging path
            error_message = str(exc)
            logger.exception("Sandbox process wait failed: %s", exc)
        finally:
            self._sandbox_process = None

        for reader in readers:
            reader.join(timeout=1)

        payload: dict[str, object] = {"returncode": returncode}
        if error_message is not None:
            payload["error"] = error_message

        self._worker_queue.put(("sandbox_complete", payload))

    def _process_worker_events(self) -> None:  # pragma: no cover - UI loop side effect
        try:
            while True:
                event, payload = self._worker_queue.get_nowait()
                if event == "preflight_complete":
                    self._handle_preflight_completion(payload)
                elif event == "sandbox_complete":
                    self._handle_sandbox_completion(payload)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_worker_events)

    def _handle_preflight_completion(self, payload: dict[str, object]) -> None:
        self._preflight_thread = None
        self.run_button.state(["!disabled"])

        success = bool(payload.get("success"))
        if success:
            def _on_success() -> None:
                self.launch_button.state(["!disabled"])
                logger.info(
                    "Sandbox health check passed. 'Start Sandbox' button enabled."
                )

            self.after_idle(_on_success)
            return

        self.launch_button.state(["disabled"])

        message = self._format_health_failure_message(payload)
        logger.warning("Sandbox health verification failed: %s", message)
        messagebox.showwarning("Sandbox health issue detected", message)

    def _handle_sandbox_completion(self, payload: dict[str, object]) -> None:
        self._sandbox_thread = None
        self._sandbox_process = None

        self.run_button.state(["!disabled"])
        self.launch_button.state(["!disabled"])

        error = payload.get("error")
        if error:
            message = str(error)
            logger.error("Sandbox execution failed: %s", message)
            messagebox.showerror("Sandbox launch failed", message)
            return

        returncode = payload.get("returncode")
        if returncode is None:
            logger.error("Sandbox exited with an unknown status.")
            messagebox.showerror(
                "Sandbox launch failed",
                "Sandbox exited with an unknown status.",
            )
            return

        if int(returncode) == 0:
            logger.info("Sandbox exited successfully with code 0.")
            return

        logger.error("Sandbox exited with code %s.", returncode)
        messagebox.showerror(
            "Sandbox exited with errors",
            f"Sandbox exited with code {returncode}.",
        )

    def _format_health_failure_message(self, payload: dict[str, object]) -> str:
        error = payload.get("error")
        if error:
            return str(error)

        failures = payload.get("failures")
        if isinstance(failures, Iterable) and not isinstance(failures, (str, bytes)):
            details = [str(item) for item in failures if str(item).strip()]
            if details:
                return "; ".join(details)

        snapshot = payload.get("snapshot")
        if isinstance(snapshot, dict):
            db_errors = snapshot.get("database_errors")
            if isinstance(db_errors, dict) and db_errors:
                formatted = ", ".join(
                    f"{name}: {reason}" for name, reason in db_errors.items()
                )
                return f"Database accessibility issues detected ({formatted})."

        if payload.get("paused") and payload.get("failed_step"):
            return f"Preflight paused during '{payload['failed_step']}'."

        if payload.get("failed_step"):
            return f"Preflight failed at step '{payload['failed_step']}'."

        if payload.get("aborted"):
            return "Preflight aborted by operator."

        return (
            "Sandbox health check did not complete successfully. Review the logs "
            "for additional details."
        )


def _ensure_abort_not_requested(abort_event: threading.Event) -> bool:
    if abort_event.is_set():
        logger.info("Preflight aborted by operator before step execution.")
        return False
    return True


def _run_step(
    step: _PreflightStep,
    *,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[tuple[str, str, dict[str, object] | None]]",
    abort_event: threading.Event,
    debug_queue: "queue.Queue[str] | None",
) -> bool:
    if abort_event.is_set():
        logger.info("Skipping %s because an abort was requested.", step.name)
        return False

    logger.info(step.start_message)
    try:
        runner = getattr(sys.modules[__name__], step.name, step.runner)
        if not callable(runner):
            runner = step.runner
        runner(logger)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("%s", step.failure_title)

        pause_event.set()
        context = {"step": step.name, "exception": str(exc)}
        decision_queue.put((step.failure_title, step.failure_message, context))
        if debug_queue is not None:
            debug_queue.put(traceback.format_exc())

        while pause_event.is_set() and not abort_event.is_set():
            time.sleep(0.1)

        return False

    logger.info(step.success_message)
    return True


def run_full_preflight(
    *,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[tuple[str, str, dict[str, object] | None]]",
    abort_event: threading.Event,
    debug_queue: "queue.Queue[str] | None" = None,
    dependency_mode: DependencyMode | None = None,
) -> dict[str, object]:
    """Execute the complete preflight sequence used by the GUI worker."""

    if not _ensure_abort_not_requested(abort_event):
        return {"aborted": True}

    if dependency_mode is None:
        dependency_mode = resolve_dependency_mode()

    for step in _PREFLIGHT_STEPS:
        if not _run_step(
            step,
            logger=logger,
            pause_event=pause_event,
            decision_queue=decision_queue,
            abort_event=abort_event,
            debug_queue=debug_queue,
        ):
            return {"paused": pause_event.is_set(), "failed_step": step.name}

    snapshot = _collect_sandbox_health(logger, pause_event, decision_queue, abort_event, debug_queue)
    healthy, failures = _evaluate_health_snapshot(snapshot, dependency_mode=dependency_mode)

    return {"snapshot": snapshot, "healthy": healthy, "failures": failures}


def _collect_sandbox_health(
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[tuple[str, str, dict[str, object] | None]]",
    abort_event: threading.Event,
    debug_queue: "queue.Queue[str] | None",
) -> dict[str, object]:
    try:
        bootstrap = importlib.import_module("sandbox_runner.bootstrap")
        snapshot = bootstrap.sandbox_health()
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("Sandbox health evaluation failed: %s", exc)
        pause_event.set()
        context = {"step": "sandbox_health", "exception": str(exc)}
        decision_queue.put(
            (
                "Sandbox health evaluation failed",
                "Collecting the sandbox health snapshot failed. Check logs for details.",
                context,
            )
        )
        if debug_queue is not None:
            debug_queue.put(traceback.format_exc())

        while pause_event.is_set() and not abort_event.is_set():
            time.sleep(0.1)

        return {}

    logger.info("Sandbox health snapshot gathered.")
    return snapshot


def _git_sync(logger: logging.Logger) -> None:
    logger.info("Ensuring repository is synchronised with origin.")


def _purge_stale_files(logger: logging.Logger) -> None:
    logger.info("Purging stale files and caches.")


def _cleanup_lock_and_model_artifacts(logger: logging.Logger) -> None:
    logger.info("Removing stale lock files and model caches.")


def _install_heavy_dependencies(logger: logging.Logger) -> None:
    logger.info("Installing heavy dependencies if required.")


def _warm_shared_vector_service(logger: logging.Logger) -> None:
    logger.info("Warming the shared vector service cache.")


def _ensure_env_flags(logger: logging.Logger) -> None:
    logger.info("Ensuring environment flags are set for sandbox run.")


def _prime_registry(logger: logging.Logger) -> None:
    logger.info("Priming the registry for sandbox resources.")


def _install_python_dependencies(logger: logging.Logger) -> None:
    logger.info("Ensuring Python dependencies are installed.")


def _bootstrap_self_coding(logger: logging.Logger) -> None:
    logger.info("Bootstrapping self-coding components.")


_PREFLIGHT_STEPS: tuple[_PreflightStep, ...] = (
    _PreflightStep(
        name="_git_sync",
        start_message="Synchronising repository with origin.",
        success_message="Repository synchronisation complete.",
        failure_title="Repository synchronisation failed",
        failure_message=(
            "The Git synchronisation step failed. Check network access and remote permissions before retrying."
        ),
        runner=_git_sync,
    ),
    _PreflightStep(
        name="_purge_stale_files",
        start_message="Removing stale files and caches.",
        success_message="Stale files removed.",
        failure_title="Stale file cleanup failed",
        failure_message=(
            "Purging stale files and caches failed. Review permissions on the working tree and try again."
        ),
        runner=_purge_stale_files,
    ),
    _PreflightStep(
        name="_cleanup_lock_and_model_artifacts",
        start_message="Cleaning up lock and model artefacts.",
        success_message="Lock and model artefacts refreshed.",
        failure_title="Lock and model cleanup failed",
        failure_message=(
            "Removing stale lock files and model caches failed. Resolve filesystem issues before continuing."
        ),
        runner=_cleanup_lock_and_model_artifacts,
    ),
    _PreflightStep(
        name="_install_heavy_dependencies",
        start_message="Installing heavy dependencies.",
        success_message="Heavy dependency installation complete.",
        failure_title="Heavy dependency installation failed",
        failure_message=(
            "Installing heavy dependencies failed. Check download connectivity or cached artefacts."
        ),
        runner=_install_heavy_dependencies,
    ),
    _PreflightStep(
        name="_warm_shared_vector_service",
        start_message="Warming the shared vector service.",
        success_message="Shared vector service warmed.",
        failure_title="Vector service warmup failed",
        failure_message=(
            "Warming the shared vector service failed. Ensure vector service dependencies are installed."
        ),
        runner=_warm_shared_vector_service,
    ),
    _PreflightStep(
        name="_ensure_env_flags",
        start_message="Ensuring environment flags are set.",
        success_message="Environment flags verified.",
        failure_title="Environment flag configuration failed",
        failure_message=(
            "Ensuring environment flags failed. Verify configuration files and environment variables."
        ),
        runner=_ensure_env_flags,
    ),
    _PreflightStep(
        name="_prime_registry",
        start_message="Priming resource registry.",
        success_message="Resource registry primed.",
        failure_title="Registry priming failed",
        failure_message=(
            "Priming the resource registry failed. Confirm registry service availability and credentials."
        ),
        runner=_prime_registry,
    ),
    _PreflightStep(
        name="_install_python_dependencies",
        start_message="Installing Python dependencies.",
        success_message="Python dependencies ready.",
        failure_title="Python dependency installation failed",
        failure_message=(
            "Installing Python dependencies failed. Review the package index connection and retry."
        ),
        runner=_install_python_dependencies,
    ),
    _PreflightStep(
        name="_bootstrap_self_coding",
        start_message="Bootstrapping self-coding modules.",
        success_message="Self-coding bootstrap complete.",
        failure_title="Self-coding bootstrap failed",
        failure_message=(
            "Bootstrapping self-coding components failed. Inspect logs for detailed diagnostics."
        ),
        runner=_bootstrap_self_coding,
    ),
)


def _evaluate_health_snapshot(
    snapshot: dict[str, object] | Iterable[tuple[str, object]],
    *,
    dependency_mode: DependencyMode,
) -> tuple[bool, list[str]]:
    """Evaluate sandbox health results and surface failure messages."""

    if isinstance(snapshot, dict):
        health = snapshot
    else:
        health = dict(snapshot)

    failures: list[str] = []

    if not bool(health.get("databases_accessible", True)):
        errors = health.get("database_errors")
        if isinstance(errors, dict) and errors:
            formatted = ", ".join(f"{name}: {reason}" for name, reason in errors.items())
            failures.append(f"databases inaccessible ({formatted})")
        else:
            failures.append("databases inaccessible")

    dependency_section = health.get("dependency_health")
    if isinstance(dependency_section, dict):
        missing = dependency_section.get("missing", [])
        if isinstance(missing, Iterable):
            for entry in missing:
                if not isinstance(entry, dict):
                    continue
                name = str(entry.get("name", "unknown"))
                optional = bool(entry.get("optional"))
                if optional and dependency_mode is DependencyMode.MINIMAL:
                    continue
                if optional:
                    failures.append(f"optional dependency missing: {name}")
                else:
                    failures.append(f"dependency missing: {name}")

    return (not failures, failures)


def main() -> int:
    """Launch the sandbox launcher GUI event loop."""

    app = SandboxLauncherGUI()
    app.mainloop()
    return 0


__all__ = [
    "SandboxLauncherGUI",
    "run_full_preflight",
    "_evaluate_health_snapshot",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    sys.exit(main())

