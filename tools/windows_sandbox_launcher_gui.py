"""GUI shell and preflight orchestration for the Windows sandbox."""

from __future__ import annotations

import importlib
import logging
import os
import queue
import sys
import threading
import time
import traceback
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TextIO

import tkinter as tk
from tkinter import font as tk_font
from tkinter import messagebox
from tkinter import ttk

from dependency_health import DependencyMode, resolve_dependency_mode
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler

import auto_env_setup
import bootstrap_self_coding
from neurosales.scripts import setup_heavy_deps
import prime_registry
from vector_service.vectorizer import SharedVectorService


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE_PATH = REPO_ROOT / "menace_gui_logs.txt"
LOG_FILE_ENV = "MENACE_GUI_LOG_PATH"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _path_exists(candidate: object) -> bool:
    """Best-effort existence check for :func:`_purge_stale_files`."""

    exists = getattr(candidate, "exists", None)
    if callable(exists):
        try:
            return bool(exists())
        except OSError:
            return False
    return False


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

    def __init__(
        self, *, log_file_path: str | Path | None = None
    ) -> None:  # pragma: no cover - UI construction
        super().__init__()

        # Window metadata
        self.title("Windows Sandbox Launcher")
        self.geometry("720x520")

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

        # UI and state trackers
        self._paused_step_index: int | None = None
        self._last_failed_step: str | None = None
        self._latest_pause_context: dict[str, object] | None = None
        self._debug_visible = False
        self._preflight_start_time: float | None = None
        self._elapsed_job: str | None = None
        self._pause_dialog_presented = False
        self.status_var = tk.StringVar(value="Preflight idle")

        configured_path = log_file_path or os.environ.get(LOG_FILE_ENV) or LOG_FILE_PATH
        self._log_file_path = Path(configured_path)

        # Build UI layout
        self._build_layout()

        # Logging sinks
        self._initialise_file_logging()

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
        status_tab.rowconfigure(1, weight=0)
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

        self.debug_container = ttk.Labelframe(
            status_tab, text="Debug Details", padding=(8, 8)
        )
        self.debug_container.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.debug_container.columnconfigure(0, weight=1)
        self.debug_container.rowconfigure(0, weight=1)

        self.debug_text = tk.Text(
            self.debug_container,
            wrap="word",
            state=tk.DISABLED,
            height=8,
        )
        self.debug_text.grid(row=0, column=0, sticky="nsew")

        debug_scrollbar = ttk.Scrollbar(
            self.debug_container, orient="vertical", command=self.debug_text.yview
        )
        debug_scrollbar.grid(row=0, column=1, sticky="ns")
        self.debug_text.configure(yscrollcommand=debug_scrollbar.set)

        self.debug_container.grid_remove()

        # Control buttons row
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        for column in range(3):
            button_frame.columnconfigure(column, weight=1)

        self.run_button = ttk.Button(
            button_frame, text="Run Preflight", command=self._on_run_preflight
        )
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.retry_button = ttk.Button(
            button_frame,
            text="Retry Step",
            command=self._on_retry_step,
            state=tk.DISABLED,
        )
        self.retry_button.grid(row=0, column=1, sticky="ew", padx=5)
        self.retry_button.grid_remove()

        self.launch_button = ttk.Button(
            button_frame,
            text="Start Sandbox",
            command=self._on_launch_sandbox,
            state=tk.DISABLED,
        )
        self.launch_button.grid(row=0, column=2, sticky="ew", padx=(5, 0))

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=(5, 0))

        self.debug_toggle = ttk.Button(
            control_frame,
            text="Show Debug Details",
            command=self._toggle_debug_panel,
            state=tk.DISABLED,
        )
        self.debug_toggle.pack(side=tk.LEFT)

        self.status_label = ttk.Label(
            control_frame, textvariable=self.status_var, anchor="e"
        )
        self.status_label.pack(side=tk.RIGHT)

    def _configure_log_tags(self) -> None:
        default_font = tk_font.nametofont("TkDefaultFont")
        bold_font = default_font.copy()
        bold_font.configure(weight="bold")

        self.log_text.tag_configure("info", foreground="#1b5e20")
        self.log_text.tag_configure("warning", foreground="#e65100", font=bold_font)
        self.log_text.tag_configure("error", foreground="#b71c1c", font=bold_font)

    def _toggle_debug_panel(self) -> None:  # pragma: no cover - UI interaction
        self._set_debug_visible(not self._debug_visible)

    def _set_debug_visible(self, visible: bool) -> None:
        if visible == self._debug_visible:
            return

        self._debug_visible = visible
        if visible:
            self.debug_container.grid()
            self.debug_toggle.configure(text="Hide Debug Details")
        else:
            self.debug_container.grid_remove()
            self.debug_toggle.configure(text="Show Debug Details")

    def _clear_debug_panel(self) -> None:
        self.debug_text.configure(state=tk.NORMAL)
        self.debug_text.delete("1.0", tk.END)
        self.debug_text.configure(state=tk.DISABLED)
        self.debug_toggle.state(["disabled"])
        self._set_debug_visible(False)

    def _update_debug_panel(self, traces: Iterable[str]) -> None:
        cleaned = [trace for trace in traces if trace]
        if not cleaned:
            return

        self.debug_text.configure(state=tk.NORMAL)
        if self.debug_text.index("end-1c") != "1.0":
            self.debug_text.insert(tk.END, "\n---\n")
        for trace in cleaned:
            self.debug_text.insert(tk.END, trace.rstrip() + "\n")
        self.debug_text.configure(state=tk.DISABLED)
        self.debug_text.see(tk.END)

        self.debug_toggle.state(["!disabled"])
        self._set_debug_visible(True)

    def _update_elapsed_timer(self) -> None:
        self._elapsed_job = None
        self._refresh_elapsed_timer()

    def _refresh_elapsed_timer(self, *, schedule_next: bool = True) -> None:
        if self._preflight_start_time is None:
            self.status_var.set("Preflight idle")
            return

        elapsed = max(0.0, time.monotonic() - self._preflight_start_time)
        if self._preflight_thread and self._preflight_thread.is_alive():
            prefix = "Preflight running"
        elif self._paused_step_index is not None:
            prefix = "Preflight paused"
        else:
            prefix = "Preflight ready"

        self.status_var.set(f"{prefix}: {elapsed:0.1f}s")

        if schedule_next:
            self._elapsed_job = self.after(500, self._update_elapsed_timer)

    def _ensure_elapsed_timer_running(self) -> None:
        if self._elapsed_job is not None:
            self.after_cancel(self._elapsed_job)
            self._elapsed_job = None
        self._refresh_elapsed_timer()
        self._elapsed_job = self.after(500, self._update_elapsed_timer)

    def _finish_elapsed_timer(self, prefix: str) -> None:
        if self._elapsed_job is not None:
            self.after_cancel(self._elapsed_job)
            self._elapsed_job = None

        elapsed = 0.0
        if self._preflight_start_time is not None:
            elapsed = max(0.0, time.monotonic() - self._preflight_start_time)

        self.status_var.set(f"{prefix}: {elapsed:0.1f}s")
        self._preflight_start_time = None
        self._paused_step_index = None

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

        if pause_event is None or decision_queue is None:
            return

        if pause_event.is_set():
            try:
                title, message, context = decision_queue.get_nowait()
            except queue.Empty:
                return

            messagebox.showerror(title=title, message=message)
            self._pause_dialog_presented = True
            if context is not None:
                self._latest_pause_context = context
                logger.debug("Preflight paused at step %s", context.get("step"))

    def _initialise_file_logging(self) -> None:
        if self._file_listener is not None:
            return

        log_queue: "queue.Queue[logging.LogRecord]" = queue.Queue()
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            self._log_file_path, maxBytes=1_048_576, backupCount=5, encoding="utf-8"
        )
        queue_handler = QueueHandler(log_queue)
        listener = QueueListener(log_queue, file_handler)

        listener.start()
        logging.getLogger().addHandler(queue_handler)

        self._file_log_queue = log_queue
        self._file_queue_handler = queue_handler
        self._file_handler = file_handler
        self._file_listener = listener

    def _shutdown_file_logging(self) -> None:
        if self._file_listener is not None:
            self._file_listener.stop()
            self._file_listener = None

        if self._file_queue_handler is not None:
            logging.getLogger().removeHandler(self._file_queue_handler)
            self._file_queue_handler = None

        if self._file_handler is not None:
            self._file_handler.close()
            self._file_handler = None

        self._file_log_queue = None

    def _start_preflight_thread(self, start_index: int) -> None:
        self.pause_event.clear()
        self.abort_event.clear()

        self._preflight_thread = threading.Thread(
            target=self._run_preflight_worker,
            args=(start_index,),
            name="sandbox-preflight",
            daemon=True,
        )
        self._preflight_thread.start()

    # ------------------------------------------------------------------
    # Preflight orchestration

    def _on_run_preflight(self) -> None:  # pragma: no cover - UI interaction
        if self._preflight_thread and self._preflight_thread.is_alive():
            logger.info("Preflight already running; ignoring duplicate request.")
            return

        logger.info("Preflight requested. Preparing execution environment.")
        self.launch_button.state(["disabled"])
        self.run_button.state(["disabled"])
        self.retry_button.state(["disabled"])
        self.retry_button.grid_remove()

        self._paused_step_index = None
        self._last_failed_step = None
        self._latest_pause_context = None
        self._pause_dialog_presented = False

        self._clear_debug_panel()

        self._preflight_start_time = time.monotonic()
        self._ensure_elapsed_timer_running()

        self._start_preflight_thread(0)

    def _on_retry_step(self) -> None:  # pragma: no cover - UI interaction
        if self._preflight_thread and self._preflight_thread.is_alive():
            logger.info("Cannot retry while preflight is already running.")
            return

        if self._paused_step_index is None:
            logger.info("No paused preflight step available to retry.")
            return

        step_name = self._last_failed_step or str(self._paused_step_index)
        logger.info("Retrying preflight step %s.", step_name)

        self.run_button.state(["disabled"])
        self.retry_button.state(["disabled"])
        self.retry_button.grid_remove()

        if self._preflight_start_time is None:
            self._preflight_start_time = time.monotonic()
        self._ensure_elapsed_timer_running()

        resume_index = self._paused_step_index
        self._paused_step_index = None
        self._pause_dialog_presented = False

        self._start_preflight_thread(resume_index)

    def _run_preflight_worker(self, start_index: int) -> None:
        if start_index:
            logger.info(
                "Resuming preflight sequence from step index %s.", start_index
            )
        else:
            logger.info("Phase 5 preflight sequence starting.")
        debug_queue: "queue.Queue[str]" = queue.Queue()

        try:
            result = run_full_preflight(
                logger=logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.abort_event,
                debug_queue=debug_queue,
                start_index=start_index,
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

        traces: list[str] = []
        try:
            while True:
                traces.append(debug_queue.get_nowait())
        except queue.Empty:
            pass

        if traces:
            payload["debug_traces"] = traces

        payload["start_index"] = start_index

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

        debug_traces = payload.get("debug_traces")
        if isinstance(debug_traces, Iterable) and not isinstance(
            debug_traces, (str, bytes)
        ):
            self._update_debug_panel(debug_traces)

        success = bool(payload.get("success"))
        paused = bool(payload.get("paused"))
        aborted = bool(payload.get("aborted"))
        error_message = payload.get("error")

        if success:
            self.pause_event.clear()
            self.abort_event.clear()
            self._paused_step_index = None
            self._pause_dialog_presented = False
            self.retry_button.state(["disabled"])
            self.retry_button.grid_remove()

            def _on_success() -> None:
                self.launch_button.state(["!disabled"])
                logger.info(
                    "Sandbox health check passed. 'Start Sandbox' button enabled."
                )

            self.after_idle(_on_success)
            self._finish_elapsed_timer("Preflight completed")
            return

        if paused:
            self.launch_button.state(["disabled"])
            failed_index = payload.get("failed_index")
            if isinstance(failed_index, int):
                self._paused_step_index = failed_index
            else:
                start_index = payload.get("start_index")
                if isinstance(start_index, int):
                    self._paused_step_index = max(0, start_index)

            failed_step = payload.get("failed_step")
            if failed_step:
                self._last_failed_step = str(failed_step)

            self.retry_button.state(["!disabled"])
            self.retry_button.grid()
            self._refresh_elapsed_timer()

            message = self._format_health_failure_message(payload)
            logger.warning("Sandbox health verification paused: %s", message)
            if not self._pause_dialog_presented:
                messagebox.showwarning("Preflight paused", message)
            self._pause_dialog_presented = False
            return

        self.launch_button.state(["disabled"])
        self.retry_button.state(["disabled"])
        self.retry_button.grid_remove()

        prefix = "Preflight aborted" if aborted else "Preflight finished"
        self._finish_elapsed_timer(prefix)
        self._pause_dialog_presented = False

        message = self._format_health_failure_message(payload)
        if error_message:
            logger.error("Sandbox preflight failed: %s", message)
            messagebox.showerror("Preflight failed", message)
        else:
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

    def destroy(self) -> None:  # pragma: no cover - UI teardown
        if self._elapsed_job is not None:
            self.after_cancel(self._elapsed_job)
            self._elapsed_job = None
        self._shutdown_file_logging()
        super().destroy()

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
    start_index: int = 0,
) -> dict[str, object]:
    """Execute the complete preflight sequence used by the GUI worker."""

    if not _ensure_abort_not_requested(abort_event):
        return {"aborted": True}

    if dependency_mode is None:
        dependency_mode = resolve_dependency_mode()

    total_steps = len(_PREFLIGHT_STEPS)
    if start_index < 0 or start_index > total_steps:
        raise ValueError(f"start_index {start_index} outside valid range 0..{total_steps}")

    for index, step in enumerate(_PREFLIGHT_STEPS[start_index:], start=start_index):
        if not _run_step(
            step,
            logger=logger,
            pause_event=pause_event,
            decision_queue=decision_queue,
            abort_event=abort_event,
            debug_queue=debug_queue,
        ):
            return {
                "paused": pause_event.is_set(),
                "failed_step": step.name,
                "failed_index": index,
            }

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

        return {}

    logger.info("Sandbox health snapshot gathered.")
    return snapshot


def _git_sync(logger: logging.Logger) -> None:
    logger.info("Ensuring repository is synchronised with origin.")

    subprocess.run(["git", "fetch", "origin"], check=True, cwd=REPO_ROOT)
    subprocess.run(
        ["git", "reset", "--hard", "origin/main"], check=True, cwd=REPO_ROOT
    )

    logger.info("Repository reset to origin/main.")


def _purge_stale_files(logger: logging.Logger) -> None:
    logger.info("Purging stale files and caches.")

    iter_cleanup = getattr(bootstrap_self_coding, "_iter_cleanup_targets", None)
    tracked_candidates: tuple[object, ...] = ()
    if callable(iter_cleanup):
        tracked_candidates = tuple(iter_cleanup())

    before_count = sum(1 for candidate in tracked_candidates if _path_exists(candidate))

    bootstrap_self_coding.purge_stale_files()

    after_count = sum(1 for candidate in tracked_candidates if _path_exists(candidate))
    removed = max(before_count - after_count, 0)

    logger.info("Purged %d stale sandbox artefacts.", removed)


def _cleanup_lock_and_model_artifacts(logger: logging.Logger) -> None:
    logger.info("Removing stale lock files and model caches.")

    removed_files = 0
    removed_directories = 0

    def _remove_path(path: Path) -> None:
        nonlocal removed_files, removed_directories
        try:
            if path.is_dir():
                shutil.rmtree(path)
                removed_directories += 1
            else:
                path.unlink()
                removed_files += 1
        except FileNotFoundError:
            return

    sandbox_lock_dir = REPO_ROOT / "sandbox_data"
    for candidate in sandbox_lock_dir.glob("*.lock"):
        _remove_path(candidate)

    hf_cache = Path.home() / ".cache" / "huggingface" / "transformers"
    for candidate in hf_cache.glob("*.lock"):
        _remove_path(candidate)

    incomplete_patterns = ("**/*.incomplete", "**/*.tmp", "**/*lock*")
    for pattern in incomplete_patterns:
        for candidate in hf_cache.glob(pattern):
            if pattern == "**/*lock*" and not getattr(candidate, "is_dir", lambda: False)():
                # Avoid double-counting regular lock files handled earlier.
                continue
            _remove_path(candidate)

    logger.info(
        "Removed %d stale files and %d stale directories.",
        removed_files,
        removed_directories,
    )


def _install_heavy_dependencies(logger: logging.Logger) -> None:
    logger.info("Installing heavy dependencies if required.")

    setup_heavy_deps.main(download_only=True)

    logger.info("Heavy dependency setup completed in download-only mode.")


def _warm_shared_vector_service(logger: logging.Logger) -> None:
    logger.info("Warming the shared vector service cache.")

    SharedVectorService()

    logger.info("Shared vector service initialised.")


def _ensure_env_flags(logger: logging.Logger) -> None:
    logger.info("Ensuring environment flags are set for sandbox run.")

    auto_env_setup.ensure_env()
    os.environ["SANDBOX_ENABLE_BOOTSTRAP"] = "1"
    os.environ["SANDBOX_ENABLE_SELF_CODING"] = "1"

    logger.info("Sandbox bootstrap and self-coding flags enabled.")


def _prime_registry(logger: logging.Logger) -> None:
    logger.info("Priming the registry for sandbox resources.")

    prime_registry.main()

    logger.info("Registry primed successfully.")


def _install_python_dependencies(logger: logging.Logger) -> None:
    logger.info("Ensuring Python dependencies are installed.")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        check=True,
        cwd=REPO_ROOT,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "jsonschema"],
        check=True,
        cwd=REPO_ROOT,
    )

    logger.info("Python dependency installation complete.")


def _bootstrap_self_coding(logger: logging.Logger) -> None:
    logger.info("Bootstrapping self-coding components.")

    bootstrap_self_coding.bootstrap_self_coding("AICounterBot")

    logger.info("Self-coding bootstrap invoked for AICounterBot.")


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

