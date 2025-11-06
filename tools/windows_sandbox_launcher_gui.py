"""GUI for launching Windows sandbox environments."""

from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox
from tkinter import ttk
from typing import Callable, Optional


def run_full_preflight(
    *,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: queue.Queue[tuple[str, str]],
    abort_event: threading.Event,
) -> None:
    """Execute the sandbox preflight checks using *logger* for progress."""

    logger.info("starting Windows sandbox preflight routine")

    def _validate_local_configuration() -> None:
        logger.info("validating local configuration")

    def _verify_dependency_availability() -> None:
        logger.info("verifying dependency availability")

    def _check_sandbox_integrity() -> None:
        logger.info("checking sandbox artefact integrity")

    steps: list[tuple[str, str, Callable[[], None]]] = [
        (
            "Configuration validation failed",
            "The configuration validation step reported an error. Continue the preflight?",
            _validate_local_configuration,
        ),
        (
            "Dependency verification failed",
            "Dependency verification did not complete successfully. Continue the preflight?",
            _verify_dependency_availability,
        ),
        (
            "Sandbox artefact check failed",
            "Sandbox artefact integrity checks were not successful. Continue the preflight?",
            _check_sandbox_integrity,
        ),
    ]

    for failure_title, failure_message, step in steps:
        if abort_event.is_set():
            logger.info("preflight routine aborted before completing remaining steps")
            return
        try:
            step()
        except Exception:
            logger.exception(failure_title)
            pause_event.set()
            decision_queue.put((failure_title, failure_message))
            while pause_event.is_set() and not abort_event.is_set():
                time.sleep(0.1)
            if abort_event.is_set():
                logger.info("preflight routine aborted during paused step")
                return

    logger.info("preflight routine completed successfully")


class _LogQueueHandler(logging.Handler):
    """Forward log records from worker threads to a :class:`queue.Queue`."""

    _LEVEL_TO_TAG = {
        logging.DEBUG: "info",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def __init__(self, log_queue: queue.Queue[tuple[str, str]]) -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - defensive fallback
            self.handleError(record)
            return
        tag = self._LEVEL_TO_TAG.get(record.levelno, "info")
        self._queue.put((tag, message))


class SandboxLauncherGUI(tk.Tk):
    """Tkinter-based GUI shell for managing sandbox launch tasks."""

    DEFAULT_GEOMETRY = "900x600"
    WINDOW_TITLE = "Windows Sandbox Launcher"

    def __init__(self) -> None:
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.geometry(self.DEFAULT_GEOMETRY)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._decision_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._log_handler: Optional[_LogQueueHandler] = None
        self._logger = logging.getLogger(__name__ + ".preflight")
        self._preflight_thread: Optional[threading.Thread] = None
        self._pause_event = threading.Event()
        self._abort_event = threading.Event()
        self._awaiting_decision = False
        self._max_log_lines = 1000

        self._create_widgets()
        self._configure_layout()
        self._setup_logging()

    def _create_widgets(self) -> None:
        """Instantiate and configure all widgets used by the GUI."""
        self.notebook = ttk.Notebook(self)

        # Status Tab
        self.status_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.status_frame, text="Status")

        fixed_font = tkfont.nametofont("TkFixedFont")

        self.log_text = tk.Text(
            self.status_frame,
            wrap="word",
            state="disabled",
            font=fixed_font,
            background="white",
        )
        self.log_vertical_scroll = ttk.Scrollbar(
            self.status_frame, orient="vertical", command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=self.log_vertical_scroll.set)

        # Pre-create tags for log styling.
        self.log_text.tag_configure("info", foreground="black")
        self.log_text.tag_configure("success", foreground="darkgreen")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("error", foreground="red")

        self.footer_frame = ttk.Frame(self)
        self.run_preflight_button = ttk.Button(
            self.footer_frame,
            text="Run Preflight",
            command=self.on_run_preflight,
            state="normal",
        )
        self.start_sandbox_button = ttk.Button(
            self.footer_frame,
            text="Start Sandbox",
            command=self.on_start_sandbox,
            state="disabled",
        )

    def _configure_layout(self) -> None:
        """Configure the geometry management for all widgets."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.footer_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        # Status frame layout
        self.status_frame.columnconfigure(0, weight=1)
        self.status_frame.rowconfigure(0, weight=1)

        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_vertical_scroll.grid(row=0, column=1, sticky="ns")

        # Footer buttons
        self.footer_frame.columnconfigure(0, weight=1)
        self.footer_frame.columnconfigure(1, weight=1)

        self.run_preflight_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.start_sandbox_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _setup_logging(self) -> None:
        """Initialise the shared logger/queue used by worker threads."""

        handler = _LogQueueHandler(self._log_queue)
        handler.setFormatter(
            logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
        )
        self._log_handler = handler

        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if handler not in self._logger.handlers:
            self._logger.addHandler(handler)

        self.log_text.configure(state="disabled")
        self.after(100, self._poll_log_queue)

    def _poll_log_queue(self) -> None:
        """Flush queued log records into the text widget."""

        drained: list[tuple[str, str]] = []
        try:
            while True:
                drained.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass

        if drained:
            self.log_text.configure(state="normal")
            for tag, message in drained:
                self.log_text.insert(tk.END, message + "\n", tag)
            self._trim_log()
            self.log_text.configure(state="disabled")
            self.log_text.see(tk.END)

        self.after(100, self._poll_log_queue)

    def _trim_log(self) -> None:
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines <= self._max_log_lines:
            return
        excess = lines - self._max_log_lines
        self.log_text.delete("1.0", f"{excess + 1}.0")

    def on_run_preflight(self) -> None:
        """Kick off the sandbox preflight routine in a background worker."""

        if self._preflight_thread and self._preflight_thread.is_alive():
            return

        self._abort_event.clear()
        self._pause_event.clear()
        self._flush_decision_queue()
        self.run_preflight_button.configure(state="disabled")
        self._logger.info("launching preflight worker")

        thread = threading.Thread(target=self._run_preflight_worker, daemon=True)
        self._preflight_thread = thread
        thread.start()
        self.after(100, self._monitor_preflight_worker)

    def on_start_sandbox(self) -> None:
        """Placeholder callback for the sandbox start action."""
        # Future implementation will start the sandbox.
        print("Start Sandbox button pressed")

    def _run_preflight_worker(self) -> None:
        """Invoke the preflight routine while reporting via the shared logger."""

        try:
            run_full_preflight(
                logger=self._logger,
                pause_event=self._pause_event,
                decision_queue=self._decision_queue,
                abort_event=self._abort_event,
            )
        except Exception:  # pragma: no cover - log and propagate to UI
            self._logger.exception("preflight routine failed")
        else:
            if not self._abort_event.is_set():
                self._logger.info("preflight worker completed")

    def _monitor_preflight_worker(self) -> None:
        """Re-enable UI controls once the worker thread terminates."""

        thread = self._preflight_thread
        if thread and thread.is_alive():
            self._handle_worker_pause()
            self.after(100, self._monitor_preflight_worker)
            return

        self._preflight_thread = None
        self.run_preflight_button.configure(state="normal")
        if self._abort_event.is_set():
            self._handle_abort_cleanup()

    def on_close(self) -> None:
        """Handle application shutdown."""

        if self._log_handler is not None:
            self._logger.removeHandler(self._log_handler)
        self.destroy()

    def _handle_worker_pause(self) -> None:
        if not self._pause_event.is_set() or self._awaiting_decision:
            return

        try:
            title, message = self._decision_queue.get_nowait()
        except queue.Empty:
            return

        self._awaiting_decision = True
        should_continue = messagebox.askyesno(title=title, message=message)
        if should_continue:
            self._logger.info("user opted to continue after pause")
            self._pause_event.clear()
        else:
            self._logger.info("user requested preflight cancellation")
            self._abort_event.set()
            self._pause_event.clear()
        self._awaiting_decision = False

    def _handle_abort_cleanup(self) -> None:
        self._abort_event.clear()
        self._pause_event.clear()
        self._flush_decision_queue()
        self.start_sandbox_button.configure(state="disabled")
        self._logger.info("preflight routine cancelled by user")

    def _flush_decision_queue(self) -> None:
        while True:
            try:
                self._decision_queue.get_nowait()
            except queue.Empty:
                break


__all__ = ["SandboxLauncherGUI"]
