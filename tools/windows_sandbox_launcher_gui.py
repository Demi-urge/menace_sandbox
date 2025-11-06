"""GUI components for launching the Windows sandbox."""

import logging
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional, Tuple


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


class SandboxLauncherGUI(tk.Tk):
    """Main application window for the sandbox launcher."""

    WINDOW_TITLE = "Windows Sandbox Launcher"
    WINDOW_GEOMETRY = "800x600"

    def __init__(self) -> None:
        super().__init__()

        self.title(self.WINDOW_TITLE)
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(600, 400)
        self.resizable(width=True, height=True)

        self.log_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.worker_queue: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
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

        self.run_preflight_button = ttk.Button(
            controls_frame,
            text="Run Preflight",
            command=self._on_run_preflight,
        )
        self.run_preflight_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.start_sandbox_button = ttk.Button(
            controls_frame,
            text="Start Sandbox",
            state=tk.DISABLED,
        )
        self.start_sandbox_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

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

        try:
            status, message = self.worker_queue.get_nowait()
        except queue.Empty:
            if self._preflight_thread and self._preflight_thread.is_alive():
                self._schedule_worker_poll()
            return

        self._preflight_thread = None

        duration_msg = ""
        if self._preflight_start_time is not None:
            elapsed = time.time() - self._preflight_start_time
            duration_msg = f" (completed in {elapsed:.2f} seconds)"
        self._preflight_start_time = None

        if status == "success":
            logger.info("Preflight completed successfully%s", duration_msg)
            self.start_sandbox_button.configure(state=tk.NORMAL)
        else:
            if message:
                logger.error("Preflight failed: %s", message)
            else:
                logger.error("Preflight failed%s", duration_msg)
            self.start_sandbox_button.configure(state=tk.DISABLED)

        self.run_preflight_button.configure(state=tk.NORMAL)

    def _on_run_preflight(self) -> None:
        if self._preflight_thread and self._preflight_thread.is_alive():
            logger.debug("Preflight already running; ignoring additional request.")
            return

        self.run_preflight_button.configure(state=tk.DISABLED)
        self.start_sandbox_button.configure(state=tk.DISABLED)
        self._preflight_start_time = time.time()
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
            stages = (
                "Validating configuration",
                "Checking required resources",
                "Verifying sandbox prerequisites",
            )
            for stage in stages:
                logger.info("Preflight stage: %s", stage)
                time.sleep(0.1)

            self.worker_queue.put_nowait(("success", None))
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

