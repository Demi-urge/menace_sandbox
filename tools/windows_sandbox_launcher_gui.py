"""Tkinter GUI for launching and monitoring the Windows sandbox."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk


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

        self._run_preflight_btn = run_preflight
        self._start_sandbox_btn = start_sandbox
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

        delay = 100 if drained else 250
        self.after(delay, self._drain_log_queue)
    # endregion callbacks


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


__all__ = ["SandboxLauncherGUI"]


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    gui = SandboxLauncherGUI()
    gui.mainloop()
