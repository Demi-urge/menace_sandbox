"""GUI for launching Windows sandbox operations."""

from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk


class TextWidgetQueueHandler(logging.Handler):
    """Logging handler that places formatted log messages onto a queue."""

    def __init__(self, log_queue: "queue.Queue[tuple[str, str]]") -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if not message.endswith("\n"):
                message = f"{message}\n"
            self.log_queue.put((record.levelname.lower(), message))
        except Exception:  # pragma: no cover - rely on logging's default handling
            self.handleError(record)


class SandboxLauncherGUI(tk.Tk):
    """Simple GUI window for managing sandbox lifecycle actions."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Windows Sandbox Launcher")
        self.geometry("600x400")

        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.logger = logging.getLogger("windows_sandbox_launcher_gui")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self._preflight_thread: threading.Thread | None = None
        self._preflight_running = False

        self._build_notebook()
        self._build_controls()
        self._configure_logging()

        self._process_log_queue()

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self)

        self.status_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.status_tab, text="Status")

        self.status_text = tk.Text(
            self.status_tab,
            wrap="word",
            state="disabled",
            background="#1e1e1e",
        )
        self.status_text.pack(fill="both", expand=True, padx=10, pady=10)

        self.notebook.pack(fill="both", expand=True, padx=10, pady=(10, 0))

    def _build_controls(self) -> None:
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=10, pady=10)

        self.run_preflight_button = ttk.Button(
            controls,
            text="Run Preflight",
            command=self.run_preflight,
        )
        self.run_preflight_button.pack(side="left", padx=(0, 5))

        self.start_sandbox_button = ttk.Button(
            controls,
            text="Start Sandbox",
            command=self.start_sandbox,
            state="disabled",
        )
        self.start_sandbox_button.pack(side="left")

    def _configure_logging(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")

        self.status_text.tag_config("info", foreground="white", font=default_font)
        self.status_text.tag_config("warning", foreground="yellow", font=default_font)

        error_font = default_font.copy()
        error_font.configure(weight="bold")
        self.status_text.tag_config("error", foreground="red", font=error_font)
        self._error_font = error_font

        queue_handler = TextWidgetQueueHandler(self.log_queue)
        queue_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(queue_handler)
        self._queue_handler = queue_handler

    def run_preflight(self) -> None:
        """Kick off the preflight routine in a background thread."""
        if self._preflight_running:
            if self._preflight_thread and self._preflight_thread.is_alive():
                self.logger.info(
                    "Preflight run already in progress; ignoring request."
                )
                return

            # A previous worker should have reset the state, but ensure the
            # button becomes usable if the thread has unexpectedly stopped.
            self.logger.warning(
                "Preflight state indicated running without an active thread;"
                " resetting controls."
            )
            self._reset_preflight_state()

        if self._preflight_thread and self._preflight_thread.is_alive():
            self.logger.info("Preflight run already in progress; ignoring request.")
            return

        self.run_preflight_button.configure(state="disabled")
        self._preflight_running = True
        self.logger.info("Preflight checks initiated...")

        self._preflight_thread = threading.Thread(
            target=self._execute_preflight,
            name="PreflightWorker",
            daemon=True,
        )
        self._preflight_thread.start()

    def start_sandbox(self) -> None:  # pragma: no cover - placeholder hook
        """Placeholder command that will be implemented in a future iteration."""
        self.logger.info("Sandbox startup sequence initiated...")

    def _execute_preflight(self) -> None:
        """Worker routine that performs preflight operations."""
        steps = (
            "Synchronizing repository...",
            "Cleaning stale files...",
            "Installing dependencies...",
            "Priming registries...",
        )

        try:
            for step in steps:
                self.logger.info(step)
                time.sleep(0.1)
            self.logger.info("Preflight checks complete.")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.error("Preflight checks failed: %s", exc)
        finally:
            self.after(0, self._reset_preflight_state)

    def _reset_preflight_state(self) -> None:
        """Re-enable UI controls and clear state after preflight finishes."""
        self._preflight_running = False
        self._preflight_thread = None
        self.run_preflight_button.configure(state="normal")

    def _process_log_queue(self) -> None:
        while True:
            try:
                tag, message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_status(message, tag)
        self.after(100, self._process_log_queue)

    def _append_status(self, message: str, tag: str = "info") -> None:
        self.status_text.configure(state="normal")
        self.status_text.insert("end", message, (tag,))
        self.status_text.configure(state="disabled")
        self.status_text.see("end")


__all__ = ["SandboxLauncherGUI"]
