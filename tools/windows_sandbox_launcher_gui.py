"""Minimal GUI scaffolding for launching the Windows sandbox."""

from __future__ import annotations

import logging
import queue
import tkinter as tk
from tkinter import font as tk_font
from tkinter import ttk


class _QueueLogHandler(logging.Handler):
    """Logging handler that pushes formatted records into a queue."""

    def __init__(self, message_queue: queue.Queue[tuple[str, str]]) -> None:
        super().__init__()
        self._queue = message_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - UI side effect
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - formatting errors fall back to repr
            message = repr(record)

        level = "info"
        if record.levelno >= logging.ERROR:
            level = "error"
        elif record.levelno >= logging.WARNING:
            level = "warning"

        self._queue.put((level, message))


class SandboxLauncherGUI(tk.Tk):
    """Basic window shell for future sandbox launcher features."""

    def __init__(self) -> None:
        super().__init__()

        # Window metadata
        self.title("Windows Sandbox Launcher")
        self.geometry("720x480")

        # Initialize Tk themed widgets
        self.style = ttk.Style(self)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        else:  # pragma: no cover - dependent on platform themes
            self.style.theme_use("default")

        # Main layout container
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Notebook with status tab for log display
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

        self.log_text = tk.Text(log_container, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        self.log_scrollbar = ttk.Scrollbar(
            log_container, orient="vertical", command=self.log_text.yview
        )
        self.log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)

        self._configure_log_tags()

        # Logging integration
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.log_handler = _QueueLogHandler(self.log_queue)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        root_logger = logging.getLogger()
        if self.log_handler not in root_logger.handlers:
            root_logger.addHandler(self.log_handler)
        if root_logger.level == logging.NOTSET:
            root_logger.setLevel(logging.INFO)

        self.logger = logging.getLogger(__name__)
        self.logger.propagate = True

        # Periodically poll for log updates
        self.after(100, self._drain_log_queue)

        # Action buttons placed beneath the notebook
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.run_button = ttk.Button(button_frame, text="Run Preflight")
        self.run_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.launch_button = ttk.Button(
            button_frame, text="Start Sandbox", state="disabled"
        )
        self.launch_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _configure_log_tags(self) -> None:
        """Configure tag styles for log message severities."""

        default_font = tk_font.nametofont("TkDefaultFont")
        self._bold_font = default_font.copy()
        self._bold_font.configure(weight="bold")

        self.log_text.tag_configure("info", foreground="#1b5e20")
        self.log_text.tag_configure("warning", foreground="#e65100", font=self._bold_font)
        self.log_text.tag_configure("error", foreground="#b71c1c", font=self._bold_font)

    def _drain_log_queue(self) -> None:  # pragma: no cover - UI loop side effect
        """Drain queued log entries and render them in the text widget."""

        try:
            while True:
                level, message = self.log_queue.get_nowait()
                if level not in {"info", "warning", "error"}:
                    level = "info"

                self.log_text.configure(state="normal")
                self.log_text.insert("end", f"{message}\n", level)
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
        except queue.Empty:
            pass

        self.after(100, self._drain_log_queue)


__all__ = ["SandboxLauncherGUI"]
