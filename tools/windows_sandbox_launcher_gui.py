"""GUI for launching the Windows sandbox environment."""

from __future__ import annotations

import logging
import queue
from typing import Any, Tuple

import tkinter as tk
from tkinter import ttk


class TextWidgetHandler(logging.Handler):
    """Logging handler that forwards records to a queue for the GUI log display."""

    def __init__(self, log_queue: "queue.Queue[Tuple[str, str]]", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - standard logging docstring not required
        """Format and enqueue the record for the GUI thread."""

        try:
            message = self.format(record)
            level = record.levelname.lower()
            self._log_queue.put_nowait((level, message))
        except Exception:  # pragma: no cover - defensive fallback mirrors logging.Handler
            self.handleError(record)


class SandboxLauncherGUI(tk.Tk):
    """Simple GUI used for launching sandbox workflows."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.log_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._configure_root()
        self._build_widgets()
        self._configure_log_tags()
        self.log_handler = TextWidgetHandler(self.log_queue)
        self._schedule_log_drain()

    def _configure_root(self) -> None:
        """Configure the main window properties and default styles."""
        self.title("Windows Sandbox Launcher")
        self.geometry("640x480")

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

    def _build_widgets(self) -> None:
        """Build the notebook, log display, and control buttons."""
        container = ttk.Frame(self, padding=10)
        container.grid(row=0, column=0, sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(container)
        notebook.grid(row=0, column=0, sticky="nsew")

        status_frame = ttk.Frame(notebook, padding=(10, 10))
        status_frame.pack_propagate(False)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            status_frame,
            state="disabled",
            wrap="word",
            height=15,
            width=60,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        notebook.add(status_frame, text="Status")

        controls = ttk.Frame(container)
        controls.grid(row=1, column=0, pady=(10, 0), sticky="ew")
        controls.columnconfigure((0, 1), weight=1)

        self.preflight_button = ttk.Button(controls, text="Run Preflight")
        self.preflight_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.start_button = ttk.Button(controls, text="Start Sandbox", state="disabled")
        self.start_button.grid(row=0, column=1, padx=(5, 0), sticky="ew")

    def _configure_log_tags(self) -> None:
        """Configure styled tags used for log rendering."""

        self.log_text.tag_configure("info", foreground="white")
        self.log_text.tag_configure("warning", foreground="yellow")
        self.log_text.tag_configure("error", foreground="red", font=("TkDefaultFont", 9, "bold"))

    def _schedule_log_drain(self) -> None:
        """Schedule periodic draining of the log queue on the Tkinter thread."""

        self._drain_log_queue()

    def _drain_log_queue(self) -> None:
        """Process queued log messages and append them to the text widget."""

        while True:
            try:
                level, message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_log_message(level, message)

        self.after(100, self._drain_log_queue)

    def _append_log_message(self, level: str, message: str) -> None:
        """Append a formatted log message to the text widget with auto-scroll."""

        tag = level if level in {"info", "warning", "error"} else "info"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n", (tag,))
        self.log_text.configure(state="disabled")
        self.log_text.see("end")


__all__ = ["SandboxLauncherGUI", "TextWidgetHandler"]
