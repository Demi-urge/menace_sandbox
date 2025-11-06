"""GUI for launching the Windows sandbox environment."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class SandboxLauncherGUI(tk.Tk):
    """Simple GUI used for launching sandbox workflows."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._configure_root()
        self._build_widgets()

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


__all__ = ["SandboxLauncherGUI"]
