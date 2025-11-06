"""Minimal GUI scaffolding for launching the Windows sandbox."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


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

        self.log_text = tk.Text(log_container, wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        self.log_scrollbar = ttk.Scrollbar(
            log_container, orient="vertical", command=self.log_text.yview
        )
        self.log_scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)

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


__all__ = ["SandboxLauncherGUI"]
