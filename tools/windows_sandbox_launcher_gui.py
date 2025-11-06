"""GUI for launching Windows sandbox environments."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont


class SandboxLauncherGUI(tk.Tk):
    """Tkinter-based GUI shell for managing sandbox launch tasks."""

    DEFAULT_GEOMETRY = "900x600"
    WINDOW_TITLE = "Windows Sandbox Launcher"

    def __init__(self) -> None:
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.geometry(self.DEFAULT_GEOMETRY)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._create_widgets()
        self._configure_layout()

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

    def on_run_preflight(self) -> None:
        """Placeholder callback for the preflight run action."""
        # Future implementation will trigger preflight checks.
        print("Run Preflight button pressed")

    def on_start_sandbox(self) -> None:
        """Placeholder callback for the sandbox start action."""
        # Future implementation will start the sandbox.
        print("Start Sandbox button pressed")

    def on_close(self) -> None:
        """Handle application shutdown."""
        self.destroy()


__all__ = ["SandboxLauncherGUI"]
