"""GUI for launching the Windows sandbox."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class SandboxLauncherGUI(tk.Tk):
    """Simple GUI for running preflight checks and launching the sandbox."""

    WINDOW_TITLE = "Sandbox Launcher"
    WINDOW_GEOMETRY = "720x480"

    def __init__(self) -> None:
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.geometry(self.WINDOW_GEOMETRY)
        self._configure_styles()
        self._build_layout()

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TNotebook", padding=10)
        style.configure("TButton", padding=(12, 6))

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 6))

        status_frame = ttk.Frame(notebook)
        status_frame.pack_propagate(False)
        notebook.add(status_frame, text="Status")

        self.status_text = tk.Text(
            status_frame,
            wrap="word",
            state="disabled",
            bg=self.cget("bg"),
            relief="flat",
        )
        self.status_text.pack(fill="both", expand=True, padx=8, pady=8)

        buttons_frame = ttk.Frame(self)
        buttons_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        buttons_frame.columnconfigure((0, 1), weight=1)

        self.preflight_button = ttk.Button(
            buttons_frame,
            text="Run Preflight",
            command=self.run_preflight,
        )
        self.preflight_button.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.start_button = ttk.Button(
            buttons_frame,
            text="Start Sandbox",
            command=self.start_sandbox,
            state="disabled",
        )
        self.start_button.grid(row=0, column=1, sticky="ew", padx=(6, 0))

    def append_log(self, message: str) -> None:
        """Append a message to the status log."""
        self.status_text.configure(state="normal")
        self.status_text.insert("end", f"{message}\n")
        self.status_text.see("end")
        self.status_text.configure(state="disabled")

    def run_preflight(self) -> None:
        """Callback for the Run Preflight button."""
        self.append_log("Preflight checks started...")

    def start_sandbox(self) -> None:
        """Callback for the Start Sandbox button."""
        self.append_log("Sandbox launch initiated.")


__all__ = ["SandboxLauncherGUI"]
