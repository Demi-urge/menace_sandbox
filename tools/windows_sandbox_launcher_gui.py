"""GUI scaffolding for the Menace sandbox launcher."""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk


class SandboxLauncherGUI(tk.Tk):
    """Primary window for the sandbox launcher application."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Menace Sandbox Launcher")
        self.geometry("900x600")

        self._create_widgets()

    def _create_widgets(self) -> None:
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        status_frame = ttk.Frame(self.notebook)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        self.notebook.add(status_frame, text="Status")

        self.log_text = scrolledtext.ScrolledText(
            status_frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.run_preflight_button = ttk.Button(button_frame, text="Run Preflight")
        self.run_preflight_button.pack(side=tk.LEFT, padx=(0, 5))

        self.start_sandbox_button = ttk.Button(
            button_frame,
            text="Start Sandbox",
            state=tk.DISABLED,
        )
        self.start_sandbox_button.pack(side=tk.LEFT)

    # ------------------------------------------------------------------
    # Log helpers
    # ------------------------------------------------------------------
    def append_log(self, message: str) -> None:
        """Append a message to the log area."""

        self.log_text.configure(state=tk.NORMAL)
        try:
            self.log_text.insert(tk.END, message)
            if not message.endswith("\n"):
                self.log_text.insert(tk.END, "\n")
            self.log_text.see(tk.END)
        finally:
            self.log_text.configure(state=tk.DISABLED)

    def clear_log(self) -> None:
        """Clear all log content."""

        self.log_text.configure(state=tk.NORMAL)
        try:
            self.log_text.delete("1.0", tk.END)
        finally:
            self.log_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Button state helpers
    # ------------------------------------------------------------------
    def enable_run_preflight(self) -> None:
        self.run_preflight_button.configure(state=tk.NORMAL)

    def disable_run_preflight(self) -> None:
        self.run_preflight_button.configure(state=tk.DISABLED)

    def enable_start_sandbox(self) -> None:
        self.start_sandbox_button.configure(state=tk.NORMAL)

    def disable_start_sandbox(self) -> None:
        self.start_sandbox_button.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Focus helpers
    # ------------------------------------------------------------------
    def focus_run_preflight_button(self) -> None:
        self.run_preflight_button.focus_set()

    def focus_start_sandbox_button(self) -> None:
        self.start_sandbox_button.focus_set()

    def focus_log(self) -> None:
        self.log_text.focus_set()
