"""GUI for launching Windows sandbox operations."""

import tkinter as tk
from tkinter import ttk


class SandboxLauncherGUI(tk.Tk):
    """Simple GUI window for managing sandbox lifecycle actions."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Windows Sandbox Launcher")
        self.geometry("600x400")

        self._build_notebook()
        self._build_controls()

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self)

        self.status_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.status_tab, text="Status")

        self.status_text = tk.Text(self.status_tab, wrap="word", state="disabled")
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

    def run_preflight(self) -> None:  # pragma: no cover - placeholder hook
        """Placeholder command that will be implemented in a future iteration."""
        self._append_status("Preflight checks initiated...\n")

    def start_sandbox(self) -> None:  # pragma: no cover - placeholder hook
        """Placeholder command that will be implemented in a future iteration."""
        self._append_status("Sandbox startup sequence initiated...\n")

    def _append_status(self, message: str) -> None:
        self.status_text.configure(state="normal")
        self.status_text.insert("end", message)
        self.status_text.configure(state="disabled")
        self.status_text.see("end")


__all__ = ["SandboxLauncherGUI"]
