"""Tkinter GUI for launching and monitoring the Windows sandbox."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk


class SandboxLauncherGUI(tk.Tk):
    """User interface for running the sandbox preflight and launch steps."""

    WINDOW_TITLE = "Windows Sandbox Launcher"
    WINDOW_GEOMETRY = "900x600"

    def __init__(self) -> None:
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.geometry(self.WINDOW_GEOMETRY)

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
            command=self._on_run_preflight,
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
    def _on_run_preflight(self) -> None:  # pragma: no cover - GUI callback
        self.log_message("Running preflight checks...", "info")

    def _on_start_sandbox(self) -> None:  # pragma: no cover - GUI callback
        self.log_message("Starting sandbox...", "info")
    # endregion callbacks


__all__ = ["SandboxLauncherGUI"]


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    gui = SandboxLauncherGUI()
    gui.mainloop()
