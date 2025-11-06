"""Minimal Windows sandbox launcher GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class SandboxLauncherGUI(tk.Tk):
    """Basic GUI window containing status output and sandbox controls."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._configure_window()
        self._build_layout()

    def _configure_window(self) -> None:
        """Configure the top-level window attributes."""
        self.title("Windows Sandbox Launcher")
        self.geometry("640x480")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def _build_layout(self) -> None:
        """Create the notebook, status tab, and control buttons."""
        container = ttk.Frame(self, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(container)
        notebook.grid(row=0, column=0, sticky="nsew")

        status_frame = ttk.Frame(notebook, padding=12)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        self.status_text = tk.Text(status_frame, wrap="word", height=10)
        self.status_text.grid(row=0, column=0, sticky="nsew")

        notebook.add(status_frame, text="Status")

        controls = self._create_controls(container)
        controls.grid(row=1, column=0, pady=(12, 0), sticky="ew")

    def _create_controls(self, parent: ttk.Frame) -> ttk.Frame:
        """Return a frame containing the preflight and sandbox buttons."""
        frame = ttk.Frame(parent)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        self.preflight_button = ttk.Button(frame, text="Run Preflight")
        self.preflight_button.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.start_button = ttk.Button(frame, text="Start Sandbox", state="disabled")
        self.start_button.grid(row=0, column=1, padx=(6, 0), sticky="ew")

        return frame
