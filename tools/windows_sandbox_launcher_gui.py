"""GUI components for launching the Windows sandbox."""

import tkinter as tk
from tkinter import ttk


class SandboxLauncherGUI(tk.Tk):
    """Main application window for the sandbox launcher."""

    WINDOW_TITLE = "Windows Sandbox Launcher"
    WINDOW_GEOMETRY = "800x600"

    def __init__(self) -> None:
        super().__init__()

        self.title(self.WINDOW_TITLE)
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(600, 400)
        self.resizable(width=True, height=True)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_notebook()
        self._build_controls()

    def _build_notebook(self) -> None:
        """Create the notebook and status log tab."""

        self.notebook = ttk.Notebook(self)

        status_frame = ttk.Frame(self.notebook)
        status_frame.rowconfigure(0, weight=1)
        status_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            status_frame,
            wrap="word",
            state="disabled",
            background=self.cget("background"),
            relief="flat",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self.notebook.add(status_frame, text="Status")
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def _build_controls(self) -> None:
        """Create the control buttons below the notebook."""

        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)

        self.run_preflight_button = ttk.Button(
            controls_frame,
            text="Run Preflight",
        )
        self.run_preflight_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.start_sandbox_button = ttk.Button(
            controls_frame,
            text="Start Sandbox",
            state=tk.DISABLED,
        )
        self.start_sandbox_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

