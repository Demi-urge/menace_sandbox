"""GUI scaffolding for the Menace sandbox launcher."""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable, Iterable, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import ttk


class SandboxLauncherGUI(tk.Tk):
    """Primary window for the sandbox launcher application."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Menace Sandbox Launcher")
        self.geometry("900x600")

        self._create_widgets()
        self._initialize_state()

    def _initialize_state(self) -> None:
        self._preflight_thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._decision_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._abort_event = threading.Event()
        self._ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._preflight_steps: list[tuple[str, Callable[[], None]]] = []

        self.run_preflight_button.configure(command=self._on_run_preflight)

        self.after(100, self._process_decisions)
        self.after(100, self._drain_ui_queue)

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
    # Preflight configuration helpers
    # ------------------------------------------------------------------
    def configure_preflight_steps(
        self, steps: Iterable[Tuple[str, Callable[[], None]]]
    ) -> None:
        self._preflight_steps = list(steps)

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

    def _submit_to_ui(self, callback: Callable[[], None]) -> None:
        """Schedule a callback to run on the Tkinter UI thread."""

        self._ui_queue.put(callback)

    def _drain_ui_queue(self) -> None:
        """Process pending UI callbacks."""

        while True:
            try:
                callback = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback()
            finally:
                self._ui_queue.task_done()

        self.after(100, self._drain_ui_queue)

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

    # ------------------------------------------------------------------
    # Preflight flow management
    # ------------------------------------------------------------------
    def _clear_decision_queue(self) -> None:
        while True:
            try:
                self._decision_queue.get_nowait()
                self._decision_queue.task_done()
            except queue.Empty:
                break

    def _on_run_preflight(self) -> None:
        if self._preflight_thread and self._preflight_thread.is_alive():
            self.append_log("Preflight is already running.")
            return

        self.disable_run_preflight()
        self.disable_start_sandbox()

        self._abort_event.clear()
        self._pause_event.clear()
        self._clear_decision_queue()

        self.append_log("Starting preflight checks...")

        self._preflight_thread = threading.Thread(
            target=self._run_preflight_flow,
            daemon=True,
        )
        self._preflight_thread.start()

    def _run_preflight_flow(self) -> None:
        errors: list[tuple[str, Exception]] = []
        steps: Sequence[Tuple[str, Callable[[], None]]] = list(self._preflight_steps)

        if not steps:
            self._submit_to_ui(
                lambda: self.append_log("No preflight steps have been configured.")
            )

        for title, step in steps:
            if self._abort_event.is_set():
                break

            self._submit_to_ui(
                lambda title=title: self.append_log(
                    f"Running preflight step: {title}"
                )
            )

            try:
                step()
            except Exception as exc:  # noqa: BLE001 - direct user feedback
                errors.append((title, exc))
                self._submit_to_ui(
                    lambda title=title, exc=exc: self.append_log(
                        f"Error during preflight step '{title}': {exc}"
                    )
                )
                self._pause_event.set()
                self._decision_queue.put(
                    (
                        title,
                        f"Step '{title}' failed with error: {exc}\nDo you want to continue?",
                    )
                )

                while self._pause_event.is_set() and not self._abort_event.is_set():
                    time.sleep(0.1)

            if self._abort_event.is_set():
                break

        aborted = self._abort_event.is_set()
        self._preflight_thread = None

        if aborted:
            self._submit_to_ui(lambda: self.append_log("Preflight run was aborted."))
        else:
            self._submit_to_ui(
                lambda errors=errors: self._on_preflight_finished(
                    success=not errors, errors=errors
                )
            )

    def _on_preflight_finished(
        self, *, success: bool, errors: Sequence[Tuple[str, Exception]]
    ) -> None:
        self.enable_run_preflight()

        if success:
            self.append_log("Preflight completed successfully.")
            self.enable_start_sandbox()
            return

        for title, exc in errors:
            self.append_log(f"Step '{title}' encountered an issue: {exc}")

        self.append_log("Preflight completed with issues.")
        self.disable_start_sandbox()

    def _process_decisions(self) -> None:
        if self._pause_event.is_set():
            try:
                title, message = self._decision_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                if title:
                    self.append_log(f"Decision required: {title}")

                continue_run = messagebox.askyesno("Continue?", message)
                self._decision_queue.task_done()
                if continue_run:
                    self.append_log("User elected to continue preflight.")
                    self._pause_event.clear()
                else:
                    self.append_log("User elected to abort preflight.")
                    self._abort_event.set()
                    self.abort_preflight()

        self.after(200, self._process_decisions)

    def abort_preflight(self) -> None:
        thread = self._preflight_thread
        if not thread:
            self.enable_run_preflight()
            self.disable_start_sandbox()
            return

        self._abort_event.set()
        self._pause_event.clear()
        self._clear_decision_queue()
        self.append_log("Stopping preflight run...")

        thread.join(timeout=5.0)
        if thread.is_alive():
            self.append_log("Warning: preflight thread did not finish within timeout.")

        self._preflight_thread = None
        self.enable_run_preflight()
        self.disable_start_sandbox()
        self.append_log("Preflight run aborted.")
