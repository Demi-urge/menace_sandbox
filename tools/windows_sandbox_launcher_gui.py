"""GUI for launching the Windows sandbox."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from tkinter import font, ttk


LOGGER = logging.getLogger(__name__)
LOG_QUEUE: "queue.Queue[tuple[str, str]]" = queue.Queue()


class QueueLoggingHandler(logging.Handler):
    """Logging handler that forwards log records to a shared queue."""

    def __init__(self, log_queue: "queue.Queue[tuple[str, str]]") -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - see base class
        try:
            message = self.format(record)
            self._queue.put((record.levelname, message))
        except Exception:  # pragma: no cover - logging infrastructure safety net
            self.handleError(record)


class SandboxLauncherGUI(tk.Tk):
    """Simple GUI for running preflight checks and launching the sandbox."""

    WINDOW_TITLE = "Sandbox Launcher"
    WINDOW_GEOMETRY = "720x480"

    def __init__(self) -> None:
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.geometry(self.WINDOW_GEOMETRY)
        LOGGER.setLevel(logging.INFO)

        self._is_preflight_running = False
        self._preflight_thread: threading.Thread | None = None

        self._log_handler = QueueLoggingHandler(LOG_QUEUE)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        LOGGER.addHandler(self._log_handler)

        self._configure_styles()
        self._build_layout()
        self._configure_text_tags()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_after_id: int | None = None
        self._poll_log_queue()

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

    def _configure_text_tags(self) -> None:
        base_font = font.nametofont(self.status_text.cget("font"))
        bold_font = base_font.copy()
        bold_font.configure(weight="bold")

        tag_styles = {
            "DEBUG": {"foreground": "#6b6b6b"},
            "INFO": {},
            "WARNING": {"foreground": "#b58900", "font": bold_font},
            "ERROR": {"foreground": "#dc322f", "font": bold_font},
            "CRITICAL": {
                "foreground": "#ffffff",
                "background": "#dc322f",
                "font": bold_font,
            },
        }

        for tag, options in tag_styles.items():
            self.status_text.tag_configure(tag, **options)

    def _on_close(self) -> None:
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            finally:
                self._poll_after_id = None

        LOGGER.removeHandler(self._log_handler)
        self._log_handler.close()
        self.destroy()

    def _poll_log_queue(self) -> None:
        try:
            while True:
                level, message = LOG_QUEUE.get_nowait()
                self._insert_message(level, message)
        except queue.Empty:
            pass
        finally:
            if self.winfo_exists():
                self._poll_after_id = self.after(100, self._poll_log_queue)

    def _insert_message(self, level: str, message: str) -> None:
        self.status_text.configure(state="normal")
        tag = level if level in self.status_text.tag_names() else None
        self.status_text.insert("end", f"{message}\n", tag)
        self.status_text.see("end")
        self.status_text.configure(state="disabled")

    def append_log(self, message: str) -> None:
        """Append a message to the status log."""
        self._insert_message("INFO", message)

    def run_preflight(self) -> None:
        """Callback for the Run Preflight button."""
        if self._is_preflight_running:
            LOGGER.info(
                "event=preflight status=ignored reason=already_running"
            )
            return

        self._is_preflight_running = True
        self.preflight_button.state(["disabled"])
        self.start_button.state(["disabled"])

        self._preflight_thread = threading.Thread(
            target=self._run_preflight,
            daemon=True,
        )
        self._preflight_thread.start()

    def start_sandbox(self) -> None:
        """Callback for the Start Sandbox button."""
        LOGGER.info("Sandbox launch initiated.")

    def _run_preflight(self) -> None:
        """Execute the preflight process in a background thread."""
        success = False
        try:
            LOGGER.info("event=preflight status=started")
            LOGGER.info(
                "event=preflight status=running step=validate_environment"
            )
            # Placeholder for actual preflight logic
            LOGGER.info("event=preflight status=running step=collect_metadata")
            success = True
            LOGGER.info("event=preflight status=completed result=success")
        except Exception:
            LOGGER.exception("event=preflight status=failed")
        finally:
            try:
                self.after(0, self._on_preflight_finished, success)
            except tk.TclError:
                # GUI was likely closed before the callback could be scheduled.
                LOGGER.debug("event=preflight status=cleanup action=after_failed")

    def _on_preflight_finished(self, success: bool) -> None:
        """Handle GUI state updates when the preflight thread completes."""
        self._is_preflight_running = False
        self._preflight_thread = None
        self.preflight_button.state(["!disabled"])
        if success:
            self.start_button.state(["!disabled"])
        else:
            self.start_button.state(["disabled"])


__all__ = ["SandboxLauncherGUI"]
