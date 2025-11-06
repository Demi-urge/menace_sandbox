"""GUI components for launching the Windows sandbox."""

import logging
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import messagebox, ttk
from typing import Any, Callable, Optional, Tuple


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TkLogQueueHandler(logging.Handler):
    """Custom logging handler that routes records into a queue for the GUI."""

    LEVEL_TAGS = {
        "debug": "debug",
        "info": "info",
        "warning": "warning",
        "error": "error",
        "critical": "critical",
    }

    def __init__(self, log_queue: "queue.Queue[Tuple[str, str]]") -> None:
        super().__init__(level=logging.DEBUG)
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        if not message.endswith("\n"):
            message += "\n"

        tag = self.LEVEL_TAGS.get(record.levelname.lower(), "info")
        try:
            self.log_queue.put_nowait((tag, message))
        except queue.Full:
            # Drop the log if the queue is full to avoid blocking the UI thread.
            pass


@dataclass(slots=True)
class PreflightStep:
    """Callable preflight step description."""

    description: str
    executor: Callable[..., None]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def run(self) -> None:
        self.executor(*self.args, **self.kwargs)


@dataclass(slots=True)
class PauseDecision:
    """Metadata describing a pause triggered by a failed preflight step."""

    title: str
    message: str
    context: dict[str, Any]
    step: PreflightStep


class _PreflightAborted(Exception):
    """Raised internally when the preflight workflow is aborted."""


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

        self.log_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.worker_queue: "queue.Queue[Tuple[str, Optional[str]]]" = queue.Queue()
        self.decision_queue: "queue.Queue[PauseDecision]" = queue.Queue()
        self.pause_event = threading.Event()
        self.abort_event = threading.Event()
        self._queue_handler = TkLogQueueHandler(self.log_queue)
        self._queue_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(self._queue_handler)
        logger.propagate = False

        self._queue_after_id: Optional[int] = None
        self._worker_after_id: Optional[int] = None
        self._preflight_thread: Optional[threading.Thread] = None
        self._preflight_start_time: Optional[float] = None
        self._drain_running = True
        self._state_lock = threading.Lock()
        self._resume_action: Optional[str] = None
        self._current_pause: Optional[PauseDecision] = None
        self._awaiting_user_decision = False

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_notebook()
        self._build_controls()
        self._schedule_log_drain()

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
            background="#1e1e1e",
            foreground="#ffffff",
            relief="flat",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.log_text.tag_configure("debug", foreground="#bfbfbf")
        self.log_text.tag_configure("info", foreground="#ffffff")
        self.log_text.tag_configure("warning", foreground="#ffd700")
        self.log_text.tag_configure(
            "error", foreground="#ff5555", font=("TkDefaultFont", 10, "bold")
        )
        self.log_text.tag_configure(
            "critical", foreground="#ff0000", font=("TkDefaultFont", 10, "bold")
        )

        self.notebook.add(status_frame, text="Status")
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

    def _build_controls(self) -> None:
        """Create the control buttons below the notebook."""

        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        controls_frame.columnconfigure(2, weight=0)

        self.run_preflight_button = ttk.Button(
            controls_frame,
            text="Run Preflight",
            command=self._on_run_preflight,
        )
        self.run_preflight_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.start_sandbox_button = ttk.Button(
            controls_frame,
            text="Start Sandbox",
            state=tk.DISABLED,
        )
        self.start_sandbox_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.retry_step_button = ttk.Button(
            controls_frame,
            text="Retry Step",
            state=tk.DISABLED,
            command=self._on_retry_step,
        )
        self.retry_step_button.grid(row=0, column=2, sticky="ew", padx=(5, 0))
        self.retry_step_button.grid_remove()

    def _schedule_log_drain(self) -> None:
        if self._drain_running:
            self._queue_after_id = self.after(100, self._drain_log_queue)

    def _drain_log_queue(self) -> None:
        if not self._drain_running:
            return

        flushed = False
        while True:
            try:
                tag, message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                if not flushed:
                    self.log_text.configure(state=tk.NORMAL)
                    flushed = True
                self.log_text.insert(tk.END, message, tag)

        if flushed:
            self.log_text.configure(state=tk.DISABLED)
            self.log_text.see(tk.END)

        self._schedule_log_drain()

    def _schedule_worker_poll(self) -> None:
        if self._drain_running and self._worker_after_id is None:
            self._worker_after_id = self.after(100, self._poll_worker_queue)

    def _poll_worker_queue(self) -> None:
        self._worker_after_id = None

        if not self._drain_running:
            return

        if self.pause_event.is_set():
            self._process_pause_decision()

        try:
            status, message = self.worker_queue.get_nowait()
        except queue.Empty:
            if self._preflight_thread and self._preflight_thread.is_alive():
                self._schedule_worker_poll()
            return

        self._preflight_thread = None

        duration_msg = ""
        if self._preflight_start_time is not None:
            elapsed = time.time() - self._preflight_start_time
            duration_msg = f" (completed in {elapsed:.2f} seconds)"
        self._preflight_start_time = None

        if status == "success":
            logger.info("Preflight completed successfully%s", duration_msg)
            self.start_sandbox_button.configure(state=tk.NORMAL)
        elif status == "aborted":
            logger.info("Preflight aborted by user%s", duration_msg)
            self.start_sandbox_button.configure(state=tk.DISABLED)
        else:
            if message:
                logger.error("Preflight failed: %s", message)
            else:
                logger.error("Preflight failed%s", duration_msg)
            self.start_sandbox_button.configure(state=tk.DISABLED)

        self._clear_pause_ui()
        self.pause_event.clear()
        self.abort_event.clear()
        self.run_preflight_button.configure(state=tk.NORMAL)

    def _on_run_preflight(self) -> None:
        if self._preflight_thread and self._preflight_thread.is_alive():
            logger.debug("Preflight already running; ignoring additional request.")
            return

        self.run_preflight_button.configure(state=tk.DISABLED)
        self.start_sandbox_button.configure(state=tk.DISABLED)
        self._preflight_start_time = time.time()
        self.pause_event.clear()
        self.abort_event.clear()
        self._clear_pause_ui()
        self._drain_decision_queue()
        with self._state_lock:
            self._resume_action = None
        self._current_pause = None
        logger.info("Starting preflight checks at %s", time.strftime("%H:%M:%S"))

        self._preflight_thread = threading.Thread(
            target=self._run_preflight_task,
            name="PreflightThread",
            daemon=True,
        )
        self._preflight_thread.start()
        self._schedule_worker_poll()

    def _run_preflight_task(self) -> None:
        try:
            for step in self._get_preflight_steps():
                self._execute_preflight_step(step)

            try:
                self.worker_queue.put_nowait(("success", None))
            except queue.Full:
                logger.error("Unable to report preflight completion; queue full")
        except _PreflightAborted as aborted:
            try:
                self.worker_queue.put_nowait(("aborted", str(aborted)))
            except queue.Full:
                logger.error("Unable to report preflight abort; queue full")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Unexpected error during preflight execution")
            try:
                self.worker_queue.put_nowait(("error", str(exc)))
            except queue.Full:
                logger.error("Unable to report preflight failure; queue full")

    def _on_close(self) -> None:
        self._drain_running = False
        if self._queue_after_id is not None:
            try:
                self.after_cancel(self._queue_after_id)
            except tk.TclError:
                pass
            self._queue_after_id = None

        if self._worker_after_id is not None:
            try:
                self.after_cancel(self._worker_after_id)
            except tk.TclError:
                pass
            self._worker_after_id = None

        logger.removeHandler(self._queue_handler)
        self.destroy()

    # ------------------------------------------------------------------
    def _get_preflight_steps(self) -> list[PreflightStep]:
        return [
            PreflightStep("Validating configuration", self._validate_configuration),
            PreflightStep("Checking required resources", self._check_required_resources),
            PreflightStep(
                "Verifying sandbox prerequisites", self._verify_sandbox_prerequisites
            ),
        ]

    def _validate_configuration(self) -> None:
        time.sleep(0.1)

    def _check_required_resources(self) -> None:
        time.sleep(0.1)

    def _verify_sandbox_prerequisites(self) -> None:
        time.sleep(0.1)

    def _execute_preflight_step(self, step: PreflightStep) -> None:
        while True:
            if self.abort_event.is_set():
                raise _PreflightAborted(f"Aborted during {step.description}")

            logger.info("Preflight stage: %s", step.description)
            try:
                step.run()
            except Exception as exc:
                logger.exception("Preflight step failed: %s", step.description)
                self._handle_step_failure(step, exc)
                while self.pause_event.is_set():
                    if self.abort_event.is_set():
                        raise _PreflightAborted(
                            f"Aborted during {step.description} after failure"
                        )
                    time.sleep(0.05)

                action = self._consume_resume_action()
                if action == "retry":
                    logger.info("Retrying preflight step: %s", step.description)
                    continue
                else:
                    logger.info(
                        "Continuing after failure in preflight step: %s",
                        step.description,
                    )
                break
            else:
                break

    def _handle_step_failure(self, step: PreflightStep, exc: Exception) -> None:
        with self._state_lock:
            self._resume_action = None

        context = {
            "step": step.description,
            "exception": repr(exc),
        }
        decision = PauseDecision(
            title=f"{step.description} failed",
            message=(
                f"The step '{step.description}' encountered an error:\n{exc}\n\n"
                "Would you like to continue running the preflight checks?"
            ),
            context=context,
            step=step,
        )

        try:
            self.decision_queue.put_nowait(decision)
        except queue.Full:
            logger.error("Unable to queue pause decision for step: %s", step.description)

        self.pause_event.set()

    def _consume_resume_action(self) -> Optional[str]:
        with self._state_lock:
            action = self._resume_action
            self._resume_action = None
        return action

    def _process_pause_decision(self) -> None:
        if self._current_pause is None:
            try:
                self._current_pause = self.decision_queue.get_nowait()
            except queue.Empty:
                return
            else:
                self._show_pause_ui(self._current_pause)

        if self._current_pause is None or self._awaiting_user_decision:
            return

        self._awaiting_user_decision = True
        decision = messagebox.askyesno(
            title=self._current_pause.title,
            message=self._current_pause.message,
        )
        self._awaiting_user_decision = False

        if decision:
            self._resume_from_pause("continue")
        else:
            self._abort_preflight()

    def _resume_from_pause(self, action: str) -> None:
        with self._state_lock:
            self._resume_action = action

        self.pause_event.clear()
        self._clear_pause_ui()
        self._current_pause = None
        logger.info("Resuming preflight after pause with action: %s", action)

    def _abort_preflight(self) -> None:
        self.abort_event.set()
        self.pause_event.clear()
        self._clear_pause_ui()
        self._current_pause = None
        logger.info("Aborting preflight at user request")

    def _on_retry_step(self) -> None:
        if not self.pause_event.is_set() or self._current_pause is None:
            return

        logger.info("User requested retry for step: %s", self._current_pause.step.description)
        self._resume_from_pause("retry")

    def _show_pause_ui(self, decision: PauseDecision) -> None:
        self.retry_step_button.configure(state=tk.NORMAL)
        self.retry_step_button.grid()

        logger.error("Preflight paused: %s", decision.context.get("exception", ""))

    def _clear_pause_ui(self) -> None:
        self.retry_step_button.configure(state=tk.DISABLED)
        self.retry_step_button.grid_remove()

    def _drain_decision_queue(self) -> None:
        while True:
            try:
                self.decision_queue.get_nowait()
            except queue.Empty:
                break

