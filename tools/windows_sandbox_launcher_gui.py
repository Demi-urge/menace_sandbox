"""GUI for launching Windows sandbox operations."""

from __future__ import annotations

import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import tkinter.messagebox as messagebox
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from typing import Callable, Iterable, Mapping, Sequence

from dependency_health import DependencyMode, current_dependency_mode
from menace_sandbox.auto_env_setup import ensure_env
from menace_sandbox.bootstrap_self_coding import (
    bootstrap_self_coding,
    purge_stale_files,
)
from menace_sandbox.lock_utils import is_lock_stale
from menace_sandbox.vector_service.vectorizer import SharedVectorService
from neurosales.scripts import setup_heavy_deps
from prime_registry import main as prime_registry_main
from sandbox_runner.bootstrap import sandbox_health


REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE_PATH = Path(__file__).with_name("menace_gui_logs.txt")
HF_TRANSFORMERS_CACHE = Path.home() / ".cache" / "huggingface" / "transformers"
try:
    LOCK_STALE_TIMEOUT = max(300, int(os.getenv("SANDBOX_LOCK_TIMEOUT", "300")))
except ValueError:
    LOCK_STALE_TIMEOUT = 300


class TextWidgetQueueHandler(logging.Handler):
    """Logging handler that places formatted log messages onto a queue."""

    def __init__(
        self, log_queue: "queue.Queue[tuple[str, str, str | None]]"
    ) -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if not message.endswith("\n"):
                message = f"{message}\n"
            debug_text: str | None = None
            if record.levelno >= logging.ERROR:
                debug_text = message
            extra_detail = getattr(record, "debug_detail", None)
            if extra_detail:
                details = str(extra_detail)
                debug_text = f"{message}{details}\n" if debug_text is None else debug_text
            self.log_queue.put((record.levelname.lower(), message, debug_text))
        except Exception:  # pragma: no cover - rely on logging's default handling
            self.handleError(record)


@dataclass(slots=True)
class PauseContext:
    """Metadata describing a paused preflight step."""

    title: str
    message: str
    handler: Callable[[], None] | None
    step_index: int
    context: dict[str, object] | None = None


class SandboxLauncherGUI(tk.Tk):
    """Simple GUI window for managing sandbox lifecycle actions."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Windows Sandbox Launcher")
        self.geometry("600x400")

        self.log_queue: "queue.Queue[tuple[str, str, str | None]]" = queue.Queue()
        self.logger = logging.getLogger("windows_sandbox_launcher_gui")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self._preflight_thread: threading.Thread | None = None
        self._preflight_running = False
        self._preflight_successful = False
        self._sandbox_thread: threading.Thread | None = None
        self._sandbox_process: subprocess.Popen[str] | None = None
        self._sandbox_running = False
        self.pause_event = threading.Event()
        self.abort_event = threading.Event()
        self.decision_queue: "queue.Queue[tuple[str, str | None, dict[str, object] | None]]" = queue.Queue()

        self._pause_context_lock = threading.Lock()
        self._pause_context: PauseContext | None = None
        self._resume_lock = threading.Lock()
        self._resume_action = "continue"
        self._pause_controls_visible = False
        self._max_debug_lines = 400
        self._debug_visible = False

        self.elapsed_var = tk.StringVar(value="Elapsed: 00:00:00")
        self._elapsed_start: float | None = None
        self._elapsed_last: float = 0.0
        self._elapsed_running = False

        self._build_notebook()
        self._build_controls()
        self._build_pause_controls()
        self._configure_logging()

        self.after(1000, self._update_elapsed_time)
        self._process_log_queue()

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self)

        self.status_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.status_tab, text="Status")

        self.status_paned = ttk.PanedWindow(self.status_tab, orient=tk.VERTICAL)
        self.status_paned.pack(fill="both", expand=True, padx=10, pady=10)

        status_container = ttk.Frame(self.status_paned)
        status_container.columnconfigure(0, weight=1)
        status_container.rowconfigure(0, weight=1)
        self.status_text = tk.Text(
            status_container,
            wrap="word",
            state="disabled",
            background="#1e1e1e",
        )
        self.status_text.grid(row=0, column=0, sticky="nsew")
        self.status_paned.add(status_container, weight=4)

        self.debug_container = ttk.Labelframe(
            self.status_paned, text="Debug Details"
        )
        self.debug_container.columnconfigure(0, weight=1)
        self.debug_container.rowconfigure(0, weight=1)
        self.debug_text = tk.Text(
            self.debug_container,
            wrap="word",
            state="disabled",
            background="#1e1e1e",
        )
        self.debug_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

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

        self.debug_toggle_var = tk.BooleanVar(value=False)
        self.debug_toggle = ttk.Checkbutton(
            controls,
            text="Show Debug",
            variable=self.debug_toggle_var,
            command=self._toggle_debug_panel,
        )
        self.debug_toggle.pack(side="right", padx=(5, 0))

        self.elapsed_label = ttk.Label(controls, textvariable=self.elapsed_var)
        self.elapsed_label.pack(side="right")

    def _build_pause_controls(self) -> None:
        self.pause_controls = ttk.Frame(self)
        self.pause_status_var = tk.StringVar(value="")
        self.pause_status_label = ttk.Label(
            self.pause_controls,
            textvariable=self.pause_status_var,
            anchor="w",
            wraplength=360,
        )
        self.pause_status_label.pack(side="left", expand=True, fill="x")

        self.retry_button = ttk.Button(
            self.pause_controls,
            text="Retry Step",
            command=self._on_retry_step,
            state="disabled",
        )
        self.retry_button.pack(side="left", padx=(5, 0))

        self.resume_button = ttk.Button(
            self.pause_controls,
            text="Resume",
            command=self._on_resume_step,
            state="disabled",
        )
        self.resume_button.pack(side="left", padx=(5, 0))

        self.abort_button = ttk.Button(
            self.pause_controls,
            text="Abort",
            command=self._on_abort_preflight,
            state="disabled",
        )
        self.abort_button.pack(side="left", padx=(5, 0))

        self.pause_controls.pack(fill="x", padx=10, pady=(0, 10))
        self.pause_controls.pack_forget()

    def _configure_logging(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")

        self.status_text.tag_config("info", foreground="white", font=default_font)
        self.status_text.tag_config("warning", foreground="yellow", font=default_font)

        error_font = default_font.copy()
        error_font.configure(weight="bold")
        self.status_text.tag_config("error", foreground="red", font=error_font)
        self._error_font = error_font

        self.debug_text.configure(font=default_font, foreground="white")

        queue_handler = TextWidgetQueueHandler(self.log_queue)
        queue_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(queue_handler)
        self._queue_handler = queue_handler

        try:
            LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        except OSError:
            file_handler = None
        if file_handler is not None:
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)
            self._file_handler = file_handler
        else:
            self._file_handler = None

    def run_preflight(self) -> None:
        """Kick off the preflight routine in a background thread."""
        if self._preflight_running:
            if self._preflight_thread and self._preflight_thread.is_alive():
                self.logger.info(
                    "Preflight run already in progress; ignoring request."
                )
                return

            # A previous worker should have reset the state, but ensure the
            # button becomes usable if the thread has unexpectedly stopped.
            self.logger.warning(
                "Preflight state indicated running without an active thread;"
                " resetting controls."
            )
            self._reset_preflight_state()

        if self._preflight_thread and self._preflight_thread.is_alive():
            self.logger.info("Preflight run already in progress; ignoring request.")
            return

        self.run_preflight_button.configure(state="disabled")
        self.start_sandbox_button.configure(state="disabled")
        self._preflight_successful = False
        self.pause_event.clear()
        self.abort_event.clear()
        self._clear_pause_context()
        self._set_resume_action("continue")
        self._update_pause_controls()
        while not self.decision_queue.empty():
            try:
                self.decision_queue.get_nowait()
            except queue.Empty:  # pragma: no cover - race with concurrent reader
                break
        self._preflight_running = True
        self.logger.info("Preflight checks initiated...")

        self._elapsed_start = time.perf_counter()
        self._elapsed_last = 0.0
        self._elapsed_running = True
        self._refresh_elapsed_display()

        self._preflight_thread = threading.Thread(
            target=self._execute_preflight,
            name="PreflightWorker",
            daemon=True,
        )
        self._preflight_thread.start()

    def start_sandbox(self) -> None:
        """Launch the sandbox in a background worker thread."""
        if self._sandbox_running:
            if self._sandbox_thread and self._sandbox_thread.is_alive():
                self.logger.info(
                    "Sandbox launch already in progress; ignoring duplicate request."
                )
                return

            self.logger.warning(
                "Sandbox state indicated running without an active worker; resetting controls."
            )
            self._sandbox_running = False

        if self._sandbox_thread and self._sandbox_thread.is_alive():
            self.logger.info(
                "Sandbox launch already in progress; ignoring duplicate request."
            )
            return

        self.logger.info("Sandbox startup sequence initiated...")
        self.run_preflight_button.configure(state="disabled")
        self.start_sandbox_button.configure(state="disabled")
        self._sandbox_running = True
        self._sandbox_thread = threading.Thread(
            target=self._launch_sandbox,
            name="SandboxLauncher",
            daemon=True,
        )
        self._sandbox_thread.start()

    def _execute_preflight(self) -> None:
        """Worker routine that performs preflight operations."""

        steps: tuple[tuple[str, str, str, Callable[[], None]], ...] = (
            (
                "Synchronizing repository",
                "Synchronizing repository...",
                "Repository synchronized successfully.",
                self._git_sync,
            ),
            (
                "Cleaning stale files",
                "Purging stale files and caches...",
                "Stale files removed.",
                self._cleanup_stale_files,
            ),
            (
                "Installing heavy dependencies",
                "Fetching heavy dependencies...",
                "Heavy dependencies ready.",
                self._install_heavy_dependencies,
            ),
            (
                "Warming vector service",
                "Priming shared vector service caches...",
                "Vector service warmed.",
                self._warm_shared_vector_service,
            ),
            (
                "Ensuring environment",
                "Ensuring environment configuration...",
                "Environment prepared.",
                self._ensure_env_flags,
            ),
            (
                "Priming registry",
                "Priming bot registry...",
                "Bot registry primed.",
                self._prime_registry,
            ),
            (
                "Installing Python dependencies",
                "Installing Python dependencies...",
                "Python dependencies installed.",
                self._install_python_dependencies,
            ),
            (
                "Bootstrapping self-coding",
                "Bootstrapping self-coding services...",
                "Self-coding bootstrap complete.",
                self._bootstrap_self_coding,
            ),
        )

        try:
            index = 0
            total_steps = len(steps)
            while index < total_steps:
                if self.abort_event.is_set():
                    title = steps[index][0]
                    self.logger.info("Preflight aborted before '%s'.", title)
                    return

                title, start_message, success_message, handler = steps[index]
                self.logger.info(start_message)
                try:
                    handler()
                except Exception as exc:  # pragma: no cover - requires GUI interaction
                    self.logger.exception("%s failed", title)
                    failure_message = f"{title} failed: {exc}"
                    context = {
                        "step": getattr(handler, "__name__", title),
                        "exception": str(exc),
                    }
                    self.pause_event.set()
                    self._set_pause_context(
                        title=title,
                        message=failure_message,
                        handler=handler,
                        step_index=index,
                        context=context,
                    )
                    self.decision_queue.put((title, failure_message, context))
                    self._update_pause_state(title, failure_message, context)
                    action = self._await_resume(title)
                    if action is None:
                        return
                    if action == "restart":
                        self.logger.info(
                            "Restarting preflight from first step after '%s'.",
                            title,
                        )
                        index = 0
                        continue
                    if action == "retry":
                        self.logger.info(
                            "Retrying preflight step '%s' per user request.", title
                        )
                        continue
                    index += 1
                    continue

                self.logger.info(success_message)
                action = self._await_resume(title)
                if action is None:
                    return
                if action == "restart":
                    self.logger.info(
                        "Restarting preflight from first step at user request."
                    )
                    index = 0
                    continue
                if action == "retry":
                    self.logger.info(
                        "Repeating preflight step '%s' at user request.", title
                    )
                    continue
                index += 1

            if self.abort_event.is_set():
                self.logger.info("Preflight aborted after completing queued steps.")
                return
            self.logger.info("Preflight checks complete.")
            self._run_post_preflight_health_check()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Preflight checks failed: %s", exc)
        finally:
            self.after(0, self._reset_preflight_state)

    def _await_resume(self, title: str) -> str | None:
        """Wait for the pause event to clear and return the requested action."""

        while self.pause_event.is_set() and not self.abort_event.is_set():
            time.sleep(0.05)
        if self.abort_event.is_set():
            self.logger.info("Preflight aborted during '%s'.", title)
            return None
        with self._resume_lock:
            action = self._resume_action
            self._resume_action = "continue"
        return action

    def _git_sync(self) -> None:
        """Synchronise the repository with origin/main."""

        commands = (
            ["git", "fetch", "origin"],
            ["git", "reset", "--hard", "origin/main"],
        )
        for command in commands:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self._log_subprocess_result(command, result)

    def _cleanup_stale_files(self) -> None:
        """Remove stale lock files and model cache directories."""

        purge_stale_files()
        removed_files: list[Path] = []
        for target in self._iter_lock_files():
            if self._lock_is_stale(target):
                try:
                    target.unlink()
                    removed_files.append(target)
                except FileNotFoundError:
                    continue
        for removed in removed_files:
            self.logger.info("Removed stale lock file: %s", removed)

        removed_dirs: list[Path] = []
        for directory in self._iter_stale_model_directories():
            try:
                shutil.rmtree(directory)
                removed_dirs.append(directory)
            except FileNotFoundError:
                continue
        for directory in removed_dirs:
            self.logger.info("Removed stale model cache: %s", directory)

    def _iter_lock_files(self) -> Iterable[Path]:
        sandbox_lock_dir = REPO_ROOT / "sandbox_data"
        for path in sandbox_lock_dir.glob("*.lock"):
            if path.is_file():
                yield path

        if HF_TRANSFORMERS_CACHE.exists():
            for path in HF_TRANSFORMERS_CACHE.glob("*.lock"):
                if path.is_file():
                    yield path

    def _lock_is_stale(self, path: Path) -> bool:
        try:
            if is_lock_stale(path.as_posix(), timeout=LOCK_STALE_TIMEOUT):
                return True
        except Exception:  # pragma: no cover - best effort fall back
            pass
        try:
            return time.time() - path.stat().st_mtime > LOCK_STALE_TIMEOUT
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def _iter_stale_model_directories(self) -> Iterable[Path]:
        if not HF_TRANSFORMERS_CACHE.exists():
            return

        cutoff = time.time() - 3600

        def maybe_collect(path: Path) -> Iterable[Path]:
            try:
                if not path.is_dir():
                    return []
            except OSError:
                return []

            name = path.name.lower()
            if name.endswith((".partial", ".incomplete", ".tmp")):
                return [path]

            try:
                entries = list(path.iterdir())
            except FileNotFoundError:
                return []
            except OSError:
                return []

            lock_files = [entry for entry in entries if entry.suffix == ".lock"]
            if lock_files and all(self._lock_is_stale(lock) for lock in lock_files):
                return [path]

            if not entries:
                try:
                    if path.stat().st_mtime < cutoff:
                        return [path]
                except OSError:
                    return [path]

            return []

        downloads_dir = HF_TRANSFORMERS_CACHE / "downloads"
        if downloads_dir.exists():
            for candidate in downloads_dir.iterdir():
                yield from maybe_collect(candidate)

        for model_dir in HF_TRANSFORMERS_CACHE.glob("models--*"):
            snapshots = model_dir / "snapshots"
            if snapshots.exists():
                for snapshot in snapshots.iterdir():
                    yield from maybe_collect(snapshot)
            refs_dir = model_dir / "refs"
            if refs_dir.exists():
                for ref in refs_dir.iterdir():
                    yield from maybe_collect(ref)

    def _install_heavy_dependencies(self) -> None:
        setup_heavy_deps.main(download_only=True)

    def _warm_shared_vector_service(self) -> None:
        SharedVectorService()

    def _ensure_env_flags(self) -> None:
        ensure_env()
        os.environ["SANDBOX_ENABLE_BOOTSTRAP"] = "1"
        os.environ["SANDBOX_ENABLE_SELF_CODING"] = "1"

    def _prime_registry(self) -> None:
        prime_registry_main()

    def _install_python_dependencies(self) -> None:
        commands = (
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-e",
                str(REPO_ROOT),
            ],
            [sys.executable, "-m", "pip", "install", "jsonschema"],
        )
        for command in commands:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self._log_subprocess_result(command, result)

    def _bootstrap_self_coding(self) -> None:
        bootstrap_self_coding("AICounterBot")

    def _log_subprocess_result(
        self, command: Iterable[str], result: subprocess.CompletedProcess[str]
    ) -> None:
        cmd_display = " ".join(command)
        if result.stdout:
            self.logger.info("%s stdout:\n%s", cmd_display, result.stdout.strip())
        if result.stderr:
            self.logger.info("%s stderr:\n%s", cmd_display, result.stderr.strip())

    def _run_post_preflight_health_check(self) -> None:
        try:
            snapshot = sandbox_health()
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Sandbox health check failed: %s", exc)
            self._preflight_successful = False
            self.after(
                0,
                lambda message=f"Sandbox health check failed: {exc}": self._notify_health_failure(
                    [message]
                ),
            )
            return

        dependency_mode = current_dependency_mode()
        healthy, failures = _evaluate_health_snapshot(
            snapshot, dependency_mode=dependency_mode
        )
        if healthy:
            self.logger.info(
                "Sandbox health check succeeded; 'Start Sandbox' is now available."
            )
            self._preflight_successful = True
            self.after(0, self._handle_health_success)
            return

        summary = "; ".join(failures) if failures else "unknown issues"
        self.logger.warning(
            "Sandbox health check reported issues: %s", summary
        )
        self._preflight_successful = False
        self.after(0, lambda details=list(failures): self._notify_health_failure(details))

    def _handle_health_success(self) -> None:
        self.start_sandbox_button.configure(state="normal")
        self._append_status(
            "Sandbox health check succeeded. You may start the sandbox.\n"
        )

    def _notify_health_failure(self, failures: Sequence[str]) -> None:
        message_lines = [
            "Sandbox health issues detected."
        ]
        if failures:
            message_lines.append("")
            message_lines.extend(f"- {failure}" for failure in failures)
        message = "\n".join(message_lines)
        messagebox.showwarning("Sandbox health check", message)
        self._append_status(f"{message}\n", "warning")
        self.start_sandbox_button.configure(state="disabled")

    def _reset_preflight_state(self) -> None:
        """Re-enable UI controls and clear state after preflight finishes."""
        self._preflight_running = False
        self._preflight_thread = None
        if self._elapsed_running and self._elapsed_start is not None:
            self._elapsed_last = time.perf_counter() - self._elapsed_start
        self._elapsed_running = False
        self._elapsed_start = None
        self._refresh_elapsed_display()
        self.pause_event.clear()
        self._clear_pause_context()
        self._update_pause_controls()
        if not self._sandbox_running:
            self.run_preflight_button.configure(state="normal")
            self.start_sandbox_button.configure(
                state="normal" if self._preflight_successful else "disabled"
            )

    def _process_log_queue(self) -> None:
        while True:
            try:
                payload = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                if not isinstance(payload, tuple):
                    continue
                if len(payload) == 3:
                    tag, message, debug_text = payload
                elif len(payload) == 2:
                    tag, message = payload
                    debug_text = None
                else:
                    continue
                self._append_status(message, tag)
                if debug_text:
                    self._append_debug_message(debug_text)

        self._drain_decision_queue()
        self._update_pause_controls()
        self.after(100, self._process_log_queue)

    def _append_status(self, message: str, tag: str = "info") -> None:
        self.status_text.configure(state="normal")
        self.status_text.insert("end", message, (tag,))
        self.status_text.configure(state="disabled")
        self.status_text.see("end")

    def _drain_decision_queue(self) -> None:
        while True:
            try:
                payload = self.decision_queue.get_nowait()
            except queue.Empty:
                break
            if not isinstance(payload, tuple) or not payload:
                continue
            title = str(payload[0])
            message: str | None = None
            context: dict[str, object] | None = None
            if len(payload) > 1 and payload[1] is not None:
                message = str(payload[1])
            if len(payload) > 2 and isinstance(payload[2], dict):
                context = payload[2]

            detail_lines = [line for line in (message,) if line]
            if context and context.get("exception"):
                detail_lines.append(str(context["exception"]))

            if detail_lines:
                combined = "\n".join(detail_lines)
                self._append_debug_message(
                    f"Pause details for {title}:\n{combined}\n"
                )

    def _append_debug_message(self, message: str) -> None:
        if not message:
            return
        if not message.endswith("\n"):
            message = f"{message}\n"
        self.debug_text.configure(state="normal")
        self.debug_text.insert("end", message)
        self._trim_debug_log()
        self.debug_text.configure(state="disabled")
        if self.debug_toggle_var.get():
            self.debug_text.see("end")

    def _trim_debug_log(self) -> None:
        lines = int(self.debug_text.index("end-1c").split(".")[0])
        if lines <= self._max_debug_lines:
            return
        excess = lines - self._max_debug_lines
        self.debug_text.delete("1.0", f"{excess + 1}.0")

    def _toggle_debug_panel(self) -> None:
        if self.debug_toggle_var.get():
            if not self._debug_visible:
                self.status_paned.add(self.debug_container, weight=1)
                self._debug_visible = True
                self.debug_text.see("end")
        elif self._debug_visible:
            self.status_paned.forget(self.debug_container)
            self._debug_visible = False

    def _get_pause_context(self) -> PauseContext | None:
        with self._pause_context_lock:
            return self._pause_context

    def _set_pause_context(
        self,
        *,
        title: str,
        message: str,
        handler: Callable[[], None] | None,
        step_index: int,
        context: dict[str, object] | None = None,
    ) -> None:
        pause_context = PauseContext(
            title=title,
            message=message,
            handler=handler,
            step_index=step_index,
            context=context,
        )
        with self._pause_context_lock:
            self._pause_context = pause_context
        self.after(0, self._update_pause_controls)

    def _clear_pause_context(self) -> None:
        with self._pause_context_lock:
            self._pause_context = None

    def _set_resume_action(self, action: str) -> None:
        with self._resume_lock:
            self._resume_action = action

    def _refresh_elapsed_display(self) -> None:
        seconds = self._elapsed_last
        if self._elapsed_running and self._elapsed_start is not None:
            seconds = time.perf_counter() - self._elapsed_start
            self._elapsed_last = seconds
        elapsed_seconds = int(seconds)
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        self.elapsed_var.set(f"Elapsed: {hours:02d}:{minutes:02d}:{secs:02d}")

    def _update_elapsed_time(self) -> None:
        self._refresh_elapsed_display()
        self.after(1000, self._update_elapsed_time)

    def _update_pause_controls(self) -> None:
        if self.pause_event.is_set():
            context = self._get_pause_context()
            detail_lines: list[str] = []
            if context and context.message:
                detail_lines.append(str(context.message))
            else:
                detail_lines.append("Preflight paused.")
            if context and context.context and context.context.get("exception"):
                detail_lines.append(str(context.context["exception"]))
            if not (context and context.handler):
                detail_lines.append(
                    "Retry Step will restart the full preflight sequence."
                )
            self.pause_status_var.set("\n".join(detail_lines))
            if not self._pause_controls_visible:
                self.pause_controls.pack(fill="x", padx=10, pady=(0, 10))
                self._pause_controls_visible = True
            self.retry_button.state(["!disabled"])
            self.resume_button.state(["!disabled"])
            self.abort_button.state(["!disabled"])
        else:
            if self._pause_controls_visible:
                self.pause_controls.pack_forget()
                self._pause_controls_visible = False
            self.pause_status_var.set("")
            self.retry_button.state(["disabled"])
            self.resume_button.state(["disabled"])
            self.abort_button.state(["disabled"])

    def _on_retry_step(self) -> None:
        context = self._get_pause_context()
        if context and context.handler is not None:
            self.logger.info("User requested retry of '%s'.", context.title)
            self._set_resume_action("retry")
        else:
            if context:
                self.logger.info(
                    "Retry requested without handler; restarting after '%s'.",
                    context.title,
                )
            else:
                self.logger.info("Retry requested without pause context; restarting.")
            self._set_resume_action("restart")
        self.pause_event.clear()
        self._clear_pause_context()
        self._update_pause_controls()

    def _on_resume_step(self) -> None:
        context = self._get_pause_context()
        if context:
            self.logger.info("User opted to continue after '%s'.", context.title)
        else:
            self.logger.info("User opted to continue preflight.")
        self._set_resume_action("continue")
        self.pause_event.clear()
        self._clear_pause_context()
        self._update_pause_controls()

    def _on_abort_preflight(self) -> None:
        context = self._get_pause_context()
        if context:
            self.logger.warning("User aborted preflight after '%s'.", context.title)
        else:
            self.logger.warning("User aborted preflight while paused.")
        self.abort_event.set()
        self._set_resume_action("continue")
        self.pause_event.clear()
        self._clear_pause_context()
        self._update_pause_controls()

    def _update_pause_state(
        self, title: str, message: str, context: dict[str, object] | None = None
    ) -> None:
        details = message or "No additional details provided."
        if context and context.get("exception"):
            details = f"{details}\nException: {context['exception']}"
        self.after(
            0,
            lambda: self._append_status(
                f"Preflight paused: {title} encountered an error. {details}\n",
                "warning",
            ),
        )

    def _launch_sandbox(self) -> None:
        """Launch the sandbox process and stream its output into the log queue."""

        command = [sys.executable, "-m", "start_autonomous_sandbox"]
        self.logger.info("Executing sandbox command: %s", " ".join(command))

        try:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to start sandbox process: %s", exc)
            self.after(
                0,
                lambda error=exc: self._handle_sandbox_exit(
                    returncode=None, error=error
                ),
            )
            return

        self._sandbox_process = process

        readers: list[threading.Thread] = []

        def stream_reader(stream: Iterable[str] | None, severity: str) -> None:
            if stream is None:
                return
            for raw_line in stream:
                line = raw_line.rstrip("\r\n")
                if severity == "info":
                    self.logger.info("[sandbox] %s", line)
                else:
                    lowered = line.lower()
                    if any(keyword in lowered for keyword in ("error", "failed", "exception")):
                        self.logger.error("[sandbox] %s", line)
                    else:
                        self.logger.warning("[sandbox] %s", line)

        for stream, severity in (
            (process.stdout, "info"),
            (process.stderr, "warning"),
        ):
            if stream is None:
                continue
            reader = threading.Thread(
                target=stream_reader,
                args=(stream, severity),
                name=f"SandboxStream-{severity}",
                daemon=True,
            )
            reader.start()
            readers.append(reader)

        returncode: int | None = None
        error: Exception | None = None
        try:
            returncode = process.wait()
        except Exception as exc:  # pragma: no cover - defensive logging
            error = exc
            self.logger.exception("Sandbox process encountered an error: %s", exc)
            try:
                process.kill()
            except Exception:
                pass
        finally:
            for reader in readers:
                reader.join(timeout=1)
            try:
                if process.stdout:
                    process.stdout.close()
                if process.stderr:
                    process.stderr.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

        self.after(
            0,
            lambda rc=returncode, err=error: self._handle_sandbox_exit(
                returncode=rc, error=err
            ),
        )

    def _handle_sandbox_exit(
        self, *, returncode: int | None, error: Exception | None
    ) -> None:
        """Reset UI state after the sandbox process finishes."""

        self._sandbox_running = False
        self._sandbox_thread = None
        self._sandbox_process = None

        self.run_preflight_button.configure(state="normal")
        self.start_sandbox_button.configure(
            state="normal" if self._preflight_successful else "disabled"
        )

        if error is not None:
            message = f"Sandbox failed to start: {error}"
            self.logger.error(message)
            messagebox.showerror("Sandbox launch failed", message)
            return

        if returncode is None:
            self.logger.warning(
                "Sandbox process exited with unknown status; controls re-enabled."
            )
            return

        if returncode == 0:
            self.logger.info("Sandbox process exited successfully.")
        else:
            message = (
                f"Sandbox process exited with code {returncode}."
                " Check logs for details."
            )
            self.logger.error(message)
            messagebox.showerror("Sandbox exited", message)


def _evaluate_health_snapshot(
    snapshot: Mapping[str, object], *, dependency_mode: DependencyMode
) -> tuple[bool, list[str]]:
    """Return whether ``snapshot`` indicates a healthy sandbox and any failures."""

    failures: list[str] = []

    databases_accessible = bool(snapshot.get("databases_accessible", True))
    if not databases_accessible:
        raw_errors = snapshot.get("database_errors")
        if isinstance(raw_errors, Mapping) and raw_errors:
            detail = ", ".join(f"{name}: {error}" for name, error in raw_errors.items())
        else:
            detail = "unknown error"
        failures.append(f"Sandbox databases inaccessible ({detail})")

    dependency_info = snapshot.get("dependency_health")
    missing_entries: Sequence[object] = ()
    if isinstance(dependency_info, Mapping):
        raw_missing = dependency_info.get("missing")
        if isinstance(raw_missing, Sequence) and not isinstance(raw_missing, (str, bytes)):
            missing_entries = raw_missing

    dependency_failures: list[str] = []
    for entry in missing_entries:
        optional = False
        description = "unknown dependency"
        category: str | None = None
        reason: str | None = None

        if isinstance(entry, Mapping):
            description = str(
                entry.get("name")
                or entry.get("description")
                or entry.get("reason")
                or description
            )
            optional = bool(entry.get("optional", False))
            category_value = entry.get("category")
            if category_value:
                category = str(category_value)
            reason_value = entry.get("reason") or entry.get("description")
            if reason_value:
                reason = str(reason_value)
        else:
            description = str(entry)

        if optional and dependency_mode is not DependencyMode.STRICT:
            continue

        parts = [description]
        extra_bits = [value for value in (category, reason) if value]
        if optional:
            extra_bits.append("optional")
        if extra_bits:
            parts.append(f"({'; '.join(extra_bits)})")
        dependency_failures.append(" ".join(parts))

    if dependency_failures:
        failures.append(
            "Missing dependencies: " + ", ".join(sorted(set(dependency_failures)))
        )

    return not failures, failures


def main() -> None:
    """Launch the sandbox GUI event loop."""

    app = SandboxLauncherGUI()
    app.mainloop()


__all__ = ["SandboxLauncherGUI", "_evaluate_health_snapshot", "main"]


if __name__ == "__main__":
    main()
