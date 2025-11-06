"""GUI scaffolding for the Menace sandbox launcher."""

from __future__ import annotations

import importlib
import logging
import os
import queue
import shlex
import subprocess
import threading
import time
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import ttk
from tkinter import font as tkfont

from sandbox_runner import bootstrap as sandbox_bootstrap

REPO_ROOT = Path(__file__).resolve().parent.parent


class _GuiLogHandler(logging.Handler):
    """Forward log records into a Tkinter-safe queue."""

    def __init__(self, log_queue: "queue.Queue[tuple[int, str]]") -> None:
        super().__init__()
        self._queue = log_queue
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - logging API
        try:
            message = self.format(record).rstrip()
            self._queue.put((record.levelno, message))
        except Exception:  # noqa: BLE001 - logging API mandates broad handling
            self.handleError(record)


class _SkipFileFilter(logging.Filter):
    """Prevent GUI-only messages from reaching persistent handlers."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - logging API
        return not getattr(record, "gui_only", False)


class SandboxLauncherGUI(tk.Tk):
    """Primary window for the sandbox launcher application."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Menace Sandbox Launcher")
        self.geometry("900x600")

        self._show_debug_var = tk.BooleanVar(value=False)
        self._elapsed_var = tk.StringVar(value="Elapsed: 00:00")

        self._create_widgets()
        self._initialize_state()

    def _initialize_state(self) -> None:
        self._preflight_thread: threading.Thread | None = None
        self._sandbox_thread: threading.Thread | None = None
        self._pause_event = threading.Event()
        self._decision_queue: queue.Queue[
            tuple[str, str, Callable[[], None] | None]
        ] = queue.Queue()
        self._abort_event = threading.Event()
        self._ui_queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._log_queue: queue.Queue[tuple[int, str]] = queue.Queue()
        self._log_handler: _GuiLogHandler | None = None
        self._preflight_steps: list[tuple[str, Callable[[], None]]] = []
        self._last_health_snapshot: dict[str, Any] | None = None
        self._sandbox_process: subprocess.Popen[str] | None = None
        self._retry_requested = False
        self._elapsed_start: float | None = None
        self._elapsed_job: str | None = None
        self._logger: logging.Logger | None = None
        self._closing = False

        self._setup_file_logging()
        self._configure_log_widget()

        self.run_preflight_button.configure(command=self._on_run_preflight)
        self.start_sandbox_button.configure(command=self._on_start_sandbox)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(100, self._process_decisions)
        self.after(100, self._drain_ui_queue)
        self.after(100, self._poll_log_queue)

    def _create_widgets(self) -> None:
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        status_frame = ttk.Frame(self.notebook)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=3)
        status_frame.rowconfigure(1, weight=1)
        self.notebook.add(status_frame, text="Status")

        self.log_text = scrolledtext.ScrolledText(
            status_frame,
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        base_font = tkfont.nametofont(self.log_text.cget("font"))
        self._log_bold_font = base_font.copy()
        self._log_bold_font.configure(weight="bold")

        self.debug_container = ttk.Labelframe(status_frame, text="Debug Details")
        self.debug_text = scrolledtext.ScrolledText(
            self.debug_container,
            state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.debug_text.pack(fill=tk.BOTH, expand=True)
        self.debug_container.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        self.debug_container.grid_remove()

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

        self.show_debug_check = ttk.Checkbutton(
            button_frame,
            text="Show Debug",
            variable=self._show_debug_var,
            command=self._on_toggle_debug,
        )
        self.show_debug_check.pack(side=tk.LEFT, padx=(10, 5))

        self.elapsed_label = ttk.Label(button_frame, textvariable=self._elapsed_var)
        self.elapsed_label.pack(side=tk.RIGHT)

    def _setup_file_logging(self) -> logging.Logger:
        """Configure logging to persistent storage and the GUI."""

        logger = logging.getLogger("menace.sandbox.launcher")
        logger.setLevel(logging.INFO)

        log_dir = REPO_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "sandbox_launcher.log"

        file_handler: RotatingFileHandler | None = None
        for handler in logger.handlers:
            if isinstance(handler, RotatingFileHandler):
                file_handler = handler
                break

        if file_handler is None:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=1_000_000,
                backupCount=5,
                encoding="utf-8",
            )
            logger.addHandler(file_handler)

        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        if not any(isinstance(filt, _SkipFileFilter) for filt in file_handler.filters):
            file_handler.addFilter(_SkipFileFilter())

        handler = self._log_handler
        if handler is None:
            handler = _GuiLogHandler(self._log_queue)
            self._log_handler = handler
        if handler not in logger.handlers:
            logger.addHandler(handler)

        logger.propagate = False
        self._logger = logger
        return logger

    def _configure_log_widget(self) -> None:
        """Prepare the log text widget styling."""

        background = "#1e1e1e"
        self.log_text.configure(background=background, foreground="white", insertbackground="white")
        self.log_text.tag_configure("default", foreground="white", background=background)
        self.log_text.tag_configure("info", foreground="white", background=background)
        self.log_text.tag_configure("warning", foreground="yellow", background=background)
        self.log_text.tag_configure(
            "error", foreground="red", background=background, font=self._log_bold_font
        )
        self.log_text.tag_configure(
            "critical", foreground="red", background=background, font=self._log_bold_font
        )

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
    def log_message(self, level: int, message: str, *, gui_only: bool = False) -> None:
        """Emit *message* at *level* through the configured logger."""

        logger = self._logger
        if logger is None:
            return

        normalized = message.rstrip("\n")
        extra: dict[str, bool] | None = {"gui_only": True} if gui_only else None
        if extra is None:
            logger.log(level, normalized)
        else:
            logger.log(level, normalized, extra=extra)

    def append_log(
        self, message: str, *, level: int = logging.INFO, log_to_file: bool = True
    ) -> None:
        """Append *message* to the log pane and optionally persist it."""

        self.log_message(level, message, gui_only=not log_to_file)

    def clear_log(self) -> None:
        """Clear all log content."""

        self.log_text.configure(state=tk.NORMAL)
        try:
            self.log_text.delete("1.0", tk.END)
        finally:
            self.log_text.configure(state=tk.DISABLED)

    def _append_to_log_widget(self, message: str, level: int, tag: str | None = None) -> None:
        """Render a message in the log widget according to *level*."""

        tag_name = tag or self._log_tag_for_level(level)
        try:
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, message + "\n", tag_name)
            self.log_text.see(tk.END)
        except tk.TclError:
            return
        finally:
            try:
                self.log_text.configure(state=tk.DISABLED)
            except tk.TclError:
                pass

    def _log_tag_for_level(self, level: int) -> str:
        """Return the text tag representing *level*."""

        if level >= logging.CRITICAL:
            return "critical"
        if level >= logging.ERROR:
            return "error"
        if level >= logging.WARNING:
            return "warning"
        if level >= logging.INFO:
            return "info"
        return "default"

    def _append_debug(self, message: str) -> None:
        """Append debug-oriented *message* to the debug pane and log file."""

        logger = self._logger
        if logger is not None:
            logger.debug(message.rstrip("\n"))
        self.debug_text.configure(state=tk.NORMAL)
        try:
            self.debug_text.insert(tk.END, message)
            if not message.endswith("\n"):
                self.debug_text.insert(tk.END, "\n")
            self.debug_text.see(tk.END)
        finally:
            self.debug_text.configure(state=tk.DISABLED)

    def _submit_to_ui(self, callback: Callable[[], None]) -> None:
        """Schedule a callback to run on the Tkinter UI thread."""

        self._ui_queue.put(callback)

    def _on_toggle_debug(self) -> None:
        """Update the debug panel visibility."""

        if self._show_debug_var.get():
            self.debug_container.grid()
        else:
            self.debug_container.grid_remove()

    def _start_elapsed_timer(self) -> None:
        """Begin updating the elapsed time label."""

        self._elapsed_start = time.time()
        if self._elapsed_job is not None:
            self.after_cancel(self._elapsed_job)
            self._elapsed_job = None
        self._update_elapsed_time()

    def _stop_elapsed_timer(self) -> None:
        """Stop updating the elapsed time label."""

        if self._elapsed_job is not None:
            self.after_cancel(self._elapsed_job)
            self._elapsed_job = None
        self._elapsed_start = None
        self._elapsed_var.set("Elapsed: 00:00")

    def _update_elapsed_time(self) -> None:
        """Refresh the elapsed time label."""

        if self._elapsed_start is None:
            self._elapsed_var.set("Elapsed: 00:00")
            self._elapsed_job = None
            return

        elapsed = int(time.time() - self._elapsed_start)
        minutes, seconds = divmod(elapsed, 60)
        self._elapsed_var.set(f"Elapsed: {minutes:02d}:{seconds:02d}")
        self._elapsed_job = self.after(1000, self._update_elapsed_time)

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

    def _poll_log_queue(self) -> None:
        """Flush queued log records into the text widget."""

        while True:
            try:
                level, message = self._log_queue.get_nowait()
            except queue.Empty:
                break

            try:
                tag = self._log_tag_for_level(level)
                self._append_to_log_widget(str(message), level, tag)
            finally:
                self._log_queue.task_done()

        if not self._closing:
            self.after(100, self._poll_log_queue)

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
    def _wait_for_pause_to_clear(self) -> None:
        """Block until the pause flag is cleared or an abort is requested."""

        while self._pause_event.is_set() and not self._abort_event.is_set():
            time.sleep(0.1)

    def _run_command(self, args: Sequence[str], *, cwd: Path | None = None) -> None:
        """Execute *args* streaming output to the GUI log."""

        logger = self._logger
        display = " ".join(shlex.quote(str(part)) for part in args)
        if logger is not None:
            logger.info("$ %s", display)

        process = subprocess.Popen(
            [str(part) for part in args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
        assert process.stdout is not None
        with process.stdout:
            for line in process.stdout:
                normalized = line.rstrip("\n")
                if logger is not None:
                    logger.info("%s", normalized)
        retcode = process.wait()
        if retcode:
            raise subprocess.CalledProcessError(retcode, args)

    def _remove_path(self, path: Path) -> None:
        """Remove a file if it exists and report the action."""

        try:
            if path.exists():
                path.unlink()
                logger = self._logger
                if logger is not None:
                    logger.info("Removed stale file: %s", path)
        except IsADirectoryError:
            return
        except OSError as exc:
            raise RuntimeError(f"Failed to remove {path}: {exc}") from exc

    def _remove_directory(self, path: Path) -> None:
        """Remove a directory tree if it exists and log the outcome."""

        if not path.exists():
            return
        import shutil

        try:
            shutil.rmtree(path)
            logger = self._logger
            if logger is not None:
                logger.info("Removed stale directory: %s", path)
        except OSError as exc:
            raise RuntimeError(f"Failed to remove directory {path}: {exc}") from exc

    def _build_default_preflight_steps(self) -> list[tuple[str, Callable[[], None]]]:
        """Return the default preflight steps when none are configured."""

        return [
            ("Git reset", self._preflight_step_git_reset),
            ("Cleanup", self._preflight_step_cleanup),
            ("Heavy dependencies", self._preflight_step_heavy_dependencies),
            ("Warm caches", self._preflight_step_warm_caches),
            ("Environment setup", self._preflight_step_environment_setup),
            ("Registry & packages", self._preflight_step_registry_and_packages),
            ("Self-coding bootstrap", self._preflight_step_self_coding_bootstrap),
        ]

    def _preflight_step_git_reset(self) -> None:
        self._run_command(["git", "fetch", "origin"], cwd=REPO_ROOT)
        self._run_command(["git", "reset", "--hard", "origin/main"], cwd=REPO_ROOT)

    def _preflight_step_cleanup(self) -> None:
        import bootstrap_self_coding

        bootstrap_self_coding.purge_stale_files()

        sandbox_dir = REPO_ROOT / "sandbox_data"
        if sandbox_dir.exists():
            for lock_file in sandbox_dir.glob("*.lock"):
                self._remove_path(lock_file)

        hf_lock_dir = Path.home() / ".cache" / "huggingface" / "transformers"
        if hf_lock_dir.exists():
            for lock_file in hf_lock_dir.glob("*.lock"):
                self._remove_path(lock_file)

        hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if hub_dir.exists():
            for stale_dir in hub_dir.rglob("*.incomplete"):
                if stale_dir.is_dir():
                    self._remove_directory(stale_dir)

    def _preflight_step_heavy_dependencies(self) -> None:
        logger = self._logger
        if logger is not None:
            logger.info("Preparing heavy dependencies in download-only mode...")

        try:
            module = importlib.import_module("neurosales.scripts.setup_heavy_deps")

            main = getattr(module, "main", None)
            if callable(main):
                main(download_only=True)
            else:
                runner = getattr(module, "run", None)
                if not callable(runner):
                    raise RuntimeError(
                        "setup_heavy_deps module does not expose a callable 'main' or 'run'"
                    )
                runner(download_only=True)
        except Exception:
            if logger is not None:
                logger.exception("Heavy dependency preparation failed")
            raise
        else:
            if logger is not None:
                logger.info("Heavy dependency preparation finished successfully")

    def _preflight_step_warm_caches(self) -> None:
        from vector_service.vectorizer import SharedVectorService

        try:
            SharedVectorService()
        except Exception as exc:
            raise RuntimeError("Failed to initialize SharedVectorService") from exc

    def _preflight_step_environment_setup(self) -> None:
        import auto_env_setup

        auto_env_setup.ensure_env()
        os.environ["SANDBOX_ENABLE_BOOTSTRAP"] = "1"
        os.environ["SANDBOX_ENABLE_SELF_CODING"] = "1"

    def _preflight_step_registry_and_packages(self) -> None:
        import prime_registry

        logger = self._logger
        if logger is not None:
            logger.info("Refreshing sandbox registry...")

        try:
            prime_registry.main()
        except Exception:
            if logger is not None:
                logger.exception("Registry refresh failed")
            raise
        else:
            if logger is not None:
                logger.info("Registry refresh completed successfully.")

        python_exe = Path(sys.executable)
        self._run_command([python_exe, "-m", "pip", "install", "-e", str(REPO_ROOT)])
        self._run_command([python_exe, "-m", "pip", "install", "jsonschema"])

    def _preflight_step_self_coding_bootstrap(self) -> None:
        import bootstrap_self_coding

        bootstrap_self_coding.bootstrap_self_coding("AICounterBot")

    def _clear_decision_queue(self) -> None:
        while True:
            try:
                self._decision_queue.get_nowait()
                self._decision_queue.task_done()
            except queue.Empty:
                break

    def _on_run_preflight(self) -> None:
        if self._preflight_thread and self._preflight_thread.is_alive():
            self.log_message(logging.WARNING, "Preflight is already running.")
            return

        self.disable_run_preflight()
        self.disable_start_sandbox()

        self._abort_event.clear()
        self._pause_event.clear()
        self._clear_decision_queue()
        self._start_elapsed_timer()

        self.log_message(logging.INFO, "Starting preflight checks...")

        self._preflight_thread = threading.Thread(
            target=self._run_preflight_flow,
            daemon=True,
        )
        self._preflight_thread.start()

    # ------------------------------------------------------------------
    # Sandbox process management
    # ------------------------------------------------------------------
    def _on_start_sandbox(self) -> None:
        if self._sandbox_thread and self._sandbox_thread.is_alive():
            self.log_message(logging.WARNING, "Sandbox process is already running.")
            return

        self.disable_run_preflight()
        self.disable_start_sandbox()

        self.log_message(logging.INFO, "Launching sandbox...")

        self._sandbox_thread = threading.Thread(
            target=self._launch_sandbox_process,
            daemon=True,
        )
        self._sandbox_thread.start()

    def _launch_sandbox_process(self) -> None:
        command = [
            sys.executable,
            "-m",
            "menace_sandbox.start_autonomous_sandbox",
        ]
        display = " ".join(shlex.quote(str(part)) for part in command)
        logger = self._logger
        if logger is not None:
            logger.info("$ %s", display)

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception:  # noqa: BLE001 - direct user feedback
            if logger is not None:
                logger.exception("Failed to launch sandbox process")
            self._submit_to_ui(self._on_sandbox_finished)
            return

        self._sandbox_process = process
        try:
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    normalized = line.rstrip("\n")
                    if logger is not None:
                        logger.info("%s", normalized)

            retcode = process.wait()
        finally:
            self._sandbox_process = None

        self._submit_to_ui(lambda retcode=retcode: self._on_sandbox_finished(retcode))

    def _on_sandbox_finished(self, retcode: int | None = None) -> None:
        self._sandbox_thread = None

        if retcode is None:
            self.log_message(logging.ERROR, "Sandbox process did not start.")
        else:
            level = logging.INFO if retcode == 0 else logging.ERROR
            self.log_message(level, f"Sandbox process exited with code {retcode}.")

        self.enable_run_preflight()

        health_ok, health_issues = self._run_post_preflight_health_check()
        if health_ok:
            self.log_message(logging.INFO, "Sandbox health check passed.")
            self.enable_start_sandbox()
        else:
            self.log_message(logging.WARNING, "Sandbox health check reported issues.")
            for issue in health_issues:
                self.log_message(logging.WARNING, f"- {issue}")
            self.disable_start_sandbox()

    def _run_preflight_flow(self) -> None:
        errors: list[tuple[str, Exception]] = []
        steps: Sequence[Tuple[str, Callable[[], None]]] = (
            list(self._preflight_steps) or self._build_default_preflight_steps()
        )
        logger = self._logger

        if not steps:
            if logger is not None:
                logger.warning("No preflight steps have been configured.")

        for title, step in steps:
            if self._abort_event.is_set():
                break

            if logger is not None:
                logger.info("Running preflight step: %s", title)

            while not self._abort_event.is_set():
                try:
                    step()
                except Exception as exc:  # noqa: BLE001 - direct user feedback
                    if logger is not None:
                        logger.exception(
                            "Error during preflight step '%s': %s", title, exc
                        )
                    stack = traceback.format_exc()
                    self._submit_to_ui(
                        lambda title=title, stack=stack: self._append_debug(
                            f"Stack trace for '{title}':\n{stack}"
                        )
                    )
                    self._retry_requested = False
                    self._pause_event.set()
                    self._decision_queue.put(
                        (
                            title,
                            f"Step '{title}' failed with error: {exc}\nDo you want to continue?",
                            step,
                        )
                    )

                    while (
                        self._pause_event.is_set()
                        and not self._abort_event.is_set()
                    ):
                        time.sleep(0.1)

                    if self._abort_event.is_set():
                        break

                    if self._retry_requested:
                        self._retry_requested = False
                        continue

                    errors.append((title, exc))
                    break

                else:
                    if logger is not None:
                        logger.info(
                            "Preflight step '%s' completed successfully.", title
                        )
                    self._wait_for_pause_to_clear()
                    break

            if self._abort_event.is_set():
                break

        aborted = self._abort_event.is_set()
        self._preflight_thread = None

        if aborted:
            self._submit_to_ui(self._on_preflight_aborted)
        else:
            self._submit_to_ui(
                lambda errors=errors: self._on_preflight_finished(
                    success=not errors, errors=errors
                )
            )

    def _on_preflight_aborted(self) -> None:
        """Handle UI updates when the preflight flow is aborted."""

        self.log_message(logging.WARNING, "Preflight run was aborted.")
        self._stop_elapsed_timer()

    def _on_preflight_finished(
        self, *, success: bool, errors: Sequence[Tuple[str, Exception]]
    ) -> None:
        self.enable_run_preflight()
        self._stop_elapsed_timer()

        health_ok, health_issues = self._run_post_preflight_health_check()

        if success:
            self.log_message(logging.INFO, "Preflight completed successfully.")
        else:
            for title, exc in errors:
                self.log_message(
                    logging.ERROR, f"Step '{title}' encountered an issue: {exc}"
                )

            self.log_message(logging.WARNING, "Preflight completed with issues.")
            self.disable_start_sandbox()

        if success and health_ok:
            self.log_message(logging.INFO, "Sandbox health check passed.")
            self.enable_start_sandbox()
            return

        if success:
            self.log_message(logging.WARNING, "Sandbox health check reported issues.")
            for issue in health_issues:
                self.log_message(logging.WARNING, f"- {issue}")

            issues_text = "\n".join(f"- {issue}" for issue in health_issues) or "No additional details available."
            messagebox.showwarning(
                "Sandbox health issues",
                "Sandbox health check reported issues:\n\n" + issues_text,
            )

        if not success:
            return

        self.disable_start_sandbox()

    def _run_post_preflight_health_check(self) -> tuple[bool, list[str]]:
        """Execute the sandbox health probe and store the latest snapshot."""

        health_issues: list[str] = []
        try:
            snapshot = sandbox_bootstrap.sandbox_health()
        except Exception as exc:  # noqa: BLE001 - provide direct feedback
            snapshot = {"error": str(exc)}
            health_issues.append(f"Health check failed: {exc}")
            healthy = False
        else:
            healthy, health_issues = self._evaluate_health_snapshot(snapshot)

        self._last_health_snapshot = snapshot
        return healthy, health_issues

    def _evaluate_health_snapshot(
        self, health: Mapping[str, Any]
    ) -> tuple[bool, list[str]]:
        """Inspect *health* and return a success flag alongside issues."""

        issues: list[str] = []

        if not health.get("self_improvement_thread_alive", True):
            issues.append("Self-improvement thread is not running.")

        if not health.get("databases_accessible", True):
            db_errors = health.get("database_errors")
            if isinstance(db_errors, Mapping) and db_errors:
                details = ", ".join(
                    f"{name}: {error}" for name, error in sorted(db_errors.items())
                )
                issues.append(f"Databases inaccessible ({details}).")
            else:
                issues.append("Databases inaccessible.")

        if not health.get("stub_generator_initialized", True):
            issues.append("Stub generator is not initialised.")

        dependency_info = health.get("dependency_health")
        if isinstance(dependency_info, Mapping):
            missing_entries = [
                item
                for item in dependency_info.get("missing", [])
                if isinstance(item, Mapping)
            ]
            if missing_entries:
                required = [
                    item for item in missing_entries if not item.get("optional", False)
                ]
                optional = [
                    item for item in missing_entries if item.get("optional", False)
                ]
                if required:
                    issues.append(
                        "Missing required dependencies: "
                        + ", ".join(
                            sorted(str(item.get("name", "unknown")) for item in required)
                        )
                    )
                if optional:
                    issues.append(
                        "Missing optional dependencies: "
                        + ", ".join(
                            sorted(str(item.get("name", "unknown")) for item in optional)
                        )
                    )

        return not issues, issues

    def get_last_health_snapshot(self) -> dict[str, Any] | None:
        """Return the most recent sandbox health data."""

        return self._last_health_snapshot

    def _process_decisions(self) -> None:
        if self._pause_event.is_set():
            try:
                title, message, retryable = self._decision_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                if title:
                    self.log_message(logging.INFO, f"Decision required: {title}")
                choice = self._show_pause_dialog(title or "Continue?", message, retryable)
                self._decision_queue.task_done()
                if choice == "continue":
                    self.log_message(logging.INFO, "User elected to continue preflight.")
                    self._retry_requested = False
                    self._pause_event.clear()
                elif choice == "retry" and retryable is not None:
                    self.log_message(
                        logging.INFO, "User elected to retry the preflight step."
                    )
                    self._retry_requested = True
                    self._pause_event.clear()
                else:
                    self.log_message(logging.WARNING, "User elected to abort preflight.")
                    self._abort_event.set()
                    self.abort_preflight()

        self.after(200, self._process_decisions)

    def _show_pause_dialog(
        self, title: str, message: str, retryable: Callable[[], None] | None
    ) -> str:
        """Display a modal pause dialog and return the selected action."""

        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text=message, wraplength=400, justify=tk.LEFT).pack(
            padx=20, pady=15
        )

        result: dict[str, str] = {"value": "abort"}

        def choose(value: str) -> None:
            result["value"] = value
            dialog.destroy()

        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        continue_button = ttk.Button(
            button_frame,
            text="Continue",
            command=lambda: choose("continue"),
        )
        continue_button.pack(side=tk.LEFT)

        if retryable is not None:
            retry_button = ttk.Button(
                button_frame,
                text="Retry Step",
                command=lambda: choose("retry"),
            )
            retry_button.pack(side=tk.LEFT, padx=(10, 0))

        ttk.Button(
            button_frame,
            text="Abort",
            command=lambda: choose("abort"),
        ).pack(side=tk.RIGHT)

        dialog.protocol("WM_DELETE_WINDOW", lambda: choose("abort"))
        dialog.bind("<Escape>", lambda _event: choose("abort"))
        continue_button.focus_set()

        self.wait_window(dialog)
        return result["value"]

    def abort_preflight(self) -> None:
        thread = self._preflight_thread
        if not thread:
            self.enable_run_preflight()
            self.disable_start_sandbox()
            self._stop_elapsed_timer()
            return

        self._abort_event.set()
        self._pause_event.clear()
        self._clear_decision_queue()
        self.log_message(logging.INFO, "Stopping preflight run...")

        thread.join(timeout=5.0)
        if thread.is_alive():
            self.log_message(
                logging.WARNING, "Warning: preflight thread did not finish within timeout."
            )

        self._preflight_thread = None
        self.enable_run_preflight()
        self.disable_start_sandbox()
        self.log_message(logging.WARNING, "Preflight run aborted.")
        self._stop_elapsed_timer()

    def _terminate_sandbox_process(self) -> None:
        process = self._sandbox_process
        if not process:
            return

        if process.poll() is None:
            logger = self._logger
            if logger is not None:
                logger.info("Terminating sandbox process...")
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                if logger is not None:
                    logger.warning("Killing sandbox process...")
                process.kill()
                process.wait()

        self._sandbox_process = None

    def _join_thread(self, thread: threading.Thread | None, name: str) -> None:
        if not thread:
            return

        thread.join(timeout=5.0)
        if thread.is_alive():
            self.log_message(
                logging.WARNING, f"Warning: {name} did not finish within timeout."
            )

    def _on_close(self) -> None:
        self.abort_preflight()
        self._terminate_sandbox_process()
        self._join_thread(self._sandbox_thread, "sandbox thread")
        self._join_thread(self._preflight_thread, "preflight thread")
        self._closing = True
        logger = self._logger
        if logger is not None and self._log_handler is not None:
            logger.removeHandler(self._log_handler)
            self._log_handler.close()
            self._log_handler = None
        self._poll_log_queue()
        super().destroy()


def main() -> None:
    """Launch the sandbox GUI."""

    gui = SandboxLauncherGUI()
    gui.mainloop()


if __name__ == "__main__":
    main()
