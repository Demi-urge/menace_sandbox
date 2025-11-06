"""GUI for launching Windows sandbox environments."""

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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox
from tkinter import ttk
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence, TextIO

from auto_env_setup import ensure_env
from bootstrap_self_coding import bootstrap_self_coding, purge_stale_files
from neurosales.scripts import setup_heavy_deps
from prime_registry import main as prime_registry_main
from dependency_health import DependencyMode, resolve_dependency_mode


_REPO_ROOT = Path(__file__).resolve().parents[1]

_LOCK_PATTERNS: tuple[tuple[str, str], ...] = (
    ("sandbox_data", "*.lock"),
    ("sandbox_data", "*.lock.*"),
    ("sandbox_data", "*.lock*"),
    ("logs", "*.lock"),
    ("logs", "*.lock.*"),
    ("vector_service", "*.lock"),
    ("vector_service", "*.lock.*"),
)

_STALE_DIRECTORIES: tuple[Path, ...] = (
    _REPO_ROOT / "sandbox_data" / "tmp",
    _REPO_ROOT / "sandbox_data" / "cache",
)


@contextmanager
def _temporary_environment(overrides: dict[str, str]) -> Iterator[None]:
    """Temporarily apply ``overrides`` to :data:`os.environ`."""

    original: dict[str, Optional[str]] = {}
    try:
        for key, value in overrides.items():
            original[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, previous in original.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


@dataclass(frozen=True)
class _PreflightStep:
    """Descriptor for a preflight action executed by :func:`run_full_preflight`."""

    name: str
    failure_title: str
    failure_message: str
    runner: Callable[[], None]


def _git_fetch_and_reset(logger: logging.Logger) -> None:
    """Synchronise the working tree with the upstream repository."""

    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git executable not found on PATH")

    logger.info("Fetching latest repository changes")
    subprocess.run(
        [git, "fetch", "--all", "--prune"],
        cwd=_REPO_ROOT,
        check=True,
    )

    branch_proc = subprocess.run(
        [git, "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    branch = branch_proc.stdout.strip() or "HEAD"
    target = "origin/HEAD" if branch == "HEAD" else f"origin/{branch}"

    logger.info("Resetting working tree to %s", target)
    subprocess.run([git, "reset", "--hard", target], cwd=_REPO_ROOT, check=True)
    logger.info("Repository refresh completed successfully")


def _purge_stale_state(logger: logging.Logger) -> None:
    """Remove known stale files produced by previous sandbox runs."""

    logger.info("Purging stale sandbox state")
    purge_stale_files()
    logger.info("Stale sandbox files removed")


def _remove_lock_artifacts(logger: logging.Logger) -> None:
    """Delete stale lock files and temporary directories."""

    failures: list[str] = []
    removed: list[Path] = []

    for directory, pattern in _LOCK_PATTERNS:
        base = _REPO_ROOT / directory
        if not base.exists():
            continue
        for candidate in base.glob(pattern):
            try:
                if candidate.is_dir():
                    shutil.rmtree(candidate)
                else:
                    candidate.unlink()
                removed.append(candidate)
            except FileNotFoundError:
                continue
            except OSError as exc:  # pragma: no cover - surface via interception
                failures.append(f"{candidate}: {exc}")

    for directory in _STALE_DIRECTORIES:
        if not directory.exists():
            continue
        try:
            shutil.rmtree(directory)
            removed.append(directory)
        except FileNotFoundError:
            continue
        except OSError as exc:  # pragma: no cover - surface via interception
            failures.append(f"{directory}: {exc}")

    for entry in removed:
        try:
            logger.info("Removed stale artefact %s", entry.relative_to(_REPO_ROOT))
        except ValueError:  # pragma: no cover - defensive logging
            logger.info("Removed stale artefact %s", entry)

    if failures:
        joined = "; ".join(failures)
        raise RuntimeError(f"Failed to remove stale artefacts: {joined}")

    logger.info("Lock artefact cleanup completed")


def _prefetch_heavy_dependencies(logger: logging.Logger) -> None:
    """Warm heavy dependency assets without installing additional packages."""

    logger.info("Prefetching heavy dependency assets")
    setup_heavy_deps.main(download_only=True)
    logger.info("Heavy dependency prefetch finished")


def _warm_shared_vector_service(logger: logging.Logger) -> None:
    """Instantiate :class:`SharedVectorService` to prime embedding caches."""

    logger.info("Initialising SharedVectorService")
    from vector_service import SharedVectorService

    service = SharedVectorService()
    service.vectorise("text", {"text": "sandbox preflight warm-up"})
    logger.info("SharedVectorService warm-up completed")


def _ensure_environment(logger: logging.Logger) -> None:
    """Generate or update the sandbox environment file with safe overrides."""

    overrides = {
        "MENACE_NON_INTERACTIVE": "1",
        "MENACE_SAFE": "0",
        "MENACE_SUPPRESS_PROMETHEUS_FALLBACK_NOTICE": "1",
        "SANDBOX_DISABLE_CLEANUP": "1",
    }
    logger.info("Ensuring .env configuration with overrides")
    with _temporary_environment(overrides):
        ensure_env()
    logger.info("Environment configuration verified")


def _prime_self_coding_registry(logger: logging.Logger) -> None:
    """Populate the bot registry cache for faster self-coding start-up."""

    logger.info("Priming self-coding registry cache")
    prime_registry_main()
    logger.info("Self-coding registry primed successfully")


def _run_pip_commands(logger: logging.Logger) -> None:
    """Execute required ``pip install`` commands."""

    commands: list[tuple[str, ...]] = [
        (sys.executable, "-m", "pip", "install", "-r", str(_REPO_ROOT / "requirements.txt")),
    ]

    extras = _REPO_ROOT / "requirements-extra.txt"
    if extras.exists():
        commands.append(
            (sys.executable, "-m", "pip", "install", "-r", str(extras))
        )

    for command in commands:
        logger.info("Executing pip command: %s", " ".join(command))
        subprocess.run(command, check=True)

    logger.info("Pip dependency installation completed")


def _bootstrap_ai_counter_bot(logger: logging.Logger) -> None:
    """Trigger self-coding bootstrap for ``AICounterBot``."""

    logger.info("Bootstrapping self-coding manager for AICounterBot")
    bootstrap_self_coding("AICounterBot")
    logger.info("AICounterBot bootstrap completed")


def run_full_preflight(
    *,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: queue.Queue[tuple[str, str]],
    abort_event: threading.Event,
) -> None:
    """Execute the sandbox preflight checks using *logger* for progress."""

    logger.info("starting Windows sandbox preflight routine")

    steps: list[_PreflightStep] = [
        _PreflightStep(
            name="Git repository refresh",
            failure_title="Repository refresh failed",
            failure_message=(
                "Synchronising the repository with 'git fetch' and 'git reset' "
                "did not complete successfully. Continue the preflight?"
            ),
            runner=lambda: _git_fetch_and_reset(logger),
        ),
        _PreflightStep(
            name="Stale file cleanup",
            failure_title="Stale file cleanup failed",
            failure_message=(
                "Cleaning up stale sandbox state failed. Continue the preflight?"
            ),
            runner=lambda: _purge_stale_state(logger),
        ),
        _PreflightStep(
            name="Lock artefact removal",
            failure_title="Lock artefact removal failed",
            failure_message=(
                "Removing stale lock files or directories failed. Continue the preflight?"
            ),
            runner=lambda: _remove_lock_artifacts(logger),
        ),
        _PreflightStep(
            name="Heavy dependency prefetch",
            failure_title="Heavy dependency prefetch failed",
            failure_message=(
                "Downloading heavy dependency assets failed. Continue the preflight?"
            ),
            runner=lambda: _prefetch_heavy_dependencies(logger),
        ),
        _PreflightStep(
            name="SharedVectorService warm-up",
            failure_title="SharedVectorService warm-up failed",
            failure_message=(
                "Initialising SharedVectorService did not succeed. Continue the preflight?"
            ),
            runner=lambda: _warm_shared_vector_service(logger),
        ),
        _PreflightStep(
            name="Environment configuration",
            failure_title="Environment configuration failed",
            failure_message=(
                "Ensuring the sandbox environment configuration failed. Continue the preflight?"
            ),
            runner=lambda: _ensure_environment(logger),
        ),
        _PreflightStep(
            name="Self-coding registry priming",
            failure_title="Registry priming failed",
            failure_message=(
                "Priming the self-coding registry failed. Continue the preflight?"
            ),
            runner=lambda: _prime_self_coding_registry(logger),
        ),
        _PreflightStep(
            name="Pip dependency installation",
            failure_title="Pip installation failed",
            failure_message=(
                "One of the pip installation commands failed. Continue the preflight?"
            ),
            runner=lambda: _run_pip_commands(logger),
        ),
        _PreflightStep(
            name="AICounterBot bootstrap",
            failure_title="AICounterBot bootstrap failed",
            failure_message=(
                "Bootstrapping the AICounterBot self-coding manager failed. Continue the preflight?"
            ),
            runner=lambda: _bootstrap_ai_counter_bot(logger),
        ),
    ]

    for step in steps:
        if abort_event.is_set():
            logger.info("preflight routine aborted before completing remaining steps")
            return

        logger.info("Starting preflight step: %s", step.name)
        try:
            step.runner()
        except Exception:
            logger.exception(step.failure_title)
            pause_event.set()
            decision_queue.put((step.failure_title, step.failure_message))
            while pause_event.is_set() and not abort_event.is_set():
                time.sleep(0.1)
            if abort_event.is_set():
                logger.info("preflight routine aborted during paused step")
                return
            logger.info("Resuming preflight after operator confirmation")
        else:
            logger.info("Preflight step succeeded: %s", step.name)

    logger.info("preflight routine completed successfully")


def _dependency_failure_messages(
    dependency_health: Mapping[str, Any] | None,
    *,
    dependency_mode: DependencyMode,
) -> list[str]:
    """Return user-facing failure reasons derived from dependency metadata."""

    if not isinstance(dependency_health, Mapping):
        return []

    missing: Sequence[Mapping[str, Any]] = tuple(
        item
        for item in dependency_health.get("missing", [])
        if isinstance(item, Mapping)
    )

    if not missing:
        return []

    required = [item for item in missing if not item.get("optional", False)]
    optional = [item for item in missing if item.get("optional", False)]

    failures: list[str] = []
    if required:
        failures.append(
            "missing required dependencies: "
            + ", ".join(sorted(str(item.get("name", "unknown")) for item in required))
        )
    if dependency_mode is not DependencyMode.MINIMAL and optional:
        failures.append(
            "missing optional dependencies in strict mode: "
            + ", ".join(sorted(str(item.get("name", "unknown")) for item in optional))
        )
    return failures


def _evaluate_health_snapshot(
    health: Mapping[str, Any],
    *,
    dependency_mode: DependencyMode | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate sandbox health metadata and return success flag and failures."""

    mode = dependency_mode or resolve_dependency_mode()
    failures: list[str] = []

    if not health.get("databases_accessible", True):
        db_errors = health.get("database_errors")
        if isinstance(db_errors, Mapping) and db_errors:
            details = ", ".join(
                f"{name}: {error}" for name, error in sorted(db_errors.items())
            )
            failures.append(f"databases inaccessible ({details})")
        else:
            failures.append("databases inaccessible")

    failures.extend(
        _dependency_failure_messages(
            health.get("dependency_health"),
            dependency_mode=mode,
        )
    )

    return not failures, failures


def _collect_sandbox_health() -> Mapping[str, Any]:
    """Return the current sandbox health snapshot."""

    from sandbox_runner import bootstrap as sandbox_bootstrap

    return sandbox_bootstrap.sandbox_health()


class _LogQueueHandler(logging.Handler):
    """Forward log records from worker threads to a :class:`queue.Queue`."""

    _LEVEL_TO_TAG = {
        logging.DEBUG: "info",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def __init__(self, log_queue: queue.Queue[tuple[str, str]]) -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        try:
            message = self.format(record)
        except Exception:  # pragma: no cover - defensive fallback
            self.handleError(record)
            return
        tag = self._LEVEL_TO_TAG.get(record.levelno, "info")
        self._queue.put((tag, message))


class SandboxLauncherGUI(tk.Tk):
    """Tkinter-based GUI shell for managing sandbox launch tasks."""

    DEFAULT_GEOMETRY = "900x600"
    WINDOW_TITLE = "Windows Sandbox Launcher"

    def __init__(self) -> None:
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.geometry(self.DEFAULT_GEOMETRY)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self._log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._decision_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._log_handler: Optional[_LogQueueHandler] = None
        self._logger = logging.getLogger(__name__ + ".preflight")
        self._launch_logger = logging.getLogger(__name__ + ".launch")
        self._preflight_thread: Optional[threading.Thread] = None
        self._sandbox_thread: Optional[threading.Thread] = None
        self._sandbox_process: Optional[subprocess.Popen[str]] = None
        self._pause_event = threading.Event()
        self._abort_event = threading.Event()
        self._sandbox_abort_event = threading.Event()
        self._awaiting_decision = False
        self._max_log_lines = 1000
        self._health_snapshot: Optional[Mapping[str, Any]] = None
        self._health_failures: list[str] = []

        self._create_widgets()
        self._configure_layout()
        self._setup_logging()

    def _create_widgets(self) -> None:
        """Instantiate and configure all widgets used by the GUI."""
        self.notebook = ttk.Notebook(self)

        # Status Tab
        self.status_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.status_frame, text="Status")

        fixed_font = tkfont.nametofont("TkFixedFont")

        self.log_text = tk.Text(
            self.status_frame,
            wrap="word",
            state="disabled",
            font=fixed_font,
            background="white",
        )
        self.log_vertical_scroll = ttk.Scrollbar(
            self.status_frame, orient="vertical", command=self.log_text.yview
        )
        self.log_text.configure(yscrollcommand=self.log_vertical_scroll.set)

        # Pre-create tags for log styling.
        self.log_text.tag_configure("info", foreground="black")
        self.log_text.tag_configure("success", foreground="darkgreen")
        self.log_text.tag_configure("warning", foreground="orange")
        self.log_text.tag_configure("error", foreground="red")

        self.footer_frame = ttk.Frame(self)
        self.run_preflight_button = ttk.Button(
            self.footer_frame,
            text="Run Preflight",
            command=self.on_run_preflight,
            state="normal",
        )
        self.start_sandbox_button = ttk.Button(
            self.footer_frame,
            text="Start Sandbox",
            command=self.on_start_sandbox,
            state="disabled",
        )

    def _configure_layout(self) -> None:
        """Configure the geometry management for all widgets."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.footer_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)

        # Status frame layout
        self.status_frame.columnconfigure(0, weight=1)
        self.status_frame.rowconfigure(0, weight=1)

        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_vertical_scroll.grid(row=0, column=1, sticky="ns")

        # Footer buttons
        self.footer_frame.columnconfigure(0, weight=1)
        self.footer_frame.columnconfigure(1, weight=1)

        self.run_preflight_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.start_sandbox_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

    def _setup_logging(self) -> None:
        """Initialise the shared logger/queue used by worker threads."""

        handler = _LogQueueHandler(self._log_queue)
        handler.setFormatter(
            logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
        )
        self._log_handler = handler

        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        if handler not in self._logger.handlers:
            self._logger.addHandler(handler)

        self._launch_logger.setLevel(logging.INFO)
        self._launch_logger.propagate = False
        if handler not in self._launch_logger.handlers:
            self._launch_logger.addHandler(handler)

        self.log_text.configure(state="disabled")
        self.after(100, self._poll_log_queue)

    def _poll_log_queue(self) -> None:
        """Flush queued log records into the text widget."""

        drained: list[tuple[str, str]] = []
        try:
            while True:
                drained.append(self._log_queue.get_nowait())
        except queue.Empty:
            pass

        if drained:
            self.log_text.configure(state="normal")
            for tag, message in drained:
                self.log_text.insert(tk.END, message + "\n", tag)
            self._trim_log()
            self.log_text.configure(state="disabled")
            self.log_text.see(tk.END)

        self.after(100, self._poll_log_queue)

    def _trim_log(self) -> None:
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines <= self._max_log_lines:
            return
        excess = lines - self._max_log_lines
        self.log_text.delete("1.0", f"{excess + 1}.0")

    def on_run_preflight(self) -> None:
        """Kick off the sandbox preflight routine in a background worker."""

        if self._preflight_thread and self._preflight_thread.is_alive():
            return

        self._abort_event.clear()
        self._pause_event.clear()
        self._flush_decision_queue()
        self.run_preflight_button.configure(state="disabled")
        self._logger.info("launching preflight worker")

        thread = threading.Thread(target=self._run_preflight_worker, daemon=True)
        self._preflight_thread = thread
        thread.start()
        self.after(100, self._monitor_preflight_worker)

    def on_start_sandbox(self) -> None:
        """Launch the autonomous sandbox in a background worker."""

        if self._sandbox_thread and self._sandbox_thread.is_alive():
            return

        self.run_preflight_button.configure(state="disabled")
        self.start_sandbox_button.configure(state="disabled")
        self._sandbox_abort_event.clear()
        self._launch_logger.info("starting autonomous sandbox process")

        thread = threading.Thread(target=self._launch_sandbox_worker, daemon=True)
        self._sandbox_thread = thread
        thread.start()

    def _run_preflight_worker(self) -> None:
        """Invoke the preflight routine while reporting via the shared logger."""

        try:
            run_full_preflight(
                logger=self._logger,
                pause_event=self._pause_event,
                decision_queue=self._decision_queue,
                abort_event=self._abort_event,
            )
        except Exception:  # pragma: no cover - log and propagate to UI
            self._logger.exception("preflight routine failed")
        else:
            if not self._abort_event.is_set():
                self._logger.info("preflight worker completed")

    def _monitor_preflight_worker(self) -> None:
        """Re-enable UI controls once the worker thread terminates."""

        thread = self._preflight_thread
        if thread and thread.is_alive():
            self._handle_worker_pause()
            self.after(100, self._monitor_preflight_worker)
            return

        self._preflight_thread = None
        self.run_preflight_button.configure(state="normal")
        if self._abort_event.is_set():
            self._handle_abort_cleanup()
        else:
            self._handle_post_preflight_completion()

    def on_close(self) -> None:
        """Handle application shutdown."""

        self._request_sandbox_abort()
        if self._log_handler is not None:
            self._logger.removeHandler(self._log_handler)
            self._launch_logger.removeHandler(self._log_handler)
        self.destroy()

    def _forward_stream_lines(self, stream: TextIO, label: str, tag: str) -> None:
        """Relay *stream* output to the UI log with the provided *tag*."""

        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                self._log_queue.put((tag, f"[sandbox {label}] {line}"))
        except Exception as exc:  # pragma: no cover - log unexpected failures
            self._log_queue.put(("error", f"failed to read sandbox {label}: {exc}"))
        finally:
            try:
                stream.close()
            except Exception:  # pragma: no cover - stream close best effort
                pass

    def _request_sandbox_abort(self) -> None:
        """Terminate the sandbox process when the user aborts the launch."""

        self._sandbox_abort_event.set()
        process = self._sandbox_process
        if not process or process.poll() is not None:
            return

        self._launch_logger.warning("terminating sandbox process at user request")
        try:
            process.terminate()
        except Exception:
            self._launch_logger.exception("failed to terminate sandbox process")
            return

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._launch_logger.warning(
                "sandbox process did not exit after terminate; killing"
            )
            try:
                process.kill()
            except Exception:
                self._launch_logger.exception("failed to kill sandbox process")

    def _launch_sandbox_worker(self) -> None:
        """Spawn ``start_autonomous_sandbox`` and monitor its lifecycle."""

        if self._sandbox_abort_event.is_set():
            self.after(0, self._handle_sandbox_completion, None, True)
            return

        command = [
            sys.executable,
            "-c",
            "import start_autonomous_sandbox as sas; sas.main([])",
        ]

        try:
            process = subprocess.Popen(
                command,
                cwd=_REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception:
            self._launch_logger.exception("failed to start autonomous sandbox process")
            self.after(0, self._handle_sandbox_completion, None, False)
            return

        self._sandbox_process = process
        stream_threads: list[threading.Thread] = []
        for label, stream, tag in (
            ("stdout", process.stdout, "info"),
            ("stderr", process.stderr, "error"),
        ):
            if stream is None:
                continue
            thread = threading.Thread(
                target=self._forward_stream_lines,
                args=(stream, label, tag),
                daemon=True,
            )
            thread.start()
            stream_threads.append(thread)

        returncode: Optional[int] = None
        aborted = False
        try:
            returncode = process.wait()
            aborted = self._sandbox_abort_event.is_set()
        except Exception:
            self._launch_logger.exception("error while waiting for sandbox process")
        finally:
            for thread in stream_threads:
                thread.join(timeout=1)
            self._sandbox_process = None
            self._sandbox_abort_event.clear()
            self.after(0, self._handle_sandbox_completion, returncode, aborted)

    def _handle_sandbox_completion(
        self, returncode: Optional[int], aborted: bool
    ) -> None:
        """Update UI state once the sandbox process completes."""

        self._sandbox_abort_event.clear()
        self._sandbox_thread = None
        self.run_preflight_button.configure(state="normal")

        if aborted:
            self._log_queue.put(
                ("warning", "sandbox launch aborted by user; preflight re-enabled")
            )
            self._sync_launch_button_with_health()
            return

        if returncode == 0:
            self._log_queue.put(("success", "sandbox process exited successfully"))
            self.start_sandbox_button.configure(state="normal")
            return

        code_display = "unknown" if returncode is None else str(returncode)
        self._log_queue.put(
            ("error", f"sandbox process exited with status {code_display}")
        )
        self.start_sandbox_button.configure(state="disabled")

    def _handle_worker_pause(self) -> None:
        if not self._pause_event.is_set() or self._awaiting_decision:
            return

        try:
            title, message = self._decision_queue.get_nowait()
        except queue.Empty:
            return

        self._awaiting_decision = True
        should_continue = messagebox.askyesno(title=title, message=message)
        if should_continue:
            self._logger.info("user opted to continue after pause")
            self._pause_event.clear()
        else:
            self._logger.info("user requested preflight cancellation")
            self._abort_event.set()
            self._pause_event.clear()
        self._awaiting_decision = False

    def _handle_abort_cleanup(self) -> None:
        self._abort_event.clear()
        self._pause_event.clear()
        self._flush_decision_queue()
        self.start_sandbox_button.configure(state="disabled")
        self._health_snapshot = None
        self._health_failures = []
        self._logger.info("preflight routine cancelled by user")

    def _flush_decision_queue(self) -> None:
        while True:
            try:
                self._decision_queue.get_nowait()
            except queue.Empty:
                break

    def _sync_launch_button_with_health(self) -> None:
        """Enable the launch control when the cached health snapshot is valid."""

        if self._health_snapshot and not self._health_failures:
            self.start_sandbox_button.configure(state="normal")
        else:
            self.start_sandbox_button.configure(state="disabled")

    def _handle_post_preflight_completion(self) -> None:
        """Run sandbox health checks and update UI state accordingly."""

        try:
            health = _collect_sandbox_health()
        except Exception:
            self.start_sandbox_button.configure(state="disabled")
            self._health_snapshot = None
            self._health_failures = []
            self._logger.exception("sandbox health check failed")
            messagebox.showwarning(
                title="Sandbox health check failed",
                message=(
                    "An unexpected error occurred while checking sandbox health.\n\n"
                    "Review the logs for details and rerun the preflight when ready."
                ),
            )
            return

        healthy, failures = _evaluate_health_snapshot(
            health, dependency_mode=resolve_dependency_mode()
        )
        self._health_snapshot = health
        self._health_failures = failures

        if healthy:
            self.start_sandbox_button.configure(state="normal")
            self._logger.info(
                "sandbox health check passed; Start Sandbox button enabled"
            )
            return

        self.start_sandbox_button.configure(state="disabled")
        summary = "; ".join(failures) if failures else "unknown issues"
        self._logger.warning("sandbox health check reported issues: %s", summary)
        details = "\n".join(f"• {failure}" for failure in failures) or "Unknown issues"
        messagebox.showwarning(
            title="Sandbox health check failed",
            message=(
                "The sandbox health check detected issues:\n\n"
                f"{details}\n\n"
                "Address the reported problems and rerun the preflight to retry."
            ),
        )


__all__ = ["SandboxLauncherGUI", "run_full_preflight"]
