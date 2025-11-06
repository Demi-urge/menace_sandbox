"""Tkinter GUI for launching and monitoring the Windows sandbox."""

from __future__ import annotations

import datetime as _dt
import importlib
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Sequence

from dependency_health import DependencyMode, resolve_dependency_mode


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


class SandboxLauncherGUI(tk.Tk):
    """User interface for running the sandbox preflight and launch steps."""

    WINDOW_TITLE = "Windows Sandbox Launcher"
    WINDOW_GEOMETRY = "900x600"

    def __init__(self) -> None:
        super().__init__()
        self.title(self.WINDOW_TITLE)
        self.geometry(self.WINDOW_GEOMETRY)

        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.preflight_thread: threading.Thread | None = None
        self.preflight_abort = threading.Event()
        self.pause_event = threading.Event()
        self.retry_event = threading.Event()
        self.decision_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.debug_queue: "queue.Queue[str]" = queue.Queue()

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
        self._install_logging()
        self.after(100, self._drain_log_queue)

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

    def _install_logging(self) -> None:
        """Attach a queue-backed handler for cross-thread logging."""

        logger.setLevel(logging.INFO)
        logger.propagate = False

        self._queue_handler = _QueueLogHandler(self.log_queue)
        self._queue_handler.setFormatter(logging.Formatter("%(levelname)s — %(message)s"))

        if self._queue_handler not in logger.handlers:
            logger.addHandler(self._queue_handler)

    def _configure_weights(self) -> None:
        """Apply grid weights to keep the layout responsive."""

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)

        self._status_frame.columnconfigure(0, weight=1)
        self._status_frame.rowconfigure(0, weight=1)

        self._control_frame.columnconfigure(0, weight=1)
        self._control_frame.columnconfigure(1, weight=1)
        self._control_frame.columnconfigure(2, weight=0)
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
            command=self._run_preflight_clicked,
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

        retry_step = ttk.Button(
            parent,
            text="Retry Step",
            style="Sandbox.TButton",
            command=self._on_retry_step,
            state=tk.DISABLED,
        )
        retry_step.grid(row=0, column=2, padx=(6, 0), sticky=tk.EW)
        retry_step.grid_remove()

        self._run_preflight_btn = run_preflight
        self._start_sandbox_btn = start_sandbox
        self._retry_step_btn = retry_step
    # endregion widget builders

    # region public API
    def log_message(self, message: str, tag: str = "info") -> None:
        """Append ``message`` to the status log using the provided tag."""

        self._status_text.configure(state=tk.NORMAL)
        self._status_text.insert(tk.END, f"{message}\n", (tag,))
        self._status_text.see(tk.END)
        self._status_text.configure(state=tk.DISABLED)
 
    def _enable_start_button(self) -> None:
        """Enable the Start Sandbox button on the GUI thread."""

        self._start_sandbox_btn.configure(state=tk.NORMAL)

    def _show_health_warning(self, failures: Sequence[str]) -> None:
        """Display a warning dialog summarising sandbox health issues."""

        if not failures:
            details = "No additional details were provided."
        else:
            bullet_list = "\n".join(f"• {item}" for item in failures)
            details = f"{bullet_list}"

        messagebox.showwarning(
            "Sandbox health issues detected",
            "The sandbox health check reported problems.\n\n"
            "Please address the following before launching:\n\n"
            f"{details}",
        )
    # endregion public API

    # region callbacks
    def _run_preflight_clicked(self) -> None:  # pragma: no cover - GUI callback
        if self.preflight_thread and self.preflight_thread.is_alive():
            return

        self._run_preflight_btn.configure(state=tk.DISABLED)
        self._start_sandbox_btn.configure(state=tk.DISABLED)
        self.preflight_abort.clear()
        self.retry_event.clear()
        self.pause_event.clear()

        self.log_message("Running preflight checks...", "info")

        self.preflight_thread = threading.Thread(
            target=self._execute_preflight,
            name="PreflightThread",
            daemon=True,
        )
        self.preflight_thread.start()

    def _on_start_sandbox(self) -> None:  # pragma: no cover - GUI callback
        self.log_message("Starting sandbox...", "info")

    def _execute_preflight(self) -> None:
        """Run the Phase 5 preflight orchestration on a worker thread."""

        success = False
        health_snapshot: dict | None = None
        health_failures: list[str] = []
        completion_message = "Preflight aborted."
        completion_tag = "warning"

        try:
            logger.info("Starting sandbox preflight checks (Phase 5)...")

            runner = globals().get("run_full_preflight")
            if runner is None:
                raise RuntimeError("Preflight runner is unavailable")

            health_snapshot = runner(
                logger=logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.preflight_abort,
                retry_event=self.retry_event,
                debug_queue=self.debug_queue,
            )

            if self.preflight_abort.is_set():
                completion_message = "Preflight aborted."
                completion_tag = "warning"
            else:
                dependency_mode = resolve_dependency_mode()

                if health_snapshot is None:
                    healthy = False
                    health_failures = [
                        "Sandbox health check did not return a status snapshot."
                    ]
                else:
                    healthy, health_failures = _evaluate_health_snapshot(
                        health_snapshot,
                        dependency_mode=dependency_mode,
                    )

                if healthy:
                    success = True
                    completion_message = (
                        "Preflight completed successfully. Sandbox is healthy."
                    )
                    completion_tag = "success"
                    logger.info("Sandbox health check passed under %s mode.", dependency_mode)
                    self.after(0, self._enable_start_button)
                else:
                    completion_message = (
                        "Preflight completed with sandbox health warnings."
                    )
                    completion_tag = "warning"
                    summary = "; ".join(health_failures) if health_failures else "unknown issues"
                    logger.warning("Sandbox health check reported issues: %s", summary)
                    self.after(0, self._show_health_warning, health_failures)
        except Exception:  # pragma: no cover - surfaced via logging
            logger.exception("Sandbox preflight failed.")
            completion_message = "Preflight failed. Check logs for details."
            completion_tag = "error"
        finally:
            self.log_queue.put((completion_message, completion_tag))
            self.after(0, self._on_preflight_done, success)

    def _on_preflight_done(self, success: bool) -> None:
        """Re-enable controls after the preflight thread finishes."""

        self.preflight_thread = None
        self._run_preflight_btn.configure(state=tk.NORMAL)
        if success:
            self._start_sandbox_btn.configure(state=tk.NORMAL)
        else:
            self._start_sandbox_btn.configure(state=tk.DISABLED)

    def _drain_log_queue(self) -> None:
        """Drain queued log records and append them to the status view."""

        drained = False
        while True:
            try:
                message, tag = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_message(message, tag)
            drained = True

        if self.pause_event.is_set():
            self._retry_step_btn.configure(state=tk.NORMAL)
            self._retry_step_btn.grid()
            try:
                title, message = self.decision_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                should_retry = messagebox.askyesno(title, message)
                if should_retry:
                    self.pause_event.clear()
                    self.log_queue.put(("Continuing…", "info"))
                    self.retry_event.set()
                    self._retry_step_btn.configure(state=tk.DISABLED)
                    self._retry_step_btn.grid_remove()
                else:
                    self.preflight_abort.set()
                    self.log_queue.put(("Preflight aborted by user.", "warning"))
                    self._retry_step_btn.configure(state=tk.DISABLED)
                    self._retry_step_btn.grid_remove()
                    self.after(0, self._on_preflight_done, False)
        else:
            self._retry_step_btn.configure(state=tk.DISABLED)
            self._retry_step_btn.grid_remove()

        delay = 100 if drained else 250
        self.after(delay, self._drain_log_queue)

    def _on_retry_step(self) -> None:  # pragma: no cover - GUI callback
        if not self.pause_event.is_set():
            return

        self.pause_event.clear()
        self.retry_event.set()
        self.log_queue.put(("Continuing…", "info"))
        self._retry_step_btn.configure(state=tk.DISABLED)
        self._retry_step_btn.grid_remove()
    # endregion callbacks


@dataclass(frozen=True)
class _PreflightStep:
    """Container describing a preflight step and its failure metadata."""

    label: str
    failure_title: str
    failure_message: str
    runner_name: str


class _PreflightWorker:
    """Coordinate execution of the sandbox preflight sequence."""

    def __init__(
        self,
        *,
        logger: logging.Logger,
        pause_event: threading.Event,
        decision_queue: "queue.Queue[tuple[str, str]]",
        abort_event: threading.Event,
        retry_event: threading.Event,
        debug_queue: "queue.Queue[str]",
        steps: Sequence[_PreflightStep],
    ) -> None:
        self._logger = logger
        self._pause_event = pause_event
        self._decision_queue = decision_queue
        self._abort_event = abort_event
        self._retry_event = retry_event
        self._debug_queue = debug_queue
        self._steps = steps

    def run(self) -> dict | None:
        """Execute the configured preflight steps in sequence."""

        return self._execute_preflight()

    def _execute_preflight(self) -> dict | None:
        """Sequentially execute steps while respecting abort semantics."""

        for step in self._steps:
            if self._abort_event.is_set():
                self._logger.info("Preflight aborted before %s", step.label)
                break

            try:
                runner = getattr(sys.modules[__name__], step.runner_name)
            except AttributeError as exc:  # pragma: no cover - defensive guard
                self._logger.exception("Runner %s is not available", step.runner_name)
                self._debug_queue.put(str(exc))
                break
            if not callable(runner):  # pragma: no cover - defensive guard
                self._logger.error("Runner %s is not callable", step.runner_name)
                break

            action = lambda func=runner: func(self._logger)
            should_continue = self._run_step(
                step.label,
                step.failure_title,
                step.failure_message,
                action,
            )
            if not should_continue:
                break

            if self._abort_event.is_set():
                self._logger.info("Preflight aborted after %s", step.label)
                break

        if self._abort_event.is_set():
            return None

        health_snapshot: dict | None = None

        def _capture_health() -> None:
            nonlocal health_snapshot

            module = importlib.import_module("sandbox_runner.bootstrap")
            health_probe = getattr(module, "sandbox_health", None)
            if not callable(health_probe):
                raise RuntimeError("sandbox_health() is not available")
            health_snapshot = health_probe()

        should_continue = self._run_step(
            "Evaluating sandbox health",
            "Sandbox health check failed",
            "Capturing the sandbox health snapshot failed.",
            _capture_health,
        )
        if not should_continue:
            return None

        return health_snapshot

    def _run_step(
        self,
        label: str,
        failure_title: str,
        failure_message: str,
        action: Callable[[], None],
    ) -> bool:
        """Execute a single step and handle pause/retry/abort semantics."""

        while True:
            if self._abort_event.is_set():
                self._logger.info("Preflight aborted before %s", label)
                return False

            self._logger.info("Starting %s", label)
            try:
                action()
            except Exception as exc:  # pragma: no cover - logged via GUI
                self._logger.exception("%s failed", label)
                self._debug_queue.put(str(exc))
                self._pause_event.set()
                self._decision_queue.put(
                    (
                        failure_title,
                        f"{failure_message}\n\n{exc}",
                    )
                )
                decision = self._await_decision()
                if decision == "retry":
                    continue
                if decision == "abort":
                    return False

                # ``skip`` – continue to the next step after logging.
                self._logger.warning(
                    "Continuing after failure in %s", label
                )
                return True

            self._logger.info("Finished %s", label)
            return True

    def _await_decision(self) -> str:
        """Wait for the GUI to signal retry, skip, or abort."""

        while True:
            if self._abort_event.is_set():
                return "abort"
            if self._retry_event.is_set():
                self._retry_event.clear()
                self._pause_event.clear()
                return "retry"
            if not self._pause_event.is_set():
                if self._retry_event.is_set():
                    self._retry_event.clear()
                    self._pause_event.clear()
                    return "retry"
                time.sleep(0.05)
                if self._retry_event.is_set():
                    self._retry_event.clear()
                    self._pause_event.clear()
                    return "retry"
                return "skip"
            time.sleep(0.05)


def run_full_preflight(
    *,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[tuple[str, str]]",
    abort_event: threading.Event,
    retry_event: threading.Event,
    debug_queue: "queue.Queue[str]",
) -> None:
    """Run the Phase 5 sandbox preflight orchestration."""

    worker = _PreflightWorker(
        logger=logger,
        pause_event=pause_event,
        decision_queue=decision_queue,
        abort_event=abort_event,
        retry_event=retry_event,
        debug_queue=debug_queue,
        steps=_DEFAULT_STEPS,
    )
    return worker.run()


def _git_sync(logger: logging.Logger) -> None:
    """Ensure the local repository mirrors the remote state."""

    logger.info("Fetching latest repository state…")
    _run_subprocess(logger, ["git", "fetch", "origin"], cwd=REPO_ROOT)

    branch_proc = _run_subprocess(
        logger,
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT,
        timeout=60,
    )
    branch = branch_proc.stdout.strip() or "HEAD"
    target = f"origin/{branch}" if branch != "HEAD" else "origin/HEAD"

    logger.info("Resetting local branch to %s", target)
    _run_subprocess(
        logger,
        ["git", "reset", "--hard", target],
        cwd=REPO_ROOT,
        timeout=120,
    )


def _purge_stale_files(logger: logging.Logger) -> None:
    """Remove temporary files that could affect subsequent runs."""

    from bootstrap_self_coding import purge_stale_files as _purge

    logger.info("Purging stale bootstrap artefacts…")
    _purge()


def _delete_lock_files(logger: logging.Logger) -> None:
    """Remove stale lock files that block dependency installs."""

    sandbox_dir = REPO_ROOT / "sandbox_data"
    if sandbox_dir.exists():
        for pattern in ("*.lock", "*.lck", "*.lock.json"):
            for path in sandbox_dir.glob(pattern):
                _safe_unlink(path, logger)
            for path in sandbox_dir.rglob(pattern):
                _safe_unlink(path, logger)

    hf_root = Path(
        os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface")
    ).expanduser()

    if hf_root.exists():
        logger.info("Cleaning Hugging Face cache locks in %s", hf_root)
        for pattern in ("**/*.lock", "**/*.json.lock"):
            for path in hf_root.glob(pattern):
                _safe_unlink(path, logger)
        _purge_stale_model_caches(hf_root, logger)
    else:
        logger.info("Hugging Face cache directory not found: %s", hf_root)


def _warm_shared_vector_service(logger: logging.Logger) -> None:
    """Download heavy dependencies and warm the shared vector service cache."""

    from neurosales.scripts import setup_heavy_deps
    from vector_service import SharedVectorService

    logger.info("Downloading heavy dependencies (download only)…")
    setup_heavy_deps.main(download_only=True)

    logger.info("Instantiating SharedVectorService to warm caches…")
    service = SharedVectorService()
    warm_payload = {"text": "sandbox warm-up"}
    try:
        vector = service.vectorise("text", warm_payload)
        logger.info("Vector warm-up produced %d dimensions", len(vector))
    except Exception:
        logger.warning(
            "SharedVectorService warm-up vectorisation failed; continuing after instantiation",
            exc_info=True,
        )


def _ensure_env_flags(logger: logging.Logger) -> None:
    """Ensure environment prerequisites are satisfied."""

    from auto_env_setup import ensure_env

    logger.info("Ensuring sandbox environment configuration…")
    ensure_env()

    overrides = {
        "MENACE_LIGHT_IMPORTS": "1",
        "MENACE_SKIP_CREATE": "1",
        "MENACE_SKIP_STRIPE_ROUTER": "1",
    }
    for key, value in overrides.items():
        previous = os.environ.get(key)
        os.environ[key] = value
        if previous != value:
            logger.info("Set %s=%s", key, value)


def _prime_registry(logger: logging.Logger) -> None:
    """Prime the self-coding registry to avoid runtime delays."""

    from prime_registry import main as prime_main

    logger.info("Priming self-coding registry…")
    prime_main()


def _install_dependencies(logger: logging.Logger) -> None:
    """Run required pip commands for the sandbox."""

    logger.info("Installing sandbox dependencies in editable mode…")
    _run_subprocess(
        logger,
        [sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT)],
        cwd=REPO_ROOT,
        timeout=900,
    )

    logger.info("Installing jsonschema dependency…")
    _run_subprocess(
        logger,
        [sys.executable, "-m", "pip", "install", "jsonschema"],
        cwd=REPO_ROOT,
        timeout=600,
    )


def _bootstrap_self_coding(logger: logging.Logger) -> None:
    """Bootstrap the AI counter bot dependencies."""

    from bootstrap_self_coding import bootstrap_self_coding as bootstrap

    logger.info("Bootstrapping self-coding for AICounterBot…")
    bootstrap("AICounterBot")


def _run_subprocess(
    logger: logging.Logger,
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Execute *args* and forward captured output to *logger*."""

    display_cmd = " ".join(args)
    logger.info("Running command: %s", display_cmd)

    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error(
            "Command timed out after %s seconds: %s", timeout, display_cmd, exc_info=True
        )
        _log_subprocess_streams(logger, exc.stdout, exc.stderr)
        raise
    except subprocess.CalledProcessError as exc:
        logger.error(
            "Command failed with exit code %s: %s", exc.returncode, display_cmd, exc_info=True
        )
        _log_subprocess_streams(logger, exc.stdout, exc.stderr)
        raise

    _log_subprocess_streams(logger, completed.stdout, completed.stderr)
    return completed


def _log_subprocess_streams(
    logger: logging.Logger,
    stdout: str | None,
    stderr: str | None,
) -> None:
    """Forward ``stdout`` and ``stderr`` to *logger* when available."""

    if stdout:
        for line in stdout.splitlines():
            logger.info("[stdout] %s", line)
    if stderr:
        for line in stderr.splitlines():
            logger.info("[stderr] %s", line)


def _safe_unlink(path: Path, logger: logging.Logger) -> None:
    """Attempt to remove ``path`` if it exists, logging failures."""

    try:
        if path.exists():
            path.unlink()
            logger.info("Removed stale lock file: %s", path)
    except IsADirectoryError:
        return
    except OSError as exc:
        logger.warning("Failed to remove %s: %s", path, exc)


def _purge_stale_model_caches(cache_root: Path, logger: logging.Logger) -> None:
    """Remove Hugging Face model caches that appear stale."""

    hub_dir = cache_root / "hub"
    if not hub_dir.exists():
        return

    cutoff = _dt.datetime.now(tz=_dt.timezone.utc) - _dt.timedelta(days=30)
    size_threshold = 1 * 1024 * 1024  # 1 MiB

    for model_dir in hub_dir.glob("models--*"):
        if not model_dir.is_dir():
            continue

        try:
            stat = model_dir.stat()
        except OSError as exc:
            logger.warning("Unable to stat %s: %s", model_dir, exc)
            continue

        mtime = _dt.datetime.fromtimestamp(stat.st_mtime, tz=_dt.timezone.utc)
        size = _dir_size(model_dir, limit_bytes=size_threshold)
        is_old = mtime < cutoff
        is_small = size <= size_threshold

        if not (is_old or is_small):
            continue

        try:
            shutil.rmtree(model_dir)
            logger.info("Removed stale model cache: %s", model_dir)
        except OSError as exc:
            logger.warning("Failed to remove model cache %s: %s", model_dir, exc)


def _dir_size(path: Path, *, limit_bytes: int | None = None) -> int:
    """Return size of ``path`` in bytes, short-circuiting at ``limit_bytes``."""

    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                continue
            if limit_bytes is not None and total > limit_bytes:
                return total
    return total


def _evaluate_health_snapshot(
    snapshot: dict,
    *,
    dependency_mode: DependencyMode,
) -> tuple[bool, list[str]]:
    """Evaluate the health snapshot collected after preflight."""

    failures: list[str] = []

    if not snapshot.get("databases_accessible", True):
        db_errors = snapshot.get("database_errors", {})
        if db_errors:
            detail = ", ".join(
                f"{name}: {error}" for name, error in sorted(db_errors.items())
            )
        else:
            detail = "databases inaccessible"
        failures.append(f"databases inaccessible: {detail}")

    dependency_health = snapshot.get("dependency_health", {})
    missing = dependency_health.get("missing", [])
    for dep in missing:
        name = dep.get("name", "unknown dependency")
        optional = dep.get("optional", False)
        if optional and dependency_mode is DependencyMode.MINIMAL:
            continue
        failures.append(f"missing dependency: {name}")

    return (not failures, failures)


_DEFAULT_STEPS: Sequence[_PreflightStep] = (
    _PreflightStep(
        label="Fetching latest Git snapshot",
        failure_title="Git fetch failed",
        failure_message="Updating the repository from origin failed.",
        runner_name="_git_sync",
    ),
    _PreflightStep(
        label="Purging stale state",
        failure_title="State purge failed",
        failure_message="Purging stale artefacts encountered an error.",
        runner_name="_purge_stale_files",
    ),
    _PreflightStep(
        label="Removing lock artefacts",
        failure_title="Lock artefact removal failed",
        failure_message="Removing stale lock files was unsuccessful.",
        runner_name="_delete_lock_files",
    ),
    _PreflightStep(
        label="Warming shared vector service",
        failure_title="Vector service warm-up failed",
        failure_message="Warming shared vector service failed.",
        runner_name="_warm_shared_vector_service",
    ),
    _PreflightStep(
        label="Ensuring environment",
        failure_title="Environment validation failed",
        failure_message="Ensuring environment prerequisites failed.",
        runner_name="_ensure_env_flags",
    ),
    _PreflightStep(
        label="Priming self-coding registry",
        failure_title="Self-coding registry priming failed",
        failure_message="Priming the self-coding registry failed.",
        runner_name="_prime_registry",
    ),
    _PreflightStep(
        label="Running pip commands",
        failure_title="Pip commands failed",
        failure_message="Executing pip commands failed.",
        runner_name="_install_dependencies",
    ),
    _PreflightStep(
        label="Bootstrapping AI counter bot",
        failure_title="AI counter bot bootstrap failed",
        failure_message="Bootstrapping the AI counter bot failed.",
        runner_name="_bootstrap_self_coding",
    ),
)


class _QueueLogHandler(logging.Handler):
    """Simple handler that forwards log records into a queue."""

    def __init__(self, log_queue: "queue.Queue[tuple[str, str]]") -> None:
        super().__init__()
        self._queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - thin wrapper
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return

        level = record.levelno
        if level >= logging.ERROR:
            tag = "error"
        elif level >= logging.WARNING:
            tag = "warning"
        else:
            tag = "info"
        self._queue.put((message, tag))


__all__ = ["SandboxLauncherGUI", "run_full_preflight"]


if __name__ == "__main__":  # pragma: no cover - manual launch helper
    gui = SandboxLauncherGUI()
    gui.mainloop()
