"""GUI for launching the Windows sandbox."""

from __future__ import annotations

import logging
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from collections.abc import Callable, Sequence
from logging.handlers import RotatingFileHandler
from pathlib import Path
from tkinter import font, messagebox, ttk
from typing import Any, TextIO

from dependency_health import DependencyMode, resolve_dependency_mode

REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX_DATA_DIR = REPO_ROOT / "sandbox_data"
VECTOR_SERVICE_DIR = REPO_ROOT / "vector_service"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"


LOGGER = logging.getLogger(__name__)
LOG_QUEUE: "queue.Queue[tuple[str, str]]" = queue.Queue()
PAUSE_EVENT = threading.Event()
DECISION_QUEUE: "queue.Queue[tuple[str, str, dict[str, Any] | None]]" = queue.Queue()
RETRY_EVENT = threading.Event()


_LOCK_FILE_PATTERNS: tuple[tuple[Path, str], ...] = (
    (SANDBOX_DATA_DIR, "*.lock"),
    (SANDBOX_DATA_DIR, "*.lock.*"),
    (VECTOR_SERVICE_DIR, "*.lock"),
)

_STALE_DIRECTORIES: tuple[Path, ...] = (
    VECTOR_SERVICE_DIR / "cache",
    VECTOR_SERVICE_DIR / "model_cache",
    SANDBOX_DATA_DIR / "vector_cache",
)

_STEP_FAILURE_TITLES: dict[str, str] = {
    "_git_sync": "Repository synchronisation failed",
    "_purge_stale_files": "Stale file purge failed",
    "_cleanup_lock_and_model_artifacts": "Lock and model cleanup failed",
    "_install_heavy_dependencies": "Heavy dependency installation failed",
    "_warm_shared_vector_service": "Vector service warmup failed",
    "_ensure_env_flags": "Environment preparation failed",
    "_prime_registry": "Registry priming failed",
    "_install_python_dependencies": "Python dependency installation failed",
    "_bootstrap_self_coding": "Self-coding bootstrap failed",
}

_STEP_FAILURE_MESSAGES: dict[str, str] = {
    "_cleanup_lock_and_model_artifacts": (
        "Removing stale lock files and model caches failed. "
        "Retrying may resolve intermittent filesystem issues."
    ),
    "_install_python_dependencies": (
        "Installing Python dependencies from requirements.txt failed. "
        "Check the output log for the failing package."
    ),
}


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _run_command(
    command: list[str],
    *,
    logger: logging.Logger,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    logger.info("Running command: %s", _format_command(command))
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.stdout:
        logger.info("stdout: %s", result.stdout.strip())
    if result.stderr:
        logger.info("stderr: %s", result.stderr.strip())
    logger.info("Command exited with code %s", result.returncode)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command {_format_command(command)} failed with exit code {result.returncode}"
        )
    return result


def _ensure_within_repo(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:  # pragma: no cover - defensive safety check
        raise RuntimeError(f"Refusing to modify path outside repository: {path}") from exc
    return resolved


def _iter_lock_files() -> list[Path]:
    candidates: list[Path] = []
    for base, pattern in _LOCK_FILE_PATTERNS:
        if not base.exists():
            continue
        candidates.extend(sorted(base.glob(pattern)))
    return candidates


def _remove_file(path: Path, *, logger: logging.Logger) -> None:
    resolved = _ensure_within_repo(path)
    if not resolved.exists():
        return
    if resolved.is_dir():
        raise RuntimeError(f"Expected file but found directory: {resolved}")
    resolved.unlink()
    logger.info("Removed stale file: %s", resolved)


def _remove_directory(path: Path, *, logger: logging.Logger) -> None:
    resolved = _ensure_within_repo(path)
    if not resolved.exists():
        return
    if not resolved.is_dir():
        raise RuntimeError(f"Expected directory but found file: {resolved}")
    shutil.rmtree(resolved)
    logger.info("Removed stale directory: %s", resolved)


def _git_sync(logger: logging.Logger) -> None:
    """Run a lightweight git status check to ensure repository accessibility."""

    _run_command(["git", "status", "--short"], logger=logger, cwd=REPO_ROOT)


def _purge_stale_files(logger: logging.Logger) -> None:
    """Remove stale bootstrap files via :mod:`bootstrap_self_coding`."""

    logger.info("Purging stale bootstrap files")
    from bootstrap_self_coding import purge_stale_files

    purge_stale_files()


def _cleanup_lock_and_model_artifacts(logger: logging.Logger) -> None:
    """Delete leftover lock files and cached model directories."""

    logger.info("Removing stale lock files and model caches")
    for candidate in _iter_lock_files():
        _remove_file(candidate, logger=logger)
    for directory in _STALE_DIRECTORIES:
        _remove_directory(directory, logger=logger)


def _install_heavy_dependencies(logger: logging.Logger) -> None:
    """Install heavy dependencies using the neurosales setup helper."""

    logger.info("Ensuring heavy dependencies are downloaded")
    from neurosales.scripts import setup_heavy_deps

    setup_heavy_deps.main(download_only=True)


def _warm_shared_vector_service(logger: logging.Logger) -> None:
    """Instantiate :class:`vector_service.SharedVectorService` to warm caches."""

    logger.info("Warming shared vector service")
    from vector_service import SharedVectorService

    service = SharedVectorService()
    vectorise = getattr(service, "vectorise", None)
    if callable(vectorise):
        vectorise("text", {"text": "sandbox warmup"})
    else:  # pragma: no cover - fallback when warmup unavailable
        logger.warning("SharedVectorService.vectorise unavailable; skipping warmup")


def _ensure_env_flags(logger: logging.Logger) -> None:
    """Ensure required environment variables and env files exist."""

    logger.info("Setting sandbox feature flags")
    flags = {
        "SANDBOX_ENABLE_WINDOWS_SELF_CODING": "1",
        "SANDBOX_ENABLE_TRANSFORMERS": "1",
        "SANDBOX_ENABLE_OPENAI": "1",
        "SANDBOX_ENABLE_RELEVANCY_RADAR": "1",
    }
    for key, value in flags.items():
        previous = os.environ.get(key)
        os.environ[key] = value
        if previous != value:
            logger.info("Set %s=%s", key, value)
    from auto_env_setup import ensure_env

    ensure_env()


def _prime_registry(logger: logging.Logger) -> None:
    """Prime the application registry for downstream services."""

    logger.info("Priming registry module")
    from prime_registry import main as prime_main

    prime_main()


def _install_python_dependencies(logger: logging.Logger) -> None:
    """Install Python dependencies from :mod:`requirements.txt`."""

    if not REQUIREMENTS_FILE.exists():
        logger.info("Requirements file %s not found; skipping installation", REQUIREMENTS_FILE)
        return
    _run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        logger=logger,
        cwd=REPO_ROOT,
    )


def _bootstrap_self_coding(logger: logging.Logger) -> None:
    """Invoke :func:`bootstrap_self_coding.bootstrap_self_coding`."""

    bot_name = os.getenv("SANDBOX_SELF_CODING_BOT", "InformationSynthesisBot")
    logger.info("Bootstrapping self-coding for %s", bot_name)
    from bootstrap_self_coding import bootstrap_self_coding

    bootstrap_self_coding(bot_name)


_PRE_FLIGHT_STEP_NAMES: tuple[str, ...] = (
    "_git_sync",
    "_purge_stale_files",
    "_cleanup_lock_and_model_artifacts",
    "_install_heavy_dependencies",
    "_warm_shared_vector_service",
    "_ensure_env_flags",
    "_prime_registry",
    "_install_python_dependencies",
    "_bootstrap_self_coding",
)


def _resolve_step(name: str) -> Callable[[logging.Logger], None]:
    func = globals().get(name)
    if func is None or not callable(func):
        raise RuntimeError(f"Preflight step {name} is not callable")
    return func  # type: ignore[return-value]


def _step_failure_details(step: str, exc: Exception) -> tuple[str, str, dict[str, Any]]:
    title = _STEP_FAILURE_TITLES.get(step, f"Step {step} failed")
    default_body = f"Step {step} failed with error: {exc}"
    message = _STEP_FAILURE_MESSAGES.get(step, default_body)
    context = {"step": step, "exception": repr(exc)}
    return title, message, context


def _wait_for_pause_resolution(
    pause_event: threading.Event,
    abort_event: threading.Event,
) -> None:
    while pause_event.is_set() and not abort_event.is_set():
        pause_event.wait(0.2)


def run_full_preflight(
    *,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[tuple[str, str, dict[str, Any] | None]]",
    abort_event: threading.Event,
    debug_queue: "queue.Queue[str]",
    retry_event: threading.Event,
    dependency_mode: DependencyMode | None = None,
) -> dict[str, Any]:
    """Execute the full sandbox preflight workflow."""

    dependency_mode = dependency_mode or resolve_dependency_mode()
    result: dict[str, Any] = {
        "steps": [],
        "aborted": False,
        "healthy": False,
        "failures": [],
    }

    def _step(name: str, func: Callable[[logging.Logger], None]) -> bool:
        while True:
            if abort_event.is_set():
                logger.info("Preflight aborted before step %s", name)
                result["aborted"] = True
                return False
            logger.info("event=preflight status=running step=%s", name)
            try:
                func(logger)
            except Exception as exc:  # pragma: no cover - runtime safety
                logger.exception("Preflight step %s failed", name)
                try:
                    debug_queue.put_nowait(traceback.format_exc())
                except queue.Full:  # pragma: no cover - unbounded queue
                    pass
                title, message, context = _step_failure_details(name, exc)
                context.setdefault("exception_type", type(exc).__name__)
                context.setdefault("step", name)
                decision_queue.put((title, message, context))
                pause_event.set()
                _wait_for_pause_resolution(pause_event, abort_event)
                if abort_event.is_set():
                    result["aborted"] = True
                    result["failed_step"] = name
                    return False
                if retry_event.is_set():
                    logger.info(
                        "event=preflight status=retry action=restart step=%s", name
                    )
                    retry_event.clear()
                    pause_event.clear()
                    continue
                result["failed_step"] = name
                return False
            else:
                retry_event.clear()
                result["steps"].append(name)
                return True

    for step_name in _PRE_FLIGHT_STEP_NAMES:
        step_callable = _resolve_step(step_name)
        if not _step(step_name, step_callable):
            return result

    from sandbox_runner import bootstrap as bootstrap_module

    while True:
        try:
            logger.info("Collecting sandbox health snapshot")
            snapshot = bootstrap_module.sandbox_health()
            result["snapshot"] = snapshot
            healthy, failures = _evaluate_health_snapshot(
                snapshot,
                dependency_mode=dependency_mode,
            )
            result["healthy"] = healthy
            result["failures"] = failures
            if healthy:
                logger.info(
                    "event=preflight status=health_verification result=healthy"
                )
                break
            logger.warning(
                "event=preflight status=health_verification result=unhealthy issues=%s",
                failures,
            )
            summary = (
                "\n".join(failures)
                if failures
                else "Unknown preflight health failures"
            )
        except Exception:
            logger.exception(
                "event=preflight status=health_verification result=error"
            )
            failures = [
                "The sandbox health check raised an exception. See logs for details.",
            ]
            result["snapshot"] = {}
            result["healthy"] = False
            result["failures"] = failures
            summary = "\n".join(failures)

        decision_queue.put(
            (
                "Sandbox health check failed",
                "Preflight health checks reported issues. Review the log for details.",
                {"step": "health_check", "details": summary},
            )
        )
        pause_event.set()
        _wait_for_pause_resolution(pause_event, abort_event)
        if abort_event.is_set():
            result["aborted"] = True
            break
        if retry_event.is_set():
            LOGGER.info(
                "event=preflight status=retry action=restart step=health_check"
            )
            retry_event.clear()
            pause_event.clear()
            continue
        break
    return result


def _evaluate_health_snapshot(
    snapshot: dict[str, Any], *, dependency_mode: DependencyMode
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not snapshot.get("databases_accessible", True):
        errors = snapshot.get("database_errors") or {}
        if isinstance(errors, dict) and errors:
            details = ", ".join(f"{db}: {err}" for db, err in errors.items())
        else:
            details = "unknown reason"
        failures.append(f"Sandbox databases inaccessible ({details})")

    dependency_health = snapshot.get("dependency_health") or {}
    missing = dependency_health.get("missing", []) if isinstance(dependency_health, dict) else []
    for item in missing:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "unknown"))
        optional = bool(item.get("optional", False))
        if optional and dependency_mode in {DependencyMode.RELAXED, DependencyMode.MINIMAL}:
            continue
        failures.append(f"Missing dependency: {name}")

    return (not failures, failures)


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
        self._is_sandbox_running = False
        self._sandbox_thread: threading.Thread | None = None
        self._sandbox_process: subprocess.Popen[str] | None = None
        self._can_launch_sandbox = False
        self._abort_event = threading.Event()

        self._pause_title_var = tk.StringVar(value="")
        self._pause_message_var = tk.StringVar(value="")
        self._paused_context: dict[str, Any] | None = None
        self._pause_visible = False
        self._debug_visible = tk.BooleanVar(value=False)
        self._debug_details: list[str] = []
        self._elapsed_var = tk.StringVar(value="Preflight idle")
        self._elapsed_start: float | None = None
        self._elapsed_job: int | None = None

        self._log_handler = QueueLoggingHandler(LOG_QUEUE)
        self._log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        LOGGER.addHandler(self._log_handler)

        log_file = REPO_ROOT / "menace_gui_logs.txt"
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        self._file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        LOGGER.addHandler(self._file_handler)

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
        self.status_text.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        debug_controls = ttk.Frame(status_frame)
        debug_controls.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Checkbutton(
            debug_controls,
            text="Show Debug Details",
            variable=self._debug_visible,
            command=self._toggle_debug_visibility,
        ).pack(side="left")

        self.debug_frame = ttk.Frame(status_frame)
        self.debug_frame.pack(fill="both", expand=False, padx=8, pady=(0, 8))
        self.debug_text = tk.Text(
            self.debug_frame,
            wrap="word",
            height=8,
            state="disabled",
            bg=self.cget("bg"),
            relief="groove",
        )
        self.debug_text.pack(fill="both", expand=True)
        self.debug_frame.pack_forget()

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

        pause_title_font = font.nametofont("TkDefaultFont").copy()
        pause_title_font.configure(weight="bold")

        self.pause_frame = ttk.Frame(buttons_frame)
        self.pause_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.pause_frame.columnconfigure(0, weight=1)

        self.pause_title_label = ttk.Label(
            self.pause_frame,
            textvariable=self._pause_title_var,
            font=pause_title_font,
        )
        self.pause_title_label.grid(row=0, column=0, columnspan=3, sticky="w")

        self.pause_message_label = ttk.Label(
            self.pause_frame,
            textvariable=self._pause_message_var,
            wraplength=640,
            justify="left",
        )
        self.pause_message_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 8))

        self.retry_button = ttk.Button(
            self.pause_frame,
            text="Retry Step",
            command=self._on_retry_step,
            state="disabled",
        )
        self.retry_button.grid(row=2, column=0, sticky="w", padx=(0, 6))

        self.resume_button = ttk.Button(
            self.pause_frame,
            text="Resume",
            command=self._on_resume_preflight,
        )
        self.resume_button.grid(row=2, column=1, sticky="w", padx=(0, 6))

        self.abort_button = ttk.Button(
            self.pause_frame,
            text="Abort Preflight",
            command=self._on_abort_preflight_clicked,
        )
        self.abort_button.grid(row=2, column=2, sticky="w")

        self.pause_frame.grid_remove()

        self.elapsed_label = ttk.Label(
            buttons_frame,
            textvariable=self._elapsed_var,
            anchor="e",
        )
        self.elapsed_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(12, 0))

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

    def _toggle_debug_visibility(self) -> None:
        if self._debug_visible.get():
            self.debug_frame.pack(fill="both", expand=False, padx=8, pady=(0, 8))
            self._refresh_debug_text()
        else:
            self.debug_frame.pack_forget()

    def _refresh_debug_text(self) -> None:
        self.debug_text.configure(state="normal")
        self.debug_text.delete("1.0", "end")
        if self._debug_details:
            content = "\n\n".join(self._debug_details)
            self.debug_text.insert("1.0", content)
        self.debug_text.configure(state="disabled")

    def _add_debug_detail(self, detail: str) -> None:
        self._debug_details.append(detail)
        if len(self._debug_details) > 100:
            del self._debug_details[:-100]
        if self._debug_visible.get():
            self._refresh_debug_text()

    def _clear_debug_details(self) -> None:
        self._debug_details.clear()
        if self._debug_visible.get():
            self._refresh_debug_text()

    def _show_pause_controls(self) -> None:
        if not self._pause_visible:
            self.pause_frame.grid()
            self._pause_visible = True

    def _hide_pause_controls(self) -> None:
        if self._pause_visible:
            self.pause_frame.grid_remove()
            self._pause_visible = False
        self._pause_title_var.set("")
        self._pause_message_var.set("")
        self._paused_context = None
        self.retry_button.state(["disabled"])

    def _update_pause_prompt(
        self,
        title: str,
        message: str,
        context: dict[str, Any] | None,
    ) -> None:
        self._paused_context = dict(context or {})
        self._pause_title_var.set(title)
        lines: list[str] = []
        if message:
            lines.append(message)
        step = self._paused_context.get("step")
        if step:
            lines.append(f"Step: {step}")
        details = self._paused_context.get("details")
        if details:
            lines.append(str(details))
        exception_type = self._paused_context.get("exception_type")
        if exception_type and not details:
            lines.append(f"Exception type: {exception_type}")
        exception = self._paused_context.get("exception")
        if exception:
            lines.append(str(exception))
        self._pause_message_var.set("\n\n".join(lines))
        if step:
            self.retry_button.state(["!disabled"])
        else:
            self.retry_button.state(["disabled"])
        self._show_pause_controls()

    def _on_resume_preflight(self) -> None:
        if not PAUSE_EVENT.is_set():
            return
        LOGGER.info("event=preflight status=user_action action=continue")
        if self._paused_context:
            LOGGER.debug(
                "event=preflight status=resume context=%s", self._paused_context
            )
        RETRY_EVENT.clear()
        PAUSE_EVENT.clear()
        self._hide_pause_controls()

    def _on_retry_step(self) -> None:
        if not PAUSE_EVENT.is_set():
            return
        LOGGER.info(
            "event=preflight status=user_action action=retry step=%s",
            (self._paused_context or {}).get("step"),
        )
        if self._paused_context:
            LOGGER.debug(
                "event=preflight status=retry context=%s", self._paused_context
            )
        RETRY_EVENT.set()
        PAUSE_EVENT.clear()
        self._hide_pause_controls()

    def _on_abort_preflight_clicked(self) -> None:
        LOGGER.info("event=preflight status=user_action action=abort")
        self._abort_preflight()
        self._hide_pause_controls()

    def _start_elapsed_timer(self) -> None:
        self._elapsed_start = time.time()
        self._update_elapsed_time()

    def _update_elapsed_time(self) -> None:
        if self._elapsed_start is None:
            self._elapsed_var.set("Preflight idle")
            self._elapsed_job = None
            return
        elapsed = int(time.time() - self._elapsed_start)
        self._elapsed_var.set(f"Preflight running: {elapsed} s")
        try:
            self._elapsed_job = self.after(1000, self._update_elapsed_time)
        except tk.TclError:
            self._elapsed_job = None

    def _stop_elapsed_timer(self, message: str = "Preflight idle") -> None:
        if self._elapsed_job is not None:
            try:
                self.after_cancel(self._elapsed_job)
            except tk.TclError:
                pass
            self._elapsed_job = None
        self._elapsed_start = None
        self._elapsed_var.set(message)

    def _on_close(self) -> None:
        LOGGER.info("event=gui status=closing")
        self._abort_preflight()
        self._terminate_sandbox_process()

        preflight_thread = self._preflight_thread
        if preflight_thread and preflight_thread.is_alive():
            LOGGER.info("event=gui status=waiting thread=preflight")
            preflight_thread.join(timeout=5)
            if preflight_thread.is_alive():
                LOGGER.warning("event=gui status=thread_still_running thread=preflight")
            else:
                self._preflight_thread = None

        sandbox_thread = self._sandbox_thread
        if sandbox_thread and sandbox_thread.is_alive():
            LOGGER.info("event=gui status=waiting thread=sandbox")
            sandbox_thread.join(timeout=5)
            if sandbox_thread.is_alive():
                LOGGER.warning("event=gui status=thread_still_running thread=sandbox")
            else:
                self._sandbox_thread = None

        self._stop_elapsed_timer()

        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            finally:
                self._poll_after_id = None

        self._drain_decision_queue()
        self._drain_log_queue()
        self._persist_final_state()

        for handler in (self._log_handler, self._file_handler):
            try:
                handler.flush()
            except Exception:  # pragma: no cover - defensive cleanup
                LOGGER.debug("event=gui status=cleanup action=handler_flush_failed")
            finally:
                try:
                    LOGGER.removeHandler(handler)
                finally:
                    handler.close()

        try:
            self.destroy()
        except tk.TclError:
            LOGGER.debug("event=gui status=cleanup action=destroy_failed")

    def _poll_log_queue(self) -> None:
        try:
            try:
                while True:
                    level, message = LOG_QUEUE.get_nowait()
                    self._insert_message(level, message)
            except queue.Empty:
                pass

            prompts: list[tuple[str, str, dict[str, Any] | None]] = []
            try:
                while True:
                    prompt = DECISION_QUEUE.get_nowait()
                    if len(prompt) == 3:
                        prompts.append(prompt)
                    else:
                        title, text = prompt[:2]
                        prompts.append((title, text, None))
            except queue.Empty:
                pass

            if prompts:
                title, prompt_message, context = prompts[-1]
                self._update_pause_prompt(title, prompt_message, context)
            elif PAUSE_EVENT.is_set():
                self._show_pause_controls()
            else:
                self._hide_pause_controls()
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
        self._can_launch_sandbox = False
        self._abort_event.clear()
        PAUSE_EVENT.clear()
        RETRY_EVENT.clear()
        self._drain_decision_queue()
        self._hide_pause_controls()
        self._clear_debug_details()
        self._start_elapsed_timer()

        self._preflight_thread = threading.Thread(
            target=self._run_preflight,
            daemon=True,
        )
        self._preflight_thread.start()

    def start_sandbox(self) -> None:
        """Callback for the Start Sandbox button."""
        if self._is_sandbox_running:
            LOGGER.info(
                "event=sandbox status=ignored reason=already_running"
            )
            return

        if not self._can_launch_sandbox:
            LOGGER.warning(
                "event=sandbox status=blocked reason=preflight_incomplete"
            )
            return

        LOGGER.info("event=sandbox status=launch_initiated")
        self._is_sandbox_running = True
        self.preflight_button.state(["disabled"])
        self.start_button.state(["disabled"])

        self._sandbox_thread = threading.Thread(
            target=self._launch_sandbox_process,
            daemon=True,
        )
        self._sandbox_thread.start()

    def _launch_sandbox_process(self) -> None:
        """Launch ``start_autonomous_sandbox`` in a background thread."""

        command = [sys.executable, "-m", "start_autonomous_sandbox"]
        stdout_thread: threading.Thread | None = None
        stderr_thread: threading.Thread | None = None
        returncode: int | None = None
        error: BaseException | None = None

        try:
            LOGGER.info(
                "event=sandbox status=starting command=%s",
                _format_command(command),
            )
            self._sandbox_process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            process = self._sandbox_process
            if process.stdout is None or process.stderr is None:
                raise RuntimeError("Sandbox process streams unavailable")

            stdout_thread = threading.Thread(
                target=self._forward_process_stream,
                args=(process.stdout, logging.INFO, "stdout"),
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=self._forward_process_stream,
                args=(process.stderr, logging.ERROR, "stderr"),
                daemon=True,
            )

            stdout_thread.start()
            stderr_thread.start()

            returncode = self._sandbox_process.wait()
            if returncode == 0:
                LOGGER.info("event=sandbox status=exited result=success code=%s", returncode)
            else:
                LOGGER.error("event=sandbox status=exited result=failure code=%s", returncode)
        except Exception as exc:  # pragma: no cover - defensive handling
            error = exc
            LOGGER.exception("event=sandbox status=failed reason=exception")
            if self._sandbox_process and self._sandbox_process.poll() is None:
                try:
                    self._sandbox_process.terminate()
                except Exception:  # pragma: no cover - best-effort cleanup
                    LOGGER.exception("event=sandbox status=terminate_failed")
        finally:
            if stdout_thread is not None:
                stdout_thread.join(timeout=1)
            if stderr_thread is not None:
                stderr_thread.join(timeout=1)

            self._sandbox_process = None
            self._is_sandbox_running = False

            try:
                self.after(0, self._on_sandbox_finished, returncode, error)
            except tk.TclError:
                LOGGER.debug("event=sandbox status=cleanup action=after_failed")
            finally:
                self._sandbox_thread = None

    def _forward_process_stream(
        self,
        stream: TextIO,
        level: int,
        label: str,
    ) -> None:
        """Forward a subprocess pipe to the GUI logger line-by-line."""

        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip()
                if not line:
                    continue
                LOGGER.log(level, "event=sandbox stream=%s message=%s", label, line)
        finally:
            try:
                stream.close()
            except Exception:  # pragma: no cover - defensive cleanup
                LOGGER.debug("event=sandbox status=cleanup action=stream_close_failed")

    def _terminate_sandbox_process(self) -> None:
        """Terminate the sandbox process if it is currently running."""

        process = self._sandbox_process
        if process is None or process.poll() is not None:
            return

        LOGGER.info("event=sandbox status=terminating reason=gui_close")
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                LOGGER.warning(
                    "event=sandbox status=terminate timeout=5 action=kill"
                )
                process.kill()
        except Exception:
            LOGGER.exception("event=sandbox status=terminate_failed")
        finally:
            self._sandbox_process = None
            self._is_sandbox_running = False

    def _on_sandbox_finished(
        self, returncode: int | None, error: BaseException | None
    ) -> None:
        """Handle GUI state updates when the sandbox process completes."""

        self.preflight_button.state(["!disabled"])
        if self._can_launch_sandbox:
            self.start_button.state(["!disabled"])
        else:
            self.start_button.state(["disabled"])

        if error is not None:
            messagebox.showerror(
                "Sandbox launch failed",
                "An unexpected error occurred while launching the sandbox.\n"
                "Check the application log for details.",
            )
        elif returncode and returncode != 0:
            messagebox.showerror(
                "Sandbox exited with errors",
                "The sandbox process exited with code " f"{returncode}.",
            )
        elif returncode == 0:
            messagebox.showinfo(
                "Sandbox exited",
                "The sandbox process completed successfully.",
            )

    def _run_preflight(self) -> None:
        """Execute the preflight process in a background thread."""
        preflight_success = False
        health_ok = False
        health_failures: list[str] = []
        try:
            LOGGER.info("event=preflight status=started")
            dependency_mode = resolve_dependency_mode()
            debug_queue: "queue.Queue[str]" = queue.Queue()
            result = run_full_preflight(
                logger=LOGGER,
                pause_event=PAUSE_EVENT,
                decision_queue=DECISION_QUEUE,
                abort_event=self._abort_event,
                debug_queue=debug_queue,
                retry_event=RETRY_EVENT,
                dependency_mode=dependency_mode,
            )
            aborted = bool(result.get("aborted"))
            failures = result.get("failures") or []
            failed_step = result.get("failed_step")

            if result.get("healthy") and not aborted and not failed_step:
                preflight_success = True
                LOGGER.info("event=preflight status=completed result=success")
            else:
                if aborted:
                    LOGGER.info("event=preflight status=aborted")
                if failed_step:
                    LOGGER.warning(
                        "event=preflight status=failed step=%s", failed_step
                    )
            for failure in failures:
                LOGGER.error("event=preflight failure=%s", failure)

            if preflight_success:
                while True:
                    try:
                        LOGGER.info(
                            "event=preflight status=health_verification action=start"
                        )
                        from sandbox_runner import bootstrap as bootstrap_module

                        health_snapshot = bootstrap_module.sandbox_health()
                        LOGGER.info(
                            "event=preflight status=health_snapshot details=%s",
                            health_snapshot,
                        )
                        health_ok, health_failures = _evaluate_health_snapshot(
                            health_snapshot,
                            dependency_mode=dependency_mode,
                        )
                        if health_ok:
                            LOGGER.info(
                                "event=preflight status=health_verification result=healthy"
                            )
                        else:
                            LOGGER.warning(
                                "event=preflight status=health_verification result=unhealthy issues=%s",
                                health_failures,
                            )
                    except Exception:
                        LOGGER.exception(
                            "event=preflight status=health_verification result=error"
                        )
                        health_failures = [
                            "The sandbox health check raised an exception. See logs for details.",
                        ]
                        health_ok = False

                    if health_ok:
                        break

                    decision_queue.put(
                        (
                            "Sandbox health check failed",
                            "Preflight health checks reported issues. Review the log for details.",
                            {"step": "health_check", "details": "\n".join(health_failures)},
                        )
                    )
                    pause_event.set()
                    _wait_for_pause_resolution(pause_event, abort_event)
                    if abort_event.is_set():
                        result["aborted"] = True
                        break
                    if retry_event.is_set():
                        LOGGER.info(
                            "event=preflight status=retry action=restart step=health_check"
                        )
                        retry_event.clear()
                        pause_event.clear()
                        continue
                    break

            while not debug_queue.empty():
                detail = debug_queue.get_nowait()
                LOGGER.debug("event=preflight detail=%s", detail)
                try:
                    self.after(0, self._add_debug_detail, detail)
                except tk.TclError:
                    LOGGER.debug(
                        "event=preflight status=cleanup action=debug_queue_dropped"
                    )
        except Exception:
            LOGGER.exception("event=preflight status=failed")
        finally:
            try:
                self.after(
                    0,
                    self._on_preflight_finished,
                    preflight_success,
                    health_ok,
                    tuple(health_failures),
                )
            except tk.TclError:
                # GUI was likely closed before the callback could be scheduled.
                LOGGER.debug("event=preflight status=cleanup action=after_failed")

    def _abort_preflight(self) -> None:
        self._abort_event.set()
        PAUSE_EVENT.clear()
        RETRY_EVENT.clear()

    def _drain_decision_queue(self) -> None:
        while True:
            try:
                DECISION_QUEUE.get_nowait()
            except queue.Empty:
                break

    def _drain_log_queue(self) -> None:
        while True:
            try:
                LOG_QUEUE.get_nowait()
            except queue.Empty:
                break

    def _persist_final_state(self) -> None:
        """Flush state that should survive the GUI shutdown."""

        try:
            self._file_handler.flush()
        except Exception:  # pragma: no cover - defensive cleanup
            LOGGER.debug("event=gui status=cleanup action=file_handler_flush_failed")

    def _on_preflight_finished(
        self,
        preflight_success: bool,
        health_ok: bool,
        health_failures: Sequence[str] | None = None,
    ) -> None:
        """Handle GUI state updates when the preflight thread completes."""
        self._is_preflight_running = False
        self._preflight_thread = None
        self.preflight_button.state(["!disabled"])
        if preflight_success and health_ok:
            self._stop_elapsed_timer("Preflight completed")
        elif self._abort_event.is_set():
            self._stop_elapsed_timer("Preflight aborted")
        else:
            self._stop_elapsed_timer("Preflight finished")
        self._hide_pause_controls()
        RETRY_EVENT.clear()
        if preflight_success and health_ok:
            self._can_launch_sandbox = True
            self.start_button.state(["!disabled"])
        else:
            self._can_launch_sandbox = False
            self.start_button.state(["disabled"])
            if preflight_success and not health_ok:
                issues = "\n".join(health_failures or ()) or (
                    "Sandbox health checks reported issues. "
                    "See the application log for additional details."
                )
                LOGGER.warning(
                    "event=preflight status=health_warning issues=%s",
                    issues,
                )
                messagebox.showwarning(
                    "Sandbox health check failed",
                    "Sandbox health checks reported issues:\n\n" + issues,
                )


def main() -> None:
    """Run the Windows sandbox launcher GUI."""

    app = SandboxLauncherGUI()
    app.mainloop()


if __name__ == "__main__":
    main()


__all__ = [
    "SandboxLauncherGUI",
    "run_full_preflight",
    "_evaluate_health_snapshot",
    "main",
]
