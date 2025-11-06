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
import traceback
import tkinter as tk
from collections.abc import Callable, Sequence
from pathlib import Path
from tkinter import font, messagebox, ttk
from typing import Any

from dependency_health import DependencyMode, resolve_dependency_mode

REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX_DATA_DIR = REPO_ROOT / "sandbox_data"
VECTOR_SERVICE_DIR = REPO_ROOT / "vector_service"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"


LOGGER = logging.getLogger(__name__)
LOG_QUEUE: "queue.Queue[tuple[str, str]]" = queue.Queue()
PAUSE_EVENT = threading.Event()
DECISION_QUEUE: "queue.Queue[tuple[str, str, dict[str, Any] | None]]" = queue.Queue()


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
            decision_queue.put((title, message, context))
            pause_event.set()
            _wait_for_pause_resolution(pause_event, abort_event)
            if abort_event.is_set():
                result["aborted"] = True
            result["failed_step"] = name
            return False
        else:
            result["steps"].append(name)
            return True

    for step_name in _PRE_FLIGHT_STEP_NAMES:
        step_callable = _resolve_step(step_name)
        if not _step(step_name, step_callable):
            return result

    from sandbox_runner import bootstrap as bootstrap_module

    logger.info("Collecting sandbox health snapshot")
    snapshot = bootstrap_module.sandbox_health()
    result["snapshot"] = snapshot
    healthy, failures = _evaluate_health_snapshot(snapshot, dependency_mode=dependency_mode)
    result["healthy"] = healthy
    result["failures"] = failures
    if not healthy:
        summary = "\n".join(failures) if failures else "Unknown preflight health failures"
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
        self._abort_event = threading.Event()

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
        self._abort_preflight()
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
            try:
                while True:
                    level, message = LOG_QUEUE.get_nowait()
                    self._insert_message(level, message)
            except queue.Empty:
                pass

            if PAUSE_EVENT.is_set():
                prompts: list[tuple[str, str, dict[str, Any] | None]] = []
                try:
                    while True:
                        prompts.append(DECISION_QUEUE.get_nowait())
                except queue.Empty:
                    pass

                for item in prompts:
                    if len(item) == 3:
                        title, message, context = item
                    else:
                        title, message = item[:2]
                        context = None
                    should_continue = messagebox.askyesno(
                        title=title,
                        message=message,
                    )
                    if should_continue:
                        LOGGER.info(
                            "event=preflight status=user_action action=continue"
                        )
                        if context:
                            LOGGER.debug(
                                "event=preflight status=resume context=%s", context
                            )
                        PAUSE_EVENT.clear()
                    else:
                        LOGGER.info(
                            "event=preflight status=user_action action=abort"
                        )
                        self._abort_preflight()
                        break
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
        self._abort_event.clear()
        PAUSE_EVENT.clear()
        self._drain_decision_queue()

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

            while not debug_queue.empty():
                detail = debug_queue.get_nowait()
                LOGGER.debug("event=preflight detail=%s", detail)
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

    def _drain_decision_queue(self) -> None:
        while True:
            try:
                DECISION_QUEUE.get_nowait()
            except queue.Empty:
                break

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
            self.start_button.state(["!disabled"])
        else:
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


__all__ = [
    "SandboxLauncherGUI",
    "run_full_preflight",
    "_evaluate_health_snapshot",
]
