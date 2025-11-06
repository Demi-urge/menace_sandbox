"""GUI helpers and orchestration utilities for the sandbox preflight."""

from __future__ import annotations

import logging
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

try:  # pragma: no cover - optional runtime dependency
    from dependency_health import DependencyMode
except Exception:  # pragma: no cover - fallback if package missing
    DependencyMode = type("DependencyMode", (), {"STRICT": object(), "MINIMAL": object()})  # type: ignore

try:  # pragma: no cover - optional runtime dependency
    from sandbox_runner import bootstrap as sandbox_bootstrap
except Exception:  # pragma: no cover - fallback if module missing
    sandbox_bootstrap = None  # type: ignore

try:  # pragma: no cover - optional runtime dependency
    import auto_env_setup
except Exception:  # pragma: no cover - fallback if module missing
    auto_env_setup = None  # type: ignore

try:  # pragma: no cover - optional runtime dependency
    import bootstrap_self_coding
except Exception:  # pragma: no cover - fallback if module missing
    bootstrap_self_coding = None  # type: ignore

try:  # pragma: no cover - optional runtime dependency
    import prime_registry
except Exception:  # pragma: no cover - fallback if module missing
    prime_registry = None  # type: ignore

try:  # pragma: no cover - optional runtime dependency
    from vector_service import SharedVectorService
except Exception:  # pragma: no cover - fallback if module missing
    SharedVectorService = None  # type: ignore


DecisionPayload = tuple[str, str, dict[str, Any] | None]


PAUSE_EVENT = threading.Event()
DECISION_QUEUE: "queue.Queue[DecisionPayload]" = queue.Queue()


class TextWidgetHandler(logging.Handler):
    """Logging handler that forwards records to a queue for the GUI log display."""

    def __init__(self, log_queue: "queue.Queue[Tuple[str, str]]", *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - standard logging docstring not required
        """Format and enqueue the record for the GUI thread."""

        try:
            message = self.format(record)
            level = record.levelname.lower()
            self._log_queue.put_nowait((level, message))
        except Exception:  # pragma: no cover - defensive fallback mirrors logging.Handler
            self.handleError(record)


class SandboxLauncherGUI(tk.Tk):
    """Simple GUI used for launching sandbox workflows."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.log_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.decision_queue = DECISION_QUEUE
        self.debug_queue: "queue.Queue[str]" = queue.Queue()
        self.pause_event = PAUSE_EVENT
        self.abort_event = threading.Event()
        self._preflight_thread: threading.Thread | None = None
        self._preflight_result: Dict[str, Any] | None = None
        self._preflight_completion_handled = False
        self._pending_decision = False
        self._last_decision: DecisionPayload | None = None
        self._configure_root()
        self._build_widgets()
        self._configure_log_tags()
        self.log_handler = TextWidgetHandler(self.log_queue)
        self.log_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
        )
        self._schedule_log_drain()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_root(self) -> None:
        """Configure the main window properties and default styles."""
        self.title("Windows Sandbox Launcher")
        self.geometry("640x480")

        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

    def _build_widgets(self) -> None:
        """Build the notebook, log display, and control buttons."""
        container = ttk.Frame(self, padding=10)
        container.grid(row=0, column=0, sticky="nsew")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(container)
        notebook.grid(row=0, column=0, sticky="nsew")

        status_frame = ttk.Frame(notebook, padding=(10, 10))
        status_frame.pack_propagate(False)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            status_frame,
            state="disabled",
            wrap="word",
            height=15,
            width=60,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        notebook.add(status_frame, text="Status")

        controls = ttk.Frame(container)
        controls.grid(row=1, column=0, pady=(10, 0), sticky="ew")
        controls.columnconfigure((0, 1), weight=1)

        self.preflight_button = ttk.Button(controls, text="Run Preflight")
        self.preflight_button.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.preflight_button.configure(command=self._start_preflight)

        self.start_button = ttk.Button(controls, text="Start Sandbox", state="disabled")
        self.start_button.grid(row=0, column=1, padx=(5, 0), sticky="ew")

    def _configure_log_tags(self) -> None:
        """Configure styled tags used for log rendering."""

        self.log_text.tag_configure("info", foreground="white")
        self.log_text.tag_configure("warning", foreground="yellow")
        self.log_text.tag_configure("error", foreground="red", font=("TkDefaultFont", 9, "bold"))

    def _schedule_log_drain(self) -> None:
        """Schedule periodic draining of the log queue on the Tkinter thread."""

        self._drain_log_queue()

    def _drain_log_queue(self) -> None:
        """Process queued log messages and append them to the text widget."""

        while True:
            try:
                level, message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_log_message(level, message)

        self._process_decisions()
        self.after(100, self._drain_log_queue)

    def _append_log_message(self, level: str, message: str) -> None:
        """Append a formatted log message to the text widget with auto-scroll."""

        tag = level if level in {"info", "warning", "error"} else "info"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n", (tag,))
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _drain_decision_queue(self) -> None:
        while True:
            try:
                self.decision_queue.get_nowait()
            except queue.Empty:
                break

    def _process_decisions(self) -> None:
        if not self.pause_event.is_set():
            self._pending_decision = False
            self._last_decision = None
            return

        while True:
            try:
                payload = self.decision_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._last_decision = payload

        if self._pending_decision:
            return

        payload = self._last_decision
        if payload is None:
            return

        title, message, context = payload
        extra = None
        if context:
            extra = context.get("exception") or context.get("details")
            if extra and extra not in message:
                message = f"{message}\n\nDetails: {extra}"

        self._pending_decision = True
        try:
            should_continue = messagebox.askyesno(title, message, parent=self)
        finally:
            self._pending_decision = False

        self._last_decision = None

        if should_continue:
            self.pause_event.clear()
            return

        self.abort_event.set()
        self.pause_event.clear()

        thread = self._preflight_thread
        if thread and thread.is_alive():
            thread.join(timeout=0.5)
            if thread.is_alive():
                return
        self._on_preflight_complete(self._preflight_result, None)

    def _start_preflight(self) -> None:
        """Launch the preflight workflow on a background thread."""

        if self._preflight_thread and self._preflight_thread.is_alive():
            return

        self.abort_event.clear()
        self.pause_event.clear()
        self._pending_decision = False
        self._last_decision = None
        self._drain_decision_queue()
        self._preflight_result = None
        self._preflight_completion_handled = False

        self.preflight_button.configure(state="disabled")
        if self.start_button.winfo_exists():
            self.start_button.configure(state="disabled")

        self._preflight_thread = threading.Thread(
            target=self._run_preflight_worker,
            name="sandbox-preflight",
            daemon=True,
        )
        self._preflight_thread.start()

    def _run_preflight_worker(self) -> None:
        """Execute the preflight controller and report progress to the GUI."""

        logger = logging.getLogger("sandbox.preflight")
        logger.setLevel(logging.INFO)
        handler = self.log_handler
        if handler not in logger.handlers:
            logger.addHandler(handler)
        logger.propagate = False

        result: Dict[str, Any] | None = None
        error: BaseException | None = None

        try:
            logger.info("Starting sandbox preflight sequence")
            result = run_full_preflight(
                logger=logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.abort_event,
                debug_queue=self.debug_queue,
            )

            if result.get("aborted"):
                logger.warning("Preflight aborted by operator")
            elif result.get("healthy"):
                logger.info("Preflight completed successfully")
            else:
                failures = result.get("failures", [])
                if failures:
                    logger.warning("Preflight completed with outstanding issues: %s", failures)
                else:
                    logger.info("Preflight completed")
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Preflight sequence encountered an unexpected error")
            error = exc
        finally:
            self._preflight_result = result
            if handler in logger.handlers:
                logger.removeHandler(handler)
            logger.propagate = True
            self.after(0, lambda: self._on_preflight_complete(result, error))

    def _on_preflight_complete(
        self,
        result: Dict[str, Any] | None,
        error: BaseException | None,
    ) -> None:
        """Re-enable controls after the preflight run finishes."""

        if self._preflight_completion_handled:
            return

        self._preflight_completion_handled = True

        if self.preflight_button.winfo_exists():
            self.preflight_button.configure(state="normal")
        self._preflight_thread = None

        if not self.winfo_exists():
            return

        if error is not None:
            if self.start_button.winfo_exists():
                self.start_button.configure(state="disabled")
            messagebox.showerror(
                "Preflight Error",
                f"Preflight encountered an unexpected error:\n\n{error}",
                parent=self,
            )
            return

        result = result or {}
        aborted = bool(result.get("aborted"))
        healthy = bool(result.get("healthy"))
        failures = result.get("failures") or []

        if healthy and not aborted:
            if self.start_button.winfo_exists():
                self.start_button.configure(state="normal")
            messagebox.showinfo(
                "Preflight Complete",
                "Sandbox is healthy and ready to start.",
                parent=self,
            )
            return

        if self.start_button.winfo_exists():
            self.start_button.configure(state="disabled")

        if aborted:
            messagebox.showinfo(
                "Preflight Aborted",
                "Preflight was aborted. You can rerun the checks when ready.",
                parent=self,
            )
            return

        failure_message = "\n".join(str(item) for item in failures) or "Unknown issues detected."
        messagebox.showwarning(
            "Preflight Issues Detected",
            f"Sandbox health checks reported issues:\n\n{failure_message}\n\nPlease resolve and rerun preflight.",
            parent=self,
        )

    def _on_close(self) -> None:
        """Request shutdown of the worker thread before closing the GUI."""

        if self._preflight_thread and self._preflight_thread.is_alive():
            self.abort_event.set()
            self.pause_event.clear()
        self.destroy()

    def destroy(self) -> None:  # type: ignore[override]
        """Ensure worker threads are stopped before tearing down the window."""

        thread = getattr(self, "_preflight_thread", None)
        if thread and thread.is_alive():
            self.abort_event.set()
            self.pause_event.clear()
            thread.join(timeout=1)
        super().destroy()


_STEP_DEFINITIONS: list[tuple[str, str, str]] = [
    ("_git_sync", "Synchronising sandbox repository", "Repository sync failed"),
    ("_purge_stale_files", "Purging stale artefacts", "Stale artefact purge failed"),
    (
        "_cleanup_lock_and_model_artifacts",
        "Removing stale lock files and model caches",
        "Lock and model cleanup failed",
    ),
    (
        "_install_heavy_dependencies",
        "Downloading heavy dependencies",
        "Heavy dependency installation failed",
    ),
    (
        "_warm_shared_vector_service",
        "Priming shared vector service",
        "Vector service warmup failed",
    ),
    (
        "_ensure_env_flags",
        "Ensuring environment flags",
        "Environment configuration failed",
    ),
    ("_prime_registry", "Priming registry cache", "Registry priming failed"),
    (
        "_install_python_dependencies",
        "Installing sandbox Python dependencies",
        "Python dependency installation failed",
    ),
    (
        "_bootstrap_self_coding",
        "Bootstrapping self-coding subsystem",
        "Self-coding bootstrap failed",
    ),
]


def run_full_preflight(
    *,
    logger: logging.Logger,
    pause_event: threading.Event | None = None,
    decision_queue: "queue.Queue[DecisionPayload]" | None = None,
    abort_event: threading.Event | None = None,
    debug_queue: "queue.Queue[str]" | None = None,
    dependency_mode: "DependencyMode" = DependencyMode.STRICT,
) -> Dict[str, Any]:
    """Execute the full sandbox preflight sequence."""

    pause_event = pause_event or PAUSE_EVENT
    decision_queue = decision_queue or DECISION_QUEUE
    abort_event = abort_event or threading.Event()

    pause_event.clear()

    for func_name, action, failure_title in _STEP_DEFINITIONS:
        if abort_event.is_set():
            logger.info("Abort requested before '%s'; stopping", action)
            return {"aborted": True, "healthy": False, "failures": ["aborted"]}

        logger.info("Starting step: %s", action)
        func: Callable[[logging.Logger], None] = globals()[func_name]

        try:
            func(logger)
        except Exception as exc:  # pragma: no cover - exercised via unit tests
            logger.exception("Step '%s' failed", action)
            if debug_queue is not None:
                debug_queue.put(str(exc))
            payload: DecisionPayload = (
                failure_title,
                f"{action}\n\n{exc}",
                {
                    "step": func_name,
                    "action": action,
                    "exception": repr(exc),
                },
            )
            pause_event.set()
            decision_queue.put(payload)
            return {"aborted": False, "healthy": False, "failures": [failure_title]}
        else:
            pause_event.clear()
            logger.info("Completed step: %s", action)

    if abort_event.is_set():
        logger.info("Abort requested before health evaluation; stopping")
        return {"aborted": True, "healthy": False, "failures": ["aborted"]}

    logger.info("Gathering sandbox health snapshot")
    snapshot: Dict[str, Any] = {}
    if sandbox_bootstrap is not None:
        snapshot = sandbox_bootstrap.sandbox_health()
    healthy, failures = _evaluate_health_snapshot(snapshot, dependency_mode=dependency_mode)
    if healthy:
        logger.info("Sandbox health check passed")
    else:
        logger.warning("Sandbox health check reported issues: %s", failures)

    pause_event.clear()
    return {"healthy": healthy, "failures": failures, "snapshot": snapshot}


def _evaluate_health_snapshot(
    snapshot: Dict[str, Any],
    *,
    dependency_mode: "DependencyMode",
) -> Tuple[bool, list[str]]:
    """Interpret the sandbox health snapshot produced by the bootstrap module."""

    failures: list[str] = []

    if not snapshot.get("databases_accessible", True):
        errors = snapshot.get("database_errors", {})
        if errors:
            rendered = ", ".join(f"{name}: {error}" for name, error in errors.items())
        else:
            rendered = "unspecified errors"
        failures.append(f"databases inaccessible: {rendered}")

    dependency_section = snapshot.get("dependency_health", {})
    missing: Iterable[Dict[str, Any]] = dependency_section.get("missing", [])  # type: ignore[assignment]

    for item in missing:
        name = item.get("name", "unknown dependency")
        optional = bool(item.get("optional"))
        if optional and dependency_mode == DependencyMode.MINIMAL:
            continue
        descriptor = "optional" if optional else "required"
        failures.append(f"{descriptor} dependency missing: {name}")

    return (not failures, failures)


def _git_sync(logger: logging.Logger) -> None:
    """Ensure the sandbox repository is reachable."""

    repo_path = pathlib.Path(os.environ.get("SANDBOX_REPO_PATH", pathlib.Path.cwd()))
    git_dir = repo_path / ".git"
    if not git_dir.exists():
        logger.info("Skipping Git sync; repository not found at %s", repo_path)
        return

    logger.info("Fetching latest changes from origin")
    _run_subprocess(logger, ["git", "fetch", "origin"], cwd=repo_path)
    logger.info("Resetting repository to origin/main")
    _run_subprocess(logger, ["git", "reset", "--hard", "origin/main"], cwd=repo_path)


def _purge_stale_files(logger: logging.Logger) -> None:
    """Remove stale bootstrap artefacts if the helper is available."""

    if bootstrap_self_coding is None:
        logger.info("Bootstrap helpers unavailable; skipping stale file purge")
        return
    bootstrap_self_coding.purge_stale_files()


def _cleanup_lock_and_model_artifacts(logger: logging.Logger) -> None:
    """Delete stale lock files and incomplete model cache directories."""

    sandbox_removed = _delete_matching_files(
        logger,
        _resolve_directory(os.environ.get("SANDBOX_DATA_DIR"), default="sandbox_data"),
        "*.lock",
    )
    logger.info("Removed %d sandbox lock files", sandbox_removed)

    hf_transformers_dir = pathlib.Path.home() / ".cache" / "huggingface" / "transformers"
    hf_removed = _delete_matching_files(logger, hf_transformers_dir, "*.lock")
    logger.info("Removed %d Hugging Face lock files", hf_removed)

    stale_removed = _delete_stale_model_directories(logger)
    logger.info("Removed %d stale model directories", stale_removed)


def _warm_shared_vector_service(logger: logging.Logger) -> None:
    """Warm the shared vector service if available."""

    if SharedVectorService is None:
        logger.info("SharedVectorService unavailable; skipping warmup")
        return

    service = SharedVectorService()
    logger.info("Shared vector service initialised: %s", type(service).__name__)


def _ensure_env_flags(logger: logging.Logger) -> None:
    """Ensure required environment flags are configured."""

    if auto_env_setup is None:
        logger.info("auto_env_setup unavailable; skipping environment validation")
        return
    auto_env_setup.ensure_env()
    os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
    os.environ.setdefault("SANDBOX_RECURSIVE_ISOLATED", "1")
    os.environ.setdefault("SELF_TEST_RECURSIVE_ISOLATED", "1")
    os.environ.setdefault("MENACE_ENVIRONMENT", "sandbox")


def _prime_registry(logger: logging.Logger) -> None:
    """Prime the registry cache if the helper module is present."""

    if prime_registry is None:
        logger.info("prime_registry helper unavailable; skipping priming step")
        return
    prime_registry.main()


def _install_python_dependencies(logger: logging.Logger) -> None:
    """Install sandbox Python dependencies using pip."""

    project_root = pathlib.Path(__file__).resolve().parent.parent
    logger.info("Installing project in editable mode via pip")
    _run_subprocess(
        logger,
        [sys.executable, "-m", "pip", "install", "-e", str(project_root)],
    )

    logger.info("Ensuring jsonschema is available")
    _run_subprocess(logger, [sys.executable, "-m", "pip", "install", "jsonschema"])


def _install_heavy_dependencies(logger: logging.Logger) -> None:
    """Install optional heavyweight dependencies if accessible."""

    try:
        from neurosales.scripts import setup_heavy_deps
    except Exception:  # pragma: no cover - fallback if package missing
        logger.info("neurosales setup helpers unavailable; skipping dependency installation")
        return

    setup_heavy_deps.main(download_only=True)


def _bootstrap_self_coding(logger: logging.Logger) -> None:
    """Bootstrap the self-coding subsystem if the helper exists."""

    if bootstrap_self_coding is None:
        logger.info("bootstrap_self_coding module unavailable; skipping bootstrap")
        return
    bootstrap_self_coding.bootstrap_self_coding("AICounterBot")


def _resolve_directory(path: str | None, *, default: str) -> pathlib.Path:
    """Return a resolved directory path, falling back to *default* when needed."""

    candidate = pathlib.Path(path) if path else pathlib.Path(default)
    return candidate.expanduser().resolve()


def _delete_matching_files(logger: logging.Logger, base_dir: pathlib.Path, pattern: str) -> int:
    """Remove files matching *pattern* inside *base_dir*, returning the count removed."""

    if not base_dir.exists():
        logger.info("Directory %s does not exist; skipping file pattern %s", base_dir, pattern)
        return 0

    removed = 0
    for candidate in base_dir.glob(pattern):
        if not candidate.is_file():
            continue
        try:
            candidate.unlink()
        except FileNotFoundError:  # pragma: no cover - race condition guard
            continue
        except OSError as exc:
            logger.warning("Failed to remove file %s: %s", candidate, exc)
        else:
            removed += 1
    return removed


def _delete_stale_model_directories(logger: logging.Logger) -> int:
    """Remove incomplete model cache directories from known locations."""

    suffixes = (".tmp", ".partial", ".incomplete")
    removed = 0

    candidates = [
        _resolve_directory(os.environ.get("SANDBOX_DATA_DIR"), default="sandbox_data") / "models",
        pathlib.Path.home() / ".cache" / "huggingface" / "hub",
        pathlib.Path.home() / ".cache" / "huggingface" / "transformers",
    ]

    for base_dir in candidates:
        if not base_dir.exists():
            continue
        for path in base_dir.glob("*"):
            if not path.is_dir():
                continue
            if not _is_stale_model_directory(path, suffixes=suffixes):
                continue
            try:
                shutil.rmtree(path)
            except FileNotFoundError:  # pragma: no cover - race condition guard
                continue
            except OSError as exc:
                logger.warning("Failed to remove model directory %s: %s", path, exc)
            else:
                removed += 1
    return removed


def _is_stale_model_directory(path: pathlib.Path, *, suffixes: Sequence[str]) -> bool:
    """Return ``True`` when *path* appears to be a stale model cache directory."""

    name = path.name.lower()
    if any(name.endswith(suffix) for suffix in suffixes):
        return True
    marker_files = {"download.tmp", "partial", "incomplete"}
    return any((path / marker).exists() for marker in marker_files)


def _run_subprocess(
    logger: logging.Logger,
    args: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run *args* capturing output for logging and raise on failure."""

    try:
        completed = subprocess.run(
            list(args),
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Executable not found for command: {args[0]}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        message = stderr or stdout or str(exc)
        raise RuntimeError(f"Command {' '.join(args)} failed: {message}") from exc

    _log_command_output(logger, args, completed.stdout, stream="stdout")
    _log_command_output(logger, args, completed.stderr, stream="stderr")
    return completed


def _log_command_output(
    logger: logging.Logger,
    args: Sequence[str],
    output: str,
    *,
    stream: str,
) -> None:
    """Emit command output to *logger* line-by-line for GUI consumption."""

    if not output:
        return
    command = " ".join(args)
    for line in output.strip().splitlines():
        logger.info("[%s %s] %s", stream, command, line)


__all__ = [
    "SandboxLauncherGUI",
    "TextWidgetHandler",
    "run_full_preflight",
    "_evaluate_health_snapshot",
]
