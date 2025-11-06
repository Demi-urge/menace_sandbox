"""GUI helpers and orchestration utilities for the sandbox preflight."""

from __future__ import annotations

import logging
import os
import pathlib
import queue
import subprocess
import threading
import time
from typing import Any, Callable, Dict, Iterable, Tuple

import tkinter as tk
from tkinter import ttk

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
        self.decision_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.debug_queue: "queue.Queue[str]" = queue.Queue()
        self.pause_event = threading.Event()
        self.abort_event = threading.Event()
        self.retry_event = threading.Event()
        self._preflight_thread: threading.Thread | None = None
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

        self.after(100, self._drain_log_queue)

    def _append_log_message(self, level: str, message: str) -> None:
        """Append a formatted log message to the text widget with auto-scroll."""

        tag = level if level in {"info", "warning", "error"} else "info"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n", (tag,))
        self.log_text.configure(state="disabled")
        self.log_text.see("end")

    def _start_preflight(self) -> None:
        """Launch the preflight workflow on a background thread."""

        if self._preflight_thread and self._preflight_thread.is_alive():
            return

        self.abort_event.clear()
        self.retry_event.clear()
        self.pause_event.clear()

        self.preflight_button.configure(state="disabled")

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

        try:
            logger.info("Starting sandbox preflight sequence")
            result = run_full_preflight(
                logger=logger,
                pause_event=self.pause_event,
                decision_queue=self.decision_queue,
                abort_event=self.abort_event,
                retry_event=self.retry_event,
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
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Preflight sequence encountered an unexpected error")
        finally:
            if handler in logger.handlers:
                logger.removeHandler(handler)
            logger.propagate = True
            self.after(0, self._on_preflight_complete)

    def _on_preflight_complete(self) -> None:
        """Re-enable controls after the preflight run finishes."""

        if self.preflight_button.winfo_exists():
            self.preflight_button.configure(state="normal")
        self._preflight_thread = None

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
    ("_delete_lock_files", "Removing stale lock files", "Lock artefact removal failed"),
    ("_warm_shared_vector_service", "Priming shared vector service", "Vector service warmup failed"),
    ("_ensure_env_flags", "Ensuring environment flags", "Environment configuration failed"),
    ("_prime_registry", "Priming registry cache", "Registry priming failed"),
    ("_install_dependencies", "Installing heavy dependencies", "Dependency installation failed"),
    ("_bootstrap_self_coding", "Bootstrapping self-coding subsystem", "Self-coding bootstrap failed"),
]


def run_full_preflight(
    *,
    logger: logging.Logger,
    pause_event: threading.Event,
    decision_queue: "queue.Queue[Tuple[str, str]]",
    abort_event: threading.Event,
    retry_event: threading.Event,
    debug_queue: "queue.Queue[str]",
    dependency_mode: "DependencyMode" = DependencyMode.STRICT,
) -> Dict[str, Any]:
    """Execute the full sandbox preflight sequence."""

    pause_event.clear()
    retry_event.clear()

    for func_name, action, failure_title in _STEP_DEFINITIONS:
        if abort_event.is_set():
            logger.info("Abort requested before '%s'; stopping", action)
            return {"aborted": True, "healthy": False, "failures": ["aborted"]}

        logger.info("Starting step: %s", action)
        func: Callable[[logging.Logger], None] = globals()[func_name]

        while True:
            retry_requested = False
            try:
                func(logger)
            except Exception as exc:  # pragma: no cover - exercised via unit tests
                logger.exception("Step '%s' failed", action)
                debug_queue.put(str(exc))
                pause_event.set()
                decision_queue.put((failure_title, f"{action}\n\n{exc}"))

                while True:
                    if abort_event.is_set():
                        logger.warning("Abort requested while paused after '%s'", action)
                        return {"aborted": True, "healthy": False, "failures": [failure_title]}
                    if retry_event.is_set():
                        logger.info("Retrying step: %s", action)
                        retry_event.clear()
                        pause_event.clear()
                        retry_requested = True
                        break
                    if not pause_event.is_set():
                        logger.info("Continuing after failure of step: %s", action)
                        break
                    time.sleep(0.1)

                if retry_requested:
                    continue

                if pause_event.is_set():
                    # Retry requested, continue loop to re-run the step.
                    continue
                break
            else:
                pause_event.clear()
                logger.info("Completed step: %s", action)
                break

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

    try:
        subprocess.run(
            ["git", "-C", str(repo_path), "status"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("git executable not available") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"git status failed: {exc.stderr.decode().strip()}") from exc


def _purge_stale_files(logger: logging.Logger) -> None:
    """Remove stale bootstrap artefacts if the helper is available."""

    if bootstrap_self_coding is None:
        logger.info("Bootstrap helpers unavailable; skipping stale file purge")
        return
    bootstrap_self_coding.purge_stale_files()


def _delete_lock_files(logger: logging.Logger) -> None:
    """Delete stale lock files from the sandbox data directory."""

    data_dir = os.environ.get("SANDBOX_DATA_DIR")
    if not data_dir:
        logger.info("SANDBOX_DATA_DIR not set; skipping lock cleanup")
        return

    base = pathlib.Path(data_dir)
    if not base.exists():
        logger.info("Sandbox data directory %s does not exist", base)
        return

    removed = 0
    for lock_file in base.glob("**/*.lock"):
        try:
            lock_file.unlink()
        except FileNotFoundError:  # pragma: no cover - race condition guard
            continue
        else:
            removed += 1

    logger.info("Removed %d stale lock files", removed)


def _warm_shared_vector_service(logger: logging.Logger) -> None:
    """Warm the shared vector service if available."""

    if SharedVectorService is None:
        logger.info("SharedVectorService unavailable; skipping warmup")
        return

    service = SharedVectorService()
    service.vectorise("sanity check")


def _ensure_env_flags(logger: logging.Logger) -> None:
    """Ensure required environment flags are configured."""

    if auto_env_setup is None:
        logger.info("auto_env_setup unavailable; skipping environment validation")
        return
    auto_env_setup.ensure_env()


def _prime_registry(logger: logging.Logger) -> None:
    """Prime the registry cache if the helper module is present."""

    if prime_registry is None:
        logger.info("prime_registry helper unavailable; skipping priming step")
        return
    prime_registry.main()


def _install_dependencies(logger: logging.Logger) -> None:
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
    bootstrap_self_coding.bootstrap_self_coding()


__all__ = [
    "SandboxLauncherGUI",
    "TextWidgetHandler",
    "run_full_preflight",
    "_evaluate_health_snapshot",
]
