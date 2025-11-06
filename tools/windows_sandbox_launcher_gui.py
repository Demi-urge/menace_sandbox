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
from pathlib import Path
from tkinter import ttk
from typing import Callable, Iterable

from menace_sandbox.auto_env_setup import ensure_env
from menace_sandbox.bootstrap_self_coding import (
    bootstrap_self_coding,
    purge_stale_files,
)
from menace_sandbox.lock_utils import is_lock_stale
from menace_sandbox.vector_service.vectorizer import SharedVectorService
from neurosales.scripts import setup_heavy_deps
from prime_registry import main as prime_registry_main


REPO_ROOT = Path(__file__).resolve().parent.parent
HF_TRANSFORMERS_CACHE = Path.home() / ".cache" / "huggingface" / "transformers"
try:
    LOCK_STALE_TIMEOUT = max(300, int(os.getenv("SANDBOX_LOCK_TIMEOUT", "300")))
except ValueError:
    LOCK_STALE_TIMEOUT = 300


class TextWidgetQueueHandler(logging.Handler):
    """Logging handler that places formatted log messages onto a queue."""

    def __init__(self, log_queue: "queue.Queue[tuple[str, str]]") -> None:
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            if not message.endswith("\n"):
                message = f"{message}\n"
            self.log_queue.put((record.levelname.lower(), message))
        except Exception:  # pragma: no cover - rely on logging's default handling
            self.handleError(record)


class SandboxLauncherGUI(tk.Tk):
    """Simple GUI window for managing sandbox lifecycle actions."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Windows Sandbox Launcher")
        self.geometry("600x400")

        self.log_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
        self.logger = logging.getLogger("windows_sandbox_launcher_gui")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        self._preflight_thread: threading.Thread | None = None
        self._preflight_running = False
        self.pause_event = threading.Event()
        self.abort_event = threading.Event()
        self.decision_queue: "queue.Queue[tuple[str, str | None, dict[str, object] | None]]" = queue.Queue()

        self._build_notebook()
        self._build_controls()
        self._configure_logging()

        self._process_log_queue()

    def _build_notebook(self) -> None:
        self.notebook = ttk.Notebook(self)

        self.status_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.status_tab, text="Status")

        self.status_text = tk.Text(
            self.status_tab,
            wrap="word",
            state="disabled",
            background="#1e1e1e",
        )
        self.status_text.pack(fill="both", expand=True, padx=10, pady=10)

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

    def _configure_logging(self) -> None:
        default_font = tkfont.nametofont("TkDefaultFont")

        self.status_text.tag_config("info", foreground="white", font=default_font)
        self.status_text.tag_config("warning", foreground="yellow", font=default_font)

        error_font = default_font.copy()
        error_font.configure(weight="bold")
        self.status_text.tag_config("error", foreground="red", font=error_font)
        self._error_font = error_font

        queue_handler = TextWidgetQueueHandler(self.log_queue)
        queue_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(queue_handler)
        self._queue_handler = queue_handler

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
        self.pause_event.clear()
        self.abort_event.clear()
        while not self.decision_queue.empty():
            try:
                self.decision_queue.get_nowait()
            except queue.Empty:  # pragma: no cover - race with concurrent reader
                break
        self._preflight_running = True
        self.logger.info("Preflight checks initiated...")

        self._preflight_thread = threading.Thread(
            target=self._execute_preflight,
            name="PreflightWorker",
            daemon=True,
        )
        self._preflight_thread.start()

    def start_sandbox(self) -> None:  # pragma: no cover - placeholder hook
        """Placeholder command that will be implemented in a future iteration."""
        self.logger.info("Sandbox startup sequence initiated...")

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
            for title, start_message, success_message, handler in steps:
                if self.abort_event.is_set():
                    self.logger.info("Preflight aborted before '%s'.", title)
                    return

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
                    self.decision_queue.put((title, failure_message, context))
                    self._update_pause_state(title, failure_message, context)
                    if not self._await_resume(title):
                        return
                    continue

                self.logger.info(success_message)
                if not self._await_resume(title):
                    return
            self.logger.info("Preflight checks complete.")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception("Preflight checks failed: %s", exc)
        finally:
            self.after(0, self._reset_preflight_state)

    def _await_resume(self, title: str) -> bool:
        """Wait for the pause event to clear or abort if requested."""

        while self.pause_event.is_set() and not self.abort_event.is_set():
            time.sleep(0.05)
        if self.abort_event.is_set():
            self.logger.info("Preflight aborted during '%s'.", title)
            return False
        return True

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

    def _reset_preflight_state(self) -> None:
        """Re-enable UI controls and clear state after preflight finishes."""
        self._preflight_running = False
        self._preflight_thread = None
        self.run_preflight_button.configure(state="normal")

    def _process_log_queue(self) -> None:
        while True:
            try:
                tag, message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._append_status(message, tag)
        if self.pause_event.is_set():
            self._handle_preflight_pause()

        self.after(100, self._process_log_queue)

    def _append_status(self, message: str, tag: str = "info") -> None:
        self.status_text.configure(state="normal")
        self.status_text.insert("end", message, (tag,))
        self.status_text.configure(state="disabled")
        self.status_text.see("end")

    def _handle_preflight_pause(self) -> None:
        try:
            payload = self.decision_queue.get_nowait()
        except queue.Empty:
            return

        title: str
        message: str | None
        context: dict[str, object] | None
        if not isinstance(payload, tuple):
            return
        if len(payload) == 3:
            title, message, context = payload
        else:
            title = payload[0]
            message = payload[1] if len(payload) > 1 else None
            context = None

        detail_lines = [message or "No additional details provided."]
        if context and context.get("exception"):
            detail_lines.append(str(context["exception"]))
        prompt = (
            f"{title} failed. Continue preflight?\n\nDetails: "
            + "\n".join(detail_lines)
        )
        user_choice = messagebox.askyesno("Preflight paused", prompt)
        if user_choice:
            self.logger.info("User opted to continue after '%s'.", title)
            self.pause_event.clear()
        else:
            self.logger.warning("User aborted preflight after '%s'.", title)
            self.abort_event.set()
            self.pause_event.clear()

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


__all__ = ["SandboxLauncherGUI"]
