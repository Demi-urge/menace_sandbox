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
        self._preflight_successful = False
        self._sandbox_thread: threading.Thread | None = None
        self._sandbox_process: subprocess.Popen[str] | None = None
        self._sandbox_running = False
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
        self.start_sandbox_button.configure(state="disabled")
        self._preflight_successful = False
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
            if self.abort_event.is_set():
                self.logger.info("Preflight aborted after completing queued steps.")
                return
            self.logger.info("Preflight checks complete.")
            self._run_post_preflight_health_check()
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
        if not self._sandbox_running:
            self.run_preflight_button.configure(state="normal")
            self.start_sandbox_button.configure(
                state="normal" if self._preflight_successful else "disabled"
            )

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


__all__ = ["SandboxLauncherGUI", "_evaluate_health_snapshot"]
