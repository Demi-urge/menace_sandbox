"""Client for menace_visual_agent_2 service."""

from __future__ import annotations

import os
import subprocess
import time
import logging
import threading
import tempfile
from collections import deque
from concurrent.futures import Future
from typing import Iterable, Dict, Any, Callable, Deque, Tuple

from .audit_logger import log_event
from filelock import FileLock


class _ContextFileLock(FileLock):
    """FileLock variant whose ``acquire`` works as a context manager."""

    class _Guard:
        def __init__(self, lock: FileLock) -> None:
            self.lock = lock

        def __enter__(self) -> FileLock:
            return self.lock

        def __exit__(self, exc_type, exc, tb) -> None:
            if self.lock.is_locked:
                try:
                    self.lock.release()
                except Exception:
                    pass

    def _pid_running(self, pid: int) -> bool:
        """Return True if ``pid`` refers to a running process."""
        try:
            os.kill(pid, 0)
        except Exception:
            return False
        return True

    def acquire(self, *args, **kwargs):  # type: ignore[override]
        lock_path = getattr(self, "lock_file", None)
        if lock_path and os.path.exists(lock_path):
            try:
                with open(lock_path, "r") as fh:
                    pid = int(fh.read().strip() or "0")
                if not self._pid_running(pid):
                    os.remove(lock_path)
            except Exception:
                try:
                    os.remove(lock_path)
                except Exception:
                    pass
        super().acquire(*args, **kwargs)
        if lock_path:
            try:
                with open(lock_path, "w") as fh:
                    fh.write(str(os.getpid()))
            except Exception:
                pass
        return self._Guard(self)

    def release(self, *args, **kwargs) -> None:  # type: ignore[override]
        try:
            super().release(*args, **kwargs)
        finally:
            try:
                os.remove(self.lock_file)
            except FileNotFoundError:
                pass
            except Exception:
                pass

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

logger = logging.getLogger(__name__)

# Default instructions prepended before messages sent to the visual agent.
DEFAULT_MESSAGE_PREFIX = (
    "Improve Menace by enhancing error handling and modifying existing bots."
)

# Allow overriding the default via environment variable for easier testing
# and custom deployments.
SELF_IMPROVEMENT_PREFIX = os.getenv("VA_MESSAGE_PREFIX", DEFAULT_MESSAGE_PREFIX)

GLOBAL_LOCK_PATH = os.getenv(
    "VISUAL_AGENT_LOCK_FILE",
    os.path.join(tempfile.gettempdir(), "visual_agent.lock"),
)
_global_lock = _ContextFileLock(GLOBAL_LOCK_PATH)


class VisualAgentClient:
    """Interface used by SelfCodingEngine to contact a visual agent."""

    def __init__(
        self,
        urls: Iterable[str] | None = None,
        token: str | None = None,
        poll_interval: float | None = None,
        token_refresh_cmd: str | None = None,
    ) -> None:
        default_urls = (
            os.getenv("VISUAL_AGENT_URLS")
            or os.getenv("VISUAL_DESKTOP_URL", "http://127.0.0.1:8001")
        ).split(";")
        self.urls = list(filter(None, urls or default_urls))
        self.token = token or os.getenv("VISUAL_AGENT_TOKEN", "")
        self.poll_interval = poll_interval or float(
            os.getenv("VISUAL_AGENT_POLL_INTERVAL", "5")
        )
        self.token_refresh_cmd = token_refresh_cmd or os.getenv(
            "VISUAL_TOKEN_REFRESH_CMD"
        )
        self.open_run_id: str | None = None
        self.active = False
        self._lock = threading.Lock()
        self._queue: Deque[Tuple[Callable[[], Any], Future]] = deque()
        self._cv = threading.Condition()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------------
    def _enqueue(self, func: Callable[[], Any]) -> Future:
        fut: Future = Future()
        with self._cv:
            self._queue.append((func, fut))
            self._cv.notify()
        return fut

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                func, fut = self._queue.popleft()
            try:
                with _global_lock.acquire():
                    result = func()
                fut.set_result(result)
            except Exception as exc:
                fut.set_exception(exc)

    def _poll(self, base: str) -> tuple[bool, str]:
        if not requests:
            return False, "requests unavailable"

        attempts = 0
        delay = self.poll_interval
        while True:
            try:
                resp = requests.get(f"{base}/status", timeout=10)
            except Exception as exc:  # pragma: no cover - network issues
                attempts += 1
                if attempts >= 3:
                    return False, f"status poll failed: {exc}"
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code != 200:
                attempts += 1
                if attempts >= 3:
                    return False, f"unexpected status {resp.status_code}"
                time.sleep(delay)
                delay *= 2
                continue

            data = resp.json()
            if not data.get("active", False):
                return True, str(data.get("status", "done"))

            time.sleep(self.poll_interval)

    def _refresh_token(self) -> bool:
        if not self.token_refresh_cmd:
            return False

        for attempt in range(3):
            proc = subprocess.run(
                self.token_refresh_cmd,
                shell=True,
                text=True,
                capture_output=True,
            )
            output = (proc.stdout + proc.stderr).strip()
            if proc.returncode == 0 and proc.stdout.strip():
                self.token = proc.stdout.strip()
                return True
            logger.warning(
                "token refresh attempt %s failed: %s",
                attempt + 1,
                output,
            )
            if attempt < 2:
                time.sleep(1.0)
        return False

    def _refresh_token_async(self) -> Future:
        """Run ``_refresh_token`` in the background respecting the global lock."""

        def run() -> bool:
            return self._refresh_token()

        return self._enqueue(run)

    def _send(self, base: str, prompt: str) -> tuple[bool, str]:
        if not requests:
            return False, "requests unavailable"

        with self._lock:
            if self.active:
                return False, "busy"
            self.active = True

            def sender() -> tuple[bool, str]:
                if self.open_run_id is None:
                    try:
                        self.open_run_id = log_event(
                            "visual_agent_run",
                            {"url": base, "prompt": prompt, "status": "started"},
                        )
                    except Exception:
                        self.open_run_id = None
                delay = 1.0
                for attempt in range(3):
                    try:
                        resp = requests.post(
                            f"{base}/run",
                            headers={"x-token": self.token},
                            json={"prompt": prompt, "branch": None},
                            timeout=10,
                        )
                    except Exception as exc:
                        if attempt == 2:
                            return False, f"connection error: {exc}"
                        time.sleep(delay)
                        delay *= 2
                        continue

                    if resp.status_code == 401:
                        self._refresh_token_async()
                        return False, "unauthorized"
                    if resp.status_code == 202:
                        return self._poll(base)
                    if resp.status_code >= 500 and attempt < 2:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    snippet = resp.text[:200]
                    return False, f"status {resp.status_code}: {snippet}"
                return False, "failed"

        try:
            return sender()
        except Exception as exc:  # pragma: no cover - network issues
            return False, f"exception {exc}"
        finally:
            with self._lock:
                self.active = False

    def _send_revert(self, base: str) -> tuple[bool, str]:
        if not requests:
            return False, "requests unavailable"

        with self._lock:
            if self.active:
                return False, "busy"
            self.active = True

            def sender() -> tuple[bool, str]:
                delay = 1.0
                for attempt in range(3):
                    try:
                        resp = requests.post(
                            f"{base}/revert",
                            headers={"x-token": self.token},
                            timeout=10,
                        )
                    except Exception as exc:
                        if attempt == 2:
                            return False, f"connection error: {exc}"
                        time.sleep(delay)
                        delay *= 2
                        continue

                    if resp.status_code == 401:
                        self._refresh_token_async()
                        return False, "unauthorized"
                    if resp.status_code == 202:
                        return self._poll(base)
                    if resp.status_code >= 500 and attempt < 2:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    snippet = resp.text[:200]
                    return False, f"status {resp.status_code}: {snippet}"
                return False, "failed"

        try:
            return sender()
        except Exception as exc:  # pragma: no cover - network issues
            return False, f"exception {exc}"
        finally:
            with self._lock:
                self.active = False

    def ask_async(self, messages: Iterable[Dict[str, str]]) -> Future:
        prompt = SELF_IMPROVEMENT_PREFIX + "\n\n" + "\n".join(
            m.get("content", "") for m in messages
        )

        def run() -> Dict[str, Any]:
            last_reason = "failed"
            for url in self.urls:
                ok, reason = self._send(url, prompt)
                if ok:
                    return {"choices": [{"message": {"content": reason}}]}
                last_reason = reason
            raise RuntimeError(f"visual agent failed: {last_reason}")

        return self._enqueue(run)

    def ask(self, messages: Iterable[Dict[str, str]]) -> Dict[str, Any]:
        """Synchronously send ``messages`` to the visual agent."""
        fut = self.ask_async(messages)
        return fut.result()

    def revert_async(self) -> Future:
        """Trigger a revert asynchronously via the visual agent."""

        def run() -> bool:
            last_reason = "failed"
            for url in self.urls:
                ok, reason = self._send_revert(url)
                if ok:
                    return True
                last_reason = reason
            raise RuntimeError(f"visual agent revert failed: {last_reason}")

        return self._enqueue(run)

    def revert(self) -> bool:
        fut = self.revert_async()
        return fut.result()

    # ------------------------------------------------------------------
    def resolve_run_log(self, outcome: str) -> None:
        """Mark the most recent /run call as resolved with ``outcome``."""
        if not self.open_run_id:
            return
        try:
            log_event(
                "visual_agent_run_result",
                {"run_id": self.open_run_id, "outcome": outcome},
            )
        finally:
            self.open_run_id = None
            with self._lock:
                self.active = False


class VisualAgentClientStub:
    """Fallback client used when requests or the real client is unavailable."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        logger.info("VisualAgentClientStub active")
        self.urls: list[str] = []
        self.open_run_id: str | None = None
        self.active = False
        self._lock = threading.Lock()

    def ask_async(self, messages: Iterable[Dict[str, str]]) -> Future:
        fut: Future = Future()
        fut.set_result({"choices": [{"message": {"content": ""}}]})
        return fut

    def ask(self, messages: Iterable[Dict[str, str]]) -> Dict[str, Any]:
        return self.ask_async(messages).result()

    def revert_async(self) -> Future:
        fut: Future = Future()
        fut.set_result(False)
        return fut

    def revert(self) -> bool:
        return self.revert_async().result()

    def resolve_run_log(self, outcome: str) -> None:
        self.open_run_id = None
        with self._lock:
            self.active = False


__all__ = ["VisualAgentClient", "VisualAgentClientStub"]

