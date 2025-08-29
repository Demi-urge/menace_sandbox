"""Client for menace_visual_agent_2 service."""

from __future__ import annotations

import os
import subprocess
import time
import logging
import threading
import tempfile
import errno
from contextlib import suppress
from collections import deque
from concurrent.futures import Future
from typing import Iterable, Dict, Any, Callable, Deque, Tuple
from pathlib import Path
import json

from . import metrics_exporter

from .audit_logger import log_event

from .lock_utils import SandboxLock, LOCK_TIMEOUT

# Backwards compatibility for tests importing the old name
_ContextFileLock = SandboxLock
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

logger = logging.getLogger(__name__)


class _LocalQueue:
    """Minimal persistent queue for failed requests."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, item: dict) -> None:
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(item) + "\n")


def _queue_local_task(queue: _LocalQueue, item: dict) -> None:
    try:
        queue.append(item)
    except Exception as exc:  # pragma: no cover - fs errors
        logger.warning("failed to queue task locally: %s", exc)


class VisualAgentError(RuntimeError):
    """Base class for visual agent client errors."""


class VisualAgentConnectionError(VisualAgentError):
    """Raised when communication with the visual agent repeatedly fails."""

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
_global_lock = SandboxLock(GLOBAL_LOCK_PATH)


class VisualAgentClient:
    """Interface used by SelfCodingEngine to contact a visual agent."""

    def __init__(
        self,
        urls: Iterable[str] | None = None,
        token: str | None = None,
        poll_interval: float | None = None,
        token_refresh_cmd: str | None = None,
        queue_warning_threshold: int | None = None,
        metrics_interval: float | None = None,
        status_interval: float | None = None,
        flush_interval: float | None = None,
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
        warn = queue_warning_threshold
        if warn is None:
            env_val = os.getenv("VISUAL_AGENT_QUEUE_THRESHOLD")
            warn = int(env_val) if env_val and env_val.isdigit() else None
        self.queue_warning_threshold = warn
        self.metrics_interval = metrics_interval or float(
            os.getenv("VISUAL_AGENT_METRICS_INTERVAL", "30")
        )
        self.status_interval = status_interval if status_interval is not None else float(
            os.getenv("VISUAL_AGENT_STATUS_INTERVAL", "0")
        )
        self.flush_interval = flush_interval or float(
            os.getenv("VISUAL_AGENT_FLUSH_INTERVAL", "60")
        )
        self.open_run_id: str | None = None
        self.active = False
        self._lock = threading.Lock()
        self._queue: Deque[Tuple[Callable[[], Any], Future, float]] = deque()
        self._wait_lock = threading.Lock()
        self._wait_times: Deque[float] = deque(maxlen=100)
        self._cv = threading.Condition()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self._stop_event = threading.Event()
        data_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
        self._local_queue = _LocalQueue(data_dir / "visual_agent_client_queue.jsonl")
        if self.queue_warning_threshold is not None and self.urls and requests:
            self._metrics_thread = threading.Thread(
                target=self._metrics_loop, daemon=True
            )
            self._metrics_thread.start()
        else:
            self._metrics_thread = None
        if self.status_interval > 0 and self.urls and requests:
            self._status_thread = threading.Thread(
                target=self._status_loop, daemon=True
            )
            self._status_thread.start()
        else:
            self._status_thread = None

        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    # ------------------------------------------------------------------
    def _enqueue(self, func: Callable[[], Any]) -> Future:
        fut: Future = Future()
        with self._cv:
            self._queue.append((func, fut, time.time()))
            self._cv.notify()
        return fut

    def _worker_loop(self) -> None:
        while True:
            with self._cv:
                while not self._queue:
                    self._cv.wait()
                func, fut, ts = self._queue.popleft()
            wait = time.time() - ts
            self._record_wait_time(wait)
            try:
                with _global_lock.acquire():
                    result = func()
                fut.set_result(result)
            except Exception as exc:
                fut.set_exception(exc)

    def _record_wait_time(self, delta: float) -> None:
        with self._wait_lock:
            self._wait_times.append(delta)
            avg = sum(self._wait_times) / len(self._wait_times)
        gauge = getattr(metrics_exporter, "visual_agent_wait_time", None)
        if gauge is not None:
            try:
                gauge.set(avg)
            except Exception as exc:
                logger.exception("failed to set wait time metric: %s", exc)

    def _metrics_loop(self) -> None:
        while not self._stop_event.wait(self.metrics_interval):
            for base in self.urls:
                try:
                    resp = requests.get(f"{base}/metrics", timeout=5)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    qsize = data.get("queue")
                    if (
                        isinstance(qsize, int)
                        and self.queue_warning_threshold is not None
                        and qsize > self.queue_warning_threshold
                    ):
                        logger.warning(
                            "visual agent queue size %s exceeds threshold %s",
                            qsize,
                            self.queue_warning_threshold,
                        )
                    break
                except Exception:
                    continue

    def _status_loop(self) -> None:
        while not self._stop_event.wait(self.status_interval):
            for base in self.urls:
                try:
                    resp = requests.get(f"{base}/status", timeout=5)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    qsize = data.get("queue")
                    if isinstance(qsize, int):
                        gauge = getattr(metrics_exporter, "visual_agent_queue_depth", None)
                        if gauge is not None:
                            try:
                                gauge.set(qsize)
                            except Exception as exc:
                                logger.exception("failed to set queue depth metric: %s", exc)
                        break
                except Exception:
                    continue

    def _flush_loop(self) -> None:
        self.flush_local_queue()
        while not self._stop_event.wait(self.flush_interval):
            self.flush_local_queue()

    def flush_local_queue(self) -> None:
        path = self._local_queue.path
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with self._local_queue._lock:
            if not path.exists():
                return
            try:
                path.rename(tmp_path)
            except Exception:
                return
            # ensure new file for further queueing
            open(path, "a", encoding="utf-8").close()

        try:
            with open(tmp_path, "r", encoding="utf-8") as fh:
                items = [json.loads(l) for l in fh if l.strip()]
        except Exception as exc:  # pragma: no cover - read errors
            logger.warning("failed to load local queue: %s", exc)
            items = []

        remaining: list[dict] = []
        for item in items:
            action = item.get("action")
            success = False
            if action == "run":
                for url in self.urls:
                    ok, _ = self._send(url, item.get("prompt", ""))
                    if ok:
                        success = True
                        break
            elif action == "revert":
                for url in self.urls:
                    ok, _ = self._send_revert(url)
                    if ok:
                        success = True
                        break
            if not success:
                remaining.append(item)

        for item in remaining:
            _queue_local_task(self._local_queue, item)

        try:
            tmp_path.unlink()
        except Exception:
            pass

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
                logger.warning(
                    "status poll attempt %s to %s failed: %s",
                    attempts,
                    base,
                    exc,
                )
                if attempts >= 3:
                    msg = f"status poll failed after {attempts} attempts: {exc}"
                    logger.error(msg)
                    raise VisualAgentConnectionError(msg)
                time.sleep(delay)
                delay *= 2
                continue

            if resp.status_code != 200:
                attempts += 1
                logger.warning(
                    "status poll attempt %s to %s returned %s",
                    attempts,
                    base,
                    resp.status_code,
                )
                if attempts >= 3:
                    msg = f"unexpected status {resp.status_code}"
                    logger.error(msg)
                    raise VisualAgentConnectionError(msg)
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

        last_stdout = ""
        last_stderr = ""
        for attempt in range(3):
            proc = subprocess.run(
                self.token_refresh_cmd,
                shell=True,
                text=True,
                capture_output=True,
            )
            last_stdout = proc.stdout.strip()
            last_stderr = proc.stderr.strip()
            output = (proc.stdout + proc.stderr).strip()
            if proc.returncode == 0 and last_stdout:
                self.token = last_stdout
                return True
            logger.warning(
                "token refresh attempt %s failed: %s",
                attempt + 1,
                output,
            )
            if attempt < 2:
                time.sleep(1.0)
        logger.error(
            "token refresh failed after %s attempts: %s",
            attempt + 1,
            last_stderr or last_stdout,
        )
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
                retried = False
                for attempt in range(3):
                    try:
                        resp = requests.post(
                            f"{base}/run",
                            headers={"Authorization": f"Bearer {self.token}"},
                            json={"prompt": prompt, "branch": None},
                            timeout=10,
                        )
                    except Exception as exc:
                        logger.warning(
                            "send attempt %s to %s failed: %s",
                            attempt + 1,
                            base,
                            exc,
                        )
                        if attempt == 2:
                            msg = f"connection error after {attempt + 1} attempts: {exc}"
                            logger.error(msg)
                            _queue_local_task(self._local_queue, {"action": "run", "prompt": prompt})
                            return True, msg
                        time.sleep(delay)
                        delay *= 2
                        continue

                    if resp.status_code == 401:
                        if not retried:
                            retried = True
                            if self._refresh_token():
                                continue
                            return False, "token refresh failed"
                        return False, "unauthorized"
                    if resp.status_code == 409:
                        self._poll(base)
                        continue
                    if resp.status_code == 202:
                        return self._poll(base)
                    if resp.status_code >= 500:
                        logger.warning(
                            "send attempt %s to %s returned %s",
                            attempt + 1,
                            base,
                            resp.status_code,
                        )
                        if attempt == 2:
                            msg = f"server error status {resp.status_code}"
                            logger.error(msg)
                            _queue_local_task(self._local_queue, {"action": "run", "prompt": prompt})
                            return True, msg
                        time.sleep(delay)
                        delay *= 2
                        continue
                    snippet = resp.text[:200]
                    return False, f"status {resp.status_code}: {snippet}"
                _queue_local_task(self._local_queue, {"action": "run", "prompt": prompt})
                return True, "queued locally"

        try:
            return sender()
        except PermissionError:
            raise
        except VisualAgentError:
            raise
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
                retried = False
                for attempt in range(3):
                    try:
                        resp = requests.post(
                            f"{base}/revert",
                            headers={"Authorization": f"Bearer {self.token}"},
                            timeout=10,
                        )
                    except Exception as exc:
                        logger.warning(
                            "revert attempt %s to %s failed: %s",
                            attempt + 1,
                            base,
                            exc,
                        )
                        if attempt == 2:
                            msg = f"connection error after {attempt + 1} attempts: {exc}"
                            logger.error(msg)
                            _queue_local_task(self._local_queue, {"action": "revert"})
                            return True, msg
                        time.sleep(delay)
                        delay *= 2
                        continue

                    if resp.status_code == 401:
                        if not retried:
                            retried = True
                            if self._refresh_token():
                                continue
                            return False, "token refresh failed"
                        return False, "unauthorized"
                    if resp.status_code == 409:
                        self._poll(base)
                        continue
                    if resp.status_code == 202:
                        return self._poll(base)
                    if resp.status_code >= 500:
                        logger.warning(
                            "revert attempt %s to %s returned %s",
                            attempt + 1,
                            base,
                            resp.status_code,
                        )
                        if attempt == 2:
                            msg = f"server error status {resp.status_code}"
                            logger.error(msg)
                            _queue_local_task(self._local_queue, {"action": "revert"})
                            return True, msg
                        time.sleep(delay)
                        delay *= 2
                        continue
                    snippet = resp.text[:200]
                    return False, f"status {resp.status_code}: {snippet}"
                _queue_local_task(self._local_queue, {"action": "revert"})
                return True, "queued locally"

        try:
            return sender()
        except PermissionError:
            raise
        except VisualAgentError:
            raise
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
        with self._lock:
            if self.active:
                fut.set_exception(RuntimeError("409 busy"))
                return fut
            self.active = True
        try:
            fut.set_result({"choices": [{"message": {"content": ""}}]})
        finally:
            with self._lock:
                self.active = False
        return fut

    def ask(self, messages: Iterable[Dict[str, str]]) -> Dict[str, Any]:
        return self.ask_async(messages).result()

    def revert_async(self) -> Future:
        fut: Future = Future()
        with self._lock:
            if self.active:
                fut.set_exception(RuntimeError("409 busy"))
                return fut
            self.active = True
        try:
            fut.set_result(False)
        finally:
            with self._lock:
                self.active = False
        return fut

    def revert(self) -> bool:
        return self.revert_async().result()

    def resolve_run_log(self, outcome: str) -> None:
        self.open_run_id = None
        with self._lock:
            self.active = False


__all__ = [
    "VisualAgentClient",
    "VisualAgentClientStub",
    "VisualAgentError",
    "VisualAgentConnectionError",
]

