"""Shared bootstrap coordination utilities.

A lightweight gate tracks in-progress and completed bootstrap helpers so that
re-entrant callers reuse the prior result instead of re-running heavy setup.
"""
from __future__ import annotations

import inspect
import logging
import os
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, Generic, Optional, Tuple, TypeVar

import bootstrap_metrics

T = TypeVar("T")


class _BootstrapStep(Generic[T]):
    def __init__(self) -> None:
        self.status: str = "pending"
        self.result: Optional[T] = None
        self.event = threading.Event()


class BootstrapManager:
    """Coordinate bootstrap helpers across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._steps: Dict[Tuple[str, str | None], _BootstrapStep[object]] = {}
        self._ready_event = threading.Event()
        self._ready_state: Optional[bool] = None
        self._ready_error: Optional[str] = None
        self._recent_attempts: Dict[Tuple[str, str | None], Deque[float]] = {}
        self._reentry_window_seconds = 5.0
        self._attempt_threshold = max(
            2, int(os.getenv("BOOTSTRAP_DENSITY_ALERT_THRESHOLD", "3") or 3)
        )
        self._thread_context = threading.local()

    @staticmethod
    def _module_label(caller: Dict[str, object]) -> str:
        module = caller.get("caller_module")
        try:
            return str(module) if module else "unknown"
        except Exception:
            return "unknown"

    def _stack(self) -> list[Tuple[str, str | None]]:
        stack = getattr(self._thread_context, "stack", None)
        if stack is None:
            stack = []
            self._thread_context.stack = stack
        return stack

    @staticmethod
    def _caller_details() -> Dict[str, object]:
        frame = inspect.currentframe()
        caller_frame = frame.f_back if frame else None
        # Walk one additional level to reach the external caller of run_once
        caller_frame = caller_frame.f_back if caller_frame else None
        module = inspect.getmodule(caller_frame) if caller_frame else None
        caller_info = {
            "caller_module": module.__name__ if module else None,
            "caller_function": caller_frame.f_code.co_name if caller_frame else None,
            "caller_lineno": caller_frame.f_lineno if caller_frame else None,
        }
        del frame
        del caller_frame
        return caller_info

    def _log_lifecycle(
        self,
        logger: logging.Logger,
        *,
        step: str,
        fingerprint: str | None,
        module: str,
        action: str,
        state_from: str | None,
        state_to: str | None,
        caller: Dict[str, object],
        message: str,
        level: int = logging.INFO,
    ) -> None:
        extra = {
            "event": "bootstrap-lifecycle",
            "bootstrap_step": step,
            "fingerprint": fingerprint,
            "bootstrap_module": module,
            "action": action,
            "state_from": state_from,
            "state_to": state_to,
        }
        extra.update(caller)
        logger.log(level, message, extra=extra)

    def _record_attempt_density(
        self,
        *,
        logger: logging.Logger,
        key: Tuple[str, str | None],
        caller: Dict[str, object],
        module: str,
    ) -> None:
        now = time.monotonic()
        attempts = self._recent_attempts.get(key)
        if attempts is None:
            attempts = deque()
            self._recent_attempts[key] = attempts

        while attempts and now - attempts[0] > self._reentry_window_seconds:
            attempts.popleft()

        attempts.append(now)
        attempt_count = len(attempts)
        if attempt_count > 1:
            window = round(now - attempts[0], 2)
            level = logging.WARNING if attempt_count < self._attempt_threshold else logging.ERROR
            logger.log(
                level,
                "multiple bootstrap attempts detected in short window",
                extra={
                    "event": "bootstrap-density",
                    "bootstrap_step": key[0],
                    "fingerprint": key[1],
                    "bootstrap_module": module,
                    "attempts_in_window": attempt_count,
                    "window_seconds": window,
                    "threshold": self._attempt_threshold,
                    **caller,
                },
            )
        bootstrap_metrics.record_attempt_density(
            module=module,
            step=key[0],
            attempts_in_window=attempt_count,
            window_seconds=self._reentry_window_seconds,
            logger=None if attempt_count == 1 else logger,
            fingerprint=key[1],
            caller=caller,
        )

    def run_once(
        self,
        step: str,
        func: Callable[[], T],
        *,
        logger: logging.Logger | None = None,
        fingerprint: object | None = None,
    ) -> T:
        """Execute ``func`` at most once for the given ``step`` key.

        Additional calls while the step is running will wait for the first
        invocation to complete. Subsequent calls return the cached result
        immediately. The ``fingerprint`` parameter allows callers to include
        configuration details in the cache key without worrying about
        hashability—the manager normalises it to a ``repr`` string.
        """

        log = logger or logging.getLogger(__name__)
        caller_info = self._caller_details()
        module_label = self._module_label(caller_info)
        fingerprint_key = repr(fingerprint) if fingerprint is not None else None
        key = (step, fingerprint_key)
        stack = self._stack()
        if key in stack:
            bootstrap_metrics.record_bootstrap_recursion(
                module=module_label,
                step=step,
                logger=log,
                fingerprint=fingerprint_key,
                caller=caller_info,
            )
            log.error(
                "recursive bootstrap invocation detected",
                extra={
                    "event": "bootstrap-recursion",
                    "bootstrap_step": step,
                    "fingerprint": fingerprint_key,
                    "module_name": module_label,
                    **caller_info,
                },
            )
            existing = self._steps.get(key)
            if existing and existing.status == "completed":
                return existing.result  # type: ignore[return-value]
            raise RuntimeError(
                f"Recursive bootstrap detected for step {step} (fingerprint={fingerprint_key})"
            )

        stack.append(key)
        try:
            with self._lock:
                self._record_attempt_density(
                    logger=log, key=key, caller=caller_info, module=module_label
                )
                existing = self._steps.get(key)
                if existing and existing.status == "completed":
                    bootstrap_metrics.record_bootstrap_guard(
                        "ready",
                        logger=log,
                        module=module_label,
                        step=step,
                        fingerprint=fingerprint_key,
                        caller=caller_info,
                    )
                    bootstrap_metrics.record_bootstrap_skip(
                        "ready",
                        logger=log,
                        module=module_label,
                        step=step,
                        fingerprint=fingerprint_key,
                        caller=caller_info,
                    )
                    bootstrap_metrics.record_bootstrap_entry(
                        "already bootstrapped",
                        logger=log,
                        module=module_label,
                        step=step,
                        fingerprint=fingerprint_key,
                    )
                    self._log_lifecycle(
                        log,
                        step=step,
                        fingerprint=fingerprint_key,
                        module=module_label,
                        action="skip",
                        state_from=existing.status,
                        state_to=existing.status,
                        caller=caller_info,
                        message="bootstrap helper skipped (cached)",
                    )
                    return existing.result  # type: ignore[return-value]
                if existing and existing.status == "running":
                    bootstrap_metrics.record_bootstrap_guard(
                        "queued",
                        logger=log,
                        module=module_label,
                        step=step,
                        fingerprint=fingerprint_key,
                        caller=caller_info,
                    )
                    bootstrap_metrics.record_bootstrap_skip(
                        "in-flight",
                        logger=log,
                        module=module_label,
                        step=step,
                        fingerprint=fingerprint_key,
                        caller=caller_info,
                    )
                    bootstrap_metrics.record_bootstrap_entry(
                        "skipped (in-flight)",
                        logger=log,
                        module=module_label,
                        step=step,
                        fingerprint=fingerprint_key,
                    )
                    self._log_lifecycle(
                        log,
                        step=step,
                        fingerprint=fingerprint_key,
                        module=module_label,
                        action="skip",
                        state_from=existing.status,
                        state_to=existing.status,
                        caller=caller_info,
                        message="bootstrap helper already running",
                    )
                    waiter = existing.event
                else:
                    entry = existing or _BootstrapStep()
                    previous_status = entry.status
                    entry.status = "running"
                    entry.event.clear()
                    self._steps[key] = entry
                    waiter = None

                    self._log_lifecycle(
                        log,
                        step=step,
                        fingerprint=fingerprint_key,
                        module=module_label,
                        action="start",
                        state_from=previous_status,
                        state_to=entry.status,
                        caller=caller_info,
                        message="bootstrap helper starting",
                    )
                    bootstrap_metrics.record_bootstrap_lifecycle(
                        "start",
                        module=module_label,
                        step=step,
                    )

            if waiter:
                waiter.wait()
                with self._lock:
                    cached = self._steps.get(key)
                    if cached and cached.status == "completed":
                        bootstrap_metrics.record_bootstrap_guard(
                            "ready",
                            logger=log,
                            module=module_label,
                            step=step,
                            fingerprint=fingerprint_key,
                            caller=caller_info,
                        )
                        bootstrap_metrics.record_bootstrap_skip(
                            "ready",
                            logger=log,
                            module=module_label,
                            step=step,
                            fingerprint=fingerprint_key,
                            caller=caller_info,
                        )
                        bootstrap_metrics.record_bootstrap_entry(
                            "already bootstrapped",
                            logger=log,
                            module=module_label,
                            step=step,
                            fingerprint=fingerprint_key,
                        )
                        return cached.result  # type: ignore[return-value]
                    if cached:
                        previous_status = cached.status
                        cached.status = "running"
                        cached.event.clear()
                    else:
                        previous_status = None
                bootstrap_metrics.record_bootstrap_entry(
                    "fresh start",
                    logger=log,
                    module=module_label,
                    step=step,
                    fingerprint=fingerprint_key,
                    retry=True,
                )
                bootstrap_metrics.record_bootstrap_lifecycle(
                    "start",
                    module=module_label,
                    step=step,
                )
                self._log_lifecycle(
                    log,
                    step=step,
                    fingerprint=fingerprint_key,
                    module=module_label,
                    action="start",
                    state_from=previous_status,
                    state_to="running",
                    caller=caller_info,
                    message="bootstrap helper restarting after wait",
                )
                return func()  # fall back to executing if the first attempt failed

            bootstrap_metrics.record_bootstrap_entry(
                "fresh start",
                logger=log,
                module=module_label,
                step=step,
                fingerprint=fingerprint_key,
            )

            try:
                result = func()
            except Exception:
                with self._lock:
                    failure = self._steps.get(key)
                    if failure:
                        failure.status = "pending"
                        failure.event.set()
                self._log_lifecycle(
                    log,
                    step=step,
                    fingerprint=fingerprint_key,
                    module=module_label,
                    action="fail",
                    state_from="running",
                    state_to="failed",
                    caller=caller_info,
                    message="bootstrap helper failed",
                    level=logging.ERROR,
                )
                bootstrap_metrics.record_bootstrap_lifecycle(
                    "fail",
                    module=module_label,
                    step=step,
                )
                log.exception(
                    "bootstrap helper failed",
                    extra={
                        "bootstrap_step": step,
                        "fingerprint": fingerprint_key,
                        "bootstrap_module": module_label,
                        "status": "failed",
                        **caller_info,
                    },
                )
                raise

            with self._lock:
                completed = self._steps.get(key)
                if completed:
                    completed.result = result
                    completed.status = "completed"
                    completed.event.set()

            self._log_lifecycle(
                log,
                step=step,
                fingerprint=fingerprint_key,
                module=module_label,
                action="finish",
                state_from="running",
                state_to="completed",
                caller=caller_info,
                message="bootstrap helper finished",
            )
            bootstrap_metrics.record_bootstrap_lifecycle(
                "finish",
                module=module_label,
                step=step,
            )
            return result
        finally:
            stack.pop()

    def mark_ready(self, *, ready: bool = True, error: str | None = None) -> None:
        """Publish bootstrap readiness state for gated callers.

        The readiness gate is shared by all modules that need to defer work
        until the bootstrap pipeline has been prepared by an orchestrator.
        """

        with self._lock:
            self._ready_state = bool(ready)
            self._ready_error = error
            self._ready_event.set()

    def wait_until_ready(
        self,
        *,
        timeout: float | None = 10.0,
        check: Callable[[], bool] | None = None,
        poll_interval: float = 0.2,
        description: str | None = None,
    ) -> bool:
        """Wait for bootstrap readiness without re-entering bootstrap logic.

        A best-effort ``check`` callback may be provided to poll a central gate
        (for example the dependency broker) while waiting.  Calls always
        respect ``timeout`` to avoid deadlocks when the readiness gate is
        unreachable.
        """

        desc = description or "bootstrap readiness"
        if timeout is not None and timeout <= 0:
            raise TimeoutError(f"Timed out immediately waiting for {desc}; timeout must be positive")

        if self._ready_event.is_set():
            if self._ready_state:
                return True
            raise RuntimeError(self._ready_error or f"{desc} gate reported failure")

        deadline = None if timeout is None else time.monotonic() + timeout
        last_error: Exception | None = None

        while True:
            if check is not None:
                try:
                    if check():
                        self.mark_ready()
                        return True
                except Exception as exc:  # pragma: no cover - defensive guard
                    last_error = exc

            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining is not None and remaining <= 0:
                break

            waited = self._ready_event.wait(poll_interval if remaining is None else min(poll_interval, remaining))
            if waited:
                if self._ready_state:
                    return True
                raise RuntimeError(self._ready_error or f"{desc} gate reported failure")

        message = self._ready_error or f"Timed out waiting for {desc}"
        if last_error is not None:
            raise TimeoutError(message) from last_error
        raise TimeoutError(message)


bootstrap_manager = BootstrapManager()

__all__ = ["BootstrapManager", "bootstrap_manager"]
