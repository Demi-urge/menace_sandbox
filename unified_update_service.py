from __future__ import annotations

"""Unify dependency updates with automated deployment."""

import logging
import os
import subprocess
import threading
import time
from threading import Event
from typing import Optional

from .dependency_update_service import DependencyUpdateService
from .deployment_bot import DeploymentBot, DeploymentSpec
from .rollback_manager import RollbackManager


class UnifiedUpdateService:
    """Run dependency updates and redeploy automatically."""

    def __init__(
        self,
        updater_service: Optional[DependencyUpdateService] = None,
        deployer: Optional[DeploymentBot] = None,
        rollback_mgr: Optional[RollbackManager] = None,
        *,
        max_retries: int = 0,
        retry_backoff_seconds: float = 1.0,
        retry_backoff_cap_seconds: float = 30.0,
        test_scope: str = "smoke",
        smoke_marker: str = "smoke",
        full_test_flag_env: str = "UNIFIED_UPDATE_FULL_SUITE",
    ) -> None:
        self.update_service = updater_service or DependencyUpdateService()
        self.deployer = deployer or DeploymentBot()
        self.rollback_mgr = rollback_mgr
        self.max_retries = max_retries
        self.retry_backoff_seconds = max(retry_backoff_seconds, 0.0)
        self.retry_backoff_cap_seconds = max(retry_backoff_cap_seconds, 0.0)
        self.test_scope = test_scope
        self.smoke_marker = smoke_marker
        self.full_test_flag_env = full_test_flag_env
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _is_truthy(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _pytest_cmd(self) -> list[str]:
        force_full_suite = self._is_truthy(os.getenv(self.full_test_flag_env, "0"))
        if self.test_scope == "full" or force_full_suite:
            return ["pytest", "-q"]
        return ["pytest", "-q", "-m", self.smoke_marker]

    @staticmethod
    def _is_terminal_cycle_error(exc: Exception) -> bool:
        return isinstance(exc, subprocess.CalledProcessError) and exc.returncode == 2

    @staticmethod
    def _root_cause_category(exc: Exception) -> str:
        msg = str(exc).lower()
        if any(p in msg for p in ("optional dependency", "extra", "not installed")):
            return "missing optional pkg"
        if any(p in msg for p in ("connection", "timeout", "dns", "network")):
            return "infra dependency"
        if (
            isinstance(exc, subprocess.CalledProcessError)
            and exc.returncode == 2
            or any(p in msg for p in ("importerror", "modulenotfounderror", "cannot import name"))
        ):
            return "import contract"
        return "unknown"

    def _cycle(self) -> None:
        for attempt in range(self.max_retries + 1):
            try:
                verify_host = os.getenv("DEP_VERIFY_HOST")
                self.update_service._run_once(verify_host=verify_host)
                try:
                    spec = DeploymentSpec(name="auto-update", resources={}, env={})
                except Exception as spec_exc:
                    raise RuntimeError(f"failed to build deployment spec: {spec_exc}") from spec_exc
                self.deployer.deploy("auto-update", [], spec)
                subprocess.run(self._pytest_cmd(), check=True)

                nodes = [n for n in os.getenv("NODES", "").split(",") if n]
                if nodes:
                    batch = int(os.getenv("ROLLOUT_BATCH_SIZE", "1"))
                    first, rest = nodes[:batch], nodes[batch:]
                    events = []
                    bus = getattr(self.deployer, "event_bus", None)
                    if first and bus:
                        bus.subscribe("nodes:update", lambda t, e: events.append(e))
                    if first:
                        self.deployer.auto_update_nodes(first)
                    if first and any(e.get("status") != "success" for e in events):
                        raise RuntimeError("staged rollout failed")
                    if rest:
                        self.deployer.auto_update_nodes(rest)
                self.logger.info("update cycle successful on attempt %s", attempt + 1)
                break
            except Exception as exc:
                category = self._root_cause_category(exc)
                self.logger.error(
                    "deployment failed on attempt %s (root-cause=%s): %s",
                    attempt + 1,
                    category,
                    exc,
                )
                if self.rollback_mgr:
                    try:
                        nodes = [n for n in os.getenv("NODES", "").split(",") if n]
                        self.rollback_mgr.auto_rollback("latest", nodes)
                    except Exception as rb_exc:
                        self.logger.error("rollback failed: %s", rb_exc)
                if self._is_terminal_cycle_error(exc):
                    self.logger.error("terminal test failure detected; aborting remaining retries")
                    raise
                if attempt == self.max_retries:
                    raise
                delay = min(self.retry_backoff_seconds * (2**attempt), self.retry_backoff_cap_seconds)
                if delay > 0:
                    time.sleep(delay)

    def run_continuous(self, interval: float = 86400.0, *, stop_event: Optional[Event] = None) -> None:
        if stop_event is None:
            stop_event = Event()

        def _loop() -> None:
            while not stop_event.is_set():
                try:
                    self._cycle()
                except Exception as exc:
                    self.logger.error("cycle failed and will be retried on next interval: %s", exc)
                if stop_event.wait(interval):
                    break

        threading.Thread(target=_loop, daemon=True).start()


__all__ = ["UnifiedUpdateService"]
