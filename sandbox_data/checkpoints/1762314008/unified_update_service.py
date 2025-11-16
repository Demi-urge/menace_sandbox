from __future__ import annotations

"""Unify dependency updates with automated deployment."""

import logging
import os
import subprocess
import threading
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
    ) -> None:
        self.update_service = updater_service or DependencyUpdateService()
        self.deployer = deployer or DeploymentBot()
        self.rollback_mgr = rollback_mgr
        self.max_retries = max_retries
        self.logger = logging.getLogger(self.__class__.__name__)

    def _cycle(self) -> None:
        for attempt in range(self.max_retries + 1):
            try:
                verify_host = os.getenv("DEP_VERIFY_HOST")
                self.update_service._run_once(verify_host=verify_host)
                spec = DeploymentSpec(resources={})
                self.deployer.deploy("auto-update", [], spec)
                subprocess.run(["pytest", "-q"], check=True)

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
                self.logger.error("deployment failed on attempt %s: %s", attempt + 1, exc)
                if self.rollback_mgr:
                    try:
                        nodes = [n for n in os.getenv("NODES", "").split(",") if n]
                        self.rollback_mgr.auto_rollback("latest", nodes)
                    except Exception as rb_exc:
                        self.logger.error("rollback failed: %s", rb_exc)
                if attempt == self.max_retries:
                    raise

    def run_continuous(self, interval: float = 86400.0, *, stop_event: Optional[Event] = None) -> None:
        if stop_event is None:
            stop_event = Event()

        def _loop() -> None:
            while not stop_event.is_set():
                self._cycle()
                if stop_event.wait(interval):
                    break

        threading.Thread(target=_loop, daemon=True).start()


__all__ = ["UnifiedUpdateService"]
