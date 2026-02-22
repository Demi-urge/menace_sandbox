from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import logging

from db_router import GLOBAL_ROUTER
from scope_utils import Scope, build_scope_clause, apply_scope

from .evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from .borderline_bucket import BorderlineBucket
from .error_flags import RAISE_ERRORS

if TYPE_CHECKING:  # pragma: no cover - heavy dependency
    from .deployment_bot import DeploymentBot
    from .resource_allocation_optimizer import ResourceAllocationOptimizer
    from .data_bot import DataBot
    from .watchdog import Watchdog


@dataclass
class TrialConfig:
    roi_threshold: float = 0.0
    success_threshold: float = 0.5
    interval: int = 60


class TrialMonitor:
    """Monitor trial bots and disable those that perform poorly."""

    def __init__(
        self,
        deployer: DeploymentBot,
        optimizer: ResourceAllocationOptimizer,
        data_bot: DataBot,
        history_db: EvolutionHistoryDB | None = None,
        *,
        config: TrialConfig | None = None,
        watchdog: Watchdog | None = None,
        borderline_bucket: BorderlineBucket | None = None,
    ) -> None:
        self.deployer = deployer
        self.optimizer = optimizer
        self.data_bot = data_bot
        self.history_db = history_db or EvolutionHistoryDB()
        self.config = config or TrialConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.watchdog = watchdog
        self.borderline_bucket = borderline_bucket or BorderlineBucket()
        self.failure_count = 0
        self._task: Optional[asyncio.Task] = None
        self.running = False

    def start(self) -> None:
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._loop())

    def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()

    async def _loop(self) -> None:
        while self.running:
            try:
                self.check_trials()
            except Exception as exc:
                self.logger.exception("trial check failed: %s", exc)
                self.failure_count += 1
                if self.watchdog:
                    self.watchdog.escalate(f"trial monitor loop failed: {exc}")
                if RAISE_ERRORS:
                    raise
            await asyncio.sleep(self.config.interval)

    def _success_rate(self, bot: str) -> float:
        df = self.data_bot.db.fetch(20)
        if hasattr(df, "empty"):
            df = df[df["bot"] == bot]
            total = len(df)
            errors = int(df["errors"].sum()) if total else 0
        else:
            rows = [r for r in df if r.get("bot") == bot]
            total = len(rows)
            errors = sum(int(r.get("errors", 0)) for r in rows)
        if total == 0:
            return 1.0
        return 1.0 - errors / total

    def check_trials(self) -> None:
        trials = self.deployer.db.trials("active")
        pending = self.borderline_bucket.pending()
        for trial in trials:
            bot_id = trial["bot_id"]
            deploy_id = trial["deploy_id"]
            router = getattr(self.deployer.bot_db, "router", GLOBAL_ROUTER)
            menace_id = getattr(router, "menace_id", os.getenv("MENACE_ID", ""))
            clause, params = build_scope_clause("bots", Scope.LOCAL, menace_id)
            query = apply_scope("SELECT name FROM bots WHERE id=?", clause)
            row = self.deployer.bot_db.conn.execute(
                query,
                (bot_id, *params),
            ).fetchone()
            name = row[0] if row else str(bot_id)
            roi = self.optimizer._roi(name)
            success = self._success_rate(name)
            if roi < self.config.roi_threshold or success < self.config.success_threshold:
                try:
                    self.deployer.bot_db.update_bot(bot_id, status="disabled")
                except Exception as exc:
                    self.logger.exception("bot disable failed: %s", exc)
                    if self.watchdog:
                        self.watchdog.escalate(f"trial bot disable failed: {exc}")
                    if RAISE_ERRORS:
                        raise
                try:
                    self.deployer.rollback(deploy_id)
                except Exception as exc:
                    self.logger.exception("rollback failed: %s", exc)
                    if self.watchdog:
                        self.watchdog.escalate(f"trial rollback failed: {exc}")
                    if RAISE_ERRORS:
                        raise
                self.deployer.db.update_trial(trial["id"], "failed")
            else:
                self.deployer.db.update_trial(trial["id"], "passed")

            bot_key = str(bot_id)
            if bot_key in pending:
                if (
                    roi >= self.config.roi_threshold
                    and success >= self.config.success_threshold
                ):
                    try:
                        self.borderline_bucket.promote(bot_key)
                        self.logger.info(
                            "promoted borderline trial %s (roi=%.3f, success=%.3f)",
                            name,
                            roi,
                            success,
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception(
                            "borderline promote failed for %s", name
                        )
                else:
                    try:
                        self.borderline_bucket.terminate(bot_key)
                        self.logger.info(
                            "terminated borderline trial %s (roi=%.3f, success=%.3f)",
                            name,
                            roi,
                            success,
                        )
                    except Exception:  # pragma: no cover - best effort
                        self.logger.exception(
                            "borderline terminate failed for %s", name
                        )
                pending.pop(bot_key, None)
            if self.history_db:
                try:
                    self.history_db.add(
                        EvolutionEvent(
                            action="trial_result",
                            before_metric=0.0,
                            after_metric=roi,
                            roi=roi,
                            predicted_roi=0.0,
                        )
                    )
                except Exception as exc:
                    self.logger.exception("history record failed: %s", exc)
                    if self.watchdog:
                        self.watchdog.escalate(f"trial history update failed: {exc}")
                    if RAISE_ERRORS:
                        raise


__all__ = ["TrialMonitor", "TrialConfig"]
