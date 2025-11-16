from __future__ import annotations

"""Clone high-ROI workflows from :class:`PathwayDB`.

The :class:`WorkflowCloner` service looks up the best performing
pathways and replicates their workflows by either invoking a genetic
algorithm manager or the :class:`BotCreationBot`. Each clone receives a
small random variation before being launched. Outcomes are stored in
:class:`EvolutionHistoryDB` for later comparison.
"""

import asyncio
import logging
import random
import threading
import time
from contextlib import closing
from typing import Optional, TYPE_CHECKING

logger = logging.getLogger(__name__)

from .neuroplasticity import PathwayDB
from .evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from .bot_planning_bot import PlanningTask
from .database_manager import add_model
from . import mutation_logger as MutationLogger

if TYPE_CHECKING:  # pragma: no cover - optional dependencies
    from .ga_clone_manager import GALearningManager
    from .bot_creation_bot import BotCreationBot


class WorkflowCloner:
    """Monitor PathwayDB and clone the best workflows."""

    def __init__(
        self,
        pathway_db: PathwayDB | None = None,
        *,
        ga_manager: GALearningManager | None = None,
        bot_creator: BotCreationBot | None = None,
        history_db: EvolutionHistoryDB | None = None,
        interval: int = 3600,
    ) -> None:
        self.db = pathway_db or PathwayDB()
        self.ga_manager = ga_manager
        self.bot_creator = bot_creator
        self.history = history_db or EvolutionHistoryDB()
        self.interval = interval
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._last_event_ids: dict[int, int] = {}

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while self.running:
            try:
                self.clone_top_workflows()
            except Exception:
                logger.exception("clone cycle failed")
            time.sleep(self.interval)

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=0)
            self._thread = None

    # ------------------------------------------------------------------
    def _variation(self, tasks: list[str]) -> list[str]:
        var = tasks[:]
        if len(var) > 1:
            i, j = random.sample(range(len(var)), 2)
            var[i], var[j] = var[j], var[i]
        return var

    def _clone(self, pid: int) -> None:
        with closing(self.db.conn.cursor()) as cur:
            cur.execute(
                "SELECT actions FROM pathways WHERE id=?",
                (pid,),
            )
            row = cur.fetchone()
        if not row:
            return
        actions = row[0]
        tasks = [a.strip() for a in actions.split("->") if a.strip()]
        if not tasks:
            return
        with closing(self.db.conn.cursor()) as cur:
            cur.execute(
                "SELECT avg_roi FROM metadata WHERE pathway_id=?",
                (pid,),
            )
            before_row = cur.fetchone()
        before = float(before_row[0] or 0.0) if before_row else 0.0
        after = before
        if self.ga_manager:
            try:
                res = self.ga_manager.run_cycle(tasks[0])
                after = float(res.roi)
            except Exception:
                logger.exception("ga run failed")
                after = before
        if self.bot_creator:
            try:
                model_id = add_model(f"clone_{pid}", source="workflow_clone")
                plans = [
                    PlanningTask(
                        description=t,
                        complexity=1.0,
                        frequency=1.0,
                        expected_time=1.0,
                        actions=[t],
                    )
                    for t in self._variation(tasks)
                ]
                asyncio.run(
                    self.bot_creator.create_bots(plans, model_id=model_id)
                )
            except Exception:
                logger.exception("bot creation failed")
        parent = self._last_event_ids.get(pid)
        event_id = self.history.add(
            EvolutionEvent(
                action="workflow_clone",
                before_metric=before,
                after_metric=after,
                roi=after - before,
                workflow_id=pid,
                reason="clone top workflow",
                trigger="top_pathways",
                performance=after - before,
                parent_event_id=parent,
            )
        )
        MutationLogger.log_mutation(
            change="workflow_clone",
            reason="clone top workflow",
            trigger="top_pathways",
            performance=after - before,
            workflow_id=pid,
            before_metric=before,
            after_metric=after,
            parent_id=event_id,
        )
        self._last_event_ids[pid] = event_id

    def clone_top_workflows(self, limit: int = 3) -> None:
        top = self.db.top_pathways(limit=limit)
        for pid, _score in top:
            self._clone(pid)


__all__ = ["WorkflowCloner"]
