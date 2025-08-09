from __future__ import annotations

"""Manage experimental variants and track their performance."""

import asyncio
import logging
from typing import Dict, Iterable, List, Tuple

from .experiment_manager import ExperimentManager, ExperimentResult
from .evolution_history_db import EvolutionHistoryDB
from .workflow_cloner import WorkflowCloner
from . import mutation_logger as MutationLogger

logger = logging.getLogger(__name__)


class VariantManager:
    """Spawn and compare workflow or bot variants."""

    def __init__(
        self,
        experiment_manager: ExperimentManager,
        *,
        history_db: EvolutionHistoryDB | None = None,
        workflow_cloner: WorkflowCloner | None = None,
    ) -> None:
        self.experiment_manager = experiment_manager
        self.history_db = history_db or EvolutionHistoryDB()
        self.workflow_cloner = workflow_cloner

    # ------------------------------------------------------------------
    def spawn_variant(self, parent_event_id: int, variant_name: str) -> int:
        """Clone configuration from *parent_event_id* and log a new branch.

        Returns the mutation event id for the new variant.
        """
        workflow_id: int | None = None
        try:
            row = self.history_db.conn.execute(
                "SELECT workflow_id FROM evolution_history WHERE rowid=?",
                (parent_event_id,),
            ).fetchone()
            if row and row[0] is not None:
                workflow_id = int(row[0])
                if self.workflow_cloner:
                    try:
                        self.workflow_cloner._clone(workflow_id)  # type: ignore[attr-defined]
                    except Exception as exc:  # pragma: no cover - best effort
                        logger.warning("workflow clone failed for %s: %s", workflow_id, exc)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "failed retrieving workflow for parent %s: %s", parent_event_id, exc
            )
        event_id = MutationLogger.log_mutation(
            change=variant_name,
            reason="spawn_variant",
            trigger="variant_manager",
            performance=0.0,
            workflow_id=workflow_id or 0,
            parent_id=parent_event_id,
        )
        return event_id

    # ------------------------------------------------------------------
    def compare_variants(
        self, parent_event_id: int
    ) -> Tuple[List[ExperimentResult], Dict[Tuple[str, str], Tuple[float, float]]]:
        """Run experiments for all variants branching from *parent_event_id*.

        Returns experiment results and pairwise comparison statistics.
        """
        children = self.history_db.fetch_children(parent_event_id)
        variants = [row[1] for row in children if row]
        if not variants:
            return [], {}
        results = asyncio.run(
            self.experiment_manager.run_experiments(variants)
        )
        comparisons = self.experiment_manager.compare_variants(results)
        return results, comparisons


__all__ = ["VariantManager"]
