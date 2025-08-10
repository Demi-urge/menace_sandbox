from __future__ import annotations

"""Manage experimental variants and track their performance."""

import asyncio
import logging
from typing import Dict, Iterable, List, Tuple

from .experiment_manager import ExperimentManager, ExperimentResult
from .evolution_history_db import EvolutionHistoryDB
from .workflow_cloner import WorkflowCloner
from . import mutation_logger as MutationLogger
from .mutation_lineage import MutationLineage

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
        metric = 0.0
        try:
            row = self.history_db.conn.execute(
                "SELECT workflow_id, after_metric FROM evolution_history WHERE rowid=?",
                (parent_event_id,),
            ).fetchone()
            if row:
                if row[0] is not None:
                    workflow_id = int(row[0])
                    if self.workflow_cloner:
                        try:
                            self.workflow_cloner._clone(workflow_id)  # type: ignore[attr-defined]
                        except Exception as exc:  # pragma: no cover - best effort
                            logger.warning("workflow clone failed for %s: %s", workflow_id, exc)
                if row[1] is not None:
                    metric = float(row[1])
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
            before_metric=metric,
            after_metric=metric,
            parent_id=parent_event_id,
        )
        return event_id

    # ------------------------------------------------------------------
    def ab_test_branch(
        self, failing_patch_id: int, variant_name: str
    ) -> Tuple[int, List[ExperimentResult]]:
        """Clone the most promising ancestor branch for A/B testing.

        ``failing_patch_id`` identifies the latest patch that performed
        poorly.  We backtrack its lineage using
        :meth:`MutationLineage.backtrack_failed_path` and analyse ROI deltas
        along that path to find the ancestor with the strongest positive ROI
        trend.  That patch is cloned via
        :meth:`MutationLineage.clone_branch_for_ab_test` and fed into the
        existing experiment pipeline.  The resulting mutation event id and
        experiment results are returned.
        """

        lineage: MutationLineage | None = getattr(
            self.experiment_manager, "lineage", None
        )
        if not lineage or not getattr(lineage, "patch_db", None):
            return 0, []

        # Determine ancestor patches with positive ROI trends
        path = lineage.backtrack_failed_path(failing_patch_id)
        if not path:
            return 0, []

        best_patch: int | None = None
        best_delta = float("-inf")
        try:
            with lineage.patch_db._connect() as conn:  # type: ignore[attr-defined]
                for pid in path:
                    row = conn.execute(
                        "SELECT roi_delta FROM patch_history WHERE id=?",
                        (pid,),
                    ).fetchone()
                    if not row:
                        continue
                    delta = float(row[0])
                    if delta > best_delta:
                        best_delta = delta
                        best_patch = pid
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("failed analysing ROI trend: %s", exc)
            return 0, []

        if best_patch is None or best_delta <= 0:
            # no promising ancestor
            return 0, []

        new_patch_id = lineage.clone_branch_for_ab_test(best_patch, variant_name)
        if not new_patch_id:
            return 0, []

        roi_val = 0.0
        results: List[ExperimentResult] = []
        try:
            # run the cloned branch through existing experiments
            results = asyncio.run(
                self.experiment_manager.run_experiments([variant_name])
            )
            roi_val = results[0].roi if results else 0.0
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "failed running experiments for %s: %s", variant_name, exc
            )

        event_id = MutationLogger.log_mutation(
            change=variant_name,
            reason="ab_test_branch",
            trigger="variant_manager",
            performance=roi_val,
            workflow_id=0,
            before_metric=0.0,
            after_metric=roi_val,
            parent_id=best_patch,
        )
        try:
            MutationLogger.record_mutation_outcome(
                event_id,
                after_metric=roi_val,
                roi=roi_val,
                performance=roi_val,
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "failed recording mutation outcome for %s: %s", variant_name, exc
            )

        return event_id, results

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

        # Map each variant to its corresponding event rowid upfront so we can
        # record outcomes even if experiment execution fails.
        name_to_id = {row[1]: row[0] for row in children if row}

        results: List[ExperimentResult] = []
        try:
            results = asyncio.run(
                self.experiment_manager.run_experiments(variants)
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("variant experiments failed: %s", exc)

        if results:
            for res in results:
                event_id = name_to_id.get(res.variant)
                if event_id is None:
                    continue
                try:
                    MutationLogger.record_mutation_outcome(
                        event_id,
                        after_metric=res.roi,
                        roi=res.roi,
                        performance=res.roi,
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning(
                        "failed recording mutation outcome for %s: %s", res.variant, exc
                    )
            comparisons = self.experiment_manager.compare_variants(results)
        else:
            # experiments failed; mark each variant with neutral outcome
            for event_id in name_to_id.values():
                try:
                    MutationLogger.record_mutation_outcome(
                        event_id, after_metric=0.0, roi=0.0, performance=0.0
                    )
                except Exception as exc:  # pragma: no cover - best effort
                    logger.warning(
                        "failed recording mutation outcome: %s", exc
                    )
            comparisons = {}
        return results, comparisons


__all__ = ["VariantManager"]
