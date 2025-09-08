from __future__ import annotations

"""Automated experiment manager for testing bot variants."""

import asyncio
from dataclasses import dataclass
from typing import Iterable, Dict, List
import math
import logging
from scipy import stats

from .mutation_lineage import MutationLineage

logger = logging.getLogger(__name__)


from .model_automation_pipeline import ModelAutomationPipeline, AutomationResult
try:  # pragma: no cover - fallback for light imports in tests
    from vector_service.context_builder import ContextBuilder
except Exception:  # pragma: no cover - best effort
    from vector_service import ContextBuilder  # type: ignore
from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from .prediction_manager_bot import PredictionManager
from .experiment_history_db import ExperimentHistoryDB, ExperimentLog, TestLog


@dataclass
class ExperimentResult:
    variant: str
    roi: float
    metrics: Dict[str, float]
    sample_size: int = 1
    variance: float = 0.0


class ExperimentManager:
    """Deploy alternative bot implementations and measure performance."""

    def __init__(
        self,
        data_bot: DataBot,
        capital_bot: CapitalManagementBot,
        pipeline: ModelAutomationPipeline | None = None,
        *,
        context_builder: ContextBuilder,
        prediction_manager: PredictionManager | None = None,
        experiment_db: ExperimentHistoryDB | None = None,
        p_threshold: float = 0.05,
        lineage: MutationLineage | None = None,
    ) -> None:
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.context_builder = context_builder
        if pipeline is None:
            self.pipeline = ModelAutomationPipeline(
                data_bot=self.data_bot,
                capital_manager=self.capital_bot,
                context_builder=context_builder,
            )
        else:
            if not hasattr(pipeline, "context_builder"):
                raise ValueError("pipeline must expose a context_builder")
            if pipeline.context_builder is not context_builder:
                raise ValueError("pipeline and context_builder must match")
            self.pipeline = pipeline
        self.prediction_manager = prediction_manager
        self.experiment_db = experiment_db or ExperimentHistoryDB()
        self.p_threshold = p_threshold
        self.lineage = lineage or MutationLineage()

    async def _run_variant(self, name: str, energy: int) -> AutomationResult:
        try:
            self.context_builder.build_context(name)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("context retrieval failed for %s: %s", name, exc)
        return await asyncio.to_thread(self.pipeline.run, name, energy=energy)

    # ------------------------------------------------------------------
    # lineage helpers
    def backtrack_failed_path(self, patch_id: int) -> List[int]:
        """Return patch lineage from ``patch_id`` back to last successful patch."""
        try:
            return self.lineage.backtrack_failed_path(patch_id)
        except Exception as exc:
            logger.warning("failed backtracking lineage: %s", exc)
            return []

    def clone_branch_for_ab_test(
        self,
        patch_id: int,
        description: str,
        vectors: List[tuple[str, float]] | None = None,
    ) -> int | None:
        """Clone a patch branch for A/B testing and return new patch id."""
        try:
            return self.lineage.clone_branch_for_ab_test(patch_id, description, vectors)
        except Exception as exc:
            logger.warning("failed cloning branch for A/B test: %s", exc)
            return None

    async def run_experiments(self, variants: Iterable[str], energy: int = 1) -> List[ExperimentResult]:
        results: List[ExperimentResult] = []
        tasks = [self._run_variant(v, energy) for v in variants]
        outs = await asyncio.gather(*tasks, return_exceptions=True)
        for name, res in zip(variants, outs):
            if isinstance(res, Exception):
                continue
            roi_val = res.roi.roi if res.roi else 0.0
            metrics = {
                "cpu": 0.0,
                "memory": 0.0,
            }
            try:
                df = self.data_bot.db.fetch(limit=1)
                if not getattr(df, "empty", True):
                    metrics["cpu"] = float(df["cpu"].iloc[0])
                    metrics["memory"] = float(df["memory"].iloc[0])
            except Exception as exc:
                logger.warning("failed fetching resource metrics: %s", exc)
            try:
                prev_vals = self.experiment_db.variant_values(name)
            except Exception as exc:
                logger.warning("failed retrieving previous ROI for %s: %s", name, exc)
                prev_vals = []
            sample_vals = prev_vals + [roi_val]
            from statistics import variance

            n = len(sample_vals)
            var = variance(sample_vals) if n > 1 else 0.0
            exp_res = ExperimentResult(
                variant=name,
                roi=roi_val,
                metrics=metrics,
                sample_size=n,
                variance=var,
            )
            results.append(exp_res)
            try:
                self.experiment_db.add(
                    ExperimentLog(
                        variant=name,
                        roi=roi_val,
                        cpu=metrics["cpu"],
                        memory=metrics["memory"],
                    )
                )
            except Exception as exc:
                logger.warning("failed inserting experiment log for %s: %s", name, exc)
            # detailed logging for bot experiments
            prev = prev_vals[-1] if prev_vals else 0.0
            change_desc = roi_val - prev
            logger.info(
                "bot_variant=%s change=%.4f reason=%s trigger=%s parent=%s",
                name,
                change_desc,
                "experiment",
                "experiment",
                None,
            )
        return results

    # ------------------------------------------------------------------
    async def run_experiments_from_parent(
        self, parent_event_id: int, energy: int = 1
    ) -> List[ExperimentResult]:
        """Run experiments for variants branching from ``parent_event_id``."""

        if not self.lineage or not getattr(self.lineage, "history_db", None):
            return []
        try:
            tree = self.lineage.history_db.subtree(parent_event_id)  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("failed building lineage subtree: %s", exc)
            return []
        if not tree:
            return []
        variants = [c["action"] for c in tree.get("children", []) if c.get("action")]
        if not variants:
            return []
        return await self.run_experiments(variants, energy=energy)

    def compare_variants(self, results: Iterable[ExperimentResult]) -> Dict[tuple[str, str], tuple[float, float]]:
        """Return t-statistics and p-values comparing ROI across variants."""
        comps: Dict[tuple[str, str], tuple[float, float]] = {}
        res_list = list(results)
        for i in range(len(res_list)):
            for j in range(i + 1, len(res_list)):
                a = res_list[i]
                b = res_list[j]
                if a.sample_size and b.sample_size:
                    t_stat, p_val = stats.ttest_ind_from_stats(
                        a.roi,
                        math.sqrt(a.variance),
                        a.sample_size,
                        b.roi,
                        math.sqrt(b.variance),
                        b.sample_size,
                        equal_var=False,
                    )
                    comps[(a.variant, b.variant)] = (float(t_stat), float(p_val))
                    try:
                        self.experiment_db.add_test(
                            TestLog(
                                variant_a=a.variant,
                                variant_b=b.variant,
                                t_stat=float(t_stat),
                                p_value=float(p_val),
                            )
                        )
                    except Exception as exc:
                        logger.warning(
                            "failed inserting test log for %s vs %s: %s",
                            a.variant,
                            b.variant,
                            exc,
                        )
        return comps

    def best_variant(self, results: Iterable[ExperimentResult]) -> ExperimentResult | None:
        res_list = list(results)
        if not res_list:
            return None
        res_list.sort(key=lambda r: r.roi, reverse=True)
        best = res_list[0]
        if len(res_list) == 1:
            return best
        comps = self.compare_variants(res_list)
        for other in res_list[1:]:
            pair = (best.variant, other.variant)
            if pair not in comps:
                continue
            t_stat, p_val = comps[pair]
            if p_val >= self.p_threshold or t_stat <= 0:
                return None
        return best

    async def run_suggested_experiments(self, bot_name: str, energy: int = 1) -> ExperimentResult | None:
        """Run experiments for variants suggested by PredictionManager."""
        if not self.prediction_manager:
            return None
        variants = self.prediction_manager.get_prediction_bots_for(bot_name)
        if not variants:
            return None
        results = await self.run_experiments(variants, energy=energy)
        best = self.best_variant(results)
        if best:
            try:
                from .metrics_exporter import experiment_best_roi

                if experiment_best_roi:
                    experiment_best_roi.set(best.roi)
            except Exception as exc:
                logger.warning("failed exporting best ROI metric: %s", exc)
        return best


__all__ = [
    "ExperimentManager",
    "ExperimentResult",
]
