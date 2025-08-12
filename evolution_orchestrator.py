from __future__ import annotations

"""Orchestrate system evolution based on metrics and capital signals."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
import time

import numpy as np

from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from .self_improvement_engine import SelfImprovementEngine
from .growth_utils import growth_score
from .system_evolution_manager import SystemEvolutionManager, EvolutionCycleResult
from .evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from .evaluation_history_db import EvaluationHistoryDB
from .trend_predictor import TrendPredictor, TrendPrediction
from .bot_creation_bot import BotCreationBot
from .resource_allocation_optimizer import ResourceAllocationOptimizer
from .workflow_evolution_bot import WorkflowEvolutionBot
from .experiment_manager import ExperimentManager
from .evolution_analysis_bot import EvolutionAnalysisBot
from . import mutation_logger as MutationLogger
from .adaptive_roi_predictor import AdaptiveROIPredictor


@dataclass
class EvolutionTrigger:
    """Trigger thresholds for evolution."""

    error_rate: float = 0.1
    roi_drop: float = -0.1
    energy_threshold: float = 0.3


class EvolutionOrchestrator:
    """Monitor metrics and coordinate improvement and evolution cycles."""

    def __init__(
        self,
        data_bot: DataBot,
        capital_bot: CapitalManagementBot,
        improvement_engine: SelfImprovementEngine,
        evolution_manager: SystemEvolutionManager,
        *,
        history_db: EvolutionHistoryDB | None = None,
        triggers: EvolutionTrigger | None = None,
        bot_creator: BotCreationBot | None = None,
        resource_optimizer: ResourceAllocationOptimizer | None = None,
        workflow_evolver: WorkflowEvolutionBot | None = None,
        experiment_manager: ExperimentManager | None = None,
        analysis_bot: EvolutionAnalysisBot | None = None,
        trend_predictor: TrendPredictor | None = None,
        predictor: EvolutionPredictor | None = None,
        multi_predictor: object | None = None,
        event_bus: UnifiedEventBus | None = None,
        roi_predictor: AdaptiveROIPredictor | None = None,
        dataset_path: str | Path = "roi_eval_dataset.csv",
        retrain_interval: int = 10,
    ) -> None:
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.improvement_engine = improvement_engine
        self.evolution_manager = evolution_manager
        self.history = history_db or EvolutionHistoryDB()
        self.triggers = triggers or EvolutionTrigger()
        self.bot_creator = bot_creator
        self.resource_optimizer = resource_optimizer
        self.workflow_evolver = workflow_evolver
        self.experiment_manager = experiment_manager
        self.analysis_bot = analysis_bot
        self.predictor = predictor
        self.multi_predictor = multi_predictor
        self.trend_predictor = trend_predictor
        self.event_bus = event_bus
        self.roi_predictor = roi_predictor or AdaptiveROIPredictor()
        self.dataset_path = Path(dataset_path)
        self.retrain_interval = retrain_interval
        if self.capital_bot and getattr(self.capital_bot, "trend_predictor", None) is None:
            try:
                self.capital_bot.trend_predictor = trend_predictor
            except Exception:
                self.logger.exception("failed to set trend predictor")
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("EvolutionOrchestrator")
        self.prev_roi = self._latest_roi()
        self._cycles = 0
        self._last_workflow_benchmark = 0.0
        self._benchmark_interval = 3600
        self._workflow_roi_history: dict[str, list[float]] = {}
        self._last_mutation_id: int | None = None
        self._workflow_event_ids: dict[int | str, int] = {}
        if not self.dataset_path.exists():
            try:
                self.dataset_path.write_text(
                    "before_roi,error_rate,eval_score,actual_roi,predicted_roi,error\n",
                    encoding="utf-8",
                )
            except Exception:
                pass
        if self.event_bus:
            try:
                self.event_bus.subscribe("evolve:system", lambda *_: self.run_cycle())
            except Exception:
                self.logger.exception("event bus subscription failed")

    # ------------------------------------------------------------------
    def _latest_roi(self) -> float:
        try:
            df = self.data_bot.db.fetch(limit=50)
            if getattr(df, "empty", True):
                return 0.0
            if hasattr(df, "sum"):
                revenue = float(df["revenue"].sum())
                expense = float(df["expense"].sum())
            else:
                revenue = sum(r.get("revenue", 0.0) for r in df)
                expense = sum(r.get("expense", 0.0) for r in df)
            return revenue - expense
        except Exception:
            return 0.0

    def _error_rate(self) -> float:
        try:
            df = self.data_bot.db.fetch(limit=50)
            if getattr(df, "empty", True):
                return 0.0
            if hasattr(df, "mean"):
                return float(df["errors"].mean() or 0.0)
            return float(sum(r.get("errors", 0.0) for r in df) / len(df))
        except Exception:
            return 0.0

    def _latest_eval_score(self) -> float:
        """Return the most recent evaluation score for the improvement engine."""

        try:
            eng = getattr(self.improvement_engine, "bot_name", None)
            if not eng:
                return 0.0
            db = EvaluationHistoryDB()
            hist = db.history(eng, limit=1)
            if hist:
                return float(hist[0][0])
        except Exception:
            pass
        return 0.0

    def _append_dataset(
        self,
        before: float,
        err_rate: float,
        eval_score: float,
        actual: float,
        predicted: float,
    ) -> None:
        """Persist a single training sample to the unified dataset."""

        try:
            error = actual - predicted
            line = f"{before},{err_rate},{eval_score},{actual},{predicted},{error}\n"
            with self.dataset_path.open("a", encoding="utf-8") as fh:
                fh.write(line)
            self.logger.info(
                "roi_prediction_error",
                extra={"predicted_roi": predicted, "actual_roi": actual, "error": error},
            )
        except Exception:
            self.logger.exception("failed to append ROI dataset")

    def _load_dataset(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Load the accumulated dataset from ``dataset_path``."""

        try:
            data = np.loadtxt(self.dataset_path, delimiter=",", skiprows=1)
        except Exception:
            return None
        if data.size == 0:
            return None
        if data.ndim == 1:
            data = data[None, :]
        X = data[:, :3]
        y = data[:, 3]
        return X, y

    # ------------------------------------------------------------------
    def run_cycle(self) -> None:
        """Check triggers and run appropriate evolution steps."""
        if self.analysis_bot and hasattr(self.analysis_bot, "train"):
            try:
                self.analysis_bot.train()
            except Exception:
                self.logger.exception("analysis training failed")
        before_roi = self._latest_roi()
        delta_roi = before_roi - self.prev_roi
        self.prev_roi = before_roi
        error_rate = self._error_rate()
        try:
            seq, _, _, _ = self.roi_predictor.predict(
                [[before_roi, error_rate]], horizon=1
            )
            model_pred = float(seq[-1]) if seq else 0.0
        except TypeError:
            model_pred, _, _, _ = self.roi_predictor.predict([[before_roi, error_rate]])
        pred_roi = before_roi
        pred_err = error_rate
        if self.trend_predictor:
            try:
                pred = self.trend_predictor.predict_future_metrics(3)
                pred_roi = pred.roi
                pred_err = pred.errors
            except Exception:
                pred_roi = before_roi
                pred_err = error_rate
        energy = self.capital_bot.energy_score(
            load=0.0,
            success_rate=1.0,
            deploy_eff=1.0,
            failure_rate=error_rate,
        )
        result_roi = before_roi
        close = lambda val, thr: abs(val - thr) <= abs(thr) * 0.1
        candidates: list[str] = []
        action_reasons: dict[str, list[str]] = {}
        action_triggers: dict[str, list[str]] = {}

        # determine triggers for self improvement
        sim_reasons: list[str] = []
        sim_triggers: list[str] = []
        if error_rate > self.triggers.error_rate:
            sim_reasons.append(
                f"error_rate {error_rate:.2f} > {self.triggers.error_rate:.2f}"
            )
            sim_triggers.append("error_rate")
        if pred_err > self.triggers.error_rate:
            sim_reasons.append(
                f"pred_error_rate {pred_err:.2f} > {self.triggers.error_rate:.2f}"
            )
            sim_triggers.append("pred_error_rate")
        if close(error_rate, self.triggers.error_rate):
            sim_reasons.append(
                f"error_rate {error_rate:.2f} ~ {self.triggers.error_rate:.2f}"
            )
            sim_triggers.append("error_rate")
        if sim_reasons:
            candidates.append("self_improvement")
            action_reasons["self_improvement"] = sim_reasons
            action_triggers["self_improvement"] = sim_triggers

        # determine triggers for system evolution
        sys_reasons: list[str] = []
        sys_triggers: list[str] = []
        if delta_roi <= self.triggers.roi_drop:
            sys_reasons.append(
                f"roi_drop {delta_roi:.2f} <= {self.triggers.roi_drop:.2f}"
            )
            sys_triggers.append("roi_drop")
        if pred_roi - before_roi <= self.triggers.roi_drop:
            sys_reasons.append(
                f"pred_roi_drop {pred_roi - before_roi:.2f} <= {self.triggers.roi_drop:.2f}"
            )
            sys_triggers.append("pred_roi_drop")
        if energy < self.triggers.energy_threshold:
            sys_reasons.append(
                f"energy {energy:.2f} < {self.triggers.energy_threshold:.2f}"
            )
            sys_triggers.append("energy")
        if close(delta_roi, self.triggers.roi_drop):
            sys_reasons.append(
                f"roi_drop {delta_roi:.2f} ~ {self.triggers.roi_drop:.2f}"
            )
            sys_triggers.append("roi_drop")
        if close(energy, self.triggers.energy_threshold):
            sys_reasons.append(
                f"energy {energy:.2f} ~ {self.triggers.energy_threshold:.2f}"
            )
            sys_triggers.append("energy")
        if sys_reasons:
            candidates.append("system_evolution")
            action_reasons["system_evolution"] = sys_reasons
            action_triggers["system_evolution"] = sys_triggers

        # determine triggers for bot creation
        bc_reasons: list[str] = []
        bc_triggers: list[str] = []
        if delta_roi > abs(self.triggers.roi_drop) and self.bot_creator:
            bc_reasons.append(
                f"roi_increase {delta_roi:.2f} > {abs(self.triggers.roi_drop):.2f}"
            )
            bc_triggers.append("roi_increase")
        if bc_reasons:
            candidates.append("bot_creation")
            action_reasons["bot_creation"] = bc_reasons
            action_triggers["bot_creation"] = bc_triggers

        candidates = list(dict.fromkeys(candidates))
        predictions: dict[str, float] = {}
        variances: dict[str, float] = {}
        sequence: list[str] = []
        predicted_action_roi = 0.0
        if self.multi_predictor and hasattr(self.multi_predictor, "predict"):
            best_score = float("-inf")
            best_act: str | None = None
            for cand in candidates:
                try:
                    mean, var = self.multi_predictor.predict(cand, before_roi)
                except Exception:
                    mean = var = 0.0
                predictions[cand] = mean
                variances[cand] = var
                score = mean - var
                if score > best_score:
                    best_score = score
                    best_act = cand
                    predicted_action_roi = mean
            if best_act:
                sequence = [best_act]
        else:
            if self.analysis_bot:
                for cand in candidates:
                    try:
                        predictions[cand] = self.analysis_bot.predict(cand, before_roi)
                    except Exception:
                        predictions[cand] = 0.0
            if self.predictor:
                for cand in candidates:
                    try:
                        val = self.predictor.predict(cand, before_roi)
                        predictions[cand] = max(predictions.get(cand, 0.0), val)
                    except Exception:
                        predictions[cand] = predictions.get(cand, 0.0)
            if candidates:
                sequences = [[c] for c in candidates]
                if self.multi_predictor and len(candidates) > 1:
                    import itertools

                    for pair in itertools.permutations(candidates, 2):
                        sequences.append(list(pair))
                scores: dict[str, float] = {}
                for seq in sequences:
                    key = "->".join(seq)
                    base = sum(predictions.get(a, 0.0) for a in seq)
                    prob = 1.0
                    if self.multi_predictor and hasattr(self.multi_predictor, "predict_success"):
                        try:
                            prob = float(
                                self.multi_predictor.predict_success(
                                    1.0, 1.0, before_roi, 1.0, key
                                )
                            )
                        except Exception:
                            prob = 1.0
                    scores[key] = base * prob
                if scores:
                    best = max(scores, key=scores.get)
                    sequence = best.split("->")
                    predicted_action_roi = scores[best]
                else:
                    sequence = [candidates[0]]
                    predicted_action_roi = predictions.get(sequence[0], 0.0)
        result_values: list[float] = []
        trending_topic: str | None = None
        for act in sequence:
            if act == "self_improvement":
                self.logger.info(
                    "Triggering self improvement due to errors %.2f", error_rate
                )
                res = self.improvement_engine.run_cycle()
                trending_topic = getattr(res, "trending_topic", trending_topic)
                if res.roi:
                    result_values.append(res.roi.roi)
            elif act == "system_evolution":
                self.logger.info(
                    "Triggering system evolution due to performance drop %.2f",
                    delta_roi,
                )
                res = self.evolution_manager.run_cycle()
                trending_topic = getattr(res, "trending_topic", trending_topic)
                rois = list(res.ga_results.values())
                if rois:
                    result_values.append(sum(rois) / len(rois))
            elif act == "bot_creation":
                self.logger.info(
                    "Launching bot creation due to ROI increase %.2f", delta_roi
                )
                try:
                    from .bot_planning_bot import PlanningTask
                    import asyncio

                    task = PlanningTask(
                        description="growth",
                        complexity=1,
                        frequency=1,
                        expected_time=1.0,
                        actions=["run"],
                    )
                    asyncio.run(self.bot_creator.create_bots([task]))
                except Exception as exc:
                    self.logger.error("bot creation failed: %s", exc)
        if self.resource_optimizer:
            try:
                w_names = self.resource_optimizer.available_workflows()
                self.resource_optimizer.update_priorities(
                    self.evolution_manager.bots,
                    workflows=w_names,
                    metrics_db=self.data_bot.db,
                    prune_threshold=0.0,
                )
            except Exception:
                self.logger.exception("resource optimizer update failed")

        if sequence:
            action_seq = "->".join(sequence)
            result_roi = (
                sum(result_values) / len(result_values) if result_values else before_roi
            )
            after_roi = self._latest_roi()

            reason_parts: list[str] = []
            trigger_metrics: list[str] = []
            for act in sequence:
                rs = action_reasons.get(act, [])
                if rs:
                    reason_parts.append(f"{act}:{' & '.join(rs)}")
                trigger_metrics.extend(action_triggers.get(act, []))
            reason_str = "; ".join(reason_parts)
            trigger_str = ",".join(dict.fromkeys(trigger_metrics))

            workflow_id = 0
            parent_event_id = self._workflow_event_ids.get(workflow_id)
            event = EvolutionEvent(
                action=action_seq,
                before_metric=before_roi,
                after_metric=after_roi,
                roi=result_roi,
                predicted_roi=predicted_action_roi,
                ts=datetime.utcnow().isoformat(),
                trending_topic=trending_topic,
                reason=reason_str,
                trigger=trigger_str,
                performance=after_roi - before_roi,
                parent_event_id=parent_event_id,
                workflow_id=workflow_id,
            )
            event_id = self.history.add(event)
            self._workflow_event_ids[workflow_id] = event_id
            try:
                eff = bottleneck = 0.0
                try:
                    df = self.data_bot.db.fetch(20)
                    if hasattr(df, "empty"):
                        if not getattr(df, "empty", True):
                            eff = float(max(0.0, 100.0 - df["cpu"].mean()))
                            if "errors" in df.columns:
                                bottleneck = float(df["errors"].mean())
                    elif isinstance(df, list) and df:
                        avg_cpu = sum(r.get("cpu", 0.0) for r in df) / len(df)
                        eff = float(max(0.0, 100.0 - avg_cpu))
                        bottleneck = float(
                            sum(r.get("errors", 0.0) for r in df) / len(df)
                        )
                except Exception:
                    eff = bottleneck = 0.0
                self.data_bot.log_evolution_cycle(
                    action_seq,
                    before_roi,
                    after_roi,
                    result_roi,
                    predicted_action_roi,
                    roi_delta=after_roi - before_roi,
                    efficiency=eff,
                    bottleneck=bottleneck,
                    trending_topic=trending_topic,
                    reason=reason_str,
                    trigger=trigger_str,
                    parent_event_id=event_id,
                )
                if self.capital_bot:
                    try:
                        self.capital_bot.log_evolution_event(
                            action_seq,
                            before_roi,
                            after_roi,
                        )
                    except Exception:
                        self.logger.exception("capital event log failed")
            except Exception:
                self.logger.exception("evolution cycle logging failed")
            try:
                from .metrics_exporter import evolution_cycle_count

                if evolution_cycle_count:
                    evolution_cycle_count.inc()
            except Exception:
                self.logger.exception("metrics export failed")
            with MutationLogger.log_context(
                change=action_seq,
                reason=reason_str,
                trigger=trigger_str,
                workflow_id=0,
                before_metric=before_roi,
                parent_id=event_id,
            ) as mutation:
                mutation["after_metric"] = after_roi
                mutation["performance"] = after_roi - before_roi
                mutation["roi"] = result_roi
            self._last_mutation_id = int(mutation["event_id"])
        self._run_bot_experiments()
        self._run_workflow_experiments()
        self._cycles += 1
        if self._cycles % 10 == 0:
            self._cleanup_workflows()
        if self.predictor:
            try:
                self.predictor.train()
            except Exception:
                self.logger.exception("predictor training failed")
        if self.multi_predictor and hasattr(self.multi_predictor, "train"):
            try:
                self.multi_predictor.train()
            except Exception:
                self.logger.exception("multi predictor training failed")
        final_roi = self._latest_roi()
        eval_score = self._latest_eval_score()
        self._append_dataset(before_roi, error_rate, eval_score, final_roi, model_pred)
        if self._cycles % self.retrain_interval == 0:
            dataset = self._load_dataset()
            if dataset is not None:
                try:
                    self.roi_predictor.train(dataset)
                except Exception:
                    self.logger.exception("roi predictor retrain failed")
        
    # ------------------------------------------------------------------
    def _run_workflow_experiments(self, limit: int = 3) -> None:
        """Propose alternative workflows and optionally benchmark them."""
        if not self.workflow_evolver:
            return
        try:
            suggestions = list(self.workflow_evolver.analyse(limit))
        except Exception:
            suggestions = []
        scored: list[tuple[str, float, str]] = []
        workflow_id = 0
        parent_event_id = self._workflow_event_ids.get(workflow_id)
        for s in suggestions:
            seq = "-".join(reversed(s.sequence.split("-")))
            features = [[s.expected_roi, 0.0]]
            try:
                try:
                    seq_preds, category, _, _ = self.roi_predictor.predict(
                        features, horizon=len(features)
                    )
                except TypeError:
                    val, category, _, _ = self.roi_predictor.predict(features)
                    seq_preds = [float(val)]
                roi_est = float(seq_preds[-1]) if seq_preds else s.expected_roi
            except Exception:
                roi_est, category = s.expected_roi, "unknown"
            self.logger.info(
                "workflow candidate",
                extra={"workflow": seq, "roi_category": category, "roi_estimate": roi_est},
            )
            try:
                event_id = MutationLogger.log_mutation(
                    change=seq,
                    reason="rearrangement",
                    trigger="workflow_evolution_bot",
                    performance=0.0,
                    workflow_id=workflow_id,
                    parent_id=parent_event_id,
                )
                if hasattr(self.workflow_evolver, "_rearranged_events"):
                    self.workflow_evolver._rearranged_events[seq] = event_id
            except Exception:
                pass
            scored.append((seq, roi_est, category))
        scored = [s for s in scored if s[2] == "exponential" or s[1] > 0]
        scored.sort(key=lambda x: (-growth_score(x[2]), -x[1]))
        proposals = [s[0] for s in scored]
        main_wf = None
        if self.resource_optimizer and (
            time.time() - self._last_workflow_benchmark >= self._benchmark_interval
        ):
            try:
                names = self.resource_optimizer.available_workflows()
                if names:
                    main_wf = names[0]
                    if main_wf not in proposals:
                        proposals.append(main_wf)
            except Exception:
                main_wf = None
        base_roi = self._latest_roi()
        results: list = []
        if self.experiment_manager and proposals:
            try:
                import asyncio

                results = asyncio.run(self.experiment_manager.run_experiments(proposals))
            except Exception:
                self.logger.exception("workflow experiments failed")
        if not results:
            for name in proposals:
                try:
                    wf_key = name
                    parent = self._workflow_event_ids.get(wf_key)
                    event_id = self.history.add(
                        EvolutionEvent(
                            action=f"experiment:{name}",
                            before_metric=base_roi,
                            after_metric=base_roi,
                            roi=0.0,
                            trending_topic=None,
                            reason="workflow experiment",
                            trigger="experiment",
                            performance=0.0,
                            parent_event_id=parent,
                            workflow_id=wf_key if isinstance(wf_key, int) else None,
                        )
                    )
                    self._workflow_event_ids[wf_key] = event_id
                    MutationLogger.record_mutation_outcome(
                        event_id, after_metric=base_roi, roi=0.0, performance=0.0
                    )
                except Exception:
                    self.logger.exception("record experiment failed")
            return
        self._last_workflow_benchmark = time.time()
        try:
            best = self.experiment_manager.best_variant(results)
        except Exception:
            best = None
        try:
            mdb = getattr(self.resource_optimizer, "menace_db", None)
            if mdb:
                with mdb.engine.begin() as conn:
                    for res in results:
                        row = (
                            conn.execute(
                                mdb.workflows.select().where(
                                    mdb.workflows.c.workflow_name == res.variant
                                )
                            )
                            .mappings()
                            .fetchone()
                        )
                        if row:
                            conn.execute(
                                mdb.workflows.update()
                                .where(mdb.workflows.c.workflow_id == row["workflow_id"])
                                .values(estimated_profit_per_bot=res.roi - base_roi)
                            )
                    if best:
                        row = (
                            conn.execute(
                                mdb.workflows.select().where(
                                    mdb.workflows.c.workflow_name == best.variant
                                )
                            )
                            .mappings()
                            .fetchone()
                        )
                        if row:
                            conn.execute(
                                mdb.workflows.update()
                                .where(mdb.workflows.c.workflow_id == row["workflow_id"])
                                .values(status="winner")
                            )
        except Exception:
            self.logger.exception("workflow DB update failed")
        for res in results:
            try:
                wf_key = getattr(res, "workflow_id", res.variant)
                parent = self._workflow_event_ids.get(wf_key)
                change = res.roi - base_roi
                event_id = self.history.add(
                    EvolutionEvent(
                        action=f"experiment:{res.variant}",
                        before_metric=base_roi,
                        after_metric=res.roi,
                        roi=change,
                        trending_topic=getattr(res, "trending_topic", None),
                        reason="workflow experiment",
                        trigger="experiment",
                        performance=change,
                        parent_event_id=parent,
                        workflow_id=wf_key if isinstance(wf_key, int) else None,
                    )
                )
                self._workflow_event_ids[wf_key] = event_id
                MutationLogger.record_mutation_outcome(
                    event_id, after_metric=res.roi, roi=change, performance=change
                )
                # detailed experiment logging
                self.logger.info(
                    "workflow_variant=%s change=%.4f reason=%s trigger=%s parent=%s",
                    res.variant,
                    change,
                    "experiment",
                    "experiment",
                    parent,
                )
                vals = self._workflow_roi_history.setdefault(res.variant, [])
                vals.append(res.roi)
                if len(vals) > 5:
                    vals.pop(0)
            except Exception:
                self.logger.exception("record experiment failed")

        if main_wf and main_wf in self._workflow_roi_history:
            main_avg = sum(self._workflow_roi_history[main_wf]) / len(
                self._workflow_roi_history[main_wf]
            )
            for wf, vals in self._workflow_roi_history.items():
                if wf == main_wf or len(vals) < 3:
                    continue
                avg = sum(vals) / len(vals)
                if avg > main_avg * 1.05:
                    parent = self._workflow_event_ids.get(wf)
                    change_desc = f"avg {avg:.4f} > main {main_avg:.4f}"
                    self.logger.info(
                        "workflow_variant=%s change=%s reason=%s trigger=%s parent=%s",
                        wf,
                        change_desc,
                        "benchmark",
                        "benchmark",
                        parent,
                    )
                    self._replace_main_workflow(wf)
                    main_wf = wf
                    main_avg = avg

    # ------------------------------------------------------------------
    def _run_bot_experiments(self) -> None:
        """Run suggested bot experiments and record outcomes."""
        if not self.experiment_manager or not self.improvement_engine:
            return
        base_roi = self._latest_roi()
        res = None
        try:
            import asyncio

            res = asyncio.run(
                self.experiment_manager.run_suggested_experiments(
                    self.improvement_engine.bot_name
                )
            )
        except Exception:
            self.logger.exception("bot experiment execution failed")

        variant = getattr(res, "variant", self.improvement_engine.bot_name)
        after_roi = getattr(res, "roi", base_roi)
        change = after_roi - base_roi
        wf_key = getattr(res, "workflow_id", variant)
        parent = self._workflow_event_ids.get(wf_key)
        try:
            event_id = self.history.add(
                EvolutionEvent(
                    action=f"bot_experiment:{variant}",
                    before_metric=base_roi,
                    after_metric=after_roi,
                    roi=change,
                    trending_topic=getattr(res, "trending_topic", None),
                    reason="bot experiment",
                    trigger="experiment",
                    performance=change,
                    parent_event_id=parent,
                    workflow_id=wf_key if isinstance(wf_key, int) else None,
                )
            )
            self._workflow_event_ids[wf_key] = event_id
            MutationLogger.record_mutation_outcome(
                event_id, after_metric=after_roi, roi=change, performance=change
            )
        except Exception:
            self.logger.exception("record bot experiment failed")

    # ------------------------------------------------------------------
    def _cleanup_workflows(self) -> None:
        """Remove paused workflows from MenaceDB to keep the database small."""
        mdb = getattr(self.resource_optimizer, "menace_db", None) if self.resource_optimizer else None
        if not mdb:
            return
        try:
            with mdb.engine.begin() as conn:
                rows = (
                    conn.execute(
                        mdb.workflows.select().where(mdb.workflows.c.status == "paused")
                    )
                    .mappings()
                    .fetchall()
                )
                for row in rows:
                    conn.execute(
                        mdb.workflows.delete().where(mdb.workflows.c.workflow_id == row["workflow_id"])
                )
        except Exception:
            self.logger.exception("cleanup workflows failed")

    # ------------------------------------------------------------------
    def _replace_main_workflow(self, name: str) -> None:
        """Set *name* as the active workflow in MenaceDB."""
        mdb = getattr(self.resource_optimizer, "menace_db", None) if self.resource_optimizer else None
        if not mdb:
            return
        try:
            with mdb.engine.begin() as conn:
                row = (
                    conn.execute(
                        mdb.workflows.select().where(mdb.workflows.c.workflow_name == name)
                    )
                    .mappings()
                    .fetchone()
                )
                if row:
                    conn.execute(
                        mdb.workflows.update()
                        .where(mdb.workflows.c.workflow_id == row["workflow_id"])
                        .values(status="winner")
                    )
        except Exception:
            self.logger.exception("replace main workflow failed")



__all__ = ["EvolutionTrigger", "EvolutionOrchestrator"]
