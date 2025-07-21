from __future__ import annotations

"""Periodic self-improvement engine for the Menace system."""

import logging
import time
import asyncio
import os
import json
import tempfile
from pathlib import Path

from .self_model_bootstrap import bootstrap
from .research_aggregator_bot import ResearchAggregatorBot, ResearchItem, InfoDB
from .model_automation_pipeline import ModelAutomationPipeline, AutomationResult
from .diagnostic_manager import DiagnosticManager
from .error_bot import ErrorBot, ErrorDB
from .data_bot import MetricsDB, DataBot
from .code_database import PatchHistoryDB
from .capital_management_bot import CapitalManagementBot
from .learning_engine import LearningEngine
from .unified_event_bus import UnifiedEventBus
from .neuroplasticity import PathwayRecord, Outcome
from .self_coding_engine import SelfCodingEngine
from .action_planner import ActionPlanner
from .evolution_history_db import EvolutionHistoryDB
from .self_improvement_policy import (
    SelfImprovementPolicy,
    ConfigurableSelfImprovementPolicy,
)
from .pre_execution_roi_bot import PreExecutionROIBot, BuildTask, ROIResult
from .env_config import PRE_ROI_SCALE, PRE_ROI_BIAS, PRE_ROI_CAP

POLICY_STATE_LEN = 19


class SelfImprovementEngine:
    """Run the automation pipeline on a configurable bot."""

    def __init__(
        self,
        *,
        interval: int = 3600,
        pipeline: ModelAutomationPipeline | None = None,
        bot_name: str = "menace",
        diagnostics: DiagnosticManager | None = None,
        info_db: InfoDB | None = None,
        capital_bot: "CapitalManagementBot" | None = None,
        energy_threshold: float = 0.5,
        learning_engine: LearningEngine | None = None,
        self_coding_engine: SelfCodingEngine | None = None,
        action_planner: "ActionPlanner" | None = None,
        event_bus: UnifiedEventBus | None = None,
        evolution_history: EvolutionHistoryDB | None = None,
        data_bot: DataBot | None = None,
        patch_db: PatchHistoryDB | None = None,
        policy: SelfImprovementPolicy | None = None,
        policy_strategy: str | None = None,
        optimize_self: bool = False,
        meta_logger: object | None = None,
        module_index: "ModuleIndexDB" | None = None,
        pre_roi_bot: PreExecutionROIBot | None = None,
        pre_roi_scale: float | None = None,
        pre_roi_bias: float | None = None,
        pre_roi_cap: float | None = None,
        synergy_weight_roi: float | None = None,
        synergy_weight_efficiency: float | None = None,
        synergy_weight_resilience: float | None = None,
        synergy_weight_antifragility: float | None = None,
        state_path: Path | str | None = None,
    ) -> None:
        self.interval = interval
        self.bot_name = bot_name
        self.info_db = info_db or InfoDB()
        self.aggregator = ResearchAggregatorBot([bot_name], info_db=self.info_db)
        self.pipeline = pipeline or ModelAutomationPipeline(
            aggregator=self.aggregator, action_planner=action_planner
        )
        err_bot = ErrorBot(ErrorDB(), MetricsDB())
        self.error_bot = err_bot
        self.diagnostics = diagnostics or DiagnosticManager(MetricsDB(), err_bot)
        self.last_run = 0.0
        self.capital_bot = capital_bot
        self.energy_threshold = energy_threshold
        self.learning_engine = learning_engine
        self.self_coding_engine = self_coding_engine
        self.event_bus = event_bus
        self.evolution_history = evolution_history
        self.data_bot = data_bot
        self.patch_db = patch_db or (data_bot.patch_db if data_bot else None)
        if policy is None:
            policy = ConfigurableSelfImprovementPolicy(strategy=policy_strategy)
        self.policy = policy
        if self.policy and getattr(self.policy, "path", None):
            try:
                self.policy.load(self.policy.path)
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("policy load failed: %s", exc)
        self.optimize_self_flag = optimize_self
        self.meta_logger = meta_logger
        self.pre_roi_bot = pre_roi_bot
        self.pre_roi_scale = pre_roi_scale if pre_roi_scale is not None else PRE_ROI_SCALE
        self.pre_roi_bias = pre_roi_bias if pre_roi_bias is not None else PRE_ROI_BIAS
        self.pre_roi_cap = pre_roi_cap if pre_roi_cap is not None else PRE_ROI_CAP
        self.synergy_weight_roi = (
            synergy_weight_roi
            if synergy_weight_roi is not None
            else float(os.getenv("SYNERGY_WEIGHT_ROI", "1.0"))
        )
        self.synergy_weight_efficiency = (
            synergy_weight_efficiency
            if synergy_weight_efficiency is not None
            else float(os.getenv("SYNERGY_WEIGHT_EFFICIENCY", "1.0"))
        )
        self.synergy_weight_resilience = (
            synergy_weight_resilience
            if synergy_weight_resilience is not None
            else float(os.getenv("SYNERGY_WEIGHT_RESILIENCE", "1.0"))
        )
        self.synergy_weight_antifragility = (
            synergy_weight_antifragility
            if synergy_weight_antifragility is not None
            else float(os.getenv("SYNERGY_WEIGHT_ANTIFRAGILITY", "1.0"))
        )
        self.state_path = Path(state_path) if state_path else None
        self.roi_history: list[float] = []
        self._load_state()
        from .module_index_db import ModuleIndexDB
        self.module_index = module_index or ModuleIndexDB()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SelfImprovementEngine")
        self._cycle_running = False
        self._schedule_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        if self.event_bus:
            if self.learning_engine:
                try:
                    self.event_bus.subscribe("pathway:new", self._on_new_pathway)
                except Exception as exc:
                    self.logger.exception("failed to subscribe to pathway events: %s", exc)
            try:
                self.event_bus.subscribe(
                    "evolve:self_improve", lambda *_: self.run_cycle()
                )
            except Exception as exc:
                self.logger.exception("failed to subscribe to self_improve events: %s", exc)

    # ------------------------------------------------------------------
    def _load_state(self) -> None:
        if not self.state_path or not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.roi_history = [float(x) for x in data.get("roi_history", [])]
            self.last_run = float(data.get("last_run", self.last_run))
        except Exception as exc:
            self.logger.exception("failed to load state: %s", exc)

    # ------------------------------------------------------------------
    def _save_state(self) -> None:
        if not self.state_path:
            return
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", delete=False, dir=self.state_path.parent, encoding="utf-8") as fh:
                json.dump({"roi_history": self.roi_history, "last_run": self.last_run}, fh)
                tmp = Path(fh.name)
            os.replace(tmp, self.state_path)
        except Exception as exc:
            self.logger.exception("failed to save state: %s", exc)

    # ------------------------------------------------------------------
    def _policy_state(self) -> tuple[int, ...]:
        """Return the state tuple used by :class:`SelfImprovementPolicy`."""
        energy = 0.0
        if self.capital_bot:
            try:
                energy = self.capital_bot.energy_score(
                    load=0.0, success_rate=1.0, deploy_eff=1.0, failure_rate=0.0
                )
            except Exception as exc:
                self.logger.exception("energy_score failed: %s", exc)
                energy = 0.0
        profit = 0.0
        if self.capital_bot:
            try:
                profit = self.capital_bot.profit()
            except Exception as exc:
                self.logger.exception("profit check failed: %s", exc)
                profit = 0.0
        trend = anomaly = patch_rate = 0.0
        if self.data_bot:
            try:
                trend = self.data_bot.long_term_roi_trend(limit=200)
            except Exception as exc:
                self.logger.exception("ROI trend fetch failed: %s", exc)
                trend = 0.0
            try:
                df = self.data_bot.db.fetch(100)
                if hasattr(df, "empty"):
                    if not getattr(df, "empty", True):
                        df["roi"] = df["revenue"] - df["expense"]
                        anomaly = float(len(DataBot.detect_anomalies(df, "roi"))) / len(
                            df
                        )
                elif isinstance(df, list) and df:
                    rois = [
                        float(r.get("revenue", 0.0) - r.get("expense", 0.0)) for r in df
                    ]
                    df_list = [{"roi": r} for r in rois]
                    anomaly = float(
                        len(DataBot.detect_anomalies(df_list, "roi"))
                    ) / len(rois)
            except Exception as exc:
                self.logger.exception("anomaly detection failed: %s", exc)
                anomaly = 0.0
            if getattr(self.data_bot, "patch_db", None):
                try:
                    patch_rate = self.data_bot.patch_db.success_rate()
                except Exception as exc:
                    self.logger.exception("patch success rate lookup failed: %s", exc)
                    patch_rate = 0.0
        avg_roi = avg_complex = revert_rate = 0.0
        module_idx = 0
        module_trend = 0.0
        dim_flag = 0
        tracker = getattr(self, "tracker", None)
        syn_roi = syn_eff = syn_res = syn_af = 0.0
        if tracker is not None:
            try:
                vals = tracker.metrics_history.get("synergy_roi", [])
                if vals:
                    syn_roi = float(vals[-1])
                    if len(vals) >= 2:
                        syn_roi = float(vals[-1] - vals[-2])
            except Exception:
                syn_roi = 0.0
            try:
                eff_vals = tracker.metrics_history.get("synergy_efficiency", [])
                if eff_vals:
                    syn_eff = float(eff_vals[-1])
                    if len(eff_vals) >= 2:
                        syn_eff = float(eff_vals[-1] - eff_vals[-2])
            except Exception:
                syn_eff = 0.0
            try:
                res_vals = tracker.metrics_history.get("synergy_resilience", [])
                if res_vals:
                    syn_res = float(res_vals[-1])
                    if len(res_vals) >= 2:
                        syn_res = float(res_vals[-1] - res_vals[-2])
            except Exception:
                syn_res = 0.0
            try:
                af_vals = tracker.metrics_history.get("synergy_antifragility", [])
                if af_vals:
                    syn_af = float(af_vals[-1])
                    if len(af_vals) >= 2:
                        syn_af = float(af_vals[-1] - af_vals[-2])
            except Exception:
                syn_af = 0.0
        syn_roi *= self.synergy_weight_roi
        syn_eff *= self.synergy_weight_efficiency
        syn_res *= self.synergy_weight_resilience
        syn_af *= self.synergy_weight_antifragility
        profit += syn_roi
        energy = max(0.0, energy + syn_eff)
        pdb = self.patch_db or (self.data_bot.patch_db if self.data_bot else None)
        if pdb:
            try:
                with pdb._connect() as conn:
                    rows = conn.execute(
                        "SELECT roi_delta, complexity_delta, reverted, filename "
                        "FROM patch_history ORDER BY id DESC LIMIT ?",
                        (10,),
                    ).fetchall()
                if rows:
                    avg_roi = float(sum(r[0] for r in rows) / len(rows))
                    avg_complex = float(sum(r[1] for r in rows) / len(rows))
                    revert_rate = float(sum(1 for r in rows if r[2]) / len(rows))
                    mod_name = Path(rows[0][3]).name
                    module_idx = self.module_index.get(mod_name)
                    try:
                        total = conn.execute(
                            "SELECT SUM(roi_delta) FROM patch_history WHERE filename=?",
                            (rows[0][3],),
                        ).fetchone()
                        module_trend = float(total[0] or 0.0)
                    except Exception:
                        module_trend = 0.0
                    if self.meta_logger:
                        try:
                            dim_flag = 1 if mod_name in self.meta_logger.diminishing() else 0
                            if module_trend == 0.0:
                                module_trend = dict(self.meta_logger.rankings()).get(mod_name, 0.0)
                        except Exception as exc:  # pragma: no cover - best effort
                            self.logger.exception("meta logger stats failed: %s", exc)
            except Exception as exc:
                self.logger.exception("patch metrics failed: %s", exc)
                avg_roi = avg_complex = revert_rate = 0.0
                module_idx = 0
            try:
                kw_count, kw_recent = pdb.keyword_features()
            except Exception as exc:
                self.logger.exception("keyword feature fetch failed: %s", exc)
                kw_count = kw_recent = 0
        else:
            kw_count = kw_recent = 0
        avg_roi_delta = avg_eff = 0.0
        if self.evolution_history:
            try:
                stats = self.evolution_history.averages(limit=5)
                avg_roi_delta = float(stats.get("avg_roi_delta", 0.0))
                avg_eff = float(stats.get("avg_efficiency", 0.0))
            except Exception as exc:  # pragma: no cover - best effort
                self.logger.exception("evolution history stats failed: %s", exc)
                avg_roi_delta = avg_eff = 0.0
        return (
            int(round(profit)),
            int(round(energy * 10)),
            int(round(trend * 10)),
            int(round(anomaly * 10)),
            int(round(patch_rate * 10)),
            int(round(avg_roi * 10)),
            int(round(avg_complex * 10)),
            int(round(revert_rate * 10)),
            int(module_idx),
            int(round(module_trend * 10)),
            int(dim_flag),
            int(kw_count),
            int(kw_recent),
            int(round(avg_roi_delta * 10)),
            int(round(avg_eff)),
            int(round(syn_roi * 10)),
            int(round(syn_eff * 10)),
            int(round(syn_res * 10)),
            int(round(syn_af * 10)),
        )

    # ------------------------------------------------------------------
    def _should_trigger(self) -> bool:
        if time.time() - self.last_run >= self.interval:
            return True
        if self.policy:
            try:
                if self.policy.score(self._policy_state()) > 0:
                    return True
            except Exception as exc:
                self.logger.exception("policy scoring failed: %s", exc)
        if self.pre_roi_bot:
            try:
                forecast = self.pre_roi_bot.predict_model_roi(self.bot_name, [])
                if forecast.roi > self.pre_roi_bias:
                    return True
            except Exception as exc:
                self.logger.exception("pre ROI forecast failed: %s", exc)
        issues = self.diagnostics.diagnose()
        return bool(issues)

    def _record_state(self) -> None:
        """Store metrics and discrepancies as research items."""
        mdb = self.diagnostics.metrics
        edb = self.diagnostics.error_bot.db
        df = mdb.fetch(20)
        for row in df.itertuples(index=False):
            item = ResearchItem(
                topic="metrics",
                content=str(row._asdict()),
                timestamp=time.time(),
            )
            try:
                self.info_db.add(item)
            except Exception as exc:
                self.logger.exception("failed to record metric item: %s", exc)
        disc = edb.discrepancies()
        if "message" in disc:
            for msg in disc["message"]:
                item = ResearchItem(
                    topic="error",
                    content=str(msg),
                    timestamp=time.time(),
                    tags=["error"],
                )
                try:
                    self.info_db.add(item)
                except Exception as exc:
                    self.logger.exception("failed to record error item: %s", exc)

    def _evaluate_learning(self) -> None:
        """Benchmark the learning engine via cross-validation."""
        if not self.learning_engine:
            return
        try:
            if hasattr(self.learning_engine, "evaluate"):
                result = self.learning_engine.evaluate()
                mean_score = float(result.get("cv_score", 0.0))
                if hasattr(self.learning_engine, "persist_evaluation"):
                    try:
                        self.learning_engine.persist_evaluation(result)
                    except Exception as exc:
                        self.logger.exception("persist_evaluation failed: %s", exc)
            else:
                X, y = self.learning_engine._dataset()  # type: ignore[attr-defined]
                if not X or len(set(y)) < 2:
                    return
                from sklearn.model_selection import cross_val_score

                scores = cross_val_score(self.learning_engine.model, X, y, cv=3)
                mean_score = float(scores.mean())
                if hasattr(self.learning_engine, "persist_evaluation"):
                    try:
                        self.learning_engine.persist_evaluation(
                            {
                                "cv_score": mean_score,
                                "holdout_score": mean_score,
                                "timestamp": time.time(),
                            }
                        )
                    except Exception as exc:
                        self.logger.exception("persist_evaluation failed: %s", exc)
        except Exception as exc:
            self.logger.exception("learning evaluation failed: %s", exc)
            mean_score = 0.0
        item = ResearchItem(
            topic="learning_eval",
            content=str({"cv_score": mean_score}),
            timestamp=time.time(),
        )
        try:
            self.info_db.add(item)
        except Exception as exc:
            self.logger.exception("failed to record learning eval: %s", exc)

    def _optimize_self(self) -> tuple[int | None, bool, float]:
        """Apply a patch to this engine via :class:`SelfCodingEngine`."""
        if not self.self_coding_engine:
            return None, False, 0.0
        try:
            patch_id, reverted, delta = self.self_coding_engine.apply_patch(
                Path(__file__), "self_improvement"
            )
            return patch_id, reverted, delta
        except Exception as exc:
            self.logger.exception("self optimization failed: %s", exc)
            return None, False, 0.0

    def _on_new_pathway(self, topic: str, payload: object) -> None:
        """Incrementally train when a new pathway is logged."""
        if not self.learning_engine:
            return
        if isinstance(payload, dict):
            try:
                rec = PathwayRecord(
                    actions=payload.get("actions", ""),
                    inputs=payload.get("inputs", ""),
                    outputs=payload.get("outputs", ""),
                    exec_time=float(payload.get("exec_time", 0.0)),
                    resources=payload.get("resources", ""),
                    outcome=Outcome(payload.get("outcome", "FAILURE")),
                    roi=float(payload.get("roi", 0.0)),
                    ts=payload.get("ts", ""),
                )
                self.learning_engine.partial_train(rec)
            except Exception as exc:
                self.logger.exception("failed to process pathway record: %s", exc)

    # ------------------------------------------------------------------
    def run_cycle(self, energy: int = 1) -> AutomationResult:
        """Execute a self-improvement cycle."""
        self._cycle_running = True
        try:
            state = self._policy_state() if self.policy else (0,) * POLICY_STATE_LEN
            predicted = self.policy.score(state) if self.policy else 0.0
            self.logger.info(
                "cycle start",
                extra={"energy": energy, "predicted_roi": predicted, "state": state},
            )
            before_roi = 0.0
            if self.capital_bot:
                try:
                    before_roi = self.capital_bot.profit()
                    self.logger.info("initial ROI", extra={"value": before_roi})
                except Exception as exc:
                    self.logger.exception("profit lookup failed: %s", exc)
                    before_roi = 0.0
            if self.capital_bot:
                try:
                    energy = int(
                        round(
                            self.capital_bot.energy_score(
                                load=0.0,
                                success_rate=1.0,
                                deploy_eff=1.0,
                                failure_rate=0.0,
                                reward=None,
                            )
                            * 5
                        )
                    )
                    self.logger.info("available energy", extra={"value": energy})
                except Exception as exc:
                    self.logger.exception("energy calculation failed: %s", exc)
                    energy = 1
            if self.policy:
                try:
                    energy = max(1, int(round(energy * (1 + max(0.0, predicted)))))
                    self.logger.info("policy adjusted energy", extra={"value": energy})
                except Exception as exc:
                    self.logger.exception("policy energy adjustment failed: %s", exc)
            if self.pre_roi_bot:
                try:
                    forecast = self.pre_roi_bot.predict_model_roi(self.bot_name, [])
                    roi_pred = float(getattr(forecast, "roi", 0.0))
                    scale = 1 + max(0.0, roi_pred + self.pre_roi_bias) * self.pre_roi_scale
                    if self.pre_roi_cap:
                        scale = min(scale, self.pre_roi_cap)
                    energy = max(1, int(round(energy * scale)))
                    self.logger.info(
                        "pre_roi adjusted energy",
                        extra={"value": energy, "roi_prediction": roi_pred},
                    )
                except Exception as exc:
                    self.logger.exception("pre ROI energy adjustment failed: %s", exc)
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                try:
                    def _delta(name: str) -> float:
                        vals = tracker.metrics_history.get(name, [])
                        if not vals:
                            return 0.0
                        if len(vals) >= 2:
                            return float(vals[-1] - vals[-2])
                        return float(vals[-1])

                    syn_adj = (
                        _delta("synergy_roi") * self.synergy_weight_roi
                        + _delta("synergy_efficiency") * self.synergy_weight_efficiency
                        + _delta("synergy_resilience") * self.synergy_weight_resilience
                        + _delta("synergy_antifragility") * self.synergy_weight_antifragility
                    )
                    if syn_adj:
                        energy = int(round(energy * (1.0 + syn_adj)))
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("synergy energy adjustment failed: %s", exc)
            energy = max(1, min(int(energy), 100))
            model_id = bootstrap()
            self.logger.info("model bootstrapped", extra={"model_id": model_id})
            self.info_db.set_current_model(model_id)
            self._record_state()
            if self.learning_engine:
                try:
                    self.logger.info("training learning engine")
                    self.learning_engine.train()
                    self._evaluate_learning()
                except Exception as exc:
                    self.logger.exception("learning engine run failed: %s", exc)
            self.logger.info("running automation pipeline", extra={"energy": energy})
            result = self.pipeline.run(self.bot_name, energy=energy)
            self.logger.info("pipeline complete", extra={"roi": getattr(result.roi, 'roi', 0.0)})
            trending_topic = getattr(result, "trending_topic", None)
            patch_id = None
            reverted = False
            if self.self_coding_engine and result.package:
                try:
                    self.logger.info(
                        "applying helper patch",
                        extra={"trending_topic": trending_topic},
                    )
                    patch_id, reverted, delta = self.self_coding_engine.apply_patch(
                        Path("auto_helpers.py"),
                        "helper",
                        trending_topic=trending_topic,
                    )
                    if self.policy:
                        try:
                            self.logger.info(
                                "patch applied",
                                extra={
                                    "patch_id": patch_id,
                                    "reverted": reverted,
                                    "delta": delta,
                                },
                            )
                            st = self._policy_state()
                            syn_reward = st[-2] / 10.0 + st[-1] / 10.0
                            self.policy.update(
                                st,
                                delta + syn_reward,
                                synergy_roi_delta=st[-4] / 10.0,
                                synergy_efficiency_delta=st[-3] / 10.0,
                            )
                            if getattr(self.policy, "path", None):
                                try:
                                    self.policy.save()
                                except Exception as exc:  # pragma: no cover - best effort
                                    self.logger.exception("policy save failed: %s", exc)
                        except Exception as exc:
                            self.logger.exception(
                                "policy patch update failed: %s", exc
                            )
                    if self.optimize_self_flag:
                        self._optimize_self()
                except Exception as exc:
                    self.logger.exception("helper patch failed: %s", exc)
                    patch_id = None
                    reverted = False
            if self.error_bot:
                try:
                    self.error_bot.auto_patch_recurrent_errors()
                    self.logger.info("error auto-patching complete")
                except Exception as exc:
                    self.logger.exception("auto patch recurrent errors failed: %s", exc)
            after_roi = before_roi
            if self.capital_bot:
                try:
                    after_roi = self.capital_bot.profit()
                    self.logger.info(
                        "post-cycle ROI", extra={"before": before_roi, "after": after_roi}
                    )
                except Exception as exc:
                    self.logger.exception("post-cycle profit lookup failed: %s", exc)
                    after_roi = before_roi
            roi_value = result.roi.roi if result.roi else 0.0
            if self.evolution_history:
                try:
                    from .evolution_history_db import EvolutionEvent

                    self.evolution_history.add(
                        EvolutionEvent(
                            action="self_improvement",
                            before_metric=before_roi,
                            after_metric=after_roi,
                            roi=roi_value,
                            predicted_roi=predicted,
                            trending_topic=trending_topic,
                        )
                    )
                except Exception as exc:
                    self.logger.exception("evolution history logging failed: %s", exc)
            if self.data_bot:
                eff = bottleneck = patch_rate = trend = anomaly = 0.0
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
                except Exception as exc:
                    self.logger.exception("data fetch failed: %s", exc)
                    eff = bottleneck = 0.0
                if self.self_coding_engine and getattr(
                    self.self_coding_engine, "patch_db", None
                ):
                    try:
                        patch_rate = self.self_coding_engine.patch_db.success_rate()
                    except Exception as exc:
                        self.logger.exception("self_coding patch rate lookup failed: %s", exc)
                        patch_rate = 0.0
                if getattr(self.data_bot, "patch_db", None) and not patch_rate:
                    try:
                        patch_rate = self.data_bot.patch_db.success_rate()
                    except Exception as exc:
                        self.logger.exception("data_bot patch rate lookup failed: %s", exc)
                        patch_rate = 0.0
                try:
                    trend = self.data_bot.long_term_roi_trend(limit=200)
                except Exception as exc:
                    self.logger.exception("trend retrieval failed: %s", exc)
                    trend = 0.0
                try:
                    df_anom = self.data_bot.db.fetch(100)
                    if hasattr(df_anom, "empty"):
                        if not getattr(df_anom, "empty", True):
                            df_anom["roi"] = df_anom["revenue"] - df_anom["expense"]
                            anomaly = float(
                                len(DataBot.detect_anomalies(df_anom, "roi"))
                            ) / len(df_anom)
                    elif isinstance(df_anom, list) and df_anom:
                        rois = [
                            float(r.get("revenue", 0.0) - r.get("expense", 0.0))
                            for r in df_anom
                        ]
                        df_list = [{"roi": r} for r in rois]
                        anomaly = float(
                            len(DataBot.detect_anomalies(df_list, "roi"))
                        ) / len(rois)
                except Exception as exc:
                    self.logger.exception("anomaly calculation failed: %s", exc)
                    anomaly = 0.0
                try:
                    self.data_bot.log_evolution_cycle(
                        "self_improvement",
                        before_roi,
                        after_roi,
                        roi_value,
                        0.0,
                        patch_success=patch_rate,
                        roi_delta=after_roi - before_roi,
                        roi_trend=trend,
                        anomaly_count=anomaly,
                        efficiency=eff,
                        bottleneck=bottleneck,
                        patch_id=patch_id,
                        trending_topic=trending_topic,
                        reverted=reverted,
                    )
                    self.logger.info(
                        "cycle metrics",
                        extra={
                            "patch_success": patch_rate,
                            "roi_delta": after_roi - before_roi,
                            "roi_trend": trend,
                            "anomaly": anomaly,
                        },
                    )
                    if self.capital_bot:
                        try:
                            self.capital_bot.log_evolution_event(
                                "self_improvement",
                                before_roi,
                                after_roi,
                            )
                        except Exception as exc:
                            self.logger.exception("capital bot evolution log failed: %s", exc)
                except Exception as exc:
                    self.logger.exception("data_bot evolution logging failed: %s", exc)
            self.last_run = time.time()
            self.roi_history.append(after_roi - before_roi)
            self._save_state()
            if self.policy:
                try:
                    next_state = self._policy_state()
                    syn_reward = next_state[-2] / 10.0 + next_state[-1] / 10.0
                    self.policy.update(
                        state,
                        after_roi - before_roi + syn_reward,
                        next_state,
                        synergy_roi_delta=next_state[-4] / 10.0,
                        synergy_efficiency_delta=next_state[-3] / 10.0,
                    )
                    if getattr(self.policy, "path", None):
                        try:
                            self.policy.save()
                        except Exception as exc:  # pragma: no cover - best effort
                            self.logger.exception("policy save failed: %s", exc)
                    self.logger.info(
                        "policy updated", extra={"reward": after_roi - before_roi}
                    )
                except Exception as exc:
                    self.logger.exception("policy update failed: %s", exc)
            if self.policy and getattr(self.policy, "path", None):
                try:
                    self.policy.save()
                except Exception as exc:  # pragma: no cover - best effort
                    self.logger.exception("policy save failed: %s", exc)
            self.logger.info("cycle complete", extra={"roi": roi_value})
            return result
        finally:
            self._cycle_running = False

    async def _schedule_loop(self, energy: int = 1) -> None:
        while not self._stop_event.is_set():
            current_energy = energy
            if self.capital_bot:
                try:
                    current_energy = self.capital_bot.energy_score(
                        load=0.0,
                        success_rate=1.0,
                        deploy_eff=1.0,
                        failure_rate=0.0,
                    )
                except Exception as exc:
                    self.logger.exception("energy check failed: %s", exc)
                    current_energy = energy
            if current_energy >= self.energy_threshold and not self._cycle_running:
                try:
                    await asyncio.to_thread(
                        self.run_cycle, energy=int(round(current_energy * 5))
                    )
                except Exception as exc:
                    self.logger.exception(
                        "self improvement run_cycle failed with energy %s: %s",
                        int(round(current_energy * 5)),
                        exc,
                    )
            await asyncio.sleep(self.interval)

    def schedule(self, energy: int = 1, *, loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Task:
        """Start the scheduling loop in the background."""
        if self._schedule_task and not self._schedule_task.done():
            return self._schedule_task
        self._stop_event = asyncio.Event()
        loop = loop or asyncio.get_event_loop()
        self._schedule_task = loop.create_task(self._schedule_loop(energy))
        return self._schedule_task

    async def shutdown_schedule(self) -> None:
        """Stop the scheduler and wait for the task to finish."""
        if self._schedule_task:
            assert self._stop_event is not None
            self._stop_event.set()
            try:
                await self._schedule_task
            finally:
                self._schedule_task = None


from typing import Callable, Optional


class ImprovementEngineRegistry:
    """Register and run multiple :class:`SelfImprovementEngine` instances."""

    def __init__(self) -> None:
        self.engines: dict[str, SelfImprovementEngine] = {}

    def register_engine(self, name: str, engine: SelfImprovementEngine) -> None:
        """Add *engine* under *name*."""
        self.engines[name] = engine

    def unregister_engine(self, name: str) -> None:
        """Remove the engine referenced by *name* if present."""
        self.engines.pop(name, None)

    def run_all_cycles(self, energy: int = 1) -> dict[str, AutomationResult]:
        """Execute ``run_cycle`` on all registered engines."""
        results: dict[str, AutomationResult] = {}
        for name, eng in self.engines.items():
            if eng._should_trigger():
                results[name] = eng.run_cycle(energy=energy)
        return results

    async def run_all_cycles_async(self, energy: int = 1) -> dict[str, AutomationResult]:
        """Asynchronously execute ``run_cycle`` on all registered engines."""
        async def _run(name: str, eng: SelfImprovementEngine):
            if eng._should_trigger():
                res = await asyncio.to_thread(eng.run_cycle, energy=energy)
                return name, res
            return None

        tasks = [asyncio.create_task(_run(n, e)) for n, e in self.engines.items()]
        results: dict[str, AutomationResult] = {}
        for t in tasks:
            out = await t
            if out:
                results[out[0]] = out[1]
        return results

    def schedule_all(self, energy: int = 1, *, loop: asyncio.AbstractEventLoop | None = None) -> list[asyncio.Task]:
        """Start schedules for all engines and return the created tasks."""
        tasks: list[asyncio.Task] = []
        for eng in self.engines.values():
            tasks.append(eng.schedule(energy=energy, loop=loop))
        return tasks

    async def shutdown_all(self) -> None:
        """Gracefully stop all running schedules."""
        for eng in self.engines.values():
            await eng.shutdown_schedule()

    def autoscale(
        self,
        *,
        capital_bot: CapitalManagementBot,
        data_bot: DataBot,
        factory: Callable[[str], SelfImprovementEngine],
        max_engines: int = 5,
        min_engines: int = 1,
        create_energy: float = 0.8,
        remove_energy: float = 0.3,
        roi_threshold: float = 0.0,
        cost_per_engine: float = 0.0,
        approval_callback: Optional[Callable[[], bool]] = None,
        max_instances: Optional[int] = None,
    ) -> None:
        """Dynamically create or remove engines based on ROI and resources."""
        try:
            energy = capital_bot.energy_score(
                load=0.0,
                success_rate=1.0,
                deploy_eff=1.0,
                failure_rate=0.0,
            )
        except Exception as exc:
            self.logger.exception("autoscale energy check failed: %s", exc)
            energy = 0.0
        try:
            trend = data_bot.long_term_roi_trend(limit=200)
        except Exception as exc:
            self.logger.exception("autoscale trend fetch failed: %s", exc)
            trend = 0.0
        if not capital_bot.check_budget():
            return
        if max_instances is not None and len(self.engines) >= max_instances:
            return
        projected_roi = trend - cost_per_engine
        if (
            energy >= create_energy
            and trend > roi_threshold
            and projected_roi > 0.0
            and len(self.engines) < max_engines
        ):
            if approval_callback and not approval_callback():
                return
            name = f"engine{len(self.engines)}"
            self.register_engine(name, factory(name))
            return
        if (
            energy <= remove_energy or trend <= roi_threshold or projected_roi <= 0.0
        ) and len(self.engines) > min_engines:
            name = next(iter(self.engines))
            self.unregister_engine(name)


__all__ = ["SelfImprovementEngine", "ImprovementEngineRegistry", "auto_x"]


def auto_x(
    engines: list[SelfImprovementEngine] | None = None,
    *,
    energy: int = 1,
) -> dict[str, AutomationResult]:
    """Convenience helper to run a self‑improvement cycle."""
    registry = ImprovementEngineRegistry()
    if engines:
        for idx, eng in enumerate(engines):
            registry.register_engine(f"engine{idx}", eng)
    else:
        registry.register_engine("default", SelfImprovementEngine())
    return registry.run_all_cycles(energy=energy)
