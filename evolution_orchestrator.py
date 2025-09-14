from __future__ import annotations

"""Orchestrate system evolution based on metrics and capital signals."""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import time
import inspect
import uuid

import numpy as np

from .data_bot import DataBot
from .capital_management_bot import CapitalManagementBot
from .system_evolution_manager import SystemEvolutionManager
from .evolution_history_db import EvolutionHistoryDB, EvolutionEvent
from .evaluation_history_db import EvaluationHistoryDB
from .trend_predictor import TrendPredictor
from typing import TYPE_CHECKING, Any, Dict
from context_builder_util import create_context_builder, ensure_fresh_weights
from retry_utils import with_retry
try:  # pragma: no cover - optional dependency
    from vector_service.context_builder import ContextBuilder  # noqa: F401
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "vector_service.ContextBuilder is required for EvolutionOrchestrator"
    ) from exc
from .self_coding_manager import HelperGenerationError
from .sandbox_settings import SandboxSettings
from .threshold_service import threshold_service
try:  # pragma: no cover - optional dependency
    from . import mutation_logger as MutationLogger
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "mutation_logger is required for EvolutionOrchestrator"
    ) from exc

try:  # pragma: no cover - allow flat imports
    from .unified_event_bus import UnifiedEventBus
except Exception as exc:  # pragma: no cover - missing dependency
    raise RuntimeError(
        "unified_event_bus.UnifiedEventBus is required for EvolutionOrchestrator"
    ) from exc
try:  # pragma: no cover - allow flat imports
    from .shared_event_bus import event_bus as _SHARED_EVENT_BUS
except Exception:  # pragma: no cover - flat layout fallback
    from shared_event_bus import event_bus as _SHARED_EVENT_BUS  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .adaptive_roi_predictor import AdaptiveROIPredictor
    from .self_coding_manager import SelfCodingManager
    from .self_improvement import SelfImprovementEngine
    from .bot_creation_bot import BotCreationBot
    from .resource_allocation_optimizer import ResourceAllocationOptimizer
    from .workflow_evolution_bot import WorkflowEvolutionBot
    from .experiment_manager import ExperimentManager
    from .evolution_analysis_bot import EvolutionAnalysisBot
    from .evolution_predictor import EvolutionPredictor


@dataclass
class EvolutionTrigger:
    """Trigger thresholds for evolution."""

    error_rate: float = 0.1
    roi_drop: float = -0.1
    energy_threshold: float = 0.3


class EvolutionOrchestrator:
    """Monitor metrics and coordinate improvement and evolution cycles.

    When ``triggers`` are not provided the ROI and error thresholds are
    loaded via :class:`ThresholdService` to ensure consistent behaviour with
    other components.
    """

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
        selfcoding_manager: SelfCodingManager | None = None,
        trend_predictor: TrendPredictor | None = None,
        predictor: EvolutionPredictor | None = None,
        multi_predictor: object | None = None,
        event_bus: UnifiedEventBus | None = None,
        roi_predictor: AdaptiveROIPredictor | None = None,
        roi_gain_floor: float = 0.0,
        roi_confidence_floor: float = 0.0,
        dataset_path: str | Path = "roi_eval_dataset.csv",
        retrain_interval: int = 10,
    ) -> None:
        self.data_bot = data_bot
        self.capital_bot = capital_bot
        self.improvement_engine = improvement_engine
        self.evolution_manager = evolution_manager
        self.history = history_db or EvolutionHistoryDB()
        if triggers is None:
            bot = selfcoding_manager.bot_name if selfcoding_manager else None
            settings = getattr(data_bot, "settings", None)
            t = threshold_service.get(bot, settings)
            self.triggers = EvolutionTrigger(
                error_rate=t.error_threshold, roi_drop=t.roi_drop
            )
        else:
            self.triggers = triggers
        self.bot_creator = bot_creator
        self.resource_optimizer = resource_optimizer
        self.workflow_evolver = workflow_evolver
        self.experiment_manager = experiment_manager
        self.analysis_bot = analysis_bot
        self.selfcoding_manager = selfcoding_manager
        self.predictor = predictor
        self.multi_predictor = multi_predictor
        self.trend_predictor = trend_predictor
        # Share a single event bus across components when one is not provided
        # explicitly so that all orchestrated systems observe the same events.
        self.event_bus = event_bus or _SHARED_EVENT_BUS
        self.roi_predictor = roi_predictor
        self.roi_gain_floor = float(roi_gain_floor)
        self.roi_confidence_floor = float(roi_confidence_floor)
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
        self.prev_error = self._error_rate()
        self._cycles = 0
        self._last_workflow_benchmark = 0.0
        self._benchmark_interval = 3600
        if self.event_bus and self.selfcoding_manager:
            def _handle_degradation(topic: str, event: object) -> None:
                try:
                    self._invoke_register_patch_cycle(event)
                except Exception:
                    self.logger.exception("register_patch_cycle failed")

            self.event_bus.subscribe("degradation:detected", _handle_degradation)
        self._cached_eval_score = 0.0
        self._workflow_roi_history: dict[str, list[float]] = {}
        self._last_mutation_id: int | None = None
        self._workflow_event_ids: dict[int | str, int] = {}
        self.roi_history: list[float] = []
        self._registered_bots: set[str] = set()
        self._pending_patch_cycle: set[str] = set()
        # Token used by ``SelfCodingManager`` to verify provenance of patch calls
        self.provenance_token = uuid.uuid4().hex
        if not self.dataset_path.exists():
            try:
                self.dataset_path.write_text(
                    "before_roi,error_rate,eval_score,actual_roi,predicted_roi,error\n",
                    encoding="utf-8",
                )
            except Exception:
                self.logger.exception(
                    "failed to initialize ROI dataset at %s", self.dataset_path
                )
                raise
        if self.event_bus:
            try:
                self.event_bus.subscribe("evolve:system", lambda *_: self.run_cycle())
                self.event_bus.subscribe(
                    "self_coding:patch_applied", self._on_patch_applied
                )
                self.event_bus.subscribe(
                    "thresholds:updated", self._on_thresholds_updated
                )
            except Exception:
                self.logger.exception("event bus subscription failed")

        bus = getattr(self.data_bot, "event_bus", None) or self.event_bus
        if bus:
            try:
                bus.subscribe("data:threshold_breach", lambda _t, e: self._on_threshold_breach(e))
                bus.subscribe("bot:degraded", lambda _t, e: self._on_bot_degraded(e))
                bus.subscribe(
                    "self_coding:degradation",
                    lambda _t, e: self._on_self_coding_degradation(e),
                )
                self._degradation_subscribed = True
            except Exception:
                self.logger.exception("bot degraded subscription failed")
        else:
            try:
                self.data_bot.subscribe_degradation(self._on_bot_degraded)
                self._degradation_subscribed = True
            except Exception:
                self.logger.exception("failed to attach degradation callback")
            self._degradation_subscribed = False

    def _invoke_register_patch_cycle(self, *args: Any) -> None:
        """Invoke ``register_patch_cycle`` with provenance when supported."""
        if not self.selfcoding_manager:
            return
        try:
            self.selfcoding_manager.register_patch_cycle(
                *args, provenance_token=self.provenance_token
            )
        except TypeError:
            self.selfcoding_manager.register_patch_cycle(*args)

    # ------------------------------------------------------------------
    def register_bot(self, bot: str) -> None:
        """Subscribe to DataBot metrics for *bot* and existing registry bots."""
        if not bot:
            return
        registry = getattr(self.selfcoding_manager, "bot_registry", None)
        bots = [bot]
        if registry and getattr(registry, "graph", None):
            try:
                bots = list(registry.graph)
                if bot not in bots:
                    bots.append(bot)
            except Exception:  # pragma: no cover - best effort
                bots = [bot]
        bus = getattr(self.data_bot, "event_bus", None) or self.event_bus
        if bus and not getattr(self, "_bot_registered_listener", False):
            bus.subscribe("bot:new", self._ensure_degradation_subscription)
            self._bot_registered_listener = True
        self._ensure_degradation_subscription()
        for name in bots:
            if name in self._registered_bots:
                continue
            self._registered_bots.add(name)
            try:
                if getattr(self.data_bot, "check_degradation", None):
                    # seed baseline metrics so future deltas are meaningful
                    self.data_bot.check_degradation(
                        name, roi=0.0, errors=0.0, test_failures=0.0
                    )
            except Exception:
                self.logger.exception("failed to register bot %s for metrics", name)

    def register_patch_cycle(self, event: Dict[str, Any]) -> None:
        """Translate degradation *event* into a patch-cycle registration."""
        if not self.selfcoding_manager:
            return
        bot = str(event.get("bot", ""))
        severity = float(event.get("severity", 0.0))
        desc = (
            f"auto_patch_due_to_degradation:{bot}"
            if bot
            else "auto_patch_due_to_degradation"
        )
        self.logger.info(
            "register_patch_cycle for %s with severity %.2f", bot or "unknown", severity
        )
        context = {
            "roi_baseline": event.get("roi_baseline", 0.0),
            "errors_baseline": event.get("errors_baseline", 0.0),
            "tests_failed_baseline": event.get("tests_failed_baseline", 0.0),
            "delta_roi": event.get("delta_roi", 0.0),
            "delta_errors": event.get("delta_errors", 0.0),
            "delta_tests_failed": event.get("delta_tests_failed", 0.0),
            "severity": severity,
            "trigger": "degradation",
            "reason": "bot degraded",
        }
        try:
            self._invoke_register_patch_cycle(desc, context)
        except Exception:
            self.logger.exception("failed to register patch cycle for %s", bot)
        else:
            if bot:
                self._pending_patch_cycle.add(bot)
        registry = getattr(self.selfcoding_manager, "bot_registry", None)
        bus = (
            getattr(self.selfcoding_manager, "event_bus", None)
            or self.event_bus
            or getattr(self.data_bot, "event_bus", None)
        )
        module_path: Path | None = None
        if registry and bot and bot in getattr(registry, "graph", {}):
            try:
                module_path = Path(registry.graph.nodes[bot]["module"])
            except Exception:
                self.logger.exception("bot registry lookup failed for %s", bot)
        if not module_path or not module_path.exists():
            payload = {
                "bot": bot,
                "success": False,
                "error": "module_path_missing",
                "severity": severity,
            }
            if bus:
                try:
                    bus.publish("self_coding:patch_attempt", payload)
                except Exception:
                    self.logger.exception(
                        "failed to publish patch_attempt for %s", bot
                    )
            try:
                self.history.add(
                    EvolutionEvent(
                        action="patch_failed",
                        before_metric=float(context.get("roi_baseline", 0.0)),
                        after_metric=float(context.get("roi_baseline", 0.0)),
                        roi=0.0,
                        reason=desc,
                        trigger="degradation",
                        performance=0.0,
                    )
                )
            except Exception:
                self.logger.exception("failed to record patch attempt")
            return
        try:
            builder = create_context_builder()
            try:
                ensure_fresh_weights(builder)
            except Exception:
                self.logger.exception(
                    "failed to refresh context builder for %s", bot
                )
            self.selfcoding_manager.generate_and_patch(
                module_path,
                desc,
                context_meta=context,
                context_builder=builder,
                provenance_token=self.provenance_token,
            )
            patch_id = getattr(self.selfcoding_manager, "_last_patch_id", None)
            commit = getattr(self.selfcoding_manager, "_last_commit_hash", None)
            success = bool(patch_id and commit)
        except Exception:
            self.logger.exception("failed to self patch after degradation of %s", bot)
            patch_id = getattr(self.selfcoding_manager, "_last_patch_id", None)
            commit = getattr(self.selfcoding_manager, "_last_commit_hash", None)
            success = False
        roi_before = float(context.get("roi_baseline", 0.0))
        err_before = float(context.get("errors_baseline", 0.0))
        roi_after = (
            self.data_bot.roi(bot) if hasattr(self.data_bot, "roi") else roi_before
        )
        err_after = (
            self.data_bot.average_errors(bot)
            if hasattr(self.data_bot, "average_errors")
            else err_before
        )
        roi_delta = roi_after - roi_before
        err_delta = err_after - err_before
        try:
            self.history.add(
                EvolutionEvent(
                    action="patch" if success else "patch_failed",
                    before_metric=roi_before,
                    after_metric=roi_after if success else roi_before,
                    roi=roi_delta if success else 0.0,
                    patch_id=patch_id,
                    reason=desc,
                    trigger="degradation",
                    performance=roi_delta,
                    bottleneck=err_delta,
                )
            )
        except Exception:
            self.logger.exception("failed to record patch result")
        payload = {
            "bot": bot,
            "patch_id": patch_id,
            "commit": commit,
            "roi_before": roi_before,
            "roi_after": roi_after if success else roi_before,
            "roi_delta": roi_delta if success else 0.0,
            "errors_before": err_before,
            "errors_after": err_after,
            "error_delta": err_delta,
            "description": desc,
            "path": str(module_path),
            "severity": severity,
            "success": success,
        }
        if bus:
            try:
                bus.publish("self_coding:patch_attempt", payload)
            except Exception:
                self.logger.exception(
                    "failed to publish patch_attempt for %s", bot
                )
        if registry and success:
            try:
                def _upd() -> None:
                    registry.update_bot(bot, str(module_path), patch_id=patch_id, commit=commit)

                with_retry(_upd, logger=self.logger)
            except Exception:
                self.logger.exception("failed to update bot %s", bot)

    def _ensure_degradation_subscription(self, *_args: object) -> None:
        if getattr(self, "_degradation_subscribed", False):
            return
        try:
            self.data_bot.subscribe_degradation(self._on_bot_degraded)
            self._degradation_subscribed = True
        except Exception:
            self.logger.exception("failed to attach degradation callback")

    # ------------------------------------------------------------------
    def _on_patch_applied(self, _topic: str, event: object) -> None:
        """Update history and ROI when a self-coding patch is applied."""
        if not isinstance(event, dict):
            return
        try:
            before = float(event.get("roi_before", 0.0))
            after = float(event.get("roi_after", before))
            delta = float(event.get("roi_delta", after - before))
            patch_id = event.get("patch_id")
            desc = event.get("description", "")
            path = event.get("path", "")
            self.history.add(
                EvolutionEvent(
                    action=f"patch:{Path(path).name}" if path else "patch",
                    before_metric=before,
                    after_metric=after,
                    roi=delta,
                    patch_id=patch_id,
                    reason=desc,
                    trigger="self_coding",
                    performance=delta,
                )
            )
            self.prev_roi = after
            self.roi_history.append(after)
        except Exception:
            self.logger.exception("failed to record patch_applied event")

    # ------------------------------------------------------------------
    def _on_thresholds_updated(self, _topic: str, event: object) -> None:
        """Refresh evolution triggers when self-coding thresholds change."""
        if not isinstance(event, dict):
            return
        bot = event.get("bot")
        if self.selfcoding_manager and bot != self.selfcoding_manager.bot_name:
            return
        try:
            roi_drop = float(event.get("roi_drop", self.triggers.roi_drop))
            err = float(event.get("error_threshold", self.triggers.error_rate))
            if (
                roi_drop != self.triggers.roi_drop
                or err != self.triggers.error_rate
            ):
                self.triggers = EvolutionTrigger(
                    error_rate=err,
                    roi_drop=roi_drop,
                    energy_threshold=self.triggers.energy_threshold,
                )
        except Exception:
            self.logger.exception("failed to process threshold update")

    # ------------------------------------------------------------------
    def _refresh_thresholds(self, bot: str) -> None:
        """Reload thresholds for *bot* to keep decisions fresh."""
        try:
            settings = getattr(self.data_bot, "settings", None)
            t = threshold_service.reload(bot, settings)
            self.triggers = EvolutionTrigger(
                error_rate=float(getattr(t, "error_threshold", self.triggers.error_rate)),
                roi_drop=float(getattr(t, "roi_drop", self.triggers.roi_drop)),
                energy_threshold=self.triggers.energy_threshold,
            )
        except Exception:
            self.logger.exception("failed to refresh thresholds for %s", bot)

    # ------------------------------------------------------------------
    def _on_threshold_breach(self, event: dict) -> None:
        """Register patch cycle when a bot exceeds degradation thresholds."""
        if not self.selfcoding_manager:
            return
        bot = str(event.get("bot", ""))
        roi = self.data_bot.roi(bot) if hasattr(self.data_bot, "roi") else 0.0
        err = (
            self.data_bot.average_errors(bot)
            if hasattr(self.data_bot, "average_errors")
            else 0.0
        )
        failures = (
            self.data_bot.average_test_failures(bot)
            if hasattr(self.data_bot, "average_test_failures")
            else 0.0
        )
        context_meta = {
            **event,
            "roi": roi,
            "error_rate": err,
            "tests_failed": failures,
        }
        desc = (
            f"auto_patch_due_to_threshold_breach:{bot}"
            if bot
            else "auto_patch_due_to_threshold_breach"
        )
        try:
            self._invoke_register_patch_cycle(desc, context_meta)
        except Exception:
            self.logger.exception("failed to register patch cycle for %s", bot)
        else:
            if bot:
                self._pending_patch_cycle.add(bot)
        self._refresh_thresholds(bot or getattr(self.selfcoding_manager, "bot_name", ""))

    # ------------------------------------------------------------------
    def _on_self_coding_degradation(self, event: dict) -> None:
        """Register patch cycle when self-coding reports degradation."""
        if not self.selfcoding_manager:
            return
        bot = str(event.get("bot", ""))
        desc = f"auto_patch_due_to_degradation:{bot}"
        try:
            self._invoke_register_patch_cycle(desc, event)
            if bot:
                self._pending_patch_cycle.add(bot)
        except Exception:
            self.logger.exception("failed to register patch cycle for %s", bot)

    # ------------------------------------------------------------------
    def _on_bot_degraded(self, event: dict) -> None:
        """Handle bot degradation events by generating and applying a patch."""
        if not self.selfcoding_manager:
            return
        bot = str(event.get("bot", ""))
        roi = self.data_bot.roi(bot) if hasattr(self.data_bot, "roi") else 0.0
        err = (
            self.data_bot.average_errors(bot)
            if hasattr(self.data_bot, "average_errors")
            else 0.0
        )
        failures = (
            self.data_bot.average_test_failures(bot)
            if hasattr(self.data_bot, "average_test_failures")
            else 0.0
        )
        context_meta = {
            **event,
            "roi": roi,
            "error_rate": err,
            "tests_failed": failures,
        }
        desc = (
            f"auto_patch_due_to_degradation:{bot}"
            if bot
            else "auto_patch_due_to_degradation"
        )
        try:
            self._invoke_register_patch_cycle(desc, context_meta)
        except Exception:
            self.logger.exception("failed to register patch cycle for %s", bot)
        else:
            if bot:
                self._pending_patch_cycle.add(bot)
        self._refresh_thresholds(bot or getattr(self.selfcoding_manager, "bot_name", ""))

        bus = (
            getattr(self.selfcoding_manager, "event_bus", None)
            or self.event_bus
            or getattr(self.data_bot, "event_bus", None)
        )
        try:
            registry = getattr(self.selfcoding_manager, "bot_registry", None)
            module_path: Path | None = None
            if not registry or not bot or bot not in getattr(registry, "graph", {}):
                if bus:
                    try:
                        bus.publish("evolve:unknown_bot", {"bot": bot})
                    except Exception:
                        self.logger.exception(
                            "failed to publish unknown bot event for %s", bot
                        )
                else:
                    self.logger.error("unknown bot %s", bot)
                return
            try:
                mod_path = registry.graph.nodes[bot]["module"]
                p = Path(mod_path)
                if p.exists():
                    module_path = p
            except Exception:
                self.logger.exception("bot registry lookup failed for %s", bot)
                return
            if not module_path or not module_path.exists():
                self.logger.error("module path not found for %s", bot)
                return

            roi_baseline = float(event.get("roi_baseline", 0.0))
            delta_roi = float(event.get("delta_roi", 0.0))
            delta_errors = float(event.get("delta_errors", 0.0))
            after_roi = roi_baseline + delta_roi
            if getattr(self.data_bot, "log_evolution_cycle", None):
                try:
                    self.data_bot.log_evolution_cycle(
                        "degradation",
                        roi_baseline,
                        after_roi,
                        delta_roi,
                        after_roi,
                        roi_delta=delta_roi,
                        anomaly_count=delta_errors,
                        reason="bot degraded",
                        trigger="degradation",
                    )
                except Exception:
                    self.logger.exception(
                        "failed to log degradation cycle for %s", bot
                    )

            current_roi = after_roi
            current_err = float(event.get("errors_baseline", 0.0)) + delta_errors
            predicted_roi = current_roi
            predicted_gain = 0.0
            confidence = 0.0
            horizon = 3
            if self.roi_predictor or self.trend_predictor:
                try:
                    if self.roi_predictor:
                        features = [[current_roi, current_err]] * horizon
                        seq, _, confs, _ = self.roi_predictor.predict(
                            features, horizon=horizon
                        )
                        if seq:
                            last = seq[-1]
                            predicted_roi = float(
                                last[-1] if isinstance(last, (list, tuple)) else last
                            )
                        if confs:
                            c_last = confs[-1]
                            confidence = float(
                                c_last[-1] if isinstance(c_last, (list, tuple)) else c_last
                            )
                    elif self.trend_predictor:
                        pred = self.trend_predictor.predict_future_metrics(horizon)
                        final = pred[-1] if isinstance(pred, (list, tuple)) else pred
                        predicted_roi = float(getattr(final, "roi", current_roi))
                        confidence = float(
                            getattr(final, "confidence", getattr(final, "roi_confidence", 1.0))
                        )
                    predicted_gain = predicted_roi - current_roi
                except Exception:
                    self.logger.exception("roi prediction failed for %s", bot)
            decision = (
                "patch"
                if predicted_gain >= self.roi_gain_floor
                and confidence >= self.roi_confidence_floor
                else "skip"
            )
            try:
                self.history.add(
                    EvolutionEvent(
                        action=decision,
                        before_metric=current_roi,
                        after_metric=predicted_roi,
                        roi=predicted_gain,
                        predicted_roi=predicted_roi,
                        confidence=confidence,
                        reason=decision,
                        trigger="degradation",
                        performance=predicted_gain,
                    )
                )
            except Exception:
                self.logger.exception("failed to record roi prediction")
            if decision == "skip":
                reason = "roi_prediction" if predicted_gain < self.roi_gain_floor else "confidence"
                self.logger.info(
                    "patch_skip_%s",
                    reason,
                    extra={
                        "bot": bot,
                        "predicted_gain": predicted_gain,
                        "confidence": confidence,
                    },
                )
                if bus:
                    try:
                        bus.publish(
                            "bot:patch_skipped",
                            {"bot": bot, "reason": reason},
                        )
                    except Exception:
                        self.logger.exception(
                            "failed to publish patch_skipped for %s", bot
                        )
                return
            try:
                if not self.selfcoding_manager.should_refactor():
                    self.logger.info(
                        "patch_skip_thresholds",
                        extra={"bot": bot},
                    )
                    if bus:
                        try:
                            bus.publish(
                                "bot:patch_skipped",
                                {"bot": bot, "reason": "thresholds"},
                            )
                        except Exception:
                            self.logger.exception(
                                "failed to publish patch_skipped for %s", bot
                            )
                    return
                settings = getattr(self.data_bot, "settings", SandboxSettings())
                data_dir = Path(getattr(settings, "sandbox_data_dir", "."))
                os.environ["SANDBOX_DATA_DIR"] = str(data_dir)
                builder = create_context_builder()
                try:
                    ensure_fresh_weights(builder)
                except Exception:
                    self.logger.exception(
                        "failed to refresh context builder for %s", bot
                    )
                    if bus:
                        try:
                            bus.publish(
                                "bot:patch_failed",
                                {"bot": bot, "stage": "context", "error": "refresh_db_weights"},
                            )
                        except Exception:
                            self.logger.exception(
                                "failed to publish patch_failed for %s", bot
                            )
                    return
                self.selfcoding_manager.generate_and_patch(
                    module_path,
                    desc,
                    context_meta=context_meta,
                    context_builder=builder,
                    provenance_token=self.provenance_token,
                )
                patch_id = getattr(self.selfcoding_manager, "_last_patch_id", None)
                commit = getattr(self.selfcoding_manager, "_last_commit_hash", None)
                success = bool(patch_id and commit)
                roi_after = (
                    self.data_bot.roi(bot)
                    if hasattr(self.data_bot, "roi")
                    else current_roi
                )
                err_after = (
                    self.data_bot.average_errors(bot)
                    if hasattr(self.data_bot, "average_errors")
                    else current_err
                )
                roi_delta = roi_after - current_roi
                err_delta = err_after - current_err
                if registry and success:
                    try:
                        def _upd() -> None:
                            registry.update_bot(
                                bot,
                                str(module_path),
                                patch_id=patch_id,
                                commit=commit,
                            )

                        with_retry(_upd, logger=self.logger)
                        node = registry.graph.nodes.get(bot, {})
                        if (
                            node.get("patch_id") != patch_id
                            or node.get("commit") != commit
                        ):
                            success = False
                    except Exception:
                        self.logger.exception("failed to update bot %s", bot)
                        success = False
                payload = {
                    "bot": bot,
                    "patch_id": patch_id,
                    "commit": commit,
                    "roi_before": current_roi,
                    "roi_after": roi_after,
                    "roi_delta": roi_delta,
                    "errors_before": current_err,
                    "errors_after": err_after,
                    "error_delta": err_delta,
                    "description": desc,
                    "path": str(module_path),
                }
                if bus:
                    try:
                        topic = "bot:patched" if success else "bot:patch_failed"
                        bus.publish(topic, payload)
                    except Exception:
                        self.logger.exception(
                            "failed to publish %s for %s", topic, bot
                        )
                try:
                    self.history.add(
                        EvolutionEvent(
                            action="patch" if success else "patch_failed",
                            before_metric=current_roi,
                            after_metric=roi_after if success else current_roi,
                            roi=roi_delta if success else 0.0,
                            patch_id=patch_id,
                            reason=desc,
                            trigger="degradation",
                            performance=roi_delta,
                            bottleneck=err_delta,
                        )
                    )
                except Exception:
                    self.logger.exception("failed to record patch result")
            except HelperGenerationError as exc:
                self.logger.error(
                    "context_build_failed",
                    exc_info=True,
                    extra={"bot": bot},
                )
                if bus:
                    try:
                        bus.publish(
                            "bot:patch_failed",
                            {"bot": bot, "stage": "context", "error": str(exc)},
                        )
                    except Exception:
                        self.logger.exception(
                            "failed to publish patch_failed for %s", bot
                        )
            except Exception as exc:
                self.logger.error(
                    "patch_failed",
                    exc_info=True,
                    extra={"bot": bot},
                )
                if bus:
                    try:
                        bus.publish(
                            "bot:patch_failed",
                            {"bot": bot, "stage": "patch", "error": str(exc)},
                        )
                    except Exception:
                        self.logger.exception(
                            "failed to publish patch_failed for %s", bot
                        )
        except Exception:
            self.logger.exception(
                "failed to self patch after degradation of %s", bot
            )

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
                score = float(hist[0][0])
                self._cached_eval_score = score
                return score
        except Exception:
            self.logger.exception("failed to fetch latest eval score")
            return self._cached_eval_score
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
        if self.selfcoding_manager and self._pending_patch_cycle:
            pending = list(self._pending_patch_cycle)
            for bot in pending:
                try:
                    self._on_bot_degraded({"bot": bot})
                except Exception:
                    self.logger.exception(
                        "failed to process pending patch for %s", bot
                    )
            self._pending_patch_cycle.clear()
        before_roi = self._latest_roi()
        delta_roi = before_roi - self.prev_roi
        self.prev_roi = before_roi
        error_rate = self._error_rate()
        delta_err = error_rate - getattr(self, "prev_error", 0.0)
        self.prev_error = error_rate
        try:
            from . import mutation_logger as MutationLogger
        except Exception:  # pragma: no cover - best effort
            MutationLogger = None  # type: ignore

        def _self_patch(module: object, reason: str, trigger: str) -> None:
            if not self.selfcoding_manager:
                return
            if not self.selfcoding_manager.should_refactor():
                return
            try:
                mod = inspect.getmodule(module.__class__)
                path = Path(getattr(mod, "__file__", "")) if mod else None
                if not path or not path.exists():
                    return
                meta = {
                    "delta_roi": delta_roi,
                    "delta_errors": delta_err,
                    "roi_threshold": self.triggers.roi_drop,
                    "error_threshold": self.triggers.error_rate,
                }
                self._invoke_register_patch_cycle(reason, meta)
                self.selfcoding_manager.run_patch(
                    path,
                    reason,
                    provenance_token=self.provenance_token,
                    context_meta=meta,
                )
                after = self._latest_roi()
                patch_id = getattr(self.selfcoding_manager, "_last_patch_id", None)
                self.history.add(
                    EvolutionEvent(
                        action=f"patch:{path.name}",
                        before_metric=before_roi,
                        after_metric=after,
                        roi=after - before_roi,
                        patch_id=patch_id,
                        reason=reason,
                        trigger=trigger,
                        performance=delta_roi,
                        bottleneck=delta_err,
                    )
                )
            except Exception:
                self.logger.exception("self coding patch failed")

        if error_rate > self.triggers.error_rate:
            _self_patch(
                self.improvement_engine,
                f"error_rate {error_rate:.2f} > {self.triggers.error_rate:.2f}",
                "error_rate",
            )
        if delta_roi <= self.triggers.roi_drop:
            _self_patch(
                self.evolution_manager,
                f"roi_drop {delta_roi:.2f} <= {self.triggers.roi_drop:.2f}",
                "roi_drop",
            )

        if self.roi_predictor:
            try:
                seq, _, _, _ = self.roi_predictor.predict(
                    [[before_roi, error_rate]], horizon=1
                )
                model_pred = float(seq[-1]) if seq else 0.0
            except TypeError:
                model_pred, _, _, _ = self.roi_predictor.predict(
                    [[before_roi, error_rate]]
                )
        else:
            seq = []
            model_pred = 0.0
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

        def close(val: float, thr: float) -> bool:
            return abs(val - thr) <= abs(thr) * 0.1
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
            scores: dict[str, float] = {}
            for cand in candidates:
                try:
                    mean, var = self.multi_predictor.predict(cand, before_roi)
                except Exception:
                    mean = var = 0.0
                predictions[cand] = mean
                variances[cand] = var
                scores[cand] = mean - var
            if scores:
                best_act = max(scores, key=scores.get)
                sequence = [best_act]
                predicted_action_roi = predictions.get(best_act, 0.0)
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
                if self.selfcoding_manager:
                    failing = (
                        getattr(res, "failing_path", None)
                        or getattr(res, "failing_module", None)
                        or getattr(self.improvement_engine, "failing_path", None)
                    )
                    path: Path | None = None
                    if failing:
                        if isinstance(failing, (str, Path)):
                            path = Path(failing)
                        else:
                            mod = inspect.getmodule(
                                getattr(failing, "__class__", failing)
                            )
                            if mod:
                                path = Path(getattr(mod, "__file__", ""))
                    if path and path.exists():
                        try:
                            from . import mutation_logger as MutationLogger
                            event_id = MutationLogger.log_mutation(
                                change=f"patch:{path.name}",
                                reason="self_improvement",
                                trigger="self_improvement",
                                performance=0.0,
                                workflow_id=0,
                                before_metric=before_roi,
                            )
                            if self.selfcoding_manager.should_refactor():
                                self.selfcoding_manager.run_patch(
                                    path,
                                    f"auto_patch:{path.name}",
                                    provenance_token=self.provenance_token,
                                )
                            after_patch = self._latest_roi()
                            delta = after_patch - before_roi
                            MutationLogger.record_mutation_outcome(
                                event_id,
                                after_metric=after_patch,
                                roi=delta,
                                performance=delta,
                            )
                            registry = getattr(
                                self.selfcoding_manager, "bot_registry", None
                            )
                            if registry:
                                try:
                                    registry.register_interaction(
                                        self.selfcoding_manager.bot_name, path.stem
                                    )
                                except Exception:
                                    self.logger.exception(
                                        "bot registry update failed"
                                    )
                        except Exception:
                            self.logger.exception("self patch failed")
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
            if MutationLogger:
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
            proposals = list(
                self.workflow_evolver.generate_variants(limit, workflow_id=0)
            )
        except Exception:
            proposals = []
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
        mdb = (
            getattr(self.resource_optimizer, "menace_db", None)
            if self.resource_optimizer
            else None
        )
        if not mdb:
            return
        try:
            with mdb.engine.begin() as conn:
                rows = (
                    conn.execute(
                        mdb.workflows.select().where(
                            mdb.workflows.c.status == "paused"
                        )
                    )
                    .mappings()
                    .fetchall()
                )
                for row in rows:
                    conn.execute(
                        mdb.workflows.delete().where(
                            mdb.workflows.c.workflow_id == row["workflow_id"]
                        )
                    )
        except Exception:
            self.logger.exception("cleanup workflows failed")

    # ------------------------------------------------------------------
    def _replace_main_workflow(self, name: str) -> None:
        """Set *name* as the active workflow in MenaceDB."""
        mdb = (
            getattr(self.resource_optimizer, "menace_db", None)
            if self.resource_optimizer
            else None
        )
        if not mdb:
            return
        try:
            with mdb.engine.begin() as conn:
                row = (
                    conn.execute(
                        mdb.workflows.select().where(
                            mdb.workflows.c.workflow_name == name
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
            self.logger.exception("replace main workflow failed")


__all__ = ["EvolutionTrigger", "EvolutionOrchestrator"]
