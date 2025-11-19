"""Lightweight single-agent ROI experiment runner.

This script bypasses the meta-planner/workflow-evolution layers and talks to the
self-coding agent/manager directly so you can experiment with ROI and
risk-adjusted ROI (RAROI) in isolation. Use it when you want a single agent to
run iterative ROI cycles without any orchestration from the broader workflow
stack.

The runner expects observable metrics to come from telemetry and tracker inputs
such as revenue proxies, latency/error deltas and recovery time. These signals
are combined with the agent's own helper output to evaluate ROI deltas in a
pluggable way rather than assuming a fixed increment.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

import shared_event_bus
from context_builder_util import create_context_builder
from menace_sandbox.bot_registry import BotRegistry
from menace_sandbox.code_database import CodeDB
from menace_sandbox.coding_bot_interface import (
    fallback_helper_manager,
    prepare_pipeline_for_bootstrap,
)
from menace_sandbox.data_bot import DataBot
from menace_sandbox.menace_memory_manager import MenaceMemoryManager
from menace_sandbox.model_automation_pipeline import ModelAutomationPipeline
from menace_sandbox.self_coding_engine import SelfCodingEngine
from menace_sandbox.self_coding_manager import SelfCodingManager, internalize_coding_bot
from menace_sandbox.self_coding_thresholds import get_thresholds
from menace_sandbox.threshold_service import ThresholdService
from roi_tracker import ROITracker
from self_improvement.workflow_discovery import discover_workflow_specs
from shared_event_bus import event_bus as _GLOBAL_EVENT_BUS

# ``calculate_raroi`` is an instance method on ROITracker; alias it for clarity
# so callers can import the toolkit components directly from this runner.
calculate_raroi = ROITracker.calculate_raroi

LOGGER = logging.getLogger("single_agent_roi_runner")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


def _latest_metric_delta(tracker: ROITracker, names: Iterable[str]) -> float:
    for name in names:
        history = tracker.synergy_metrics_history.get(name) or tracker.metrics_history.get(
            name
        )
        if history:
            if len(history) >= 2:
                return history[-1] - history[-2]
            return history[-1]
    return 0.0


def compute_roi_delta(
    manager: SelfCodingManager | None,
    bot_name: str,
    helper_text: str | None,
    tracker: ROITracker,
) -> EvaluationResult:
    """Compute ROI delta from telemetry and agent output.

    The evaluator pulls recent telemetry deltas (revenue proxy, latency/error
    deltas and recovery time) from the tracker so the resulting ROI reflects real
    performance instead of a fixed increment. Errors are surfaced via
    ``EvaluationResult.error`` and a dedicated ``evaluation_error`` metric so the
    tracker can record degraded confidence.
    """

    roi_before = tracker.roi_history[-1] if tracker.roi_history else 1.0
    metrics: dict[str, float] = {}
    confidence = 0.5

    try:
        revenue_delta = _latest_metric_delta(
            tracker, ["synergy_revenue", "synergy_profitability", "synergy_projected_lucrativity"]
        )
        latency_delta = _latest_metric_delta(
            tracker, ["latency_error_rate", "synergy_latency_error_rate", "recovery_time"]
        )
        error_delta = _latest_metric_delta(
            tracker,
            [
                "hostile_failures",
                "misuse_failures",
                "synergy_hostile_failures",
                "synergy_misuse_failures",
            ],
        )

        metrics.update(
            {
                "revenue_proxy_delta": revenue_delta,
                "latency_error_delta": latency_delta,
                "failure_delta": error_delta,
            }
        )

        roi_delta = revenue_delta - max(latency_delta, 0.0) - max(error_delta, 0.0)
        roi_after = max(0.0, roi_before + roi_delta)
        confidence = min(1.0, max(0.0, 0.4 + abs(roi_delta) + (0.1 if helper_text else 0.0)))
    except Exception as exc:  # pragma: no cover - defensive path
        LOGGER.exception("ROI evaluation failed for %s", bot_name)
        metrics["evaluation_error"] = 1.0
        return EvaluationResult(roi_after=roi_before, confidence=0.0, metrics=metrics, error=str(exc))

    metrics.setdefault("evaluation_error", 0.0)
    return EvaluationResult(roi_after=roi_after, confidence=confidence, metrics=metrics)


@dataclass
class ROIConfig:
    roi_target: float
    min_confidence: float
    catastrophic_multiplier: float
    max_cycles: int
    dry_run: bool


@dataclass
class CycleResult:
    roi_before: float
    roi_after: float
    raroi: float
    confidence: float
    safety_factor: float
    suggestions: Iterable[Tuple[str, str]]
    should_stop: bool
    reason: str | None
    stopped_due_to_raroi: bool
    stopped_due_to_safety: bool
    stopped_due_to_confidence: bool


@dataclass
class EvaluationResult:
    roi_after: float
    confidence: float
    metrics: dict[str, float]
    error: str | None = None


@dataclass
class SelfCodingBootstrap:
    manager: SelfCodingManager
    engine: SelfCodingEngine
    pipeline: ModelAutomationPipeline
    data_bot: DataBot
    registry: BotRegistry
    promote_pipeline: Callable[[SelfCodingManager], None]
    threshold_service: ThresholdService


class DiscoveryPlan:
    """Track workflow discovery attempts and prioritise dormant modules."""

    def __init__(
        self,
        registry: BotRegistry,
        *,
        state_path: Path | None = None,
    ) -> None:
        self.registry = registry
        self.state_path = state_path or (REPO_ROOT / ".single_agent_discovery_plan.json")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()

    def _default_state(self) -> dict[str, object]:
        return {
            "round_id": 0,
            "attempt_counts": {},
            "attempted_in_round": {},
            "outcomes": {},
        }

    def _load_state(self) -> dict[str, object]:
        try:
            if self.state_path.exists():
                with self.state_path.open("r", encoding="utf-8") as handle:
                    raw = json.load(handle)
                    if isinstance(raw, dict):
                        return raw
        except Exception:
            LOGGER.exception("Failed to load discovery plan state from %s", self.state_path)
        return self._default_state()

    def _save_state(self) -> None:
        try:
            with self.state_path.open("w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2, sort_keys=True)
        except Exception:
            LOGGER.exception("Failed to persist discovery plan state to %s", self.state_path)

    def _extract_workflow_id(self, spec: Mapping[str, object]) -> str | None:
        metadata = spec.get("metadata") if isinstance(spec, Mapping) else None
        if isinstance(metadata, Mapping):
            workflow_id = metadata.get("workflow_id")
            if isinstance(workflow_id, str):
                return workflow_id
        workflow = spec.get("workflow") if isinstance(spec, Mapping) else None
        if isinstance(workflow, list) and workflow:
            first = workflow[0]
            if isinstance(first, str):
                return first
        return None

    def _discover_modules(self) -> list[str]:
        discovered = []
        try:
            specs = discover_workflow_specs(base_path=REPO_ROOT, logger=LOGGER)
        except Exception:
            LOGGER.exception("Failed to discover workflow specs; using empty plan")
            return []
        for spec in specs:
            module = self._extract_workflow_id(spec)
            if module:
                discovered.append(module)
        return sorted(set(discovered))

    def _dormant_modules(self) -> list[str]:
        active = set(self.registry.graph.nodes)
        return [module for module in self._discover_modules() if module not in active]

    def _prune_state(self, active_modules: set[str]) -> None:
        attempt_counts = self.state.get("attempt_counts")
        if isinstance(attempt_counts, dict):
            for key in list(attempt_counts):
                if key not in active_modules:
                    attempt_counts.pop(key, None)
        attempted_round = self.state.get("attempted_in_round")
        if isinstance(attempted_round, dict):
            for key in list(attempted_round):
                if key not in active_modules:
                    attempted_round.pop(key, None)
        outcomes = self.state.get("outcomes")
        if isinstance(outcomes, dict):
            for key in list(outcomes):
                if key not in active_modules:
                    outcomes.pop(key, None)

    def _priority_key(self, module: str, suggestions: Iterable[Tuple[str, str]] | None) -> tuple[int, int, str]:
        attempt_counts = self.state.get("attempt_counts")
        attempts = 0
        if isinstance(attempt_counts, dict):
            attempts = int(attempt_counts.get(module, 0))
        module_lower = module.lower()
        suggestion_match = 1
        if suggestions:
            for bucket, hint in suggestions:
                bucket_lower = bucket.lower() if isinstance(bucket, str) else ""
                hint_lower = hint.lower() if isinstance(hint, str) else ""
                if bucket_lower and (bucket_lower in module_lower or module_lower in bucket_lower):
                    suggestion_match = 0
                    break
                if hint_lower and module_lower in hint_lower:
                    suggestion_match = 0
                    break
        return (suggestion_match, attempts, module)

    def next_candidate(self, suggestions: Iterable[Tuple[str, str]] | None = None) -> str | None:
        modules = self._dormant_modules()
        if not modules:
            return None
        active = set(modules)
        self._prune_state(active)
        round_id = int(self.state.get("round_id", 0))
        attempted_round = self.state.get("attempted_in_round")
        if not isinstance(attempted_round, dict):
            attempted_round = {}
            self.state["attempted_in_round"] = attempted_round
        available = [m for m in modules if attempted_round.get(m) != round_id]
        if not available:
            round_id += 1
            self.state["round_id"] = round_id
            available = modules
            attempted_round.clear()
        candidate = sorted(available, key=lambda mod: self._priority_key(mod, suggestions))[0]
        attempt_counts = self.state.get("attempt_counts")
        if not isinstance(attempt_counts, dict):
            attempt_counts = {}
            self.state["attempt_counts"] = attempt_counts
        attempt_counts[candidate] = int(attempt_counts.get(candidate, 0)) + 1
        attempted_round[candidate] = round_id
        self._save_state()
        return candidate

    def record_outcome(self, module: str, *, status: str, error: str | None = None) -> None:
        outcomes = self.state.get("outcomes")
        if not isinstance(outcomes, dict):
            outcomes = {}
            self.state["outcomes"] = outcomes
        entry = {
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if error:
            entry["error"] = error
        outcomes[module] = entry
        self._save_state()

    def internalize_next(
        self,
        bootstrap: SelfCodingBootstrap,
        suggestions: Iterable[Tuple[str, str]] | None = None,
    ) -> str | None:
        candidate = self.next_candidate(suggestions)
        if candidate is None:
            return None
        thresholds = get_thresholds(candidate)
        try:
            manager = internalize_coding_bot(
                candidate,
                bootstrap.engine,
                bootstrap.pipeline,
                data_bot=bootstrap.data_bot,
                bot_registry=bootstrap.registry,
                roi_threshold=thresholds.roi_drop,
                error_threshold=thresholds.error_increase,
                test_failure_threshold=thresholds.test_failure_increase,
                threshold_service=bootstrap.threshold_service,
            )
            bootstrap.promote_pipeline(manager)
            bootstrap.manager = manager
        except Exception as exc:
            reason = f"internalize_failed: {exc}"
            LOGGER.warning("Workflow internalization skipped for %s: %s", candidate, exc)
            self.record_outcome(candidate, status="skipped", error=str(exc))
            self._publish_event(
                "workflow_discovery:skipped",
                {
                    "workflow": candidate,
                    "reason": "internalize_failed",
                    "error": str(exc),
                },
            )
            return None
        self.record_outcome(candidate, status="internalized", error=None)
        attempt_counts = self.state.get("attempt_counts")
        if isinstance(attempt_counts, dict):
            attempts_recorded = attempt_counts.get(candidate, 1)
        else:
            attempts_recorded = 1
        self._publish_event(
            "workflow_discovery:internalized",
            {
                "workflow": candidate,
                "attempts": attempts_recorded,
            },
        )
        return candidate

    def _publish_event(self, topic: str, payload: Mapping[str, object]) -> None:
        bus = getattr(shared_event_bus, "event_bus", None) or _GLOBAL_EVENT_BUS
        if bus is None:
            return
        try:
            bus.publish(topic, dict(payload))
        except Exception:  # pragma: no cover - best effort logging
            LOGGER.debug("Failed to publish discovery event %s", topic, exc_info=True)


class MetricsWriter:
    """Persist cycle metrics for auditing.

    Supports JSONL (append-only) and SQLite backends so production runs can keep
    a durable audit log alongside console output.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        self.is_jsonl = suffix in {".jsonl", ".json"}
        self.is_sqlite = suffix in {".db", ".sqlite", ".sqlite3"}

        if not (self.is_jsonl or self.is_sqlite):
            raise ValueError("--metrics-path must point to a .jsonl or SQLite file")

        if self.is_sqlite:
            self._initialise_db()

    def _initialise_db(self) -> None:
        try:
            conn = sqlite3.connect(self.path)
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cycle_metrics (
                        cycle INTEGER,
                        roi_before REAL,
                        roi_after REAL,
                        raroi REAL,
                        confidence REAL,
                        safety_factor REAL,
                        stop_reason TEXT,
                        stopped_due_to_raroi INTEGER,
                        stopped_due_to_safety INTEGER,
                        stopped_due_to_confidence INTEGER,
                        suggestions_json TEXT
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS run_summary (
                        started_at TEXT,
                        ended_at TEXT,
                        cycles_executed INTEGER,
                        stopped_due_to_raroi INTEGER,
                        stopped_due_to_safety INTEGER,
                        stopped_due_to_confidence INTEGER,
                        stop_reason TEXT
                    )
                    """
                )
        except Exception:
            LOGGER.exception("Failed to initialise metrics database at %s", self.path)
            raise

    def _serialise_suggestions(self, suggestions: Iterable[Tuple[str, str]]) -> list[dict[str, str]]:
        return [{"bucket": bucket, "hint": hint} for bucket, hint in suggestions]

    def record_cycle(self, cycle: int, result: CycleResult) -> None:
        payload = {
            "cycle": cycle,
            "roi_before": result.roi_before,
            "roi_after": result.roi_after,
            "raroi": result.raroi,
            "confidence": result.confidence,
            "safety_factor": result.safety_factor,
            "stop_reason": result.reason,
            "stopped_due_to_raroi": result.stopped_due_to_raroi,
            "stopped_due_to_safety": result.stopped_due_to_safety,
            "stopped_due_to_confidence": result.stopped_due_to_confidence,
            "suggestions": self._serialise_suggestions(result.suggestions),
        }
        try:
            if self.is_jsonl:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload) + "\n")
            else:
                conn = sqlite3.connect(self.path)
                with conn:
                    conn.execute(
                        """
                        INSERT INTO cycle_metrics (
                            cycle, roi_before, roi_after, raroi, confidence, safety_factor,
                            stop_reason, stopped_due_to_raroi, stopped_due_to_safety,
                            stopped_due_to_confidence, suggestions_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload["cycle"],
                            payload["roi_before"],
                            payload["roi_after"],
                            payload["raroi"],
                            payload["confidence"],
                            payload["safety_factor"],
                            payload["stop_reason"],
                            int(payload["stopped_due_to_raroi"]),
                            int(payload["stopped_due_to_safety"]),
                            int(payload["stopped_due_to_confidence"]),
                            json.dumps(payload["suggestions"]),
                        ),
                    )
        except Exception:
            LOGGER.exception("Failed to persist cycle %s metrics to %s", cycle, self.path)

    def record_summary(
        self,
        started_at: datetime,
        ended_at: datetime,
        cycles_executed: int,
        last_result: CycleResult | None,
    ) -> None:
        payload = {
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "cycles_executed": cycles_executed,
            "stopped_due_to_raroi": bool(last_result and last_result.stopped_due_to_raroi),
            "stopped_due_to_safety": bool(last_result and last_result.stopped_due_to_safety),
            "stopped_due_to_confidence": bool(
                last_result and last_result.stopped_due_to_confidence
            ),
            "stop_reason": last_result.reason if last_result else None,
        }
        try:
            if self.is_jsonl:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"summary": payload}) + "\n")
            else:
                conn = sqlite3.connect(self.path)
                with conn:
                    conn.execute(
                        """
                        INSERT INTO run_summary (
                            started_at, ended_at, cycles_executed, stopped_due_to_raroi,
                            stopped_due_to_safety, stopped_due_to_confidence, stop_reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload["started_at"],
                            payload["ended_at"],
                            payload["cycles_executed"],
                            int(payload["stopped_due_to_raroi"]),
                            int(payload["stopped_due_to_safety"]),
                            int(payload["stopped_due_to_confidence"]),
                            payload["stop_reason"],
                        ),
                    )
        except Exception:
            LOGGER.exception("Failed to persist summary metrics to %s", self.path)


def build_manager(bot_name: str) -> SelfCodingBootstrap:
    registry = BotRegistry()
    data_bot = DataBot(start_server=False)
    builder = create_context_builder()
    engine = SelfCodingEngine(CodeDB(), MenaceMemoryManager(), context_builder=builder)

    with fallback_helper_manager(bot_registry=registry, data_bot=data_bot) as bootstrap_manager:
        pipeline, promote_pipeline = prepare_pipeline_for_bootstrap(
            pipeline_cls=ModelAutomationPipeline,
            context_builder=builder,
            bot_registry=registry,
            data_bot=data_bot,
            bootstrap_runtime_manager=bootstrap_manager,
            manager=bootstrap_manager,
        )

    thresholds = get_thresholds(bot_name)
    threshold_service = ThresholdService()
    manager = internalize_coding_bot(
        bot_name,
        engine,
        pipeline,
        data_bot=data_bot,
        bot_registry=registry,
        roi_threshold=thresholds.roi_drop,
        error_threshold=thresholds.error_increase,
        test_failure_threshold=thresholds.test_failure_increase,
        threshold_service=threshold_service,
    )
    promote_pipeline(manager)
    return SelfCodingBootstrap(
        manager=manager,
        engine=engine,
        pipeline=pipeline,
        data_bot=data_bot,
        registry=registry,
        promote_pipeline=promote_pipeline,
        threshold_service=threshold_service,
    )


def run_cycle(
    manager: SelfCodingManager | None,
    tracker: ROITracker,
    cfg: ROIConfig,
    bot_name: str,
) -> CycleResult:
    roi_before = tracker.roi_history[-1] if tracker.roi_history else 1.0
    roi_after = roi_before
    confidence = cfg.min_confidence
    suggestions: Iterable[Tuple[str, str]] = []
    metrics: dict[str, float] = {}
    evaluation_error: str | None = None
    manager_missing_reason: str | None = None

    def _validate_inputs(result: EvaluationResult) -> tuple[float, float, dict[str, float]]:
        nonlocal evaluation_error
        validated_roi = result.roi_after
        validated_confidence = result.confidence
        validated_metrics = dict(result.metrics)

        if result.error:
            evaluation_error = result.error
            validated_metrics["evaluation_error"] = validated_metrics.get("evaluation_error", 0.0) + 1.0
            validated_confidence = 0.0

        if not math.isfinite(validated_roi) or validated_roi <= 0.0:
            evaluation_error = evaluation_error or "roi_after_invalid"
            validated_metrics["evaluation_error"] = validated_metrics.get("evaluation_error", 0.0) + 1.0
            validated_roi = roi_before

        if not math.isfinite(validated_confidence) or not (0.0 <= validated_confidence <= 1.0):
            evaluation_error = evaluation_error or "confidence_invalid"
            validated_metrics["evaluation_error"] = validated_metrics.get("evaluation_error", 0.0) + 1.0
            validated_confidence = 0.0

        return validated_roi, validated_confidence, validated_metrics

    if cfg.dry_run:
        LOGGER.info("Dry-run enabled; skipping agent invocation.")
        result = EvaluationResult(
            roi_after=roi_before + 0.01,
            confidence=1.0,
            metrics={"dry_run": 1.0},
        )
        roi_after, confidence, metrics = _validate_inputs(result)
        if manager is None:
            manager_missing_reason = "SelfCodingManager unavailable; simulating cycle"
    elif manager is not None:
        try:
            helper_text = manager.engine.generate_helper(
                description=f"Single-agent ROI probe for {bot_name}",
                strategy="roi-feedback",
            )
            LOGGER.info("Agent output (truncated): %s", helper_text[:256])
            result = compute_roi_delta(manager, bot_name, helper_text, tracker)
            roi_after, confidence, metrics = _validate_inputs(result)
        except Exception:
            LOGGER.exception("Agent invocation failed; marking ROI as unchanged.")
            confidence = 0.0
            metrics["evaluation_error"] = metrics.get("evaluation_error", 0.0) + 1.0
    else:
        LOGGER.warning("No manager available; treating cycle as no-op.")
        manager_missing_reason = "SelfCodingManager unavailable; cycle executed without agent"

    if evaluation_error:
        LOGGER.warning("Evaluation error recorded: %s", evaluation_error)

    _vertex, _predictions, should_stop, _entropy_cap = tracker.update(
        roi_before,
        roi_after,
        confidence=confidence,
        metrics=metrics,
        profile_type="single_agent",
    )

    _base_roi, raroi, suggestions = tracker.calculate_raroi(
        roi_after,
        impact_severity=min(1.0, cfg.catastrophic_multiplier),
    )
    safety_factor = raroi / roi_after if roi_after else 0.0
    suggestions = list(suggestions)

    reasons: list[str] = []
    stop_reason: str | None = None
    stop_due_to_raroi = raroi < cfg.roi_target
    stop_due_to_confidence = confidence < cfg.min_confidence
    stop_due_to_safety = safety_factor < 1.0 / max(1.0, cfg.catastrophic_multiplier)

    if stop_due_to_raroi:
        stop_reason = f"RAROI {raroi:.3f} below target {cfg.roi_target:.3f}"
    elif stop_due_to_confidence:
        stop_reason = f"Confidence {confidence:.3f} below minimum {cfg.min_confidence:.3f}"
    elif stop_due_to_safety:
        stop_reason = (
            f"Safety factor {safety_factor:.3f} below conservative bound "
            f"{1.0 / max(1.0, cfg.catastrophic_multiplier):.3f}"
        )
    elif should_stop:
        stop_reason = "Tracker requested stop (entropy or tolerance triggered)"

    if stop_reason:
        reasons.append(stop_reason)
    if manager_missing_reason:
        reasons.append(manager_missing_reason)

    reason = " | ".join(dict.fromkeys(reasons)) or None

    return CycleResult(
        roi_before=roi_before,
        roi_after=roi_after,
        raroi=raroi,
        confidence=confidence,
        safety_factor=safety_factor,
        suggestions=suggestions,
        should_stop=should_stop or stop_reason is not None,
        reason=reason,
        stopped_due_to_raroi=stop_due_to_raroi,
        stopped_due_to_safety=stop_due_to_safety,
        stopped_due_to_confidence=stop_due_to_confidence,
    )


def parse_args() -> tuple[argparse.Namespace, ROIConfig]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Metrics sources: telemetry-backed revenue proxies (e.g. synergy_revenue), "
            "latency/error deltas (latency_error_rate, hostile_failures) and recovery time. "
            "Populate tracker metrics to avoid defaulting to no-op deltas and keep RAROI "
            "aligned with production behaviour."
        ),
    )
    parser.add_argument("--bot-name", default=os.getenv("SINGLE_AGENT_BOT", "menace"))
    parser.add_argument(
        "--roi-target",
        type=float,
        default=_env_float("SINGLE_AGENT_ROI_TARGET", 1.0),
        help="RAROI threshold required to continue running additional cycles.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=_env_float("SINGLE_AGENT_MIN_CONFIDENCE", 0.6),
        help="Minimum confidence score required from the agent before continuing.",
    )
    parser.add_argument(
        "--catastrophic-risk-multiplier",
        type=float,
        default=_env_float("SINGLE_AGENT_CATASTROPHIC_MULTIPLIER", 1.5),
        help=(
            "Multiplier applied to catastrophic risk when evaluating RAROI safety. "
            "Higher values make the runner more conservative."
        ),
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=int(os.getenv("SINGLE_AGENT_MAX_CYCLES", "10")),
        help="Maximum number of single-agent cycles to attempt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=_env_bool("SINGLE_AGENT_DRY_RUN"),
        help="Skip self-coding calls and simulate ROI changes for smoke testing.",
    )
    parser.add_argument(
        "--allow-no-manager",
        action="store_true",
        help=(
            "Allow the runner to continue in simulated mode if the self-coding manager"
            " fails to start. Intended for debugging only."
        ),
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        help=(
            "Optional file path for cycle metrics. Supports .jsonl (append-only) and "
            "SQLite databases (.db/.sqlite/.sqlite3)."
        ),
    )

    args = parser.parse_args()
    cfg = ROIConfig(
        roi_target=args.roi_target,
        min_confidence=args.min_confidence,
        catastrophic_multiplier=args.catastrophic_risk_multiplier,
        max_cycles=args.max_cycles,
        dry_run=args.dry_run,
    )
    return args, cfg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args, cfg = parse_args()
    LOGGER.info(
        "Starting single-agent ROI runner for %s (dry_run=%s)", args.bot_name, cfg.dry_run
    )

    metrics_writer: MetricsWriter | None = None
    if args.metrics_path:
        try:
            metrics_writer = MetricsWriter(args.metrics_path)
            LOGGER.info("Metrics will be written to %s", args.metrics_path)
        except Exception:
            LOGGER.exception("Metrics writer disabled; continuing without persistence")

    tracker = ROITracker()
    bootstrap: SelfCodingBootstrap | None = None
    manager: SelfCodingManager | None = None
    if not cfg.dry_run:
        try:
            bootstrap = build_manager(args.bot_name)
            manager = bootstrap.manager
        except Exception:
            LOGGER.exception("Failed to initialise SelfCodingManager; continuing without it")

    if bootstrap is None and not cfg.dry_run:
        if args.allow_no_manager:
            LOGGER.warning(
                "No manager available; enabling dry-run mode for simulated ROI cycles.")
            cfg.dry_run = True
        else:
            LOGGER.error(
                "No manager available and dry-run not enabled. Use --allow-no-manager to"
                " simulate cycles when debugging."
            )
            sys.exit(1)

    discovery_plan: DiscoveryPlan | None = None
    if bootstrap is not None:
        discovery_plan = DiscoveryPlan(bootstrap.registry)

    active_bot_name = args.bot_name
    plan_suggestions: list[Tuple[str, str]] = []
    last_result: CycleResult | None = None
    start_time = datetime.utcnow()
    cycles_executed = 0
    for cycle in range(1, cfg.max_cycles + 1):
        if not cfg.dry_run and discovery_plan and bootstrap:
            assimilated = discovery_plan.internalize_next(bootstrap, plan_suggestions)
            if assimilated:
                active_bot_name = assimilated
                manager = bootstrap.manager
        result = run_cycle(manager, tracker, cfg, active_bot_name)
        last_result = result
        cycles_executed += 1
        LOGGER.info(
            "Cycle %d: ROI %.3f -> %.3f | RAROI %.3f | confidence %.3f | safety %.3f",
            cycle,
            result.roi_before,
            result.roi_after,
            result.raroi,
            result.confidence,
            result.safety_factor,
        )
        plan_suggestions = list(result.suggestions)
        for bucket, hint in plan_suggestions:
            LOGGER.info("Bottleneck suggestion (%s): %s", bucket, hint)

        if metrics_writer:
            metrics_writer.record_cycle(cycle, result)

        if result.should_stop:
            LOGGER.info("Stopping after cycle %d: %s", cycle, result.reason)
            break
        time.sleep(0.5)

    end_time = datetime.utcnow()
    if metrics_writer:
        metrics_writer.record_summary(start_time, end_time, cycles_executed, last_result)

    if last_result is not None:
        LOGGER.info(
            "Single-agent ROI run complete. Final ROI %.3f -> %.3f | RAROI %.3f | "
            "confidence %.3f | safety %.3f | reasons: %s",
            last_result.roi_before,
            last_result.roi_after,
            last_result.raroi,
            last_result.confidence,
            last_result.safety_factor,
            last_result.reason or "none recorded",
        )
    else:
        LOGGER.info("Single-agent ROI run complete with no cycles executed.")


if __name__ == "__main__":
    main()
