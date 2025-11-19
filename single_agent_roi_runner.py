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
import logging
import os
import sys
import time
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

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
from context_builder_util import create_context_builder

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


@dataclass
class EvaluationResult:
    roi_after: float
    confidence: float
    metrics: dict[str, float]
    error: str | None = None


def build_manager(bot_name: str) -> SelfCodingManager:
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
    manager = internalize_coding_bot(
        bot_name,
        engine,
        pipeline,
        data_bot=data_bot,
        bot_registry=registry,
        roi_threshold=thresholds.roi_drop,
        error_threshold=thresholds.error_increase,
        test_failure_threshold=thresholds.test_failure_increase,
        threshold_service=ThresholdService(),
    )
    promote_pipeline(manager)
    return manager


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

    reasons: list[str] = []
    stop_reason: str | None = None
    if raroi < cfg.roi_target:
        stop_reason = f"RAROI {raroi:.3f} below target {cfg.roi_target:.3f}"
    elif confidence < cfg.min_confidence:
        stop_reason = f"Confidence {confidence:.3f} below minimum {cfg.min_confidence:.3f}"
    elif safety_factor < 1.0 / max(1.0, cfg.catastrophic_multiplier):
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

    tracker = ROITracker()
    manager: SelfCodingManager | None = None
    if not cfg.dry_run:
        try:
            manager = build_manager(args.bot_name)
        except Exception:
            LOGGER.exception("Failed to initialise SelfCodingManager; continuing without it")

    if manager is None and not cfg.dry_run:
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

    last_result: CycleResult | None = None
    for cycle in range(1, cfg.max_cycles + 1):
        result = run_cycle(manager, tracker, cfg, args.bot_name)
        last_result = result
        LOGGER.info(
            "Cycle %d: ROI %.3f -> %.3f | RAROI %.3f | confidence %.3f | safety %.3f",
            cycle,
            result.roi_before,
            result.roi_after,
            result.raroi,
            result.confidence,
            result.safety_factor,
        )
        for bucket, hint in result.suggestions:
            LOGGER.info("Bottleneck suggestion (%s): %s", bucket, hint)

        if result.should_stop:
            LOGGER.info("Stopping after cycle %d: %s", cycle, result.reason)
            break
        time.sleep(0.5)

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
