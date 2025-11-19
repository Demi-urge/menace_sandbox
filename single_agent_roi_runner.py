"""Lightweight single-agent ROI experiment runner.

This script bypasses the meta-planner/workflow-evolution layers and talks to the
self-coding agent/manager directly so you can experiment with ROI and
risk-adjusted ROI (RAROI) in isolation. Use it when you want a single agent to
run iterative ROI cycles without any orchestration from the broader workflow
stack.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
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

    if cfg.dry_run:
        LOGGER.info("Dry-run enabled; skipping agent invocation.")
        roi_after = roi_before + 0.01
    elif manager is not None:
        try:
            helper_text = manager.engine.generate_helper(
                description=f"Single-agent ROI probe for {bot_name}",
                strategy="roi-feedback",
            )
            LOGGER.info("Agent output (truncated): %s", helper_text[:256])
            roi_after = roi_before + 0.05
        except Exception:
            LOGGER.exception("Agent invocation failed; marking ROI as unchanged.")
            confidence = 0.0
    else:
        LOGGER.warning("No manager available; treating cycle as no-op.")

    _vertex, _predictions, should_stop, _entropy_cap = tracker.update(
        roi_before,
        roi_after,
        confidence=confidence,
        profile_type="single_agent",
    )

    _base_roi, raroi, suggestions = tracker.calculate_raroi(
        roi_after,
        impact_severity=min(1.0, cfg.catastrophic_multiplier),
    )
    safety_factor = raroi / roi_after if roi_after else 0.0

    reason: str | None = None
    if raroi < cfg.roi_target:
        reason = f"RAROI {raroi:.3f} below target {cfg.roi_target:.3f}"
    elif confidence < cfg.min_confidence:
        reason = f"Confidence {confidence:.3f} below minimum {cfg.min_confidence:.3f}"
    elif safety_factor < 1.0 / max(1.0, cfg.catastrophic_multiplier):
        reason = (
            f"Safety factor {safety_factor:.3f} below conservative bound "
            f"{1.0 / max(1.0, cfg.catastrophic_multiplier):.3f}"
        )
    elif should_stop:
        reason = "Tracker requested stop (entropy or tolerance triggered)"

    return CycleResult(
        roi_before=roi_before,
        roi_after=roi_after,
        raroi=raroi,
        confidence=confidence,
        safety_factor=safety_factor,
        suggestions=suggestions,
        should_stop=should_stop or reason is not None,
        reason=reason,
    )


def parse_args() -> tuple[argparse.Namespace, ROIConfig]:
    parser = argparse.ArgumentParser(description=__doc__)
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

    for cycle in range(1, cfg.max_cycles + 1):
        result = run_cycle(manager, tracker, cfg, args.bot_name)
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

    LOGGER.info("Single-agent ROI run complete.")


if __name__ == "__main__":
    main()
