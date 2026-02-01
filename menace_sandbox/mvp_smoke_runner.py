"""Run MVP brain smoke workflows to validate sandbox self-healing."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

import mvp_evaluator
from logging_utils import get_logger, log_record, set_correlation_id
from menace_sandbox import patch_generator
from menace_sandbox.mvp_brain import run_mvp_pipeline
from menace_sandbox.mvp_self_debug import _apply_patch
from menace_sandbox.sandbox_rule_builder import build_rules
from menace_sandbox.stabilization.logging_wrapper import wrap_with_logging


@dataclass
class SmokeRunResult:
    success: bool
    duration: float
    stderr: str = ""
    failure: str | None = None


def _load_scoring_helpers() -> tuple[Any, Any]:
    scoring_path = Path(__file__).resolve().parents[1] / "sandbox_runner" / "scoring.py"
    scoring_spec = importlib.util.spec_from_file_location(
        "sandbox_runner_scoring", scoring_path
    )
    if scoring_spec is None or scoring_spec.loader is None:
        raise RuntimeError("Unable to load sandbox_runner.scoring for smoke workflow")
    scoring_module = importlib.util.module_from_spec(scoring_spec)
    scoring_spec.loader.exec_module(scoring_module)
    return scoring_module.record_run, scoring_module.load_summary


def _parse_step(step: str) -> tuple[str, str | None]:
    if ":" in step:
        mod, func = step.split(":", 1)
        return mod, func
    if importlib.util.find_spec(step) is not None:
        return step, None
    if "." in step:
        mod, func = step.rsplit(".", 1)
        return mod, func
    if "/" in step:
        return step.replace("/", "."), None
    raise ValueError(f"Workflow step '{step}' must include a module path")


def _validate_patch(patch_text: str) -> dict[str, object]:
    try:
        patch_generator.validate_patch_text(patch_text)
    except Exception as exc:
        return {
            "valid": False,
            "flags": ["validation_exception"],
            "context": {"error": str(exc)},
        }
    lines = patch_text.splitlines()
    change_count = 0
    if len(lines) > 1 and lines[1].startswith("change-count:"):
        try:
            change_count = int(lines[1].split(":", 1)[1].strip())
        except (TypeError, ValueError):
            change_count = 0
    if change_count <= 0:
        return {
            "valid": False,
            "flags": ["no_changes"],
            "context": {"change_count": change_count},
        }
    return {
        "valid": True,
        "flags": [],
        "context": {"format": "menace_patch", "change_count": change_count},
    }


def _compute_workflow_entropy(spec: list[dict[str, str]]) -> float:
    if not spec:
        return 0.0
    counts: dict[str, int] = {}
    total = 0
    for step in spec:
        mod = step.get("module")
        if not mod:
            continue
        counts[mod] = counts.get(mod, 0) + 1
        total += 1
    if not total:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def run_mvp_workflow_smoke(
    steps: Sequence[str],
    *,
    max_attempts: int = 2,
    log_event_prefix: str = "sandbox.mvp.self_heal",
    enforce_checks: bool = True,
) -> dict[str, Any]:
    record_run, load_summary = _load_scoring_helpers()
    logger = get_logger("sandbox.mvp.self_heal")
    correlation_id = f"mvp-workflow-smoke-{uuid.uuid4()}"
    set_correlation_id(correlation_id)

    parsed_steps = [s.strip() for s in steps if s.strip()]
    if not parsed_steps:
        raise ValueError("At least one workflow step is required")
    module_list = [_parse_step(step)[0] for step in parsed_steps]
    workflow_entropy = _compute_workflow_entropy(
        [{"module": mod} for mod in module_list]
    )
    last_entropy = 0.0
    summary_before = load_summary()
    pipeline_invocations = 0
    patch_proposed = 0
    patch_applied = 0
    patch_rejected = 0
    validation_blocked = 0
    stabilized_steps: list[str] = []
    failure_steps: list[str] = []
    roi_samples: list[float] = []
    logged_pipeline = wrap_with_logging(
        run_mvp_pipeline,
        {"log_event_prefix": f"{log_event_prefix}.pipeline."},
    )
    logged_validate = wrap_with_logging(
        _validate_patch,
        {"log_event_prefix": f"{log_event_prefix}.patch.validate."},
    )

    for step in parsed_steps:
        mod, func = _parse_step(step)
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.origin:
            raise RuntimeError(f"Module '{mod}' for workflow step '{step}' not found")
        source_path = Path(spec.origin)
        original_source = source_path.read_text(encoding="utf-8")
        prior_roi: float | None = None
        try:
            patch_applied_for_step = False
            for attempt in range(1, max_attempts + 1):
                attempt_start = time.monotonic()
                cmd = [
                    sys.executable,
                    "-c",
                    f"from {mod} import {func or 'main'} as _m; _m()",
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                attempt_runtime = time.monotonic() - attempt_start
                source_text = source_path.read_text(encoding="utf-8")
                rules = build_rules(result.stderr or result.stdout, source=source_text)
                had_prior_roi = prior_roi is not None
                payload = {
                    "source": source_text,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "rules": rules,
                }
                if prior_roi is not None:
                    payload["prior_roi"] = prior_roi
                pipeline_result = logged_pipeline(payload)
                pipeline_invocations += 1
                roi_score = pipeline_result.get("roi_score")
                if prior_roi is None and isinstance(roi_score, (int, float)):
                    prior_roi = float(roi_score)
                roi_value = float(roi_score) if isinstance(roi_score, (int, float)) else float(
                    mvp_evaluator.evaluate_roi(result.stdout, result.stderr)
                )
                roi_samples.append(roi_value)
                entropy_delta = workflow_entropy - last_entropy
                last_entropy = workflow_entropy
                run_result = SmokeRunResult(
                    success=result.returncode == 0,
                    duration=attempt_runtime,
                    stderr=result.stderr,
                    failure=(result.stderr or result.stdout) if result.returncode else None,
                )
                record_run(
                    run_result,
                    metrics={
                        "runtime": attempt_runtime,
                        "roi": roi_value,
                        "entropy_delta": entropy_delta,
                        "coverage": {"executed_functions": [f"{mod}:{func or 'main'}"]},
                    },
                )
                if result.returncode == 0:
                    print(f"{step}: ok after {attempt} attempt(s)")
                    logger.info(
                        "sandbox workflow stabilized",
                        extra=log_record(
                            workflow_step=step,
                            attempt=attempt,
                            correlation_id=correlation_id,
                            roi=roi_value,
                            entropy=workflow_entropy,
                        ),
                    )
                    if patch_applied_for_step:
                        stabilized_steps.append(step)
                    break
                failure_steps.append(step)
                patch_text = str(pipeline_result.get("patch_text") or "")
                modified_source = str(pipeline_result.get("modified_source") or "")
                validation = logged_validate(patch_text)
                if patch_text:
                    logger.info(
                        "mvp brain proposed patch",
                        extra=log_record(
                            workflow_step=step,
                            attempt=attempt,
                            patch_text_length=len(patch_text),
                            roi=roi_value,
                            entropy=workflow_entropy,
                        ),
                    )
                    patch_proposed += 1
                if not validation.get("valid"):
                    print(f"{step}: patch rejected (flags={validation.get('flags')})")
                    logger.info(
                        "sandbox patch rejected",
                        extra=log_record(
                            workflow_step=step,
                            attempt=attempt,
                            flags=list(validation.get("flags", [])),
                            roi=roi_value,
                            entropy=workflow_entropy,
                        ),
                    )
                    patch_rejected += 1
                    validation_blocked += 1
                    break
                roi_delta = pipeline_result.get("roi_delta", {})
                delta_total = 0.0
                if isinstance(roi_delta, Mapping):
                    data = roi_delta.get("data")
                    if isinstance(data, Mapping):
                        try:
                            delta_total = float(data.get("total", 0.0))
                        except (TypeError, ValueError):
                            delta_total = 0.0
                if (had_prior_roi and delta_total <= 0) or not modified_source:
                    print(f"{step}: patch rejected (roi_delta={delta_total})")
                    logger.info(
                        "sandbox patch rejected",
                        extra=log_record(
                            workflow_step=step,
                            attempt=attempt,
                            roi_delta=delta_total,
                            roi=roi_value,
                            entropy=workflow_entropy,
                        ),
                    )
                    patch_rejected += 1
                    break
                _apply_patch(source_path, modified_source)
                print(f"{step}: patch applied (attempt {attempt})")
                logger.info(
                    "sandbox patch applied",
                    extra=log_record(
                        workflow_step=step,
                        attempt=attempt,
                        roi=roi_value,
                        entropy=workflow_entropy,
                    ),
                )
                patch_applied += 1
                patch_applied_for_step = True
        finally:
            source_path.write_text(original_source, encoding="utf-8")

    summary_after = load_summary()
    run_delta = (summary_after.get("runs") or 0) - (summary_before.get("runs") or 0)
    roi_total_delta = (summary_after.get("roi_total") or 0.0) - (
        summary_before.get("roi_total") or 0.0
    )
    roi_count_delta = (summary_after.get("roi_count") or 0) - (
        summary_before.get("roi_count") or 0
    )
    entropy_delta_total = (summary_after.get("entropy_total") or 0.0) - (
        summary_before.get("entropy_total") or 0.0
    )
    checks = {
        "pipeline_invocations": pipeline_invocations,
        "patches_proposed": patch_proposed,
        "patches_applied": patch_applied,
        "patches_rejected": patch_rejected,
        "validation_blocked": validation_blocked,
        "stabilized_steps": stabilized_steps,
        "failure_steps": failure_steps,
        "run_delta": run_delta,
        "roi_total_delta": roi_total_delta,
        "roi_count_delta": roi_count_delta,
        "entropy_total_delta": entropy_delta_total,
    }
    print("[SANDBOX] MVP self-heal checks:")
    for key, value in checks.items():
        print(f"  - {key}: {value}")
    logger.info(
        "sandbox mvp self-heal checks",
        extra=log_record(
            correlation_id=correlation_id,
            **checks,
        ),
    )

    if enforce_checks:
        if not pipeline_invocations:
            raise RuntimeError("MVP brain was not invoked during smoke workflow")
        if not patch_proposed:
            raise RuntimeError("MVP brain did not propose any patches")
        if not validation_blocked:
            raise RuntimeError("Patch validation did not block any invalid patches")
        if not stabilized_steps:
            raise RuntimeError("No workflow steps stabilized after patching")
        if not failure_steps:
            raise RuntimeError("Workflow steps did not trigger deterministic failures")
        if run_delta <= 0 or entropy_delta_total == 0.0:
            raise RuntimeError("Workflow metrics failed to update in run summary")
        if roi_count_delta <= 0:
            raise RuntimeError("ROI metrics failed to update in run summary")

    return checks


__all__ = ["run_mvp_workflow_smoke"]
