"""Layer-2 orchestrator for deterministic MVP self-debugging."""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from yaml_fallback import get_yaml

from menace_sandbox.mvp_brain import run_mvp_pipeline
from menace_sandbox.stabilization.logging_wrapper import (
    StabilizationLoggingWrapper,
    wrap_with_logging,
)
from menace_sandbox.stabilization.patch_validator import validate_patch_text
from sandbox_results_logger import record_self_debug_metrics

yaml = get_yaml("mvp_self_debug", warn=False)


@dataclasses.dataclass(frozen=True)
class RunResult:
    stdout: str
    stderr: str
    returncode: int


@dataclasses.dataclass(frozen=True)
class PatchAttempt:
    patch_payload: Mapping[str, Any]
    diff_text: str
    validation: Mapping[str, Any]
    target_path: Path


@dataclasses.dataclass(frozen=True)
class LoopResult:
    attempts: int
    final_run: RunResult
    roi_delta: Mapping[str, Any] | None
    patch_attempts: tuple[PatchAttempt, ...]
    exit_reason: str
    classification: Mapping[str, Any] | None
    patch_validity: Mapping[str, Any] | None


def _run_target(path: Path) -> RunResult:
    completed = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return RunResult(
        stdout=completed.stdout or "",
        stderr=completed.stderr or "",
        returncode=completed.returncode,
    )


def _normalize_task_rules(candidate: Any) -> list[dict[str, Any]] | None:
    if isinstance(candidate, Mapping):
        if "rules" in candidate:
            candidate = candidate.get("rules")
        elif "patch_rules" in candidate:
            candidate = candidate.get("patch_rules")
        else:
            return None
    if isinstance(candidate, Sequence) and not isinstance(
        candidate, (str, bytes, bytearray)
    ):
        rules = [dict(rule) for rule in candidate if isinstance(rule, Mapping)]
        return rules or None
    return None


def _load_task_rules(task_input: Path | None) -> list[dict[str, Any]] | None:
    if task_input is None:
        return None
    raw = task_input.read_text(encoding="utf-8")
    parsed: Any | None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = yaml.safe_load(raw)
    return _normalize_task_rules(parsed)


def _classification_context(
    classification: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    status: str | None = None
    matched_rule_id: str | None = None
    if isinstance(classification, Mapping):
        raw_status = classification.get("status")
        if isinstance(raw_status, str):
            status = raw_status
        data = classification.get("data")
        if isinstance(data, Mapping):
            raw_rule_id = data.get("matched_rule_id")
            if isinstance(raw_rule_id, str):
                matched_rule_id = raw_rule_id
    return status, matched_rule_id


def _classification_metrics(
    classification: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    status, matched_rule_id = _classification_context(classification)
    if status is None and matched_rule_id is None:
        return None
    return {"status": status, "matched_rule_id": matched_rule_id}


def _fallback_rule(
    *,
    classification: Mapping[str, Any] | None,
    source: str,
) -> list[dict[str, Any]]:
    status, matched_rule_id = _classification_context(classification)
    return [
        {
            "type": "replace",
            "id": f"noop-{(status or 'unknown').lower()}",
            "description": "No-op rule to keep deterministic patch generation.",
            "anchor": "__MVP_SELF_DEBUG_NOOP__",
            "replacement": "__MVP_SELF_DEBUG_NOOP__",
            "anchor_kind": "literal",
            "count": 1,
            "allow_zero_matches": True,
            "meta": {
                "source": "mvp_self_debug",
                "error_category": status or "unknown",
                "error_rule_id": matched_rule_id or "unknown",
                "source_length": len(source),
            },
        }
    ]


def _build_patch_rules(
    source: str,
    *,
    classification: Mapping[str, Any] | None,
    task_rules: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if task_rules is not None:
        return [dict(rule) for rule in task_rules]

    status, matched_rule_id = _classification_context(classification)

    rule_catalog: dict[str, list[dict[str, Any]]] = {
        "ContractViolation": [
            {
                "type": "replace",
                "id": "fix-addition",
                "description": "Ensure add uses addition instead of subtraction.",
                "anchor": "return a - b",
                "replacement": "return a + b",
                "anchor_kind": "literal",
                "count": 1,
                "allow_zero_matches": False,
                "meta": {
                    "source": "mvp_self_debug",
                    "error_category": status or "unknown",
                    "error_rule_id": matched_rule_id or "unknown",
                },
            }
        ],
        "SyntaxError": [
            {
                "type": "replace",
                "id": "fix-missing-def-colon",
                "description": "Add missing colon to MVP function definition.",
                "anchor": "def add(a, b)",
                "replacement": "def add(a, b):",
                "anchor_kind": "literal",
                "count": 1,
                "allow_zero_matches": False,
                "meta": {
                    "source": "mvp_self_debug",
                    "error_category": status or "unknown",
                    "error_rule_id": matched_rule_id or "unknown",
                },
            }
        ],
        "MissingReturn": [
            {
                "type": "replace",
                "id": "fix-missing-return",
                "description": "Replace placeholder pass with explicit return.",
                "anchor": "pass",
                "replacement": "return None",
                "anchor_kind": "literal",
                "count": 1,
                "allow_zero_matches": False,
                "meta": {
                    "source": "mvp_self_debug",
                    "error_category": status or "unknown",
                    "error_rule_id": matched_rule_id or "unknown",
                },
            }
        ],
        "TypeError-Mismatch": [
            {
                "type": "replace",
                "id": "fix-type-mismatch-addition",
                "description": "Coerce addition operands to integers.",
                "anchor": "return a + b",
                "replacement": "return int(a) + int(b)",
                "anchor_kind": "literal",
                "count": 1,
                "allow_zero_matches": False,
                "meta": {
                    "source": "mvp_self_debug",
                    "error_category": status or "unknown",
                    "error_rule_id": matched_rule_id or "unknown",
                },
            }
        ],
    }

    if status in rule_catalog:
        return rule_catalog[status]

    return _fallback_rule(classification=classification, source=source)


def _apply_patch(target_path: Path, modified_source: str) -> Path:
    target_path.write_text(modified_source, encoding="utf-8")
    return target_path


def _validate_menace_patch_text(patch_text: str) -> dict[str, Any]:
    """Validate Menace patch text for Layer-2 self-debugging."""

    try:
        validation: Any = validate_patch_text(patch_text)
    except Exception as exc:  # pragma: no cover - safety fallback
        validation = {
            "valid": False,
            "flags": ["validation_exception"],
            "context": {"error": str(exc)},
        }
    if not isinstance(validation, Mapping):
        validation = {
            "valid": False,
            "flags": ["validation_invalid_payload"],
            "context": {"payload_type": type(validation).__name__},
        }
    validation.setdefault("valid", False)
    validation.setdefault("flags", [])
    validation.setdefault("context", {})
    if validation.get("valid"):
        validation["context"]["format"] = "menace_patch"
    return validation


def _roi_delta_total(pipeline_result: Mapping[str, Any]) -> float:
    roi_delta = pipeline_result.get("roi_delta")
    if isinstance(roi_delta, Mapping):
        data = roi_delta.get("data")
        if isinstance(data, Mapping):
            try:
                return float(data.get("total", 0.0))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _loop(
    target_path: Path,
    *,
    max_attempts: int,
    task_rules: Sequence[Mapping[str, Any]] | None = None,
    metrics_logger: StabilizationLoggingWrapper | None = None,
) -> LoopResult:
    logged_run = wrap_with_logging(_run_target, {"log_event_prefix": "mvp.run."})
    logged_pipeline = wrap_with_logging(
        run_mvp_pipeline, {"log_event_prefix": "mvp.pipeline."}
    )
    logged_validate = wrap_with_logging(
        _validate_menace_patch_text, {"log_event_prefix": "mvp.patch.validate."}
    )

    attempts = 0
    patch_attempts: list[PatchAttempt] = []
    current_path = target_path
    previous_roi: float | None = None
    roi_delta: Mapping[str, Any] | None = None
    last_classification: dict[str, Any] | None = None
    last_validation: Mapping[str, Any] | None = None
    exit_reason = "unknown"

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        run_result = logged_run(current_path)
        if run_result.returncode == 0:
            source = current_path.read_text(encoding="utf-8")
            rules = _build_patch_rules(
                source,
                classification=None,
                task_rules=task_rules,
            )
            payload: dict[str, Any] = {
                "source": source,
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
                "returncode": run_result.returncode,
                "rules": rules,
            }
            if previous_roi is not None:
                payload["prior_roi"] = previous_roi
            pipeline_result = logged_pipeline(payload)
            current_roi = pipeline_result.get("roi_score")
            if previous_roi is None and isinstance(current_roi, (int, float)):
                previous_roi = current_roi
            roi_delta = (
                pipeline_result.get("roi_delta")
                if isinstance(pipeline_result, Mapping)
                else None
            )
            exit_reason = "success"
            if metrics_logger:
                metrics_logger.log_metrics(
                    "mvp.self_debug.exit",
                    attempts=attempts,
                    classification=last_classification,
                    patch_validity=last_validation,
                    roi_delta=roi_delta,
                    roi_delta_total=_roi_delta_total(pipeline_result)
                    if isinstance(pipeline_result, Mapping)
                    else None,
                    exit_reason=exit_reason,
                )
            return LoopResult(
                attempts=attempts,
                final_run=run_result,
                roi_delta=roi_delta,
                patch_attempts=tuple(patch_attempts),
                exit_reason=exit_reason,
                classification=last_classification,
                patch_validity=last_validation,
            )

        source = current_path.read_text(encoding="utf-8")
        classification: Mapping[str, Any] | None = None
        if task_rules is None:
            preflight_payload: dict[str, Any] = {
                "source": source,
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
                "returncode": run_result.returncode,
                "rules": [],
            }
            if previous_roi is not None:
                preflight_payload["prior_roi"] = previous_roi
            preflight_result = logged_pipeline(preflight_payload)
            if isinstance(preflight_result, Mapping):
                classification = preflight_result.get("classification")
        last_classification = _classification_metrics(classification)

        rules = _build_patch_rules(
            source,
            classification=classification,
            task_rules=task_rules,
        )
        had_prior_roi = previous_roi is not None
        payload: dict[str, Any] = {
            "source": source,
            "stdout": run_result.stdout,
            "stderr": run_result.stderr,
            "returncode": run_result.returncode,
            "rules": rules,
        }
        if previous_roi is not None:
            payload["prior_roi"] = previous_roi
        pipeline_result = logged_pipeline(payload)
        current_roi = (
            pipeline_result.get("roi_score")
            if isinstance(pipeline_result, Mapping)
            else None
        )
        if previous_roi is None and isinstance(current_roi, (int, float)):
            previous_roi = current_roi
        patch_text = (
            pipeline_result.get("patch_text")
            if isinstance(pipeline_result, Mapping)
            else ""
        )
        modified_source = (
            pipeline_result.get("modified_source")
            if isinstance(pipeline_result, Mapping)
            else ""
        )
        patch_payload = (
            pipeline_result if isinstance(pipeline_result, Mapping) else {}
        )

        if not modified_source:
            roi_delta = (
                pipeline_result.get("roi_delta")
                if isinstance(pipeline_result, Mapping)
                else None
            )
            exit_reason = "no_patch_generated"
            if metrics_logger:
                metrics_logger.log_metrics(
                    "mvp.self_debug.exit",
                    attempts=attempts,
                    classification=last_classification,
                    patch_validity=last_validation,
                    roi_delta=roi_delta,
                    roi_delta_total=_roi_delta_total(patch_payload),
                    exit_reason=exit_reason,
                )
            return LoopResult(
                attempts=attempts,
                final_run=run_result,
                roi_delta=roi_delta,
                patch_attempts=tuple(patch_attempts),
                exit_reason=exit_reason,
                classification=last_classification,
                patch_validity=last_validation,
            )

        diff_text = str(patch_text or "")
        validation = logged_validate(diff_text)
        last_validation = validation if isinstance(validation, Mapping) else None
        patch_attempt = PatchAttempt(
            patch_payload=patch_payload,
            diff_text=diff_text,
            validation=validation,
            target_path=current_path,
        )
        patch_attempts.append(patch_attempt)
        if metrics_logger:
            metrics_logger.log_metrics(
                "mvp.self_debug.attempt",
                attempt=attempts,
                classification=last_classification,
                patch_validity=last_validation,
                roi_delta=roi_delta,
                roi_delta_total=_roi_delta_total(patch_payload),
            )

        if not validation.get("valid", False):
            roi_delta = (
                pipeline_result.get("roi_delta")
                if isinstance(pipeline_result, Mapping)
                else None
            )
            exit_reason = "invalid_patch"
            if metrics_logger:
                metrics_logger.log_metrics(
                    "mvp.self_debug.exit",
                    attempts=attempts,
                    classification=last_classification,
                    patch_validity=last_validation,
                    roi_delta=roi_delta,
                    roi_delta_total=_roi_delta_total(patch_payload),
                    exit_reason=exit_reason,
                )
            return LoopResult(
                attempts=attempts,
                final_run=run_result,
                roi_delta=roi_delta,
                patch_attempts=tuple(patch_attempts),
                exit_reason=exit_reason,
                classification=last_classification,
                patch_validity=last_validation,
            )

        if had_prior_roi and _roi_delta_total(patch_payload) <= 0:
            roi_delta = (
                pipeline_result.get("roi_delta")
                if isinstance(pipeline_result, Mapping)
                else None
            )
            exit_reason = "non_positive_roi_delta"
            if metrics_logger:
                metrics_logger.log_metrics(
                    "mvp.self_debug.exit",
                    attempts=attempts,
                    classification=last_classification,
                    patch_validity=last_validation,
                    roi_delta=roi_delta,
                    roi_delta_total=_roi_delta_total(patch_payload),
                    exit_reason=exit_reason,
                )
            return LoopResult(
                attempts=attempts,
                final_run=run_result,
                roi_delta=roi_delta,
                patch_attempts=tuple(patch_attempts),
                exit_reason=exit_reason,
                classification=last_classification,
                patch_validity=last_validation,
            )

        if attempt == max_attempts:
            roi_delta = (
                pipeline_result.get("roi_delta")
                if isinstance(pipeline_result, Mapping)
                else None
            )
            exit_reason = "max_attempts_reached"
            if metrics_logger:
                metrics_logger.log_metrics(
                    "mvp.self_debug.exit",
                    attempts=attempts,
                    classification=last_classification,
                    patch_validity=last_validation,
                    roi_delta=roi_delta,
                    roi_delta_total=_roi_delta_total(patch_payload),
                    exit_reason=exit_reason,
                )
            return LoopResult(
                attempts=attempts,
                final_run=run_result,
                roi_delta=roi_delta,
                patch_attempts=tuple(patch_attempts),
                exit_reason=exit_reason,
                classification=last_classification,
                patch_validity=last_validation,
            )

        temp_dir = Path(tempfile.mkdtemp(prefix="mvp_self_debug_"))
        temp_path = temp_dir / current_path.name
        temp_path.write_text(source, encoding="utf-8")
        current_path = _apply_patch(temp_path, modified_source)

    return LoopResult(
        attempts=attempts,
        final_run=logged_run(current_path),
        roi_delta=roi_delta,
        patch_attempts=tuple(patch_attempts),
        exit_reason=exit_reason,
        classification=last_classification,
        patch_validity=last_validation,
    )


def _format_summary(result: LoopResult) -> str:
    lines: list[str] = [
        f"Attempts: {result.attempts}",
        f"Return code: {result.final_run.returncode}",
    ]
    if result.final_run.stdout:
        lines.append(f"Stdout: {result.final_run.stdout.strip()}")
    if result.final_run.stderr:
        lines.append(f"Stderr: {result.final_run.stderr.strip()}")
    if result.roi_delta:
        lines.append("ROI delta: computed")
    if result.patch_attempts:
        lines.append(f"Patch attempts: {len(result.patch_attempts)}")
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        default="toy.py",
        help="Target script path (default: toy.py)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum debug iterations before stopping",
    )
    parser.add_argument(
        "--task-input",
        type=Path,
        default=None,
        help="Optional JSON/YAML task input containing patch rules.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    target_path = Path(args.target)
    task_rules = _load_task_rules(args.task_input)
    correlation_id = f"mvp-self-debug-{uuid.uuid4()}"
    metrics_logger = StabilizationLoggingWrapper.start(
        correlation_id=correlation_id, source="mvp_self_debug"
    )
    try:
        result = _loop(
            target_path,
            max_attempts=max(1, args.max_attempts),
            task_rules=task_rules,
            metrics_logger=metrics_logger,
        )
    finally:
        metrics_logger.close()
    record_self_debug_metrics(
        {
            "ts": datetime.utcnow().isoformat(),
            "correlation_id": correlation_id,
            "source": "mvp_self_debug",
            "attempts": result.attempts,
            "classification": result.classification,
            "patch_validity": result.patch_validity,
            "roi_delta_total": _roi_delta_total(result.roi_delta or {}),
            "roi_delta": result.roi_delta,
            "exit_reason": result.exit_reason,
        }
    )
    print(_format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
