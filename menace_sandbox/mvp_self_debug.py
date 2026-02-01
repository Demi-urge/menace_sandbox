"""Layer-2 orchestrator for deterministic MVP self-debugging."""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import yaml

from error_ontology import classify_error
import mvp_evaluator
from menace_sandbox import patch_generator
from menace_sandbox.stabilization.logging_wrapper import wrap_with_logging
from menace_sandbox.stabilization.patch_validator import validate_patch_text
from menace_sandbox.stabilization.roi import compute_roi_delta


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


def _classify_error(stderr: str, returncode: int) -> Mapping[str, Any] | None:
    if returncode == 0:
        return None
    error_text = stderr.strip() or f"process exited with code {returncode}"
    return classify_error(error_text)


def _build_error_report(
    *,
    stderr: str,
    returncode: int,
    classification: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "stderr": stderr,
        "returncode": returncode,
        "classification": dict(classification or {}),
    }


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
        ]
    }

    if status in rule_catalog:
        return rule_catalog[status]

    return _fallback_rule(classification=classification, source=source)


def _render_unified_diff(original: str, modified: str, filename: str) -> str:
    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(),
            modified.splitlines(),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
        )
    )
    if not diff_lines:
        return ""
    header = f"diff --git a/{filename} b/{filename}"
    return "\n".join([header, *diff_lines]) + "\n"


def _apply_patch(target_path: Path, modified_source: str) -> Path:
    target_path.write_text(modified_source, encoding="utf-8")
    return target_path


def _loop(
    target_path: Path,
    *,
    max_attempts: int,
    task_rules: Sequence[Mapping[str, Any]] | None = None,
) -> LoopResult:
    logged_run = wrap_with_logging(_run_target, {"log_event_prefix": "mvp.run."})
    logged_classify = wrap_with_logging(
        _classify_error, {"log_event_prefix": "mvp.classify."}
    )
    logged_generate = wrap_with_logging(
        patch_generator.generate_patch, {"log_event_prefix": "mvp.patch.generate."}
    )
    logged_validate = wrap_with_logging(
        validate_patch_text, {"log_event_prefix": "mvp.patch.validate."}
    )
    logged_evaluate_roi = wrap_with_logging(
        mvp_evaluator.evaluate_roi, {"log_event_prefix": "mvp.roi.evaluate."}
    )
    logged_compute_delta = wrap_with_logging(
        compute_roi_delta, {"log_event_prefix": "mvp.roi.delta."}
    )

    attempts = 0
    patch_attempts: list[PatchAttempt] = []
    current_path = target_path
    previous_roi: float | None = None
    roi_delta: Mapping[str, Any] | None = None

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        run_result = logged_run(current_path)
        current_roi = logged_evaluate_roi(run_result.stdout, run_result.stderr)
        if previous_roi is None:
            previous_roi = current_roi

        classification = logged_classify(run_result.stderr, run_result.returncode)
        if run_result.returncode == 0:
            roi_delta = logged_compute_delta(
                {"roi": previous_roi}, {"roi": current_roi}
            )
            return LoopResult(
                attempts=attempts,
                final_run=run_result,
                roi_delta=roi_delta,
                patch_attempts=tuple(patch_attempts),
            )

        source = current_path.read_text(encoding="utf-8")
        error_report = _build_error_report(
            stderr=run_result.stderr,
            returncode=run_result.returncode,
            classification=classification,
        )
        rules = _build_patch_rules(
            source,
            classification=classification,
            task_rules=task_rules,
        )
        patch_payload = logged_generate(source, error_report, rules, validate_syntax=True)
        modified_source = ""
        if isinstance(patch_payload, Mapping):
            data = patch_payload.get("data")
            if isinstance(data, Mapping):
                modified_source = str(data.get("modified_source") or "")

        if not modified_source:
            roi_delta = logged_compute_delta(
                {"roi": previous_roi}, {"roi": current_roi}
            )
            return LoopResult(
                attempts=attempts,
                final_run=run_result,
                roi_delta=roi_delta,
                patch_attempts=tuple(patch_attempts),
            )

        diff_text = _render_unified_diff(source, modified_source, current_path.name)
        validation = logged_validate(diff_text)
        patch_attempt = PatchAttempt(
            patch_payload=patch_payload,
            diff_text=diff_text,
            validation=validation,
            target_path=current_path,
        )
        patch_attempts.append(patch_attempt)

        if not validation.get("valid", False):
            roi_delta = logged_compute_delta(
                {"roi": previous_roi}, {"roi": current_roi}
            )
            return LoopResult(
                attempts=attempts,
                final_run=run_result,
                roi_delta=roi_delta,
                patch_attempts=tuple(patch_attempts),
            )

        if attempt == max_attempts:
            roi_delta = logged_compute_delta(
                {"roi": previous_roi}, {"roi": current_roi}
            )
            return LoopResult(
                attempts=attempts,
                final_run=run_result,
                roi_delta=roi_delta,
                patch_attempts=tuple(patch_attempts),
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
    result = _loop(
        target_path,
        max_attempts=max(1, args.max_attempts),
        task_rules=task_rules,
    )
    print(_format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
