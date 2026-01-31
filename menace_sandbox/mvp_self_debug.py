"""Layer-2 orchestrator for deterministic MVP self-debugging."""

from __future__ import annotations

import argparse
import dataclasses
import difflib
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

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


def _build_patch_rules(source: str) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    if "return a - b" in source:
        rules.append(
            {
                "type": "replace",
                "id": "fix-addition",
                "description": "Ensure add uses addition instead of subtraction.",
                "anchor": "return a - b",
                "replacement": "return a + b",
                "anchor_kind": "literal",
                "count": 1,
                "allow_zero_matches": False,
                "meta": {"source": "mvp_self_debug"},
            }
        )
    return rules


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
        rules = _build_patch_rules(source)
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
    args = parser.parse_args(list(argv) if argv is not None else None)

    target_path = Path(args.target)
    result = _loop(target_path, max_attempts=max(1, args.max_attempts))
    print(_format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
