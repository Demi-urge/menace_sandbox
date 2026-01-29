"""Minimal MVP workflow execution helpers.

This module implements an end-to-end pipeline for MVP tasks:
generation (build deterministic Python code) → execution (run the code in a
controlled subprocess) → evaluation (score execution output for ROI).

Constraints:
    - Standard-library-only implementation.
    - No recursion.
    - No concurrency.
    - No networking.
    - No dynamic imports.
"""

from __future__ import annotations

import datetime
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

import mvp_codegen
import mvp_evaluator
import mvp_executor

__all__ = ["execute_task"]

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class _TaskSpec:
    """Normalized task specification for execution."""

    objective: str
    constraints: list[str]


def execute_task(task_dict: dict) -> dict:
    """Execute a task workflow end-to-end and return a JSON-serializable payload.

    Args:
        task_dict: Input task payload that must contain an objective and optional constraints.

    Returns:
        A JSON-serializable dictionary with objective, constraints, generated code,
        execution output, error details, ROI score, timestamps, duration, and success flag.
    """
    started_at = _now_iso8601()
    start_time = time.time()
    try:
        generated_code = ""
        execution_output = ""
        execution_error = ""
        evaluation_error = ""
        roi_score = 0.0
        success = False
        spec: _TaskSpec | None = None
        exec_stdout = ""
        exec_stderr = ""

        try:
            if not isinstance(task_dict, dict):
                execution_error = "task_dict must be a dict"
            else:
                objective_value = task_dict.get("objective")
                if not isinstance(objective_value, str) or not objective_value.strip():
                    execution_error = "objective must be a non-empty string"
                else:
                    objective = objective_value.strip()
                    constraints_raw = task_dict.get("constraints")
                    constraints_errors: list[str] = []
                    constraints = _normalize_constraints(constraints_raw, constraints_errors)
                    if constraints_errors:
                        execution_error = constraints_errors[0]
                    else:
                        spec = _TaskSpec(objective=objective, constraints=constraints)
        except Exception as exc:  # pragma: no cover - defensive
            execution_error = _sanitize_exception(exc)

        if spec is not None:
            try:
                generation_task = {
                    "objective": spec.objective,
                    "constraints": spec.constraints,
                }
                generated_code = mvp_codegen.run_generation(generation_task)
                if not isinstance(generated_code, str):
                    generated_code = str(generated_code)
                if not generated_code.strip():
                    execution_error = "code generation failed"
            except Exception as exc:  # pragma: no cover - defensive
                execution_error = _sanitize_exception(exc)

        if spec is not None and generated_code and not execution_error:
            try:
                exec_stdout, exec_stderr = mvp_executor.execute_untrusted(generated_code)
                execution_output = f"STDOUT:\n{exec_stdout}\n\nSTDERR:\n{exec_stderr}"
                if isinstance(exec_stderr, str) and exec_stderr.strip().startswith("error:"):
                    execution_error = exec_stderr.strip()
            except Exception as exc:  # pragma: no cover - defensive
                execution_error = _sanitize_exception(exc)
                execution_output = ""
        elif spec is not None and not generated_code and not execution_error:
            execution_error = "execution skipped: no generated code"

        if spec is not None:
            try:
                if execution_error and not exec_stderr:
                    exec_stderr = execution_error
                roi_score = float(mvp_evaluator.evaluate_roi(exec_stdout, exec_stderr))
                if roi_score == -1.0:
                    evaluation_error = "evaluation failed"
            except Exception as exc:  # pragma: no cover - defensive
                evaluation_error = _sanitize_exception(exc)
                roi_score = 0.0

        success = bool(spec and generated_code and not execution_error and not evaluation_error)
        finished_at = _now_iso8601()
        end_time = time.time()
        duration_ms = _elapsed_ms(start_time, end_time)
        sanitized_output = _sanitize_error_output(execution_output)

        sanitized_objective = _sanitize_error_output(spec.objective) if spec else ""
        sanitized_constraints = (
            [_sanitize_error_output(constraint) for constraint in spec.constraints] if spec else []
        )

        return {
            "objective": sanitized_objective,
            "constraints": sanitized_constraints,
            "generated_code": _sanitize_error_output(generated_code),
            "execution_output": sanitized_output,
            "execution_error": _sanitize_error_output(execution_error),
            "evaluation_error": _sanitize_error_output(evaluation_error),
            "roi_score": float(roi_score),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "success": success,
        }
    except Exception as exc:  # pragma: no cover - defensive
        finished_at = _now_iso8601()
        end_time = time.time()
        return {
            "success": False,
            "error": _sanitize_exception(exc),
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": _elapsed_ms(start_time, end_time),
        }


def _now_iso8601() -> str:
    """Return an ISO-8601 timestamp with timezone info when available."""
    return datetime.datetime.now().astimezone().isoformat()


def _elapsed_ms(start: float, end: float) -> int:
    """Return elapsed time in integer milliseconds."""
    return int((end - start) * 1000)


def _normalize_constraints(constraints: object, errors: list[str]) -> list[str]:
    """Normalize constraints into a list of strings."""
    if constraints is None:
        return []
    if isinstance(constraints, dict):
        return [f"{key}: {value}" for key, value in constraints.items()]
    if isinstance(constraints, str):
        cleaned = constraints.strip()
        return [cleaned] if cleaned else []
    if _is_sequence_of_strings(constraints):
        return [item.strip() for item in constraints if item.strip()]

    errors.append("constraints must be a dict, list of strings, or string when provided")
    return []


def _is_sequence_of_strings(value: object) -> bool:
    """Return True when the value is a non-string sequence of strings."""
    if isinstance(value, (str, bytes)):
        return False
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(item, str) for item in value)


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape sequences from text."""
    return _ANSI_ESCAPE_RE.sub("", text)


def _strip_tracebacks(text: str) -> str:
    """Remove traceback-looking lines from output."""
    cleaned_lines: list[str] = []
    in_traceback = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Traceback (most recent call last):"):
            in_traceback = True
            continue
        if in_traceback:
            if not line.startswith(" ") and not line.startswith("\t"):
                in_traceback = False
            else:
                continue
        if stripped.startswith('File "'):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _sanitize_error_output(text: str) -> str:
    """Remove ANSI codes and stack traces from error output."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = _strip_ansi(text)
    cleaned = _strip_tracebacks(cleaned)
    return cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def _sanitize_exception(exc: BaseException | str) -> str:
    """Return a sanitized exception message without ANSI or tracebacks."""
    message = str(exc) if exc else "unexpected error"
    return _sanitize_error_output(message) or "unexpected error"
