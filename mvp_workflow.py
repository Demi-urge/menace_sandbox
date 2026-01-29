"""Minimal MVP workflow execution helpers."""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from typing import Any

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def execute_task(task_dict: dict[str, object]) -> dict[str, object]:
    """Execute a task workflow end-to-end and return a JSON-serializable payload.

    Args:
        task_dict: Input task payload that must contain an objective and optional constraints.

    Returns:
        A JSON-serializable dictionary with generated code, execution output, errors,
        ROI score, timestamps, and success flag.
    """
    started_at = datetime.datetime.now(datetime.timezone.utc)
    errors: list[str] = []
    generated_code = ""
    execution_output = ""
    roi_score = 0.0
    success = False

    if not isinstance(task_dict, dict):
        errors.append("task_dict must be a dict")
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        return _build_response(
            generated_code,
            execution_output,
            errors,
            roi_score,
            started_at,
            finished_at,
            success,
        )

    objective_value = task_dict.get("objective")
    if not isinstance(objective_value, str) or not objective_value.strip():
        errors.append("objective must be a non-empty string")
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        return _build_response(
            generated_code,
            execution_output,
            errors,
            roi_score,
            started_at,
            finished_at,
            success,
        )

    objective = objective_value.strip()
    constraints_raw = task_dict.get("constraints")
    constraints = _normalize_constraints(constraints_raw, errors)
    if errors:
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        return _build_response(
            generated_code,
            execution_output,
            errors,
            roi_score,
            started_at,
            finished_at,
            success,
        )

    try:
        generation_result = _generate_code(objective, constraints)
        generated_code = str(generation_result.get("generated_code", ""))
        errors.extend(_coerce_error_list(generation_result.get("errors")))
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"generation error: {exc}")

    try:
        execution_result = _execute_code(generated_code, timeout_s=5.0)
        execution_output = str(execution_result.get("execution_output", ""))
        errors.extend(_coerce_error_list(execution_result.get("errors")))
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"execution error: {exc}")
        execution_result = {"execution_output": "", "errors": [f"execution error: {exc}"]}

    try:
        evaluation_result = _evaluate_result(generated_code, execution_result)
        roi_score = float(evaluation_result.get("roi_score", 0.0))
        errors.extend(_coerce_error_list(evaluation_result.get("errors")))
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"evaluation error: {exc}")
        roi_score = 0.0

    success = not errors and bool(generated_code)
    finished_at = datetime.datetime.now(datetime.timezone.utc)
    return _build_response(
        generated_code,
        execution_output,
        errors,
        roi_score,
        started_at,
        finished_at,
        success,
    )


def _generate_code(objective: str, constraints: list[str]) -> dict[str, object]:
    """Generate deterministic Python code for the given task.

    Args:
        objective: The task objective.
        constraints: Normalized constraint strings.

    Returns:
        A dictionary containing generated code and error messages.
    """
    errors: list[str] = []
    objective_text = objective.strip()
    constraints_payload = constraints

    payload = {
        "objective": objective_text,
        "constraints": constraints_payload,
    }
    payload_json = json.dumps(payload, sort_keys=True)

    code_lines = [
        '"""Auto-generated script."""',
        "import json",
        "",
        "def main() -> None:",
        f"    payload = {payload_json!r}",
        "    print(payload)",
        "",
        "if __name__ == '__main__':",
        "    main()",
        "",
    ]
    return {"generated_code": "\n".join(code_lines), "errors": errors}


def _execute_code(code: str, timeout_s: float) -> dict[str, object]:
    """Execute code in a temporary file and capture sanitized output.

    Args:
        code: Python source code to execute.
        timeout_s: Timeout in seconds.

    Returns:
        A dictionary with execution output and errors.
    """
    if not code.strip():
        return {"execution_output": "", "errors": ["code is empty"]}

    errors: list[str] = []
    output = ""
    tmp_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as tmp_file:
            tmp_file.write(code)
            tmp_path = tmp_file.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
        stdout = _strip_ansi(result.stdout or "")
        stderr = _strip_ansi(result.stderr or "")
        output = stdout

        if result.returncode != 0:
            errors.append(_sanitize_error(stderr) or "execution failed")
    except subprocess.TimeoutExpired:
        errors.append("timeout")
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"execution error: {exc}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return {"execution_output": output, "errors": errors}


def _evaluate_result(code: str, exec_result: dict[str, object]) -> dict[str, object]:
    """Evaluate execution results and provide an ROI score.

    Args:
        code: Generated code string.
        exec_result: Execution result dictionary.

    Returns:
        A dictionary with ROI score and errors.
    """
    errors: list[str] = []
    execution_errors = _coerce_error_list(exec_result.get("errors"))
    execution_output = str(exec_result.get("execution_output", ""))

    if not code.strip():
        errors.append("no code generated")

    base_score = 1.0 if not execution_errors and code.strip() else 0.0
    output_bonus = min(len(execution_output) / 200.0, 1.0) * 0.2
    roi_score = max(0.0, min(1.0, base_score + output_bonus))

    return {"roi_score": float(roi_score), "errors": errors}


def _normalize_constraints(constraints: object, errors: list[str]) -> list[str]:
    """Normalize constraints into a list of strings.

    Args:
        constraints: Raw constraints payload.
        errors: Shared error list to append to.

    Returns:
        A list of constraint strings.
    """
    if constraints is None:
        return []
    if isinstance(constraints, dict):
        return [f"{key}: {value}" for key, value in constraints.items()]
    if _is_sequence_of_strings(constraints):
        return [item.strip() for item in constraints if item.strip()]

    errors.append("constraints must be a dict or sequence of strings when provided")
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


def _sanitize_error(stderr: str) -> str:
    """Reduce stderr output to a single sanitized line."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "execution error"
    return lines[-1]


def _coerce_error_list(raw_errors: object) -> list[str]:
    """Ensure errors are represented as a list of strings."""
    if isinstance(raw_errors, list):
        return [str(item) for item in raw_errors if str(item)]
    if isinstance(raw_errors, str) and raw_errors:
        return [raw_errors]
    return []


def _build_response(
    generated_code: str,
    execution_output: str,
    errors: list[str],
    roi_score: float,
    started_at: datetime.datetime,
    finished_at: datetime.datetime,
    success: bool,
) -> dict[str, object]:
    """Build the JSON-serializable response payload."""
    return {
        "generated_code": generated_code,
        "execution_output": execution_output,
        "errors": errors,
        "roi_score": float(roi_score),
        "timestamps": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
        },
        "success": bool(success),
    }
