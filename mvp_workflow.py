"""Minimal MVP workflow execution helpers."""

from __future__ import annotations

import ast
import datetime
import os
import re
import json
import subprocess
import sys
import tempfile
from typing import Any, Optional

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def execute_task(task_dict: dict) -> dict:
    """Execute a task workflow end-to-end and return a JSON-serializable payload.

    Args:
        task_dict: Input task payload that must contain an objective and optional constraints.

    Returns:
        A JSON-serializable dictionary describing the workflow output.
    """
    started_at = datetime.datetime.now(datetime.timezone.utc)
    errors: list[str] = []
    generated_code = ""
    execution_output = ""
    roi_score = 0.0
    success = False

    normalized_task: dict[str, Any] = {}
    try:
        normalized_task, validation_errors = _normalize_task(task_dict)
        errors.extend(validation_errors)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"task normalization error: {exc}")

    try:
        generated_code = _generate_code(
            normalized_task.get("objective", ""),
            normalized_task.get("constraints"),
        )
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"generation error: {exc}")

    try:
        is_valid, validation_errors = _validate_code(generated_code)
        if not is_valid:
            errors.extend(validation_errors)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"validation error: {exc}")

    try:
        code_to_execute = generated_code if not errors else ""
        execution_output, execution_errors = _execute_code(code_to_execute, timeout_s=5.0)
        if execution_errors:
            errors.extend(execution_errors)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"execution error: {exc}")

    try:
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)
        roi_score = _evaluate_result(generated_code, execution_output, errors, duration_ms)
        success = bool(roi_score > 0 and not errors)
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"evaluation error: {exc}")
        success = False
        roi_score = 0.0
        finished_at = datetime.datetime.now(datetime.timezone.utc)
        duration_ms = int((finished_at - started_at).total_seconds() * 1000)
    else:
        if "finished_at" not in locals():
            finished_at = datetime.datetime.now(datetime.timezone.utc)
            duration_ms = int((finished_at - started_at).total_seconds() * 1000)

    return {
        "generated_code": generated_code,
        "execution_output": execution_output,
        "errors": errors,
        "roi_score": roi_score,
        "timestamps": {
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_ms": duration_ms,
        },
        "success": success,
    }


def _normalize_task(task_dict: dict) -> tuple[dict[str, Any], list[str]]:
    """Normalize task input and collect validation errors.

    Args:
        task_dict: Raw task input.

    Returns:
        A tuple of normalized task payload and a list of validation errors.
    """
    errors: list[str] = []
    normalized: dict[str, Any] = {"objective": "", "constraints": {}}

    if not isinstance(task_dict, dict):
        errors.append("task_dict must be a dict")
        return normalized, errors

    objective = task_dict.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        errors.append("objective must be a non-empty string")
    else:
        normalized["objective"] = objective.strip()

    constraints = task_dict.get("constraints")
    if constraints is None:
        normalized["constraints"] = {}
    elif isinstance(constraints, dict):
        normalized["constraints"] = constraints
    elif isinstance(constraints, list) and all(isinstance(item, str) for item in constraints):
        normalized["constraints"] = {"items": [item.strip() for item in constraints if item.strip()]}
    else:
        errors.append("constraints must be a dict or list of strings when provided")
        normalized["constraints"] = {}

    return normalized, errors


def _generate_code(objective: str, constraints: Optional[Any]) -> str:
    """Generate deterministic Python code for the given task."""
    objective_text = str(objective or "").strip()
    if not objective_text:
        return ""

    constraints_repr = "none"
    if constraints is not None:
        constraints_repr = repr(constraints)

    payload = {
        "objective": objective_text,
        "constraints": constraints_repr,
        "status": "generated",
    }
    payload_json = json.dumps(payload, sort_keys=True)

    code_lines = [
        '"""Auto-generated script.',
        f"Objective: {objective_text}",
        f"Constraints: {constraints_repr}",
        '"""',
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
    return "\n".join(code_lines)


def _validate_code(code: str) -> tuple[bool, list[str]]:
    """Validate generated code for syntax, imports, and forbidden calls."""
    errors: list[str] = []
    if not code or not code.strip():
        return False, ["code is empty"]

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"syntax error: {exc.msg}"]
    except Exception as exc:  # pragma: no cover - defensive
        return False, [f"validation error: {exc}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if not _is_stdlib_module(module):
                    errors.append(f"forbidden import: {module}")
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                errors.append("relative import not allowed")
            else:
                module = node.module.split(".")[0]
                if not _is_stdlib_module(module):
                    errors.append(f"forbidden import: {module}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in {"exec", "eval", "__import__"}:
                errors.append(f"forbidden call: {node.func.id}")
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "importlib":
                    errors.append("dynamic import detected")

    if not errors and _find_dynamic_import_calls(tree):
        errors.append("dynamic import detected")

    return not errors, errors


def _is_stdlib_module(module_name: str) -> bool:
    """Determine if a module belongs to the standard library."""
    stdlib_modules = getattr(sys, "stdlib_module_names", None)
    if stdlib_modules is None:
        stdlib_modules = set(sys.builtin_module_names)
    return module_name in stdlib_modules


def _find_dynamic_import_calls(tree: ast.AST) -> bool:
    """Detect dynamic import usage in an AST."""
    forbidden_names = {"__import__", "importlib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                return True
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in forbidden_names:
                    return True
    return False


def _execute_code(code: str, timeout_s: float = 2.0) -> tuple[str, list[str]]:
    """Execute code in a temporary file and capture sanitized output."""
    if not code or not code.strip():
        return "", ["code is empty"]

    tmp_path: str | None = None
    errors: list[str] = []
    stdout = ""

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
        stdout = _ANSI_ESCAPE_RE.sub("", result.stdout or "")
        stderr = _ANSI_ESCAPE_RE.sub("", result.stderr or "")

        if result.returncode != 0:
            errors.append(_redact_error(stderr) or "execution failed")
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

    return stdout, errors


def _redact_error(stderr: str) -> str:
    """Reduce stderr output to a single sanitized line."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "execution error"
    return lines[-1]


def _evaluate_result(code: str, exec_output: str, errors: list[str], duration_ms: int) -> float:
    """Compute a deterministic ROI score between 0 and 1."""
    success_score = 0.5 if not errors else 0.0
    output_score = min(len(exec_output) / 200.0, 1.0) * 0.3
    error_penalty = min(len(errors) * 0.1, 0.4)
    latency_penalty = min(duration_ms / 2000.0, 1.0) * 0.2
    code_penalty = 0.1 if not code.strip() else 0.0

    score = success_score + output_score - error_penalty - latency_penalty - code_penalty
    return max(0.0, min(1.0, score))
