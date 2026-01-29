"""Minimal MVP workflow execution helpers."""

from __future__ import annotations

import ast
import datetime
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

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
        generation_result = _generate_code(normalized_task)
        generated_code = generation_result.get("generated_code", "")
        if generation_result.get("error"):
            errors.append(str(generation_result["error"]))
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"generation error: {exc}")

    validation_result: dict[str, Any] = {}
    try:
        validation_result = _validate_code(generated_code)
        if validation_result.get("error"):
            errors.append(str(validation_result["error"]))
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"validation error: {exc}")

    try:
        code_to_execute = generated_code if not validation_result.get("error") else ""
        execution_result = _execute_code(code_to_execute, timeout_s=5.0)
        if execution_result.get("error"):
            errors.append(str(execution_result["error"]))
        execution_output = execution_result.get("stdout", "")
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"execution error: {exc}")

    try:
        evaluation_result = _evaluate_result(normalized_task, generated_code, execution_output, errors)
        roi_score = float(evaluation_result.get("roi_score", 0.0))
        success = bool(evaluation_result.get("success", False))
        if evaluation_result.get("error"):
            errors.append(str(evaluation_result["error"]))
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"evaluation error: {exc}")
        success = False
        roi_score = 0.0

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


def _generate_code(task: dict[str, Any]) -> dict[str, Any]:
    """Generate deterministic Python code for the given task.

    Args:
        task: Normalized task payload.

    Returns:
        A dictionary containing generated code and an optional error.
    """
    try:
        objective = str(task.get("objective", "")).strip()
        constraints = task.get("constraints", {})
        if not objective:
            return {"generated_code": "", "error": "objective is empty"}
        if constraints is None:
            constraints = {}
        if not isinstance(constraints, dict):
            return {"generated_code": "", "error": "constraints must be a dict"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"generated_code": "", "error": f"generation error: {exc}"}

    constraints_items = ", ".join(f"{key}={value}" for key, value in sorted(constraints.items()))
    constraints_summary = constraints_items if constraints_items else "none"

    code = (
        "def main():\n"
        f"    objective = {objective!r}\n"
        f"    constraints = {constraints_summary!r}\n"
        "    print(f\"Objective: {objective}\")\n"
        "    print(f\"Constraints: {constraints}\")\n"
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    return {"generated_code": code, "error": None}


def _validate_code(code: str) -> dict[str, Any]:
    """Validate generated code for syntax and forbidden imports.

    Args:
        code: Generated Python code.

    Returns:
        A dictionary containing validation result and an optional error message.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"error": f"syntax error: {exc.msg}"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"validation error: {exc}"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in node.names]
            for module in modules:
                if not _is_stdlib_module(module):
                    return {"error": f"forbidden import: {module}"}

    if _find_dynamic_import_calls(tree):
        return {"error": "dynamic import detected"}

    return {"error": None}


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


def _execute_code(code: str, timeout_s: float) -> dict[str, Any]:
    """Execute code after static safety checks.

    Args:
        code: Python source code to execute.
        timeout_s: Timeout in seconds.

    Returns:
        A dictionary containing stdout, stderr, and an optional error message.
    """
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
        stdout = _ANSI_ESCAPE_RE.sub("", result.stdout or "")
        stderr = _ANSI_ESCAPE_RE.sub("", result.stderr or "")

        if result.returncode != 0:
            error_line = _redact_error(stderr) or "execution failed"
            return {"stdout": stdout, "stderr": stderr, "error": error_line}

        return {"stdout": stdout, "stderr": stderr, "error": None}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "", "error": "timeout"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"stdout": "", "stderr": "", "error": f"execution error: {exc}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _redact_error(stderr: str) -> str:
    """Reduce stderr output to a single sanitized line."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "execution error"
    return lines[-1]


def _evaluate_result(
    task: dict[str, Any],
    generated_code: str,
    execution_output: str,
    errors: list[str],
) -> dict[str, Any]:
    """Compute deterministic ROI score and success flag.

    Args:
        task: Normalized task payload.
        generated_code: Generated Python code.
        execution_output: Captured stdout from execution.
        errors: Collected errors from previous steps.

    Returns:
        A dictionary containing roi_score, success, and optional error.
    """
    objective = str(task.get("objective", ""))
    objective_len = max(len(objective), 1)
    code_len = max(len(generated_code), 1)

    success = not errors
    output_bonus = 0.1 if execution_output else 0.0

    base_score = min(1.0, (objective_len / (code_len + 1)) + output_bonus)
    if not success:
        base_score = max(0.0, base_score - 0.2)

    roi_score = max(0.0, min(1.0, base_score))
    return {"roi_score": roi_score, "success": success, "error": None}
