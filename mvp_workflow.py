"""Minimal MVP workflow execution helpers."""

from __future__ import annotations

import ast
import datetime
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def execute_task(task_dict: dict) -> dict:
    """Execute a task workflow end-to-end and return a JSON-serializable payload."""
    started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    errors: list[str] = []

    if not isinstance(task_dict, dict):
        finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "generated_code": None,
            "execution_output": None,
            "errors": ["task_dict must be a dict"],
            "roi_score": 0.0,
            "started_at": started_at,
            "finished_at": finished_at,
            "success": False,
        }

    objective = task_dict.get("objective")
    constraints = task_dict.get("constraints")

    if not isinstance(objective, str) or not objective.strip():
        finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "generated_code": None,
            "execution_output": None,
            "errors": ["objective must be a non-empty string"],
            "roi_score": 0.0,
            "started_at": started_at,
            "finished_at": finished_at,
            "success": False,
        }

    if constraints is not None and not isinstance(constraints, dict):
        finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {
            "generated_code": None,
            "execution_output": None,
            "errors": ["constraints must be a dict when provided"],
            "roi_score": 0.0,
            "started_at": started_at,
            "finished_at": finished_at,
            "success": False,
        }

    task: Dict[str, Any] = {"objective": objective.strip(), "constraints": constraints or {}}

    generation_result = _generate_code(task)
    generated_code = generation_result.get("code")
    if generation_result.get("error"):
        errors.append(str(generation_result["error"]))
        exec_result = {"stdout": "", "stderr": "", "error": "generation_failed"}
    else:
        exec_result = _execute_code(generated_code, timeout_s=5.0)
        if exec_result.get("error"):
            errors.append(str(exec_result["error"]))

    exec_result_with_code = dict(exec_result)
    exec_result_with_code["code"] = generated_code or ""
    evaluation = _evaluate_result(task, exec_result_with_code)
    if evaluation.get("error"):
        errors.append(str(evaluation["error"]))

    finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return {
        "generated_code": generated_code,
        "execution_output": exec_result.get("stdout", "") if not exec_result.get("error") else "",
        "errors": errors or None,
        "roi_score": float(evaluation.get("roi_score", 0.0)),
        "started_at": started_at,
        "finished_at": finished_at,
        "success": bool(evaluation.get("success", False)),
    }


def _generate_code(task: dict) -> dict:
    """Generate deterministic Python code for the given task."""
    try:
        objective = str(task.get("objective", "")).strip()
        constraints = task.get("constraints", {})
        if not objective:
            return {"code": "", "error": "objective is empty"}
        if constraints is None:
            constraints = {}
        if not isinstance(constraints, dict):
            return {"code": "", "error": "constraints must be a dict"}
    except Exception as exc:  # pragma: no cover - defensive
        return {"code": "", "error": f"generation error: {exc}"}

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
    return {"code": code, "error": None}


def _is_stdlib_module(module_name: str) -> bool:
    stdlib_modules = getattr(sys, "stdlib_module_names", None)
    if stdlib_modules is None:
        stdlib_modules = set(sys.builtin_module_names)
    return module_name in stdlib_modules


def _find_dynamic_import_calls(tree: ast.AST) -> bool:
    forbidden_names = {"__import__", "importlib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in forbidden_names:
                return True
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id in forbidden_names:
                    return True
    return False


def _execute_code(code: str, timeout_s: float) -> dict:
    """Execute code after static safety checks."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"stdout": "", "stderr": "", "error": f"syntax error: {exc.msg}"}

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name.split(".")[0] for alias in node.names]
            for module in modules:
                if not _is_stdlib_module(module):
                    return {"stdout": "", "stderr": "", "error": f"forbidden import: {module}"}

    if _find_dynamic_import_calls(tree):
        return {"stdout": "", "stderr": "", "error": "dynamic import detected"}

    tmp_path: Optional[str] = None
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
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return "execution error"
    return lines[-1]


def _evaluate_result(task: dict, exec_result: dict) -> dict:
    """Compute deterministic ROI score and success flag."""
    objective = str(task.get("objective", ""))
    code = exec_result.get("code", "")
    stdout = exec_result.get("stdout", "")
    error = exec_result.get("error")

    objective_len = max(len(objective), 1)
    code_len = max(len(code), 1)

    success = error is None
    output_bonus = 0.1 if stdout else 0.0

    base_score = min(1.0, (objective_len / (code_len + 1)) + output_bonus)
    if not success:
        base_score = max(0.0, base_score - 0.2)

    roi_score = max(0.0, min(1.0, base_score))
    return {"roi_score": roi_score, "success": success, "error": None}
