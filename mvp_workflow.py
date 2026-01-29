"""Minimal MVP workflow execution helpers."""

from __future__ import annotations

import ast
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


@dataclass(frozen=True)
class TaskSpec:
    """Normalized task specification for execution."""

    objective: str
    constraints: list[str]


@dataclass(frozen=True)
class GenerationResult:
    """Structured output for code generation."""

    code: str
    error: str
    warnings: list[str]


@dataclass(frozen=True)
class ExecutionResult:
    """Structured output for executing generated code."""

    stdout: str
    stderr: str
    return_code: int | None
    error: str


@dataclass(frozen=True)
class EvaluationResult:
    """Structured output for deterministic ROI evaluation."""

    roi_score: float
    evaluation_error: str
    rationale: str


_EXECUTION_TIMEOUT_S = 5.0


def execute_task(task_dict: dict) -> dict:
    """Execute a task workflow end-to-end and return a JSON-serializable payload.

    Args:
        task_dict: Input task payload that must contain an objective and optional constraints.

    Returns:
        A JSON-serializable dictionary with objective, constraints, generated code,
        execution output, error details, ROI score, timestamps, duration, and success flag.
    """
    started_at = datetime.datetime.now(datetime.timezone.utc)
    started_at_iso = started_at.isoformat()
    generated_code = ""
    execution_output = ""
    execution_error = ""
    evaluation_error = ""
    roi_score = 0.0
    success = False
    spec: TaskSpec | None = None
    execution_result: dict[str, object] = {}

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
                    spec = TaskSpec(objective=objective, constraints=constraints)
    except Exception as exc:  # pragma: no cover - defensive
        execution_error = _sanitize_exception_message(exc)

    if spec is not None:
        try:
            generation_result = generate_code(spec)
            generated_code = generation_result.code
            if generation_result.error:
                execution_error = generation_result.error
        except Exception as exc:  # pragma: no cover - defensive
            execution_error = _sanitize_exception_message(exc)

    if spec is not None and generated_code:
        try:
            execution_result = _execute_code(generated_code, timeout_s=5.0)
            execution_output = str(execution_result.get("execution_output", ""))
            execution_errors = _coerce_error_list(execution_result.get("errors"))
            if execution_errors:
                execution_error = execution_errors[0]
        except Exception as exc:  # pragma: no cover - defensive
            execution_error = _sanitize_exception_message(exc)
            execution_result = {"execution_output": "", "errors": [execution_error]}

    if spec is not None:
        try:
            exec_struct = _execution_result_from_dict(execution_result)
            evaluation_result = evaluate_result(spec, exec_struct)
            roi_score = float(evaluation_result.roi_score)
            if evaluation_result.evaluation_error:
                evaluation_error = evaluation_result.evaluation_error
        except Exception as exc:  # pragma: no cover - defensive
            evaluation_error = _sanitize_exception_message(exc)
            roi_score = 0.0

    success = bool(spec and generated_code and not execution_error and not evaluation_error)
    finished_at = datetime.datetime.now(datetime.timezone.utc)
    finished_at_iso = finished_at.isoformat()
    duration_ms = int((finished_at - started_at).total_seconds() * 1000)
    sanitized_output = _sanitize_output(execution_output)

    return {
        "objective": spec.objective if spec else "",
        "constraints": spec.constraints if spec else [],
        "generated_code": _sanitize_output(generated_code),
        "execution_output": sanitized_output,
        "execution_error": _sanitize_output(execution_error),
        "evaluation_error": _sanitize_output(evaluation_error),
        "roi_score": float(roi_score),
        "started_at": started_at_iso,
        "finished_at": finished_at_iso,
        "duration_ms": duration_ms,
        "success": success,
    }


def generate_code(task: TaskSpec) -> GenerationResult:
    """Generate deterministic Python code for the given task.

    Args:
        task: Normalized TaskSpec input.

    Returns:
        Structured generation result (code, error, warnings) with no exceptions.
    """
    warnings: list[str] = []
    error = ""

    objective_text = task.objective.strip() if isinstance(task.objective, str) else ""
    constraints_payload = (
        [item for item in task.constraints if isinstance(item, str)]
        if isinstance(task.constraints, list)
        else []
    )

    if not objective_text:
        return GenerationResult(code="", error="objective must be a non-empty string", warnings=warnings)

    code_lines = [
        '"""Auto-generated script."""',
        f"OBJECTIVE = {objective_text!r}",
        f"CONSTRAINTS = {constraints_payload!r}",
        "",
        "def main() -> None:",
        "    print('Objective:')",
        "    print(OBJECTIVE)",
        "    if CONSTRAINTS:",
        "        print('Constraints:')",
        "        for constraint in CONSTRAINTS:",
        "            print(f'- {constraint}')",
        "    else:",
        "        print('Constraints: none')",
        "",
        "if __name__ == '__main__':",
        "    main()",
        "",
    ]

    sanitized_code, sanitization_warnings, sanitization_errors = _sanitize_generated_code("\n".join(code_lines))
    warnings.extend(sanitization_warnings)
    if sanitization_errors:
        error = sanitization_errors[0]
    elif not sanitized_code.strip():
        error = "generated code is empty after sanitization"
    return GenerationResult(code=sanitized_code, error=error, warnings=warnings)


def _generate_code(objective: str, constraints: list[str]) -> dict[str, object]:
    """Backward-compatible wrapper for code generation."""
    result = generate_code(TaskSpec(objective=objective, constraints=constraints))
    errors = [result.error] if result.error else []
    return {"generated_code": result.code, "errors": errors, "warnings": result.warnings}


def execute_generated_code(code: str) -> ExecutionResult:
    """Execute generated code in a temporary file with safety checks."""
    if not isinstance(code, str) or not code.strip():
        return ExecutionResult(stdout="", stderr="", return_code=None, error="code is empty or whitespace-only")

    try:
        parsed = ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}, column {exc.offset}" if exc.lineno and exc.offset else "unknown location"
        message = exc.msg or "invalid syntax"
        return ExecutionResult(
            stdout="",
            stderr="",
            return_code=None,
            error=f"syntax error at {location}: {message}",
        )

    forbidden_imports = {
        "builtins",
        "ctypes",
        "importlib",
        "inspect",
        "os",
        "subprocess",
        "sys",
    }
    forbidden_found = _find_forbidden_imports(parsed, forbidden_imports)
    if forbidden_found:
        forbidden_list = ", ".join(sorted(forbidden_found))
        return ExecutionResult(
            stdout="",
            stderr="",
            return_code=None,
            error=f"forbidden imports detected: {forbidden_list}",
        )

    if _find_dynamic_imports(parsed):
        return ExecutionResult(
            stdout="",
            stderr="",
            return_code=None,
            error="dynamic import patterns detected",
        )

    if sys.version_info >= (3, 10):
        interpreter = sys.executable
    else:
        interpreter = shutil.which("python3.10")
        if interpreter is None:
            return ExecutionResult(
                stdout="",
                stderr="",
                return_code=None,
                error="python 3.10+ interpreter not available",
            )

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(code)
            temp_path = temp_file.name

        try:
            result = subprocess.run(
                [interpreter, temp_path],
                capture_output=True,
                text=True,
                timeout=_EXECUTION_TIMEOUT_S,
                check=False,
                encoding="utf-8",
                errors="replace",
            )
            stdout = _strip_ansi(result.stdout or "")
            stderr = _strip_ansi(result.stderr or "")
            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                return_code=result.returncode,
                error="",
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _strip_ansi((exc.stdout or "").strip())
            stderr = _strip_ansi((exc.stderr or "").strip())
            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                return_code=None,
                error="execution timed out",
            )
    except Exception as exc:  # pragma: no cover - defensive
        return ExecutionResult(
            stdout="",
            stderr="",
            return_code=None,
            error=f"execution error: {exc}",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _sanitize_generated_code(code: str) -> tuple[str, list[str], list[str]]:
    """Strip forbidden imports and dynamic import patterns from generated code."""
    warnings: list[str] = []
    errors: list[str] = []
    if not isinstance(code, str) or not code.strip():
        return "", warnings, ["generated code is empty"]

    forbidden_imports = {
        "builtins",
        "ctypes",
        "importlib",
        "inspect",
        "os",
        "subprocess",
        "sys",
    }
    dynamic_patterns = ("__import__", "importlib.import_module", "importlib.reload")

    sanitized_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if any(pattern in stripped for pattern in dynamic_patterns):
            warnings.append("dynamic import pattern removed from generated code")
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            module_name = _extract_import_module(stripped)
            if module_name in forbidden_imports:
                warnings.append(f"forbidden import '{module_name}' removed from generated code")
                continue
        sanitized_lines.append(line)

    sanitized_code = "\n".join(sanitized_lines).rstrip() + "\n"

    try:
        parsed = ast.parse(sanitized_code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}, column {exc.offset}" if exc.lineno and exc.offset else "unknown location"
        errors.append(f"generated code has syntax error at {location}: {exc.msg}")
        return "", warnings, errors

    forbidden_found = _find_forbidden_imports(parsed, forbidden_imports)
    if forbidden_found:
        forbidden_list = ", ".join(sorted(forbidden_found))
        errors.append(f"forbidden imports detected after sanitization: {forbidden_list}")

    dynamic_found = _find_dynamic_imports(parsed)
    if dynamic_found:
        errors.append("dynamic import patterns detected after sanitization")

    if errors:
        return "", warnings, errors
    return sanitized_code, warnings, errors


def _extract_import_module(line: str) -> str:
    """Extract the root module name from an import line."""
    if line.startswith("import "):
        module_part = line.replace("import ", "", 1).strip()
        return module_part.split(".")[0].split()[0]
    if line.startswith("from "):
        module_part = line.replace("from ", "", 1).strip()
        return module_part.split(".")[0].split()[0]
    return ""


def _find_forbidden_imports(parsed: ast.AST, forbidden_imports: set[str]) -> set[str]:
    """Return a set of forbidden imports discovered in AST."""
    found: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".")[0]
                if root_name in forbidden_imports:
                    found.add(root_name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module.split(".")[0] if node.module else ""
            if module_name in forbidden_imports:
                found.add(module_name or "<relative>")
    return found


def _find_dynamic_imports(parsed: ast.AST) -> bool:
    """Detect dynamic import patterns in AST."""
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "__import__":
                return True
            if isinstance(node.func, ast.Attribute):
                attr = node.func
                if (
                    isinstance(attr.value, ast.Name)
                    and attr.value.id == "importlib"
                    and attr.attr in {"import_module", "reload"}
                ):
                    return True
    return False


def _execute_code(code: str, timeout_s: float) -> dict[str, object]:
    """Execute code in a temporary file and capture sanitized output.

    Args:
        code: Python source code to execute.
        timeout_s: Timeout in seconds.

    Returns:
        A dictionary with execution output and errors.
    """
    errors: list[str] = []
    stdout = ""
    stderr = ""
    exit_code: int | None = None
    timed_out = False

    if not code.strip():
        errors.append("code is empty or whitespace-only")
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "execution_output": "",
            "errors": errors,
        }

    try:
        parsed = ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}, column {exc.offset}" if exc.lineno and exc.offset else "unknown location"
        message = exc.msg or "invalid syntax"
        errors.append(f"syntax error at {location}: {message}")
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "execution_output": "",
            "errors": errors,
        }

    allowed_imports = {
        "collections",
        "datetime",
        "functools",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "time",
        "typing",
    }
    forbidden_imports: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".")[0]
                if root_name not in allowed_imports:
                    forbidden_imports.add(root_name)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module.split(".")[0] if node.module else ""
            if module_name not in allowed_imports:
                forbidden_imports.add(module_name or "<relative>")
    if forbidden_imports:
        forbidden_list = ", ".join(sorted(forbidden_imports))
        errors.append(f"forbidden imports detected: {forbidden_list}")
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "execution_output": "",
            "errors": errors,
        }

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, "generated_script.py")
            with open(script_path, "w", encoding="utf-8") as script_file:
                script_file.write(code)

            restricted_env = {
                "PATH": "/usr/bin:/bin",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "LANG": "C.UTF-8",
            }

            try:
                result = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                    env=restricted_env,
                    encoding="utf-8",
                    errors="replace",
                )
                stdout = _strip_ansi(result.stdout or "")
                stderr = _strip_ansi(result.stderr or "")
                exit_code = result.returncode

                if exit_code != 0:
                    errors.append(_sanitize_error(stderr) or "execution failed")
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = _strip_ansi((exc.stdout or "").strip())
                stderr = _strip_ansi((exc.stderr or "").strip())
                errors.append("execution timed out")
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"execution error: {exc}")

    stdout = _strip_ansi(stdout)
    stderr = _strip_ansi(stderr)
    execution_output = f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "execution_output": execution_output,
        "errors": errors,
    }


def evaluate_result(task: TaskSpec, exec_result: ExecutionResult) -> EvaluationResult:
    """Evaluate execution results and provide a deterministic ROI score.

    Heuristics (bounded to 0.0–1.0):
    - +0.60 when return_code is zero and no execution error is reported.
    - +0.25 when stdout is non-empty after stripping.
    - +0.15 when stderr is empty after stripping.

    The score is deterministic and depends only on the inputs. Failures return
    an ``evaluation_error`` string and a zero score without raising exceptions.
    """
    try:
        if not isinstance(task, TaskSpec):
            return EvaluationResult(
                roi_score=0.0,
                evaluation_error="task must be TaskSpec",
                rationale="invalid task specification",
            )
        if not isinstance(task.objective, str) or not task.objective.strip():
            return EvaluationResult(
                roi_score=0.0,
                evaluation_error="task objective must be a non-empty string",
                rationale="invalid task objective",
            )
        if not isinstance(exec_result, ExecutionResult):
            return EvaluationResult(
                roi_score=0.0,
                evaluation_error="exec_result must be ExecutionResult",
                rationale="invalid execution result",
            )

        stdout = exec_result.stdout or ""
        stderr = exec_result.stderr or ""
        return_code = exec_result.return_code
        exec_error = exec_result.error or ""

        score = 0.0
        rationale_parts: list[str] = []

        if return_code == 0 and not exec_error:
            score += 0.60
            rationale_parts.append("zero return code")
        else:
            rationale_parts.append("non-zero return code")
            if exec_error:
                rationale_parts.append("execution error present")

        if stdout.strip():
            score += 0.25
            rationale_parts.append("non-empty stdout")
        else:
            rationale_parts.append("empty stdout")

        if not stderr.strip():
            score += 0.15
            rationale_parts.append("empty stderr")
        else:
            rationale_parts.append("stderr present")

        score = max(0.0, min(1.0, score))
        rationale = _build_rationale(rationale_parts)
        return EvaluationResult(roi_score=score, evaluation_error="", rationale=rationale)
    except Exception as exc:  # pragma: no cover - defensive
        return EvaluationResult(
            roi_score=0.0,
            evaluation_error=_sanitize_exception_message(exc),
            rationale="evaluation failure",
        )


def _execution_result_from_dict(exec_result: dict[str, object]) -> ExecutionResult:
    """Coerce a dictionary into an ExecutionResult without raising exceptions."""
    try:
        stdout = str(exec_result.get("stdout", "") or "")
        stderr = str(exec_result.get("stderr", "") or "")
        exit_code_raw = exec_result.get("exit_code")
        return_code: int | None
        if isinstance(exit_code_raw, bool):
            return_code = int(exit_code_raw)
        elif isinstance(exit_code_raw, (int, float)):
            return_code = int(exit_code_raw)
        else:
            return_code = None
        errors = _coerce_error_list(exec_result.get("errors"))
        error = _sanitize_output(errors[0]) if errors else ""
        return ExecutionResult(stdout=stdout, stderr=stderr, return_code=return_code, error=error)
    except Exception as exc:  # pragma: no cover - defensive
        return ExecutionResult(stdout="", stderr="", return_code=None, error=_sanitize_exception_message(exc))


def _build_rationale(parts: list[str]) -> str:
    """Build a short, sanitized rationale string."""
    if not parts:
        return ""
    rationale = _sanitize_output("; ".join(parts))
    max_len = 160
    if len(rationale) > max_len:
        rationale = rationale[: max_len - 3].rstrip() + "..."
    return rationale


def _evaluate_result(code: str, exec_result: dict[str, object]) -> dict[str, object]:
    """Evaluate execution results and provide an ROI score.

    Args:
        code: Generated code string.
        exec_result: Execution result dictionary.

    Returns:
        A dictionary with ROI score and errors.
    """
    errors: list[str] = []
    notes: list[str] = []

    try:
        if not isinstance(exec_result, dict):
            raise TypeError("exec_result must be a dict")

        code_text = code if isinstance(code, str) else ""
        if not code_text.strip():
            errors.append("no code generated")

        exec_struct = _execution_result_from_dict(exec_result)
        task = TaskSpec(objective=code_text or "generated code", constraints=[])
        evaluation = evaluate_result(task, exec_struct)

        if evaluation.evaluation_error:
            errors.append(evaluation.evaluation_error)
        if evaluation.rationale:
            notes.append(evaluation.rationale)

        return {
            "roi_score": float(evaluation.roi_score),
            "errors": errors,
            "evaluation_notes": notes,
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {
            "roi_score": 0.0,
            "errors": [_sanitize_exception_message(exc)],
            "evaluation_notes": [],
        }


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


def _sanitize_output(text: str) -> str:
    """Ensure output contains UTF-8 text without ANSI or stack traces."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = _strip_ansi(text)
    cleaned = _strip_tracebacks(cleaned)
    return cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def _sanitize_exception_message(exc: BaseException) -> str:
    """Sanitize exception messages to avoid tracebacks and ANSI sequences."""
    message = str(exc) if exc else "unexpected error"
    return _sanitize_output(message) or "unexpected error"


def _sanitize_error(stderr: str) -> str:
    """Reduce stderr output to a single sanitized line."""
    lines = [line.strip() for line in _sanitize_output(stderr).splitlines() if line.strip()]
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
