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

Input/Output Schema:
    execute_task expects a dictionary with the following structure:
        {
            "objective": str,               # required, non-empty
            "constraints": list[str] | None # optional
        }
    It returns a JSON-serializable dictionary:
        {
            "objective": str,
            "constraints": list[str],
            "generated_code": str,
            "execution_output": str,
            "execution_error": str,
            "evaluation_error": str,
            "roi_score": float,
            "started_at": str,
            "finished_at": str,
            "duration_ms": int,
            "success": bool
        }

Error Handling:
    The workflow is defensive. It validates inputs, captures exceptions,
    sanitizes error output, and reports errors through execution_error and
    evaluation_error fields rather than raising.
"""

from __future__ import annotations

import ast
import datetime
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

import mvp_codegen
import mvp_evaluator
import mvp_executor

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


ALLOWED_IMPORTS = frozenset(
    {
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
)


def now_iso8601() -> str:
    """Return an ISO-8601 timestamp with timezone info when available."""
    return datetime.datetime.now().astimezone().isoformat()


def elapsed_ms(start: float, end: float) -> int:
    """Return elapsed time in integer milliseconds."""
    return int((end - start) * 1000)


def execute_task(task_dict: dict) -> dict:
    """Execute a task workflow end-to-end and return a JSON-serializable payload.

    Args:
        task_dict: Input task payload that must contain an objective and optional constraints.

    Returns:
        A JSON-serializable dictionary with objective, constraints, generated code,
        execution output, error details, ROI score, timestamps, duration, and success flag.
    """
    started_at = now_iso8601()
    start_time = time.time()
    generated_code = ""
    execution_output = ""
    execution_error = ""
    evaluation_error = ""
    roi_score = 0.0
    success = False
    spec: TaskSpec | None = None
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
                    spec = TaskSpec(objective=objective, constraints=constraints)
    except Exception as exc:  # pragma: no cover - defensive
        execution_error = sanitize_exception(exc)

    if spec is not None:
        try:
            generated_code = mvp_codegen.run_generation(task_dict)
            if not isinstance(generated_code, str):
                generated_code = str(generated_code)
            if not generated_code.strip():
                execution_error = "code generation failed"
        except Exception as exc:  # pragma: no cover - defensive
            execution_error = sanitize_exception(exc)

    if spec is not None and generated_code and not execution_error:
        try:
            exec_stdout, exec_stderr = mvp_executor.execute_untrusted(generated_code)
            execution_output = f"STDOUT:\n{exec_stdout}\n\nSTDERR:\n{exec_stderr}"
            if isinstance(exec_stderr, str) and exec_stderr.strip().startswith("error:"):
                execution_error = exec_stderr.strip()
        except Exception as exc:  # pragma: no cover - defensive
            execution_error = sanitize_exception(exc)
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
            evaluation_error = sanitize_exception(exc)
            roi_score = 0.0

    success = bool(spec and generated_code and not execution_error and not evaluation_error)
    finished_at = now_iso8601()
    end_time = time.time()
    duration_ms = elapsed_ms(start_time, end_time)
    sanitized_output = sanitize_error_output(execution_output)

    return {
        "objective": spec.objective if spec else "",
        "constraints": spec.constraints if spec else [],
        "generated_code": sanitize_error_output(generated_code),
        "execution_output": sanitized_output,
        "execution_error": sanitize_error_output(execution_error),
        "evaluation_error": sanitize_error_output(evaluation_error),
        "roi_score": float(roi_score),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_ms": duration_ms,
        "success": success,
    }


def generate_code(task: TaskSpec) -> GenerationResult:
    """Generate deterministic Python code for the given task.

    Deprecated: use mvp_codegen.run_generation instead.

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
        return GenerationResult(
            code="",
            error=sanitize_error_output("objective must be a non-empty string"),
            warnings=warnings,
        )

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
    warnings.extend(sanitize_error_output(warning) for warning in sanitization_warnings)
    if sanitization_errors:
        error = sanitize_error_output(sanitization_errors[0])
    elif not sanitized_code.strip():
        error = sanitize_error_output("generated code is empty after sanitization")
    return GenerationResult(code=sanitized_code, error=error, warnings=warnings)


def _generate_code(objective: str, constraints: list[str]) -> dict[str, object]:
    """Backward-compatible wrapper for code generation.

    Deprecated: prefer mvp_codegen.run_generation instead.
    """
    result = generate_code(TaskSpec(objective=objective, constraints=constraints))
    errors = [result.error] if result.error else []
    return {"generated_code": result.code, "errors": errors, "warnings": result.warnings}


def execute_generated_code(code: str) -> ExecutionResult:
    """Execute generated code with allowlisted imports, no dynamic imports, and no exec/eval.

    Deprecated: prefer mvp_executor.execute_untrusted instead.
    """
    if not isinstance(code, str) or not code.strip():
        return ExecutionResult(
            stdout="",
            stderr="",
            return_code=None,
            error=sanitize_error_output("code is empty or whitespace-only"),
        )

    try:
        ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}, column {exc.offset}" if exc.lineno and exc.offset else "unknown location"
        message = exc.msg or "invalid syntax"
        return ExecutionResult(
            stdout="",
            stderr="",
            return_code=None,
            error=sanitize_error_output(f"syntax error at {location}: {message}"),
        )

    policy_errors = _validate_code_policy(code)
    if policy_errors:
        return ExecutionResult(
            stdout="",
            stderr="",
            return_code=None,
            error=policy_errors[0],
        )

    try:
        stdout, stderr = mvp_executor.execute_untrusted(code)
        stdout = _strip_ansi(stdout)
        stderr = _strip_ansi(stderr)
        error = ""
        if stderr.strip().startswith("error:"):
            error = sanitize_error_output(stderr.strip())
        return ExecutionResult(
            stdout=stdout,
            stderr=stderr,
            return_code=None,
            error=error,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ExecutionResult(
            stdout="",
            stderr="",
            return_code=None,
            error=sanitize_exception(exc),
        )


def _sanitize_generated_code(code: str) -> tuple[str, list[str], list[str]]:
    """Strip forbidden imports and dynamic import patterns from generated code."""
    warnings: list[str] = []
    errors: list[str] = []
    if not isinstance(code, str) or not code.strip():
        return "", warnings, [sanitize_error_output("generated code is empty")]

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
    unsafe_call_patterns = ("exec(", "eval(")

    sanitized_lines: list[str] = []
    for line in code.splitlines():
        stripped = line.strip()
        if any(pattern in stripped for pattern in dynamic_patterns):
            warnings.append("dynamic import pattern removed from generated code")
            continue
        if any(pattern in stripped for pattern in unsafe_call_patterns):
            warnings.append("unsafe exec/eval pattern removed from generated code")
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
        errors.append(
            sanitize_error_output(f"generated code has syntax error at {location}: {exc.msg}")
        )
        return "", warnings, errors

    forbidden_found = contains_forbidden_imports(sanitized_code, forbidden_imports=forbidden_imports)
    if forbidden_found:
        forbidden_list = ", ".join(sorted(forbidden_found))
        errors.append(
            sanitize_error_output(
                f"forbidden imports detected after sanitization: {forbidden_list}"
            )
        )

    if contains_dynamic_imports(sanitized_code):
        errors.append(sanitize_error_output("dynamic import patterns detected after sanitization"))

    unsafe_calls = contains_exec_or_eval(sanitized_code)
    if unsafe_calls:
        unsafe_list = ", ".join(sorted(unsafe_calls))
        errors.append(
            sanitize_error_output(f"unsafe exec/eval patterns detected after sanitization: {unsafe_list}")
        )

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


def contains_forbidden_imports(
    code: str,
    *,
    forbidden_imports: set[str] | None = None,
    allowed_imports: set[str] | None = None,
) -> set[str]:
    """Return a set of forbidden imports found in code.

    Args:
        code: Python source code to scan.
        forbidden_imports: Explicitly forbidden root modules.
        allowed_imports: If provided, any import not in this set is treated as forbidden.

    Returns:
        A set of forbidden root module names.
    """
    if not isinstance(code, str) or not code.strip():
        return set()
    parsed = _parse_code(code)
    if parsed is None:
        return set()
    imports = _extract_imported_modules(parsed)
    forbidden: set[str] = set()
    if forbidden_imports:
        forbidden.update({name for name in imports if name in forbidden_imports})
    if allowed_imports is not None:
        forbidden.update({name for name in imports if name not in allowed_imports})
    return forbidden


def contains_dynamic_imports(code: str) -> bool:
    """Return True if dynamic import patterns are detected."""
    if not isinstance(code, str) or not code.strip():
        return False
    parsed = _parse_code(code)
    if parsed is None:
        return False
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


def contains_exec_or_eval(code: str) -> set[str]:
    """Return the set of unsafe builtins (exec/eval) found in code."""
    if not isinstance(code, str) or not code.strip():
        return set()
    parsed = _parse_code(code)
    if parsed is None:
        return set()
    unsafe_calls: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in {"exec", "eval"}:
                unsafe_calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id in {"builtins", "__builtins__"}
                    and func.attr in {"exec", "eval"}
                ):
                    unsafe_calls.add(func.attr)
    return unsafe_calls


def _validate_code_policy(code: str) -> list[str]:
    """Validate code against the shared import and unsafe call policy."""
    errors: list[str] = []
    forbidden_imports = contains_forbidden_imports(code, allowed_imports=set(ALLOWED_IMPORTS))
    if forbidden_imports:
        forbidden_list = ", ".join(sorted(forbidden_imports))
        errors.append(sanitize_error_output(f"forbidden imports detected: {forbidden_list}"))
        return errors
    if contains_dynamic_imports(code):
        errors.append(sanitize_error_output("dynamic import patterns detected"))
        return errors
    unsafe_calls = contains_exec_or_eval(code)
    if unsafe_calls:
        unsafe_list = ", ".join(sorted(unsafe_calls))
        errors.append(sanitize_error_output(f"unsafe call patterns detected: {unsafe_list}"))
    return errors


def _execute_code(code: str, timeout_s: float) -> dict[str, object]:
    """Execute code in a temporary file and capture sanitized output.

    Deprecated: prefer mvp_executor.execute_untrusted instead.

    Args:
        code: Python source code to execute.
        timeout_s: Timeout in seconds (ignored; mvp_executor enforces its own limits).

    Returns:
        A dictionary with execution output and errors following the unified import policy.
    """
    errors: list[str] = []
    stdout = ""
    stderr = ""
    exit_code: int | None = None
    timed_out = False
    _ = timeout_s

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
        ast.parse(code)
    except SyntaxError as exc:
        location = f"line {exc.lineno}, column {exc.offset}" if exc.lineno and exc.offset else "unknown location"
        message = exc.msg or "invalid syntax"
        errors.append(sanitize_error_output(f"syntax error at {location}: {message}"))
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "execution_output": "",
            "errors": errors,
        }

    errors.extend(_validate_code_policy(code))
    if errors:
        return {
            "stdout": "",
            "stderr": "",
            "exit_code": exit_code,
            "timed_out": timed_out,
            "execution_output": "",
            "errors": errors,
        }

    try:
        stdout, stderr = mvp_executor.execute_untrusted(code)
        stdout = _strip_ansi(stdout)
        stderr = _strip_ansi(stderr)
        if stderr.strip().startswith("error:"):
            errors.append(sanitize_error_output(stderr.strip()))
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(sanitize_exception(exc))

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

    Deprecated: prefer mvp_evaluator.evaluate_roi instead.

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

        if return_code is None or exec_error:
            rationale_parts: list[str] = []
            if return_code is None:
                rationale_parts.append("missing return code")
            if exec_error:
                rationale_parts.append("execution error present")
            rationale_parts.append("execution failure")
            return EvaluationResult(
                roi_score=0.0,
                evaluation_error="",
                rationale=_build_rationale(rationale_parts),
            )

        score = 0.0
        rationale_parts = []

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
            evaluation_error=sanitize_exception(exc),
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
        error = sanitize_error_output(errors[0]) if errors else ""
        return ExecutionResult(stdout=stdout, stderr=stderr, return_code=return_code, error=error)
    except Exception as exc:  # pragma: no cover - defensive
        return ExecutionResult(stdout="", stderr="", return_code=None, error=sanitize_exception(exc))


def _build_rationale(parts: list[str]) -> str:
    """Build a short, sanitized rationale string."""
    if not parts:
        return ""
    rationale = sanitize_error_output("; ".join(parts))
    max_len = 160
    if len(rationale) > max_len:
        rationale = rationale[: max_len - 3].rstrip() + "..."
    return rationale


def _evaluate_result(code: str, exec_result: dict[str, object]) -> dict[str, object]:
    """Evaluate execution results and provide an ROI score.

    Deprecated: prefer mvp_evaluator.evaluate_roi instead.

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
            "errors": [sanitize_exception(exc)],
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


def sanitize_error_output(text: str) -> str:
    """Remove ANSI codes and stack traces from error output."""
    if not isinstance(text, str):
        text = str(text)
    cleaned = _strip_ansi(text)
    cleaned = _strip_tracebacks(cleaned)
    return cleaned.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def sanitize_exception(exc: BaseException | str) -> str:
    """Return a sanitized exception message without ANSI or tracebacks."""
    message = str(exc) if exc else "unexpected error"
    return sanitize_error_output(message) or "unexpected error"


def _sanitize_error(stderr: str) -> str:
    """Reduce stderr output to a single sanitized line."""
    lines = [line.strip() for line in sanitize_error_output(stderr).splitlines() if line.strip()]
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


def _parse_code(code: str) -> ast.AST | None:
    """Parse code into an AST or return None on syntax errors."""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def _extract_imported_modules(parsed: ast.AST) -> set[str]:
    """Extract root module names from import statements in the AST."""
    found: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module.split(".")[0] if node.module else ""
            found.add(module_name or "<relative>")
    return found
