from __future__ import annotations

import ast
import math
import os
import subprocess
import sys
import tempfile
from typing import Callable, Optional


def run_untrusted_code(
    code: str,
    timeout_s: float = 5.0,
    memory_limit_mb: int = 256,
) -> tuple[str, str]:
    """Execute untrusted Python code in a constrained subprocess.

    Returns a tuple of (stdout, stderr) with normalized line endings.
    """

    def normalize_output(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def to_text(value: object) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            return value
        return str(value)

    def format_syntax_error(error: SyntaxError) -> str:
        message = error.msg or "invalid syntax"
        line = error.lineno
        column = error.offset
        if line is not None and column is not None:
            return f"SyntaxError: {message} (line {line}, column {column})"
        if line is not None:
            return f"SyntaxError: {message} (line {line})"
        return f"SyntaxError: {message}"

    def find_forbidden_construct(tree: ast.AST) -> Optional[str]:
        forbidden_modules = {
            "os",
            "sys",
            "subprocess",
            "socket",
            "pathlib",
            "ctypes",
            "importlib",
            "multiprocessing",
            "threading",
            "signal",
            "resource",
        }
        forbidden_calls = {"exec", "eval", "compile", "__import__", "open"}

        def is_forbidden_module(module: Optional[str]) -> Optional[str]:
            if not module:
                return None
            root = module.split(".")[0]
            if root in forbidden_modules:
                return root
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = is_forbidden_module(alias.name)
                    if root:
                        return f"ImportError: import of '{root}' is blocked"
            elif isinstance(node, ast.ImportFrom):
                root = is_forbidden_module(node.module)
                if root:
                    return f"ImportError: import of '{root}' is blocked"
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in forbidden_calls:
                    return f"ImportError: call to '{func.id}' is blocked"
                if isinstance(func, ast.Attribute) and func.attr in forbidden_calls:
                    if isinstance(func.value, ast.Name) and func.value.id == "builtins":
                        return f"ImportError: call to '{func.attr}' is blocked"
        return None

    if not code.strip():
        return "", ""

    try:
        parsed = ast.parse(code)
    except SyntaxError as exc:
        return "", format_syntax_error(exc)

    validation_error = find_forbidden_construct(parsed)
    if validation_error:
        return "", validation_error

    def build_preexec_fn() -> Optional[Callable[[], None]]:
        if os.name != "posix":
            return None

        def apply_limits() -> None:
            try:
                import resource
            except Exception:
                return

            limit_bytes = max(1, int(memory_limit_mb)) * 1024 * 1024
            for limit in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
                try:
                    resource.setrlimit(limit, (limit_bytes, limit_bytes))
                    break
                except (ValueError, OSError):
                    continue

            cpu_limit = max(1, int(math.ceil(timeout_s)))
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
            except (ValueError, OSError):
                pass

            for limit_value in (0, 1):
                try:
                    resource.setrlimit(resource.RLIMIT_NPROC, (limit_value, limit_value))
                    break
                except (ValueError, OSError):
                    continue

            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
            except (ValueError, OSError):
                pass

        return apply_limits

    stdout_text = ""
    stderr_text = ""

    try:
        with tempfile.TemporaryDirectory(prefix="mvp_executor_") as temp_dir:
            code_path = os.path.join(temp_dir, "untrusted.py")
            with open(code_path, "w", encoding="utf-8") as handle:
                handle.write(code)

            try:
                result = subprocess.run(
                    [sys.executable, "-I", "-S", "-u", code_path],
                    cwd=temp_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    close_fds=True,
                    timeout=timeout_s,
                    text=True,
                    errors="replace",
                    preexec_fn=build_preexec_fn(),
                )
                stdout_text = result.stdout
                stderr_text = result.stderr
            except subprocess.TimeoutExpired as exc:
                stdout_text = to_text(exc.stdout or "")
                stderr_text = to_text(exc.stderr or "")
                timeout_message = f"Execution timed out after {timeout_s}s"
                if stderr_text:
                    stderr_text = f"{stderr_text}\n{timeout_message}"
                else:
                    stderr_text = timeout_message
            except Exception as exc:
                stderr_text = f"Execution failed: {exc}"
    finally:
        stdout_text = normalize_output(stdout_text)
        stderr_text = normalize_output(stderr_text)

    return stdout_text, stderr_text
