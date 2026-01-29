from __future__ import annotations

import ast
import os
import signal
import subprocess
import sys
import tempfile
from typing import Callable, Optional

_MEMORY_LIMIT_MB = 256
_TIMEOUT_SECONDS = 5
BANNED_MODULES = {
    "http",
    "importlib",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
BANNED_BUILTINS = {"__import__", "compile", "eval", "exec", "input", "open"}


def _check_static_policy(tree: ast.AST) -> list[str]:
    violations: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name.split(".", 1)[0]
                if module_name in BANNED_MODULES:
                    violations.add(f"import of '{module_name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_name = node.module.split(".", 1)[0]
                if module_name in BANNED_MODULES:
                    violations.add(f"import of '{module_name}' is not allowed")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
                violations.add(f"call to '{node.func.id}' is not allowed")
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id in BANNED_BUILTINS:
                violations.add(f"use of '{node.id}' is not allowed")
    return sorted(violations)


def execute_untrusted(code: str) -> tuple[str, str]:
    """Execute untrusted Python code in a constrained subprocess.

    Returns (stdout, stderr) as normalized strings with predictable error surfaces.
    """

    def normalize_output(data: bytes) -> str:
        text = data.decode("utf-8", errors="replace")
        text = text.replace("\x00", "")
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def error_message(message: str) -> tuple[str, str]:
        return "", f"error: {message}"

    def preexec_limits() -> Optional[Callable[[], None]]:
        if os.name != "posix":
            return None

        def apply_limits() -> None:
            try:
                import resource
            except Exception:
                return

            memory_bytes = _MEMORY_LIMIT_MB * 1024 * 1024
            for limit in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
                try:
                    resource.setrlimit(limit, (memory_bytes, memory_bytes))
                except (ValueError, OSError):
                    continue

            try:
                resource.setrlimit(resource.RLIMIT_CPU, (_TIMEOUT_SECONDS, _TIMEOUT_SECONDS))
            except (ValueError, OSError):
                pass

            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
            except (ValueError, OSError):
                pass

        return apply_limits

    if not isinstance(code, str):
        return error_message("code must be a string")

    normalized_code = code.strip()
    if not normalized_code:
        return error_message("empty code")

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        detail = exc.msg or "invalid syntax"
        line = exc.lineno or 0
        col = exc.offset or 0
        return error_message(f"syntax error: {detail} (line {line}, column {col})")

    violations = _check_static_policy(tree)
    if violations:
        return "", "error: " + "; ".join(violations)

    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        temp_dir = tempfile.TemporaryDirectory(prefix="mvp_executor_")
        temp_path = temp_dir.name
        code_path = os.path.join(temp_path, "untrusted.py")
        with open(code_path, "w", encoding="utf-8") as handle:
            handle.write(code)

        env = {
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
        }

        try:
            result = subprocess.run(
                [sys.executable, code_path],
                shell=False,
                timeout=_TIMEOUT_SECONDS,
                capture_output=True,
                text=False,
                cwd=temp_path,
                env=env,
                stdin=subprocess.DEVNULL,
                preexec_fn=preexec_limits(),
            )
        except subprocess.TimeoutExpired as exc:
            stdout = normalize_output(exc.stdout or b"")
            stderr = normalize_output(exc.stderr or b"")
            timeout_msg = "error: execution timed out"
            stderr = f"{stderr}\n{timeout_msg}".strip() if stderr else timeout_msg
            return stdout, stderr
        except Exception as exc:
            return error_message(f"execution failed: {exc}")

        stdout_text = normalize_output(result.stdout or b"")
        stderr_text = normalize_output(result.stderr or b"")

        if result.returncode != 0:
            if result.returncode < 0:
                signum = -result.returncode
                try:
                    signal_name = signal.Signals(signum).name
                except Exception:
                    signal_name = f"signal {signum}"
                crash_msg = f"error: terminated by {signal_name}"
            else:
                crash_msg = f"error: exited with status {result.returncode}"
            stderr_text = f"{stderr_text}\n{crash_msg}".strip() if stderr_text else crash_msg

        return stdout_text, stderr_text
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
