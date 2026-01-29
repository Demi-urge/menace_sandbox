from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
from typing import Iterable


def run_untrusted_code(code: str, denied_imports: Iterable[str] | None = None) -> tuple[str, str]:
    """Execute untrusted Python code in a constrained subprocess.

    The function returns a tuple of (stdout, stderr) with normalized line endings.
    """
    if not code.strip():
        return "", ""

    denylist = frozenset(
        denied_imports
        if denied_imports is not None
        else {
            "os",
            "sys",
            "subprocess",
            "socket",
            "pathlib",
            "importlib",
            "ctypes",
            "resource",
            "signal",
            "multiprocessing",
            "asyncio",
            "threading",
            "inspect",
        }
    )

    def normalize_output(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    try:
        parsed = ast.parse(code, filename="untrusted.py")
    except SyntaxError as exc:
        message = f"SyntaxError: {exc.msg} (line {exc.lineno}, column {exc.offset})"
        return "", normalize_output(message)

    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name in denylist or any(name.startswith(f"{blocked}.") for blocked in denylist):
                    return "", normalize_output(f"Blocked import: {name}. Execution was not attempted.")
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module is None:
                continue
            if module in denylist or any(module.startswith(f"{blocked}.") for blocked in denylist):
                return "", normalize_output(
                    f"Blocked import: {module}. Execution was not attempted."
                )

    def apply_limits() -> None:
        try:
            import resource

            limit_bytes = 256 * 1024 * 1024
            for limit in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
                try:
                    resource.setrlimit(limit, (limit_bytes, limit_bytes))
                except (ValueError, OSError):
                    pass
            for limit in (resource.RLIMIT_NPROC, resource.RLIMIT_NOFILE):
                try:
                    resource.setrlimit(limit, (1, 1))
                except (ValueError, OSError):
                    pass
        except Exception:
            return

    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": "",
        "PATH": "",
    }

    stdout_text = ""
    stderr_text = ""
    try:
        with tempfile.TemporaryDirectory(prefix="mvp_executor_") as temp_dir:
            code_path = os.path.join(temp_dir, "untrusted.py")
            with open(code_path, "w", encoding="utf-8") as handle:
                handle.write(code)
            try:
                result = subprocess.run(
                    [sys.executable, code_path],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=5,
                    cwd=temp_dir,
                    text=True,
                    errors="replace",
                    env=env,
                    preexec_fn=apply_limits if os.name == "posix" else None,
                )
                stdout_text = result.stdout
                stderr_text = result.stderr
            except subprocess.TimeoutExpired as exc:
                stdout_text = exc.stdout or ""
                stderr_text = exc.stderr or ""
                timeout_message = "Execution timed out after 5s"
                stderr_text = f"{stderr_text}\n{timeout_message}" if stderr_text else timeout_message
            except Exception as exc:
                stderr_text = f"Execution failed: {exc}"
    finally:
        stdout_text = normalize_output(stdout_text)
        stderr_text = normalize_output(stderr_text)

    return stdout_text, stderr_text
