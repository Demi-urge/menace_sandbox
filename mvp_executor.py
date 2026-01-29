from __future__ import annotations

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

    if not code.strip():
        return "", ""

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
