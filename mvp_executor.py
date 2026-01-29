from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Tuple


def execute_generated_code(code: str) -> Tuple[str, str]:
    """Execute generated Python code in a constrained subprocess.

    Returns a tuple of (stdout, stderr) with normalized line endings.
    """
    temp_dir = tempfile.mkdtemp(prefix="mvp_executor_")
    stdout_text = ""
    stderr_text = ""
    try:
        code_path = os.path.join(temp_dir, "generated.py")
        with open(code_path, "w", encoding="utf-8") as handle:
            handle.write(code)

        try:
            parsed = ast.parse(code, filename="generated.py")
        except SyntaxError as exc:
            stderr_text = (
                f"SyntaxError: {exc.msg} (line {exc.lineno}, column {exc.offset})"
            )
            return ("", stderr_text.replace("\r\n", "\n").replace("\r", "\n"))

        disallowed = {
            "os",
            "sys",
            "subprocess",
            "socket",
            "pathlib",
            "importlib",
            "ctypes",
            "resource",
            "multiprocessing",
            "signal",
            "inspect",
            "builtins",
            "threading",
            "asyncio",
            "selectors",
        }
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_name = alias.name.split(".", 1)[0]
                    if root_name in disallowed:
                        stderr_text = f"Disallowed import detected: {root_name}"
                        return ("", stderr_text.replace("\r\n", "\n").replace("\r", "\n"))
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                root_name = node.module.split(".", 1)[0]
                if root_name in disallowed:
                    stderr_text = f"Disallowed import detected: {root_name}"
                    return ("", stderr_text.replace("\r\n", "\n").replace("\r", "\n"))

        def apply_limits() -> None:
            try:
                import resource

                limit_bytes = 256 * 1024 * 1024
                for limit in (resource.RLIMIT_AS, resource.RLIMIT_DATA):
                    try:
                        resource.setrlimit(limit, (limit_bytes, limit_bytes))
                    except (ValueError, OSError):
                        pass
                try:
                    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
                except (ValueError, OSError):
                    pass
                try:
                    resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
                except (ValueError, OSError):
                    pass
            except Exception:
                pass

        env = {"PYTHONIOENCODING": "utf-8"}
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", code_path],
                cwd=temp_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                timeout=5,
                preexec_fn=apply_limits if os.name == "posix" else None,
                start_new_session=True,
                shell=False,
            )
            stdout_text = result.stdout.decode("utf-8", errors="replace")
            stderr_text = result.stderr.decode("utf-8", errors="replace")
        except subprocess.TimeoutExpired as exc:
            stdout_text = (
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, (bytes, bytearray))
                else ""
            )
            stderr_text = (
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, (bytes, bytearray))
                else ""
            )
            stderr_text = (stderr_text + "\nExecution timed out after 5 seconds.").strip()
        except PermissionError as exc:
            stderr_text = f"PermissionError: {exc}"
        except OSError as exc:
            stderr_text = f"OSError: {exc}"
        except Exception as exc:
            stderr_text = f"Execution failed: {exc}"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    stdout_text = stdout_text.replace("\r\n", "\n").replace("\r", "\n")
    stderr_text = stderr_text.replace("\r\n", "\n").replace("\r", "\n")
    return stdout_text, stderr_text
