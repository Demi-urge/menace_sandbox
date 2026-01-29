from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Tuple

DENIED_IMPORTS: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "importlib",
        "ctypes",
        "inspect",
        "signal",
        "resource",
        "multiprocessing",
        "asyncio",
        "threading",
    }
)


def _blocked_import_message(module_name: str) -> str:
    """Return a deterministic stderr message for blocked imports."""
    return f"Blocked import: {module_name}. Execution was not attempted."


def _is_denied_module(module_name: str, denied: frozenset[str]) -> bool:
    """Return True when the module is denied or a submodule of a denied module."""
    return any(
        module_name == denied_name or module_name.startswith(f"{denied_name}.")
        for denied_name in denied
    )


def _parse_and_reject_dangerous_imports(code: str) -> Tuple[ast.Module | None, str | None]:
    """Parse code and detect dangerous imports via ast.Import/ast.ImportFrom.

    Returns a tuple of (parsed_module, stderr_message). The stderr message is
    populated for syntax errors or blocked imports, and execution should not
    be attempted when it is present.
    """
    try:
        parsed = ast.parse(code, filename="generated.py")
    except SyntaxError as exc:
        message = f"SyntaxError: {exc.msg} (line {exc.lineno}, column {exc.offset})"
        return None, message

    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_denied_module(alias.name, DENIED_IMPORTS):
                    return None, _blocked_import_message(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if _is_denied_module(node.module, DENIED_IMPORTS):
                return None, _blocked_import_message(node.module)

    return parsed, None


def execute_generated_code(code: str) -> Tuple[str, str]:
    """Execute generated Python code in a constrained subprocess.

    Returns a tuple of (stdout, stderr) with normalized line endings.
    """
    def decode_stream(raw: bytes | bytearray | None) -> str:
        if not raw:
            return ""
        text = raw.decode("utf-8", errors="replace")
        return text.replace("\r\n", "\n").replace("\r", "\n")

    temp_dir = tempfile.mkdtemp(prefix="mvp_executor_")
    stdout_text = ""
    stderr_text = ""
    try:
        code_path = os.path.join(temp_dir, "generated.py")
        with open(code_path, "w", encoding="utf-8") as handle:
            handle.write(code)

        _, parse_error = _parse_and_reject_dangerous_imports(code)
        if parse_error:
            stderr_text = parse_error.replace("\r\n", "\n").replace("\r", "\n")
            return ("", stderr_text)

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
            stdout_text = decode_stream(result.stdout)
            stderr_text = decode_stream(result.stderr)
        except subprocess.TimeoutExpired as exc:
            stdout_text = decode_stream(exc.stdout)
            stderr_text = decode_stream(exc.stderr)
            timeout_message = "Execution timed out after 5 seconds."
            if stderr_text:
                stderr_text = f"{stderr_text}\n{timeout_message}"
            else:
                stderr_text = timeout_message
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
