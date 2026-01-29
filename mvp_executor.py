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
    "builtins",
    "codecs",
    "concurrent",
    "fnmatch",
    "glob",
    "http",
    "importlib",
    "io",
    "multiprocessing",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tarfile",
    "tempfile",
    "urllib",
    "zipfile",
}
BANNED_BUILTINS = {"__import__", "compile", "eval", "exec", "input"}
ALLOWED_IMPORTS = {
    "dataclasses",
    "functools",
    "json",
    "math",
    "random",
    "re",
    "statistics",
    "string",
    "time",
    "typing",
}


def _check_static_policy(tree: ast.AST) -> list[str]:
    violations: set[str] = set()
    banned_lookup_names = BANNED_BUILTINS | BANNED_MODULES | {"__builtins__", "builtins"}

    def is_builtins_target(node: ast.AST) -> bool:
        if isinstance(node, ast.Name):
            return node.id in {"__builtins__", "builtins"}
        if isinstance(node, ast.Attribute):
            return node.attr in {"__builtins__", "builtins"} or is_builtins_target(node.value)
        if isinstance(node, ast.Subscript):
            return is_builtins_target(node.value) or is_builtins_target(node.slice)
        if isinstance(node, ast.Constant):
            return node.value in {"__builtins__", "builtins"}
        if isinstance(node, ast.Index):  # pragma: no cover - py<3.9 compatibility
            return is_builtins_target(node.value)
        return False

    def is_module_reference(node: ast.AST, names: set[str]) -> bool:
        if isinstance(node, ast.Name):
            return node.id in names
        if isinstance(node, ast.Attribute):
            return node.attr in names or is_module_reference(node.value, names)
        if isinstance(node, ast.Subscript):
            return is_module_reference(node.value, names) or is_module_reference(node.slice, names)
        if isinstance(node, ast.Index):  # pragma: no cover - py<3.9 compatibility
            return is_module_reference(node.value, names)
        return False

    def string_literal(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Index):  # pragma: no cover - py<3.9 compatibility
            return string_literal(node.value)
        return None

    def is_banned_lookup(node: ast.AST) -> bool:
        literal = string_literal(node)
        return literal in banned_lookup_names if literal is not None else False

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
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                if len(node.args) >= 2 and is_builtins_target(node.args[0]) and is_banned_lookup(node.args[1]):
                    violations.add("call to 'getattr' with builtins is not allowed")
            if isinstance(node.func, ast.Name) and node.func.id == "vars":
                if node.args and is_builtins_target(node.args[0]):
                    violations.add("call to 'vars' with builtins is not allowed")
            if isinstance(node.func, ast.Name) and node.func.id in {"globals", "locals"}:
                if node.args and any(is_builtins_target(arg) for arg in node.args):
                    violations.add(f"call to '{node.func.id}' with builtins is not allowed")
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id in BANNED_BUILTINS:
                violations.add(f"use of '{node.id}' is not allowed")
            if isinstance(node.ctx, ast.Load) and node.id in {"__builtins__", "builtins"}:
                violations.add(f"use of '{node.id}' is not allowed")
        elif isinstance(node, ast.Attribute):
            if is_builtins_target(node):
                violations.add("access to builtins is not allowed")
        elif isinstance(node, ast.Subscript):
            if is_builtins_target(node):
                violations.add("access to builtins is not allowed")
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id in {"globals", "locals", "vars"} and is_banned_lookup(node.slice):
                    violations.add(f"call to '{node.value.func.id}' with builtins is not allowed")
    return sorted(violations)


def execute_untrusted(code: str) -> tuple[str, str]:
    """Execute untrusted Python code in a constrained subprocess.

    Returns (stdout, stderr) as normalized strings with predictable error surfaces.
    """
    if os.name != "posix":
        return "", "error: unsupported platform for sandboxed execution"

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
        runner_path = os.path.join(temp_path, "runner.py")
        bootstrap_path = os.path.join(temp_path, "bootstrap.py")
        with open(code_path, "w", encoding="utf-8") as handle:
            handle.write(code)
        with open(runner_path, "w", encoding="utf-8") as handle:
            handle.write(
                (
                    "import builtins\n"
                    "import io\n"
                    "import os\n"
                    "import runpy\n"
                    "import sys\n"
                    "\n"
                    f"ALLOWED_ROOT = {temp_path!r}\n"
                    "\n"
                    "def is_allowed(path):\n"
                    "    try:\n"
                    "        resolved = os.path.realpath(path)\n"
                    "    except Exception:\n"
                    "        return False\n"
                    "    allowed_root = os.path.realpath(ALLOWED_ROOT)\n"
                    "    if resolved == allowed_root:\n"
                    "        return True\n"
                    "    return resolved.startswith(allowed_root + os.sep)\n"
                    "\n"
                    "def _guarded_open(file, *args, **kwargs):\n"
                    "    if isinstance(file, (str, bytes, os.PathLike)):\n"
                    "        target = os.fsdecode(os.fspath(file))\n"
                    "        if not is_allowed(target):\n"
                    "            raise PermissionError(f\"Access to '{target}' is not allowed\")\n"
                    "    return _ORIGINAL_OPEN(file, *args, **kwargs)\n"
                    "\n"
                    "_ORIGINAL_OPEN = builtins.open\n"
                    "builtins.open = _guarded_open\n"
                    "io.open = _guarded_open\n"
                    "\n"
                    "if len(sys.argv) < 2:\n"
                    "    raise SystemExit('Missing untrusted script path')\n"
                    "untrusted_path = sys.argv[1]\n"
                    "sys.argv = sys.argv[1:]\n"
                    "runpy.run_path(untrusted_path, run_name='__main__')\n"
                )
            )
        with open(bootstrap_path, "w", encoding="utf-8") as handle:
            handle.write(
                (
                    "import builtins\n"
                    "import runpy\n"
                    "import sys\n"
                    "\n"
                    "# Allowed imports must be explicitly listed for auditability.\n"
                    f"ALLOWED_IMPORTS = {sorted(ALLOWED_IMPORTS)!r}\n"
                    "# Banned prefixes are rejected even if they look allowed.\n"
                    "BANNED_PREFIXES = (\n"
                    "    'builtins',\n"
                    "    'ctypes',\n"
                    "    'importlib',\n"
                    "    'io',\n"
                    "    'codecs',\n"
                    "    'concurrent',\n"
                    "    'fnmatch',\n"
                    "    'glob',\n"
                    "    'multiprocessing',\n"
                    "    'os',\n"
                    "    'pathlib',\n"
                    "    'socket',\n"
                    "    'subprocess',\n"
                    "    'tarfile',\n"
                    "    'tempfile',\n"
                    "    'zipfile',\n"
                    ")\n"
                    "\n"
                    "def _is_allowed(module_name: str) -> bool:\n"
                    "    if not module_name:\n"
                    "        return False\n"
                    "    root = module_name.split('.', 1)[0]\n"
                    "    for prefix in BANNED_PREFIXES:\n"
                    "        if root == prefix or module_name.startswith(prefix + '.'):\n"
                    "            return False\n"
                    "    return root in ALLOWED_IMPORTS\n"
                    "\n"
                    "class ImportGate:\n"
                    "    def find_spec(self, fullname, path=None, target=None):\n"
                    "        if not _is_allowed(fullname):\n"
                    "            raise ImportError(f\"import of '{fullname}' is not allowed\")\n"
                    "        return None\n"
                    "\n"
                    "def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
                    "    if level:\n"
                    "        raise ImportError(\"relative imports are not allowed\")\n"
                    "    if not _is_allowed(name):\n"
                    "        raise ImportError(f\"import of '{name}' is not allowed\")\n"
                    "    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)\n"
                    "\n"
                    "_ORIGINAL_IMPORT = builtins.__import__\n"
                    "sys.meta_path.insert(0, ImportGate())\n"
                    "builtins.__import__ = _guarded_import\n"
                    "\n"
                    "if len(sys.argv) < 2:\n"
                    "    raise SystemExit('Missing untrusted script path')\n"
                    "sys.argv = sys.argv[1:]\n"
                    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
                )
            )

        env = {
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
        }

        try:
            result = subprocess.run(
                [sys.executable, bootstrap_path, runner_path, code_path],
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
