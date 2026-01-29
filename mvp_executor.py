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
    "ctypes",
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
BANNED_IMPORT_PATHS = {"asyncio.subprocess", "concurrent.futures"}
BAN_END_IMPORTS = {"asyncio": {"subprocess"}}
BANNED_CALL_PATHS = {
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    "concurrent.futures.ProcessPoolExecutor",
    "multiprocessing.Pool",
    "multiprocessing.Process",
}


def execute_untrusted(code: str) -> tuple[str, str]:
    """Execute untrusted Python code in a constrained subprocess.

    Enforces a ~256MB memory cap and a 5s timeout. Returns (stdout, stderr)
    as normalized strings with predictable error surfaces.
    """
    if os.name not in {"posix", "nt"}:
        return "", "error: unsupported platform for sandboxed execution"

    def normalize_output(data: bytes) -> str:
        text = data.decode("utf-8", errors="replace")
        text = text.replace("\x00", "")
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def error_message(message: str) -> tuple[str, str]:
        return "", f"error: {message}"

    def check_static_policy(tree: ast.AST) -> list[str]:
        violations: set[str] = set()
        banned_lookup_names = BANNED_BUILTINS | BANNED_MODULES | {"__builtins__", "builtins"}
        builtins_aliases: set[str] = set()
        importlib_aliases: set[str] = set()
        import_aliases: set[str] = set()

        def is_builtins_target(node: ast.AST) -> bool:
            if isinstance(node, ast.Name):
                return node.id in {"__builtins__", "builtins"} or node.id in builtins_aliases
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

        def dotted_name(node: ast.AST) -> Optional[str]:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                prefix = dotted_name(node.value)
                if prefix:
                    return f"{prefix}.{node.attr}"
            return None

        def register_alias(target: ast.AST, value: ast.AST) -> None:
            if not isinstance(target, ast.Name):
                return
            if is_module_reference(value, {"builtins"} | builtins_aliases):
                builtins_aliases.add(target.id)
                return
            if is_module_reference(value, {"importlib"} | importlib_aliases):
                importlib_aliases.add(target.id)
                return
            if isinstance(value, ast.Name) and value.id in {"__import__"} | import_aliases:
                import_aliases.add(target.id)
                return
            if isinstance(value, ast.Attribute) and value.attr == "__import__":
                if is_builtins_target(value.value) or is_module_reference(value.value, importlib_aliases):
                    import_aliases.add(target.id)

        def handle_import(node: ast.Import) -> None:
            for alias in node.names:
                module_name = alias.name.split(".", 1)[0]
                if alias.name in BANNED_IMPORT_PATHS:
                    violations.add(f"import of '{alias.name}' is not allowed")
                    continue
                if module_name in BANNED_MODULES:
                    violations.add(f"import of '{module_name}' is not allowed")

        def handle_import_from(node: ast.ImportFrom) -> None:
            if node.module:
                module_name = node.module.split(".", 1)[0]
                if node.module in BANNED_IMPORT_PATHS:
                    violations.add(f"import of '{node.module}' is not allowed")
                for alias in node.names:
                    banned_names = BAN_END_IMPORTS.get(node.module)
                    if banned_names and alias.name in banned_names:
                        violations.add(f"import of '{node.module}.{alias.name}' is not allowed")
                if module_name in BANNED_MODULES:
                    violations.add(f"import of '{module_name}' is not allowed")

        def handle_call(node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
                violations.add(f"call to '{node.func.id}' is not allowed")
            if isinstance(node.func, ast.Name) and node.func.id in import_aliases:
                violations.add("call to '__import__' is not allowed")
            if isinstance(node.func, ast.Attribute):
                func_name = dotted_name(node.func)
                if func_name in BANNED_CALL_PATHS:
                    violations.add(f"call to '{func_name}' is not allowed")
                if node.func.attr == "__import__" and (
                    is_builtins_target(node.func.value) or is_module_reference(node.func.value, importlib_aliases)
                ):
                    violations.add("call to '__import__' is not allowed")
            if isinstance(node.func, ast.Name) and node.func.id == "getattr":
                if len(node.args) >= 2 and is_builtins_target(node.args[0]) and is_banned_lookup(node.args[1]):
                    violations.add("call to 'getattr' with builtins is not allowed")
                if len(node.args) >= 2 and string_literal(node.args[1]) in BANNED_BUILTINS:
                    violations.add("call to 'getattr' with banned builtin name is not allowed")
            if isinstance(node.func, ast.Name) and node.func.id == "vars":
                if node.args and is_builtins_target(node.args[0]):
                    violations.add("call to 'vars' with builtins is not allowed")
            if isinstance(node.func, ast.Name) and node.func.id in {"globals", "locals"}:
                if node.args and any(is_builtins_target(arg) for arg in node.args):
                    violations.add(f"call to '{node.func.id}' with builtins is not allowed")

        def handle_name(node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load) and node.id in BANNED_BUILTINS:
                violations.add(f"use of '{node.id}' is not allowed")
            if isinstance(node.ctx, ast.Load) and node.id in {"__builtins__", "builtins"}:
                violations.add(f"use of '{node.id}' is not allowed")
            if isinstance(node.ctx, ast.Load) and node.id in builtins_aliases:
                violations.add(f"use of builtins alias '{node.id}' is not allowed")

        def handle_attribute(node: ast.Attribute) -> None:
            if is_builtins_target(node):
                violations.add("access to builtins is not allowed")

        def handle_subscript(node: ast.Subscript) -> None:
            if is_builtins_target(node):
                violations.add("access to builtins is not allowed")
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id in {"globals", "locals", "vars"} and is_banned_lookup(node.slice):
                    violations.add(f"call to '{node.value.func.id}' with builtins is not allowed")

        class PolicyVisitor(ast.NodeVisitor):
            def visit_Import(self, node: ast.Import) -> None:
                handle_import(node)
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                handle_import_from(node)
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                handle_call(node)
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                handle_name(node)

            def visit_Attribute(self, node: ast.Attribute) -> None:
                handle_attribute(node)
                self.generic_visit(node)

            def visit_Subscript(self, node: ast.Subscript) -> None:
                handle_subscript(node)
                self.generic_visit(node)

            def visit_Assign(self, node: ast.Assign) -> None:
                if node.value is not None:
                    self.visit(node.value)
                for target in node.targets:
                    register_alias(target, node.value)
                    self.visit(target)

            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                if node.value is not None:
                    self.visit(node.value)
                if node.annotation is not None:
                    self.visit(node.annotation)
                if node.target is not None and node.value is not None:
                    register_alias(node.target, node.value)
                if node.target is not None:
                    self.visit(node.target)

        PolicyVisitor().visit(tree)
        return sorted(violations)

    # POSIX uses resource-based rlimits in preexec_limits; Windows uses Job Objects.
    def preexec_limits() -> Optional[Callable[[], None]]:
        if os.name != "posix":
            return None

        def apply_limits() -> None:
            import resource

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

    def run_windows_subprocess(
        args: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess[bytes]:
        import ctypes
        from ctypes import wintypes

        JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x100
        JobObjectExtendedLimitInformation = 9

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.INT,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        def apply_windows_job_object(process_handle: int) -> int:
            job_handle = kernel32.CreateJobObjectW(None, None)
            if not job_handle:
                raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.ProcessMemoryLimit = _MEMORY_LIMIT_MB * 1024 * 1024
            if not kernel32.SetInformationJobObject(
                job_handle,
                JobObjectExtendedLimitInformation,
                ctypes.byref(info),
                ctypes.sizeof(info),
            ):
                error = ctypes.get_last_error()
                kernel32.CloseHandle(job_handle)
                raise OSError(error, "SetInformationJobObject failed")

            if not kernel32.AssignProcessToJobObject(job_handle, wintypes.HANDLE(process_handle)):
                error = ctypes.get_last_error()
                kernel32.CloseHandle(job_handle)
                raise OSError(error, "AssignProcessToJobObject failed")

            return job_handle

        def resume_windows_process(process_id: int) -> None:
            TH32CS_SNAPTHREAD = 0x00000004
            THREAD_SUSPEND_RESUME = 0x0002
            INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

            class THREADENTRY32(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wintypes.DWORD),
                    ("cntUsage", wintypes.DWORD),
                    ("th32ThreadID", wintypes.DWORD),
                    ("th32OwnerProcessID", wintypes.DWORD),
                    ("tpBasePri", wintypes.LONG),
                    ("tpDeltaPri", wintypes.LONG),
                    ("dwFlags", wintypes.DWORD),
                ]

            kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
            kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
            kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
            kernel32.Thread32First.restype = wintypes.BOOL
            kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(THREADENTRY32)]
            kernel32.Thread32Next.restype = wintypes.BOOL
            kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenThread.restype = wintypes.HANDLE
            kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
            kernel32.ResumeThread.restype = wintypes.DWORD
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL

            snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
            if snapshot == INVALID_HANDLE_VALUE:
                raise OSError(ctypes.get_last_error(), "CreateToolhelp32Snapshot failed")

            entry = THREADENTRY32()
            entry.dwSize = ctypes.sizeof(THREADENTRY32)
            success = kernel32.Thread32First(snapshot, ctypes.byref(entry))
            while success:
                if entry.th32OwnerProcessID == process_id:
                    thread_handle = kernel32.OpenThread(
                        THREAD_SUSPEND_RESUME,
                        False,
                        entry.th32ThreadID,
                    )
                    if not thread_handle:
                        kernel32.CloseHandle(snapshot)
                        raise OSError(ctypes.get_last_error(), "OpenThread failed")
                    resume_result = kernel32.ResumeThread(thread_handle)
                    kernel32.CloseHandle(thread_handle)
                    kernel32.CloseHandle(snapshot)
                    if resume_result == 0xFFFFFFFF:
                        raise OSError(ctypes.get_last_error(), "ResumeThread failed")
                    return
                success = kernel32.Thread32Next(snapshot, ctypes.byref(entry))

            kernel32.CloseHandle(snapshot)
            raise OSError(ctypes.get_last_error(), "Unable to locate primary thread")

        creationflags = getattr(subprocess, "CREATE_SUSPENDED", 0x00000004)
        proc = subprocess.Popen(
            args,
            shell=False,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            creationflags=creationflags,
        )
        job_handle: Optional[int] = None
        try:
            job_handle = apply_windows_job_object(proc._handle)
            resume_windows_process(proc.pid)
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            proc.kill()
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(
                exc.cmd,
                exc.timeout,
                stdout=stdout,
                stderr=stderr,
            ) from exc
        except Exception:
            proc.kill()
            proc.communicate()
            raise
        finally:
            if job_handle is not None:
                kernel32.CloseHandle(wintypes.HANDLE(job_handle))

        return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)

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

    violations = check_static_policy(tree)
    if violations:
        return "", "error: " + "; ".join(violations)

    temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    try:
        temp_dir = tempfile.TemporaryDirectory(prefix="mvp_executor_")
        temp_path = temp_dir.name
        code_path = os.path.join(temp_path, "untrusted.py")
        runner_path = os.path.join(temp_path, "runner.py")
        with open(code_path, "w", encoding="utf-8") as handle:
            handle.write(code)
        with open(runner_path, "w", encoding="utf-8") as handle:
            handle.write(
                (
                    "import builtins\n"
                    "import runpy\n"
                    "import sys\n"
                    "\n"
                    f"ALLOWED_ROOT = {temp_path!r}\n"
                    f"BANNED_MODULES = {sorted(BANNED_MODULES)!r}\n"
                    "PATH_SEP = '\\\\' if sys.platform.startswith('win') else '/'\n"
                    "ALT_SEP = '/' if PATH_SEP == '\\\\' else None\n"
                    "\n"
                    "def _error(message):\n"
                    "    sys.stderr.write(f\"error: {message}\\n\")\n"
                    "    raise SystemExit(1)\n"
                    "\n"
                    "def _coerce_path(value):\n"
                    "    if isinstance(value, str):\n"
                    "        return value\n"
                    "    if isinstance(value, bytes):\n"
                    "        return value.decode(sys.getfilesystemencoding(), errors='surrogateescape')\n"
                    "    fspath = getattr(value, '__fspath__', None)\n"
                    "    if fspath is None:\n"
                    "        return None\n"
                    "    return _coerce_path(fspath())\n"
                    "\n"
                    "def _is_absolute(path):\n"
                    "    if path.startswith(PATH_SEP):\n"
                    "        return True\n"
                    "    if len(path) > 2 and path[1] == ':':\n"
                    "        return path[2] in (PATH_SEP, ALT_SEP or PATH_SEP)\n"
                    "    return False\n"
                    "\n"
                    "def _normalize(path):\n"
                    "    if ALT_SEP:\n"
                    "        path = path.replace(ALT_SEP, PATH_SEP)\n"
                    "    drive = ''\n"
                    "    absolute = path.startswith(PATH_SEP)\n"
                    "    if len(path) > 1 and path[1] == ':':\n"
                    "        drive = path[:2]\n"
                    "        path = path[2:]\n"
                    "        absolute = path.startswith(PATH_SEP)\n"
                    "        if drive and not absolute:\n"
                    "            return None\n"
                    "    parts = []\n"
                    "    for part in path.split(PATH_SEP):\n"
                    "        if part in ('', '.'): \n"
                    "            continue\n"
                    "        if part == '..':\n"
                    "            if parts:\n"
                    "                parts.pop()\n"
                    "            else:\n"
                    "                return None\n"
                    "            continue\n"
                    "        parts.append(part)\n"
                    "    prefix = drive + (PATH_SEP if absolute else '')\n"
                    "    if not parts:\n"
                    "        if prefix:\n"
                    "            return prefix.rstrip(PATH_SEP) if not absolute else prefix\n"
                    "        return PATH_SEP if absolute else '.'\n"
                    "    return prefix + PATH_SEP.join(parts)\n"
                    "\n"
                    "def _resolve(path):\n"
                    "    if path is None:\n"
                    "        return None\n"
                    "    if _is_absolute(path):\n"
                    "        return _normalize(path)\n"
                    "    base = _normalize(ALLOWED_ROOT)\n"
                    "    if base is None:\n"
                    "        return None\n"
                    "    combined = base.rstrip(PATH_SEP) + PATH_SEP + path.lstrip(PATH_SEP)\n"
                    "    return _normalize(combined)\n"
                    "\n"
                    "def is_allowed(path):\n"
                    "    resolved = _resolve(path)\n"
                    "    if resolved is None:\n"
                    "        return False\n"
                    "    allowed_root = _normalize(ALLOWED_ROOT)\n"
                    "    if allowed_root is None:\n"
                    "        return False\n"
                    "    if resolved == allowed_root:\n"
                    "        return True\n"
                    "    return resolved.startswith(allowed_root.rstrip(PATH_SEP) + PATH_SEP)\n"
                    "\n"
                    "class ImportGate:\n"
                    "    def find_spec(self, fullname, path=None, target=None):\n"
                    "        root = fullname.split('.', 1)[0]\n"
                    "        if root in BANNED_MODULES or fullname in BANNED_MODULES:\n"
                    "            raise ImportError(f\"import of '{fullname}' is not allowed\")\n"
                    "        return None\n"
                    "\n"
                    "def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
                    "    if level:\n"
                    "        raise ImportError(\"relative imports are not allowed\")\n"
                    "    root = name.split('.', 1)[0]\n"
                    "    if root in BANNED_MODULES or name in BANNED_MODULES:\n"
                    "        raise ImportError(f\"import of '{name}' is not allowed\")\n"
                    "    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)\n"
                    "\n"
                    "def _guarded_open(file, *args, **kwargs):\n"
                    "    target = _coerce_path(file)\n"
                    "    if target is not None and not is_allowed(target):\n"
                    "        raise PermissionError(f\"Access to '{target}' is not allowed\")\n"
                    "    return _ORIGINAL_OPEN(file, *args, **kwargs)\n"
                    "\n"
                    "_ORIGINAL_OPEN = builtins.open\n"
                    "_ORIGINAL_IMPORT = builtins.__import__\n"
                    "builtins.open = _guarded_open\n"
                    "sys.meta_path.insert(0, ImportGate())\n"
                    "builtins.__import__ = _guarded_import\n"
                    "\n"
                    "if len(sys.argv) < 2:\n"
                    "    _error('Missing untrusted script path')\n"
                    "untrusted_path = sys.argv[1]\n"
                    "sys.argv = sys.argv[1:]\n"
                    "try:\n"
                    "    runpy.run_path(untrusted_path, run_name='__main__')\n"
                    "except (ImportError, PermissionError) as exc:\n"
                    "    _error(str(exc))\n"
                    "except SystemExit as exc:\n"
                    "    raise\n"
                    "except Exception as exc:\n"
                    "    _error(str(exc))\n"
                )
            )

        env = {
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "",
        }

        try:
            if os.name == "nt":
                result = run_windows_subprocess(
                    [sys.executable, runner_path, code_path],
                    cwd=temp_path,
                    env=env,
                    timeout=_TIMEOUT_SECONDS,
                )
            else:
                result = subprocess.run(
                    [sys.executable, runner_path, code_path],
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
