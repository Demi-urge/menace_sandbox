"""Package-level compatibility shim for ``SelfDebuggerSandbox``."""

from __future__ import annotations

from importlib import import_module
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_root_module_from_path() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "self_debugger_sandbox.py"
    if not module_path.exists():
        raise ImportError(
            "Fallback import failed: expected root module at "
            f"'{module_path}', but the file does not exist."
        )

    spec = importlib.util.spec_from_file_location("self_debugger_sandbox", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            "Fallback import failed: could not build an import spec for "
            f"'{module_path}'."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_self_debugger_sandbox() -> type:
    try:
        module = import_module("self_debugger_sandbox")
    except Exception:
        module = _load_root_module_from_path()

    cls = getattr(module, "SelfDebuggerSandbox", None)
    if cls is None:
        module_source = getattr(module, "__file__", "<unknown>")
        raise ImportError(
            "Imported module does not define 'SelfDebuggerSandbox'. "
            "Ensure self_debugger_sandbox.py exports that class. "
            f"Loaded module source: {module_source}."
        )
    return cls


SelfDebuggerSandbox = _resolve_self_debugger_sandbox()

__all__ = ["SelfDebuggerSandbox"]
