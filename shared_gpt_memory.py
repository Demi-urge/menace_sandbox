from __future__ import annotations

"""Shared GPT memory manager instance for all ChatGPT clients.

This module prefers package-relative imports (``menace_sandbox.*``) so it can
be used as part of the installed package, but it transparently falls back to
flat imports when executed as a standalone script (``python shared_gpt_memory.py``).
In that situation :mod:`importlib` is used to resolve dependencies and the
loaded module is registered under both the package and flat names so every
import path shares the same singleton state.
"""

from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType


def _ensure_package_alias() -> None:
    """Ensure ``menace_sandbox`` is registered in :mod:`sys.modules`."""

    if "menace_sandbox" in sys.modules:
        return

    package_root = Path(__file__).resolve().parent
    init_path = package_root / "__init__.py"
    spec = spec_from_file_location("menace_sandbox", init_path)
    if spec and spec.loader:
        module = module_from_spec(spec)
        module.__path__ = [str(package_root)]
        sys.modules["menace_sandbox"] = module
        spec.loader.exec_module(module)
    else:  # pragma: no cover - fallback when loader unavailable
        module = ModuleType("menace_sandbox")
        module.__path__ = [str(package_root)]
        module.__file__ = str(init_path)
        sys.modules["menace_sandbox"] = module


def _import_with_optional_package(module: str) -> ModuleType:
    """Import *module* preferring the ``menace_sandbox`` package."""

    qualified = f"menace_sandbox.{module}"
    try:
        return import_module(qualified)
    except ModuleNotFoundError as exc:
        if exc.name not in {"menace_sandbox", qualified}:
            raise

    _ensure_package_alias()

    try:
        return import_module(qualified)
    except ModuleNotFoundError as exc:
        if exc.name != qualified:
            raise

    module_obj = import_module(module)
    sys.modules.setdefault(qualified, module_obj)
    return module_obj


try:  # pragma: no cover - exercised via dedicated import test
    from .gpt_memory import GPTMemoryManager
except ImportError:  # pragma: no cover - fallback for flat layout
    GPTMemoryManager = _import_with_optional_package("gpt_memory").GPTMemoryManager

try:  # pragma: no cover - exercised via dedicated import test
    from .shared_knowledge_module import LOCAL_KNOWLEDGE_MODULE
except ImportError:  # pragma: no cover - fallback for flat layout
    LOCAL_KNOWLEDGE_MODULE = _import_with_optional_package(
        "shared_knowledge_module"
    ).LOCAL_KNOWLEDGE_MODULE

# Single global GPT memory instance reused across bots
GPT_MEMORY_MANAGER: GPTMemoryManager = LOCAL_KNOWLEDGE_MODULE.memory

_MODULE = sys.modules[__name__]
sys.modules["menace_sandbox.shared_gpt_memory"] = _MODULE
sys.modules["shared_gpt_memory"] = _MODULE

__all__ = ["GPT_MEMORY_MANAGER"]
