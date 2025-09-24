from __future__ import annotations

"""Process-wide :class:`LocalKnowledgeModule` shared across components.

The module favours package-relative imports (``menace_sandbox.*``) but gracefully
falls back to flat imports when run directly.  Both entry points register the
resulting module object in :data:`sys.modules` so that the singleton
``LOCAL_KNOWLEDGE_MODULE`` remains shared regardless of the import style.

It exposes a single :data:`LOCAL_KNOWLEDGE_MODULE` instance which combines
:class:`gpt_memory.GPTMemoryManager` with
:class:`gpt_knowledge_service.GPTKnowledgeService`.  The underlying database
location can be customised through the ``GPT_MEMORY_DB`` environment variable.
"""

import os
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
    from .local_knowledge_module import init_local_knowledge, LocalKnowledgeModule
except ImportError:  # pragma: no cover - fallback for flat layout
    _local_module = _import_with_optional_package("local_knowledge_module")
    init_local_knowledge = _local_module.init_local_knowledge
    LocalKnowledgeModule = _local_module.LocalKnowledgeModule

# Resolve database path from environment or fall back to default location.
_MEM_DB = Path(os.getenv("GPT_MEMORY_DB", "gpt_memory.db"))

# Public singleton instance reused by all modules.
LOCAL_KNOWLEDGE_MODULE: LocalKnowledgeModule = init_local_knowledge(_MEM_DB)

_MODULE = sys.modules[__name__]
sys.modules["menace_sandbox.shared_knowledge_module"] = _MODULE
sys.modules["shared_knowledge_module"] = _MODULE

__all__ = ["LOCAL_KNOWLEDGE_MODULE", "LocalKnowledgeModule"]
