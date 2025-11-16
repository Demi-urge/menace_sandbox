from __future__ import annotations

"""Process-wide :class:`LocalKnowledgeModule` shared across components."""

import importlib.util
import os
import sys
from pathlib import Path

_HELPER_NAME = "import_compat"
_PACKAGE_NAME = "menace_sandbox"

try:  # pragma: no cover - prefer package import when installed
    from menace_sandbox import import_compat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - support flat execution
    _helper_path = Path(__file__).resolve().parent / f"{_HELPER_NAME}.py"
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{_HELPER_NAME}",
        _helper_path,
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise
    import_compat = importlib.util.module_from_spec(_spec)
    sys.modules[f"{_PACKAGE_NAME}.{_HELPER_NAME}"] = import_compat
    sys.modules[_HELPER_NAME] = import_compat
    _spec.loader.exec_module(import_compat)
else:  # pragma: no cover - ensure helper aliases exist
    sys.modules.setdefault(_HELPER_NAME, import_compat)
    sys.modules.setdefault(f"{_PACKAGE_NAME}.{_HELPER_NAME}", import_compat)

import_compat.bootstrap(__name__, __file__)
load_internal = import_compat.load_internal

_local_module = load_internal("local_knowledge_module")
init_local_knowledge = _local_module.init_local_knowledge
LocalKnowledgeModule = _local_module.LocalKnowledgeModule

# Resolve database path from environment or fall back to default location.
_MEM_DB = Path(os.getenv("GPT_MEMORY_DB", "gpt_memory.db"))

# Public singleton instance reused by all modules.
LOCAL_KNOWLEDGE_MODULE: LocalKnowledgeModule = init_local_knowledge(_MEM_DB)

__all__ = ["LOCAL_KNOWLEDGE_MODULE", "LocalKnowledgeModule"]
