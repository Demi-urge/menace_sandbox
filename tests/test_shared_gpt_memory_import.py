from __future__ import annotations

import importlib
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import sys
from types import ModuleType
from typing import Dict, Iterable


def _load_flat_shared_gpt_memory(module_path: Path) -> ModuleType:
    loader = SourceFileLoader("shared_gpt_memory", str(module_path))
    spec = spec_from_loader("shared_gpt_memory", loader)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError("Unable to create module specification for shared_gpt_memory")

    module = module_from_spec(spec)
    assert module is not None  # for type checkers
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


def _clear_modules(names: Iterable[str]) -> None:
    for name in names:
        sys.modules.pop(name, None)


def test_shared_gpt_memory_import_paths() -> None:
    module_path = Path(__file__).resolve().parents[1] / "shared_gpt_memory.py"
    tracked_modules = [
        "shared_gpt_memory",
        "menace_sandbox.shared_gpt_memory",
        "menace_sandbox",
        "gpt_memory",
        "menace_sandbox.gpt_memory",
        "shared_knowledge_module",
        "menace_sandbox.shared_knowledge_module",
    ]

    saved_modules: Dict[str, ModuleType | None] = {
        name: sys.modules.get(name) for name in tracked_modules
    }

    try:
        _clear_modules(tracked_modules)

        flat_module = _load_flat_shared_gpt_memory(module_path)
        manager_from_flat = flat_module.GPT_MEMORY_MANAGER
        assert flat_module.__package__ == "menace_sandbox"
        assert manager_from_flat is flat_module.GPT_MEMORY_MANAGER
        assert (
            sys.modules["shared_gpt_memory"].GPT_MEMORY_MANAGER is manager_from_flat
        )
        assert (
            sys.modules["menace_sandbox.shared_gpt_memory"].GPT_MEMORY_MANAGER
            is manager_from_flat
        )

        package_alias = importlib.import_module("menace_sandbox.shared_gpt_memory")
        assert package_alias.GPT_MEMORY_MANAGER is manager_from_flat

        _clear_modules(["shared_gpt_memory", "menace_sandbox.shared_gpt_memory"])

        package_module = importlib.import_module("menace_sandbox.shared_gpt_memory")
        manager_from_package = package_module.GPT_MEMORY_MANAGER
        flat_alias = importlib.import_module("shared_gpt_memory")
        assert flat_alias is package_module
        assert flat_alias.GPT_MEMORY_MANAGER is manager_from_package
    finally:
        _clear_modules(["shared_gpt_memory", "menace_sandbox.shared_gpt_memory"])
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
