from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


MODULE_PATH = Path(__file__).resolve().parents[1] / "menace" / "self_coding_manager.py"


def _load_shim(module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_self_coding_manager_prefers_repo_package_fallback(monkeypatch) -> None:
    package_module = ModuleType("menace_sandbox.self_coding_manager")
    package_module.EXPORT_TOKEN = "package"
    package_module.__all__ = ["EXPORT_TOKEN"]
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_manager", package_module)

    flat_module = ModuleType("self_coding_manager")
    flat_module.EXPORT_TOKEN = "flat"
    flat_module.__all__ = ["EXPORT_TOKEN"]
    monkeypatch.setitem(sys.modules, "self_coding_manager", flat_module)

    loaded = _load_shim("menace.self_coding_manager")

    assert loaded.EXPORT_TOKEN == "package"
