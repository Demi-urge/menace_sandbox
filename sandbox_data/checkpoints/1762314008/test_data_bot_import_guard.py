from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_flat_import_exposes_scope(monkeypatch) -> None:
    module_names = ["data_bot", "menace_sandbox.data_bot"]
    saved_modules = {name: sys.modules[name] for name in module_names if name in sys.modules}
    saved_package = sys.modules.get("menace_sandbox")

    try:
        for name in module_names:
            sys.modules.pop(name, None)

        module = importlib.import_module("data_bot")
        from menace_sandbox.scope_utils import Scope as PackageScope

        assert module.__package__ == "menace_sandbox"
        assert module.Scope is PackageScope
        assert callable(module.build_scope_clause)
        assert callable(module.apply_scope)
        assert sys.modules["data_bot"] is module
        assert sys.modules["menace_sandbox.data_bot"] is module

        pkg_module = sys.modules.get("menace_sandbox")
        assert pkg_module is not None
        pkg_paths = [Path(p).resolve() for p in getattr(pkg_module, "__path__", [])]
        package_root = Path(module.__file__).resolve().parent
        assert package_root in pkg_paths
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
        if saved_package is not None:
            sys.modules["menace_sandbox"] = saved_package
        else:
            sys.modules.pop("menace_sandbox", None)
        sys.modules.update(saved_modules)
