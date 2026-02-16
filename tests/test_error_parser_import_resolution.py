from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_module(module_name: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "error_parser.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_error_parser_prefers_package_relative_imports(monkeypatch):
    menace_pkg = ModuleType("menace")
    menace_pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "menace", menace_pkg)

    packaged_dynamic_path_router = ModuleType("menace.dynamic_path_router")
    packaged_dynamic_path_router.resolve_path = lambda value: f"pkg:{value}"
    monkeypatch.setitem(sys.modules, "menace.dynamic_path_router", packaged_dynamic_path_router)

    flat_dynamic_path_router = ModuleType("dynamic_path_router")
    flat_dynamic_path_router.resolve_path = lambda value: f"flat:{value}"
    monkeypatch.setitem(sys.modules, "dynamic_path_router", flat_dynamic_path_router)

    self_improvement_pkg = ModuleType("menace.self_improvement")
    self_improvement_pkg.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "menace.self_improvement", self_improvement_pkg)

    target_region_module = ModuleType("menace.self_improvement.target_region")
    target_region_module.TargetRegion = SimpleNamespace
    target_region_module.extract_target_region = lambda trace: f"pkg-region:{trace}"
    monkeypatch.setitem(sys.modules, "menace.self_improvement.target_region", target_region_module)

    module = _load_module("menace.error_parser")

    assert module.resolve_path("x") == "pkg:x"
    assert module.extract_target_region("trace") == "pkg-region:trace"


def test_error_parser_flat_layout_uses_absolute_fallbacks(monkeypatch):
    flat_dynamic_path_router = ModuleType("dynamic_path_router")
    flat_dynamic_path_router.resolve_path = lambda value: f"flat:{value}"
    monkeypatch.setitem(sys.modules, "dynamic_path_router", flat_dynamic_path_router)

    target_region_module = ModuleType("self_improvement.target_region")
    target_region_module.TargetRegion = SimpleNamespace
    target_region_module.extract_target_region = lambda trace: f"flat-region:{trace}"

    import_compat = ModuleType("import_compat")
    import_compat.load_internal = lambda dotted: target_region_module
    monkeypatch.setitem(sys.modules, "import_compat", import_compat)

    module = _load_module("error_parser_flat_layout")

    assert module.resolve_path("x") == "flat:x"
    assert module.extract_target_region("trace") == "flat-region:trace"
