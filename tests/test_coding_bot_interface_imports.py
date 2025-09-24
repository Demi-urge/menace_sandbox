"""Ensure ``coding_bot_interface`` imports work in both flat and packaged modes."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest


_DEF_MODULES = (
    "coding_bot_interface",
    "menace_sandbox.coding_bot_interface",
)


@pytest.fixture(autouse=True)
def _stub_self_coding_modules() -> None:
    """Provide lightweight stand-ins for self-coding dependencies."""

    installed: dict[str, ModuleType | None] = {}

    def install(name: str, module: ModuleType) -> None:
        installed[name] = sys.modules.get(name)
        sys.modules[name] = module

    thresholds = ModuleType("self_coding_thresholds")
    thresholds.update_thresholds = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    thresholds._load_config = lambda *args, **kwargs: {}  # type: ignore[attr-defined]
    for key in ("self_coding_thresholds", "menace_sandbox.self_coding_thresholds"):
        install(key, thresholds)

    manager = ModuleType("self_coding_manager")

    class _StubManager:
        def __init__(self) -> None:
            self.engine = SimpleNamespace(
                generate_helper=lambda *args, **kwargs: "",
                context_builder=None,
            )
            self.bot_registry = SimpleNamespace(register_bot=lambda *args, **kwargs: None)
            self.data_bot = SimpleNamespace(reload_thresholds=lambda name: None)
            self.evolution_orchestrator = None

    manager.SelfCodingManager = _StubManager  # type: ignore[attr-defined]
    for key in ("self_coding_manager", "menace_sandbox.self_coding_manager"):
        install(key, manager)

    engine = ModuleType("self_coding_engine")

    class _ManagerContext:
        def set(self, value: object) -> object:
            return object()

        def reset(self, token: object) -> None:  # pragma: no cover - simple stub
            return None

    engine.MANAGER_CONTEXT = _ManagerContext()  # type: ignore[attr-defined]
    for key in ("self_coding_engine", "menace_sandbox.self_coding_engine"):
        install(key, engine)

    try:
        yield
    finally:
        for name, original in installed.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _clear_module_state() -> None:
    """Remove cached modules and package attributes for repeatable imports."""

    for name in _DEF_MODULES:
        sys.modules.pop(name, None)
    package = sys.modules.get("menace_sandbox")
    if isinstance(package, ModuleType):
        package_dict = getattr(package, "__dict__", {})
        if "coding_bot_interface" in package_dict:
            del package_dict["coding_bot_interface"]


def _assert_helpers_available(module: ModuleType) -> None:
    """Validate that the helper API is available without bootstrap errors."""

    assert module.__package__ == "menace_sandbox"
    assert hasattr(module, "self_coding_managed")
    assert hasattr(module, "manager_generate_helper")


def test_flat_import_bootstraps_relative_imports() -> None:
    """Importing the module flat should engage the bootstrap shims."""

    _clear_module_state()
    module = importlib.import_module("coding_bot_interface")
    _assert_helpers_available(module)


def test_package_import_remains_supported() -> None:
    """Package-style imports should continue to work for compatibility."""

    _clear_module_state()
    module = __import__("menace_sandbox", fromlist=["coding_bot_interface"]).coding_bot_interface
    _assert_helpers_available(module)
