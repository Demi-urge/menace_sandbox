from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def test_verify_optional_modules_prefers_repo_modules(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(repo_root))

    sys.modules.pop("dynamic_path_router", None)
    importlib.import_module("dynamic_path_router")

    restored: dict[str, object] = {}
    for name in list(sys.modules):
        if name == "menace_sandbox" or name.startswith("menace_sandbox."):
            restored[name] = sys.modules.pop(name)

    stubbed: dict[str, object | None] = {}

    def install_stub(name: str, module: types.ModuleType) -> None:
        if name not in stubbed:
            stubbed[name] = sys.modules.get(name)
        sys.modules[name] = module

    counter = types.SimpleNamespace(labels=lambda *a, **k: counter, inc=lambda *a, **k: None)
    metrics_stub = types.ModuleType("metrics_exporter")
    metrics_stub.sandbox_restart_total = counter
    metrics_stub.environment_failure_total = counter
    metrics_stub.sandbox_crashes_total = counter
    install_stub("metrics_exporter", metrics_stub)
    install_stub("sandbox_runner.metrics_exporter", metrics_stub)
    install_stub("menace_sandbox.metrics_exporter", metrics_stub)

    auto_env_stub = types.ModuleType("menace.auto_env_setup")
    auto_env_stub.ensure_env = lambda *_a, **_k: None
    install_stub("menace.auto_env_setup", auto_env_stub)

    class _DummyDefaultConfigManager:
        def __init__(self, *_a, **_k) -> None:
            pass

        def apply_defaults(self) -> None:
            return None

    default_stub = types.ModuleType("menace.default_config_manager")
    default_stub.DefaultConfigManager = _DummyDefaultConfigManager
    install_stub("menace.default_config_manager", default_stub)

    cycle_stub = types.ModuleType("sandbox_runner.cycle")
    cycle_stub.ensure_vector_service = lambda: None
    install_stub("sandbox_runner.cycle", cycle_stub)

    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.main = lambda *_a, **_k: None
    install_stub("sandbox_runner.cli", cli_stub)

    class _SandboxSettings:
        optional_service_versions: dict[str, str] = {}
        sandbox_data_dir = "sandbox_data"
        sandbox_required_db_files: tuple[str, ...] = ()

    def _load_sandbox_settings() -> _SandboxSettings:
        return _SandboxSettings()

    settings_stub = types.ModuleType("sandbox_settings")
    settings_stub.SandboxSettings = _SandboxSettings
    settings_stub.load_sandbox_settings = _load_sandbox_settings
    install_stub("sandbox_settings", settings_stub)

    package_stub = types.ModuleType("menace_sandbox")
    package_stub.__path__ = [str(repo_root)]
    package_stub.RAISE_ERRORS = False
    install_stub("menace_sandbox", package_stub)

    real_import_module = importlib.import_module
    module_cache: dict[str, types.ModuleType] = {}
    module_paths = {
        "relevancy_radar": repo_root / "relevancy_radar.py",
        "quick_fix_engine": repo_root / "quick_fix_engine.py",
    }
    alias_map = {
        name: base
        for base in module_paths
        for name in (base, f"menace_sandbox.{base}")
    }

    def fake_import(name: str, package: str | None = None):
        target = alias_map.get(name)
        if target:
            module = module_cache.get(target)
            if module is None:
                module = types.ModuleType(f"menace_sandbox.{target}")
                module.__file__ = str(module_paths[target])
                module.__dict__["__version__"] = "0.0"
                module_cache[target] = module
            sys.modules[name] = module
            return module
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    bootstrap = importlib.import_module("sandbox_runner.bootstrap")

    optional_modules = ("relevancy_radar", "quick_fix_engine")
    baseline_modules = set(sys.modules)
    try:
        missing = bootstrap._verify_optional_modules(optional_modules, {})
        assert missing == set()
        for base, module in module_cache.items():
            assert sys.modules.get(base) is module
            assert sys.modules.get(f"menace_sandbox.{base}") is module
    finally:
        new_modules = set(sys.modules) - baseline_modules
        for module_name in new_modules:
            sys.modules.pop(module_name, None)
        for name, module in stubbed.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.update(restored)
