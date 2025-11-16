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
    blocked_imports = {
        "relevancy_radar",
        "menace_sandbox.relevancy_radar",
        "quick_fix_engine",
        "menace_sandbox.quick_fix_engine",
    }

    def fake_import(name: str, package: str | None = None):
        if name in blocked_imports:
            raise AssertionError(f"unexpected import of optional module {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    bootstrap = importlib.import_module("sandbox_runner.bootstrap")

    optional_modules = ("relevancy_radar", "quick_fix_engine")
    baseline_modules = set(sys.modules)
    try:
        missing = bootstrap._verify_optional_modules(optional_modules, {})
        assert missing == set()
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


def test_initialize_autonomous_sandbox_prints_step_e_without_quick_fix(monkeypatch, tmp_path, capsys):
    import sandbox_runner.bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "auto_configure_env", lambda settings: None)
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_ensure_sqlite_db", lambda path: None)
    monkeypatch.setattr(bootstrap, "_start_optional_services", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "resolve_path", lambda value: Path(value).resolve())
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)

    class _DummyRegistry:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def mark_missing(self, **info):  # type: ignore[no-untyped-def]
            self.calls.append(("missing", info))

        def mark_available(self, **info):  # type: ignore[no-untyped-def]
            self.calls.append(("available", info))

        def summary(self) -> dict[str, object]:
            return {}

    registry = _DummyRegistry()
    monkeypatch.setattr(bootstrap, "dependency_registry", registry)

    bootstrap._OPTIONAL_MODULE_CACHE.clear()
    bootstrap._MISSING_OPTIONAL.clear()
    bootstrap._OPTIONAL_DEPENDENCY_WARNED.clear()

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if "quick_fix_engine" in name:
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    original_import_optional = bootstrap._import_optional_module

    def fail_quick_fix_import(name: str, *args, **kwargs):
        if "quick_fix_engine" in name:
            raise AssertionError("quick_fix_engine import attempted during verification")
        return original_import_optional(name, *args, **kwargs)

    monkeypatch.setattr(bootstrap, "_import_optional_module", fail_quick_fix_import)

    data_dir = tmp_path / "sandbox-data"
    settings = types.SimpleNamespace(
        sandbox_data_dir=str(data_dir),
        sandbox_required_db_files=(),
        optional_service_versions={"quick_fix_engine": "1.0"},
        alignment_baseline_metrics_path="",
        menace_env_file=str(tmp_path / ".env"),
    )

    bootstrap._initialize_autonomous_sandbox(
        settings,
        start_services=False,
        start_self_improvement=False,
    )

    out = capsys.readouterr().out
    assert "🧬 D: verifying optional modules" in out
    assert "🧬 E: preparing sandbox data directory" in out
    assert any(call[0] == "missing" and call[1].get("name") == "quick_fix_engine" for call in registry.calls)
    assert data_dir.exists()
