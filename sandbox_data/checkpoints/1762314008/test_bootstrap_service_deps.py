import importlib
import sys
import types
from pathlib import Path

import pytest

# Stub out heavy modules before importing bootstrap
menace = types.ModuleType("menace")
menace.RAISE_ERRORS = False
menace.__path__ = [str(Path(__file__).resolve().parents[1])]
auto_env_setup = types.ModuleType("menace.auto_env_setup")
auto_env_setup.ensure_env = lambda _path: None
menace.auto_env_setup = auto_env_setup
# expose config helpers for bootstrap imports
_dcm = importlib.import_module("menace_sandbox.default_config_manager")
_cd = importlib.import_module("menace_sandbox.config_discovery")
menace.default_config_manager = _dcm
menace.config_discovery = _cd
sys.modules["menace"] = menace
sys.modules["menace.auto_env_setup"] = auto_env_setup
sys.modules["menace.default_config_manager"] = _dcm
sys.modules["menace.config_discovery"] = _cd

sandbox_settings = types.ModuleType("sandbox_settings")

# Stub predictor dependency pulled in by cycle module
adp = types.ModuleType("adaptive_roi_predictor")
adp.load_training_data = lambda *a, **k: []
sys.modules.setdefault("adaptive_roi_predictor", adp)

# Minimal environment module required by cycle imports
env_stub = types.ModuleType("sandbox_runner.environment")
env_stub.record_error = lambda *a, **k: None
env_stub.SANDBOX_ENV_PRESETS = [{}]
env_stub.run_scenarios = lambda *a, **k: None
env_stub.ERROR_CATEGORY_COUNTS = {}
env_stub.auto_include_modules = lambda *a, **k: []
sys.modules["sandbox_runner.environment"] = env_stub

# Minimal self_improvement API to avoid heavy imports
si_api = types.ModuleType("self_improvement.api")
si_api.init_self_improvement = lambda settings: None


class _DummyThread:
    def __init__(self):
        self._alive = True

    def start(self):
        self._alive = True

    def join(self, timeout=None):
        if timeout is None:
            self._alive = False

    def is_alive(self):
        return self._alive


si_api.start_self_improvement_cycle = lambda *a, **k: _DummyThread()
si_api.stop_self_improvement_cycle = lambda: None
sys.modules["self_improvement.api"] = si_api
si_pkg = types.ModuleType("self_improvement")
si_pkg.api = si_api
sys.modules["self_improvement"] = si_pkg


class SandboxSettings:
    def __init__(self, sandbox_data_dir="", alignment_baseline_metrics_path=""):
        self.sandbox_data_dir = sandbox_data_dir
        self.alignment_baseline_metrics_path = alignment_baseline_metrics_path
        self.menace_mode = "test"
        self.menace_env_file = "env"
        self.menace_light_imports = False
        self.sandbox_required_db_files = []
        self.optional_service_versions = {
            "relevancy_radar": "1.0.0",
            "quick_fix_engine": "1.0.0",
        }
        self.patch_retries = 3
        self.patch_retry_delay = 0.1
        self.usage_queue_maxsize = 100
        self.required_env_vars = [
            "OPENAI_API_KEY",
            "DATABASE_URL",
            "MODELS",
        ]


sandbox_settings.SandboxSettings = SandboxSettings
sandbox_settings.load_sandbox_settings = lambda: SandboxSettings()
sys.modules["sandbox_settings"] = sandbox_settings

sr_cli = types.ModuleType("sandbox_runner.cli")
sr_cli.main = lambda *_a, **_k: None
sys.modules["sandbox_runner.cli"] = sr_cli

bootstrap = importlib.import_module("sandbox_runner.bootstrap")


def _stub_services(monkeypatch, versions, missing=frozenset()):
    original_import = importlib.import_module

    def fake_import(name, package=None):
        if name in missing:
            raise ModuleNotFoundError(name)
        if name in versions:
            mod = types.ModuleType(name)
            mod.__version__ = versions[name]
            if name == "relevancy_radar":
                mod.RelevancyRadar = type("RelevancyRadar", (), {})
            monkeypatch.setitem(sys.modules, name, mod)
            return mod
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)


def _settings(tmp_path):
    return SandboxSettings(sandbox_data_dir=str(tmp_path))


def test_missing_service_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("MODELS", str(tmp_path))
    _stub_services(monkeypatch, {"quick_fix_engine": "1.0.0"}, missing={"relevancy_radar"})
    caplog.set_level("WARNING")
    bootstrap.initialize_autonomous_sandbox(_settings(tmp_path))
    assert any("relevancy_radar" in r.getMessage() for r in caplog.records)
    bootstrap.shutdown_autonomous_sandbox()


def test_missing_quick_fix_engine_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("MODELS", str(tmp_path))
    _stub_services(monkeypatch, {"relevancy_radar": "1.0.0"}, missing={"quick_fix_engine"})
    caplog.set_level("WARNING")
    bootstrap.initialize_autonomous_sandbox(_settings(tmp_path))
    assert any("quick_fix_engine" in r.getMessage() for r in caplog.records)
    bootstrap.shutdown_autonomous_sandbox()


def test_version_too_old_warns(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("MODELS", str(tmp_path))
    _stub_services(monkeypatch, {"quick_fix_engine": "1.0.0", "relevancy_radar": "0.0.1"})
    caplog.set_level("WARNING")
    bootstrap.initialize_autonomous_sandbox(_settings(tmp_path))
    assert any("too old" in r.getMessage() for r in caplog.records)
    bootstrap.shutdown_autonomous_sandbox()


def test_unwritable_data_dir(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("MODELS", str(tmp_path))
    _stub_services(monkeypatch, {"quick_fix_engine": "1.0.0", "relevancy_radar": "1.0.0"})
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(
        bootstrap.Path, "touch", lambda *a, **k: (_ for _ in ()).throw(PermissionError())
    )
    settings = SandboxSettings(sandbox_data_dir=str(data_dir))
    with pytest.raises(RuntimeError, match="not writable"):
        bootstrap.initialize_autonomous_sandbox(settings)
    assert any("not writable" in r.getMessage() for r in caplog.records)
    bootstrap.shutdown_autonomous_sandbox()
