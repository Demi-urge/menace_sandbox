import json
import sys
import types
import importlib.util
from pathlib import Path

import pytest
from sandbox_settings import SandboxSettings


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def test_init_creates_synergy_weights(tmp_path, monkeypatch):
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(Path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    called = {}

    def fake_init(s):
        called["settings"] = s

    bootstrap.initialize_autonomous_sandbox = fake_init
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

    meta_stub = types.ModuleType("menace.self_improvement.meta_planning")
    meta_stub.reload_settings = lambda cfg: None
    sys.modules["menace.self_improvement.meta_planning"] = meta_stub

    init_module = _load_module("menace.self_improvement.init", Path("self_improvement/init.py"))

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.synergy_weight_file = str(tmp_path / "synergy_weights.json")
    settings.sandbox_central_logging = False

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)

    result = init_module.init_self_improvement()

    assert result is settings
    assert init_module.settings is settings
    assert called["settings"] is settings

    synergy_file = Path(settings.synergy_weight_file)
    assert synergy_file.exists()
    assert settings.synergy_weight_file == str(synergy_file)

    data = json.loads(synergy_file.read_text())
    for key, value in init_module.DEFAULT_SYNERGY_WEIGHTS.items():
        assert data[key] == value
        assert getattr(settings, f"synergy_weight_{key}") == value


def test_init_meta_planning_failure(tmp_path, monkeypatch, caplog):
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(Path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda s: None
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

    meta_stub = types.ModuleType("menace.self_improvement.meta_planning")

    def fail(cfg):  # pragma: no cover - test behaviour
        raise ValueError("boom")

    meta_stub.reload_settings = fail
    sys.modules["menace.self_improvement.meta_planning"] = meta_stub

    init_module = _load_module("menace.self_improvement.init", Path("self_improvement/init.py"))

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.synergy_weight_file = str(tmp_path / "synergy_weights.json")
    settings.sandbox_central_logging = False

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)

    caplog.set_level("ERROR")
    with pytest.raises(RuntimeError) as err:
        init_module.init_self_improvement()

    assert "failed to reload meta_planning settings" in str(err.value)
    record = next(
        r for r in caplog.records if r.message == "failed to reload meta_planning settings"
    )
    assert record.error == "boom"
