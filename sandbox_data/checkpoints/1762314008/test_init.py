import json
import sys
import types
import importlib
import importlib.util
from pathlib import Path

import pytest
from sandbox_settings import SandboxSettings
from dynamic_path_router import resolve_path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _stub_deps():
    sys.modules.setdefault("quick_fix_engine", types.ModuleType("quick_fix_engine"))
    sr_pkg = types.ModuleType("sandbox_runner")
    sr_pkg.__path__ = []
    sys.modules.setdefault("sandbox_runner", sr_pkg)
    boot = types.ModuleType("sandbox_runner.bootstrap")
    boot.initialize_autonomous_sandbox = lambda *a, **k: None
    sys.modules.setdefault("sandbox_runner.bootstrap", boot)
    sys.modules.setdefault(
        "sandbox_runner.orphan_integration",
        types.ModuleType("sandbox_runner.orphan_integration"),
    )


def test_get_default_synergy_weights_reflects_settings(monkeypatch):
    _stub_deps()
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(resolve_path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    monkeypatch.setenv(
        "DEFAULT_SYNERGY_WEIGHTS",
        json.dumps(
            {
                "roi": 1.0,
                "efficiency": 1.0,
                "resilience": 1.0,
                "antifragility": 1.0,
                "reliability": 1.0,
                "maintainability": 1.0,
                "throughput": 1.0,
            }
        ),
    )
    init_module = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )
    first = init_module.get_default_synergy_weights()

    monkeypatch.setenv(
        "DEFAULT_SYNERGY_WEIGHTS",
        json.dumps(
            {
                "roi": 2.0,
                "efficiency": 2.0,
                "resilience": 2.0,
                "antifragility": 2.0,
                "reliability": 2.0,
                "maintainability": 2.0,
                "throughput": 2.0,
            }
        ),
    )
    second = init_module.get_default_synergy_weights()
    assert first != second
    assert first["roi"] == 1.0 and second["roi"] == 2.0


def test_init_creates_synergy_weights(tmp_path, monkeypatch):
    _stub_deps()
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(resolve_path("self_improvement"))]
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

    init_module = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )

    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: {"quick_fix_engine": "1.0", "sandbox_runner": "1.0", "torch": "2.0"}.get(name, "0"),
    )

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.synergy_weight_file = str(tmp_path / "synergy_weights.json")
    settings.sandbox_central_logging = False

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)
    monkeypatch.setattr(init_module.sys.stdin, "isatty", lambda: True)

    result = init_module.init_self_improvement()

    assert result is settings
    assert init_module.settings is settings
    assert called["settings"] is settings

    synergy_file = Path(settings.synergy_weight_file)
    assert synergy_file.exists()
    assert settings.synergy_weight_file == str(synergy_file)

    data = json.loads(synergy_file.read_text())
    for key, value in init_module.get_default_synergy_weights().items():
        assert data[key] == value
        assert getattr(settings, f"synergy_weight_{key}") == value


def test_init_meta_planning_failure(tmp_path, monkeypatch, caplog):
    _stub_deps()
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(resolve_path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda s: None
    sys.modules["sandbox_runner.bootstrap"] = bootstrap

    meta_stub = types.ModuleType("menace.self_improvement.meta_planning")

    def fail(cfg):  # pragma: no cover - test behaviour
        raise ValueError("boom")

    meta_stub.reload_settings = fail
    sys.modules["menace.self_improvement.meta_planning"] = meta_stub

    init_module = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )

    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: {"quick_fix_engine": "1.0", "sandbox_runner": "1.0", "torch": "2.0"}.get(name, "0"),
    )

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.synergy_weight_file = str(tmp_path / "synergy_weights.json")
    settings.sandbox_central_logging = False

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)
    monkeypatch.setattr(init_module.sys.stdin, "isatty", lambda: True)

    caplog.set_level("ERROR")
    with pytest.raises(RuntimeError) as err:
        init_module.init_self_improvement()

    assert "failed to reload meta_planning settings" in str(err.value)
    record = next(
        r for r in caplog.records if r.message == "failed to reload meta_planning settings"
    )
    assert record.error == "boom"


def test_init_enables_auto_install_when_unattended(tmp_path, monkeypatch):
    _stub_deps()
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules["menace"] = menace_pkg
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(resolve_path("self_improvement"))]
    sys.modules["menace.self_improvement"] = si_pkg

    meta_stub = types.ModuleType("menace.self_improvement.meta_planning")
    meta_stub.reload_settings = lambda cfg: None
    sys.modules["menace.self_improvement.meta_planning"] = meta_stub

    init_module = _load_module(
        "menace.self_improvement.init", resolve_path("self_improvement/init.py")  # path-ignore
    )

    settings = SandboxSettings()
    settings.sandbox_data_dir = str(tmp_path)
    settings.synergy_weight_file = str(tmp_path / "weights.json")
    settings.sandbox_central_logging = False
    settings.auto_install_dependencies = False

    monkeypatch.setattr(init_module, "load_sandbox_settings", lambda: settings)

    called: dict[str, bool] = {}

    def fake_verify(*, auto_install: bool = False) -> None:
        called["auto_install"] = auto_install

    monkeypatch.setattr(init_module, "verify_dependencies", fake_verify)
    monkeypatch.setattr(init_module.sys.stdin, "isatty", lambda: False)

    init_module.init_self_improvement()

    assert called.get("auto_install") is True
