import importlib.util
import json
import sys
import types
from pathlib import Path

from dynamic_path_router import resolve_path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))


def _load_init():
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = []
    sys.modules.setdefault("menace", menace_pkg)
    si_pkg = types.ModuleType("menace.self_improvement")
    si_pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    sys.modules.setdefault("menace.self_improvement", si_pkg)

    bootstrap = types.ModuleType("sandbox_runner.bootstrap")
    bootstrap.initialize_autonomous_sandbox = lambda *a, **k: None
    sr_pkg = types.ModuleType("sandbox_runner")
    sys.modules.setdefault("sandbox_runner", sr_pkg)
    sys.modules.setdefault("sandbox_runner.bootstrap", bootstrap)

    spec = importlib.util.spec_from_file_location(
        "menace.self_improvement.init",
        resolve_path("self_improvement/init.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["menace.self_improvement.init"] = module
    spec.loader.exec_module(module)
    return module


init = _load_init()


def _defaults():
    return {
        "roi": 1.0,
        "efficiency": 1.0,
        "resilience": 1.0,
        "antifragility": 1.0,
        "reliability": 1.0,
        "maintainability": 1.0,
        "throughput": 1.0,
    }


def _make_settings(tmp_path, content):
    path = tmp_path / "synergy_weights.json"
    path.write_text(json.dumps(content))
    return types.SimpleNamespace(
        sandbox_data_dir=str(tmp_path), synergy_weight_file=str(path)
    )


def test_missing_entry(monkeypatch, tmp_path):
    monkeypatch.setattr(init, "get_default_synergy_weights", _defaults)
    settings = _make_settings(tmp_path, {"roi": 1.0})
    monkeypatch.setattr(init, "settings", settings)
    with pytest.raises(ValueError, match="missing synergy weight"):
        init._load_initial_synergy_weights()


def test_malformed_entry(monkeypatch, tmp_path):
    data = _defaults()
    data["efficiency"] = "high"
    monkeypatch.setattr(init, "get_default_synergy_weights", _defaults)
    settings = _make_settings(tmp_path, data)
    monkeypatch.setattr(init, "settings", settings)
    with pytest.raises(ValueError, match="efficiency"):
        init._load_initial_synergy_weights()
