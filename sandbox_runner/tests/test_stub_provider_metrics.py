import importlib.util
from pathlib import Path
import sys
import types
import pytest
from dynamic_path_router import resolve_path

MODULE_DIR = resolve_path("sandbox_runner")
ROOT_DIR = resolve_path("")
sys.path.append(str(ROOT_DIR))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stub_providers = _load("stub_providers", MODULE_DIR / ("stub_providers" + ".py"))
metrics_plugins = _load("metrics_plugins", MODULE_DIR / ("metrics_plugins" + ".py"))
menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = []
sys.modules["menace"] = menace_pkg
si_pkg = types.ModuleType("menace.self_improvement")
si_pkg.__path__ = [str(ROOT_DIR / "self_improvement")]
sys.modules["menace.self_improvement"] = si_pkg
sys.modules["menace.sandbox_settings"] = _load(
    "menace.sandbox_settings", ROOT_DIR / ("sandbox_settings" + ".py")
)
logger = types.SimpleNamespace(
    info=lambda *a, **k: None,
    warning=lambda *a, **k: None,
    exception=lambda *a, **k: None,
    debug=lambda *a, **k: None,
    error=lambda *a, **k: None,
)
sys.modules["menace.logging_utils"] = types.SimpleNamespace(
    get_logger=lambda name: logger,
    setup_logging=lambda: None,
)
si_metrics = _load(
    "menace.self_improvement.metrics",
    ROOT_DIR / "self_improvement" / ("metrics" + ".py"),
)


def test_discover_stub_providers_success_and_failure(monkeypatch):
    """Providers are returned only when load succeeds."""
    def good(inputs, ctx):
        return inputs

    class EP:
        def __init__(self, name):
            self.name = name

    monkeypatch.setattr(
        stub_providers,
        "_iter_entry_points",
        lambda: [EP("good"), EP("bad")],
    )
    monkeypatch.setattr(
        stub_providers, "_load_entry_point", lambda ep: good if ep.name == "good" else None
    )
    settings = types.SimpleNamespace(stub_providers=[], disabled_stub_providers=[])
    providers = stub_providers.discover_stub_providers(settings=settings)
    assert providers == [good]

    monkeypatch.setattr(stub_providers, "_load_entry_point", lambda ep: None)
    with pytest.raises(RuntimeError):
        stub_providers.discover_stub_providers(settings=settings)


def test_load_metrics_plugins_and_errors(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / ("ok" + ".py")).write_text(
        "def collect_metrics(prev, cur, res):\n    return {'a': 1}\n"
    )
    (plugin_dir / ("nofunc" + ".py")).write_text("x = 1\n")
    (plugin_dir / ("bad" + ".py")).write_text("raise RuntimeError('boom')\n")
    missing = tmp_path / "missing"
    plugins = metrics_plugins.load_metrics_plugins([plugin_dir, missing])
    assert len(plugins) == 1
    assert plugins[0](0, 0, None) == {"a": 1}


def test_collect_plugin_metrics_handles_errors():
    def good(prev, cur, res):
        return {"x": 1}

    def bad(prev, cur, res):
        raise RuntimeError("fail")

    merged = metrics_plugins.collect_plugin_metrics([good, bad], 0.0, 0.0, None)
    assert merged == {"x": 1}


def test_update_and_get_alignment_baseline(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ("a" + ".py")).write_text("def f():\n    return 1\n")
    baseline = tmp_path / "baseline.yaml"
    fake_settings = types.SimpleNamespace(
        alignment_baseline_metrics_path=baseline,
        sandbox_repo_path=str(repo),
        metrics_skip_dirs=[],
    )
    monkeypatch.setattr(si_metrics, "SandboxSettings", lambda: fake_settings)
    data = si_metrics._update_alignment_baseline(settings=fake_settings)
    assert baseline.exists()
    assert ("a" + ".py") in data["files"]
    loaded = si_metrics.get_alignment_metrics(settings=fake_settings)
    assert loaded == data
    missing_settings = types.SimpleNamespace(
        alignment_baseline_metrics_path=tmp_path / "nope.yaml"
    )
    assert si_metrics.get_alignment_metrics(settings=missing_settings) == {}
