import atexit
import importlib

import pytest


def test_import_hook_bypasses_bootstrap_critical_modules(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(atexit, "register", lambda func: func)
    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    metrics_file = tmp_path / "relevancy_metrics.json"
    radar = rr.RelevancyRadar(metrics_file=metrics_file)

    import_calls = []

    def fake_original(name, globals=None, locals=None, fromlist=(), level=0):
        import_calls.append(name)
        return object()

    original_import = rr.builtins.__import__
    monkeypatch.setattr(rr.builtins, "_relevancy_radar_original_import", None, raising=False)
    monkeypatch.setattr(rr.builtins, "__import__", fake_original)
    radar._install_import_hook(force=True)
    tracked = rr.builtins.__import__
    rr.builtins.__import__ = original_import

    caplog.set_level("DEBUG", logger="relevancy_radar")
    tracked("sandbox_runner.bootstrap")

    assert import_calls == ["sandbox_runner.bootstrap"]
    assert "sandbox_runner" not in radar._metrics
    assert any(
        "Bypassing relevancy import instrumentation" in rec.getMessage()
        for rec in caplog.records
    )


def test_import_hook_preserves_original_import_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(atexit, "register", lambda func: func)
    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    metrics_file = tmp_path / "relevancy_metrics.json"
    radar = rr.RelevancyRadar(metrics_file=metrics_file)

    expected = ModuleNotFoundError("No module named 'missing_pkg'")

    def fake_original(name, globals=None, locals=None, fromlist=(), level=0):
        raise expected

    original_import = rr.builtins.__import__
    monkeypatch.setattr(rr.builtins, "_relevancy_radar_original_import", None, raising=False)
    monkeypatch.setattr(rr.builtins, "__import__", fake_original)
    radar._install_import_hook(force=True)
    tracked = rr.builtins.__import__
    rr.builtins.__import__ = original_import

    with pytest.raises(ModuleNotFoundError) as exc:
        tracked("missing_pkg")

    assert exc.value is expected
