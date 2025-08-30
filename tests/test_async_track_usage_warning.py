import importlib.util
import types
import sys
import logging
from pathlib import Path


def test_async_track_usage_warns_once(monkeypatch, caplog):
    monkeypatch.delenv("SANDBOX_SUPPRESS_TELEMETRY_WARNING", raising=False)
    pkg = types.ModuleType("sandbox_runner")
    pkg.__path__ = []  # empty so submodules cannot be resolved
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(sys.modules, "relevancy_radar", types.ModuleType("relevancy_radar"))

    path = Path(__file__).resolve().parent.parent / "sandbox_runner" / "meta_logger.py"
    spec = importlib.util.spec_from_file_location("sandbox_runner.meta_logger", path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "sandbox_runner.meta_logger", mod)
    with caplog.at_level(logging.WARNING):
        spec.loader.exec_module(mod)
        mod._async_track_usage("a")
        mod._async_track_usage("b")
    warns = [r for r in caplog.records if "relevancy radar unavailable" in r.message]
    assert len(warns) == 1
