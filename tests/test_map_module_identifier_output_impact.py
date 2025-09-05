import atexit
import importlib
import types
import sys
from unittest.mock import MagicMock


def _make_radar(tmp_path, monkeypatch):
    monkeypatch.setattr(atexit, "register", lambda func: func)
    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    monkeypatch.setattr(rr.RelevancyRadar, "_install_import_hook", lambda self: None)
    radar = rr.RelevancyRadar(metrics_file=tmp_path / "metrics.json")
    radar._metrics.clear()
    radar._call_graph.clear()
    return rr, radar


def test_map_module_identifier_records_output_impact(tmp_path, monkeypatch):
    rr, radar = _make_radar(tmp_path, monkeypatch)

    # Use isolated radar instance for module level helpers
    monkeypatch.setattr(rr, "_get_default_radar", lambda: radar)
    monkeypatch.setattr(rr, "radar", radar)

    record_mock = MagicMock(side_effect=radar.record_output_impact)
    monkeypatch.setattr(radar, "record_output_impact", record_mock)

    monkeypatch.setenv("SANDBOX_ENABLE_RELEVANCY_RADAR", "1")

    dummy_adaptive = types.ModuleType("adaptive_roi_predictor")
    dummy_adaptive.load_training_data = lambda *a, **k: None
    sys.modules["adaptive_roi_predictor"] = dummy_adaptive

    dummy_env = types.ModuleType("sandbox_runner.environment")
    dummy_env.SANDBOX_ENV_PRESETS = {}
    dummy_env.auto_include_modules = lambda *a, **k: None
    dummy_env.record_error = lambda *a, **k: None
    dummy_env.ERROR_CATEGORY_COUNTS = {}
    sys.modules["sandbox_runner.environment"] = dummy_env

    cycle = importlib.reload(importlib.import_module("sandbox_runner.cycle"))

    class DummyThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self):  # pragma: no cover - trivial
            if self.target:
                self.target()

    monkeypatch.setattr(cycle.threading, "Thread", DummyThread)

    mod1 = tmp_path / "hero_mod.py"  # path-ignore
    mod1.write_text("", encoding="utf-8")
    mod2 = tmp_path / "retire_mod.py"  # path-ignore
    mod2.write_text("", encoding="utf-8")

    cycle.map_module_identifier(str(mod1), tmp_path, 10.0)
    cycle.map_module_identifier(str(mod2), tmp_path)

    assert record_mock.call_count == 2
    assert record_mock.call_args_list[0][0] == ("hero_mod", 10.0)
    assert record_mock.call_args_list[1][0] == ("retire_mod", 0.0)

