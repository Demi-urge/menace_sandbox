import atexit
import importlib
from unittest.mock import MagicMock


def _make_radar(tmp_path, monkeypatch):
    monkeypatch.setattr(atexit, "register", lambda func: func)
    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    monkeypatch.setattr(rr.RelevancyRadar, "_install_import_hook", lambda self: None)
    radar = rr.RelevancyRadar(metrics_file=tmp_path / "metrics.json")
    radar._metrics.clear()
    radar._call_graph.clear()
    return rr, radar


def test_async_tracking_records_output_impact(tmp_path, monkeypatch):
    rr, radar = _make_radar(tmp_path, monkeypatch)

    # Use isolated radar instance for module level helpers
    monkeypatch.setattr(rr, "_get_default_radar", lambda: radar)

    record_mock = MagicMock(side_effect=radar.record_output_impact)
    monkeypatch.setattr(rr, "record_output_impact", record_mock)

    import types, sys
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

    def fake_async(module, impact=None):
        if impact is not None:
            if impact > 0:
                cycle._radar_track_usage(module, impact)
            rr.record_output_impact(module, impact)

    monkeypatch.setattr(cycle, "_async_track_usage", fake_async)

    cycle._async_track_usage("hero_mod", 10.0)
    cycle._async_track_usage("retire_mod", 0.0)

    flags = radar.evaluate_final_contribution(
        compress_threshold=2.0, replace_threshold=5.0
    )

    assert flags == {"retire_mod": "retire"}
    assert "hero_mod" not in flags
    assert record_mock.call_count == 2
