import atexit
import importlib


def _create_radar(tmp_path, monkeypatch):
    """Return isolated radar instance and module for testing."""
    monkeypatch.setattr(atexit, "register", lambda func: func)
    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    monkeypatch.setattr(rr.RelevancyRadar, "_install_import_hook", lambda self: None)
    radar = rr.RelevancyRadar(metrics_file=tmp_path / "metrics.json")
    radar._metrics.clear()
    radar._call_graph.clear()
    return rr, radar


def test_dependency_pruning_respects_core_modules(tmp_path, monkeypatch):
    rr, radar = _create_radar(tmp_path, monkeypatch)

    radar._metrics = {
        "core_mod": {"imports": 0, "executions": 1, "impact": 0.0},
        "auto_mod": {"imports": 0, "executions": 1, "impact": 0.0},
        "orphan_mod": {"imports": 0, "executions": 1, "impact": 0.0},
    }

    radar._call_graph = {
        "menace_master": {"core_mod"},
        "run_autonomous": {"auto_mod"},
    }

    flags = radar.evaluate_final_contribution(
        compress_threshold=2.0, replace_threshold=5.0
    )

    assert flags == {
        "core_mod": "compress",
        "auto_mod": "compress",
        "orphan_mod": "retire",
    }


def test_positive_roi_bypasses_retirement(tmp_path, monkeypatch):
    rr, radar = _create_radar(tmp_path, monkeypatch)

    radar._metrics = {
        "profitable": {"imports": 0, "executions": 0, "impact": 50.0},
        "dormant": {"imports": 0, "executions": 0, "impact": 0.0},
    }

    radar._call_graph = {}

    flags = radar.evaluate_final_contribution(
        compress_threshold=2.0, replace_threshold=5.0
    )

    assert flags == {"dormant": "retire"}
    assert "profitable" not in flags


def test_imported_module_never_invoked_is_retired(tmp_path, monkeypatch):
    rr, radar = _create_radar(tmp_path, monkeypatch)

    radar._metrics = {
        "unused_mod": {"imports": 1, "executions": 0, "impact": 0.0},
    }

    radar._call_graph = {"other": {"unused_mod"}}

    flags = radar.evaluate_final_contribution(
        compress_threshold=2.0, replace_threshold=5.0
    )

    assert flags == {"unused_mod": "retire"}


def test_output_impact_influences_scoring(tmp_path, monkeypatch):
    rr, radar = _create_radar(tmp_path, monkeypatch)

    radar.record_output_impact("hero_mod", 10.0)
    radar.record_output_impact("retire_mod", 0.0)

    radar._call_graph = {}

    flags = radar.evaluate_final_contribution(
        compress_threshold=2.0, replace_threshold=5.0
    )

    assert flags == {"retire_mod": "retire"}
    assert "hero_mod" not in flags
