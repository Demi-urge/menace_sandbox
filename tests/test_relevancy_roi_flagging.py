import atexit
import importlib
import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


def test_roi_values_influence_flagging(tmp_path, monkeypatch):
    """Modules with positive ROI avoid being flagged."""

    monkeypatch.setattr(atexit, "register", lambda func: func)
    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    metrics_file = tmp_path / "relevancy_metrics.json"
    rr._DEFAULT_RADAR = rr.RelevancyRadar(metrics_file=metrics_file)

    rr.track_usage("high_roi", impact=5.0)
    rr.track_usage("low_roi", impact=0.0)

    flags = rr.evaluate_final_contribution(
        compress_threshold=1.0, replace_threshold=1.0, impact_weight=1.0, core_modules=[]
    )

    assert "high_roi" not in flags
    assert flags.get("low_roi") == "retire"
