import importlib
import json
import atexit

import pytest


def test_instance_tracking_and_evaluation(tmp_path, monkeypatch):
    """RelevancyRadar tracks usage events and evaluates module scores."""

    monkeypatch.setattr(atexit, "register", lambda func: func)

    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    metrics_file = tmp_path / "relevancy_metrics.json"
    radar = rr.RelevancyRadar(metrics_file=metrics_file)

    radar.track_usage("alpha", impact=0.0)
    radar.track_usage("alpha", impact=0.0)
    radar.track_usage("beta", impact=5.0)

    # Include a module with no activity to exercise the retire path
    radar._metrics["gamma"] = {"imports": 0, "executions": 0, "impact": 0.0}

    result = radar.evaluate_relevance(
        compress_threshold=1.0, replace_threshold=5.0, impact_weight=1.0
    )
    assert result == {"gamma": "retire", "alpha": "replace"}

    saved = json.loads(metrics_file.read_text())
    assert saved["alpha"]["executions"] == 2
    assert saved["beta"]["executions"] == 1
    assert saved["beta"]["impact"] == 5.0


def test_metrics_persistence_keeps_annotations(tmp_path, monkeypatch):
    """Annotations in the metrics file survive reloads and updates."""

    monkeypatch.setattr(atexit, "register", lambda func: func)

    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    metrics_file = tmp_path / "relevancy_metrics.json"
    metrics_file.write_text(
        json.dumps({"alpha": {"imports": 0, "executions": 0, "annotation": "replace"}})
    )

    radar = rr.RelevancyRadar(metrics_file=metrics_file)
    radar.track_usage("alpha", impact=0.0)

    saved = json.loads(metrics_file.read_text())
    assert saved["alpha"]["annotation"] == "replace"
    assert saved["alpha"]["executions"] == 1


def test_impact_influences_flagging(tmp_path, monkeypatch):
    """Modules with positive impact avoid being flagged when weighted."""

    monkeypatch.setattr(atexit, "register", lambda func: func)

    rr = importlib.reload(importlib.import_module("relevancy_radar"))
    metrics_file = tmp_path / "relevancy_metrics.json"
    radar = rr.RelevancyRadar(metrics_file=metrics_file)

    radar.track_usage("low", impact=0.0)
    radar.track_usage("high", impact=10.0)

    with_impact = radar.evaluate_relevance(
        compress_threshold=2.0, replace_threshold=3.0, impact_weight=1.0
    )
    without_impact = radar.evaluate_relevance(
        compress_threshold=2.0, replace_threshold=3.0, impact_weight=0.0
    )

    assert with_impact == {"low": "compress"}
    assert without_impact == {"low": "compress", "high": "compress"}
