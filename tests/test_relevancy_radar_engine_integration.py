import importlib
import json
import atexit
import types

import pytest


def test_radar_flags_and_engine_response(tmp_path, monkeypatch):
    """Active modules are logged while inactive ones are flagged and surfaced."""

    monkeypatch.setattr(atexit, "register", lambda func: func)

    rr_mod = importlib.reload(importlib.import_module("relevancy_radar"))
    metrics_file = tmp_path / "metrics.json"
    radar = rr_mod.RelevancyRadar(metrics_file=metrics_file)

    radar.track_usage("active_mod")
    radar.track_usage("mid_mod")
    radar.track_usage("mid_mod")
    radar._metrics["inactive_mod"] = {"imports": 0, "executions": 0}

    data = json.loads(metrics_file.read_text())
    assert data["active_mod"]["executions"] == 1

    class DummyEngine:
        def __init__(self, radar):
            self.relevancy_radar = radar
            self.relevancy_flags = {}
            self.event_bus = None
            self.logger = types.SimpleNamespace(
                info=lambda *a, **k: None, exception=lambda *a, **k: None
            )

    engine = DummyEngine(radar)

    def evaluate_module_relevance(self):
        compress_threshold = 1
        replace_threshold = 3
        try:
            flags = self.relevancy_radar.evaluate_relevance(
                compress_threshold, replace_threshold
            )
        except Exception:
            self.logger.exception("relevancy evaluation failed")
            return
        if flags:
            self.relevancy_flags = flags

    evaluate_module_relevance(engine)
    expected = {
        "inactive_mod": "retire",
        "active_mod": "compress",
        "mid_mod": "replace",
    }
    assert engine.relevancy_flags == expected
