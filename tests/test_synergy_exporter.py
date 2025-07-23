import importlib
import json
import time
from pathlib import Path

import pytest


def test_exporter_updates_and_stops(tmp_path, monkeypatch):
    se = importlib.import_module("menace.synergy_exporter")

    # Dummy Gauge collecting set values
    class DummyGauge:
        def __init__(self, name, doc):
            self.name = name
            self.doc = doc
            self.values = []

        def set(self, value):
            self.values.append(value)

    monkeypatch.setattr(se, "Gauge", DummyGauge)
    monkeypatch.setattr(se, "start_metrics_server", lambda port: None)

    hist_file = tmp_path / "synergy_history.json"
    hist_file.write_text(json.dumps([{"synergy_roi": 0.1}]))

    exp = se.SynergyExporter(history_file=hist_file, interval=0.05, port=1234)
    exp.start()
    try:
        # Wait for first update
        for _ in range(50):
            g = exp._gauges.get("synergy_roi")
            if g and g.values:
                break
            time.sleep(0.02)
        assert g is not None and g.values[-1] == 0.1

        # Update file with new value and wait for export
        hist_file.write_text(json.dumps([{"synergy_roi": 0.1}, {"synergy_roi": 0.2}]))
        for _ in range(50):
            if g.values and g.values[-1] == 0.2:
                break
            time.sleep(0.02)
        assert g.values[-1] == 0.2
    finally:
        exp.stop()

    assert exp._thread is not None
    assert not exp._thread.is_alive()
