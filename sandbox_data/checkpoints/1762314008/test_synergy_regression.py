import importlib
import json
import os
import sys
import types

import pytest


def _stub_roi_tracker(monkeypatch):
    mod = types.ModuleType("menace.roi_tracker")

    class DummyTracker:
        def __init__(self, *a, **k):
            self.metrics_history = {}
            self.synergy_metrics_history = {}

        def load_history(self, path: str) -> None:
            if not os.path.exists(path):
                return
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.metrics_history = data.get("metrics_history", {})
            self.synergy_metrics_history = data.get("synergy_metrics_history", {})

    mod.ROITracker = DummyTracker
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", mod)


def _write(path, metrics):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"synergy_metrics_history": metrics}, fh)


def _load_env():
    return importlib.import_module("sandbox_runner.environment")


def test_synergy_regression(tmp_path, monkeypatch):
    _stub_roi_tracker(monkeypatch)
    env = _load_env()

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write(
        a,
        {
            "synergy_roi": [0.5, -0.2, 0.7],
            "synergy_security_score": [0.1, 0.2],
        },
    )
    _write(b, {"synergy_roi": [0.2, -0.1], "synergy_security_score": [0.4]})

    res = env.aggregate_synergy_metrics([str(a), str(b)])
    assert res == [("a", pytest.approx(1.0)), ("b", pytest.approx(0.1))]

    res_sec = env.aggregate_synergy_metrics([str(a), str(b)], metric="security_score")
    assert res_sec == [("b", pytest.approx(0.4)), ("a", pytest.approx(0.3))]
