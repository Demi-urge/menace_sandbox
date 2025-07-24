import importlib.util
import sys
import types
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg

def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"menace.{name}", ROOT / f"{name}.py", submodule_search_locations=[str(ROOT)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"menace.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

sd_mod = _load("sandbox_dashboard")
SandboxDashboard = sd_mod.SandboxDashboard
ROITracker = _load("roi_tracker").ROITracker


def _make_history(path):
    tracker = ROITracker()
    tracker.update(0.0, 1.0, modules=["m"], metrics={"security_score": 0.8})
    tracker.update(1.0, 2.0, modules=["m"], metrics={"security_score": 0.9})
    tracker.save_history(str(path))


def test_roi_route(tmp_path):
    history = tmp_path / "hist.json"
    _make_history(history)
    dash = SandboxDashboard(history)
    client = dash.app.test_client()
    resp = client.get("/roi_data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["roi"] == [1.0, 1.0]
    assert data["security"] == [0.8, 0.9]


def test_load_error(tmp_path, monkeypatch, caplog):
    history = tmp_path / "hist.json"

    def boom(self, path):
        raise RuntimeError("fail")

    monkeypatch.setattr(sd_mod.ROITracker, "load_history", boom)
    dash = SandboxDashboard(history)
    client = dash.app.test_client()
    caplog.set_level("ERROR")
    resp = client.get("/roi_data")
    assert resp.status_code == 500
    assert "Failed to load ROI history" in caplog.text
    data = resp.get_json()
    assert data["error"]




def test_plot_predictions(tmp_path):
    path = tmp_path / "hist.json"
    tracker = ROITracker()
    tracker.update(
        0.0,
        1.0,
        metrics={
            "security_score": 0.8,
            "synergy_security_score": 0.05,
        },
    )
    tracker.record_metric_prediction("security_score", 0.9, 0.8)
    tracker.record_metric_prediction("synergy_security_score", 0.06, 0.05)
    tracker.record_prediction(1.1, 1.0)
    tracker.save_history(str(path))

    dash = SandboxDashboard(path)
    client = dash.app.test_client()
    resp = client.get("/plots/predictions.png")
    assert resp.status_code == 200
    assert resp.data.startswith(b"\x89PNG") or resp.data == b""


def test_weights_route(tmp_path):
    history = tmp_path / "hist.json"
    _make_history(history)
    log = tmp_path / "weights.log"
    entries = [
        {"timestamp": 1, "roi": 1.0, "efficiency": 0.5},
        {"timestamp": 2, "roi": 1.1, "efficiency": 0.6},
    ]
    log.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    dash = SandboxDashboard(history, weights_log=log)
    client = dash.app.test_client()
    resp = client.get("/weights")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["labels"] == [0, 1]
    assert data["weights"]["roi"] == [1.0, 1.1]
    assert data["weights"]["efficiency"] == [0.5, 0.6]
