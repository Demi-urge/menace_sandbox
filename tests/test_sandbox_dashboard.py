import importlib.util
import sys
import types
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
