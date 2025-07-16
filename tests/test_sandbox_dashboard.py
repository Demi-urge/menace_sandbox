from menace.sandbox_dashboard import SandboxDashboard
from menace.roi_tracker import ROITracker


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
