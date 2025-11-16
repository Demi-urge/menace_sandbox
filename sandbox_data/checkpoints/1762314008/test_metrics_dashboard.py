import importlib.util
import sys
import types
from pathlib import Path

from menace.roi_tracker import ROITracker

pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
pkg.RAISE_ERRORS = False
spec_me = importlib.util.spec_from_file_location(
    "menace.metrics_exporter",
    Path(__file__).resolve().parents[1] / "metrics_exporter.py",  # path-ignore
    submodule_search_locations=pkg.__path__,
)
me = importlib.util.module_from_spec(spec_me)
me.__package__ = "menace"
sys.modules["menace.metrics_exporter"] = me
spec_me.loader.exec_module(me)

spec_md = importlib.util.spec_from_file_location(
    "menace.metrics_dashboard",
    Path(__file__).resolve().parents[1] / "metrics_dashboard.py",  # path-ignore
    submodule_search_locations=pkg.__path__,
)
md = importlib.util.module_from_spec(spec_md)
md.__package__ = "menace"
sys.modules["menace.metrics_dashboard"] = md
spec_md.loader.exec_module(md)

pkg.metrics_exporter = me
pkg.metrics_dashboard = md
class _Gauge:
    def __init__(self):
        self.value = 0
        self._value = types.SimpleNamespace(get=lambda: self.value)

    def set(self, v):
        self.value = v

sat_stub = types.SimpleNamespace(
    synergy_trainer_iterations=_Gauge(),
    synergy_trainer_failures_total=_Gauge(),
)
sys.modules["menace.synergy_auto_trainer"] = sat_stub
pkg.synergy_auto_trainer = sat_stub
MetricsDashboard = md.MetricsDashboard
metrics_exporter = me


def _make_history(path):
    tracker = ROITracker()
    tracker.update(
        0.0,
        1.0,
        metrics={
            "security_score": 0.8,
            "projected_lucrativity": 0.5,
            "synergy_roi": 0.1,
            "synergy_security_score": 0.05,
            "synergy_profitability": 0.2,
            "synergy_revenue": 0.15,
            "synergy_projected_lucrativity": 0.3,
            "risk_index": 1.0,
            "synergy_risk_index": 0.4,
        },
    )
    tracker.record_metric_prediction("projected_lucrativity", 0.6, 0.5)
    tracker.record_metric_prediction("risk_index", 1.1, 1.0)
    tracker.record_metric_prediction("synergy_risk_index", 0.5, 0.4)
    tracker.save_history(str(path))


def test_roi_and_metric_routes(tmp_path):
    history = tmp_path / "hist.json"
    _make_history(history)
    dash = MetricsDashboard(history)
    client = dash.app.test_client()

    resp = client.get("/roi")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["roi"] == [1.0]
    assert data["labels"] == [0]
    assert data["category_counts"] == {}
    assert data["synergy_roi"] == [0.1]
    assert data["synergy_security_score"] == [0.05]
    assert data["synergy_risk_index"] == [0.4]
    assert data["synergy_risk_index_predicted"] == [0.5]
    assert data["synergy_risk_index_actual"] == [0.4]
    assert data["workflow_mae"] == {}
    assert data["workflow_variance"] == {}
    assert data["workflow_confidence"] == {}

    resp = client.get("/metrics/security_score")
    assert resp.status_code == 200
    mdata = resp.get_json()
    assert mdata["values"] == [0.8]

    resp = client.get("/metrics/projected_lucrativity")
    pdata = resp.get_json()
    assert pdata["predicted"] == [0.6]
    assert pdata["actual"] == [0.5]

    resp = client.get("/metrics/risk_index")
    ridx = resp.get_json()
    assert ridx["predicted"] == [1.1]
    assert ridx["actual"] == [1.0]
    assert ridx["synergy_values"] == [0.4]
    assert ridx["synergy_predicted"] == [0.5]
    assert ridx["synergy_actual"] == [0.4]

    resp = client.get("/metrics/synergy_security_score")
    sdata = resp.get_json()
    assert sdata["values"] == [0.05]

    resp = client.get("/metrics/synergy_profitability")
    spdata = resp.get_json()
    assert spdata["values"] == [0.2]

    resp = client.get("/metrics/synergy_revenue")
    srdata2 = resp.get_json()
    assert srdata2["values"] == [0.15]

    resp = client.get("/metrics/synergy_risk_index")
    srdata = resp.get_json()
    assert srdata["values"] == [0.4]
    assert srdata["predicted"] == [0.5]
    assert srdata["actual"] == [0.4]

    resp = client.get("/metrics/synergy_projected_lucrativity")
    spldata = resp.get_json()
    assert spldata["values"] == [0.3]

    resp = client.get("/plots/predictions.png")
    assert resp.status_code == 200
    assert resp.data.startswith(b"\x89PNG") or resp.data == b""

    metrics_exporter.container_creation_success_total.labels("img").set(3)
    metrics_exporter.container_creation_failures_total.labels("img").set(1)
    metrics_exporter.container_creation_alerts_total.labels("img").set(2)
    metrics_exporter.synergy_weight_update_failures_total.set(4)

    from menace import synergy_auto_trainer as sat

    sat.synergy_trainer_iterations.set(5)
    sat.synergy_trainer_failures_total.set(2)

    all_metrics = client.get("/metrics").get_json()
    assert all_metrics["container_creation_success_total"] == 3
    assert all_metrics["container_creation_failures_total"] == 1
    assert all_metrics["container_creation_alerts_total"] == 2
    assert all_metrics["synergy_weight_update_failures_total"] == 4
    assert all_metrics["synergy_trainer_iterations"] == 5
    assert all_metrics["synergy_trainer_failures_total"] == 2


def test_refresh_endpoint(tmp_path, monkeypatch):
    hist = tmp_path / "hist.json"
    _make_history(hist)
    tb = md.TelemetryBackend(str(tmp_path / "tel.db"))
    tb.log_prediction("wf1", 0.5, 0.4, None, None, False, 0.9, None)
    monkeypatch.setattr(md, "TelemetryBackend", lambda: tb)
    dash = MetricsDashboard(hist)
    client = dash.app.test_client()
    resp = client.get("/refresh")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["readiness"]["wf1"] == 0.9
    assert data["telemetry"][0]["workflow_id"] == "wf1"


def test_scenario_deltas_chart(tmp_path, monkeypatch):
    hist = tmp_path / "hist.json"
    _make_history(hist)
    tb = md.TelemetryBackend(str(tmp_path / "tel.db"))
    tb.log_prediction(
        "wf1",
        0.5,
        0.4,
        None,
        {"scenA": 1.0},
        False,
        0.9,
        "scenA",
        "2020-01-01",
    )
    tb.log_prediction(
        "wf1",
        0.6,
        0.5,
        None,
        {"scenA": 2.0},
        False,
        0.9,
        "scenA",
        "2020-01-02",
    )
    tb.log_prediction(
        "wf1",
        0.7,
        0.6,
        None,
        {"scenB": 3.0},
        False,
        0.9,
        "scenB",
        "2020-01-03",
    )
    monkeypatch.setattr(md, "TelemetryBackend", lambda: tb)
    dash = MetricsDashboard(hist)
    client = dash.app.test_client()
    resp = client.get("/scenario_deltas/wf1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["scenarios"]["scenA"]["roi"] == [1.0, 2.0]
    assert data["scenarios"]["scenB"]["roi"] == [3.0]

    resp = client.get(
        "/scenario_deltas/wf1?scenario=scenA&start=2020-01-02&end=2020-01-04"
    )
    data = resp.get_json()
    assert list(data["scenarios"].keys()) == ["scenA"]
    assert data["scenarios"]["scenA"]["roi"] == [2.0]
