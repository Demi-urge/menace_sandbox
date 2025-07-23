import importlib.util
import sys
import types
from pathlib import Path

from menace.roi_tracker import ROITracker

if "menace.metrics_dashboard" in sys.modules:
    from menace.metrics_dashboard import MetricsDashboard
    from menace import metrics_exporter
else:  # pragma: no cover - allow running file directly
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
    spec_me = importlib.util.spec_from_file_location(
        "menace.metrics_exporter",
        Path(__file__).resolve().parents[1] / "metrics_exporter.py",
        submodule_search_locations=pkg.__path__,
    )
    me = importlib.util.module_from_spec(spec_me)
    me.__package__ = "menace"
    sys.modules["menace.metrics_exporter"] = me
    spec_me.loader.exec_module(me)

    spec_md = importlib.util.spec_from_file_location(
        "menace.metrics_dashboard",
        Path(__file__).resolve().parents[1] / "metrics_dashboard.py",
        submodule_search_locations=pkg.__path__,
    )
    md = importlib.util.module_from_spec(spec_md)
    md.__package__ = "menace"
    sys.modules["menace.metrics_dashboard"] = md
    spec_md.loader.exec_module(md)

    pkg.metrics_exporter = me
    pkg.metrics_dashboard = md
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
    assert data["synergy_roi"] == [0.1]
    assert data["synergy_security_score"] == [0.05]
    assert data["synergy_risk_index"] == [0.4]
    assert data["synergy_risk_index_predicted"] == [0.5]
    assert data["synergy_risk_index_actual"] == [0.4]

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

    metrics_exporter.visual_agent_queue_depth.set(1)
    metrics_exporter.visual_agent_wait_time.set(0.2)
    all_metrics = client.get("/metrics").get_json()
    assert all_metrics["visual_agent_queue_depth"] == 1
    assert all_metrics["visual_agent_wait_time"] == 0.2
