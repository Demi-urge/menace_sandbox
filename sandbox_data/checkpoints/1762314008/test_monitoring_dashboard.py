import json
from menace import monitoring_dashboard as md
from menace import data_bot as db
from menace import evolution_history_db as eh
from menace import error_bot as eb


def test_data_routes(tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    mdb.add(db.MetricRecord(bot="b", cpu=1.0, memory=1.0, response_time=0.1, disk_io=0.0, net_io=0.0, errors=1))
    edb = eh.EvolutionHistoryDB(tmp_path / "e.db")
    edb.add(eh.EvolutionEvent(action="a", before_metric=1.0, after_metric=2.0, roi=1.0))
    errdb = eb.ErrorDB(tmp_path / "err.db")
    errdb.add_telemetry(
        eb.TelemetryEvent(
            bot_id="b",
            task_id="t",
            error_type=eb.ErrorType.RUNTIME_FAULT,
            stack_trace="",
            root_module="m",
        )
    )
    dash = md.MonitoringDashboard(mdb, edb, errdb)
    client = dash.app.test_client()
    resp = client.get("/metrics_data")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "cpu" in data
    resp = client.get("/evolution_data")
    assert resp.get_json()["roi"]
    resp = client.get("/error_data")
    assert resp.get_json()["count"]


def test_schedule_reports(monkeypatch, tmp_path):
    mdb = db.MetricsDB(tmp_path / "m.db")
    dash = md.MonitoringDashboard(mdb)
    called = {}

    def fake_schedule(self, options, interval=1):
        called['ok'] = options.metrics

    monkeypatch.setattr(md.ReportGenerationBot, "schedule", fake_schedule)
    dash.schedule_reports(metrics=["cpu"], recipients=["a@example.com"], interval=1)
    assert called['ok'] == ["cpu"]


