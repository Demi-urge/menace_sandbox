import types
import sys

def _prepare(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    stub = types.ModuleType("stub")
    stub.Template = lambda *a, **k: types.SimpleNamespace(render=lambda **kw: "")
    stub.safe_load = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "jinja2", stub)
    monkeypatch.setitem(sys.modules, "yaml", stub)
    monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", types.ModuleType("pyplot"))  # path-ignore
    monkeypatch.setitem(sys.modules, "sqlalchemy", types.ModuleType("sqlalchemy"))
    monkeypatch.setitem(sys.modules, "httpx", types.ModuleType("httpx"))
    monkeypatch.setitem(
        sys.modules,
        "env_config",
        types.SimpleNamespace(DATABASE_URL="sqlite:///test.db", BUDGET_MAX_INSTANCES=2),
    )
    auto_mod = types.ModuleType("menace.autoscaler")
    class DummyAutoscaler:
        def __init__(self, *a, **k) -> None:
            self.up = 0
        def scale_up(self, amount: int = 1) -> None:
            self.up += amount
        def scale_down(self, amount: int = 1) -> None:
            self.up -= amount
        def scale(self, metrics):
            pass
    auto_mod.Autoscaler = DummyAutoscaler
    monkeypatch.setitem(sys.modules, "menace.autoscaler", auto_mod)

    import menace.operational_monitor_bot as omb  # type: ignore
    import menace.data_bot as db  # type: ignore
    from menace.unified_event_bus import UnifiedEventBus  # type: ignore
    return omb, db, UnifiedEventBus

class StubES:
    def __init__(self) -> None:
        self.docs = []
    def add(self, doc_id: str, body: dict) -> None:
        self.docs.append({"id": doc_id, **body})


def test_anomaly_triggers_autoscale(tmp_path, monkeypatch):
    omb, db, UEB = _prepare(monkeypatch)
    monkeypatch.setattr(db, "Gauge", None)
    monkeypatch.setattr(omb, "IForest", None)
    monkeypatch.setattr(
        omb, "AnomalyEnsembleDetector", lambda *a, **k: types.SimpleNamespace(detect=lambda: [])
    )
    monkeypatch.setattr(
        omb, "PlaybookGenerator", lambda *a, **k: types.SimpleNamespace(generate=lambda *_: "")
    )
    mdb = db.MetricsDB(tmp_path / "m.db")
    es = StubES()
    a_db = omb.AnomalyDB(tmp_path / "a.db")
    bus = UEB()
    events = []
    bus.subscribe("autoscale:request", lambda t, e: events.append(e))

    auto = omb.Autoscaler()

    bot = omb.OperationalMonitoringBot(
        mdb,
        es,
        anomaly_db=a_db,
        autoscaler=auto,
        event_bus=bus,
        severity_threshold=50.0,
    )
    normal = db.MetricRecord("bot1", 10.0, 20.0, 0.1, 1.0, 1.0, 0)
    for _ in range(5):
        mdb.add(normal)
    anomaly = db.MetricRecord("bot1", 99.0, 99.0, 1.0, 1.0, 1.0, 5)
    mdb.add(anomaly)

    bot.detect_anomalies("bot1")

    assert auto.up == 1
    assert events and events[0]["bot"] == "bot1"
