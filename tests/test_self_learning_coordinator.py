import sys
import types

# Stub external dependencies used by planning components
sys.modules.setdefault("networkx", types.ModuleType("networkx"))
sys.modules.setdefault("pulp", types.ModuleType("pulp"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
sqlalchemy_mod = types.ModuleType("sqlalchemy")
engine_mod = types.ModuleType("sqlalchemy.engine")
class DummyEngine:
    pass
engine_mod.Engine = DummyEngine
sqlalchemy_mod.engine = engine_mod
sys.modules.setdefault("sqlalchemy", sqlalchemy_mod)
sys.modules.setdefault("sqlalchemy.engine", engine_mod)

# Stub optional jinja2 dependency used by ErrorBot
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = type("T", (), {"render": lambda self, *a, **k: ""})
sys.modules.setdefault("jinja2", jinja_mod)
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", yaml_mod)
stub_err = types.ModuleType("menace.error_bot")
stub_err.ErrorBot = object
stub_err.ErrorDB = object
sys.modules.setdefault("menace.error_bot", stub_err)

from menace.unified_event_bus import UnifiedEventBus
from menace.data_bot import MetricsDB
from menace.self_learning_coordinator import SelfLearningCoordinator
from unittest.mock import MagicMock

# Cleanup stubs so they don't affect other tests
for mod in [
    "pandas",
    "networkx",
    "pulp",
    "sqlalchemy",
    "sqlalchemy.engine",
    "jinja2",
    "yaml",
]:
    sys.modules.pop(mod, None)


class DummyEngine:
    def __init__(self) -> None:
        self.records = []
        self.evaluated = 0

    def partial_train(self, rec):
        self.records.append(rec)
        return True

    def evaluate(self):
        self.evaluated += 1
        return {"cv_score": 0.0, "holdout_score": 0.0}

    def persist_evaluation(self, res):
        self.persisted = res


def test_coordinator_trains_on_workflow(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    coord = SelfLearningCoordinator(bus, learning_engine=engine)
    coord.start()
    bus.publish(
        "workflows:new",
        {
            "workflow": ["a", "b"],
            "status": "active",
            "workflow_duration": 1.0,
            "estimated_profit_per_bot": 2.0,
        },
    )
    assert engine.records
    assert engine.records[0].actions == "a->b"


def test_coordinator_trains_on_memory(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    coord = SelfLearningCoordinator(bus, learning_engine=engine)
    coord.start()
    bus.publish("memory:new", {"key": "X"})
    assert engine.records
    assert engine.records[0].actions == "X"


def test_coordinator_trains_on_code(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    coord = SelfLearningCoordinator(bus, learning_engine=engine)
    coord.start()
    bus.publish("code:new", {"summary": "Y"})
    assert engine.records
    assert engine.records[0].actions == "Y"


def test_coordinator_trains_on_error(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    coord = SelfLearningCoordinator(bus, learning_engine=engine)
    coord.start()
    bus.publish("errors:new", {"message": "boom"})
    assert engine.records
    assert engine.records[0].actions == "boom"


def test_coordinator_evaluates_interval(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    coord = SelfLearningCoordinator(bus, learning_engine=engine, eval_interval=1)
    coord.start()
    bus.publish("memory:new", {"key": "A"})
    assert engine.evaluated == 1


def test_coordinator_trains_on_metrics(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    mdb = MetricsDB(tmp_path / "m.db")
    coord = SelfLearningCoordinator(bus, learning_engine=engine, metrics_db=mdb)
    coord.start()
    bus.publish(
        "metrics:new",
        {
            "bot": "b",
            "cpu": 1.0,
            "memory": 1.0,
            "response_time": 0.1,
            "disk_io": 0.0,
            "net_io": 0.0,
            "errors": 0,
            "revenue": 1.0,
            "expense": 0.5,
        },
    )
    assert engine.records
    assert engine.records[0].actions == "b"


def test_coordinator_trains_on_transaction(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    coord = SelfLearningCoordinator(bus, learning_engine=engine)
    coord.start()
    bus.publish("transactions:new", {"model_id": "M1", "amount": 5.0, "result": "success"})
    assert engine.records
    assert engine.records[0].actions == "M1"
    assert engine.records[0].roi == 5.0


def test_telemetry_summary_updates_training(tmp_path):
    import importlib

    class DummyErrorDB:
        def __init__(self, path, event_bus=None):
            self.telemetry = []
            self.event_bus = event_bus

        def add_telemetry(self, ev):
            self.telemetry.append(ev)
            if self.event_bus:
                self.event_bus.publish(
                    "telemetry:new", {"error_type": str(ev.error_type)}
                )

    class DummyErrorBot:
        def __init__(self, db, metrics):
            self.db = db
            self.metrics = metrics

        def summarize_telemetry(self):
            return [
                {"error_type": t.error_type, "success_rate": 0.0}
                for t in self.db.telemetry
            ]

    eb = types.ModuleType("menace.error_bot")
    eb.ErrorDB = DummyErrorDB
    eb.ErrorBot = DummyErrorBot
    sys.modules["menace.error_bot"] = eb
    elog = importlib.import_module("menace.error_logger")

    bus = UnifiedEventBus()
    engine = DummyEngine()
    mdb = MetricsDB(tmp_path / "m.db")
    edb = eb.ErrorDB(tmp_path / "e.db", event_bus=bus)
    err_bot = eb.ErrorBot(edb, mdb)
    coord = SelfLearningCoordinator(
        bus,
        learning_engine=engine,
        error_bot=err_bot,
        summary_interval=1,
    )
    coord.start()
    edb.add_telemetry(
        elog.TelemetryEvent(
            error_type=elog.ErrorType.RUNTIME_FAULT,
            resolution_status="fatal",
        )
    )
    assert len(engine.records) >= 2


def test_stop_unsubscribes_callbacks():
    class DummyBus:
        def __init__(self) -> None:
            self.subs: dict[str, list] = {}

        def subscribe(self, topic, callback):
            self.subs.setdefault(topic, []).append(callback)

        def unsubscribe(self, topic, callback):
            if topic in self.subs and callback in self.subs[topic]:
                self.subs[topic].remove(callback)
                if not self.subs[topic]:
                    del self.subs[topic]

        def publish(self, topic, event):
            for cb in list(self.subs.get(topic, [])):
                cb(topic, event)

    bus = DummyBus()
    coord = SelfLearningCoordinator(bus)
    orig = coord._on_memory
    mem_cb = MagicMock(side_effect=orig)
    coord._on_memory = mem_cb
    coord._train_all = MagicMock()
    coord.start()
    bus.publish("memory:new", {"key": "A"})
    assert mem_cb.call_count == 1
    coord.stop()
    bus.publish("memory:new", {"key": "B"})
    assert mem_cb.call_count == 1
    assert "memory:new" not in bus.subs

