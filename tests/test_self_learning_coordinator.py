import sys
import types
import importlib.machinery

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
vs_mod = types.ModuleType("vector_service")
vs_mod.CognitionLayer = object
sys.modules.setdefault("vector_service", vs_mod)

sii = types.ModuleType("menace.self_improvement.init")


class DummyLock:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    @property
    def is_locked(self):
        return True


def fake_atomic_write(path, data, *, lock=None, binary=False):
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    path.write_text(data)


sii.FileLock = DummyLock
sii._atomic_write = fake_atomic_write
si_pkg = types.ModuleType("menace.self_improvement")
si_pkg.init = sii
si_pkg.__path__ = []
sys.modules.setdefault("menace.self_improvement", si_pkg)
sys.modules.setdefault("menace.self_improvement.init", sii)

# Stub optional jinja2 dependency used by ErrorBot
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = type("T", (), {"render": lambda self, *a, **k: ""})
jinja_mod.__spec__ = importlib.machinery.ModuleSpec("jinja2", loader=None)
sys.modules.setdefault("jinja2", jinja_mod)
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", yaml_mod)
stub_err = types.ModuleType("menace.error_bot")
stub_err.ErrorBot = object
stub_err.ErrorDB = object
sys.modules.setdefault("menace.error_bot", stub_err)

# Stub MetricsDB to avoid heavy imports
stub_db = types.ModuleType("menace.data_bot")


class MetricsDB:  # noqa: D401 - simple stub
    def __init__(self, *a, **k):
        pass

    def log_training_stat(self, *a, **k):
        pass


stub_db.MetricsDB = MetricsDB
sys.modules.setdefault("menace.data_bot", stub_db)

# Stub learning engine modules to avoid heavy imports
ule = types.ModuleType("menace.unified_learning_engine")
ule.UnifiedLearningEngine = object
sys.modules.setdefault("menace.unified_learning_engine", ule)
ale = types.ModuleType("menace.action_learning_engine")
ale.ActionLearningEngine = object
sys.modules.setdefault("menace.action_learning_engine", ale)
neuro = types.ModuleType("menace.neuroplasticity")


class PathwayRecord:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Outcome:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


neuro.PathwayRecord = PathwayRecord
neuro.Outcome = Outcome
sys.modules.setdefault("menace.neuroplasticity", neuro)
le = types.ModuleType("menace.learning_engine")
le.LearningEngine = object
sys.modules.setdefault("menace.learning_engine", le)
evm = types.ModuleType("menace.evaluation_manager")


class EvaluationManager:
    def __init__(self, *engines):
        self.engines = engines

    def evaluate_all(self):
        for eng in self.engines:
            if eng and hasattr(eng, "evaluate"):
                eng.evaluate()


evm.EvaluationManager = EvaluationManager
sys.modules.setdefault("menace.evaluation_manager", evm)
stub_curr = types.ModuleType("menace.curriculum_builder")
stub_curr.CurriculumBuilder = object
sys.modules.setdefault("menace.curriculum_builder", stub_curr)

from menace.unified_event_bus import UnifiedEventBus  # noqa: E402
from menace.data_bot import MetricsDB  # noqa: E402
from menace.self_learning_coordinator import SelfLearningCoordinator  # noqa: E402
from unittest.mock import AsyncMock  # noqa: E402
import asyncio  # noqa: E402
import pytest  # noqa: E402

# Cleanup stubs so they don't affect other tests
for mod in [
    "pandas",
    "networkx",
    "pulp",
    "sqlalchemy",
    "sqlalchemy.engine",
    "jinja2",
    "yaml",
    "vector_service",
    "menace.self_improvement",
    "menace.self_improvement.init",
    "menace.error_bot",
    "menace.data_bot",
    "menace.unified_learning_engine",
    "menace.action_learning_engine",
    "menace.neuroplasticity",
    "menace.learning_engine",
    "menace.evaluation_manager",
    "menace.curriculum_builder",
]:
    sys.modules.pop(mod, None)


@pytest.fixture(autouse=True)
def _mock_atomic_write(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))


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
    bus._loop.run_until_complete(asyncio.sleep(0.1))
    assert engine.records
    assert engine.records[0].actions == "a->b"


def test_coordinator_trains_on_memory(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    coord = SelfLearningCoordinator(bus, learning_engine=engine)
    coord.start()
    bus.publish("memory:new", {"key": "X"})
    bus._loop.run_until_complete(asyncio.sleep(0.1))
    assert engine.records
    assert engine.records[0].actions == "X"


def test_coordinator_trains_on_code(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    coord = SelfLearningCoordinator(bus, learning_engine=engine)
    coord.start()
    bus.publish("code:new", {"summary": "Y"})
    bus._loop.run_until_complete(asyncio.sleep(0.1))
    assert engine.records
    assert engine.records[0].actions == "Y"


def test_coordinator_trains_on_error(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    coord = SelfLearningCoordinator(bus, learning_engine=engine)
    coord.start()
    bus.publish("errors:new", {"message": "boom"})
    bus._loop.run_until_complete(asyncio.sleep(0.1))
    assert engine.records
    assert engine.records[0].actions == "boom"


def test_coordinator_evaluates_interval(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    import os
    os.environ["SELF_LEARNING_EVAL_INTERVAL"] = "1"
    coord = SelfLearningCoordinator(bus, learning_engine=engine, eval_interval=1)
    coord.start()
    bus.publish("memory:new", {"key": "A"})
    bus._loop.run_until_complete(asyncio.sleep(0.1))
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
    bus._loop.run_until_complete(asyncio.sleep(0.1))
    assert engine.records
    assert engine.records[0].actions == "b"


def test_coordinator_trains_on_transaction(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    coord = SelfLearningCoordinator(bus, learning_engine=engine)
    coord.start()
    bus.publish("transactions:new", {"model_id": "M1", "amount": 5.0, "result": "success"})
    bus._loop.run_until_complete(asyncio.sleep(0.1))
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
        def __init__(self, db, metrics, context_builder=None):
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
    err_bot = eb.ErrorBot(edb, mdb, context_builder=None)
    import os
    os.environ["SELF_LEARNING_SUMMARY_INTERVAL"] = "1"
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
    bus._loop.run_until_complete(asyncio.sleep(0.1))
    assert len(engine.records) >= 2


def test_stop_unsubscribes_callbacks():
    class DummyBus:
        def __init__(self) -> None:
            self.subs: dict[str, list] = {}
            self.loop = asyncio.new_event_loop()

        def subscribe_async(self, topic, callback):
            self.subs.setdefault(topic, []).append(callback)

        def unsubscribe(self, topic, callback):
            if topic in self.subs and callback in self.subs[topic]:
                self.subs[topic].remove(callback)
                if not self.subs[topic]:
                    del self.subs[topic]

        def publish(self, topic, event):
            for cb in list(self.subs.get(topic, [])):
                self.loop.run_until_complete(cb(topic, event))

    bus = DummyBus()
    coord = SelfLearningCoordinator(bus)
    orig = coord._on_memory
    mem_cb = AsyncMock(side_effect=orig)
    coord._on_memory = mem_cb
    coord._train_all = AsyncMock()
    coord.start()
    bus.publish("memory:new", {"key": "A"})
    assert mem_cb.await_count == 1
    coord.stop()
    bus.publish("memory:new", {"key": "B"})
    assert mem_cb.await_count == 1
    assert "memory:new" not in bus.subs
