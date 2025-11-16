import sys
import types
import importlib.machinery

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
sys.modules.setdefault("prometheus_client", types.ModuleType("prometheus_client"))
sii = types.ModuleType("menace.self_improvement.init")
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
vs_mod = types.ModuleType("vector_service")
vs_mod.CognitionLayer = object
sys.modules.setdefault("vector_service", vs_mod)
stub_curr = types.ModuleType("menace.curriculum_builder")


class CurriculumBuilder:
    def __init__(self, error_bot, event_bus, *, threshold=3):
        self.error_bot = error_bot
        self.event_bus = event_bus
        self.threshold = threshold

    def build(self):
        summary = self.error_bot.summarize_telemetry()
        return [
            {"error_type": str(item.get("error_type", ""))}
            for item in summary
            if item.get("count", 0) >= self.threshold
        ]

    def publish(self):
        items = self.build()
        for entry in items:
            self.event_bus.publish("curriculum:new", entry)
        return items


stub_curr.CurriculumBuilder = CurriculumBuilder
sys.modules.setdefault("menace.curriculum_builder", stub_curr)

from menace.unified_event_bus import UnifiedEventBus  # noqa: E402

# Stub optional modules used by ErrorBot

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
    def __init__(self, *a, **k):
        pass


evm.EvaluationManager = EvaluationManager
sys.modules.setdefault("menace.evaluation_manager", evm)

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = type("T", (), {"render": lambda self, *a, **k: ""})
jinja_mod.__spec__ = importlib.machinery.ModuleSpec("jinja2", loader=None)
sys.modules.setdefault("jinja2", jinja_mod)
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", yaml_mod)

from menace.curriculum_builder import CurriculumBuilder  # noqa: E402
from menace.self_learning_coordinator import SelfLearningCoordinator  # noqa: E402
import asyncio  # noqa: E402
import pytest  # noqa: E402

for mod in [
    "networkx",
    "pulp",
    "pandas",
    "sqlalchemy",
    "sqlalchemy.engine",
    "prometheus_client",
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
    def __init__(self):
        self.records = []

    def partial_train(self, rec):
        self.records.append(rec)
        return True


class DummyErrorBot:
    def __init__(self, summary):
        self._summary = summary

    def summarize_telemetry(self):
        return self._summary


def test_curriculum_generation_triggers_training(tmp_path):
    bus = UnifiedEventBus()
    engine = DummyEngine()
    err_bot = DummyErrorBot([
        {"error_type": "io", "count": 5, "success_rate": 0.0},
        {"error_type": "net", "count": 1, "success_rate": 0.0},
    ])
    builder = CurriculumBuilder(err_bot, bus, threshold=2)
    coord = SelfLearningCoordinator(bus, learning_engine=engine, curriculum=builder)
    coord.start()
    builder.publish()
    bus._loop.run_until_complete(asyncio.sleep(0.1))
    assert engine.records
    assert engine.records[0].actions == "io"
