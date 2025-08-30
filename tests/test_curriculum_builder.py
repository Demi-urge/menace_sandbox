import sys
import types

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

jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = type("T", (), {"render": lambda self, *a, **k: ""})
sys.modules.setdefault("jinja2", jinja_mod)
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", yaml_mod)

from menace.curriculum_builder import CurriculumBuilder  # noqa: E402
from menace.self_learning_coordinator import SelfLearningCoordinator  # noqa: E402
import asyncio  # noqa: E402

for mod in [
    "networkx",
    "pulp",
    "pandas",
    "sqlalchemy",
    "sqlalchemy.engine",
    "prometheus_client",
]:
    sys.modules.pop(mod, None)


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
    coord = SelfLearningCoordinator(bus, learning_engine=engine, curriculum_builder=builder)
    coord.start()
    builder.publish()
    bus._loop.run_until_complete(asyncio.sleep(0.1))
    assert engine.records
    assert engine.records[0].actions == "io"
