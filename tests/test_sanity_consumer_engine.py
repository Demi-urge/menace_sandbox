import sys
import types
from pathlib import Path


# Stub dependencies required by billing.sanity_consumer
class _Bus:
    def __init__(self):
        self.handlers = []

    def subscribe(self, topic, cb):
        self.handlers.append((topic, cb))

    def publish(self, topic, event):
        for t, cb in list(self.handlers):
            if t == topic:
                cb(topic, event)


bus_module = types.ModuleType("unified_event_bus")
bus_module.UnifiedEventBus = _Bus
sys.modules["unified_event_bus"] = bus_module

sf_module = types.ModuleType("sanity_feedback")


class SanityFeedback:
    def __init__(self, engine, outcome_db=None):
        self.engine = engine
        self.outcome_db = outcome_db


sf_module.SanityFeedback = SanityFeedback
sys.modules["sanity_feedback"] = sf_module

sce_module = types.ModuleType("self_coding_engine")


class SelfCodingEngine:
    def __init__(self, *a, **k):
        pass


sce_module.SelfCodingEngine = SelfCodingEngine
sys.modules["self_coding_engine"] = sce_module

msl_module = types.ModuleType("menace_sanity_layer")
msl_module._EVENT_BUS = _Bus()
msl_module.record_event = lambda *a, **k: None
sys.modules["menace_sanity_layer"] = msl_module

dpr_module = types.ModuleType("dynamic_path_router")
dpr_module.resolve_path = lambda p: Path(p)
dpr_module.resolve_dir = lambda p: Path(p)
dpr_module.get_project_root = lambda: Path('.')
dpr_module.get_project_roots = lambda: [Path('.')]
sys.modules["dynamic_path_router"] = dpr_module

db_module = types.ModuleType("discrepancy_db")


class _DummyDB:
    def __init__(self, *a, **k):
        pass


db_module.DiscrepancyDB = _DummyDB
db_module.DiscrepancyRecord = object
sys.modules["discrepancy_db"] = db_module

code_db_module = types.ModuleType("code_database")


class _CodeDB:
    pass


code_db_module.CodeDB = _CodeDB
sys.modules["code_database"] = code_db_module

mmm_module = types.ModuleType("menace_memory_manager")


class _Memory:
    pass


mmm_module.MenaceMemoryManager = _Memory
sys.modules["menace_memory_manager"] = mmm_module

vector_pkg = types.ModuleType("vector_service")
ctx_module = types.ModuleType("vector_service.context_builder_utils")


class _Builder:
    def refresh_db_weights(self):
        pass


def get_default_context_builder(**_):
    return _Builder()


ctx_module.get_default_context_builder = get_default_context_builder
sys.modules["vector_service"] = vector_pkg
sys.modules["vector_service.context_builder_utils"] = ctx_module

import billing.sanity_consumer as sc  # noqa: E402
from unified_event_bus import UnifiedEventBus  # noqa: E402
from db_router import init_db_router  # noqa: E402


def test_injected_engine_used():
    class DummyEngine:
        pass

    engine = DummyEngine()
    consumer = sc.SanityConsumer(event_bus=UnifiedEventBus(), engine=engine)
    assert consumer._get_engine() is engine


def test_engine_instantiates_dependencies(monkeypatch, tmp_path):
    init_db_router('sc', str(tmp_path / 'local.db'), str(tmp_path / 'shared.db'))

    class DummyCodeDB:
        pass

    class DummyMemory:
        pass

    class DummyEngine:
        def __init__(self, code_db, memory_mgr, context_builder=None):
            self.code_db = code_db
            self.memory_mgr = memory_mgr
            self.context_builder = context_builder

    monkeypatch.setattr(sc, 'CodeDB', DummyCodeDB)
    monkeypatch.setattr(sc, 'MenaceMemoryManager', DummyMemory)
    monkeypatch.setattr(sc, 'SelfCodingEngine', DummyEngine)

    consumer = sc.SanityConsumer(event_bus=UnifiedEventBus())
    engine = consumer._get_engine()
    assert isinstance(engine, DummyEngine)
    assert isinstance(engine.code_db, DummyCodeDB)
    assert isinstance(engine.memory_mgr, DummyMemory)
    assert engine.context_builder is not None


def test_optional_dependency_failure(monkeypatch, tmp_path):
    init_db_router('sc2', str(tmp_path / 'local.db'), str(tmp_path / 'shared.db'))

    class FailingCodeDB:
        def __init__(self):
            raise RuntimeError('boom')

    class DummyMemory:
        pass

    captured = {}

    class DummyEngine:
        def __init__(self, code_db, memory_mgr, context_builder=None):
            captured['code_db'] = code_db
            captured['memory_mgr'] = memory_mgr
            captured['builder'] = context_builder

    monkeypatch.setattr(sc, 'CodeDB', FailingCodeDB)
    monkeypatch.setattr(sc, 'MenaceMemoryManager', DummyMemory)
    monkeypatch.setattr(sc, 'SelfCodingEngine', DummyEngine)

    consumer = sc.SanityConsumer(event_bus=UnifiedEventBus())
    engine = consumer._get_engine()
    assert engine is not None
    assert captured['memory_mgr'].__class__ is DummyMemory
    assert captured['code_db'].__class__ is object
    assert captured['builder'] is not None


def test_builder_shared_with_feedback(monkeypatch, tmp_path):
    init_db_router('sc3', str(tmp_path / 'local.db'), str(tmp_path / 'shared.db'))

    class DummyCodeDB:
        pass

    class DummyMemory:
        pass

    shared = {}

    class DummyEngine:
        def __init__(self, code_db, memory_mgr, context_builder=None):
            shared['builder'] = context_builder
            self.context_builder = context_builder

    monkeypatch.setattr(sc, 'CodeDB', DummyCodeDB)
    monkeypatch.setattr(sc, 'MenaceMemoryManager', DummyMemory)
    monkeypatch.setattr(sc, 'SelfCodingEngine', DummyEngine)

    consumer = sc.SanityConsumer(event_bus=UnifiedEventBus())
    feedback = consumer._get_feedback()
    assert getattr(feedback, 'context_builder', None) is shared['builder']
