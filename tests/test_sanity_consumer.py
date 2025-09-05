import importlib
import sys
import types

class DummyCodeDB:
    pass

class DummyMMM:
    pass

class DummyEngine:
    def __init__(self, code_db, memory_mgr):
        self.code_db = code_db
        self.memory_mgr = memory_mgr
        self.patch_calls = []
        self.update_calls = []

    def apply_patch(self, target, message, **kwargs):
        self.patch_calls.append((target, message, kwargs))
        return ("p1", True, None)

    def update_generation_params(self, meta):
        self.update_calls.append(meta)

class DummyFeedback:
    def __init__(self, engine, outcome_db=None):
        self.engine = engine

class DummyBus:
    def __init__(self):
        self.handlers = {}

    def subscribe(self, topic, handler):
        self.handlers[topic] = handler


def _import_consumer(monkeypatch):
    # stub dependencies before importing
    sys.modules['code_database'] = types.SimpleNamespace(CodeDB=DummyCodeDB)
    sys.modules['menace_memory_manager'] = types.SimpleNamespace(MenaceMemoryManager=DummyMMM)
    bus_mod = types.ModuleType('unified_event_bus')
    bus_mod.UnifiedEventBus = DummyBus
    sys.modules['unified_event_bus'] = bus_mod
    sys.modules['self_coding_engine'] = types.SimpleNamespace(SelfCodingEngine=DummyEngine)
    sys.modules['sanity_feedback'] = types.SimpleNamespace(SanityFeedback=DummyFeedback)
    sys.modules['menace_sanity_layer'] = types.SimpleNamespace(record_event=lambda *a, **k: None)
    sys.modules['dynamic_path_router'] = types.SimpleNamespace(resolve_path=lambda p: p)
    if 'billing.sanity_consumer' in sys.modules:
        del sys.modules['billing.sanity_consumer']
    sc = importlib.import_module('billing.sanity_consumer')
    return sc


def test_applies_patch_when_path_provided(monkeypatch):
    sc = _import_consumer(monkeypatch)
    bus = DummyBus()
    consumer = sc.SanityConsumer(event_bus=bus)
    event = {'metadata': {'path': 'mod'}, 'event_type': 'bug'}
    bus.handlers['billing.anomaly']('billing.anomaly', event)
    engine = consumer._get_engine()
    assert engine.patch_calls
    assert not engine.update_calls


def test_updates_params_when_no_path(monkeypatch):
    sc = _import_consumer(monkeypatch)
    bus = DummyBus()
    consumer = sc.SanityConsumer(event_bus=bus)
    event = {'metadata': {'foo': 'bar'}}
    bus.handlers['billing.anomaly']('billing.anomaly', event)
    engine = consumer._get_engine()
    assert engine.update_calls == [{'foo': 'bar'}]
    assert not engine.patch_calls
