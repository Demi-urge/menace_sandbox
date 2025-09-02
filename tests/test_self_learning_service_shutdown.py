import sys
import types
import threading
import importlib


class Dummy:
    def __init__(self, *a, **k):
        pass


class DummyMem:
    def __init__(self):
        self.conn = types.SimpleNamespace(
            execute=lambda q: types.SimpleNamespace(fetchone=lambda: (0,))
        )

    def compact(self, interval):
        pass


class DummyLocalKnowledge:
    def __init__(self):
        self.memory = DummyMem()


class DummyConfig:
    def __init__(self):
        self.persist_events = None
        self.persist_progress = None
        self.prune_interval = 1


def _stub_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    pkg_name, _, sub = name.partition(".")
    pkg = sys.modules.get(pkg_name)
    if pkg and sub:
        setattr(pkg, sub, mod)
    return mod


def _setup(monkeypatch):
    import menace  # ensure package exists
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=Dummy)
    _stub_module(monkeypatch, "menace.neuroplasticity", PathwayDB=Dummy)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=Dummy)
    _stub_module(monkeypatch, "menace.code_database", CodeDB=Dummy)
    _stub_module(monkeypatch, "menace.resource_allocation_optimizer", ROIDB=Dummy)
    _stub_module(monkeypatch, "menace.learning_engine", LearningEngine=Dummy)
    _stub_module(monkeypatch, "menace.unified_learning_engine", UnifiedLearningEngine=Dummy)
    _stub_module(monkeypatch, "menace.action_learning_engine", ActionLearningEngine=Dummy)

    class DummyCoordinator:
        def __init__(self, *a, **k):
            self.evaluation_manager = types.SimpleNamespace(evaluate_all=lambda: {})

        def start(self):
            pass

        def stop(self):
            pass

    _stub_module(
        monkeypatch,
        "menace.self_learning_coordinator",
        SelfLearningCoordinator=DummyCoordinator,
    )
    _stub_module(
        monkeypatch,
        "menace.local_knowledge_module",
        init_local_knowledge=lambda *a, **k: DummyLocalKnowledge(),
    )
    _stub_module(monkeypatch, "menace.self_services_config", SelfLearningConfig=DummyConfig)

    if "menace.self_learning_service" in sys.modules:
        sls = importlib.reload(sys.modules["menace.self_learning_service"])
    else:
        sls = importlib.import_module("menace.self_learning_service")
    monkeypatch.setattr(sls.time, "sleep", lambda s: None)
    return sls


def test_main_premature_stop_joins_threads(monkeypatch):
    sls = _setup(monkeypatch)
    stop_event = threading.Event()
    stop_event.set()
    before = set(threading.enumerate())
    sls.main(stop_event=stop_event, prune_interval=1)
    after = set(threading.enumerate())
    assert before == after


def test_run_background_shutdown_leaves_no_threads(monkeypatch):
    sls = _setup(monkeypatch)
    before = set(threading.enumerate())
    start, stop = sls.run_background(prune_interval=1)
    start()
    stop()
    after = set(threading.enumerate())
    assert before == after
