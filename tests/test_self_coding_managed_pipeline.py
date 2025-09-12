import sys
import types
import contextvars


class DummyManager:
    def __init__(self, *_, **kwargs):
        self.bot_registry = kwargs.get("bot_registry")
        self.data_bot = kwargs.get("data_bot")
        self.evolution_orchestrator = kwargs.get("evolution_orchestrator")
        self.quick_fix = kwargs.get("quick_fix") or object()


stub = types.ModuleType("menace.self_coding_manager")
stub.SelfCodingManager = DummyManager
sys.modules.setdefault("menace.self_coding_manager", stub)

sce_stub = types.ModuleType("menace.self_coding_engine")
sce_stub.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT")
sce_stub.SelfCodingEngine = object
sys.modules.setdefault("menace.self_coding_engine", sce_stub)

from menace.coding_bot_interface import self_coding_managed


class DummyRegistry:
    def __init__(self):
        self.registered = []

    def register_bot(self, name):
        self.registered.append(name)

    def update_bot(self, name, module_path):
        self.updated = (name, module_path)


class DummyDB:
    def __init__(self):
        self.logged = []

    def log_eval(self, name, metric, value):
        self.logged.append((name, metric, value))


class DummyDataBot:
    def __init__(self):
        self.db = DummyDB()

    def roi(self, name):
        return 0.0


class DummyOrchestrator:
    def __init__(self):
        self.registered = []

    def register_bot(self, name):
        self.registered.append(name)


def test_self_coding_managed_pipeline():
    registry = DummyRegistry()
    data_bot = DummyDataBot()
    orchestrator = DummyOrchestrator()
    manager = DummyManager(
        bot_registry=registry, data_bot=data_bot, evolution_orchestrator=orchestrator
    )

    @self_coding_managed
    class DummyBot:
        def __init__(self, manager=None):
            self.manager = manager
            self.bot_name = "dummy"

    DummyBot(
        manager=manager,
        bot_registry=registry,
        data_bot=data_bot,
        evolution_orchestrator=orchestrator,
    )

    assert "dummy" in registry.registered
    assert "dummy" in orchestrator.registered
    assert ("dummy", "roi", 0.0) in data_bot.db.logged
    assert ("dummy", "errors", 0.0) in data_bot.db.logged
