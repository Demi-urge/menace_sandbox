import sys
import types
import pytest
import contextvars


class DummyManager:
    def __init__(self, *_, **kwargs):
        self.bot_registry = kwargs.get("bot_registry")
        self.data_bot = kwargs.get("data_bot")
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
        self.updated = []

    def register_bot(self, name):
        self.registered.append(name)

    def update_bot(self, name, module_path):
        self.updated.append((name, module_path))


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


def test_missing_bot_registry():
    @self_coding_managed
    class Bot:
        def __init__(self):
            pass

    with pytest.raises(RuntimeError, match="Bot: BotRegistry is required"):
        Bot(data_bot=DummyDataBot(), evolution_orchestrator=DummyOrchestrator())


def test_missing_data_bot():
    @self_coding_managed
    class Bot:
        def __init__(self):
            pass

    with pytest.raises(RuntimeError, match="Bot: DataBot is required"):
        Bot(bot_registry=DummyRegistry(), evolution_orchestrator=DummyOrchestrator())


def test_missing_orchestrator():
    @self_coding_managed
    class Bot:
        def __init__(self):
            pass

    with pytest.raises(RuntimeError, match="Bot: EvolutionOrchestrator is required"):
        Bot(bot_registry=DummyRegistry(), data_bot=DummyDataBot())


def test_successful_initialisation_registers():
    registry = DummyRegistry()
    data_bot = DummyDataBot()
    orchestrator = DummyOrchestrator()

    @self_coding_managed
    class Bot:
        name = "sample"

        def __init__(self):
            pass

    Bot(
        bot_registry=registry,
        data_bot=data_bot,
        evolution_orchestrator=orchestrator,
    )

    assert registry.registered == ["sample"]
    assert orchestrator.registered == ["sample"]

