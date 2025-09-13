import sys
import types
import sys
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

# lightweight stubs for orchestrator dependencies used during auto-instantiation
evo_stub = types.ModuleType("menace.evolution_orchestrator")

class _StubOrchestrator:
    def __init__(self, data_bot, capital_bot, improvement_engine, evolution_manager, selfcoding_manager=None):
        self.data_bot = data_bot
        self.registered: list[str] = []

    def register_bot(self, name: str) -> None:
        self.registered.append(name)


evo_stub.EvolutionOrchestrator = _StubOrchestrator
sys.modules.setdefault("menace.evolution_orchestrator", evo_stub)

cap_stub = types.ModuleType("menace.capital_management_bot")

class CapitalManagementBot:
    def __init__(self, *a, **k):
        pass


cap_stub.CapitalManagementBot = CapitalManagementBot
sys.modules.setdefault("menace.capital_management_bot", cap_stub)

sie_stub = types.ModuleType("menace.self_improvement.engine")

class SelfImprovementEngine:
    def __init__(self, *a, **k):
        pass


sie_stub.SelfImprovementEngine = SelfImprovementEngine
sys.modules.setdefault("menace.self_improvement.engine", sie_stub)

sem_stub = types.ModuleType("menace.system_evolution_manager")

class SystemEvolutionManager:
    def __init__(self, bots, *a, **k):
        pass


sem_stub.SystemEvolutionManager = SystemEvolutionManager
sys.modules.setdefault("menace.system_evolution_manager", sem_stub)

import menace.coding_bot_interface as cbi
from menace.roi_thresholds import ROIThresholds

cbi.update_thresholds = lambda *a, **k: None
self_coding_managed = cbi.self_coding_managed


class DummyRegistry:
    def __init__(self):
        self.registered = []
        self.updated = []

    def register_bot(self, name, **kwargs):
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
        self._thresholds = {}

    def roi(self, name):
        return 0.0

    def reload_thresholds(self, bot=None):
        rt = ROIThresholds(roi_drop=-0.1, error_threshold=1.0, test_failure_threshold=0.0)
        self._thresholds[bot or ""] = rt
        return rt


class DummyOrchestrator:
    def __init__(self):
        self.registered = []

    def register_bot(self, name):
        self.registered.append(name)


def test_missing_bot_registry():
    with pytest.raises(TypeError):
        @self_coding_managed(data_bot=DummyDataBot())  # type: ignore[call-arg]
        class Bot:
            def __init__(self):
                pass


def test_missing_data_bot():
    with pytest.raises(TypeError):
        @self_coding_managed(bot_registry=DummyRegistry())  # type: ignore[call-arg]
        class Bot:
            def __init__(self):
                pass


def test_auto_instantiates_orchestrator():
    registry = DummyRegistry()
    data_bot = DummyDataBot()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "auto"

        def __init__(self):
            pass

    bot = Bot()
    assert registry.registered == ["auto"]
    assert bot.evolution_orchestrator.registered == ["auto"]


def test_orchestrator_autoinstantiation_failure(monkeypatch):
    class Broken(_StubOrchestrator):
        def __init__(self, *a, **k):  # pragma: no cover - simulate failure
            raise RuntimeError("boom")

    monkeypatch.setattr(evo_stub, "EvolutionOrchestrator", Broken)

    @self_coding_managed(bot_registry=DummyRegistry(), data_bot=DummyDataBot())
    class Bot:
        def __init__(self):
            pass

    with pytest.raises(RuntimeError, match="EvolutionOrchestrator is required"):
        Bot()


def test_successful_initialisation_registers():
    registry = DummyRegistry()
    data_bot = DummyDataBot()
    orchestrator = DummyOrchestrator()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "sample"

        def __init__(self):
            pass
    Bot(evolution_orchestrator=orchestrator)

    assert registry.registered == ["sample"]
    assert orchestrator.registered == ["sample"]


def test_thresholds_loaded_on_init():
    registry = DummyRegistry()
    data_bot = DummyDataBot()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "loader"

        def __init__(self):
            pass

    Bot()
    assert "loader" in data_bot._thresholds

