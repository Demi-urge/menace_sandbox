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
        self.register_calls = []

    def register_bot(self, name, **kwargs):
        self.registered.append(name)
        self.register_calls.append((name, kwargs))

    def update_bot(self, name, module_path):
        self.updated.append((name, module_path))


class ProvenanceRegistry(DummyRegistry):
    def __init__(self):
        super().__init__()
        self.provenance_updates = []
        self.attempts = 0

    def update_bot(self, name, module_path, *, patch_id=None, commit=None):
        self.attempts += 1
        if patch_id is None or commit is None:
            raise RuntimeError("patch provenance required")
        self.provenance_updates.append((name, module_path, patch_id, commit))


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


def test_module_path_falls_back_to_module_name(monkeypatch):
    registry = DummyRegistry()
    data_bot = DummyDataBot()

    def _raise_getfile(_cls):
        raise OSError("no source")

    monkeypatch.setattr(cbi.inspect, "getfile", _raise_getfile)

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "fallback"

        def __init__(self):
            pass

    Bot()

    assert registry.updated, "bot update should be attempted"
    assert registry.updated[-1][1] == Bot.__module__


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


def test_unsigned_provenance_generated_when_metadata_missing():
    registry = ProvenanceRegistry()
    data_bot = DummyDataBot()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "provless"

        def __init__(self):
            pass

    Bot()
    assert registry.registered == ["provless"]
    assert registry.attempts == 1
    assert len(registry.provenance_updates) == 1
    name, module_path, patch_id, commit = registry.provenance_updates[0]
    assert name == "provless"
    assert patch_id < 0
    assert commit.startswith("unsigned:")
    assert registry.register_calls[0][1].get("is_coding_bot") is True


def test_signed_policy_disables_self_coding_when_metadata_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MENACE_REQUIRE_SIGNED_PROVENANCE", "1")
    registry = ProvenanceRegistry()
    data_bot = DummyDataBot()

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class Bot:
        name = "strictprov"

        def __init__(self):
            pass

    Bot()
    assert registry.registered == ["strictprov"]
    assert registry.attempts == 0
    assert registry.provenance_updates == []
    assert registry.register_calls[0][1].get("is_coding_bot") is False


def test_uses_manager_provenance_for_update():
    registry = ProvenanceRegistry()
    data_bot = DummyDataBot()
    manager = DummyManager(bot_registry=registry, data_bot=data_bot)
    manager._last_patch_id = 123
    manager._last_commit_hash = "deadbeef"

    @self_coding_managed(bot_registry=registry, data_bot=data_bot, manager=manager)
    class Bot:
        name = "provbot"

        def __init__(self):
            pass

    Bot()
    assert [update[0] for update in registry.provenance_updates] == ["provbot"]
    assert registry.provenance_updates[0][2:] == (123, "deadbeef")


def test_falsey_manager_preserved_and_disabled_only_on_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = DummyRegistry()
    data_bot = DummyDataBot()

    class FalseyManager:
        def __init__(self) -> None:
            self.bot_registry = registry
            self.data_bot = data_bot

        def __bool__(self) -> bool:
            return False

    sentinel_manager = FalseyManager()

    def _fail_bootstrap(*_args, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError("bootstrap should not execute when manager supplied")

    monkeypatch.setattr(cbi, "_bootstrap_manager", _fail_bootstrap)
    monkeypatch.setattr(cbi, "_self_coding_runtime_available", lambda: True)

    @self_coding_managed(
        bot_registry=registry,
        data_bot=data_bot,
        manager=sentinel_manager,
    )
    class BotWithSentinel:
        name = "sentinel"

        def __init__(self) -> None:
            pass

    bot = BotWithSentinel()
    assert bot.manager is sentinel_manager
    assert not isinstance(bot.manager, cbi._DisabledSelfCodingManager)

    monkeypatch.setattr(cbi, "_self_coding_runtime_available", lambda: False)

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class BotWithFallback:
        name = "fallback"

        def __init__(self) -> None:
            pass

    fallback_bot = BotWithFallback()
    assert isinstance(fallback_bot.manager, cbi._DisabledSelfCodingManager)

