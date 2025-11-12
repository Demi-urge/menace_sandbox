import contextvars
import types
import sys

import pytest


class DummyManager:
    def __init__(self, *_, **kwargs):
        self.bot_registry = kwargs.get("bot_registry")
        self.data_bot = kwargs.get("data_bot")
        self.evolution_orchestrator = kwargs.get("evolution_orchestrator")
        self.quick_fix = kwargs.get("quick_fix") or object()


class DummyRegistry:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def register_bot(self, name: str, **_kwargs) -> None:
        self.registered.append(name)

    def update_bot(self, name: str, module_path: str, **_kwargs) -> None:  # pragma: no cover - minimal stub
        self.updated = (name, module_path)


class DummyDB:
    def __init__(self) -> None:
        self.logged: list[tuple[str, str, float]] = []

    def log_eval(self, name: str, metric: str, value: float) -> None:
        self.logged.append((name, metric, value))


class DummyDataBot:
    def __init__(self) -> None:
        self.db = DummyDB()

    def roi(self, name: str) -> float:  # pragma: no cover - simple stub
        return 0.0


class DummyOrchestrator:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def register_bot(self, name: str) -> None:
        self.registered.append(name)


@pytest.fixture(autouse=True)
def stub_self_coding_runtime(monkeypatch):
    manager_mod = types.ModuleType("menace.self_coding_manager")
    manager_mod.SelfCodingManager = DummyManager
    engine_mod = types.ModuleType("menace.self_coding_engine")
    engine_mod.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT")
    engine_mod.SelfCodingEngine = object
    monkeypatch.setitem(sys.modules, "menace.self_coding_manager", manager_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_manager", manager_mod)
    monkeypatch.setitem(sys.modules, "menace.self_coding_engine", engine_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_engine", engine_mod)
    yield


@pytest.fixture(autouse=True)
def reset_policy_cache():
    from menace_sandbox.self_coding_policy import get_self_coding_policy

    get_self_coding_policy.cache_clear()
    yield
    get_self_coding_policy.cache_clear()


def _instantiate_dummy_bot(cbi, registry, data_bot, orchestrator):
    manager = DummyManager(
        bot_registry=registry,
        data_bot=data_bot,
        evolution_orchestrator=orchestrator,
    )

    @cbi.self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class DummyBot:
        name = "DummyPolicyBot"

        def __init__(self, manager=None, evolution_orchestrator=None):
            self.manager = manager
            self.evolution_orchestrator = evolution_orchestrator

    DummyBot(manager=manager, evolution_orchestrator=orchestrator)
    return DummyBot


def test_all_bots_enabled_by_default(monkeypatch):
    monkeypatch.delenv("MENACE_SELF_CODING_ALLOWLIST", raising=False)
    monkeypatch.delenv("MENACE_SELF_CODING_DENYLIST", raising=False)
    from menace_sandbox import coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()
    orchestrator = DummyOrchestrator()

    _instantiate_dummy_bot(cbi, registry, data_bot, orchestrator)

    assert "DummyPolicyBot" in registry.registered
    assert "DummyPolicyBot" in orchestrator.registered


def test_denylist_disables_matching_bot(monkeypatch):
    monkeypatch.setenv("MENACE_SELF_CODING_DENYLIST", "DummyPolicyBot")
    monkeypatch.delenv("MENACE_SELF_CODING_ALLOWLIST", raising=False)
    from menace_sandbox import coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()
    orchestrator = DummyOrchestrator()

    _instantiate_dummy_bot(cbi, registry, data_bot, orchestrator)

    assert registry.registered == []
    assert orchestrator.registered == []


def test_allowlist_limits_participation(monkeypatch):
    monkeypatch.setenv("MENACE_SELF_CODING_ALLOWLIST", "OtherBot")
    monkeypatch.delenv("MENACE_SELF_CODING_DENYLIST", raising=False)
    from menace_sandbox import coding_bot_interface as cbi

    registry = DummyRegistry()
    data_bot = DummyDataBot()
    orchestrator = DummyOrchestrator()

    _instantiate_dummy_bot(cbi, registry, data_bot, orchestrator)

    assert registry.registered == []
    assert orchestrator.registered == []
