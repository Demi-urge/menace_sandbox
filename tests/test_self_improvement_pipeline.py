import contextvars
import sys
import types

import pytest

# Event log captures the order of operations across components
EVENT_LOG: list[str] = []


class FakeQuickFixEngine:
    """Minimal QuickFixEngine recording patch applications."""

    def __init__(self) -> None:
        self.applied: list[str] = []

    def apply_patch(self, desc: str) -> None:
        EVENT_LOG.append("quick_fix:apply")
        self.applied.append(desc)


class FakeBotRegistry:
    def __init__(self) -> None:
        self.graph: dict[str, dict[str, object]] = {}

    def register_bot(self, name: str) -> None:
        self.graph[name] = {"module": "", "version": 0}

    def update_bot(self, name: str, module_path: str) -> None:
        info = self.graph.setdefault(name, {"module": "", "version": 0})
        info["module"] = module_path
        info["version"] = int(info.get("version", 0)) + 1
        if info["version"] > 1:
            EVENT_LOG.append("registry:update")

    def version(self, name: str) -> int:
        return int(self.graph[name]["version"])  # type: ignore[return-value]


class FakeDB:
    def __init__(self) -> None:
        self.logged: list[tuple[str, str, float]] = []

    def log_eval(self, name: str, metric: str, value: float) -> None:
        self.logged.append((name, metric, value))


class FakeDataBot:
    def __init__(self) -> None:
        self.db = FakeDB()
        self.callbacks: list = []
        self.events: list[dict] = []

    def roi(self, _name: str) -> float:  # pragma: no cover - constant
        return 1.0

    def reload_thresholds(self, _name: str):  # pragma: no cover - constant
        return types.SimpleNamespace(
            roi_drop=-0.1, error_threshold=1.0, test_failure_threshold=0.0
        )

    def subscribe_degradation(self, cb) -> None:
        self.callbacks.append(cb)

    def check_degradation(self, bot: str, *, roi: float, errors: float) -> None:
        event = {"bot": bot, "roi": roi, "errors": errors}
        self.events.append(event)
        EVENT_LOG.append("data:degradation")
        for cb in list(self.callbacks):
            cb(event)


class FakeSelfCodingManager:
    def __init__(
        self,
        *,
        bot_registry: FakeBotRegistry,
        data_bot: FakeDataBot,
        quick_fix: FakeQuickFixEngine,
        bot_name: str = "sample",
    ) -> None:
        self.bot_registry = bot_registry
        self.data_bot = data_bot
        self.quick_fix = quick_fix
        self.bot_name = bot_name

    def register_patch_cycle(self, desc: str, event: dict | None = None) -> tuple[int, str]:
        self.quick_fix.apply_patch(desc)
        self.bot_registry.update_bot(self.bot_name, f"{self.bot_name}_patched")
        return 1, "deadbeef"


class FakeEvolutionOrchestrator:
    def __init__(self, data_bot: FakeDataBot, manager: FakeSelfCodingManager) -> None:
        self.data_bot = data_bot
        self.manager = manager
        self.patch_cycles: list[dict] = []
        self.registered: list[str] = []

    def register_bot(self, name: str) -> None:
        self.registered.append(name)
        self.data_bot.subscribe_degradation(self._on_degraded)

    def _on_degraded(self, event: dict) -> None:
        EVENT_LOG.append("orchestrator:patch_cycle")
        self.patch_cycles.append(event)
        self.manager.register_patch_cycle("degradation", event)


@pytest.fixture(autouse=True)
def _clear_event_log():
    EVENT_LOG.clear()
    yield
    EVENT_LOG.clear()


def test_self_improvement_pipeline(monkeypatch):
    scm_stub = types.ModuleType("menace.self_coding_manager")
    scm_stub.SelfCodingManager = FakeSelfCodingManager
    scm_stub.HelperGenerationError = RuntimeError
    sys.modules.setdefault("menace.self_coding_manager", scm_stub)

    sce_stub = types.ModuleType("menace.self_coding_engine")
    sce_stub.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT")
    sce_stub.SelfCodingEngine = object
    sys.modules.setdefault("menace.self_coding_engine", sce_stub)

    from menace.coding_bot_interface import self_coding_managed

    registry = FakeBotRegistry()
    data_bot = FakeDataBot()
    quick_fix = FakeQuickFixEngine()
    manager = FakeSelfCodingManager(
        bot_registry=registry, data_bot=data_bot, quick_fix=quick_fix
    )
    orchestrator = FakeEvolutionOrchestrator(data_bot, manager)

    @self_coding_managed(bot_registry=registry, data_bot=data_bot)
    class SampleBot:
        name = "sample"

        def __init__(self, manager=None, evolution_orchestrator=None):
            self.manager = manager
            self.evolution_orchestrator = evolution_orchestrator

    SampleBot(manager=manager, evolution_orchestrator=orchestrator)

    data_bot.check_degradation("sample", roi=-1.0, errors=5.0)

    assert data_bot.events, "DataBot should emit a degradation event"
    assert orchestrator.patch_cycles, "EvolutionOrchestrator should register a patch cycle"
    assert quick_fix.applied == ["degradation"], "QuickFixEngine should apply a patch"
    assert registry.version("sample") == 2, "BotRegistry should reflect new version"
    assert EVENT_LOG == [
        "data:degradation",
        "orchestrator:patch_cycle",
        "quick_fix:apply",
        "registry:update",
    ]
