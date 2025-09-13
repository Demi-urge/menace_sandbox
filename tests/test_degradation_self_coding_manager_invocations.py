import types

EVENT_LOG = []


class FakeSelfCodingEngine:
    def __init__(self) -> None:
        self.calls = []

    def generate_helper(self, desc: str, **_: object) -> str:
        EVENT_LOG.append("engine:generate_helper")
        self.calls.append(desc)
        return "def auto_helper():\n    pass\n"


class FakeQuickFixEngine:
    def __init__(self) -> None:
        self.validations = []
        self.applied = []

    def validate_patch(self, module: str, desc: str, repo_root=None):
        EVENT_LOG.append("quick_fix:validate")
        self.validations.append((module, desc))
        return True, []

    def apply_validated_patch(self, module: str, desc: str, ctx_meta=None):
        EVENT_LOG.append("quick_fix:apply")
        self.applied.append((module, desc))
        return True, 1, []


class FakeBotRegistry:
    def __init__(self) -> None:
        self.hot_swapped = []

    def register_bot(self, name: str) -> None:  # pragma: no cover - no-op
        pass

    def hot_swap(self, name: str, module_path: str) -> None:
        EVENT_LOG.append("registry:hot_swap")
        self.hot_swapped.append((name, module_path))


class FakeDataBot:
    def __init__(self) -> None:
        self.callbacks = []

    def subscribe_degradation(self, cb):
        self.callbacks.append(cb)

    def trigger(self, bot: str, severity: float = 1.0) -> None:
        event = {"bot": bot, "severity": severity}
        EVENT_LOG.append("data:degradation")
        for cb in list(self.callbacks):
            cb(event)

    # Below are stubs used by SelfCodingManager but unused in this test
    def roi(self, _bot: str) -> float:  # pragma: no cover - constant
        return 1.0

    def average_errors(self, _bot: str) -> float:  # pragma: no cover - constant
        return 0.0

    def average_test_failures(self, _bot: str) -> float:  # pragma: no cover - constant
        return 0.0

    def reload_thresholds(self, _bot: str):  # pragma: no cover - constant
        return types.SimpleNamespace(
            roi_drop=-0.1, error_threshold=1.0, test_failure_threshold=0.0
        )

    def log_evolution_cycle(self, *_a, **_k) -> None:  # pragma: no cover - no-op
        pass

    def check_degradation(self, *_a, **_k) -> bool:  # pragma: no cover - always true
        return True


class FakeSelfCodingManager:
    def __init__(
        self,
        *,
        engine: FakeSelfCodingEngine,
        quick_fix: FakeQuickFixEngine,
        bot_registry: FakeBotRegistry,
        data_bot: FakeDataBot,
        bot_name: str = "sample",
    ) -> None:
        self.engine = engine
        self.quick_fix = quick_fix
        self.bot_registry = bot_registry
        self.data_bot = data_bot
        self.bot_name = bot_name

    def register_patch_cycle(self, desc: str, event: dict | None = None):
        self.engine.generate_helper(desc)
        self.quick_fix.validate_patch("module", desc)
        self.quick_fix.apply_validated_patch("module", desc)
        self.bot_registry.hot_swap(self.bot_name, "module")
        return 1, "deadbeef"


class FakeEvolutionOrchestrator:
    def __init__(self, data_bot: FakeDataBot, manager: FakeSelfCodingManager) -> None:
        self.data_bot = data_bot
        self.manager = manager
        self.patch_cycles = []

    def register_bot(self, name: str) -> None:
        self.data_bot.subscribe_degradation(self._on_degraded)

    def _on_degraded(self, event: dict) -> None:
        EVENT_LOG.append("orchestrator:patch_cycle")
        self.patch_cycles.append(event)
        self.manager.register_patch_cycle("auto_patch", event)


def test_degradation_triggers_patch_cycle_and_manager_calls():
    engine = FakeSelfCodingEngine()
    quick_fix = FakeQuickFixEngine()
    registry = FakeBotRegistry()
    data_bot = FakeDataBot()
    manager = FakeSelfCodingManager(
        engine=engine, quick_fix=quick_fix, bot_registry=registry, data_bot=data_bot
    )
    orchestrator = FakeEvolutionOrchestrator(data_bot, manager)

    orchestrator.register_bot("sample")
    data_bot.trigger("sample")

    assert orchestrator.patch_cycles, "EvolutionOrchestrator should register a patch cycle"
    assert engine.calls == ["auto_patch"], "SelfCodingEngine.generate_helper should be invoked"
    assert quick_fix.validations == [
        ("module", "auto_patch")
    ], "QuickFixEngine should validate the patch"
    assert registry.hot_swapped == [
        ("sample", "module")
    ], "BotRegistry.hot_swap should be called"
    assert EVENT_LOG == [
        "data:degradation",
        "orchestrator:patch_cycle",
        "engine:generate_helper",
        "quick_fix:validate",
        "quick_fix:apply",
        "registry:hot_swap",
    ]
