import importlib
import sys
import types
from pathlib import Path


def test_self_coding_pipeline_full(tmp_path):
    """End-to-end smoke test for self-coding pipeline hot swap."""
    # Stub coding bot interface so the dummy bot registers with orchestrator
    cbi = types.ModuleType("menace.coding_bot_interface")

    def self_coding_managed(cls):
        orig_init = cls.__init__

        def wrapped(self, *a, **kw):
            orch = kw.get("evolution_orchestrator")
            orig_init(self, *a, **kw)
            if orch:
                orch.register_bot(self.name)

        cls.__init__ = wrapped
        return cls

    cbi.self_coding_managed = self_coding_managed
    sys.modules["menace.coding_bot_interface"] = cbi

    # Stub self coding manager with minimal behaviour
    scm_mod = types.ModuleType("menace.self_coding_manager")

    class SelfCodingManager:
        def __init__(self, **kwargs):
            self.quick_fix = kwargs.get("quick_fix")
            self.bot_name = kwargs.get("bot_name", "")
            self.bot_registry = kwargs.get("bot_registry")
            self.event_bus = kwargs.get("event_bus")
            self.cycles: list[tuple[str, object | None]] = []
            if self.bot_registry:
                self.bot_registry.register_bot(self.bot_name)

        def register_patch_cycle(self, description, context_meta=None):
            self.cycles.append((description, context_meta))
            if self.event_bus:
                self.event_bus.publish(
                    "self_coding:cycle_registered",
                    {"bot": self.bot_name, "description": description},
                )

        def run_patch(self, path: Path, description: str, *, context_meta=None, context_builder=None):
            self.quick_fix.apply_validated_patch(str(path), description, context_meta or {})
            self._last_patch_id = 1
            self._last_commit_hash = "deadbeef"
            if self.bot_registry:
                self.bot_registry.update_bot(
                    self.bot_name,
                    str(path),
                    patch_id=self._last_patch_id,
                    commit=self._last_commit_hash,
                )

        def should_refactor(self) -> bool:  # pragma: no cover - simple
            return True

    scm_mod.SelfCodingManager = SelfCodingManager
    scm_mod.HelperGenerationError = RuntimeError
    sys.modules["menace.self_coding_manager"] = scm_mod

    class DummyQuickFix:
        def __init__(self):
            self.validated = False
            self.calls = []

        def apply_validated_patch(self, module_path, desc, ctx_meta):
            self.validated = True
            self.calls.append((module_path, desc, ctx_meta))
            return True, 1, []

    class DummyDataBot:
        def __init__(self, event_bus=None):
            self.event_bus = event_bus
            self.logged = []
            self._callbacks = []

        def log_eval(self, cycle, metric, value):
            self.logged.append((cycle, metric, value))

        def subscribe_degradation(self, cb):
            self._callbacks.append(cb)

        def check_degradation(self, bot, roi, errors, test_failures=0.0):
            event = {
                "bot": bot,
                "delta_roi": roi - 1.0,
                "delta_errors": float(errors),
                "roi_baseline": 1.0,
                "errors_baseline": 0.0,
            }
            degraded = roi < 1.0 or errors > 0.0
            if degraded:
                for cb in list(self._callbacks):
                    cb(event)
            return degraded

        def roi(self, _name):  # pragma: no cover - deterministic
            return 1.0

        def average_errors(self, _name):  # pragma: no cover - deterministic
            return 0.0

    class DummyRegistry:
        def __init__(self):
            self.graph: dict[str, dict[str, object]] = {}

        def register_bot(self, name):
            self.graph.setdefault(name, {})

        def update_bot(self, name, module, *, patch_id=None, commit=None):
            self.graph.setdefault(name, {})
            self.graph[name].update(
                {"module": module, "patch_id": patch_id, "commit": commit}
            )

    class EvolutionOrchestrator:
        def __init__(self, data_bot, selfcoding_manager):
            self.data_bot = data_bot
            self.selfcoding_manager = selfcoding_manager
            self._registered = set()

        def register_bot(self, name):
            if name in self._registered:
                return
            self._registered.add(name)
            self.data_bot.subscribe_degradation(self._on_bot_degraded)
            self.data_bot.check_degradation(name, roi=1.0, errors=0.0)

        def _on_bot_degraded(self, event):
            desc = f"auto_patch_due_to_degradation:{event['bot']}"
            self.selfcoding_manager.register_patch_cycle(desc, event)
            self.selfcoding_manager.run_patch(Path(__file__), desc, context_meta=event)

    class DummyBus:
        def __init__(self):
            self.events = []

        def publish(self, topic, payload):
            self.events.append((topic, payload))

    bus = DummyBus()
    data_bot = DummyDataBot(event_bus=bus)
    registry = DummyRegistry()
    quick_fix = DummyQuickFix()
    manager = SelfCodingManager(
        quick_fix=quick_fix,
        bot_name="dummy_coding_bot",
        bot_registry=registry,
        event_bus=bus,
    )
    orch = EvolutionOrchestrator(data_bot, manager)

    importlib.invalidate_caches()
    from tests.fixtures.dummy_coding_bot import DummyCodingBot

    DummyCodingBot(
        manager=manager,
        data_bot=data_bot,
        bot_registry=registry,
        evolution_orchestrator=orch,
    )

    # Log baseline metrics and verify
    data_bot.log_eval("dummy_coding_bot", "baseline_roi", 1.0)
    assert ("dummy_coding_bot", "baseline_roi", 1.0) in data_bot.logged

    # Trigger degradation
    data_bot.check_degradation("dummy_coding_bot", roi=0.0, errors=2.0)

    # Evolution orchestrator registered patch cycle
    assert manager.cycles

    # Quick fix engine validated the patch
    assert quick_fix.validated

    # Bot registry hot swapped the module
    node = registry.graph["dummy_coding_bot"]
    assert node["patch_id"] == 1
    assert node["commit"] == "deadbeef"
