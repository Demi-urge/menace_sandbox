import importlib
import inspect
import sys
import types
from pathlib import Path
from menace.coding_bot_interface import manager_generate_helper

# Prepare menace package root for dynamic imports
ROOT = Path(__file__).resolve().parents[2]
package = types.ModuleType("menace")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("menace", package)

# Minimal dynamic_path_router to avoid filesystem interactions
_dpr = types.ModuleType("dynamic_path_router")
_dpr.resolve_path = lambda p: Path(p)
_dpr.repo_root = lambda: ROOT
_dpr.resolve_dir = lambda p: Path(p)
_dpr.path_for_prompt = lambda p: str(p)
sys.modules.setdefault("dynamic_path_router", _dpr)

# Stub sandbox runner harness
sr_pkg = types.ModuleType("sandbox_runner")
sr_pkg.__path__ = []
sys.modules.setdefault("sandbox_runner", sr_pkg)
sys.modules.setdefault("menace.sandbox_runner", sr_pkg)
th = types.ModuleType("sandbox_runner.test_harness")
th.run_tests = lambda *a, **k: types.SimpleNamespace(success=True, stdout="", duration=0)
th.TestHarnessResult = types.SimpleNamespace
sys.modules.setdefault("sandbox_runner.test_harness", th)
sys.modules.setdefault("menace.sandbox_runner.test_harness", th)


# Dummy event bus to capture interactions

class DummyBus:
    def __init__(self):
        self.subs: dict[str, list] = {}
        self.events: list[tuple[str, object]] = []

    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)

    def publish(self, topic, payload):
        self.events.append((topic, payload))
        for fn in self.subs.get(topic, []):
            fn(topic, payload)


# Provide unified event bus stub before importing data_bot
bus_mod = types.ModuleType("menace.unified_event_bus")
bus_mod.UnifiedEventBus = DummyBus
sys.modules.setdefault("menace.unified_event_bus", bus_mod)

# Minimal coding bot interface stub to avoid self-coding engine dependency
cbi_stub = types.ModuleType("menace.coding_bot_interface")
cbi_stub.self_coding_managed = lambda cls: cls
sys.modules.setdefault("menace.coding_bot_interface", cbi_stub)

# Import core modules after stubbing environment
DataBot = importlib.import_module("menace.data_bot").DataBot
MetricsDB = importlib.import_module("menace.data_bot").MetricsDB
BotRegistry = importlib.import_module("menace.bot_registry").BotRegistry


# Dummy context builder and util hooks

class DummyContextBuilder:
    def build(self, description):
        return None

    def refresh_db_weights(self):
        pass


context_builder_util = importlib.import_module("context_builder_util")
context_builder_util.create_context_builder = lambda: DummyContextBuilder()
context_builder_util.ensure_fresh_weights = lambda *a, **k: None

# Stubs for engine and quick fix components


class DummyEngine:
    def __init__(self):
        self.calls: list[str] = []

    def generate_helper(self, desc, **kwargs):
        self.calls.append(desc)
        return "code"


class DummyQuickFix:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def apply_validated_patch(self, module_path, desc, ctx_meta):
        self.calls.append((module_path, desc))
        return True, 123, []


# Self-coding manager stub invoking generate_helper and updating registry

class SelfCodingManager:
    def __init__(self, *, bot_name, bot_registry, data_bot, engine, event_bus, quick_fix):
        self.bot_name = bot_name
        self.bot_registry = bot_registry
        self.data_bot = data_bot
        self.engine = engine
        self.event_bus = event_bus
        self.quick_fix = quick_fix
        self._last_patch_id = None
        self._last_commit_hash = None

    def register_patch_cycle(self, description, context_meta=None):
        if self.event_bus:
            self.event_bus.publish(
                "self_coding:cycle_registered", {"bot": self.bot_name, "description": description}
            )
        self._last_patch_id = 123
        self._last_commit_hash = "deadbeef"
        return 123, "deadbeef"

    def run_patch(self, path: Path, description: str, *, context_meta=None, context_builder=None):
        manager_generate_helper(self, description, path=str(path))
        self.quick_fix.apply_validated_patch(str(path), description, context_meta or {})
        self._last_patch_id = 123
        self._last_commit_hash = "deadbeef"
        self.bot_registry.update_bot(
            self.bot_name, str(path), patch_id=123, commit="deadbeef"
        )
        if self.event_bus:
            self.event_bus.publish(
                "bot:patched", {"bot": self.bot_name, "commit": "deadbeef"}
            )

    def should_refactor(self) -> bool:
        return True


# Expose stub manager module
scm_mod = types.ModuleType("menace.self_coding_manager")
scm_mod.SelfCodingManager = SelfCodingManager
scm_mod.HelperGenerationError = RuntimeError
sys.modules.setdefault("menace.self_coding_manager", scm_mod)


class EvolutionOrchestrator:
    def __init__(
        self,
        data_bot,
        capital_bot,
        improvement_engine,
        evolution_manager,
        *,
        selfcoding_manager,
        event_bus,
        module_path: Path,
        **_: object,
    ) -> None:
        self.data_bot = data_bot
        self.selfcoding_manager = selfcoding_manager
        self.event_bus = event_bus
        self.module_path = module_path
        if event_bus:
            event_bus.subscribe("bot:degraded", self._on_degraded)

    def register_bot(self, bot: str) -> None:  # pragma: no cover - simple stub
        pass

    def _on_degraded(self, _topic: str, event: dict) -> None:
        bot = event.get("bot", "")
        desc = f"auto_patch_due_to_degradation:{bot}"
        self.selfcoding_manager.register_patch_cycle(desc, event)
        self.selfcoding_manager.run_patch(self.module_path, desc, context_meta=event)


# ---------------------------------------------------------------------------
# Test begins


def test_full_self_coding_flow(tmp_path, monkeypatch):
    bus = DummyBus()
    data_bot = DataBot(MetricsDB(tmp_path / "metrics.db"), event_bus=bus)
    registry = BotRegistry(event_bus=bus)

    monkeypatch.syspath_prepend(tmp_path)

    update_calls = []

    def fake_update_bot(name, module_path, patch_id=None, commit=None):
        update_calls.append((name, module_path, patch_id, commit))

    monkeypatch.setattr(registry, "update_bot", fake_update_bot)

    # Decorator capturing registration and update
    def self_coding_decorator(cls):
        module_path = inspect.getfile(cls)
        registry.register_bot(cls.name)
        registry.update_bot(cls.name, module_path)
        return cls

    cbi = types.ModuleType("menace.coding_bot_interface")
    cbi.self_coding_managed = self_coding_decorator
    sys.modules.setdefault("menace.coding_bot_interface", cbi)

    mod_path = tmp_path / "dummy_module.py"
    mod_path.write_text(
        "from menace.coding_bot_interface import self_coding_managed\n\n"
        "@self_coding_managed\n"
        "class DummyBot:\n"
        "    name = 'dummy_bot'\n"
        "    def __init__(self):\n        pass\n"
    )

    importlib.invalidate_caches()
    sys.modules.pop("dummy_module", None)
    __import__("dummy_module")

    engine = DummyEngine()
    quick_fix = DummyQuickFix()
    manager = SelfCodingManager(
        bot_name="dummy_bot",
        bot_registry=registry,
        data_bot=data_bot,
        engine=engine,
        event_bus=bus,
        quick_fix=quick_fix,
    )

    class DummyCapital:
        pass

    class DummyImprovement:
        pass

    class DummyEvolution:
        pass

    orch = EvolutionOrchestrator(
        data_bot,
        DummyCapital(),
        DummyImprovement(),
        DummyEvolution(),
        selfcoding_manager=manager,
        event_bus=bus,
        module_path=mod_path,
    )

    orch.register_bot("dummy_bot")

    data_bot.check_degradation("dummy_bot", roi=1.0, errors=0.0)
    data_bot.check_degradation("dummy_bot", roi=0.0, errors=2.0)

    assert engine.calls, "generate_helper not invoked"
    topics = [t for t, _ in bus.events]
    assert "self_coding:cycle_registered" in topics, "patch cycle not registered"
    assert "bot:patch_applied" in topics
    assert update_calls and update_calls[-1][0] == "dummy_bot", "update_bot not called"
